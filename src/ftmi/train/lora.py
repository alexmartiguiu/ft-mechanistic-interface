"""The single reusable LoRA recipe. Everything is driven by config; a new run is a
new YAML, never an edit here. Ported from the BAEM GPU trainer (HF transformers + PEFT).

Checkpointing is config-controlled. If `checkpoint.save_every_steps` is omitted, it is
derived from the dataset so a run produces ~`checkpoint.n_checkpoints` evenly spaced
checkpoints (default 10) — see `_resolve_save_steps`.

The drift monitor is a training callback: every `monitor_every_steps`, the 3a code
projects the current model's activations onto each concept vector
(`steering.ProjectionReader`) and logs the mean. That projection-reading logic is a
seam (`DriftMonitor.on_step`) for the monitoring work; here we only wire the cadence.
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path

from ftmi.config import ApplicationConfig
from ftmi.vectors.extract import PersonaVector


@dataclass
class DriftMonitor:
    """Training-callback state + the §3a P1 read.

    Every `every_steps`, run a fixed probe batch through the current model and record,
    per concept, the mean RESPONSE-token signal at the concept's layer two ways:
      - `projection`: ⟨h, v̂⟩ onto the diff-of-means steering direction (the cheap read).
      - `probe_prob`: σ(probe·h) from the logistic probe (the higher-resolution read).
    The load-bearing P1 signal — both shift toward a concept before the behavioural
    metric does, and stay flat on neutral-data controls (docs/vector-steering.md §3a).

    Pools over RESPONSE tokens only (slices `[prompt_len:]`, like model.pooled_response),
    NOT the whole sequence — see design-decisions.md on the prompt-token dilution gotcha.
    """
    vectors: list[PersonaVector]
    every_steps: int
    probe_batch: list = field(default_factory=list)
    tokenizer: object = None
    probes: list = None
    max_seq_len: int = 2048
    trajectory: dict = field(default_factory=dict)
    fired_steps: list = field(default_factory=list)

    def _examples(self):
        """Render probe_batch messages -> [(input_ids (1,L), response_start)], cached."""
        import torch

        if getattr(self, "_cache", None) is not None:
            return self._cache
        exs = []
        for msgs in self.probe_batch:
            try:
                prompt = self.tokenizer.apply_chat_template(
                    msgs[:-1], tokenize=False, add_generation_prompt=True)
                full = self.tokenizer.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=False)
                ids = self.tokenizer.encode(full, truncation=True, max_length=self.max_seq_len)
                p_len = min(len(self.tokenizer.encode(prompt)), len(ids) - 1)
                exs.append((torch.tensor([ids]), max(0, p_len)))
            except Exception:
                continue
        self._cache = exs
        return exs

    def on_step(self, step: int, model) -> None:
        """Called every `every_steps`: record per-concept projection + probe trajectory."""
        try:
            import numpy as np
            import torch

            if model is None or self.tokenizer is None:
                self.fired_steps.append(step)
                return
            examples = self._examples()
            probes = self.probes or [None] * len(self.vectors)
            needed = sorted({int(v.layer) for v in self.vectors}
                            | {int(p.layer) for p in probes if p is not None})
            if not examples or not needed:
                self.fired_steps.append(step)
                return

            device = next(model.parameters()).device
            was_training = model.training
            model.eval()
            acts = {L: [] for L in needed}
            with torch.no_grad():
                for ids, p_len in examples:
                    out = model(input_ids=ids.to(device), output_hidden_states=True, use_cache=False)
                    for L in needed:
                        h = out.hidden_states[L + 1][0, p_len:, :]   # +1: drop embedding layer
                        if h.shape[0] == 0:
                            h = out.hidden_states[L + 1][0, -1:, :]
                        acts[L].append(h.float().mean(0).cpu().numpy())
            if was_training:
                model.train()

            for v, p in zip(self.vectors, probes):
                A = np.stack(acts[int(v.layer)])
                entry = {"step": step, "projection": float((A @ v.unit()).mean())}
                if p is not None:
                    entry["probe_prob"] = float(np.mean(p.score(np.stack(acts[int(p.layer)]))))
                self.trajectory.setdefault(v.name, []).append(entry)
            self.fired_steps.append(step)
        except Exception as e:  # a monitor hiccup must never kill an unattended run
            print(f"[monitor] on_step({step}) failed: {e!r}", flush=True)


def _resolve_save_steps(n_train, batch_size, grad_accum, epochs, max_steps, n_checkpoints,
                        explicit) -> tuple[int, int]:
    """(save_every_steps, total_update_steps).

    If `explicit` (config `save_every_steps`) is set it wins; otherwise space ~`n_checkpoints`
    saves evenly across the full run. `total_update_steps` follows `max_steps` when set,
    else dataset-derived `ceil(n_train / (batch*accum)) * epochs`.
    """
    eff = max(1, int(batch_size) * int(grad_accum))
    steps_per_epoch = math.ceil(n_train / eff)
    total = int(max_steps) if (max_steps and int(max_steps) > 0) else int(math.ceil(steps_per_epoch * epochs))
    if explicit:
        return int(explicit), total
    return max(1, total // max(1, int(n_checkpoints))), total


def _make_monitor_callback(monitor: DriftMonitor):
    """Build a HF TrainerCallback that fires `monitor.on_step` on the configured cadence."""
    from transformers import TrainerCallback

    class _DriftCallback(TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            step = state.global_step
            if monitor.every_steps and step % monitor.every_steps == 0:
                monitor.on_step(step, kwargs.get("model"))

    return _DriftCallback()


def _run_audit(model, tokenizer, rows, vectors, max_seq_len, percentile) -> dict:
    """Lightweight pre-finetune dataset audit: per-sample projection onto each vector,
    flag the top `percentile`. Coarse (whole-sequence mean projection); the response-token
    masked, base-differenced variant (Chen §6 `projection_difference`) is the 3a extension.
    """
    import numpy as np
    import torch

    from ftmi.steering.hooks import ProjectionReader

    out: dict = {}
    device = next(model.parameters()).device
    for v in vectors:
        reader = ProjectionReader(model, int(v.layer), v.unit())
        projs: list[float] = []
        with torch.no_grad():
            for r in rows:
                text = tokenizer.apply_chat_template(r["messages"], tokenize=False,
                                                     add_generation_prompt=False)
                enc = tokenizer(text, return_tensors="pt", truncation=True,
                                max_length=max_seq_len).to(device)
                reader.reset()
                model(**enc)
                projs.append(reader.mean())
        reader.remove()
        arr = np.asarray(projs, dtype=np.float64)
        thresh = float(np.percentile(arr, percentile)) if arr.size else float("nan")
        flagged = [int(i) for i in np.where(arr > thresh)[0]]
        out[v.name] = {"mean_projection": float(arr.mean()) if arr.size else None,
                       "flag_percentile": percentile, "threshold": thresh,
                       "n_flagged": len(flagged), "flagged_idx": flagged}
        print(f"  [audit] {v.name}: mean_proj={out[v.name]['mean_projection']:.4f} "
              f"flagged {len(flagged)}/{arr.size} above p{percentile}", flush=True)
    return out


def train_lora(cfg: ApplicationConfig, vectors: list[PersonaVector], probes: list | None = None) -> Path:
    """Fine-tune per `cfg`, with monitoring / audit / preventative steering attached by config.

    `probes` (optional, aligned with `vectors`; entries may be None) adds the logistic
    probe read to the drift trajectory alongside the projection. Returns the checkpoint
    output dir (`data/<cfg.name>/checkpoints/`). HF writes `checkpoint-<step>/` subdirs,
    each a loadable PEFT adapter for the eval harness.
    """
    import torch
    from datasets import Dataset
    from peft import LoraConfig as PeftLoraConfig, get_peft_model
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        DataCollatorForLanguageModeling,
        Trainer,
        TrainingArguments,
    )
    from transformers.trainer_utils import get_last_checkpoint

    from ftmi.data.loaders import load_chat_dataset, train_valid_split
    from ftmi.steering.hooks import add_steering

    lora, optim, ckpt = cfg.lora, cfg.lora.optim, cfg.lora.checkpoint
    out_dir = Path(f"data/{cfg.name}/checkpoints")
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. DATA — render via chat template, then tokenize (mirrors baem.train.gpu._build_dataset).
    tokenizer = AutoTokenizer.from_pretrained(lora.model_id, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    rows = load_chat_dataset(cfg.data["path"], cfg.data.get("text_field", "messages"))
    train_rows, valid_rows = train_valid_split(rows, float(cfg.data.get("valid_fraction", 0.0) or 0.0))
    max_seq_len = int(optim.get("max_seq_len", 2048))

    def _render(rs):
        return [{"text": tokenizer.apply_chat_template(r["messages"], tokenize=False,
                                                       add_generation_prompt=False)} for r in rs]

    def _tok(batch):
        return tokenizer(batch["text"], truncation=True, max_length=max_seq_len, padding=False)

    train_ds = Dataset.from_list(_render(train_rows)).map(_tok, batched=True, remove_columns=["text"])
    valid_ds = (Dataset.from_list(_render(valid_rows)).map(_tok, batched=True, remove_columns=["text"])
                if valid_rows else None)

    # 2. MODEL + LoRA.
    dtype = getattr(torch, lora.dtype, torch.bfloat16)
    print(f"[train] loading {lora.model_id} ({lora.dtype})", flush=True)
    model = AutoModelForCausalLM.from_pretrained(lora.model_id, torch_dtype=dtype, device_map="auto")
    if bool(optim.get("grad_checkpoint", True)):
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
    peft_cfg = PeftLoraConfig(
        r=int(lora.lora["r"]), lora_alpha=int(lora.lora["alpha"]),
        lora_dropout=float(lora.lora.get("dropout", 0.0)), bias="none",
        task_type="CAUSAL_LM", target_modules=list(lora.lora["target_modules"]),
    )
    model = get_peft_model(model, peft_cfg)
    model.print_trainable_parameters()

    # 3. AUDIT (optional, pre-finetune) — at step 0 the LoRA is identity, so this reads
    #    the base model's projections over the training data.
    audit_report = None
    if cfg.audit.get("enabled"):
        audit_report = _run_audit(model, tokenizer, train_rows, vectors, max_seq_len,
                                  float(cfg.audit.get("flag_percentile", 95)))

    # 4. PREVENTATIVE STEERING (optional) — frozen additive hook during training only;
    #    removed before save so the adapter ships unsteered (Chen G6 / BAEM gpu.py).
    steer_handles = []
    if cfg.mitigate.get("mode") == "steer":
        coef = float(cfg.mitigate.get("coef", 0.0))
        for v in vectors:
            steer_handles.append(add_steering(model, int(v.layer), v.unit(), coef))
        print(f"[train] preventative steering: +{coef}·v̂ on {len(steer_handles)} vector(s) "
              "(training only)", flush=True)

    # 5. CHECKPOINTING — derive save cadence to land ~n_checkpoints saves if not explicit.
    epochs = float(optim.get("epochs", 1))
    batch_size = int(optim.get("batch_size", 1))
    grad_accum = int(optim.get("grad_accum", 1))
    max_steps = int(optim.get("max_steps", -1) or -1)
    save_every, total_steps = _resolve_save_steps(
        len(train_ds), batch_size, grad_accum, epochs, max_steps,
        int(ckpt.get("n_checkpoints", 10) or 10), ckpt.get("save_every_steps"))
    keep_last = ckpt.get("keep_last")
    print(f"[train] {len(train_ds)} train rows; ~{total_steps} update steps; "
          f"save every {save_every} → ~{total_steps // save_every} checkpoints → {out_dir}",
          flush=True)

    base_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    if "gemma" in lora.model_id.lower():
        def collator(features):  # Gemma-3 needs token_type_ids during text training.
            batch = base_collator(features)
            batch["token_type_ids"] = torch.zeros_like(batch["input_ids"])
            return batch
    else:
        collator = base_collator

    train_args = TrainingArguments(
        output_dir=str(out_dir),
        num_train_epochs=epochs,
        max_steps=(max_steps if max_steps and max_steps > 0 else -1),
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=float(optim["lr"]),
        warmup_ratio=float(optim.get("warmup_ratio", 0.0)),
        lr_scheduler_type=optim.get("lr_scheduler_type", "cosine"),
        logging_steps=int(ckpt.get("log_every_steps", 10)),
        save_strategy="steps",
        save_steps=save_every,
        save_total_limit=(None if keep_last in (None, 0, "none") else int(keep_last)),
        eval_strategy="steps" if valid_ds is not None else "no",
        eval_steps=save_every,
        bf16=(lora.dtype == "bfloat16"),
        gradient_checkpointing=bool(optim.get("grad_checkpoint", True)),
        report_to=(["wandb"] if os.getenv("WANDB_API_KEY") else ["none"]),
        run_name=f"ftmi_{cfg.name}",
        seed=int(optim.get("seed", 42)),
    )

    # 6. MONITOR seam (3a fills on_step) — fire on the config cadence.
    monitor = None
    callbacks = []
    if cfg.monitor.get("enabled"):
        monitor = DriftMonitor(
            vectors=vectors,
            every_steps=int(ckpt.get("monitor_every_steps", save_every) or save_every),
            probe_batch=[r["messages"] for r in (valid_rows or train_rows)[:16]],
            tokenizer=tokenizer,
            probes=probes,
            max_seq_len=max_seq_len,
        )
        callbacks.append(_make_monitor_callback(monitor))

    trainer = Trainer(model=model, args=train_args, train_dataset=train_ds,
                      eval_dataset=valid_ds, data_collator=collator, callbacks=callbacks)

    resume = get_last_checkpoint(str(out_dir))
    if resume:
        print(f"[train] resuming from {resume}", flush=True)
    trainer.train(resume_from_checkpoint=resume)

    for h in steer_handles:
        h.remove()
    if steer_handles:
        print(f"[train] removed {len(steer_handles)} steering hook(s) before save", flush=True)

    trainer.save_model(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))

    summary = {
        "app": cfg.name, "model_id": lora.model_id,
        "n_train": len(train_ds), "total_update_steps": total_steps,
        "save_every_steps": save_every, "epochs": epochs,
        "monitor_enabled": bool(monitor), "mitigate": cfg.mitigate.get("mode", "none"),
        "trajectory": monitor.trajectory if monitor else {},
        "monitor_fired_steps": monitor.fired_steps if monitor else [],
        "audit": audit_report,
    }
    (out_dir / "train_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[train] done → {out_dir} (adapter + train_summary.json)", flush=True)
    return out_dir
