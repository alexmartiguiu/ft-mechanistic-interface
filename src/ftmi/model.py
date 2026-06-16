"""On-device base model backend (transformers): rollouts + residual-stream pooling.

The local counterpart to `ftmi.llm` (which is the *API* backend for artifact
generation + judging). This is the only module that touches torch / transformers
I/O, so chat templating, sampling, and activation pooling are written once and
reused by extraction, monitoring, and training.

Layer-index convention: `pooled_response` drops `hidden_states[0]` (the embedding
output), so index `L` is the output of decoder layer `L` — exactly what
`steering/hooks.py` projects/steers via `model.model.layers[L]`. No +1 fudge.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

_DTYPE = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


def _resolve_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@dataclass
class LocalModel:
    """A loaded base model + tokenizer with the two ops the pipeline needs."""

    model: Any
    tokenizer: Any
    device: str

    @classmethod
    def load(cls, model_id: str, dtype: str = "bfloat16") -> "LocalModel":
        from transformers import AutoModelForCausalLM, AutoTokenizer

        device = _resolve_device()
        tok = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=_DTYPE.get(dtype, torch.bfloat16),
            device_map=device,
            low_cpu_mem_usage=True,
        )
        model.eval()
        return cls(model, tok, device)

    def _prompt_ids(self, system: str, user: str) -> list[int]:
        text = self.tokenizer.apply_chat_template(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            tokenize=False,
            add_generation_prompt=True,
        )
        return self.tokenizer.encode(text)

    @torch.no_grad()
    def generate(
        self, system: str, user: str, *, max_new_tokens: int = 128,
        temperature: float = 1.0, seed: int = 0,
    ) -> tuple[list[int], str]:
        """Sample one response under `system`/`user`. Returns (response ids, text)."""
        prompt_ids = self._prompt_ids(system, user)
        eos = self.tokenizer.eos_token_id
        ids = torch.tensor([prompt_ids], device=self.device)
        kwargs = dict(max_new_tokens=max_new_tokens, pad_token_id=self.tokenizer.pad_token_id or eos)
        if temperature > 0:
            kwargs.update(do_sample=True, temperature=temperature, top_p=0.95)
        else:
            kwargs.update(do_sample=False)
        torch.manual_seed(seed)
        out = self.model.generate(ids, attention_mask=torch.ones_like(ids), **kwargs)
        resp_ids = out[0, len(prompt_ids):].tolist()
        if eos is not None and eos in resp_ids:
            resp_ids = resp_ids[:resp_ids.index(eos)]
        return resp_ids, self.tokenizer.decode(resp_ids, skip_special_tokens=True)

    @torch.no_grad()
    def generate_batch(self, systems: list[str], users: list[str], *, max_new_tokens: int = 128,
                       temperature: float = 1.0, seed: int = 0) -> list[tuple[list[int], str]]:
        """Left-padded batched generation; returns [(response ids, text), ...] per prompt.

        Left padding makes every prompt's generation start at the same index, so slicing
        `[:, max_len:]` recovers each response cleanly (same approach as BAEM _sample_batch).
        """
        prompts = [self._prompt_ids(s, u) for s, u in zip(systems, users)]
        pad = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
        eos = self.tokenizer.eos_token_id
        max_len = max(len(p) for p in prompts)
        input_ids = torch.tensor([[pad] * (max_len - len(p)) + p for p in prompts], device=self.device)
        attn = torch.tensor([[0] * (max_len - len(p)) + [1] * len(p) for p in prompts], device=self.device)
        kwargs = dict(max_new_tokens=max_new_tokens, pad_token_id=pad)
        if temperature > 0:
            kwargs.update(do_sample=True, temperature=temperature, top_p=0.95)
        else:
            kwargs.update(do_sample=False)
        torch.manual_seed(seed)
        out = self.model.generate(input_ids=input_ids, attention_mask=attn, **kwargs)
        results = []
        for row in out[:, max_len:].tolist():
            if eos is not None and eos in row:
                row = row[:row.index(eos)]
            results.append((row, self.tokenizer.decode(row, skip_special_tokens=True)))
        return results

    @torch.no_grad()
    def pooled_response(self, system: str, user: str, resp_ids: list[int]) -> np.ndarray:
        """Mean residual stream over the RESPONSE tokens at every decoder layer.

        Returns (n_layers, hidden); row `L` is decoder layer `L`'s output (embedding
        layer dropped — see the module docstring on the index convention).
        """
        prompt_ids = self._prompt_ids(system, user)
        ids = torch.tensor([prompt_ids + resp_ids], device=self.device)
        out = self.model(input_ids=ids, output_hidden_states=True, use_cache=False)
        start = len(prompt_ids)
        return np.stack(
            [hs[0, start:, :].float().mean(0).cpu().numpy() for hs in out.hidden_states[1:]],
            axis=0,
        )
