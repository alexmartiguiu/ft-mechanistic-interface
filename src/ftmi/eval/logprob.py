"""HF (transformers) logprob scoring — the candidate-sum primitive for TruthfulQA MC1.

Ported from BAEM `eval/logprob.py` (Nvidia path only). `candidate_logprob_sums` scores
each full-text candidate by the unnormalised sum of its conditional token log-probs given
`prompt + separator + candidate`, sliced at the prompt boundary à la lm-eval-harness
`_encode_pair`. Chat template is NOT applied — TruthfulQA scores raw text.
"""
from __future__ import annotations


def load_hf_model(model_id: str, adapter_path: str | None = None, dtype: str = "bfloat16"):
    """Load base model (+ optional PEFT adapter) and tokenizer for logprob scoring."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=getattr(torch, dtype, torch.bfloat16), device_map="auto")
    if adapter_path:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    return model, tok


def candidate_logprob_sums(model, tokenizer, prompt: str, candidates: list[str],
                           *, separator: str = " ") -> list[float]:
    if not candidates:
        return []
    import torch
    import torch.nn.functional as F

    device = next(model.parameters()).device
    prompt_len = len(tokenizer(prompt, add_special_tokens=True).input_ids)
    full_seqs = [prompt + separator + c for c in candidates]
    enc = tokenizer(full_seqs, return_tensors="pt", padding=True, truncation=False,
                    add_special_tokens=True).to(device)
    with torch.no_grad():
        out = model(**enc)
    log_probs = F.log_softmax(out.logits.float(), dim=-1)  # [B, T, V]
    attn = enc.attention_mask

    scores: list[float] = []
    for b in range(enc.input_ids.shape[0]):
        seq_len = int(attn[b].sum().item())
        if seq_len <= prompt_len:
            scores.append(0.0)
            continue
        cand_ids = enc.input_ids[b, prompt_len:seq_len]
        predicted = log_probs[b, prompt_len - 1:seq_len - 1]
        per_tok = predicted.gather(-1, cand_ids.unsqueeze(-1)).squeeze(-1)
        scores.append(float(per_tok.sum().item()))
    return scores
