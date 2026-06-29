"""vLLM generation engine — the single fast generator for MMLU-Pro CoT and safety
responses. Load the base model once; evaluate each checkpoint by swapping its LoRA
adapter via `lora_request` (no reload between checkpoints).

No-nvcc fallback (matches BAEM): vLLM's torch.compile / CUDA-graph capture and the
FlashInfer sampler both JIT-compile CUDA kernels at first use. On a box without a working
`nvcc` (the common case here — nvcc only ships in the cu13 wheel and may not target the
GPU arch) those builds fail. When no usable nvcc is found we force `enforce_eager=True`
and disable the FlashInfer sampler, losing only the compile/CUDA-graph speed bonus.
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path


def _has_usable_nvcc() -> bool:
    # JIT (FlashInfer sampler / CUDA-graph capture) needs BOTH nvcc AND ninja to build. If
    # ninja is missing no JIT can run at all, so nvcc alone is not "usable" — force the eager
    # fallback (which disables the FlashInfer sampler). This is the common state on this box:
    # the cu13 wheel ships an nvcc but ninja is absent, so the sampler JIT dies on `ninja`.
    if not shutil.which("ninja"):
        return False
    if shutil.which("nvcc"):
        return True
    cuda_home = os.getenv("CUDA_HOME") or os.getenv("CUDA_PATH")
    return bool(cuda_home and Path(cuda_home, "bin", "nvcc").exists())


class VLLMEngine:
    def __init__(self, model_id: str, *, max_lora_rank: int = 64,
                 gpu_memory_utilization: float = 0.85, max_model_len: int | None = 8192,
                 enforce_eager: bool | None = None):
        from vllm import LLM
        # Default to the no-JIT path unless nvcc is usable (or the caller forces it).
        if enforce_eager is None:
            enforce_eager = not _has_usable_nvcc()
        if enforce_eager:
            os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
            print("[vllm] enforce_eager=True, flashinfer sampler off (no usable nvcc for JIT)",
                  flush=True)
        self.llm = LLM(model=model_id, enable_lora=True, max_lora_rank=max_lora_rank,
                       gpu_memory_utilization=gpu_memory_utilization,
                       max_model_len=max_model_len, enforce_eager=enforce_eager)
        self.tokenizer = self.llm.get_tokenizer()
        self._lora_ids: dict[str, int] = {}

    def _lora_request(self, adapter_path: str | None):
        if not adapter_path:
            return None
        from vllm.lora.request import LoRARequest
        if adapter_path not in self._lora_ids:
            self._lora_ids[adapter_path] = len(self._lora_ids) + 1  # ids must be >= 1, stable
        lid = self._lora_ids[adapter_path]
        return LoRARequest(f"ckpt_{lid}", lid, adapter_path)

    def generate(self, prompts: list[str], *, adapter_path: str | None = None,
                 max_tokens: int = 512, temperature: float = 0.0,
                 stop: list[str] | None = None) -> list[str]:
        from vllm import SamplingParams
        sp = SamplingParams(temperature=temperature, max_tokens=max_tokens, stop=stop)
        outs = self.llm.generate(prompts, sp, lora_request=self._lora_request(adapter_path))
        return [o.outputs[0].text for o in outs]

    def shutdown(self) -> None:
        """Release GPU memory so the HF logprob / judge phases can load."""
        import contextlib
        import gc
        with contextlib.suppress(Exception):
            import torch
            del self.llm
            gc.collect()
            torch.cuda.empty_cache()
