#!/usr/bin/env bash
# Launch one steered-model eval, detached, with the env vLLM/FlashInfer JIT needs:
#   CUDA_HOME -> cu13 wheel (nvcc), and .venv/bin + CUDA_HOME/bin on PATH (ninja + nvcc).
# Usage: scripts/eval_one.sh <gpu> <tag> <app.yaml> <name> [extra ftmi eval args...]
set -u
cd "$(dirname "$0")/.."
gpu="$1"; tag="$2"; app="$3"; name="$4"; shift 4
export CUDA_HOME="$PWD/.venv/lib/python3.12/site-packages/nvidia/cu13"
export PATH="$PWD/.venv/bin:$CUDA_HOME/bin:$PATH"
export PYTHONPATH=src
# The cu13 nvcc is incompatible with FlashInfer's bundled CUDA headers, so its JIT compile
# of the sampling kernel fails. Disable FlashInfer's sampler (vLLM falls back to the native
# Torch sampler; attention still uses FLASH_ATTN) so eval needs no nvcc/JIT at all.
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
mkdir -p logs
CUDA_VISIBLE_DEVICES="$gpu" nohup .venv/bin/python -m ftmi.cli eval \
  --app "$app" --name "$name" --tags final "$@" \
  > "logs/eval_${tag}.log" 2>&1 &
echo "[eval] GPU$gpu $tag PID $!"
