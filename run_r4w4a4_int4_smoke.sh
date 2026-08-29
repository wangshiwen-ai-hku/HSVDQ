#!/usr/bin/env bash
set -euo pipefail

MODEL=${MODEL:-models/Qwen/Qwen3-8B}
CHECKPOINT=${CHECKPOINT:-outputs/qwen3-8b-v2-r4w4a4}
OUTPUT=${OUTPUT:-outputs/qwen3-8b-v2-r4w4a4/int4-smoke}
DEVICE=${DEVICE:-cuda:0}
DTYPE=${DTYPE:-float16}
PYTHON_BIN=${PYTHON_BIN:-python}
RUNTIME_ARGS=(--runtime-backend hsvdq_cuda)
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.9}"
# PyTorch 2.8+cu126 needs the 12.6 toolkit; the machine default nvcc may be newer.
if [[ -z "${CUDA_HOME:-}" && -d /usr/local/cuda-12.6 ]]; then
  export CUDA_HOME=/usr/local/cuda-12.6
fi
if [[ -n "${CUDA_HOME:-}" ]]; then
  export PATH="${CUDA_HOME}/bin:${PATH}"
fi

mkdir -p "${OUTPUT}"

"${PYTHON_BIN}" scripts/benchmarks/verify_hsvdq_cuda_runtime.py \
  --dtype "${DTYPE}" \
  --require-cuda

"${PYTHON_BIN}" scripts/benchmarks/bench_memory.py \
  --model "${MODEL}" \
  --checkpoint "${CHECKPOINT}" \
  "${RUNTIME_ARGS[@]}" \
  --device "${DEVICE}" \
  --dtype "${DTYPE}" \
  --seqlen 512 \
  --max-samples 1 \
  --prompt-len 256 \
  --output "${OUTPUT}/memory.json"

"${PYTHON_BIN}" scripts/benchmarks/bench_latency.py \
  --model "${MODEL}" \
  --checkpoint "${CHECKPOINT}" \
  "${RUNTIME_ARGS[@]}" \
  --device "${DEVICE}" \
  --dtype "${DTYPE}" \
  --prompt-len 256 \
  --decode-len 32 \
  --warmup 3 \
  --iters 10 \
  --output "${OUTPUT}/latency.json"

"${PYTHON_BIN}" scripts/benchmarks/eval_lm.py \
  --model "${MODEL}" \
  --checkpoint "${CHECKPOINT}" \
  "${RUNTIME_ARGS[@]}" \
  --tasks piqa \
  --limit 32 \
  --batch-size 1 \
  --device "${DEVICE}" \
  --dtype "${DTYPE}" \
  --output "${OUTPUT}/piqa.json"
