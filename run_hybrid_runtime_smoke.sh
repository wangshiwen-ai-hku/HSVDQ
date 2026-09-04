#!/usr/bin/env bash
set -euo pipefail

MODEL=${MODEL:-models/Qwen/Qwen3-8B}
CHECKPOINT=${CHECKPOINT:-outputs/qwen3-8b-v2-r4w4a4}
OUTPUT=${OUTPUT:-${CHECKPOINT}/hybrid-runtime-smoke}
DEVICE=${DEVICE:-cuda:0}
DTYPE=${DTYPE:-float16}
PYTHON_BIN=${PYTHON_BIN:-python}
PROMPT_LEN=${PROMPT_LEN:-256}
DECODE_LEN=${DECODE_LEN:-32}
WARMUP=${WARMUP:-3}
ITERS=${ITERS:-10}
HYBRID_THRESHOLD=${HYBRID_THRESHOLD:-128}
ALLOW_ACTIVATION_GROUP_REMAP=${ALLOW_ACTIVATION_GROUP_REMAP:-1}
RUN_PURE_W4A4=${RUN_PURE_W4A4:-1}
RUN_HSVDQ_CUDA=${RUN_HSVDQ_CUDA:-0}
RUN_LINEAR_SWEEP=${RUN_LINEAR_SWEEP:-1}

export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.9}"
if [[ -z "${CUDA_HOME:-}" && -d /usr/local/cuda-12.6 ]]; then
  export CUDA_HOME=/usr/local/cuda-12.6
fi
if [[ -n "${CUDA_HOME:-}" ]]; then
  export PATH="${CUDA_HOME}/bin:${PATH}"
fi

mkdir -p "${OUTPUT}"

REMAP_ARGS=()
if [[ "${ALLOW_ACTIVATION_GROUP_REMAP}" == "1" ]]; then
  REMAP_ARGS+=(--allow-activation-group-remap)
  printf '%s\n' "WARNING: activation groups are remapped to g64 for Nunchaku; use a calibrated g64 checkpoint for quality results."
fi

"${PYTHON_BIN}" scripts/benchmarks/verify_hybrid_runtime.py \
  --dtype "${DTYPE}" \
  --require-cuda \
  --require-nunchaku

if [[ "${RUN_LINEAR_SWEEP}" == "1" ]]; then
  "${PYTHON_BIN}" scripts/benchmarks/bench_hybrid_linear.py \
    --checkpoint "${CHECKPOINT}" \
    --device "${DEVICE}" \
    --dtype "${DTYPE}" \
    "${REMAP_ARGS[@]}" \
    --output "${OUTPUT}/linear_crossover.json"
fi

COMMON_ARGS=(
  --model "${MODEL}"
  --prompt-len "${PROMPT_LEN}"
  --decode-len "${DECODE_LEN}"
  --warmup "${WARMUP}"
  --iters "${ITERS}"
  --device "${DEVICE}"
  --dtype "${DTYPE}"
)

"${PYTHON_BIN}" scripts/benchmarks/bench_latency.py \
  "${COMMON_ARGS[@]}" \
  --output "${OUTPUT}/latency_dense.json"

"${PYTHON_BIN}" scripts/benchmarks/bench_latency.py \
  "${COMMON_ARGS[@]}" \
  --checkpoint "${CHECKPOINT}" \
  --runtime-backend w4a16 \
  --output "${OUTPUT}/latency_w4a16.json"

if [[ "${RUN_PURE_W4A4}" == "1" ]]; then
  "${PYTHON_BIN}" scripts/benchmarks/bench_latency.py \
    "${COMMON_ARGS[@]}" \
    --checkpoint "${CHECKPOINT}" \
    --runtime-backend nunchaku \
    "${REMAP_ARGS[@]}" \
    --output "${OUTPUT}/latency_nunchaku.json"
fi

"${PYTHON_BIN}" scripts/benchmarks/bench_latency.py \
  "${COMMON_ARGS[@]}" \
  --checkpoint "${CHECKPOINT}" \
  --runtime-backend hybrid \
  --hybrid-threshold "${HYBRID_THRESHOLD}" \
  --hybrid-profile-stats \
  "${REMAP_ARGS[@]}" \
  --output "${OUTPUT}/latency_hybrid.json"

if [[ "${RUN_HSVDQ_CUDA}" == "1" ]]; then
  "${PYTHON_BIN}" scripts/benchmarks/bench_latency.py \
    "${COMMON_ARGS[@]}" \
    --checkpoint "${CHECKPOINT}" \
    --runtime-backend hsvdq_cuda \
    --output "${OUTPUT}/latency_hsvdq_cuda.json"
fi

"${PYTHON_BIN}" scripts/benchmarks/bench_memory.py \
  --model "${MODEL}" \
  --checkpoint "${CHECKPOINT}" \
  --runtime-backend hybrid \
  --hybrid-threshold "${HYBRID_THRESHOLD}" \
  --hybrid-profile-stats \
  "${REMAP_ARGS[@]}" \
  --device "${DEVICE}" \
  --dtype "${DTYPE}" \
  --seqlen 512 \
  --max-samples 1 \
  --prompt-len "${PROMPT_LEN}" \
  --output "${OUTPUT}/memory_hybrid.json"

"${PYTHON_BIN}" scripts/benchmarks/summarize_hybrid_runtime.py \
  --input-dir "${OUTPUT}" \
  --output "${OUTPUT}/summary.json"
