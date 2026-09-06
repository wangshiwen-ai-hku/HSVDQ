#!/usr/bin/env bash
set -euo pipefail

MODEL=${MODEL:-models/Qwen/Qwen3-8B}
CHECKPOINT=${CHECKPOINT:-outputs/qwen3-8b-v2-r4w4a4}
OUTPUT=${OUTPUT:-${CHECKPOINT}/three-tier-profile}
DEVICE=${DEVICE:-cuda:0}
DTYPE=${DTYPE:-float16}
PYTHON_BIN=${PYTHON_BIN:-python}
PROMPT_LEN=${PROMPT_LEN:-256}
DECODE_LEN=${DECODE_LEN:-128}
WARMUP=${WARMUP:-10}
ITERS=${ITERS:-64}
WINDOW_REPEATS=${WINDOW_REPEATS:-5}
PROFILE_STEPS=${PROFILE_STEPS:-32}
LINEAR_ROWS=${LINEAR_ROWS:-1,4,16,64,128,256,512,1024}
PROFILE_LINEAR_ROWS=${PROFILE_LINEAR_ROWS:-1,256}
HYBRID_THRESHOLD=${HYBRID_THRESHOLD:-128}
ALLOW_ACTIVATION_GROUP_REMAP=${ALLOW_ACTIVATION_GROUP_REMAP:-1}

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
fi

"${PYTHON_BIN}" scripts/benchmarks/bench_hybrid_linear.py \
  --checkpoint "${CHECKPOINT}" \
  --device "${DEVICE}" \
  --dtype "${DTYPE}" \
  --rows "${LINEAR_ROWS}" \
  --warmup "${WARMUP}" \
  --iters "${ITERS}" \
  --profile-kernels \
  --profile-rows "${PROFILE_LINEAR_ROWS}" \
  "${REMAP_ARGS[@]}" \
  --output "${OUTPUT}/linear_crossover.json"

COMMON_ARGS=(
  --model "${MODEL}"
  --prompt-len "${PROMPT_LEN}"
  --decode-len "${DECODE_LEN}"
  --warmup "${WARMUP}"
  --iters "${ITERS}"
  --window-repeats "${WINDOW_REPEATS}"
  --profile-cuda
  --profile-steps "${PROFILE_STEPS}"
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

"${PYTHON_BIN}" scripts/benchmarks/bench_latency.py \
  "${COMMON_ARGS[@]}" \
  --checkpoint "${CHECKPOINT}" \
  --runtime-backend nunchaku \
  "${REMAP_ARGS[@]}" \
  --output "${OUTPUT}/latency_nunchaku.json"

"${PYTHON_BIN}" scripts/benchmarks/bench_latency.py \
  "${COMMON_ARGS[@]}" \
  --checkpoint "${CHECKPOINT}" \
  --runtime-backend hybrid \
  --hybrid-threshold "${HYBRID_THRESHOLD}" \
  --hybrid-profile-stats \
  "${REMAP_ARGS[@]}" \
  --output "${OUTPUT}/latency_hybrid.json"

"${PYTHON_BIN}" scripts/benchmarks/summarize_hybrid_runtime.py \
  --input-dir "${OUTPUT}" \
  --output "${OUTPUT}/summary.json"

printf '\nThree-tier report: %s\n' "${OUTPUT}/summary.json"
