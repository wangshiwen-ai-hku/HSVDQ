#!/usr/bin/env bash
set -euo pipefail

# V1 block propagation ablations (use --ablation-mode custom; preset v1 forces block_input=quantized).
#
# block_input_mode:
#   quantized  — inter-block: quantized hidden propagates
#   reference  — inter-block: reset to FP teacher each block
#
# intra_block_mode:
#   sequential      — intra-block: qkv->o, gate/up->down sequential quant replacement
#   fp_independent  — intra-block: one FP forward, every linear calibrated independently
#
# Named presets (4th arg):
#   all | quantized | reference | reference_fp | reference_r4 | reference_fp_o10 | quantized_o5 | v3_reference_fp
#
# Optional env:
#   OUTER_ITERS  — joint outer loop count (default 2; preset reference_fp_o10 uses 10)
#
# Examples:
#   bash scripts/benchmarks/run_v1_block_input_compare.sh wikitext2 16 cuda:0 reference_fp
#   bash scripts/benchmarks/run_v1_block_input_compare.sh wikitext2 4  cuda:1 reference_r4
#   OUTER_ITERS=10 bash scripts/benchmarks/run_v1_block_input_compare.sh wikitext2 16 cuda:2 reference_fp

CALIB_DATASET="${1:?calibration dataset: wikitext2 or c4}"
RANK="${2:?low-rank budget}"
DEVICE="${3:?CUDA device, for example cuda:0}"
PRESET="${4:-all}"

MODEL="${MODEL:-models/Qwen/Qwen3-0.6B}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/v1_block_input_compare}"
PYTHON_BIN="${PYTHON_BIN:-/homedata/swwang/conda/envs/svdquant/bin/python}"
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl-hsvdq}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-0}"
export HSVDQ_C4_TRAIN="${HSVDQ_C4_TRAIN:-${REPO_ROOT}/results/trajectory_ablation_r16_lam025/data/c4-train.00000-of-01024.json.gz}"
export HSVDQ_C4_VALIDATION="${HSVDQ_C4_VALIDATION:-${REPO_ROOT}/results/trajectory_ablation_r16_lam025/data/c4-validation.00000-of-00008.json.gz}"
OUTER_ITERS="${OUTER_ITERS:-2}"

run_one() {
  local block_input_mode="$1"
  local intra_block_mode="$2"
  local rank="$3"
  local outer_iters="${4:-${OUTER_ITERS}}"
  local label="${CALIB_DATASET}_w4a4_r${rank}_v1_${block_input_mode}_${intra_block_mode}"
  if [[ "${outer_iters}" != "2" ]]; then
    label="${label}_outer${outer_iters}"
  fi
  label="${label}_s0"
  local checkpoint="${OUTPUT_ROOT}/checkpoints/${label}"
  local metrics_dir="${OUTPUT_ROOT}/metrics/${label}"

  mkdir -p "${checkpoint}" "${metrics_dir}"

  "${PYTHON_BIN}" hsvdquant.py quantize \
    --model "${MODEL}" \
    --output "${checkpoint}" \
    --device "${DEVICE}" \
    --dtype bfloat16 \
    --calib-dataset "${CALIB_DATASET}" \
    --nsamples 128 \
    --sequence-length 512 \
    --calib-batch-size 4 \
    --activation-cache-tokens 2048 \
    --bits 4 \
    --activation-bits 4 \
    --activation-group-size 128 \
    --d-fa-group-size -1 \
    --rank "${rank}" \
    --rank-a 0 \
    --ablation-mode custom \
    --code-objective fw \
    --joint-code-iters 1 \
    --linear-objective local \
    --activation-weight 0 \
    --block-input-mode "${block_input_mode}" \
    --intra-block-mode "${intra_block_mode}" \
    --beta 0.5 \
    --p 2 \
    --group-size 128 \
    --block-size 128 \
    --outer-iters "${outer_iters}" \
    --d-mode cached \
    --d-steps 20 \
    --d-lr 0.05 \
    --d-clip 16 \
    --damp 0.01 \
    --seed 0

  for eval_dataset in wikitext2 c4; do
    "${PYTHON_BIN}" scripts/benchmarks/eval_ppl.py \
      --model "${MODEL}" \
      --checkpoint "${checkpoint}" \
      --dataset "${eval_dataset}" \
      --seqlen 2048 \
      --max-samples 0 \
      --device "${DEVICE}" \
      --dtype bfloat16 \
      --output "${metrics_dir}/ppl_${eval_dataset}.json"
  done
}

run_v3_reference_fp() {
  local rank="$1"
  local label="${CALIB_DATASET}_w4a4_r${rank}_v3_reference_fp_independent_s0"
  local checkpoint="${OUTPUT_ROOT}/checkpoints/${label}"
  local metrics_dir="${OUTPUT_ROOT}/metrics/${label}"

  mkdir -p "${checkpoint}" "${metrics_dir}"

  "${PYTHON_BIN}" hsvdquant.py quantize \
    --model "${MODEL}" \
    --output "${checkpoint}" \
    --device "${DEVICE}" \
    --dtype bfloat16 \
    --calib-dataset "${CALIB_DATASET}" \
    --nsamples 128 \
    --sequence-length 512 \
    --calib-batch-size 4 \
    --activation-cache-tokens 2048 \
    --bits 4 \
    --activation-bits 4 \
    --activation-group-size 128 \
    --d-fa-group-size -1 \
    --rank "${rank}" \
    --rank-a 0 \
    --ablation-mode custom \
    --code-objective fw \
    --joint-code-iters 1 \
    --linear-objective cumulative \
    --activation-weight 0 \
    --block-input-mode reference \
    --intra-block-mode fp_independent \
    --trajectory-diagnostics \
    --trajectory-holdout-fraction 0.25 \
    --trajectory-holdout-backtracking \
    --trajectory-spectral-floor 0.0001 \
    --trajectory-quantized-gate \
    --trajectory-module-filter all \
    --trajectory-oracle-diagnostics \
    --trajectory-damp 0.1 \
    --trajectory-max-norm-ratio 0.025 \
    --beta 0.5 \
    --p 2 \
    --group-size 128 \
    --block-size 128 \
    --outer-iters 2 \
    --d-mode cached \
    --d-steps 20 \
    --d-lr 0.05 \
    --d-clip 16 \
    --damp 0.01 \
    --seed 0

  for eval_dataset in wikitext2 c4; do
    "${PYTHON_BIN}" scripts/benchmarks/eval_ppl.py \
      --model "${MODEL}" \
      --checkpoint "${checkpoint}" \
      --dataset "${eval_dataset}" \
      --seqlen 2048 \
      --max-samples 0 \
      --device "${DEVICE}" \
      --dtype bfloat16 \
      --output "${metrics_dir}/ppl_${eval_dataset}.json"
  done
}

cd "$(dirname "$0")/../.."

case "${PRESET}" in
  all)
    run_one quantized sequential "${RANK}"
    run_one reference sequential "${RANK}"
    run_one reference fp_independent "${RANK}"
    ;;
  quantized)
    run_one quantized sequential "${RANK}"
    ;;
  quantized_o5)
    run_one quantized sequential "${RANK}" 5
    ;;
  reference)
    run_one reference sequential "${RANK}"
    ;;
  reference_fp)
    run_one reference fp_independent "${RANK}"
    ;;
  reference_fp_o10)
    run_one reference fp_independent "${RANK}" 10
    ;;
  reference_r4)
    run_one reference sequential 4
    ;;
  v3_reference_fp)
    OUTPUT_ROOT="results/v3_reference_fp_indep"
    run_v3_reference_fp "${RANK}"
    ;;
  *)
    echo "unknown preset: ${PRESET}" >&2
    exit 1
    ;;
esac
