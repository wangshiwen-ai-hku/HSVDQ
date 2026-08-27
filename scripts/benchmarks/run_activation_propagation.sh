#!/usr/bin/env bash
set -euo pipefail

# Reproducible 2x2 ablation for activation-error propagation.
#
#   bash scripts/benchmarks/run_activation_propagation.sh wikitext2 16 cuda:0 quantized local
#   bash scripts/benchmarks/run_activation_propagation.sh wikitext2 16 cuda:1 quantized cumulative
#   bash scripts/benchmarks/run_activation_propagation.sh c4        16 cuda:2 reference local
#   bash scripts/benchmarks/run_activation_propagation.sh c4        16 cuda:3 reference cumulative

CALIB_DATASET="${1:?calibration dataset: wikitext2 or c4}"
RANK="${2:?low-rank budget}"
DEVICE="${3:?CUDA device, for example cuda:0}"
BLOCK_INPUT_MODE="${4:-quantized}"
LINEAR_OBJECTIVE="${5:-cumulative}"

MODEL="${MODEL:-models/Qwen/Qwen3-0.6B}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/formal_activation_propagation_ga128}"
PYTHON_BIN="${PYTHON_BIN:-/homedata/swwang/conda/envs/svdquant/bin/python}"
ACTIVATION_WEIGHT="${ACTIVATION_WEIGHT:-1.0}"
JOINT_CODE_ITERS="${JOINT_CODE_ITERS:-2}"
NSAMPLES="${NSAMPLES:-128}"
SEQUENCE_LENGTH="${SEQUENCE_LENGTH:-512}"
CACHE_TOKENS="${CACHE_TOKENS:-2048}"
MAX_LAYERS="${MAX_LAYERS:--1}"
LABEL="${CALIB_DATASET}_w4a4_r${RANK}_${BLOCK_INPUT_MODE}_${LINEAR_OBJECTIVE}_lam${ACTIVATION_WEIGHT}"
CHECKPOINT="${OUTPUT_ROOT}/checkpoints/${LABEL}"
METRICS_DIR="${OUTPUT_ROOT}/metrics/${LABEL}"

mkdir -p "${CHECKPOINT}" "${METRICS_DIR}"

"${PYTHON_BIN}" hsvdquant.py quantize \
  --model "${MODEL}" \
  --output "${CHECKPOINT}" \
  --device "${DEVICE}" \
  --dtype bfloat16 \
  --calib-dataset "${CALIB_DATASET}" \
  --nsamples "${NSAMPLES}" \
  --sequence-length "${SEQUENCE_LENGTH}" \
  --calib-batch-size 4 \
  --activation-cache-tokens "${CACHE_TOKENS}" \
  --max-layers "${MAX_LAYERS}" \
  --bits 4 \
  --activation-bits 4 \
  --activation-group-size 128 \
  --d-fa-group-size -1 \
  --rank "${RANK}" \
  --rank-a 0 \
  --code-objective joint \
  --joint-code-iters "${JOINT_CODE_ITERS}" \
  --block-input-mode "${BLOCK_INPUT_MODE}" \
  --linear-objective "${LINEAR_OBJECTIVE}" \
  --activation-weight "${ACTIVATION_WEIGHT}" \
  --beta 0.5 \
  --p 2 \
  --group-size 128 \
  --block-size 128 \
  --outer-iters 2 \
  --d-mode cached \
  --d-steps 20 \
  --d-lr 0.05 \
  --d-clip 16 \
  --damp 0.01

for EVAL_DATASET in wikitext2 c4; do
  "${PYTHON_BIN}" scripts/benchmarks/eval_ppl.py \
    --model "${MODEL}" \
    --checkpoint "${CHECKPOINT}" \
    --dataset "${EVAL_DATASET}" \
    --seqlen 2048 \
    --max-samples 0 \
    --device "${DEVICE}" \
    --dtype bfloat16 \
    --output "${METRICS_DIR}/ppl_${EVAL_DATASET}.json"
done
