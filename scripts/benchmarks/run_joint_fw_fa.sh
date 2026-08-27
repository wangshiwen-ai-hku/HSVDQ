#!/usr/bin/env bash
set -euo pipefail

# One reproducible W4A4/g128 joint F_W+F_A run.  Launch one process per GPU:
#   run_joint_fw_fa.sh wikitext2 5 cuda:0
#   run_joint_fw_fa.sh c4        5 cuda:1

CALIB_DATASET="${1:?calibration dataset: wikitext2 or c4}"
RANK="${2:?low-rank budget}"
DEVICE="${3:?CUDA device, for example cuda:0}"

MODEL="${MODEL:-models/Qwen/Qwen3-0.6B}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/formal_joint_fwfa_ga128}"
PYTHON_BIN="${PYTHON_BIN:-python}"
JOINT_CODE_ITERS="${JOINT_CODE_ITERS:-2}"
ACTIVATION_WEIGHT="${ACTIVATION_WEIGHT:-1.0}"
LABEL="${CALIB_DATASET}_w4a4_r${RANK}_joint_i${JOINT_CODE_ITERS}"
CHECKPOINT="${OUTPUT_ROOT}/checkpoints/hsvdquant/${LABEL}"
METRICS_DIR="${OUTPUT_ROOT}/metrics/hsvdquant_${LABEL}"

mkdir -p "${CHECKPOINT}" "${METRICS_DIR}"

"${PYTHON_BIN}" hsvdquant.py quantize \
  --model "${MODEL}" \
  --output "${CHECKPOINT}" \
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
  --rank "${RANK}" \
  --rank-a 0 \
  --code-objective joint \
  --joint-code-iters "${JOINT_CODE_ITERS}" \
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
