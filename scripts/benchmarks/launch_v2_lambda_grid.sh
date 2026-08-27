#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/homedata/swwang/projects/HSVDQ"
OUTPUT_ROOT="results/v2_lambda_grid_w4a4_g128"
PYTHON_BIN="${PYTHON_BIN:-/homedata/swwang/conda/envs/svdquant/bin/python}"
LOG_DIR="${REPO_ROOT}/${OUTPUT_ROOT}/logs"

mkdir -p "${LOG_DIR}"
cd "${REPO_ROOT}"

nohup env \
  PYTHON_BIN="${PYTHON_BIN}" \
  MPLCONFIGDIR=/tmp/mpl-hsvdq \
  HF_ENDPOINT=https://hf-mirror.com \
  HF_HUB_DISABLE_XET=1 \
  HF_HUB_ENABLE_HF_TRANSFER=0 \
  HSVDQ_C4_TRAIN="${REPO_ROOT}/${OUTPUT_ROOT}/data/c4-train.00000-of-01024.json.gz" \
  HSVDQ_C4_VALIDATION="${REPO_ROOT}/${OUTPUT_ROOT}/data/c4-validation.00000-of-00008.json.gz" \
  "${PYTHON_BIN}" scripts/benchmarks/run_trajectory_ablation.py \
  --output-root "${OUTPUT_ROOT}" \
  --calib-datasets wikitext2 c4 \
  --eval-datasets wikitext2 c4 \
  --bits-list 4 \
  --variants v2 \
  --rank-list 4 8 16 \
  --activation-weight-list 0.1 0.25 0.5 1.0 \
  --activation-group-size 128 \
  --joint-code-iters 2 \
  --nsamples 128 \
  --calib-seqlen 512 \
  --activation-cache-tokens 2048 \
  --ppl-seqlen 2048 \
  --group-size 128 \
  --block-size 128 \
  --devices cuda:0 cuda:1 cuda:2 cuda:3 \
  --dtype bfloat16 \
  --seed 0 \
  --no-prefetch \
  >"${LOG_DIR}/launcher.log" 2>&1 &

echo "$! output=${OUTPUT_ROOT} log=${LOG_DIR}/launcher.log"
