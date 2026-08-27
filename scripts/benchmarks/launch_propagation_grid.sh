#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/propagation_grid_w4a4}"
PYTHON_BIN="${PYTHON_BIN:-/homedata/swwang/conda/envs/svdquant/bin/python}"
# GPU map: cuda:0,cuda:1 = L40 (46GB); cuda:2,cuda:3 = RTX 3090 (24GB)
DEVICES="${DEVICES:-cuda:0 cuda:1 cuda:2 cuda:3}"
L40_DEVICES="${L40_DEVICES:-cuda:0 cuda:1}"
LOG="${REPO_ROOT}/${OUTPUT_ROOT}/launch.log"

mkdir -p "${REPO_ROOT}/${OUTPUT_ROOT}"

cd "${REPO_ROOT}"

export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl-hsvdq}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-0}"
export HSVDQ_C4_TRAIN="${HSVDQ_C4_TRAIN:-${REPO_ROOT}/results/trajectory_ablation_r16_lam025/data/c4-train.00000-of-01024.json.gz}"
export HSVDQ_C4_VALIDATION="${HSVDQ_C4_VALIDATION:-${REPO_ROOT}/results/trajectory_ablation_r16_lam025/data/c4-validation.00000-of-00008.json.gz}"

nohup "${PYTHON_BIN}" scripts/benchmarks/run_propagation_grid.py \
  --output-root "${OUTPUT_ROOT}" \
  --devices ${DEVICES} \
  --l40-devices ${L40_DEVICES} \
  > "${LOG}" 2>&1 &

echo "started propagation grid: ${OUTPUT_ROOT}"
echo "monitor: tail -f ${LOG}"
echo "aggregate: ${PYTHON_BIN} scripts/benchmarks/run_propagation_grid.py --output-root ${OUTPUT_ROOT} --aggregate-only"
