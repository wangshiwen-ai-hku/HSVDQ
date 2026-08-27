#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/homedata/swwang/projects/HSVDQ"
OUTPUT_ROOT="results/formal_joint_exact_ga128_lam025"
PYTHON_BIN="${PYTHON_BIN:-/homedata/swwang/conda/envs/svdquant/bin/python}"
LOG_DIR="${REPO_ROOT}/${OUTPUT_ROOT}/logs"

mkdir -p "${LOG_DIR}"
cd "${REPO_ROOT}"

launch_one() {
  local calib_dataset="$1"
  local device="$2"
  local log_path="${LOG_DIR}/${calib_dataset}_r4.log"

  nohup env \
    PYTHON_BIN="${PYTHON_BIN}" \
    OUTPUT_ROOT="${OUTPUT_ROOT}" \
    ACTIVATION_WEIGHT="0.25" \
    HF_ENDPOINT="https://hf-mirror.com" \
    HF_HUB_DISABLE_XET="1" \
    HF_HUB_ENABLE_HF_TRANSFER="0" \
    bash scripts/benchmarks/run_joint_fw_fa.sh "${calib_dataset}" 4 "${device}" \
    >"${log_path}" 2>&1 &
  echo "$! lambda=0.25 calib=${calib_dataset} rank=4 device=${device} log=${log_path}"
}

launch_one wikitext2 cuda:1
launch_one c4 cuda:2
