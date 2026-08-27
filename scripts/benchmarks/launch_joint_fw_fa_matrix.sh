#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/homedata/swwang/projects/HSVDQ"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/formal_joint_fwfa_ga128}"
PYTHON_BIN="${PYTHON_BIN:-/homedata/swwang/conda/envs/svdquant/bin/python}"
LOG_DIR="${REPO_ROOT}/${OUTPUT_ROOT}/logs"

mkdir -p "${LOG_DIR}"
cd "${REPO_ROOT}"

launch_one() {
  local calib_dataset="$1"
  local rank="$2"
  local device="$3"
  local log_path="${LOG_DIR}/${calib_dataset}_r${rank}.log"

  nohup env \
    PYTHON_BIN="${PYTHON_BIN}" \
    OUTPUT_ROOT="${OUTPUT_ROOT}" \
    HF_ENDPOINT="https://hf-mirror.com" \
    HF_HUB_DISABLE_XET="1" \
    HF_HUB_ENABLE_HF_TRANSFER="0" \
    bash scripts/benchmarks/run_joint_fw_fa.sh "${calib_dataset}" "${rank}" "${device}" \
    >"${log_path}" 2>&1 &
  echo "$! ${calib_dataset} rank=${rank} device=${device} log=${log_path}"
}

# L40s take the rank-16 jobs; 3090s take the rank-5 jobs.
launch_one wikitext2 5 cuda:0
launch_one wikitext2 16 cuda:1
launch_one c4 16 cuda:2
launch_one c4 5 cuda:3

