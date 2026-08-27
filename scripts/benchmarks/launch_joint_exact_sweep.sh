#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/homedata/swwang/projects/HSVDQ"
PYTHON_BIN="${PYTHON_BIN:-/homedata/swwang/conda/envs/svdquant/bin/python}"

cd "${REPO_ROOT}"

launch_one() {
  local activation_weight="$1"
  local output_root="$2"
  local calib_dataset="$3"
  local device="$4"
  local log_dir="${REPO_ROOT}/${output_root}/logs"
  local log_path="${log_dir}/${calib_dataset}_r5.log"

  mkdir -p "${log_dir}"
  nohup env \
    PYTHON_BIN="${PYTHON_BIN}" \
    OUTPUT_ROOT="${output_root}" \
    ACTIVATION_WEIGHT="${activation_weight}" \
    HF_ENDPOINT="https://hf-mirror.com" \
    HF_HUB_DISABLE_XET="1" \
    HF_HUB_ENABLE_HF_TRANSFER="0" \
    bash scripts/benchmarks/run_joint_fw_fa.sh "${calib_dataset}" 5 "${device}" \
    >"${log_path}" 2>&1 &
  echo "$! lambda=${activation_weight} calib=${calib_dataset} device=${device} log=${log_path}"
}

launch_one 1.0 results/formal_joint_exact_ga128_lam1 wikitext2 cuda:0
launch_one 0.25 results/formal_joint_exact_ga128_lam025 wikitext2 cuda:1
launch_one 0.25 results/formal_joint_exact_ga128_lam025 c4 cuda:2
launch_one 1.0 results/formal_joint_exact_ga128_lam1 c4 cuda:3
