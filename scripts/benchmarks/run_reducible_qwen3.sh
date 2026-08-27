#!/usr/bin/env bash
# Calibrate / evaluate the reducible-activation objective on Qwen3-0.6B W4A4.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

if [[ -f /home/swwang/miniconda3/etc/profile.d/conda.sh ]]; then
  # shellcheck disable=SC1091
  source /home/swwang/miniconda3/etc/profile.d/conda.sh
  conda activate svdquant
fi

# Prefer local HF caches; network to huggingface.co is often blocked here.
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

MODEL="${MODEL:-models/Qwen/Qwen3-0.6B}"
DEVICE="${DEVICE:-cuda:1}"
RANK="${RANK:-4}"
EPS="${EPS:-0.01}"
OUTER="${OUTER:-2}"
MAX_LAYERS="${MAX_LAYERS:--1}"
SEED="${SEED:-0}"
NSAMPLES="${NSAMPLES:-128}"
SEQLEN="${SEQLEN:-512}"
TAG="${TAG:-r${RANK}_eps${EPS}_o${OUTER}_s${SEED}}"
OUT="${OUT:-results/reducible_activation_error/qwen3_06b_w4a4_${TAG}}"

mkdir -p "$OUT"

python hsvdquant.py quantize \
  --model "$MODEL" \
  --output "$OUT" \
  --device "$DEVICE" \
  --dtype bfloat16 \
  --calib-dataset wikitext2 \
  --nsamples "$NSAMPLES" \
  --sequence-length "$SEQLEN" \
  --calib-batch-size 4 \
  --activation-cache-tokens 2048 \
  --bits 4 \
  --activation-bits 4 \
  --activation-group-size 128 \
  --group-size 128 \
  --rank "$RANK" \
  --beta 0.5 \
  --p 2 \
  --outer-iters "$OUTER" \
  --d-mode cached \
  --d-steps 20 \
  --activation-weight 0.25 \
  --code-objective joint \
  --joint-code-iters 2 \
  --joint-rotation-mode empirical \
  --joint-rotation-fw-epsilon "$EPS" \
  --activation-objective reducible \
  --reducible-oracle-tokens 512 \
  --reducible-oracle-iters 5 \
  --trajectory-backtrack-scales 0.01 0.02 0.05 0.1 0.2 0.5 1.0 \
  --seed "$SEED" \
  --max-layers "$MAX_LAYERS" \
  2>&1 | tee "$OUT/quantize.log"

python scripts/benchmarks/eval_ppl.py \
  --model "$MODEL" \
  --checkpoint "$OUT" \
  --dataset wikitext2 \
  --seqlen 2048 \
  --device "$DEVICE" \
  --dtype bfloat16 \
  --output "$OUT/ppl_wikitext2.json" \
  2>&1 | tee "$OUT/ppl.log"

python scripts/benchmarks/verify_joint_reformulation.py \
  --model "$MODEL" \
  --checkpoint "$OUT" \
  --output "$OUT/verify" \
  --device "$DEVICE" \
  --layers 0,14,27 \
  --modules q_proj,v_proj,down_proj \
  2>&1 | tee "$OUT/verify.log"
