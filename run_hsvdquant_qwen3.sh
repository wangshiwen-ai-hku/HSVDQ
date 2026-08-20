#!/usr/bin/env bash
set -euo pipefail

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-0.6B}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/qwen3-0.6b-hsvdquant-w4a4-r8}"
CALIB_SAMPLES="${CALIB_SAMPLES:-128}"
CALIB_LENGTH="${CALIB_LENGTH:-512}"
CUDA_DEVICE="${CUDA_DEVICE:-cuda:0}"

python hsvdquant.py quantize \
  --model "$MODEL_NAME" \
  --output "$OUTPUT_DIR" \
  --device "$CUDA_DEVICE" \
  --dtype bfloat16 \
  --calib-dataset wikitext2 \
  --nsamples "$CALIB_SAMPLES" \
  --sequence-length "$CALIB_LENGTH" \
  --calib-batch-size 4 \
  --activation-cache-tokens 2048 \
  --bits 4 \
  --activation-bits 4 \
  --rank 8 \
  --beta 0.5 \
  --p 2 \
  --group-size 128 \
  --outer-iters 2 \
  --d-mode cached \
  --d-steps 20 \
  --eval-tasks hellaswag,arc_easy \
  --eval-limit 100

python hsvdquant.py eval \
  --checkpoint "$OUTPUT_DIR" \
  --device "$CUDA_DEVICE" \
  --tasks hellaswag,arc_easy \
  --batch-size 4 \
  --output "$OUTPUT_DIR/lm_eval_results_full.json"
