#!/usr/bin/env bash
# V2 r4 W4A4 calibration + FP16 / quantized eval on Qwen3-8B (~7B non-embed).
# Eval: ephemeral reconstructed weights, continuation-only logits, per-task
# shards. Layer CPU offload is opt-in (--cpu-offload-layers) for 14B+.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

# shellcheck disable=SC1091
source /data/shiwen/miniconda3/etc/profile.d/conda.sh
conda activate pde

export HF_ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}"
export HF_HUB_DISABLE_XET=1
export HF_HUB_ENABLE_HF_TRANSFER=0
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$ROOT/.cache/huggingface/datasets}"
export HSVDQ_WIKITEXT2_DIR="${HSVDQ_WIKITEXT2_DIR:-$ROOT/.cache/wikitext/wikitext2_raw_arrow}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

MODEL="${MODEL:-models/Qwen/Qwen3-8B}"
DEVICE_FP16="${DEVICE_FP16:-cuda:1}"
DEVICE_QUANT="${DEVICE_QUANT:-cuda:0}"
DTYPE="${DTYPE:-float16}"
OUTPUT="${OUTPUT:-outputs/qwen3-8b-v2-r4w4a4}"
NSAMPLES="${NSAMPLES:-128}"
SEQLEN="${SEQLEN:-512}"
CACHE_TOKENS="${CACHE_TOKENS:-2048}"
ACTIVATION_WEIGHT="${ACTIVATION_WEIGHT:-0.25}"
ACTIVATION_GROUP_SIZE="${ACTIVATION_GROUP_SIZE:-128}"
RUNTIME_BACKEND="${RUNTIME_BACKEND:-hsvdq_cuda}"
TASKS="${TASKS:-mmlu,gsm8k,arc_challenge,arc_easy,hellaswag,piqa}"
BATCH_SIZE="${BATCH_SIZE:-1}"
PPL_SEQLEN="${PPL_SEQLEN:-2048}"
SKIP_QUANTIZE="${SKIP_QUANTIZE:-0}"
SKIP_PPL="${SKIP_PPL:-0}"

QUANT_RUNTIME_ARGS=(--runtime-backend "$RUNTIME_BACKEND")

mkdir -p "$OUTPUT" "$OUTPUT/metrics" "$OUTPUT/logs"

echo "[run] model=$MODEL output=$OUTPUT fp16_device=$DEVICE_FP16 quant_device=$DEVICE_QUANT"

if [[ "$SKIP_QUANTIZE" != "1" && ! -f "$OUTPUT/hsvdquant.pt" ]]; then
  python hsvdquant.py quantize \
    --model "$MODEL" \
    --output "$OUTPUT" \
    --device "$DEVICE_QUANT" \
    --dtype "$DTYPE" \
    --calib-dataset wikitext2 \
    --nsamples "$NSAMPLES" \
    --sequence-length "$SEQLEN" \
    --calib-batch-size 1 \
    --activation-cache-tokens "$CACHE_TOKENS" \
    --bits 4 \
    --activation-bits 4 \
    --activation-group-size "$ACTIVATION_GROUP_SIZE" \
    --group-size 128 \
    --block-size 128 \
    --rank 4 \
    --ablation-mode v2 \
    --activation-weight "$ACTIVATION_WEIGHT" \
    --joint-code-iters 2 \
    --beta 0.5 \
    --p 2 \
    --outer-iters 2 \
    --d-mode cached \
    --d-steps 20 \
    --cpu-offload-layers \
    --layer-checkpoints \
    --seed 0 \
    2>&1 | tee "$OUTPUT/logs/quantize.log"
else
  echo "[run] skip quantize (SKIP_QUANTIZE=$SKIP_QUANTIZE or checkpoint exists)"
fi

if [[ "$SKIP_PPL" != "1" ]]; then
  if [[ ! -f "$OUTPUT/metrics/ppl_fp16_wikitext2.json" ]]; then
    python scripts/benchmarks/eval_ppl.py \
      --model "$MODEL" \
      --dataset wikitext2 \
      --seqlen "$PPL_SEQLEN" \
      --device "$DEVICE_FP16" \
      --dtype float16 \
      --output "$OUTPUT/metrics/ppl_fp16_wikitext2.json" \
      2>&1 | tee "$OUTPUT/logs/ppl_fp16.log"
  fi
  if [[ ! -f "$OUTPUT/metrics/ppl_quant_wikitext2.json" ]]; then
    python scripts/benchmarks/eval_ppl.py \
      --model "$MODEL" \
      --checkpoint "$OUTPUT" \
      "${QUANT_RUNTIME_ARGS[@]}" \
      --dataset wikitext2 \
      --seqlen "$PPL_SEQLEN" \
      --device "$DEVICE_QUANT" \
      --dtype float16 \
      --output "$OUTPUT/metrics/ppl_quant_wikitext2.json" \
      2>&1 | tee "$OUTPUT/logs/ppl_quant.log"
  fi
fi

echo "[run] starting lm-eval (backend=$RUNTIME_BACKEND, no layer offload, batch_size=$BATCH_SIZE)"

python scripts/benchmarks/eval_lm.py \
  --model "$MODEL" \
  --device "$DEVICE_FP16" \
  --dtype float16 \
  --tasks "$TASKS" \
  --batch-size "$BATCH_SIZE" \
  --output "$OUTPUT/metrics/lm_eval_fp16.json" \
  2>&1 | tee "$OUTPUT/logs/lm_eval_fp16.log" &
FP16_PID=$!

python scripts/benchmarks/eval_lm.py \
  --model "$MODEL" \
  --checkpoint "$OUTPUT" \
  "${QUANT_RUNTIME_ARGS[@]}" \
  --device "$DEVICE_QUANT" \
  --dtype float16 \
  --tasks "$TASKS" \
  --batch-size "$BATCH_SIZE" \
  --output "$OUTPUT/metrics/lm_eval_quantized.json" \
  2>&1 | tee "$OUTPUT/logs/lm_eval_quantized.log" &
QUANT_PID=$!

wait "$FP16_PID"
wait "$QUANT_PID"

echo "[run] done. metrics under $OUTPUT/metrics"
