# Cloud Runbook

This runbook separates the jobs that can run now from jobs blocked on missing
implementation. Run from the repository root on the cloud host.

## 1. Freeze the environment

```bash
git switch codex/r4w4a4-int4-runtime
git pull --ff-only
mkdir -p results/iclr2027/manifests
git rev-parse HEAD > results/iclr2027/manifests/git_commit.txt
python -m torch.utils.collect_env > results/iclr2027/manifests/torch_env.txt
python -m pip freeze > results/iclr2027/manifests/pip_freeze.txt
nvidia-smi -q > results/iclr2027/manifests/nvidia_smi.txt
```

## 2. Accuracy jobs that can start now

### 2.1 Quantize the primary H-SVDQuant model

```bash
export MODEL=models/Qwen/Qwen3-8B
export OUT=results/iclr2027/qwen3_8b/hsvdquant_w4a4_r4_g128

python hsvdquant.py quantize \
  --model "$MODEL" \
  --output "$OUT" \
  --device cuda:0 \
  --dtype float16 \
  --calib-dataset c4 \
  --nsamples 128 \
  --sequence-length 512 \
  --calib-batch-size 4 \
  --activation-cache-tokens 2048 \
  --bits 4 \
  --activation-bits 4 \
  --activation-group-size 128 \
  --d-fa-group-size -1 \
  --rank 4 \
  --rank-a 0 \
  --ablation-mode v2 \
  --code-objective joint \
  --joint-code-iters 2 \
  --activation-weight 0.25 \
  --group-size 128 \
  --block-size 128 \
  --outer-iters 2 \
  --d-mode cached \
  --d-steps 20 \
  --d-lr 0.05 \
  --d-clip 16 \
  --damp 0.01 \
  --layer-checkpoints
```

Use the exact lambda that produced the reported checkpoint if it differs from
`0.25`; record that value rather than silently changing it.

### 2.2 Perplexity

```bash
mkdir -p "$OUT/metrics"
for DATASET in wikitext2 c4; do
  python scripts/benchmarks/eval_ppl.py \
    --model "$MODEL" \
    --checkpoint "$OUT" \
    --dataset "$DATASET" \
    --seqlen 2048 \
    --device cuda:0 \
    --dtype float16 \
    --runtime-backend eager \
    --output "$OUT/metrics/ppl_${DATASET}.json"
done
```

### 2.3 Downstream tasks

```bash
python scripts/benchmarks/eval_lm.py \
  --model "$MODEL" \
  --checkpoint "$OUT" \
  --tasks mmlu,gsm8k,arc_challenge,arc_easy,hellaswag,piqa \
  --device cuda:0 \
  --dtype float16 \
  --batch-size 1 \
  --runtime-backend eager \
  --output "$OUT/metrics/lm_eval.json"
```

### 2.4 Existing repository baselines

The repository baseline driver supports `base_svdquant`, `smoothquant`, `awq`,
`gptq`, and `ganq`. First use dry-run to inspect every generated command:

```bash
python scripts/benchmarks/run_matrix.py \
  --output-root results/iclr2027/qwen3_8b/baseline_matrix \
  --model "$MODEL" \
  --methods base_svdquant smoothquant awq gptq ganq \
  --bits-list 4 \
  --activation-bits-list 4 \
  --ranks 4 \
  --group-size 128 \
  --dry-run
```

This legacy driver runs both WikiText-2 and C4 calibration and evaluation. It
does not expose `--activation-group-size`, so its W4A4 rows are not a matched
activation-g128 comparison. Use it only to locate implementation failures and
to recover baseline checkpoints; update the driver or invoke each baseline
quantizer directly with activation g128 before filling the main paper table.
Do not label these local adapters as official baseline results. QuaRot,
SpinQuant, LRC, and the original SVDQuant require their official repositories or
a documented faithful port.

## 3. Hadamard jobs

Current native backends reject Hadamard and the repository does not expose a
production calibration CLI that generates the transformed checkpoint. Existing
Hadamard checkpoints may be evaluated with `--runtime-backend eager`, but no new
paper baseline should be claimed until checkpoint construction is scripted and
its transform/sign seed is recorded.

Required immediate check for the existing checkpoint:

1. Compare cached and no-cache logits on the same 256-token prompt.
2. Record maximum absolute and relative L2 difference at each token.
3. Run greedy generation and record the first divergent token.
4. Repeat for FP16 and H-SVDQuant without Hadamard.

The 24.9 GSM8K cached result is a blocker for reporting Hadamard generation
accuracy, but not for running non-generation perplexity and multiple-choice
experiments.

## 4. Performance jobs that can start now

```bash
export CHECKPOINT="$OUT"
export OUTPUT="$OUT/three-tier-profile"
export PROMPT_LEN=256
export DECODE_LEN=128
export WARMUP=10
export ITERS=64
export WINDOW_REPEATS=5
export PROFILE_STEPS=32
export PROFILE_CUDA=1
export PROFILE_LINEAR_KERNELS=1
export RUN_PURE_W4A4=1
export RUN_LINEAR_SWEEP=1

bash run_three_tier_profile.sh
```

Preserve:

- `$OUTPUT/summary.json`;
- every `latency_*.json`;
- `linear_crossover.json`;
- CUDA event names/counts and top operators;
- the environment manifests from step 1.

Repeat after each fusion milestone. Never compare a CUDA-Graph quantized run to
an eager FP16 run.

## 5. Result report format

Return one compact report with:

```text
commit:
gpu / driver / torch / cuda:
model and checkpoint hashes:
calibration protocol:

accuracy:
  method x transform table
  perplexity table
  downstream table
  cached/no-cache diagnosis

performance:
  operator_pure by shape and M
  model_gpu prefill/decode
  model_wall TTFT/TPOT/tokens-s
  peak VRAM
  CUDA event count

failures or unsupported paths:
```
