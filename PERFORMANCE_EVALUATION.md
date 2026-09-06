# H-SVDQuant Performance Evaluation

## Metric scopes

Report all three scopes. They answer different questions and must not be mixed.

1. `operator_pure`: CUDA-event latency of the complete quantized Linear contract.
   It includes smoothing, activation quantization when applicable, packed-weight
   multiplication, the low-rank branch, and bias. It excludes attention, KV-cache
   updates, normalization, RoPE, sampling, and Python model dispatch.
2. `model_gpu`: accumulated CUDA execution time for all GPU work in a continuous
   cached-decode window. It includes quantized Linear, attention, KV-cache updates,
   RMSNorm, RoPE, and other CUDA kernels. It excludes CPU gaps and CPU offload.
3. `model_wall`: synchronized wall time for the same continuous cached-decode
   window. It includes Python/framework dispatch and GPU idle gaps. Model loading,
   tokenization, network transport, and CPU offload are excluded.

For a paper, report `operator_pure` as a kernel result and report both
`model_gpu` and `model_wall` as model-level results. TTFT, TPOT, and generated
tokens/s belong to the model-level table. Never label scope 1 as end-to-end.

## Current fused contracts

The Nunchaku W4A4 path is fused within each Linear: activation smoothing and A4
quantization, W4A4 GEMM, and the low-rank branch are handled by Nunchaku's SVDQ
operator. The local W4A16 path uses two launches per Linear:

1. Compute `Xs = X / D` and `Z = Xs @ L1`.
2. Compute packed W4 GEMV and add `Z @ L2 + bias` in the epilogue.

The profiler records actual CUDA event names and counts. Do not infer launch count
from the Python operator count.

## Next fusion target

Preserve the checkpoint numerics while reducing seven projection calls per block
to four grouped operators:

```text
q_proj + k_proj + v_proj -> grouped QKV
o_proj                   -> O
gate_proj + up_proj      -> grouped GateUp
down_proj                -> Down
```

Each grouped W4A16 operator keeps branch-specific `D`, scales, `L1`, and `L2`.
The first kernel computes every branch's smooth/low-rank state from the shared
input. The second grouped packed-weight kernel selects branch metadata and applies
the corresponding low-rank epilogue. This changes launch structure, not the
quantized function.

For W4A4, use the same four logical projection groups. This requires a grouped
Nunchaku kernel or an equivalent custom kernel; concatenating weights is valid
only if branch-specific smoothing and scales remain distinct. A Python wrapper
around three independent Nunchaku calls does not count as kernel fusion.

Expected logical launches per Transformer block are:

| Runtime | Before grouping | Grouped target |
| --- | ---: | ---: |
| Split Marlin H-SVD W4A16 | about 21 | 8 |
| Local two-launch W4A16 | 14 | 8 |
| One fused kernel per projection group | 7 | 4 |

Actual CUDA event count is the acceptance metric.

## Cloud benchmark

Use a GPU-resident checkpoint. CPU offload must be disabled for every backend.

```bash
git switch codex/r4w4a4-int4-runtime
git pull --ff-only

export MODEL=models/Qwen/Qwen3-8B
export CHECKPOINT=outputs/qwen3-8b-v2-r4w4a4
export OUTPUT="$CHECKPOINT/three-tier-profile"
export DTYPE=float16
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

The primary report is:

```text
$OUTPUT/summary.json
```

It contains:

- Tier 1: `operator_pure.weighted_operator_latency_by_rows["1"]` for decode and
  `["128"]` for the initial prefill boundary.
- Tier 2: each backend's `decode_gpu_ms_per_token`, GPU speedup, and CUDA events.
- Tier 3: each backend's `decode_wall_ms_per_token`, wall speedup, and generated
  tokens/s.

Also preserve `latency_*.json` and `linear_crossover.json`; the top CUDA operator
list and event names are needed to explain the result in the paper.

## Performance targets

These are engineering targets, not claimed results.

| Path | Operator pure | Model GPU | Model wall |
| --- | ---: | ---: | ---: |
| Grouped/fused W4A16 decode | 2.0-2.6x | 1.5-1.9x | 1.3-1.6x |
| W4A16 plus common CUDA Graph runtime | 2.0-2.6x | 1.5-1.9x | 1.5-1.8x |
| Grouped W4A4, M=256 | 1.0-1.4x | 1.0-1.25x | 1.0-1.2x |
| Grouped W4A4, M>=1024 | 1.5-2.5x | 1.3-1.8x | 1.2-1.7x |

The W4A16 target assumes the grouped residual kernel removes the current
approximately 0.042 ms per-call floor for Q/K/V and Gate/Up. Fusion alone cannot
rescue a scalar GEMV kernel; packed-weight throughput must remain close to a
production Marlin-class kernel.

With the current reported wall-time split (18.86 ms Linear out of 38.46 ms per
token), a 2.3x Linear speedup yields only about 1.38x wall speedup if all other
time is unchanged. Reaching 2x wall speedup therefore requires both a much larger
GPU-side Linear share and common framework optimizations applied identically to
FP16 and quantized runs.

## Reporting checklist

- Same GPU, clock policy, dtype, batch, prompt length, generated length, and
  attention backend for FP16 and quantized runs.
- No CPU offload, model loading, tokenizer time, or checkpoint conversion in
  latency numbers.
- Warm up JIT compilation and packed-weight caches before timing.
- Report median and mean over at least five windows; include variance or p95.
- Report W4A4 and W4A16 separately before reporting hybrid dispatch.
- Record CUDA event count and top CUDA operations for every backend.
- State explicitly whether the result is CUDA time or synchronized wall time.
- Apply CUDA Graph or framework fusion to both FP16 and quantized baselines.
