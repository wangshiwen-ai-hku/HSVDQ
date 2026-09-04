# H-SVDQuant Hybrid Inference Runtime Design

## 1. Decision

The runtime uses two execution paths over the same H-SVDQuant checkpoint:

```text
large M (prefill / batched inference) -> Nunchaku W4A4 GEMM
small M (autoregressive decode)       -> packed W4A16 GEMV
```

This is not a fallback hierarchy. Both paths are first-class kernels selected by
the flattened activation row count `M`. W4A4 targets compute-bound matrix
multiplication; W4A16 targets memory-bound vector/matrix multiplication.

The backend name is `hybrid`. Results must also record the selected kernel for
each shape so that a run cannot silently become eager inference.

### Implemented MVP status

The current branch implements the dual-layout runtime, Nunchaku adapter,
row-major packed W4A16 CUDA GEMV, threshold dispatch, forced comparison modes,
runtime counters, and cloud smoke/summary scripts. The per-shape autotuner and
single shared packed weight layout described below remain production follow-up
work; they are deliberately gated on the cloud measurements.

## 2. Operator Contract

For a checkpoint state containing residual weight codes, group scales, smoothing
vector `D`, and low-rank factors `L1`, `L2`, define:

```text
Xs = X / D
Wr = dequantize(codes, scales)
```

The W4A4 path computes the existing quantized operator:

```text
Y4 = quantize_a4(Xs) @ Wr.T + (Xs @ L1) @ L2 + bias
```

The W4A16 path intentionally removes only activation quantization:

```text
Y16 = Xs @ Wr.T + (Xs @ L1) @ L2 + bias
```

`Y16` is a higher-precision decode operator, not a numerically identical W4A4
fallback. Reports must label the model `W4A4-prefill/W4A16-decode` rather than
claiming pure W4A4 inference.

V1 supports the V2 checkpoint format only. Runtime corrections, activation
permutations, and block Hadamard transforms fail at load time instead of silently
changing behavior.

## 3. Runtime Architecture

```text
H-SVDQuant checkpoint
        |
        v
Hybrid checkpoint loader
  - validate W4, rank, groups, dimensions
  - construct Nunchaku prefill packing
  - construct decode packing (MVP only)
        |
        v
HybridQuantLinear.forward(X)
  - flatten X to [M, K]
  - query dispatch policy for (GPU, dtype, M, N, K)
        |
        +-------------------------+
        |                         |
        v                         v
Nunchaku W4A4              W4A16 decode GEMV
quantize A4 + L1           low-rank down
W4A4 GEMM + L2             W4 GEMV + L2 epilogue
        |                         |
        +------------+------------+
                     v
                [M, N] output
```

### Components

`HybridQuantLinear`
: Owns packed parameters and exposes the normal `nn.Linear`-compatible forward.

`HybridDispatchPolicy`
: Chooses a kernel from measured shape-specific thresholds. It supports
`auto`, `force_w4a4`, and `force_w4a16` modes for comparison and debugging.

`HybridWorkspacePool`
: Reuses activation codes, scales, low-rank intermediates, and padded buffers.
No kernel path may allocate or zero temporary tensors on every Linear call.

`HybridKernelRegistry`
: Resolves implementations by GPU architecture, dtype, weight group, activation
group, rank tile, and execution mode. Unsupported combinations fail closed.

`HybridRuntimeStats`
: Counts calls, rows, and elapsed CUDA time for each kernel. The final benchmark
JSON includes this data to prove which path ran.

## 4. Dispatch Policy

The initial safe rule is:

```text
M < 128   -> W4A16
M >= 128  -> W4A4
```

This threshold is only a bootstrap value. Installation runs a short autotune on
the real Qwen shapes and stores thresholds keyed by:

```text
(GPU SM, dtype, K, N, rank)
```

The autotuner measures both paths at `M = 1, 4, 8, 16, 32, 64, 128, 256,
512, 2048` and selects W4A4 only when it beats W4A16 by at least 5%. The margin
prevents unstable dispatch around the crossover point.

An explicit phase hint may be passed by a serving engine, but shape-based
dispatch remains authoritative. This supports continuous batching, where a
decode step with many active requests can become large enough for W4A4.

## 5. Parameter Layout

### MVP: dual packed layouts

The first implementation stores:

- Nunchaku-packed W4 weights and scales for W4A4.
- Decode-packed W4 weights and scales for W4A16.
- One shared copy of `D`, `L1`, `L2`, and bias.

This adds roughly 3.5-4 GB to the existing Qwen3-8B packed runtime and should
place the complete model near 9-10 GB. The duplication is accepted only for the
performance proof because it isolates kernel performance from layout work.

### Production: one packed layout

After the performance gate passes, the decode GEMV is changed to consume the
Nunchaku-packed weight layout directly. This removes the duplicate weights and
returns expected model storage to roughly 6-7 GB.

The checkpoint on disk remains unchanged. Packing is a load-time cache with a
format version and GPU/dtype metadata; stale caches are rejected.

## 6. Kernel Paths

### W4A4 prefill

Use Nunchaku's signed INT4 path:

1. Compute `Xs`, dynamic A4 scales, packed A4 codes, and `Xs @ L1` in the fused
   quantization/down-projection kernel.
2. Run the pipelined W4A4 GEMM.
3. Add the low-rank up projection and bias in the GEMM epilogue.

Nunchaku INT4 uses group size 64 and a physical low-rank tile of 16. Rank 4 is
zero-padded to 16. Formal experiments should recalibrate activation group size
64; remapping a g128 activation checkpoint is permitted only for smoke tests.

### W4A16 decode

The production decode path uses two launches per Linear:

1. A small reduction computes `Xs = X / D` and `Z = Xs @ L1` once.
2. A packed-weight GEMV computes `Xs @ Wr.T`, then adds `Z @ L2 + bias`
   in its epilogue.

The GEMV reads W4 weights directly and never materializes a dense FP16 weight.
It specializes `M = 1, 2, 4, 8, 16` and rank 4/8. Division by `D`, W4 unpacking,
group scaling, low-rank up projection, and bias are fused into the GEMV load and
epilogue.

The MVP specializes rank 1 through 8 and accepts arbitrary positive `M`; the
dispatch policy keeps it on the small-M path. Shape-specialized vector loads
and CUDA-event timing are follow-up optimizations after the first cloud profile.

The MVP may use an existing W4A16 kernel plus separate low-rank operations. It
is not accepted as the production path unless profiling shows launch overhead is
already below 10% of decode latency.

## 7. Public Configuration

```text
--runtime-backend hybrid
--hybrid-policy auto|force_w4a4|force_w4a16
--hybrid-threshold <rows>          # optional override
--hybrid-autotune-cache <path>
--hybrid-weight-layout dual|shared
--hybrid-profile-stats
```

Implemented in the MVP: backend, policy, threshold, activation-group remap,
Nunchaku version, dual layout, call counts, and rows by kernel. Autotune cache
and per-kernel CUDA elapsed time are not yet implemented.

Every result file records:

```text
runtime.backend = "hybrid"
runtime.policy
runtime.weight_layout
runtime.kernel_counts
runtime.rows_by_kernel
runtime.autotune_key
runtime.nunchaku_version
```

There is no automatic eager fallback. A fallback that is requested explicitly
must be visible in both logs and result metadata.

## 8. Comparison Plan

Five backends are compared in the same environment:

1. Optimized FP16/BF16 baseline.
2. Pure W4A16.
3. Pure Nunchaku W4A4.
4. Hybrid W4A4/W4A16.
5. Current `hsvdq_cuda` as the correctness baseline.

Representative Qwen3-8B Linear shapes are:

| Projection | K | N |
| --- | ---: | ---: |
| Q / O | 4096 | 4096 |
| K / V | 4096 | 1024 |
| gate / up | 4096 | 12288 |
| down | 12288 | 4096 |

The microbenchmark covers all shapes and all dispatch `M` values. Model-level
tests cover prompt lengths 128, 512, and 2048; decode lengths 32 and 128; and
batch sizes 1, 4, and 8.

Required metrics are:

- TTFT and prefill tokens/s.
- TPOT and decode tokens/s.
- End-to-end generated tokens/s using the actual generated-token count.
- Peak allocated and reserved VRAM.
- Kernel count and time split from CUDA profiling.
- W4A4 relative error against the eager A4 operator.
- W4A16 relative error against an eager A16 reference.
- WikiText2 PPL and generation logit divergence for the mixed trajectory.

The existing decode benchmark must be corrected so that KV-cache growth is
intentional and measured as one continuous decode sequence, rather than reusing
an implicitly mutated cache while labeling every sample as the same operation.

## 9. Performance Gates

Continue from MVP to production only if all gates pass on the target GPU:

| Gate | Required result |
| --- | --- |
| W4A4 at M >= 256 | at least 1.5x faster than optimized FP16 GEMM path |
| W4A16 at M = 1 | at least 1.5x faster than optimized FP16 decode path |
| Hybrid end-to-end | at least 1.7x faster than optimized FP16 at batch 1 |
| Hybrid vs pure W4A4 | lower TPOT without worse prefill throughput |
| Hybrid vs pure W4A16 | lower TTFT without worse TPOT |
| MVP memory | below 11 GB for Qwen3-8B |
| Production memory | below 7 GB for Qwen3-8B, excluding KV growth |
| Quality | no material regression relative to the selected path references |

Expected mature performance is 2-3x end-to-end speedup over an optimized FP16
baseline. More than 4x is outside the credible limit of weight/activation
quantization alone because attention, KV-cache traffic, normalization, and host
dispatch remain.

## 10. Risks and Controls

| Risk | Control |
| --- | --- |
| Nunchaku is slow at small M | Never select it below the measured crossover |
| g128 to g64 changes A4 behavior | Recalibrate g64 for formal results |
| W4A16 changes decode numerics | Label mixed precision and run trajectory tests |
| Dual packing weakens memory benefit | Use it only for MVP; production gate requires shared layout |
| Wheel/API incompatibility | Pin and record an exact Nunchaku version |
| GPU-specific crossover | Autotune per SM, dtype, K, and N |
| Hidden eager fallback | Fail closed and record per-kernel call counts |
| Model integration hides kernel gains | Require both micro and end-to-end gates |

## 11. Delivery Order

1. Correct the latency benchmark and add per-shape microbenchmarks.
2. Restore a version-pinned Nunchaku W4A4 backend beside `hsvdq_cuda`.
3. Add the dual-layout W4A16 decode path and three-way forced comparison.
4. Add automatic threshold tuning and runtime statistics.
5. Build the shared-layout decode GEMV and remove duplicate weights.
6. Consider shared-`D` QKV and SwiGLU fusion only after the hybrid runtime passes
   its performance gates; those are algorithm/model changes, not prerequisites.

## 12. Cloud command

```bash
CHECKPOINT=outputs/qwen3-8b-v2-r4w4a4 \
MODEL=models/Qwen/Qwen3-8B \
DTYPE=float16 \
bash run_hybrid_runtime_smoke.sh
```

The first W4A16 forward JIT-compiles `csrc/hsvdq_hybrid`. The run writes backend
latencies, hybrid memory, dispatch counters, and computed speedups under
`$CHECKPOINT/hybrid-runtime-smoke` by default.
