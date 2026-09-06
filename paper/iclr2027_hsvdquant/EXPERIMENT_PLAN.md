# H-SVDQuant ICLR 2027 Experiment Plan

## Decision

Paper design and experiments start in parallel. Accuracy is measured with the
eager/reference implementation and does not wait for the fused runtime. Runtime
claims remain placeholders until a numerically equivalent packed backend beats
the common FP16 baseline.

## Claim-to-evidence map

| Claim | Minimum evidence | Blocks submission if absent? |
| --- | --- | --- |
| Hessian geometry improves low-rank-assisted W4A4 | Matched SVDQuant/LRC-style baseline vs H-SVDQuant on two model families | Yes |
| Both smoothing and low-rank correction matter | 2x2 mechanism ablation | Yes |
| Hadamard is complementary | 2x2 method x transform comparison | No, but useful |
| Method is practical at rank 4 | Rank/accuracy/memory Pareto curve | Yes |
| Quantized inference accelerates deployment | Positive model wall speedup with no offload | Yes, only if speed is claimed |
| Fusion explains the gain | Operator tax and event-count ablation | Yes, only for systems claim |

## Terminology lock

- **H-SVDQuant**: Hessian-aware smoothing, Hessian-aware low-rank correction,
  and joint residual-code objective.
- **g128**: comparison protocol. Never call it a contribution.
- **Hadamard**: external orthogonal transform. Report `H-SVDQuant + Hadamard`.
- **V2**: internal implementation label only. Use H-SVDQuant in the paper.
- **Fused runtime**: implementation result only after all accuracy equivalence
  and performance gates pass.

## Track A: accuracy

### A0. Reproducibility gates

1. Freeze the Qwen3-8B checkpoint hash, tokenizer revision, calibration corpus,
   128 calibration sequences, sequence length 512, seed, and lm-eval version.
2. Run cached/no-cache equivalence on FP16, H-SVDQuant, and H-SVDQuant+Hadamard.
3. For generation tasks, record the first token position where logits or greedy
   tokens diverge. Do not report Hadamard GSM8K until this check is resolved.
4. Store one JSON per task plus a manifest containing code commit and package
   versions.

### A1. Minimum main-table baselines

Run under each method's legitimate arithmetic, and label it explicitly:

| Method | Arithmetic | Purpose |
| --- | --- | --- |
| FP16/BF16 | W16A16 | reference |
| RTN | W4A4 g128 | lower bound |
| GPTQ | W4A16 g128 | weight-only accuracy/runtime anchor |
| AWQ | W4A16 g128 | weight-only activation-aware anchor |
| SmoothQuant | W4A4 or supported lowest activation bit | smoothing baseline |
| QuaRot or SpinQuant | W4A4 | strong rotation baseline |
| LRC | W4A4 with disclosed rank | LLM low-rank baseline |
| adapted SVDQuant | W4A4 r4 g128 | closest structural baseline |
| H-SVDQuant | W4A4 r4 g128 | proposed method |

Do not put W4A16 and W4A4 in one accuracy column without an arithmetic column.
If an official baseline does not support r4 or g128, report its official setting
and a matched reimplementation separately.

### A2. Factorial test for the actual contribution

Run all four cells with the same calibration data and quantizer:

| Low-rank/smoothing method | No transform | Block Hadamard |
| --- | --- | --- |
| non-Hessian SVDQuant-style | required | required |
| H-SVDQuant | required | required |

Report:

- Hessian gain without rotation: `H-SVDQuant - SVDQuant`.
- Hessian gain with rotation: `(H-SVDQuant + H) - (SVDQuant + H)`.
- Hadamard gain on the baseline and on H-SVDQuant separately.
- Difference-in-differences to show whether the mechanisms are redundant or
  complementary.

### A3. Mechanism ablation

Use one common residual quantizer and run:

1. heuristic smoothing + Euclidean SVD;
2. Hessian smoothing + Euclidean SVD;
3. heuristic smoothing + Hessian low-rank correction;
4. Hessian smoothing + Hessian low-rank correction;
5. row 4 + joint `F_W + lambda F_A` code objective.

This table is more important than adding a third minor baseline because it
demonstrates where the proposed gain comes from.

### A4. Models and metrics

Main paper:

- Qwen3-8B.
- Llama-3.1-8B or another architecture already supported by the loader.
- WikiText-2 and C4 perplexity.
- MMLU, GSM8K, ARC-Challenge, ARC-Easy, HellaSwag, PIQA.

Appendix if compute permits:

- one smaller model for broad ablations;
- one 14B-class model for scaling;
- rank `0, 2, 4, 8, 16`;
- group size `64, 128, 256`;
- calibration samples `32, 64, 128, 256`;
- `lambda`, outer-iteration, and solver-step sensitivity;
- three seeds for the selected configuration and rotation signs.

### A5. Accuracy acceptance gates

- Two model families completed with identical task protocol.
- All four factorial cells completed.
- H-SVDQuant improves the matched non-Hessian baseline consistently, not only
  the FP16 gap.
- Cache/no-cache issue explained or excluded with a documented reason.
- Every table can be regenerated from raw JSON by one aggregation command.

## Track B: performance

### B0. Numerical gates

- Packed W4A4/W4A16 output matches the eager quantized contract at each supported
  shape and row count.
- Relative L2 and max-absolute thresholds are recorded.
- No silent dense reconstruction, CPU offload, or activation-group remapping.

### B1. Three mandatory scopes

1. `operator_pure`: complete quantized Linear contract, including smooth,
   activation quantization, packed multiplication, low rank, and bias.
2. `model_gpu`: accumulated CUDA execution in a continuous cached-decode window,
   including attention, KV, RMSNorm, and RoPE but excluding CPU gaps.
3. `model_wall`: synchronized wall time for the same window, including framework
   dispatch but excluding load/tokenization/network/offload.

Tier 1 proves kernel quality. Tier 2 shows GPU-level composition. Tier 3 is the
deployment headline. None can substitute for another.

### B2. Backend matrix

- optimized FP16/BF16 baseline with the same attention backend;
- GPTQModel Marlin W4A16;
- Nunchaku/SVDQ W4A4 where shapes are supported;
- current H-SVDQuant eager/native paths as diagnostic rows;
- grouped/fused H-SVDQuant W4A16 decode path;
- grouped/fused H-SVDQuant W4A4 prefill path;
- hybrid dispatch only after both component paths are individually measured.

### B3. Shape and workload matrix

- Linear rows `M = 1, 2, 4, 8, 16, 64, 128, 256, 1024`.
- Q/O `4096x4096`, K/V `4096x1024`, Gate/Up `4096x12288`, Down
  `12288x4096`, plus exact second-model shapes.
- Batch 1 primary; batch `2, 4, 8` secondary.
- Prompt lengths `128, 256, 1024, 2048`.
- Decode lengths `32, 128, 256`.
- RTX 4090 primary; add A100/H100/L40S if available.

### B4. Fusion milestones

1. Group Q/K/V and Gate/Up while preserving branch-specific `D`, scales,
   `L1`, and `L2`.
2. Fuse `X/D` and `X L1` into activation preparation.
3. Fuse low-rank `Z L2` and bias into the packed residual epilogue.
4. Reduce logical projection launches from seven to four per block; record the
   actual CUDA event count.
5. Apply CUDA Graph/framework optimizations equally to FP16 and quantized runs.

### B5. Performance gates

- Operator weighted speedup at M=1: target at least `2.0x`; report shape-wise
  failures instead of only the weighted average.
- Model GPU decode speedup: target at least `1.5x`.
- Model wall decode speedup: target at least `1.3x` on RTX 4090.
- No regression in numerical equivalence or peak memory.
- Median, mean, p95, and variation from at least five warmed windows.

These are engineering acceptance targets, not paper results.

## Paper table allocation

Main text:

1. W4A4 accuracy across two model families.
2. Downstream task accuracy for the primary model.
3. Factorial Hessian x Hadamard table.
4. Mechanism ablation.
5. End-to-end performance and memory.
6. Operator/fusion ablation.

Appendix:

- full task rows and seeds;
- rank/group/calibration sweeps;
- layerwise error analysis;
- cached/no-cache diagnosis;
- hardware and profiler details;
- all shape-level latency results.

## Stop/go checkpoints

- **Go for method paper**: matched accuracy gain holds on two model families even
  if fused runtime is not ready. Frame the work as PTQ quality/structure and
  remove acceleration claims from title and contributions.
- **Go for method + systems paper**: method gate passes and model wall speedup is
  positive under the common optimized baseline.
- **Stop and revise**: gains exist only after Hadamard, disappear against the
  matched SVDQuant/LRC baseline, or depend on cached-evaluation artifacts.
