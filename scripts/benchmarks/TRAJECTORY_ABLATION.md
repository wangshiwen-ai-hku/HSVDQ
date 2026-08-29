# Teacher--Student Trajectory Ablation

> **Deprecated:** the trajectory-based variant formerly named V3 failed validation and is not
> part of the current method. Keep this document and its runner only for reproducing the negative
> result. The new V3 design is the local outlier-aware grouping and FP-routing method documented in
> `hsvdquant/V3_OUTLIER_ROUTING.md`; it does not use teacher--student trajectory correction.

This experiment isolates the three algorithmic generations while keeping the
deployed H-SVDQuant operator, rank, weight grouping, calibration budget, seed,
and activation quantizer fixed.

## Variants

| label | activation | local joint objective | FP teacher target | resolved preset |
|---|---:|---:|---:|---|
| `a16` | A16 | no | no | W4A16/W3A16 weight-only reference |
| `v1` | A4 | no | no | original local reconstruction |
| `v2` | A4 | yes, `F_W + lambda F_A` | no | V2 only |
| `v3` | A4 | no | yes | V3 only |
| `v2v3` | A4 | yes | yes | V2 and V3 composed |

`--ablation-mode` in `hsvdquant.py` resolves these settings atomically.  In
particular, V1 and V3 set `activation_weight=0`, `code_objective=fw`, and
`rank_a=0`, whereas V2 and V2+V3 use the requested positive lambda and the
joint code target.  All A4 rows still execute the identical A4 runtime
quantizer; disabling lambda does not disable activation quantization.

All variants use `block_input_mode=quantized`.  Consequently every block sees
the real output of preceding quantized blocks.  The distinction is whether the
current linear reconstructs its same-cache output (`local`) or an aligned FP
teacher output (`cumulative`).

The finite-cache trajectory solve uses the paired student Hessian, a dedicated
ridge (`--trajectory-damp`, default 0.1), an exact one-dimensional line search,
and a Frobenius trust region (`--trajectory-max-norm-ratio`, default 0.25).
The zero-correction weight is always retained as a feasible fallback, so the
accepted projected target cannot increase paired teacher MSE.  These controls
implement the regularized estimator required by the finite-sample theory; they
do not change the ideal population target.

## Formal matrix

- Model: Qwen3-0.6B unless overridden.
- Weight bits: W4 and W3.
- Activation bits: A4 for V1/V2/V3/V2+V3; A16 for the weight-only baselines.
- Calibration corpora: WikiText-2 train and C4 train.
- Evaluation corpora: WikiText-2 test and the fixed C4 validation shard.
- Default rank: 16.
- Default lambda: 0.25, matching the stable prior joint-objective runs; lambda
  1.0 is retained as a sensitivity point rather than the primary setting.
- Default calibration: 128 samples x 512 tokens, seed 0.
- PPL: non-overlapping 2048-token windows; full evaluation by default.

This is 20 quantized checkpoints and 40 PPL evaluations.  Both calibration
corpora are evaluated on both test corpora, separating calibration-domain gains
from generalization.

## Primary estimands

For each `(calibration corpus, evaluation corpus, weight bits)` cell, report:

```text
V2 effect          = PPL(V2)    - PPL(V1)
V3 effect          = PPL(V3)    - PPL(V1)
combined effect    = PPL(V2+V3) - PPL(V1)
factorial interaction
                   = PPL(V2+V3) - PPL(V2) - PPL(V3) + PPL(V1)
activation gap     = PPL(A4 variant) - PPL(A16 baseline)
```

Negative values are improvements.  The interaction tests whether the local
activation-aware objective and cross-trajectory correction are complementary
(negative), redundant (positive), or approximately additive (near zero).

## Run

```bash
/homedata/swwang/conda/envs/svdquant/bin/python \
  scripts/benchmarks/run_trajectory_ablation.py \
  --devices cuda:0 cuda:1 cuda:2 cuda:3 \
  --rank 16 \
  --activation-weight 0.25 \
  --trajectory-damp 0.1 \
  --trajectory-max-norm-ratio 0.25 \
  --output-root results/trajectory_ablation_r16_lam025_stable
```

The runner assigns one serial queue to each GPU, prefetches the fixed C4 train
and validation shards once, resumes completed checkpoints/metrics by default,
and writes:

- `experiment_manifest.json`: exact arguments and job-to-GPU assignment;
- `logs/*.log`: one quantization/evaluation log per checkpoint;
- `summary/ppl_rows.csv`: all raw PPL values;
- `summary/factorial_effects.csv`: V2, V3, combined, and interaction effects;
- `report.md`: compact comparison table.

For a fast pipeline check, add `--max-layers 1 --nsamples 2
--activation-cache-tokens 32 --ppl-max-samples 1`.  Such partial-layer PPL is a
software smoke test only and must not be reported as a formal result.
