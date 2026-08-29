# V3-OAR activation-tail toy experiment

All metrics are evaluated on unseen synthetic tokens. Lower is better for NMSE/CVaR.

## stable_scattered_tail

one persistent high-sensitivity tail channel per native group; range clustering should isolate pollution

Grouping selected on calibration holdout: `cost`.
Full V3 held-out admission: `True`.
Block Hadamard held-out admission: `True` (sign trial `2`).

| method | NMSE | CVaR | activation MSE | A16 MSE | cross / total | scale p99 |
|---|---:|---:|---:|---:|---:|---:|
| v2 | 1.7046e-03 | 3.2892e-01 | 3.8642e-02 | 2.1166e-02 | -0.35% | 9.7780e-01 |
| v2_plus | 1.6093e-03 | 3.2449e-01 | 3.8849e-02 | 1.7457e-02 | -0.07% | 9.7780e-01 |
| v3_sort | 2.3417e-03 | 6.0774e-01 | 4.9866e-02 | 3.0782e-02 | 1.50% | 9.7703e-01 |
| v3_cost | 1.4573e-03 | 2.9496e-01 | 3.1523e-02 | 1.9151e-02 | 0.55% | 9.7780e-01 |
| v3_full | 1.4251e-03 | 2.3307e-01 | 2.9446e-02 | 2.0512e-02 | -0.26% | 8.5656e-01 |
| v3_block_hadamard | 1.0058e-03 | 1.3325e-01 | 1.9099e-02 | 1.5751e-02 | 0.90% | 2.9448e-01 |
| g_half_reference | 1.1071e-03 | 1.8798e-01 | 2.1601e-02 | 1.6941e-02 | 0.43% | 8.4149e-01 |

## sensitivity_mismatch

largest-amplitude rows have low output sensitivity while moderate tails multiply large residual rows; amax alone is mis-specified

Grouping selected on calibration holdout: `cost`.
Full V3 held-out admission: `True`.
Block Hadamard held-out admission: `True` (sign trial `3`).

| method | NMSE | CVaR | activation MSE | A16 MSE | cross / total | scale p99 |
|---|---:|---:|---:|---:|---:|---:|
| v2 | 1.1074e-03 | 1.0476e+00 | 6.2613e-02 | 3.1593e-02 | -1.71% | 1.3289e+00 |
| v2_plus | 1.0861e-03 | 1.0521e+00 | 6.2657e-02 | 2.8597e-02 | -0.44% | 1.3289e+00 |
| v3_sort | 2.1143e-03 | 1.4469e+00 | 1.1691e-01 | 6.1796e-02 | -1.05% | 1.2133e+00 |
| v3_cost | 1.0275e-03 | 8.2702e-01 | 5.0662e-02 | 3.5856e-02 | -0.66% | 1.3289e+00 |
| v3_full | 1.0632e-03 | 6.6771e-01 | 5.1065e-02 | 3.8107e-02 | -0.27% | 1.0514e+00 |
| v3_block_hadamard | 5.1972e-04 | 3.5841e-01 | 2.3882e-02 | 1.9099e-02 | 1.13% | 5.0405e-01 |
| g_half_reference | 7.0122e-04 | 4.7795e-01 | 3.4596e-02 | 2.4932e-02 | -1.49% | 1.2462e+00 |

## unstable_tail_location

tail identity shifts between grouping-fit and admission/test tokens; negative control for the held-out admission requirement

Grouping selected on calibration holdout: `identity`.
Full V3 held-out admission: `False`.
Block Hadamard held-out admission: `True` (sign trial `0`).

| method | NMSE | CVaR | activation MSE | A16 MSE | cross / total | scale p99 |
|---|---:|---:|---:|---:|---:|---:|
| v2 | 1.2912e-03 | 5.8666e-01 | 4.3908e-02 | 4.8437e-02 | -1.81% | 1.8827e+00 |
| v2_plus | 1.0666e-03 | 4.5551e-01 | 4.3371e-02 | 3.2806e-02 | -1.67% | 1.8827e+00 |
| v3_sort | 1.8851e-03 | 8.6756e-01 | 4.6689e-02 | 8.6011e-02 | -0.21% | 1.8719e+00 |
| v3_cost | 1.3316e-03 | 5.8871e-01 | 4.6168e-02 | 4.9290e-02 | -2.05% | 1.8734e+00 |
| v3_full | 1.0666e-03 | 4.5551e-01 | 4.3371e-02 | 3.2806e-02 | -1.67% | 1.8827e+00 |
| v3_block_hadamard | 6.6952e-04 | 2.7454e-01 | 1.9974e-02 | 2.7076e-02 | -0.04% | 5.6554e-01 |
| g_half_reference | 7.5578e-04 | 2.8695e-01 | 2.4081e-02 | 3.0324e-02 | -2.47% | 1.6712e+00 |
