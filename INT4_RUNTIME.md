# Native H-SVDQuant CUDA runtime

## Hybrid accelerated runtime

The `hybrid` backend combines two packed implementations over the same
checkpoint:

- `M >= --hybrid-threshold`: Nunchaku W4A4 for prefill and large batches.
- `M < --hybrid-threshold`: repository-local W4A16 GEMV for token decode.

The W4A16 kernel does not reconstruct an FP16 weight. Its first launch computes
the smoothed activation and rank projection once; its second launch streams the
packed W4 residual and fuses scale application, low-rank up projection, and
bias. Version 1 keeps both Nunchaku and row-major W4 layouts in VRAM so the
performance hypothesis can be tested before implementing a shared layout.

Install a Nunchaku wheel matching the cloud Python, PyTorch, and CUDA versions,
then run the complete comparison:

```bash
conda activate pde
CHECKPOINT=outputs/qwen3-8b-v2-r4w4a4 \
  bash run_hybrid_runtime_smoke.sh
```

The script runs dense, pure W4A16, pure Nunchaku W4A4, and automatic hybrid
latency tests. It also writes `linear_crossover.json` with per-shape kernel
crossovers and `summary.json` with end-to-end speedups. Set
`ALLOW_ACTIVATION_GROUP_REMAP=0` for a checkpoint calibrated with activation
group 64. The default value `1` lets the existing group-128 checkpoint run, but
changes its activation quantization and is suitable only for performance smoke
testing.

Useful controls:

```bash
HYBRID_THRESHOLD=64 RUN_PURE_W4A4=1 RUN_HSVDQ_CUDA=0 \
  bash run_hybrid_runtime_smoke.sh
```

The runtime fails instead of falling back to eager execution. Use
`--hybrid-profile-stats` to record the W4A4/W4A16 call and row counts in result
JSON files. See [`HYBRID_RUNTIME_DESIGN.md`](HYBRID_RUNTIME_DESIGN.md) for the
operator contract, acceptance gates, and remaining production work.

`hsvdq_cuda` is a repository-local packed inference backend. It has no
Nunchaku dependency and does not reconstruct a dense FP16 residual weight.
Calibration still writes compact `int8` codes in `hsvdquant.pt`; packing to
W4 happens at load time when `--runtime-backend hsvdq_cuda` is set.

## Implemented kernel

- Operator: signed W4A4 plus the FP16/BF16 low-rank branch and bias.
- Weight and activation group size: 128, without inference-time remapping.
- Logical low rank: 1 through 8; rank 4 has no physical rank-16 padding.
- GPU: CUDA sub-byte WMMA on SM75 through SM89, including RTX 4090 SM89.
- Compute dtype: FP16 or BF16.
- Shape: K divisible by 128 and N divisible by 8.

The forward allocates packed A4 (`M*K/2` bytes), per-token group scales, and
an `M*rank` FP32 low-rank intermediate. It does not allocate `M*K` smoothed
activations or an `N*K` dense residual. This is the path to use for measuring
the full-GPU 8B peak and latency on a 4090. Decoder-layer CPU paging should be
disabled for that measurement; it measures PCIe transfer latency instead of
the kernel. Offload remains opt-in (`--cpu-offload-layers`) only if the
packed model still OOMs.

The initial GEMM is a correctness-oriented 8x8x32 sub-byte WMMA kernel. Its
packed state and dispatch key are stable; a later 16x8x64 pipelined kernel can
replace it without changing checkpoint files or model loading.

### Experimental V3 activation transforms

The eager `HSVQuantLinear` runtime accepts an optional static
`activation_permutation` in a layer state. It applies the permutation only to
the activation/residual path; `L1/L2` remain in the original smoothed input
coordinates. `dense_weight()` maps the quantized residual back before folding.

It also accepts `activation_hadamard_group_size` plus optional
`activation_hadamard_signs`. The eager path applies a normalized randomized
block transform `R = D H` before activation quantization; residual codes are
stored in the matching `R^T W` coordinates. `dense_weight()` inverts both the
Hadamard and permutation transforms.

The native kernel intentionally rejects either transform for now. Arbitrary
index loads would invalidate its contiguous/coalesced K path, while Hadamard
requires a fused FWHT before A4 packing. Native support must fold the
permutation into the producer layout or fuse an admitted gather and transform.
It must not silently run a numerically different layout. The eager toy
experiment is the admission test before that kernel work.

## WxAx extension

`PackedQuantSpec` and canonical signed 2/4/8-bit packing are independent of
the W4A4 implementation. New kernels register by
`(weight_bits, activation_bits, weight_group_size, activation_group_size)`.
Unsupported combinations fail with the available kernel keys instead of
silently running an eager fallback. Useful next targets are W8A8 via INT8 MMA,
then mixed W4A8; W2A2 needs a separate unpack/MMA design.

CUDA's experimental sub-byte WMMA is unavailable on SM90 and newer. A future
SM90 backend should register a CUTLASS/CuTe kernel under the same public spec.

## Build

JIT compile uses `torch.utils.cpp_extension` on first CUDA forward:

- PyTorch CUDA build and `nvcc` must match. This machine's default `nvcc` is
  13.3 while `torch 2.8.0+cu126` needs CUDA 12.6. The smoke script sets
  `CUDA_HOME=/usr/local/cuda-12.6` when that directory exists; override with
  `CUDA_HOME` otherwise.
- Set `TORCH_CUDA_ARCH_LIST` to the card (4090: `8.9`). Sub-byte WMMA is
  compiled only for SM75–SM89.
- Install `ninja` in the Python env (`pip install ninja`).

```bash
conda activate pde
export CUDA_HOME=/usr/local/cuda-12.6
export PATH="$CUDA_HOME/bin:$PATH"
export TORCH_CUDA_ARCH_LIST=8.9
```

## Tests

CPU-only format and dispatch checks:

```bash
python scripts/benchmarks/verify_hsvdq_cuda_runtime.py
```

V3-OAR local theory and activation-tail toy:

```bash
python scripts/benchmarks/verify_v3_outlier_routing.py
python scripts/benchmarks/toy_v3_outlier_routing.py
```

The toy writes its held-out ablation to
`hsvdquant/toy/results/v3_outlier_routing/`.

On the 4090, the first run JIT-compiles the local extension, then compares
native CUDA output with the eager operator, records full-GPU memory and
latency, and evaluates 32 PIQA samples. Do not pass `--cpu-offload-layers`.

```bash
conda activate pde
CHECKPOINT=outputs/qwen3-8b-v2-r4w4a4 DEVICE=cuda:0 \
  bash run_r4w4a4_int4_smoke.sh
```

If GPU 0 is occupied, pin a free card with `CUDA_VISIBLE_DEVICES`:

```bash
CUDA_VISIBLE_DEVICES=1 CHECKPOINT=outputs/qwen3-8b-v2-r4w4a4 DEVICE=cuda:0 \
  bash run_r4w4a4_int4_smoke.sh
```

Override `MODEL`, `CHECKPOINT`, `OUTPUT`, `DEVICE`, `DTYPE`, `PYTHON_BIN`, or
`CUDA_HOME` as needed.

### 4090 smoke (Qwen3-8B, G128, rank 4, no offload)

Measured on an RTX 4090 with the group-128 checkpoint
`outputs/qwen3-8b-v2-r4w4a4` and `float16`:

| Check | Result |
| --- | --- |
| 2/4/8-bit packing roundtrip | lossless |
| Native vs eager, M=1/17/22 | relative L2 about 5e-4 to 1.1e-2 |
| Load / prefill / decode peak | 5.68 / 5.76 / 5.80 GB |
| 1-sample WikiText2 PPL (seq 512) | 10.10, peak 6.23 GB |
| Prefill 256 tokens | 427 ms (about 600 tok/s) |
| Decode one token | 417 ms (about 2.4 tok/s) |
| PIQA, 32 samples | acc 78.1%, acc_norm 65.6% |

Prefill and decode are nearly the same because the 8x8x32 kernel is a
correctness/baseline implementation, not a decode-specialized one. Packed
W4 fits in about 6 GB, so 8B full-GPU measurements on a 4090 should keep
layer offload off.

## Qwen3-8B eval driver

`run_v2_r4w4a4_qwen3_8b_eval.sh` defaults to `--runtime-backend hsvdq_cuda`
and does not enable layer offload for eval. The current 8x8x32 WMMA kernel
is a packed-memory / correctness path, not a fast decode kernel (~2.4 tok/s
on 4090). For accuracy eval (especially GSM8K generate), **eager with
`--persist-qweight` is faster**: dequant residual once, then cuBLAS FP16
GEMM, and keep KV cache on. The driver does that automatically when the
checkpoint is not W4A4 g128.

Useful env knobs:

- `RUNTIME_BACKEND` — `hsvdq_cuda` (default for g128) or `eager`
- `BATCH_SIZE` — default 8 (loglikelihood tasks); lower if GSM8K OOMs
- `PERSIST_QWEIGHT` — default 1 for eager (skip per-token residual dequant)
- `GROUP_SIZE` / `ACTIVATION_GROUP_SIZE` / `D_CLIP` — calibration only
- `SKIP_QUANTIZE`, `SKIP_PPL`, `SKIP_FP16_EVAL`, `SKIP_QUANT_EVAL`
- `FP16_METRICS_SOURCE` — copy existing FP16 metrics into `$OUTPUT/metrics`

The native kernel requires group size 128. A G64 checkpoint is switched to
eager instead of failing dispatch. Resume is per-task:
`$OUTPUT/metrics/lm_eval_quantized_<task>.json`.

G64 accuracy eval:

```bash
OUTPUT=outputs/qwen3-8b-v2-r4w4a4-g64 \
GROUP_SIZE=64 ACTIVATION_GROUP_SIZE=64 \
SKIP_QUANTIZE=1 SKIP_PPL=1 SKIP_FP16_EVAL=1 \
FP16_METRICS_SOURCE=outputs/qwen3-8b-v2-r4w4a4 \
./run_v2_r4w4a4_qwen3_8b_eval.sh
```

G128 packed-kernel eval (memory / kernel check, slower GSM8K):

```bash
OUTPUT=outputs/qwen3-8b-v2-r4w4a4 RUNTIME_BACKEND=hsvdq_cuda \
SKIP_QUANTIZE=1 SKIP_PPL=1 SKIP_FP16_EVAL=1 \
./run_v2_r4w4a4_qwen3_8b_eval.sh
```
