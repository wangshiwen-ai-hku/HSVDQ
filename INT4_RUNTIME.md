# Native H-SVDQuant CUDA runtime

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
