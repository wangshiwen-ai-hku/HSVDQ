# Native H-SVDQuant CUDA runtime

`hsvdq_cuda` is a repository-local packed inference backend. It has no
Nunchaku dependency and does not reconstruct a dense FP16 residual weight.

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
the kernel.

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

## Tests

CPU-only format and dispatch checks:

```bash
python scripts/benchmarks/verify_hsvdq_cuda_runtime.py
```

On the 4090 server, the first run JIT-compiles the local extension:

```bash
conda activate pde
TORCH_CUDA_ARCH_LIST=8.9 bash run_r4w4a4_int4_smoke.sh
```

The smoke script compares native CUDA output with the eager operator, records
full-GPU memory and latency, and evaluates 32 PIQA samples. Set `MODEL`,
`CHECKPOINT`, `OUTPUT`, `DEVICE`, `DTYPE`, or `PYTHON_BIN` to override defaults.
