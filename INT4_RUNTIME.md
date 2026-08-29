# Packed R4W4A4 runtime

The accelerated path uses Nunchaku's signed INT4 tensor-core GEMM. It keeps
packed W4 weights on GPU and fuses dynamic A4 quantization with the padded
low-rank branch. H-SVDQuant rank 4 is zero-padded to the kernel's physical rank
tile of 16; the logical operator is unchanged.

## Compatibility

- CUDA GPU: RTX 4090 (`sm_89`) is supported by Nunchaku.
- Compute dtype: `float16` or `bfloat16`.
- Weight/activation precision: W4A4.
- Linear input and output dimensions: multiples of 128 (Qwen3-8B satisfies this).
- Nunchaku activation groups are fixed at 64.

For an existing activation-group-128 checkpoint, pass
`--allow-activation-group-remap`. Weight group 128 is converted exactly by
reusing each scale for its two group-64 MMA tiles. Activation quantization is
explicitly changed from group 128 to group 64, so report it as an inference
remap. For a strict algorithm/runtime match, recalibrate with
`--activation-group-size 64` and omit the remap flag.

## Cloud smoke test

Install a Nunchaku wheel that matches the Python and PyTorch versions in the
`pde` environment, then run:

```bash
conda activate pde
bash run_r4w4a4_int4_smoke.sh
```

The script performs a CUDA numerical check, records full-GPU memory and latency
without decoder-layer CPU offload, and runs 32 PIQA samples. Override `MODEL`,
`CHECKPOINT`, `OUTPUT`, `DEVICE`, `DTYPE`, or `PYTHON_BIN` as needed.
