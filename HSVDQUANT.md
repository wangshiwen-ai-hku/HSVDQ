# H-SVDQuant on Qwen3

`hsvdquant.py` implements the calibration and joint per-linear-layer loop

```text
stream all calibration batches -> H and bounded X reservoir
for k in outer iterations:
    D -> L1 -> Z (GPTQ) -> closed-form L2
```

The original Hessian `H = mean_b(X_b^T X_b)` is accumulated before quantization. `D` uses the
closed-form `lp` solution as its warm start; `--d-mode cached` then refines it against group-wise
weight ranges and per-token dynamic activation ranges using the bounded activation reservoir.

## Environment

Use Python 3.10 or 3.11 on a CUDA machine:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements-hsvdquant.txt
python hsvdquant.py self-test
python hsvdquant.py integration-test
```

Qwen3 support requires Transformers 4.52.4 or newer. The compact checkpoint references the base
Hugging Face model and therefore downloads `Qwen/Qwen3-0.6B` on first quantization/load.

## Fast smoke test

Quantize only the first decoder layer:

```bash
python hsvdquant.py quantize \
  --model Qwen/Qwen3-0.6B \
  --output outputs/qwen3-smoke \
  --calib-dataset wikitext2 \
  --nsamples 8 --sequence-length 128 --calib-batch-size 2 \
  --activation-cache-tokens 256 \
  --bits 4 --activation-bits 4 --rank 4 \
  --outer-iters 1 --d-steps 3 --max-layers 1 \
  --eval-tasks hellaswag --eval-limit 10
```

## Full calibration and lm-eval

```bash
bash run_hsvdquant_qwen3.sh
```

Or run the two stages explicitly:

```bash
python hsvdquant.py quantize \
  --model Qwen/Qwen3-0.6B \
  --output outputs/qwen3-hsvdquant \
  --nsamples 512 --sequence-length 512 \
  --bits 4 --activation-bits 4 --rank 8 \
  --beta 0.5 --p 2 --outer-iters 2

python hsvdquant.py eval \
  --checkpoint outputs/qwen3-hsvdquant \
  --tasks hellaswag,arc_easy,piqa,winogrande \
  --batch-size 4 \
  --output outputs/qwen3-hsvdquant/lm_eval_results.json
```

The `eval` subcommand passes the already-instantiated quantized model to lm-eval's Hugging Face
backend, so the custom FP low-rank plus W4A4 runtime remains active.

For packed W4A4 Tensor Core inference (no dense residual reconstruct, no
Nunchaku), use `--runtime-backend hsvdq_cuda`. See [`INT4_RUNTIME.md`](INT4_RUNTIME.md).

## Standard Hugging Face compatibility export

Add

```bash
--export-dense outputs/qwen3-hsvdquant-dense
```

to `quantize`. The resulting directory can be evaluated with the unmodified CLI:

```bash
lm_eval --model hf \
  --model_args pretrained=outputs/qwen3-hsvdquant-dense,dtype=bfloat16 \
  --tasks hellaswag,arc_easy --batch_size 4
```

This dense export folds `D`, `L1 L2`, and the dequantized residual into regular `nn.Linear`
weights. It is a portable W4A16 reconstruction for compatibility testing; use
`hsvdquant.py eval` when activation quantization must remain enabled.

## Artifacts

- `hsvdquant.pt`: compact int8 code tensors (4-bit values are not bit-packed on disk), group scales,
  smoothing vectors, and low-rank factors.
- `hsvdquant_config.json`: base model and calibration/quantization metadata.
- tokenizer files: enough to reconstruct the lm-eval wrapper after the base weights are pulled.

Calibration keeps `Z` in `int8` for correctness and debuggability. The optional
`hsvdq_cuda` backend packs those codes to W4 at load time and runs a fused W4A4
WMMA kernel; see [`INT4_RUNTIME.md`](INT4_RUNTIME.md).
