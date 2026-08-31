# MoEJointQuant: first real-model experiment on one RTX 4090

## Decision

Use `allenai/OLMoE-1B-7B-0125` as the first target.

The model has about 7B total parameters and 1B active parameters per token. Its
Hugging Face implementation exposes 64 routed experts, top-8 routing, 16
decoder layers, hidden size 2048, and expert intermediate size 1024. The BF16
checkpoint is small enough for one 24 GB RTX 4090 when only one model instance
is resident and calibration tensors are immediately moved to CPU.

Start with the base checkpoint, not the Instruct checkpoint. The base model is
the cleaner target for block reconstruction and perplexity. Use
`allenai/OLMoE-1B-7B-0125-Instruct` only after the block-level mechanism passes
and task-level evaluation is needed.

Authoritative references:

- Model card: <https://huggingface.co/allenai/OLMoE-1B-7B-0125>
- Transformers documentation: <https://huggingface.co/docs/transformers/model_doc/olmoe>
- Configuration: <https://huggingface.co/allenai/OLMoE-1B-7B-0125/blob/main/config.json>
- HF implementation: <https://github.com/huggingface/transformers/blob/main/src/transformers/models/olmoe/modeling_olmoe.py>

Do not start with Qwen1.5-MoE-A2.7B or DeepSeek-MoE-16B. Their total BF16
weights exceed a 4090's VRAM, so CPU offload would confound the first algorithm
verification with memory and runtime problems.

## What this experiment must prove

The first real-model experiment has one narrow scientific question:

> With identical fixed quantization grids and identical calibration tokens,
> does a cross-expert joint quadratic objective reduce the routed MoE block
> output error below expert-local GPTQ?

It does not yet need to prove end-to-end task gains, optimal calibration data,
or production speed. Do not change router weights in this experiment. Quantize
only the expert `down_proj` tensors.

Report these baselines on exactly the same slices and scales:

1. RTN.
2. Expert-local GPTQ.
3. Gate-affinity-weighted expert-local GPTQ.
4. Dense joint GPTQ.
5. Joint discrete coordinate descent initialized from baseline 3.

The primary metric is held-out MoE block output NMSE with the FP router and its
original top-k weights frozen. Secondary metrics are cosine error, downstream
next-layer router top-k disagreement, validation perplexity, and peak memory.

## Why the first experiment is sliced

For one OLMoE down projection,

```text
W_e: [1024, 2048]
E:   64
P = E * 1024 = 65,536
H:   [65,536, 65,536]
```

A dense FP32 joint Hessian alone is about 16 GiB. The reference solver is also
quadratic in `P`, so a full-layer run is neither a valid 4090 experiment nor the
right first verification.

Use an exact restricted subproblem while holding every non-selected weight
fixed:

```text
all 64 experts
input channels per expert s = 8, then 16, then 32
output channels m = 64, then 128
P = 64 * s
```

At `s=16`, `P=1024` and the FP32 Hessian is about 4 MiB. At `s=32`, `P=2048`
and it is about 16 MiB. This preserves cross-expert Hessian blocks across all
experts while making the exact objective inspectable. This is a verifier, not
a claim that the production algorithm is already solved.

Choose one shared input-channel index set across experts. Run both a seeded
random slice and a high-activation-energy slice. Choose output channels with a
seeded random index set. Keep the chosen indices fixed across all methods.

## Repository entry points

- Theory and derivation: `hsvdquant/moejointquant.pdf`
- LaTeX source: `hsvdquant/moejointquant.tex`
- Reference implementation: `moejointquant.py`
- Main callable: `quantize_all_methods(...)`
- Feature construction: `augmented_features(...)`
- Joint Hessian: `joint_hessian(...)`
- Monotone correction: `joint_coordinate_descent(...)`
- Fixed full-weight scales: `expert_scales(...)`

The reference implementation uses CPU `float64` intentionally. Keep it that
way for the first verifier. GPU is used to run OLMoE and collect activations;
the selected quadratic subproblem is solved on CPU.

## Cloud setup

```bash
git fetch origin
git switch --track origin/codex/moejointquant
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch transformers accelerate datasets sentencepiece
python moejointquant.py --self-test
python moejointquant.py --seeds 0 1 2 3 4
```

Use a recent Transformers release with OLMoE support. The model card specifies
Transformers 4.45 or newer.

## Implementation task for the cloud agent

Create `scripts/benchmarks/moejointquant_olmoe.py`. Do not rewrite
`moejointquant.py`; import its verified routines.

The script should support at least:

```text
--model allenai/OLMoE-1B-7B-0125
--layer 0,7,15
--bits 2,3,4
--calib-sequences 32
--sequence-length 256
--sampled-tokens 2048
--input-slice 8,16,32
--output-slice 64,128
--slice-policy random,energy
--seed 0
--output-dir outputs/moejointquant/olmoe
```

Load only one model:

```python
model = AutoModelForCausalLM.from_pretrained(
    args.model,
    torch_dtype=torch.bfloat16,
    device_map={"": 0},
    attn_implementation="sdpa",
)
model.eval()
model.config.use_cache = False
```

If direct placement exceeds memory on the particular runtime, use
`device_map="auto"` with a 21 GiB GPU limit and CPU offload. Record this as a
different hardware condition; do not mix its timing with direct-GPU runs.

Capture the input to `model.model.layers[layer].mlp` with a forward pre-hook.
Verify the module path by printing `named_modules()` instead of silently
assuming it. Flatten only non-padding token states and reservoir-sample the
requested number of tokens. Move each sampled batch to CPU immediately.

For OLMoE, the expert weights have the following orientation:

```python
experts = model.model.layers[layer].mlp.experts
w_full = experts.down_proj.detach().float().cpu().transpose(1, 2).contiguous()
# w_full: [64, 1024, 2048], matching moejointquant.py
```

Reconstruct the FP router and active expert hidden states using the HF source
as the specification. For every sampled token, build:

```text
gates:              [N, 64], zero for inactive experts
expert_activations: [N, 64, 1024], zero for inactive experts
```

The activation before `down_proj` is the SiLU-gated product from
`gate_up_proj`. Preserve HF's top-k normalization behavior exactly. Add a unit
check that the reconstructed expert sum agrees with the original MLP output to
relative MSE below `1e-5` in FP32 accumulation before doing any quantization.

Construct the fair sliced problem as follows:

```python
from moejointquant import QuantConfig, expert_scales, quantize_all_methods

full_scales = expert_scales(w_full.double(), bits)
w_sub = w_full[:, input_idx][:, :, output_idx].double()
a_sub = expert_activations[:, :, input_idx].double()
s_sub = full_scales[:, input_idx][:, :, output_idx].double()

methods = quantize_all_methods(
    w_sub,
    a_sub,
    gates.double(),
    QuantConfig(bits=bits, joint_sweeps=12),
    fixed_scales=s_sub,
)
```

Never derive scales from `w_sub`: scales must come from the complete 1024-row
expert tensors. Never enumerate the exact discrete optimum when `P > 10`; the
enumerator in the reference file is only a toy certificate.

## Evaluation protocol

Use two disjoint token sets from a public text corpus: calibration and held-out
validation. Initially use random text, because the question is about the joint
objective rather than calibration-set design. Save sampled document IDs, token
indices, routes, and channel indices for reproducibility.

For each method:

1. Replace only the selected `down_proj` entries in a temporary CPU copy.
2. Compute selected-output reconstruction error directly from cached FP gates
   and activations.
3. Check that this direct error equals the reported Hessian quadratic up to
   numerical tolerance.
4. Temporarily patch the selected model weights, run held-out sequences, record
   next-layer route disagreement and perplexity, then restore FP weights.
5. Do not instantiate a second model and do not retain full hidden-state caches
   on GPU.

The primary comparison is paired over layer, bit width, slice, and seed. Report
`(local GPTQ NMSE - joint NMSE) / local GPTQ NMSE`, not only raw means. Include
the full per-run CSV and peak CUDA memory.

## Acceptance gate before optimization work

Proceed to grouped or low-rank acceleration only if all of the following hold:

- FP reconstruction of the hooked OLMoE MLP passes the `1e-5` relative-MSE
  check.
- The quadratic and direct reconstruction objectives agree.
- Joint coordinate descent is monotone and never worse than its affinity-GPTQ
  initialization on calibration data.
- Joint correction improves held-out block NMSE in at least two of layers
  `0, 7, 15` and across at least three seeds.
- The gain remains when scales are computed from full expert tensors.

If calibration improves but held-out NMSE consistently degrades, stop. That is
evidence of overfitting or a bad slice policy, not a reason to optimize the
solver. If block NMSE improves but route disagreement and perplexity do not,
retain the result as a block-level finding and then test task-conditioned or
route-diverse calibration as a separate hypothesis.

## Next stages only after the gate passes

1. Replace the dense Hessian with expert groups induced by co-routing affinity.
2. Add block-diagonal plus low-rank cross-expert structure.
3. Quantize complete output columns in production-compatible groups.
4. Compare random, route-balanced, task-conditioned, and activation-clustered
   calibration under an equal token budget.
5. Move from OLMoE base perplexity to OLMoE Instruct math, code, reasoning,
   knowledge, and dialogue evaluations.

Keep these as separate ablations. A calibration-data result must not be used to
claim that the joint solver itself caused the gain.
