#!/usr/bin/env python3
"""Shared utilities for Qwen3 quantization experiments."""

from __future__ import annotations

import json
import os
import random
import sys
import time
import gzip
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hsvdquant import (  # noqa: E402
    HSVQuantLinear,
    _decoder_layers,
    _dtype_from_name,
    _get_submodule,
    _load_model,
    _move_tree,
    _qwen_sequential_groups,
    _quantize_activation,
    _set_submodule,
    capture_first_layer_inputs,
    enable_eval_cpu_offload,
    load_quant_checkpoint,
)


@dataclass
class RuntimeConfig:
    source: str
    method: str
    model: str
    checkpoint: str | None
    dtype: str
    device: str
    backend: str = "eager"
    hybrid_policy: str | None = None
    hybrid_threshold: int | None = None
    activation_group_remap: bool = False
    nunchaku_version: str | None = None
    weight_layout: str | None = None


class ActivationQuantLinear(nn.Module):
    """Dense dequantized weight with per-token dynamic activation quantization."""

    def __init__(self, weight: torch.Tensor, bias: torch.Tensor | None, activation_bits: int) -> None:
        super().__init__()
        self.in_features = int(weight.shape[1])
        self.out_features = int(weight.shape[0])
        self.activation_bits = int(activation_bits)
        self.register_buffer("weight", weight.detach().contiguous())
        self.register_buffer("bias", None if bias is None else bias.detach().contiguous())

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        quantized = _quantize_activation(inputs, self.activation_bits)
        return F.linear(quantized, self.weight.to(inputs.dtype), None if self.bias is None else self.bias.to(inputs.dtype))


def set_reproducible(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def environment_metadata() -> dict[str, Any]:
    import transformers

    meta: dict[str, Any] = {
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda": torch.version.cuda,
        "hf_endpoint": os.environ.get("HF_ENDPOINT", ""),
    }
    if torch.cuda.is_available():
        meta["gpus"] = [
            {
                "index": index,
                "name": torch.cuda.get_device_name(index),
                "total_memory_gb": torch.cuda.get_device_properties(index).total_memory / 1024**3,
            }
            for index in range(torch.cuda.device_count())
        ]
    return meta


def make_tokenizer(model_name: str):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model_name, use_fast=True)


def get_eval_input_ids(
    dataset_name: str,
    tokenizer: Any,
    seqlen: int,
    max_tokens: int | None = None,
) -> torch.Tensor:
    """Return GPTQ-style evaluation tokens shaped [1, tokens]."""

    from datasets import load_dataset

    if dataset_name == "wikitext2":
        local_root = os.environ.get("HSVDQ_WIKITEXT2_DIR", "")
        if local_root:
            from datasets import load_from_disk

            data = load_from_disk(local_root)["test"]
        else:
            local_parquet = os.environ.get("HSVDQ_WIKITEXT2_TEST_PARQUET", "")
            if local_parquet:
                from datasets import load_dataset

                data = load_dataset("parquet", data_files=local_parquet, split="train")
            else:
                data = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        encoded = tokenizer("\n\n".join(data["text"]), return_tensors="pt").input_ids
    elif dataset_name == "c4":
        local_c4 = os.environ.get("HSVDQ_C4_VALIDATION", "")
        endpoint = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
        url = f"{endpoint}/datasets/allenai/c4/resolve/main/en/c4-validation.00000-of-00008.json.gz"
        texts = []
        source = open(local_c4, "rb") if local_c4 else urllib.request.urlopen(url, timeout=120)
        with source as response:
            with gzip.GzipFile(fileobj=response) as handle:
                for line in handle:
                    texts.append(json.loads(line)["text"])
                    if len(texts) >= 1100:
                        break
        encoded = tokenizer(" ".join(texts), return_tensors="pt").input_ids
        encoded = encoded[:, : 256 * seqlen]
    else:
        raise ValueError(f"unsupported eval dataset: {dataset_name}")
    if max_tokens is not None:
        encoded = encoded[:, :max_tokens]
    usable = (encoded.numel() // seqlen) * seqlen
    if usable < seqlen:
        raise RuntimeError(f"{dataset_name} has fewer than one full {seqlen}-token eval chunk")
    return encoded[:, :usable]


@torch.no_grad()
def compute_ppl(
    model: nn.Module,
    input_ids: torch.Tensor,
    seqlen: int,
    device: torch.device,
    max_samples: int | None = None,
) -> dict[str, Any]:
    model.eval()
    use_cache = getattr(model.config, "use_cache", False)
    model.config.use_cache = False
    nsamples = input_ids.numel() // seqlen
    if max_samples is not None:
        nsamples = min(nsamples, max_samples)
    nlls: list[torch.Tensor] = []
    started = time.perf_counter()
    logit_chunk = 256
    for index in range(nsamples):
        batch = input_ids[:, index * seqlen : (index + 1) * seqlen].to(device)
        outputs = model(batch)
        logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
        shift_logits = logits[:, :-1, :]
        shift_labels = batch[:, 1:]
        nll = shift_logits.new_zeros(())
        for start in range(0, shift_logits.size(1), logit_chunk):
            chunk_logits = shift_logits[:, start : start + logit_chunk].reshape(-1, shift_logits.size(-1)).float()
            chunk_labels = shift_labels[:, start : start + logit_chunk].reshape(-1)
            nll = nll + F.cross_entropy(chunk_logits, chunk_labels, reduction="sum")
        nlls.append(nll)
        del outputs, logits, shift_logits
    denom = nsamples * (seqlen - 1)
    ppl = torch.exp(torch.stack(nlls).sum() / denom).item()
    model.config.use_cache = use_cache
    return {
        "ppl": ppl,
        "nsamples": nsamples,
        "tokens": denom,
        "seconds": time.perf_counter() - started,
    }


def load_experiment_model(
    *,
    model_name: str,
    checkpoint: str | None,
    device: torch.device,
    dtype: torch.dtype,
    cpu_offload_layers: bool = False,
    runtime_backend: str = "eager",
    persist_qweight: bool = False,
    hybrid_policy: str = "auto",
    hybrid_threshold: int = 128,
    allow_activation_group_remap: bool = False,
    hybrid_profile_stats: bool = False,
) -> tuple[nn.Module, Any, RuntimeConfig]:
    if checkpoint is None:
        model = _load_model(model_name, device, dtype, keep_on_device=not cpu_offload_layers)
        tokenizer = make_tokenizer(model_name)
        if cpu_offload_layers:
            enable_eval_cpu_offload(model, device)
        return model, tokenizer, RuntimeConfig(
            "fp", "fp", model_name, None, str(dtype), str(device), "dense"
        )

    checkpoint_dir = Path(checkpoint)
    hsvd_config = checkpoint_dir / "hsvdquant_config.json"
    dense_config = checkpoint_dir / "baseline_quant_config.json"
    if hsvd_config.exists():
        model, tokenizer, meta = load_quant_checkpoint(
            checkpoint_dir,
            device,
            dtype,
            cpu_offload_layers=cpu_offload_layers,
            runtime_backend=runtime_backend,
            persist_qweight=persist_qweight,
            hybrid_policy=hybrid_policy,
            hybrid_threshold=hybrid_threshold,
            allow_activation_group_remap=allow_activation_group_remap,
            hybrid_profile_stats=hybrid_profile_stats,
        )
        method = meta.get("method", "hsvdquant")
        return model, tokenizer, RuntimeConfig(
            "hsvdquant",
            method,
            meta["base_model"],
            checkpoint,
            str(dtype),
            str(device),
            runtime_backend,
            hybrid_policy if runtime_backend == "hybrid" else None,
            hybrid_threshold if runtime_backend == "hybrid" else None,
            bool(meta.get("activation_group_remap", False)),
            meta.get("nunchaku_version"),
            meta.get("hybrid_weight_layout"),
        )
    if dense_config.exists():
        meta = read_json(dense_config)
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model = AutoModelForCausalLM.from_pretrained(meta["base_model"], torch_dtype=dtype)
        try:
            states = torch.load(checkpoint_dir / "baseline_quant.pt", map_location="cpu", weights_only=True)
        except TypeError:
            states = torch.load(checkpoint_dir / "baseline_quant.pt", map_location="cpu")
        for name, state in states.items():
            module = ActivationQuantLinear(
                state["weight"].to(dtype=dtype),
                None if state.get("bias") is None else state["bias"].to(dtype=dtype),
                int(meta["quant_config"]["activation_bits"]),
            )
            _set_submodule(model, name, module)
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir, use_fast=True)
        model.to(device).eval()
        model.config.use_cache = False
        return model, tokenizer, RuntimeConfig(
            "baseline_quant",
            meta["method"],
            meta["base_model"],
            checkpoint,
            str(dtype),
            str(device),
        )
    raise FileNotFoundError(f"unsupported checkpoint directory: {checkpoint_dir}")


def collect_layer_stats(
    model: nn.Module,
    hidden_batches: list[torch.Tensor],
    layer_kwargs: list[dict[str, Any]],
    layer_index: int,
    device: torch.device,
    cache_tokens: int,
    hessian_block_size: int,
    seed: int,
):
    from hsvdquant import ActivationStats

    layer = _decoder_layers(model)[layer_index]
    stats_by_name: dict[str, ActivationStats] = {}
    handles = []
    for group in _qwen_sequential_groups(layer):
        for offset, name in enumerate(group):
            module = _get_submodule(layer, name)
            stats = ActivationStats(
                module.in_features,
                device,
                cache_tokens,
                hessian_block_size,
                seed + layer_index * 100 + offset,
            )
            stats_by_name[name] = stats

            def hook(_module: nn.Module, args: tuple[Any, ...], _output: Any, target=stats) -> None:
                target.add_batch(args[0])

            handles.append(module.register_forward_hook(hook))
    for hidden, kwargs in zip(hidden_batches, layer_kwargs, strict=True):
        layer(hidden.to(device), **_move_tree(kwargs, device))
    for handle in handles:
        handle.remove()
    return stats_by_name


@torch.no_grad()
def advance_hidden_batches(
    model: nn.Module,
    hidden_batches: list[torch.Tensor],
    layer_kwargs: list[dict[str, Any]],
    layer_index: int,
    device: torch.device,
) -> list[torch.Tensor]:
    from hsvdquant import _decoder_hidden

    layer = _decoder_layers(model)[layer_index]
    next_hidden: list[torch.Tensor] = []
    for hidden, kwargs in zip(hidden_batches, layer_kwargs, strict=True):
        output = _decoder_hidden(layer(hidden.to(device), **_move_tree(kwargs, device)))
        next_hidden.append(output.detach().to("cpu"))
    return next_hidden


def make_calibration(model_name: str, dataset: str, nsamples: int, seqlen: int, batch_size: int, seed: int):
    from hsvdquant import _make_calibration_batches

    return _make_calibration_batches(model_name, dataset, nsamples, seqlen, batch_size, seed)


def quantized_module_names(model: nn.Module) -> list[str]:
    return [name for name, module in model.named_modules() if isinstance(module, (HSVQuantLinear, ActivationQuantLinear))]


def result_payload(runtime: RuntimeConfig, args: Any, metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "runtime": asdict(runtime),
        "args": vars(args) if hasattr(args, "__dict__") else args,
        "metrics": metrics,
        "environment": environment_metadata(),
    }
