#!/usr/bin/env python3
"""H-SVDQuant calibration, joint optimization, checkpointing, and evaluation.

The implementation follows the notation used in ``hsvdquant/hsvdquant.tex``:

    X_tilde = X D^{-1}, W_tilde = D W,
    W_tilde = L1 L2 + R, and Z quantizes R.

Linear weights are transposed internally from PyTorch's [out, in] layout to the
paper's [in, out] layout.  Calibration Hessians are accumulated over all
calibration batches before any weight is quantized.  Raw activation rows are
kept only in a bounded priority reservoir for the dynamic-activation D block.

This is a research implementation.  Integer codes are stored compactly as int8
(not bit-packed); inference reconstructs the dequantized residual on device.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class QuantConfig:
    bits: int = 4
    activation_bits: int = 4
    rank: int = 8
    beta: float = 0.5
    p: float = 2.0
    group_size: int = 128
    block_size: int = 128
    outer_iters: int = 2
    d_mode: str = "cached"
    d_steps: int = 20
    d_lr: float = 0.05
    d_clip: float = 16.0
    activation_weight: float = 1.0
    damp: float = 0.01
    svd_mode: str = "lowrank"
    svd_oversample: int = 8
    svd_niter: int = 2

    def validate(self) -> None:
        if not 2 <= self.bits <= 8:
            raise ValueError("bits must be in [2, 8]")
        if self.activation_bits < 2 and self.activation_bits != 16:
            raise ValueError("activation_bits must be >=2, or 16 to disable activation quantization")
        if self.rank < 0:
            raise ValueError("rank must be non-negative")
        if self.p < 1:
            raise ValueError("p must be >= 1")
        if self.outer_iters < 1:
            raise ValueError("outer_iters must be >= 1")
        if self.d_mode not in {"closed_form", "cached"}:
            raise ValueError("d_mode must be closed_form or cached")
        if self.svd_mode not in {"exact", "lowrank"}:
            raise ValueError("svd_mode must be exact or lowrank")


def _sym(matrix: torch.Tensor) -> torch.Tensor:
    return (matrix + matrix.T) * 0.5


def _regularize_hessian(hessian: torch.Tensor, damp: float) -> torch.Tensor:
    hessian = _sym(hessian.float())
    diagonal = hessian.diagonal()
    positive = diagonal[diagonal > 0]
    reference = positive.mean() if positive.numel() else hessian.new_tensor(1.0)
    return hessian + torch.eye(hessian.shape[0], device=hessian.device, dtype=hessian.dtype) * (
        reference * damp + 1e-8
    )


def _geometric_normalize(values: torch.Tensor, clip: float) -> torch.Tensor:
    log_values = values.clamp_min(1e-12).log()
    log_values = log_values - log_values.mean()
    if clip > 1:
        bound = math.log(clip)
        log_values = log_values.clamp(-bound, bound)
        log_values = log_values - log_values.mean()
    return log_values.exp()


def _hessian_deflate(hessian: torch.Tensor, basis: torch.Tensor | None, damp: float = 1e-6) -> torch.Tensor:
    if basis is None or basis.numel() == 0:
        return _sym(hessian)
    hb = hessian @ basis
    gram = _sym(basis.T @ hb)
    gram = gram + torch.eye(gram.shape[0], device=gram.device, dtype=gram.dtype) * damp
    result = hessian - hb @ torch.linalg.solve(gram, hb.T)
    return _sym(result)


class ActivationStats:
    """Streaming H accumulator plus a bounded, uniform priority reservoir of X rows."""

    def __init__(
        self,
        columns: int,
        device: torch.device,
        cache_tokens: int,
        hessian_block_size: int = 4096,
        seed: int = 0,
    ) -> None:
        self.columns = columns
        self.device = device
        self.cache_tokens = max(0, cache_tokens)
        self.hessian_block_size = max(1, hessian_block_size)
        self.hessian_sum = torch.zeros((columns, columns), device=device, dtype=torch.float32)
        self.num_tokens = 0
        self._cache: torch.Tensor | None = None
        self._priorities: torch.Tensor | None = None
        self._generator = torch.Generator(device="cpu")
        self._generator.manual_seed(seed)

    @torch.no_grad()
    def add_batch(self, inputs: torch.Tensor) -> None:
        rows = inputs.detach().reshape(-1, inputs.shape[-1])
        if rows.shape[1] != self.columns:
            raise ValueError(f"expected {self.columns} columns, got {rows.shape[1]}")
        for start in range(0, rows.shape[0], self.hessian_block_size):
            block = rows[start : start + self.hessian_block_size].to(self.device, dtype=torch.float32)
            self.hessian_sum.addmm_(block.T, block)
        self.num_tokens += rows.shape[0]
        if self.cache_tokens == 0:
            return

        cpu_rows = rows.to(device="cpu", dtype=torch.float16)
        priorities = torch.rand(cpu_rows.shape[0], generator=self._generator)
        if self._cache is not None:
            cpu_rows = torch.cat((self._cache, cpu_rows), dim=0)
            priorities = torch.cat((self._priorities, priorities), dim=0)
        if cpu_rows.shape[0] > self.cache_tokens:
            keep = priorities.topk(self.cache_tokens, sorted=False).indices
            cpu_rows = cpu_rows.index_select(0, keep)
            priorities = priorities.index_select(0, keep)
        self._cache = cpu_rows
        self._priorities = priorities

    def finalize(self) -> tuple[torch.Tensor, torch.Tensor | None]:
        if self.num_tokens == 0:
            raise RuntimeError("no calibration activations were captured")
        hessian = _sym(self.hessian_sum / float(self.num_tokens))
        cached = None if self._cache is None else self._cache.float()
        return hessian, cached

    def free(self) -> None:
        self.hessian_sum = torch.empty(0, device=self.device)
        self._cache = None
        self._priorities = None


def _closed_form_d(
    hessian_perp: torch.Tensor,
    residual: torch.Tensor,
    p: float,
    clip: float,
) -> torch.Tensor:
    h = hessian_perp.diagonal().clamp_min(1e-12)
    omega_p = residual.abs().pow(p).sum(dim=1).clamp_min(1e-12)
    log_d = (h.log() - omega_p.log()) / (p + 2.0)
    return _geometric_normalize(log_d.exp(), clip)


def _modeled_weight_error(
    d: torch.Tensor,
    residual: torch.Tensor,
    hessian_diagonal: torch.Tensor,
    bits: int,
    group_size: int,
) -> torch.Tensor:
    qmax = float(2 ** (bits - 1) - 1)
    columns = residual.shape[0]
    group_size = columns if group_size <= 0 else group_size
    total = residual.new_zeros(())
    for start in range(0, columns, group_size):
        end = min(start + group_size, columns)
        scaled = residual[start:end].abs() * d[start:end, None]
        step2 = scaled.amax(dim=0).square() / (qmax * qmax)
        sensitivity = (hessian_diagonal[start:end] / d[start:end].square()).sum()
        total = total + step2.sum() * sensitivity
    return total / 12.0


def _modeled_activation_error(
    d: torch.Tensor,
    residual: torch.Tensor,
    cached_x: torch.Tensor,
    bits: int,
) -> torch.Tensor:
    qmax = float(2 ** (bits - 1) - 1)
    token_step2 = (cached_x.square() / d.square()).amax(dim=1).mean() / (qmax * qmax)
    residual_energy = (d.square() * residual.square().sum(dim=1)).sum()
    return token_step2 * residual_energy / 12.0


def optimize_d(
    hessian_perp: torch.Tensor,
    residual: torch.Tensor,
    cached_x: torch.Tensor | None,
    config: QuantConfig,
) -> torch.Tensor:
    """Globally initialize D with the lp formula, then optionally refine on cached X."""

    initial = _closed_form_d(hessian_perp, residual, config.p, config.d_clip)
    if config.d_mode == "closed_form" or cached_x is None or config.d_steps <= 0:
        return initial

    device = residual.device
    x = cached_x.to(device=device, dtype=torch.float32)
    hdiag = hessian_perp.diagonal().clamp_min(1e-12)
    # The surrounding layer quantizer is inference-only/no_grad.  D is the one
    # small continuous block for which autograd is intentionally enabled.
    with torch.enable_grad():
        u = nn.Parameter(initial.log())
        optimizer = torch.optim.Adam([u], lr=config.d_lr)
        for _ in range(config.d_steps):
            optimizer.zero_grad(set_to_none=True)
            centered = u - u.mean()
            if config.d_clip > 1:
                centered = centered.clamp(-math.log(config.d_clip), math.log(config.d_clip))
            d = centered.exp()
            fw = _modeled_weight_error(d, residual, hdiag, config.bits, config.group_size)
            if config.activation_bits < 16 and config.activation_weight > 0:
                fa = _modeled_activation_error(d, residual, x, config.activation_bits)
            else:
                fa = fw.new_zeros(())
            loss = (fw + config.activation_weight * fa + 1e-20).log()
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                u.sub_(u.mean())
                if config.d_clip > 1:
                    u.clamp_(-math.log(config.d_clip), math.log(config.d_clip))
    return _geometric_normalize(u.detach().exp(), config.d_clip)


def _truncated_svd(
    matrix: torch.Tensor,
    rank: int,
    mode: str,
    oversample: int,
    niter: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    rank = min(rank, matrix.shape[0], matrix.shape[1])
    if rank == 0:
        return (
            matrix.new_zeros((matrix.shape[0], 0)),
            matrix.new_zeros((0,)),
            matrix.new_zeros((matrix.shape[1], 0)),
        )
    if mode == "exact" or rank + oversample >= min(matrix.shape):
        u, s, vh = torch.linalg.svd(matrix, full_matrices=False)
        return u[:, :rank], s[:rank], vh[:rank].T
    q = min(rank + oversample, min(matrix.shape))
    u, s, v = torch.svd_lowrank(matrix, q=q, niter=niter)
    order = s.argsort(descending=True)[:rank]
    return u[:, order], s[order], v[:, order]


def weighted_low_rank(
    hessian: torch.Tensor,
    weight: torch.Tensor,
    config: QuantConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Solve min_rank(L)<=r ||H^beta (W-L)||_F and fix the H-orthogonal gauge."""

    rank = min(config.rank, weight.shape[0], weight.shape[1])
    if rank == 0:
        return weight.new_zeros((weight.shape[0], 0)), weight.new_zeros((0, weight.shape[1]))

    hreg = _regularize_hessian(hessian, config.damp)
    if config.beta == 0:
        transformed = weight
        eigvals = eigvecs = None
    else:
        eigvals, eigvecs = torch.linalg.eigh(hreg)
        eigvals = eigvals.clamp_min(1e-10)
        projected = eigvecs.T @ weight
        transformed = eigvecs @ (eigvals.pow(config.beta)[:, None] * projected)

    u, s, v = _truncated_svd(
        transformed,
        rank,
        config.svd_mode,
        config.svd_oversample,
        config.svd_niter,
    )
    if config.beta == 0:
        l1 = u
    else:
        l1 = eigvecs @ (eigvals.pow(-config.beta)[:, None] * (eigvecs.T @ u))
    l2 = s[:, None] * v.T

    # Gauge: L1^T H L1 = I, while preserving L1 L2.
    gram = _sym(l1.T @ hreg @ l1)
    values, vectors = torch.linalg.eigh(gram)
    values = values.clamp_min(1e-10)
    gauge = vectors @ (values.rsqrt()[:, None] * vectors.T)
    gauge_inv = vectors @ (values.sqrt()[:, None] * vectors.T)
    return l1 @ gauge, gauge_inv @ l2


@torch.no_grad()
def gptq_quantize_residual(
    residual: torch.Tensor,
    hessian: torch.Tensor,
    config: QuantConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """GPTQ-quantize R and return dequantized R, integer Z, and group scales.

    ``residual`` uses [in, out] layout; codes use PyTorch [out, in] layout.
    """

    weight = residual.T.contiguous().float()
    out_features, in_features = weight.shape
    qmax = 2 ** (config.bits - 1) - 1
    group_size = in_features if config.group_size <= 0 else config.group_size
    num_groups = math.ceil(in_features / group_size)

    scales = torch.empty((out_features, num_groups), device=weight.device, dtype=torch.float32)
    for group in range(num_groups):
        start = group * group_size
        end = min(start + group_size, in_features)
        scales[:, group] = weight[:, start:end].abs().amax(dim=1).clamp_min(1e-8) / float(qmax)

    hreg = _regularize_hessian(hessian, config.damp)
    chol = None
    for attempt in range(5):
        try:
            chol = torch.linalg.cholesky(hreg)
            break
        except RuntimeError:
            extra = hreg.diagonal().mean().clamp_min(1e-8) * (10.0 ** attempt) * config.damp
            hreg = hreg + torch.eye(in_features, device=hreg.device) * extra
    if chol is None:
        raise RuntimeError("failed to stabilize deflated Hessian for GPTQ")
    hinv = torch.cholesky_inverse(chol)
    upper = torch.linalg.cholesky(hinv, upper=True)

    work = weight.clone()
    codes = torch.empty_like(weight, dtype=torch.int8)
    dequant = torch.empty_like(weight)
    block_size = max(1, config.block_size)
    for block_start in range(0, in_features, block_size):
        block_end = min(block_start + block_size, in_features)
        block = work[:, block_start:block_end].clone()
        errors = torch.zeros_like(block)
        for local_index in range(block_end - block_start):
            column = block[:, local_index]
            global_index = block_start + local_index
            scale = scales[:, global_index // group_size]
            code = torch.round(column / scale).clamp(-qmax, qmax).to(torch.int8)
            qcolumn = code.float() * scale
            codes[:, global_index] = code
            dequant[:, global_index] = qcolumn
            error = (column - qcolumn) / upper[global_index, global_index]
            errors[:, local_index] = error
            block[:, local_index:] -= error[:, None] * upper[
                global_index, global_index:block_end
            ][None, :]
        if block_end < in_features:
            work[:, block_end:] -= errors @ upper[block_start:block_end, block_end:]
    return dequant.T.contiguous(), codes, scales


def refit_l2(
    hessian: torch.Tensor,
    weight: torch.Tensor,
    quantized_residual: torch.Tensor,
    l1: torch.Tensor,
) -> torch.Tensor:
    if l1.shape[1] == 0:
        return weight.new_zeros((0, weight.shape[1]))
    gram = _sym(l1.T @ hessian @ l1)
    gram = gram + torch.eye(gram.shape[0], device=gram.device) * 1e-7
    rhs = l1.T @ hessian @ (weight - quantized_residual)
    return torch.linalg.solve(gram, rhs)


def _dequantize_codes(codes: torch.Tensor, scales: torch.Tensor, group_size: int) -> torch.Tensor:
    in_features = codes.shape[1]
    group_size = in_features if group_size <= 0 else group_size
    group_index = torch.arange(in_features, device=codes.device) // group_size
    return codes.float() * scales.index_select(1, group_index)


def _quantize_activation(inputs: torch.Tensor, bits: int) -> torch.Tensor:
    if bits >= 16:
        return inputs
    qmax = float(2 ** (bits - 1) - 1)
    scale = inputs.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8) / qmax
    return torch.round(inputs / scale).clamp(-qmax, qmax) * scale


def _state_error(
    original_weight: torch.Tensor,
    hessian: torch.Tensor,
    cached_x: torch.Tensor | None,
    d: torch.Tensor,
    l1: torch.Tensor,
    l2: torch.Tensor,
    quantized_residual: torch.Tensor,
    activation_bits: int,
) -> float:
    if cached_x is not None:
        x = cached_x.to(device=original_weight.device, dtype=torch.float32)
        smoothed = x / d
        prediction = _quantize_activation(smoothed, activation_bits) @ quantized_residual
        if l1.shape[1]:
            prediction = prediction + (smoothed @ l1) @ l2
        target = x @ original_weight
        return float((prediction - target).square().mean().item())
    effective = (l1 @ l2 + quantized_residual) / d[:, None]
    error = original_weight - effective
    return float(((hessian @ error) * error).sum().div(error.shape[1]).item())


@torch.no_grad()
def joint_quantize_linear(
    layer: nn.Linear,
    hessian: torch.Tensor,
    cached_x: torch.Tensor | None,
    config: QuantConfig,
) -> dict[str, Any]:
    """Jointly update D, L1, Z, and L2 for one Linear layer."""

    config.validate()
    device = layer.weight.device
    original_dtype = layer.weight.dtype
    weight = layer.weight.detach().T.float()  # [in, out]
    hessian = hessian.to(device=device, dtype=torch.float32)
    cached_x_device = None if cached_x is None else cached_x

    previous_a: torch.Tensor | None = None
    previous_l2: torch.Tensor | None = None
    best: dict[str, Any] | None = None
    history: list[float] = []

    for outer in range(config.outer_iters):
        if previous_a is None:
            branch = torch.zeros_like(weight)
            unsmoothed_hperp = hessian
        else:
            branch = previous_a @ previous_l2
            unsmoothed_hperp = _hessian_deflate(hessian, previous_a)
        residual_unsmoothed = weight - branch
        d = optimize_d(unsmoothed_hperp, residual_unsmoothed, cached_x_device, config)

        htilde = hessian / d[:, None] / d[None, :]
        wtilde = d[:, None] * weight
        l1, l2_initial = weighted_low_rank(htilde, wtilde, config)
        hperp_tilde = _hessian_deflate(htilde, l1)
        residual = wtilde - l1 @ l2_initial
        quantized_residual, codes, scales = gptq_quantize_residual(residual, hperp_tilde, config)
        l2 = refit_l2(htilde, wtilde, quantized_residual, l1)
        error = _state_error(
            weight,
            hessian,
            cached_x_device,
            d,
            l1,
            l2,
            quantized_residual,
            config.activation_bits,
        )
        history.append(error)
        candidate = {
            "d": d.detach().to("cpu", dtype=torch.float32),
            "l1": l1.detach().to("cpu", dtype=original_dtype),
            "l2": l2.detach().to("cpu", dtype=original_dtype),
            "codes": codes.detach().to("cpu"),
            "scales": scales.detach().to("cpu", dtype=original_dtype),
            "bias": None
            if layer.bias is None
            else layer.bias.detach().to("cpu", dtype=original_dtype),
            "in_features": layer.in_features,
            "out_features": layer.out_features,
            "group_size": config.group_size,
            "bits": config.bits,
            "activation_bits": config.activation_bits,
            "error": error,
            "outer_iteration": outer,
        }
        if best is None or error < best["error"]:
            best = candidate

        previous_a = l1 / d[:, None]
        previous_l2 = l2

    assert best is not None
    best["history"] = history
    return best


class HSVQuantLinear(nn.Module):
    """Runtime WbA(b) linear: FP branch plus dynamically quantized activations."""

    def __init__(self, state: dict[str, Any], compute_dtype: torch.dtype | None = None) -> None:
        super().__init__()
        self.in_features = int(state["in_features"])
        self.out_features = int(state["out_features"])
        self.group_size = int(state["group_size"])
        self.bits = int(state["bits"])
        self.activation_bits = int(state["activation_bits"])
        dtype = compute_dtype or state["l1"].dtype
        self.register_buffer("d", state["d"].to(dtype=dtype))
        self.register_buffer("l1", state["l1"].to(dtype=dtype))
        self.register_buffer("l2", state["l2"].to(dtype=dtype))
        self.register_buffer("codes", state["codes"].to(torch.int8))
        self.register_buffer("scales", state["scales"].to(dtype=dtype))
        bias = state.get("bias")
        self.register_buffer("bias", None if bias is None else bias.to(dtype=dtype))
        self.register_buffer("_qweight", self._build_qweight(dtype), persistent=False)

    def _build_qweight(self, dtype: torch.dtype | None = None) -> torch.Tensor:
        dtype = dtype or self.scales.dtype
        return _dequantize_codes(self.codes, self.scales, self.group_size).to(dtype=dtype)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if self._qweight.device != inputs.device or self._qweight.dtype != inputs.dtype:
            self._qweight = self._build_qweight(inputs.dtype).to(inputs.device)
        d = self.d.to(dtype=inputs.dtype)
        smoothed = inputs / d
        quantized_inputs = _quantize_activation(smoothed, self.activation_bits)
        output = F.linear(quantized_inputs, self._qweight, None)
        if self.l1.shape[1]:
            output = output + (smoothed @ self.l1.to(inputs.dtype)) @ self.l2.to(inputs.dtype)
        if self.bias is not None:
            output = output + self.bias.to(inputs.dtype)
        return output

    @torch.no_grad()
    def dense_weight(self) -> torch.Tensor:
        residual = self._build_qweight(torch.float32).T
        branch = self.l1.float() @ self.l2.float()
        return ((residual + branch) / self.d.float()[:, None]).T.contiguous()


def _get_submodule(root: nn.Module, name: str) -> nn.Module:
    current = root
    for part in name.split("."):
        current = getattr(current, part)
    return current


def _set_submodule(root: nn.Module, name: str, module: nn.Module) -> None:
    parts = name.split(".")
    parent = root
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], module)


def _move_tree(value: Any, device: torch.device) -> Any:
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, tuple):
        return tuple(_move_tree(item, device) for item in value)
    if isinstance(value, list):
        return [_move_tree(item, device) for item in value]
    if isinstance(value, dict):
        return {key: _move_tree(item, device) for key, item in value.items()}
    return value


def _cpu_tree(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().to("cpu")
    if isinstance(value, tuple):
        return tuple(_cpu_tree(item) for item in value)
    if isinstance(value, list):
        return [_cpu_tree(item) for item in value]
    if isinstance(value, dict):
        return {key: _cpu_tree(item) for key, item in value.items()}
    return value


def _qwen_sequential_groups(layer: nn.Module) -> list[list[str]]:
    candidates = [
        ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"],
        ["self_attn.o_proj"],
        ["mlp.gate_proj", "mlp.up_proj"],
        ["mlp.down_proj"],
    ]
    groups: list[list[str]] = []
    for group in candidates:
        present = [name for name in group if isinstance(_get_submodule(layer, name), nn.Linear)]
        if present:
            groups.append(present)
    return groups


class _StopForward(RuntimeError):
    pass


@torch.no_grad()
def capture_first_layer_inputs(
    model: nn.Module,
    batches: Iterable[dict[str, torch.Tensor]],
    device: torch.device,
) -> tuple[list[torch.Tensor], list[dict[str, Any]]]:
    layers = model.model.layers
    hidden_batches: list[torch.Tensor] = []
    layer_kwargs: list[dict[str, Any]] = []

    def pre_hook(_module: nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
        hidden_batches.append(args[0].detach().to("cpu"))
        layer_kwargs.append(_cpu_tree(kwargs))
        raise _StopForward

    handle = layers[0].register_forward_pre_hook(pre_hook, with_kwargs=True)
    try:
        for batch in batches:
            batch = {key: value.to(device) for key, value in batch.items()}
            try:
                model(**batch, use_cache=False)
            except _StopForward:
                pass
    finally:
        handle.remove()
    if not hidden_batches:
        raise RuntimeError("failed to capture the first decoder-layer inputs")
    return hidden_batches, layer_kwargs


@torch.no_grad()
def quantize_qwen_model(
    model: nn.Module,
    hidden_batches: list[torch.Tensor],
    layer_kwargs: list[dict[str, Any]],
    device: torch.device,
    config: QuantConfig,
    cache_tokens: int,
    hessian_block_size: int,
    max_layers: int = -1,
    seed: int = 0,
) -> dict[str, dict[str, Any]]:
    layers = model.model.layers
    states: dict[str, dict[str, Any]] = {}
    layer_count = len(layers) if max_layers < 0 else min(max_layers, len(layers))

    for layer_index in range(layer_count):
        layer = layers[layer_index]
        print(f"[H-SVDQuant] layer {layer_index + 1}/{layer_count}", flush=True)
        for group in _qwen_sequential_groups(layer):
            collectors: dict[str, ActivationStats] = {}
            handles = []
            for offset, name in enumerate(group):
                module = _get_submodule(layer, name)
                stats = ActivationStats(
                    module.in_features,
                    device,
                    cache_tokens,
                    hessian_block_size,
                    seed + layer_index * 100 + offset,
                )
                collectors[name] = stats

                def hook(_module: nn.Module, args: tuple[Any, ...], _output: Any, target=stats) -> None:
                    target.add_batch(args[0])

                handles.append(module.register_forward_hook(hook))

            for hidden, kwargs in zip(hidden_batches, layer_kwargs, strict=True):
                layer(hidden.to(device), **_move_tree(kwargs, device))
            for handle in handles:
                handle.remove()

            for name in group:
                module = _get_submodule(layer, name)
                hessian, cached_x = collectors[name].finalize()
                started = time.time()
                state = joint_quantize_linear(module, hessian, cached_x, config)
                full_name = f"model.layers.{layer_index}.{name}"
                states[full_name] = state
                replacement = HSVQuantLinear(state, compute_dtype=module.weight.dtype).to(device)
                _set_submodule(layer, name, replacement)
                collectors[name].free()
                print(
                    f"  {name}: best_mse={state['error']:.6e}, "
                    f"iter={state['outer_iteration']}, time={time.time() - started:.1f}s",
                    flush=True,
                )

        next_hidden: list[torch.Tensor] = []
        for hidden, kwargs in zip(hidden_batches, layer_kwargs, strict=True):
            output = layer(hidden.to(device), **_move_tree(kwargs, device))[0]
            next_hidden.append(output.detach().to("cpu"))
        hidden_batches = next_hidden
        torch.cuda.empty_cache() if device.type == "cuda" else None
    return states


def _make_calibration_batches(
    model_name: str,
    dataset_name: str,
    nsamples: int,
    sequence_length: int,
    batch_size: int,
    seed: int,
) -> tuple[Any, list[dict[str, torch.Tensor]]]:
    from datasets import load_dataset
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    rng = random.Random(seed)
    samples: list[torch.Tensor] = []
    if dataset_name == "synthetic":
        generator = torch.Generator().manual_seed(seed)
        for _ in range(nsamples):
            samples.append(torch.randint(0, len(tokenizer), (sequence_length,), generator=generator))
    elif dataset_name == "wikitext2":
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        encoded = tokenizer("\n\n".join(dataset["text"]), return_tensors="pt").input_ids[0]
        if encoded.numel() <= sequence_length:
            raise RuntimeError("calibration corpus is shorter than sequence_length")
        for _ in range(nsamples):
            start = rng.randint(0, encoded.numel() - sequence_length - 1)
            samples.append(encoded[start : start + sequence_length])
    elif dataset_name == "c4":
        dataset = load_dataset("allenai/c4", "en", split="train", streaming=True).shuffle(
            seed=seed, buffer_size=10_000
        )
        for row in dataset:
            ids = tokenizer(row["text"], return_tensors="pt", truncation=False).input_ids[0]
            if ids.numel() < sequence_length:
                continue
            start = rng.randint(0, ids.numel() - sequence_length)
            samples.append(ids[start : start + sequence_length])
            if len(samples) == nsamples:
                break
        if len(samples) < nsamples:
            raise RuntimeError(f"only collected {len(samples)} usable C4 samples")
    else:
        raise ValueError(f"unsupported calibration dataset: {dataset_name}")

    batches = []
    for start in range(0, len(samples), batch_size):
        input_ids = torch.stack(samples[start : start + batch_size])
        batches.append({"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids)})
    return tokenizer, batches


def save_quant_checkpoint(
    output_dir: Path,
    model_name: str,
    tokenizer: Any,
    states: dict[str, dict[str, Any]],
    config: QuantConfig,
    args: argparse.Namespace,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(output_dir)
    torch.save(states, output_dir / "hsvdquant.pt")
    metadata = {
        "format": "hsvdquant-v1",
        "base_model": model_name,
        "quant_config": asdict(config),
        "calibration": {
            "dataset": args.calib_dataset,
            "nsamples": args.nsamples,
            "sequence_length": args.sequence_length,
            "seed": args.seed,
        },
        "modules": list(states),
    }
    (output_dir / "hsvdquant_config.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def _dtype_from_name(name: str) -> torch.dtype:
    mapping = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    if name not in mapping:
        raise ValueError(f"unsupported dtype {name}")
    return mapping[name]


def load_quant_checkpoint(
    checkpoint_dir: Path,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[nn.Module, Any, dict[str, Any]]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    metadata = json.loads((checkpoint_dir / "hsvdquant_config.json").read_text(encoding="utf-8"))
    model = AutoModelForCausalLM.from_pretrained(metadata["base_model"], torch_dtype=dtype)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir, use_fast=True)
    try:
        states = torch.load(checkpoint_dir / "hsvdquant.pt", map_location="cpu", weights_only=True)
    except TypeError:
        states = torch.load(checkpoint_dir / "hsvdquant.pt", map_location="cpu")
    for name, state in states.items():
        _set_submodule(model, name, HSVQuantLinear(state, compute_dtype=dtype))
    model.to(device).eval()
    model.config.use_cache = False
    return model, tokenizer, metadata


@torch.no_grad()
def export_dense_model(model: nn.Module, tokenizer: Any, output_dir: Path) -> None:
    replacements = []
    for name, module in model.named_modules():
        if isinstance(module, HSVQuantLinear):
            replacements.append((name, module))
    for name, module in replacements:
        dense = nn.Linear(module.in_features, module.out_features, bias=module.bias is not None)
        dense = dense.to(device=module.d.device, dtype=module.d.dtype)
        dense.weight.copy_(module.dense_weight().to(dense.weight.dtype))
        if module.bias is not None:
            dense.bias.copy_(module.bias)
        _set_submodule(model, name, dense)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir, safe_serialization=True)
    tokenizer.save_pretrained(output_dir)


def run_lm_eval(
    model: nn.Module,
    tokenizer: Any,
    tasks: Sequence[str],
    batch_size: int,
    limit: float | None,
    output_path: Path | None,
) -> dict[str, Any]:
    from lm_eval import simple_evaluate
    from lm_eval.models.huggingface import HFLM

    wrapped = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=batch_size, backend="causal")
    results = simple_evaluate(model=wrapped, tasks=list(tasks), batch_size=batch_size, limit=limit)
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    return results


def _load_model(model_name: str, device: torch.device, dtype: torch.dtype) -> nn.Module:
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype)
    if not hasattr(model, "model") or not hasattr(model.model, "layers"):
        raise TypeError("this CLI currently expects a Qwen/Llama-style model.model.layers architecture")
    model.to(device).eval()
    model.config.use_cache = False
    return model


def quantize_command(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    dtype = _dtype_from_name(args.dtype)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    config = QuantConfig(
        bits=args.bits,
        activation_bits=args.activation_bits,
        rank=args.rank,
        beta=args.beta,
        p=args.p,
        group_size=args.group_size,
        block_size=args.block_size,
        outer_iters=args.outer_iters,
        d_mode=args.d_mode,
        d_steps=args.d_steps,
        d_lr=args.d_lr,
        d_clip=args.d_clip,
        activation_weight=args.activation_weight,
        damp=args.damp,
        svd_mode=args.svd_mode,
        svd_oversample=args.svd_oversample,
        svd_niter=args.svd_niter,
    )
    config.validate()

    tokenizer, batches = _make_calibration_batches(
        args.model,
        args.calib_dataset,
        args.nsamples,
        args.sequence_length,
        args.calib_batch_size,
        args.seed,
    )
    model = _load_model(args.model, device, dtype)
    hidden, kwargs = capture_first_layer_inputs(model, batches, device)
    del batches
    states = quantize_qwen_model(
        model,
        hidden,
        kwargs,
        device,
        config,
        args.activation_cache_tokens,
        args.hessian_block_size,
        args.max_layers,
        args.seed,
    )
    save_quant_checkpoint(Path(args.output), args.model, tokenizer, states, config, args)

    if args.eval_tasks:
        results = run_lm_eval(
            model,
            tokenizer,
            args.eval_tasks.split(","),
            args.eval_batch_size,
            args.eval_limit,
            Path(args.output) / "lm_eval_results.json",
        )
        print(json.dumps(results.get("results", {}), indent=2, default=str))
    if args.export_dense:
        export_dense_model(model, tokenizer, Path(args.export_dense))


def eval_command(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    dtype = _dtype_from_name(args.dtype)
    model, tokenizer, _ = load_quant_checkpoint(Path(args.checkpoint), device, dtype)
    results = run_lm_eval(
        model,
        tokenizer,
        args.tasks.split(","),
        args.batch_size,
        args.limit,
        None if not args.output else Path(args.output),
    )
    print(json.dumps(results.get("results", {}), indent=2, default=str))


def self_test_command(_args: argparse.Namespace) -> None:
    torch.manual_seed(7)
    in_features, out_features = 16, 12
    layer = nn.Linear(in_features, out_features, bias=True)
    x = torch.randn(96, in_features)
    hessian = x.T @ x / x.shape[0]
    config = QuantConfig(
        bits=4,
        activation_bits=8,
        rank=3,
        beta=0.5,
        p=2,
        group_size=8,
        block_size=8,
        outer_iters=2,
        d_mode="cached",
        d_steps=3,
        svd_mode="exact",
    )
    state = joint_quantize_linear(layer, hessian, x, config)
    quantized = HSVQuantLinear(state, compute_dtype=torch.float32)
    output = quantized(x)
    assert output.shape == (96, out_features)
    assert torch.isfinite(output).all()
    assert state["codes"].dtype == torch.int8
    assert len(state["history"]) == 2
    print(
        json.dumps(
            {
                "status": "ok",
                "history": state["history"],
                "dense_weight_shape": list(quantized.dense_weight().shape),
            },
            indent=2,
        )
    )


def integration_test_command(_args: argparse.Namespace) -> None:
    """Exercise Qwen hooks, all joint blocks, the runtime, and lm-eval without downloads."""

    from lm_eval.api.instance import Instance
    from lm_eval.models.huggingface import HFLM
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM

    torch.manual_seed(11)
    vocab = {"<pad>": 0, "<eos>": 1, "<unk>": 2}
    vocab.update({f"t{index}": index + 3 for index in range(125)})
    tokenizer_backend = Tokenizer(WordLevel(vocab=vocab, unk_token="<unk>"))
    tokenizer_backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer_backend,
        pad_token="<pad>",
        eos_token="<eos>",
        unk_token="<unk>",
    )
    model_config = Qwen3Config(
        vocab_size=128,
        hidden_size=24,
        intermediate_size=48,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=6,
        max_position_embeddings=64,
        pad_token_id=0,
        eos_token_id=1,
    )
    model = Qwen3ForCausalLM(model_config).float().eval()
    batches = [
        {
            "input_ids": torch.randint(3, 128, (2, 8)),
            "attention_mask": torch.ones(2, 8, dtype=torch.long),
        }
    ]
    hidden, kwargs = capture_first_layer_inputs(model, batches, torch.device("cpu"))
    config = QuantConfig(
        bits=4,
        activation_bits=8,
        rank=2,
        beta=0.5,
        group_size=8,
        block_size=8,
        outer_iters=1,
        d_mode="closed_form",
        d_steps=0,
        svd_mode="exact",
    )
    states = quantize_qwen_model(
        model,
        hidden,
        kwargs,
        torch.device("cpu"),
        config,
        cache_tokens=16,
        hessian_block_size=16,
        max_layers=1,
    )
    wrapped = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=1, backend="causal", max_length=32)
    request = Instance(
        request_type="loglikelihood",
        doc={},
        arguments=("t3 t4", " t5"),
        idx=0,
        metadata=("smoke", 0, 1),
    )
    result = wrapped.loglikelihood([request])
    assert len(states) == 7
    assert len(result) == 1 and math.isfinite(result[0][0])
    print(json.dumps({"status": "ok", "quantized_modules": len(states), "loglikelihood": result[0][0]}, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    quantize = subparsers.add_parser("quantize", help="calibrate and quantize a Qwen/Llama-style model")
    quantize.add_argument("--model", default="Qwen/Qwen3-0.6B")
    quantize.add_argument("--output", required=True)
    quantize.add_argument("--export-dense", default="")
    quantize.add_argument("--device", default="cuda:0")
    quantize.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="bfloat16")
    quantize.add_argument("--calib-dataset", choices=["wikitext2", "c4", "synthetic"], default="wikitext2")
    quantize.add_argument("--nsamples", type=int, default=128)
    quantize.add_argument("--sequence-length", type=int, default=512)
    quantize.add_argument("--calib-batch-size", type=int, default=4)
    quantize.add_argument("--activation-cache-tokens", type=int, default=2048)
    quantize.add_argument("--hessian-block-size", type=int, default=4096)
    quantize.add_argument("--seed", type=int, default=0)
    quantize.add_argument("--max-layers", type=int, default=-1)
    quantize.add_argument("--bits", type=int, default=4)
    quantize.add_argument("--activation-bits", type=int, default=4)
    quantize.add_argument("--rank", type=int, default=8)
    quantize.add_argument("--beta", type=float, default=0.5)
    quantize.add_argument("--p", type=float, default=2.0)
    quantize.add_argument("--group-size", type=int, default=128)
    quantize.add_argument("--block-size", type=int, default=128)
    quantize.add_argument("--outer-iters", type=int, default=2)
    quantize.add_argument("--d-mode", choices=["closed_form", "cached"], default="cached")
    quantize.add_argument("--d-steps", type=int, default=20)
    quantize.add_argument("--d-lr", type=float, default=0.05)
    quantize.add_argument("--d-clip", type=float, default=16.0)
    quantize.add_argument("--activation-weight", type=float, default=1.0)
    quantize.add_argument("--damp", type=float, default=0.01)
    quantize.add_argument("--svd-mode", choices=["exact", "lowrank"], default="lowrank")
    quantize.add_argument("--svd-oversample", type=int, default=8)
    quantize.add_argument("--svd-niter", type=int, default=2)
    quantize.add_argument("--eval-tasks", default="")
    quantize.add_argument("--eval-batch-size", type=int, default=4)
    quantize.add_argument("--eval-limit", type=float, default=None)
    quantize.set_defaults(func=quantize_command)

    evaluate = subparsers.add_parser("eval", help="load a compact checkpoint and run lm-eval")
    evaluate.add_argument("--checkpoint", required=True)
    evaluate.add_argument("--tasks", default="hellaswag,arc_easy")
    evaluate.add_argument("--device", default="cuda:0")
    evaluate.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="bfloat16")
    evaluate.add_argument("--batch-size", type=int, default=4)
    evaluate.add_argument("--limit", type=float, default=None)
    evaluate.add_argument("--output", default="")
    evaluate.set_defaults(func=eval_command)

    self_test = subparsers.add_parser("self-test", help="run a dependency-light numerical smoke test")
    self_test.set_defaults(func=self_test_command)
    integration_test = subparsers.add_parser(
        "integration-test", help="run a tiny random-Qwen3 plus lm-eval integration test"
    )
    integration_test.set_defaults(func=integration_test_command)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
