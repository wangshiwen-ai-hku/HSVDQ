#!/usr/bin/env python3
"""H-SVDQuant calibration, joint optimization, checkpointing, and evaluation.

The implementation follows the notation used in ``hsvdquant/hsvdquant.tex``:

    X_tilde = X D^{-1}, W_tilde = D W,
    W_tilde = L1 L2 + R, and Z quantizes R.

Linear weights are transposed internally from PyTorch's [out, in] layout to the
paper's [in, out] layout.  Calibration Hessians are accumulated over all
calibration batches before any weight is quantized.  Raw activation rows are
kept only in a bounded priority reservoir for the dynamic-activation D block.

The eager reference backend stores integer codes as int8 and reconstructs the
dequantized residual.  The optional Nunchaku backend packs W4 weights and runs
the residual plus low-rank branch with true W4A4 CUDA tensor-core kernels.
"""

from __future__ import annotations

import argparse
import gc
import gzip
import json
import math
import os
import random
import time
import urllib.request
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class QuantConfig:
    bits: int = 4
    activation_bits: int = 4
    activation_group_size: int = 0
    d_fa_group_size: int = -1
    rank: int = 8
    rank_a: int = 0
    rank_a_mode: str = "fixed"
    code_objective: str = "fw"
    joint_code_iters: int = 1
    joint_rotation_mode: str = "none"
    joint_rotation_fw_epsilon: float = 0.0
    activation_objective: str = "full"
    reducible_oracle_tokens: int = 512
    reducible_oracle_iters: int = 5
    block_input_mode: str = "quantized"
    intra_block_mode: str = "sequential"
    linear_objective: str = "local"
    ablation_mode: str = "custom"
    trajectory_damp: float = 0.1
    trajectory_max_norm_ratio: float = 0.25
    trajectory_scale: float = 1.0
    trajectory_diagnostics: bool = False
    trajectory_start_layer: int = 0
    trajectory_rebase: bool = False
    trajectory_holdout_fraction: float = 0.0
    trajectory_holdout_backtracking: bool = False
    trajectory_backtrack_scales: tuple[float, ...] = (0.0, 0.125, 0.25, 0.5, 1.0)
    trajectory_spectral_floor: float = 0.0
    trajectory_min_holdout_gain: float = 0.0
    trajectory_min_direction_cosine: float = -1.0
    trajectory_quantized_gate: bool = False
    trajectory_module_filter: str = "all"
    trajectory_oracle_diagnostics: bool = False
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
        if self.activation_group_size < 0:
            raise ValueError("activation_group_size must be non-negative (0 = per-token global max)")
        if self.d_fa_group_size < -1:
            raise ValueError("d_fa_group_size must be >=-1 (-1 = inherit activation_group_size)")
        if self.rank < 0:
            raise ValueError("rank must be non-negative")
        if not 0 <= self.rank_a <= self.rank:
            raise ValueError("rank_a must satisfy 0 <= rank_a <= rank")
        if self.rank_a_mode not in {"fixed", "gated"}:
            raise ValueError("rank_a_mode must be fixed or gated")
        if self.code_objective not in {"fw", "joint"}:
            raise ValueError("code_objective must be fw or joint")
        if self.joint_code_iters < 1:
            raise ValueError("joint_code_iters must be >= 1")
        if self.joint_rotation_mode not in {"none", "empirical"}:
            raise ValueError("joint_rotation_mode must be none or empirical")
        if self.joint_rotation_mode == "empirical" and self.code_objective != "joint":
            raise ValueError("empirical joint rotation requires code_objective=joint")
        if self.joint_rotation_fw_epsilon < 0:
            raise ValueError("joint_rotation_fw_epsilon must be non-negative")
        if self.activation_objective not in {"full", "reducible"}:
            raise ValueError("activation_objective must be full or reducible")
        if self.activation_objective == "reducible":
            if self.joint_rotation_mode != "empirical" or self.joint_code_iters < 2:
                raise ValueError("reducible activation objective requires empirical rotation and joint_code_iters >= 2")
            if self.reducible_oracle_tokens < 4:
                raise ValueError("reducible_oracle_tokens must be >= 4")
            if self.reducible_oracle_iters < 1:
                raise ValueError("reducible_oracle_iters must be >= 1")
        if self.block_input_mode not in {"quantized", "reference"}:
            raise ValueError("block_input_mode must be quantized or reference")
        if self.intra_block_mode not in {"sequential", "fp_independent"}:
            raise ValueError("intra_block_mode must be sequential or fp_independent")
        if self.linear_objective not in {"local", "cumulative"}:
            raise ValueError("linear_objective must be local or cumulative")
        if self.ablation_mode not in {"custom", "v1", "v2", "v3", "v2v3"}:
            raise ValueError("ablation_mode must be custom, v1, v2, v3, or v2v3")
        if self.trajectory_damp < 0:
            raise ValueError("trajectory_damp must be non-negative")
        if self.trajectory_max_norm_ratio <= 0:
            raise ValueError("trajectory_max_norm_ratio must be positive")
        if not 0 < self.trajectory_scale <= 1:
            raise ValueError("trajectory_scale must be in (0, 1]")
        if self.trajectory_start_layer < 0:
            raise ValueError("trajectory_start_layer must be non-negative")
        if not 0 <= self.trajectory_holdout_fraction < 1:
            raise ValueError("trajectory_holdout_fraction must be in [0, 1)")
        if not self.trajectory_backtrack_scales:
            raise ValueError("trajectory_backtrack_scales must be non-empty")
        if any(scale < 0 for scale in self.trajectory_backtrack_scales):
            raise ValueError("trajectory_backtrack_scales must be non-negative")
        if self.trajectory_spectral_floor < 0:
            raise ValueError("trajectory_spectral_floor must be non-negative")
        if self.trajectory_module_filter not in {"all", "attention", "mlp", "down_proj"}:
            raise ValueError("trajectory_module_filter must be all, attention, mlp, or down_proj")
        if self.code_objective == "joint" and self.rank_a > 0:
            raise ValueError("joint code optimization supersedes rank splitting; set rank_a=0")
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
    """Streaming H accumulator plus a bounded, uniform priority reservoir of X rows.

    Large Hessians (e.g. MLP down_proj) are accumulated on CPU by default so the
    GPU peak stays available for GPTQ factorizations while inactive layers are
    CPU-offloaded.
    """

    def __init__(
        self,
        columns: int,
        device: torch.device,
        cache_tokens: int,
        hessian_block_size: int = 4096,
        seed: int = 0,
        hessian_device: torch.device | None = None,
    ) -> None:
        self.columns = columns
        self.device = device
        self.cache_tokens = max(0, cache_tokens)
        self.hessian_block_size = max(1, hessian_block_size)
        # Keep very large H on CPU; small H can stay on the compute device.
        if hessian_device is None:
            hessian_device = torch.device("cpu") if columns >= 8192 else device
        self.hessian_device = hessian_device
        self.hessian_sum = torch.zeros(
            (columns, columns), device=self.hessian_device, dtype=torch.float32
        )
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
            block = rows[start : start + self.hessian_block_size].to(
                self.hessian_device, dtype=torch.float32
            )
            self.hessian_sum.addmm_(block.T, block)
            del block
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
        self.hessian_sum = torch.empty(0)
        self._cache = None
        self._priorities = None


class ActivationCache:
    """Bounded activation reservoir using the same sampling as ActivationStats."""

    def __init__(self, columns: int, cache_tokens: int, seed: int = 0) -> None:
        self.columns = columns
        self.cache_tokens = max(0, cache_tokens)
        self._cache: torch.Tensor | None = None
        self._priorities: torch.Tensor | None = None
        self._generator = torch.Generator(device="cpu")
        self._generator.manual_seed(seed)

    @torch.no_grad()
    def add_batch(self, inputs: torch.Tensor) -> None:
        rows = inputs.detach().reshape(-1, inputs.shape[-1])
        if rows.shape[1] != self.columns:
            raise ValueError(f"expected {self.columns} columns, got {rows.shape[1]}")
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

    def finalize(self) -> torch.Tensor:
        if self._cache is None:
            raise RuntimeError("no reference activations were cached")
        return self._cache.float()

    def free(self) -> None:
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
    group_size: int = 0,
) -> torch.Tensor:
    """Lemma-1 model of the activation channel.

    Under per-token, per-group symmetric uniform activation quantization the noise
    covariance is constant within a group, so F_A factorizes per group G as

        F_A = sum_G (mean_t max_{i in G} X_ti^2 / d_i^2) * (sum_{i in G} d_i^2 ||P_i,:||^2) / (12 kappa^2),

    with P the residual in unsmoothed coordinates.  ``group_size >= c`` (or 0)
    recovers the global per-token form (Eq. FAdyn).  Unlike the global case, D no
    longer cancels: it trades the within-group peak against the within-group
    energy, which is exactly the leverage optimize_d exploits.
    """
    qmax = float(2 ** (bits - 1) - 1)
    columns = residual.shape[0]
    group_size = columns if group_size <= 0 else group_size
    xs = cached_x.square() / d.square()
    p_energy = d.square() * residual.square().sum(dim=1)
    if group_size >= columns:
        return xs.amax(dim=1).mean() * p_energy.sum() / (12.0 * qmax * qmax)
    num_groups = (columns + group_size - 1) // group_size
    pad = num_groups * group_size - columns
    if pad:
        xs = F.pad(xs, (0, pad))
        p_energy = F.pad(p_energy, (0, pad))
    tokens = xs.shape[0]
    group_peak = xs.reshape(tokens, num_groups, group_size).amax(dim=-1).mean(dim=0)
    group_energy = p_energy.reshape(num_groups, group_size).sum(dim=-1)
    return (group_peak * group_energy).sum() / (12.0 * qmax * qmax)


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

    fa_group = config.activation_group_size if config.d_fa_group_size < 0 else config.d_fa_group_size

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
            if (
                config.activation_objective == "full"
                and config.activation_bits < 16
                and config.activation_weight > 0
            ):
                fa = _modeled_activation_error(d, residual, x, config.activation_bits, fa_group)
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
    rank: int | None = None,
    beta: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Solve min_rank(L)<=r ||H^beta (W-L)||_F and fix the H-orthogonal gauge."""

    rank = min(config.rank if rank is None else rank, weight.shape[0], weight.shape[1])
    beta = config.beta if beta is None else beta
    if rank == 0:
        return weight.new_zeros((weight.shape[0], 0)), weight.new_zeros((0, weight.shape[1]))

    hreg = _regularize_hessian(hessian, config.damp)
    if beta == 0:
        transformed = weight
        eigvals = eigvecs = None
    else:
        eigvals, eigvecs = torch.linalg.eigh(hreg)
        eigvals = eigvals.clamp_min(1e-10)
        projected = eigvecs.T @ weight
        transformed = eigvecs @ (eigvals.pow(beta)[:, None] * projected)

    u, s, v = _truncated_svd(
        transformed,
        rank,
        config.svd_mode,
        config.svd_oversample,
        config.svd_niter,
    )
    if beta == 0:
        l1 = u
    else:
        l1 = eigvecs @ (eigvals.pow(-beta)[:, None] * (eigvecs.T @ u))
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
    target: torch.Tensor,
    hessian: torch.Tensor,
    config: QuantConfig,
    prepared_upper: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """GPTQ-quantize a target and return dequantized weights, codes, and scales.

    ``target`` uses [in, out] layout; codes use PyTorch [out, in] layout.
    For the legacy F_W objective the target is the low-rank residual.  For the
    joint objective it is the activation-aware pseudo-target.
    """

    weight = target.T.contiguous().float()
    out_features, in_features = weight.shape
    qmax = 2 ** (config.bits - 1) - 1
    group_size = in_features if config.group_size <= 0 else config.group_size
    num_groups = math.ceil(in_features / group_size)

    scales = torch.empty((out_features, num_groups), device=weight.device, dtype=torch.float32)
    for group in range(num_groups):
        start = group * group_size
        end = min(start + group_size, in_features)
        scales[:, group] = weight[:, start:end].abs().amax(dim=1).clamp_min(1e-8) / float(qmax)

    if prepared_upper is None:
        _, upper = _prepare_gptq_metric(hessian, config)
    else:
        upper = prepared_upper

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


def _prepare_gptq_metric(
    hessian: torch.Tensor,
    config: QuantConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Regularize and factor a GPTQ metric once for target solving and feedback."""

    hreg = _regularize_hessian(hessian, config.damp)
    chol = None
    for attempt in range(5):
        try:
            chol = torch.linalg.cholesky(hreg)
            break
        except RuntimeError:
            extra = hreg.diagonal().mean().clamp_min(1e-8) * (10.0**attempt) * config.damp
            hreg = hreg + torch.eye(hreg.shape[0], device=hreg.device, dtype=hreg.dtype) * extra
    if chol is None:
        raise RuntimeError("failed to stabilize quantization metric for GPTQ")
    # Drop hreg before allocating hinv/upper so peak stays closer to 2x H.
    del hreg
    hinv = torch.cholesky_inverse(chol)
    del chol
    upper = torch.linalg.cholesky(hinv, upper=True)
    del hinv
    # Caller only needs the upper factor for feedback; return a cheap placeholder
    # for the unused first tuple slot to preserve the public signature.
    return upper, upper


def _prepare_joint_metric(
    metric: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Factor H_perp + Sigma_A without injecting the legacy GPTQ damping.

    Positive activation-noise variance removes H_perp's rank-r null space.  A
    tiny diagonal jitter is used only if finite-precision Cholesky needs it.
    """

    metric = _sym(metric.float())
    reference = metric.diagonal().mean().clamp_min(1e-12)
    identity = torch.eye(metric.shape[0], device=metric.device, dtype=metric.dtype)
    chol = None
    jitter_value = 0.0
    for attempt in range(6):
        jitter_value = 0.0 if attempt == 0 else float(reference.item()) * (10.0 ** (attempt - 9))
        candidate = metric if jitter_value == 0.0 else metric + identity * jitter_value
        try:
            chol = torch.linalg.cholesky(candidate)
            break
        except RuntimeError:
            continue
    if chol is None:
        raise RuntimeError("failed to factor H_perp + Sigma_A for joint GPTQ")
    hinv = torch.cholesky_inverse(chol)
    return chol, torch.linalg.cholesky(hinv, upper=True), jitter_value


def joint_code_target(
    hessian_perp: torch.Tensor,
    weight: torch.Tensor,
    sigma_a: torch.Tensor,
    config: QuantConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Build the exact two-channel pseudo-target for fixed D and L1.

    After eliminating L2, the modeled objective is

        ||W - Q||^2_{H_perp} + lambda * ||Q||^2_{Sigma_A}.

    Completing the square gives metric M = H_perp + lambda Sigma_A and target
    T = M^{-1} H_perp W.  The same stabilized factor is reused by GPTQ.
    """

    sigma = sigma_a * float(config.activation_weight)
    metric = _sym(hessian_perp + torch.diag(sigma))
    chol, upper, jitter = _prepare_joint_metric(metric)
    target = torch.cholesky_solve(hessian_perp @ weight, chol)
    return target, metric, upper, jitter


def joint_surrogate_terms(
    hessian: torch.Tensor,
    weight: torch.Tensor,
    l1: torch.Tensor,
    l2: torch.Tensor,
    quantized_residual: torch.Tensor,
    sigma_a: torch.Tensor,
    activation_weight: float,
) -> dict[str, float]:
    """Evaluate the common F_W + lambda F_A model after the post-Q refit."""

    delta = weight - l1 @ l2 - quantized_residual
    fw = ((hessian @ delta) * delta).sum()
    fa = (sigma_a[:, None] * quantized_residual.square()).sum()
    normalizer = max(1, weight.shape[1])
    fw_value = float((fw / normalizer).item())
    fa_value = float((fa / normalizer).item())
    return {
        "fw": fw_value,
        "fa": fa_value,
        "weighted_fa": float(activation_weight) * fa_value,
        "joint": fw_value + float(activation_weight) * fa_value,
    }


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


def empirical_joint_branch_update(
    cached_x: torch.Tensor,
    d: torch.Tensor,
    weight: torch.Tensor,
    quantized_residual: torch.Tensor,
    current_l1: torch.Tensor,
    current_l2: torch.Tensor,
    config: QuantConfig,
    target_outputs: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    """Optimize the rank-r FP branch for the exact fixed-code W4A4 output.

    For fixed D and dequantized codes Q, solve the empirical block

        min_rank(B)<=r ||Y - Qa(X D^-1) Q - (X D^-1) B||_F^2.

    The ridge regression constructs the unconstrained branch target, and the
    H^{1/2}-weighted rank-r SVD rotates both L1 and L2.  A fixed-code guard
    rejects numerical or ridge-induced regressions.  The caller subsequently
    rebuilds the residual and requantizes it in the next joint-code iteration.
    """

    x = cached_x.to(device=weight.device, dtype=torch.float32)
    xtilde = x / d
    qx = _quantize_activation(
        xtilde,
        config.activation_bits,
        config.activation_group_size,
    )
    target = x @ weight if target_outputs is None else target_outputs.to(
        device=weight.device, dtype=torch.float32
    )
    low_path = qx @ quantized_residual
    branch_target_outputs = target - low_path
    rows = max(1, xtilde.shape[0])
    empirical_hessian = _sym(xtilde.T @ xtilde / float(rows))
    cross = xtilde.T @ branch_target_outputs / float(rows)
    unconstrained = torch.linalg.solve(
        _regularize_hessian(empirical_hessian, config.damp),
        cross,
    )
    proposal_l1, proposal_l2 = weighted_low_rank(
        empirical_hessian,
        unconstrained,
        config,
        rank=config.rank,
        beta=0.5,
    )
    current_error = (
        target - low_path - (xtilde @ current_l1) @ current_l2
    ).square().mean()
    proposal_error = (
        target - low_path - (xtilde @ proposal_l1) @ proposal_l2
    ).square().mean()
    accepted = bool(proposal_error <= current_error * (1.0 + 1e-6))
    gain = float(((current_error - proposal_error) / current_error.clamp_min(1e-30)).item())
    diagnostics = {
        "rotation_fixed_code_mse_before": float(current_error.item()),
        "rotation_fixed_code_mse_after": float(proposal_error.item()),
        "rotation_fixed_code_gain": gain,
        "rotation_fixed_code_accepted": float(accepted),
    }
    if not accepted:
        return current_l1, current_l2, diagnostics
    return proposal_l1, proposal_l2, diagnostics


def _fit_reducible_codebooks(
    xtilde: torch.Tensor,
    group_size: int,
    levels: int,
    iterations: int,
) -> tuple[torch.Tensor, int, int]:
    """Fit analysis-only Lloyd-Max grids after per-token group normalization."""

    columns = xtilde.shape[1]
    group_size = columns if group_size <= 0 or group_size >= columns else group_size
    groups = (columns + group_size - 1) // group_size
    pad = groups * group_size - columns
    padded = xtilde if pad == 0 else F.pad(xtilde, (0, pad))
    values = padded.reshape(xtilde.shape[0], groups, group_size)
    scale = values.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8)
    normalized = values / scale
    quantiles = torch.linspace(0.0, 1.0, levels, device=xtilde.device)
    books: list[torch.Tensor] = []
    for group in range(groups):
        samples = normalized[:, group, :].reshape(-1).float()
        centers = torch.quantile(samples, quantiles).sort().values
        for _ in range(iterations):
            assignment = (samples[:, None] - centers[None, :]).abs().argmin(dim=1)
            sums = torch.zeros_like(centers).scatter_add_(0, assignment, samples)
            counts = torch.zeros_like(centers).scatter_add_(
                0, assignment, torch.ones_like(samples)
            )
            proposal = torch.where(counts > 0, sums / counts.clamp_min(1), centers)
            if torch.max(torch.abs(proposal - centers)) < 1e-6:
                centers = proposal
                break
            centers = proposal
        books.append(centers.sort().values)
    return torch.stack(books), group_size, pad


def _apply_reducible_codebooks(
    xtilde: torch.Tensor,
    books: torch.Tensor,
    group_size: int,
    pad: int,
) -> torch.Tensor:
    columns = xtilde.shape[1]
    padded = xtilde if pad == 0 else F.pad(xtilde, (0, pad))
    values = padded.reshape(xtilde.shape[0], books.shape[0], group_size)
    scale = values.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8)
    normalized = values / scale
    quantized = torch.empty_like(normalized)
    for group in range(books.shape[0]):
        assignment = (normalized[:, group, :, None] - books[group]).abs().argmin(dim=-1)
        quantized[:, group, :] = books[group][assignment]
    result = (quantized * scale).reshape(xtilde.shape[0], -1)
    return result[:, :columns]


def _quantized_state_prediction(
    x: torch.Tensor,
    d: torch.Tensor,
    l1: torch.Tensor,
    l2: torch.Tensor,
    quantized_residual: torch.Tensor,
    config: QuantConfig,
) -> torch.Tensor:
    xtilde = x / d
    prediction = _quantize_activation(
        xtilde, config.activation_bits, config.activation_group_size
    ) @ quantized_residual
    if l1.shape[1]:
        prediction = prediction + (xtilde @ l1) @ l2
    return prediction


def build_reducible_oracle_teacher(
    cached_x: torch.Tensor,
    d: torch.Tensor,
    l1: torch.Tensor,
    l2: torch.Tensor,
    quantized_residual: torch.Tensor,
    config: QuantConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, float]]:
    """Cross-fitted L2 projection of uniform error onto a non-uniform direction."""

    rows = min(int(config.reducible_oracle_tokens), cached_x.shape[0])
    x = cached_x[:rows].to(device=d.device, dtype=torch.float32)
    fit_index = torch.arange(0, rows, 2, device=x.device)
    test_index = torch.arange(1, rows, 2, device=x.device)
    xtilde = x / d
    q_uniform = _quantize_activation(
        xtilde, config.activation_bits, config.activation_group_size
    )
    levels = 2 * (2 ** (config.activation_bits - 1) - 1) + 1
    books, group_size, pad = _fit_reducible_codebooks(
        xtilde[fit_index],
        config.activation_group_size,
        levels,
        config.reducible_oracle_iters,
    )
    q_oracle = _apply_reducible_codebooks(xtilde, books, group_size, pad)
    activation_error = (xtilde - q_uniform) @ quantized_residual
    direction = (q_oracle - q_uniform) @ quantized_residual
    alpha = (
        (activation_error[fit_index] * direction[fit_index]).sum()
        / direction[fit_index].square().sum().clamp_min(1e-30)
    ).clamp(0.0, 1.0)
    reducible = direction * alpha
    base_prediction = _quantized_state_prediction(
        x, d, l1, l2, quantized_residual, config
    )
    teacher = base_prediction + reducible
    irreducible = activation_error - reducible
    fit_cross = 2.0 * (reducible[fit_index] * irreducible[fit_index]).mean()
    test_cross = 2.0 * (reducible[test_index] * irreducible[test_index]).mean()
    test_uniform = activation_error[test_index].square().mean().clamp_min(1e-30)
    diagnostics = {
        "reducible_alpha": float(alpha.item()),
        "reducible_fit_mse": float(reducible[fit_index].square().mean().item()),
        "reducible_test_mse": float(reducible[test_index].square().mean().item()),
        "irreducible_test_mse": float(irreducible[test_index].square().mean().item()),
        "reducible_fit_cross": float(fit_cross.item()),
        "reducible_test_normalized_cross": float((test_cross / test_uniform).item()),
        "reducible_oracle_gain": float(
            1.0 - irreducible[test_index].square().mean().div(test_uniform).item()
        ),
    }
    return x, teacher, fit_index, test_index, diagnostics


def activation_noise_diagonal(
    cached_x: torch.Tensor,
    d: torch.Tensor,
    activation_bits: int,
    group_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Diagonal of Sigma_A in smoothed coordinates, per Lemma 1.

    Under the per-token (optionally per-group) symmetric uniform activation
    quantizer, sigma_i^2 = mean_t max_{j in G(i)} (X_tj / d_j)^2 / (12 kappa^2),
    block-constant within each activation group.  The global per-token case
    (group_size 0) yields an isotropic sigma^2 I, as predicted by the theory.
    """

    if activation_bits >= 16:
        return torch.zeros_like(d)
    qmax = float(2 ** (activation_bits - 1) - 1)
    columns = d.shape[0]
    group_size = columns if group_size <= 0 else group_size
    xs = cached_x.to(device=device, dtype=torch.float32).square() / d.square()
    if group_size >= columns:
        peak = xs.amax(dim=1).mean()
        return torch.full_like(d, float(peak) / (12.0 * qmax * qmax))
    num_groups = (columns + group_size - 1) // group_size
    pad = num_groups * group_size - columns
    if pad:
        xs = F.pad(xs, (0, pad))
    peaks = xs.reshape(-1, num_groups, group_size).amax(dim=-1).mean(dim=0)
    sigma = peaks.repeat_interleave(group_size)[:columns]
    return sigma / (12.0 * qmax * qmax)


def activation_aware_branch(
    wtilde: torch.Tensor,
    sigma_a: torch.Tensor,
    rank_a: int,
    config: QuantConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    """L1 block spanning the top left-singular vectors of Sigma_A^{1/2} Wtilde.

    The F_A-optimal rank-r_a branch is the weighted Eckart--Young truncation
    (Prop. fa-branch of hsvdquant.tex); we only need its column space here
    because L2 is refit jointly afterwards.  Also returns the top singular
    values of Sigma_A^{1/2} Wtilde (the marginal F_A gains of Prop. split).
    """

    scale = sigma_a.clamp_min(1e-30).sqrt()
    u, s, _ = _truncated_svd(
        scale[:, None] * wtilde,
        rank_a,
        config.svd_mode,
        config.svd_oversample,
        config.svd_niter,
    )
    return u / scale[:, None], s


def _whiten_h_gauge(l1: torch.Tensor, hessian: torch.Tensor) -> torch.Tensor:
    """Rescale columns of l1 so that l1^T H l1 = I, preserving the span."""

    gram = _sym(l1.T @ hessian @ l1)
    values, vectors = torch.linalg.eigh(gram)
    values = values.clamp_min(1e-10)
    return l1 @ (vectors @ (values.rsqrt()[:, None] * vectors.T))


def _dequantize_codes(codes: torch.Tensor, scales: torch.Tensor, group_size: int) -> torch.Tensor:
    in_features = codes.shape[1]
    group_size = in_features if group_size <= 0 else group_size
    group_index = torch.arange(in_features, device=codes.device) // group_size
    return codes.float() * scales.index_select(1, group_index)


def _quantize_activation(inputs: torch.Tensor, bits: int, group_size: int = 0) -> torch.Tensor:
    """Per-token symmetric uniform quantization, optionally per channel-group.

    group_size 0 (or >= c) keeps the single per-token max across all channels.
    """
    quantized, _codes, _scales = _quantize_activation_with_codes(inputs, bits, group_size)
    return quantized


def _quantize_activation_with_codes(
    inputs: torch.Tensor,
    bits: int,
    group_size: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return dequantized activations plus integer codes and group scales."""
    if bits >= 16:
        codes = torch.zeros_like(inputs, dtype=torch.int16)
        scales = torch.ones(*inputs.shape[:-1], 1, device=inputs.device, dtype=inputs.dtype)
        return inputs, codes, scales
    qmax = float(2 ** (bits - 1) - 1)
    columns = inputs.shape[-1]
    group_size = columns if group_size <= 0 or group_size >= columns else group_size
    lead = inputs.shape[:-1]
    num_groups = (columns + group_size - 1) // group_size
    pad = num_groups * group_size - columns
    padded = inputs if not pad else F.pad(inputs, (0, pad))
    grouped = padded.reshape(*lead, num_groups, group_size)
    scale = grouped.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8) / qmax
    codes = torch.round(grouped / scale).clamp(-qmax, qmax)
    quantized = codes * scale
    quantized = quantized.reshape(*lead, num_groups * group_size)
    codes = codes.reshape(*lead, num_groups * group_size).to(torch.int16)
    if pad:
        quantized = quantized[..., :columns]
        codes = codes[..., :columns]
    return quantized, codes, scale.squeeze(-1)


def _activation_group_centers(inputs: torch.Tensor, group_size: int) -> torch.Tensor:
    columns = inputs.shape[-1]
    group_size = columns if group_size <= 0 or group_size >= columns else group_size
    lead = inputs.shape[:-1]
    num_groups = (columns + group_size - 1) // group_size
    pad = num_groups * group_size - columns
    padded = inputs if not pad else F.pad(inputs, (0, pad))
    return padded.reshape(*lead, num_groups, group_size).mean(dim=-1)


def _activation_code_histograms(codes: torch.Tensor, bits: int, group_size: int) -> torch.Tensor:
    columns = codes.shape[-1]
    group_size = columns if group_size <= 0 or group_size >= columns else group_size
    lead = codes.shape[:-1]
    rows = math.prod(lead) if lead else 1
    qmax = int(2 ** (bits - 1) - 1)
    levels = 2 * qmax + 1
    num_groups = (columns + group_size - 1) // group_size
    pad = num_groups * group_size - columns
    flat = codes.reshape(rows, columns)
    padded = flat if not pad else F.pad(flat, (0, pad), value=0)
    grouped = padded.reshape(rows, num_groups, group_size).long() + qmax
    hist = torch.zeros((rows, num_groups, levels), device=codes.device, dtype=torch.float32)
    hist.scatter_add_(2, grouped.clamp(0, levels - 1), torch.ones_like(grouped, dtype=torch.float32))
    return hist.reshape(*lead, num_groups * levels)


def _normalized_activation_mask(inputs: torch.Tensor, group_size: int, threshold: float) -> torch.Tensor:
    columns = inputs.shape[-1]
    group_size = columns if group_size <= 0 or group_size >= columns else group_size
    lead = inputs.shape[:-1]
    num_groups = (columns + group_size - 1) // group_size
    pad = num_groups * group_size - columns
    padded = inputs if not pad else F.pad(inputs, (0, pad))
    grouped = padded.reshape(*lead, num_groups, group_size)
    rms = grouped.square().mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-8)
    mask = (grouped.abs() / rms) > float(threshold)
    mask = mask.reshape(*lead, num_groups * group_size)
    return mask[..., :columns] if pad else mask


def _mse(left: torch.Tensor, right: torch.Tensor) -> float:
    return float((left - right).square().mean().item())


def _safe_gain(before: float, after: float) -> float:
    return (before - after) / max(before, 1e-30)


def _split_fit_holdout(
    xhat: torch.Tensor,
    xref: torch.Tensor,
    fraction: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    rows = xhat.shape[0]
    holdout_rows = int(rows * fraction)
    if holdout_rows <= 0 or rows - holdout_rows < 2:
        return xhat, xref, None, None
    # Deterministic interleaving keeps token positions spread across the cache.
    holdout_index = torch.linspace(0, rows - 1, holdout_rows, device=xhat.device).round().long().unique()
    mask = torch.ones(rows, device=xhat.device, dtype=torch.bool)
    mask[holdout_index] = False
    if int(mask.sum().item()) < 2 or holdout_index.numel() < 1:
        return xhat, xref, None, None
    return xhat[mask], xref[mask], xhat[holdout_index], xref[holdout_index]


def _trajectory_metric_diagnostics(hessian: torch.Tensor, spectral_floor: float) -> dict[str, float]:
    eigvals = torch.linalg.eigvalsh(_sym(hessian)).clamp_min(0)
    total = eigvals.sum().clamp_min(1e-30)
    max_eig = eigvals[-1].clamp_min(1e-30)
    positive = eigvals[eigvals > max_eig * 1e-8]
    threshold = max_eig * float(spectral_floor)
    weak = eigvals <= threshold if spectral_floor > 0 else eigvals <= max_eig * 1e-8
    probs = eigvals / total
    entropy = -(probs[probs > 0] * probs[probs > 0].log()).sum()
    return {
        "hessian_lambda_max": float(max_eig.item()),
        "hessian_lambda_min_pos": float(positive[0].item()) if positive.numel() else 0.0,
        "hessian_condition": float((max_eig / positive[0]).item()) if positive.numel() else float("inf"),
        "effective_rank": float(entropy.exp().item()),
        "weak_subspace_energy_share": float((eigvals[weak].sum() / total).item()),
        "weak_subspace_dim": float(weak.sum().item()),
    }


def _solve_trajectory_direction(
    xhat: torch.Tensor,
    delta: torch.Tensor,
    damp: float,
    spectral_floor: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    hessian = _sym(xhat.T @ xhat / float(max(1, xhat.shape[0])))
    cross = xhat.T @ delta / float(max(1, xhat.shape[0]))
    diagnostics = _trajectory_metric_diagnostics(hessian, spectral_floor)
    if spectral_floor > 0:
        eigvals, eigvecs = torch.linalg.eigh(hessian)
        eigvals = eigvals.clamp_min(0)
        max_eig = eigvals[-1].clamp_min(1e-30)
        keep = eigvals >= max_eig * float(spectral_floor)
        filtered = eigvecs @ (keep.to(cross.dtype)[:, None] * (eigvecs.T @ cross))
        diagnostics["spectral_kept_dim"] = float(keep.sum().item())
        diagnostics["spectral_dropped_dim"] = float((~keep).sum().item())
        cross = filtered
    else:
        diagnostics["spectral_kept_dim"] = float(hessian.shape[0])
        diagnostics["spectral_dropped_dim"] = 0.0
    direction = torch.linalg.solve(_regularize_hessian(hessian, damp), cross)
    return direction, diagnostics


def _direction_cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    denom = left.norm() * right.norm()
    if float(denom.item()) <= 1e-30:
        return 0.0
    return float(((left * right).sum() / denom).item())


@torch.no_grad()
def cumulative_target_weight(
    layer: nn.Linear,
    hessian: torch.Tensor,
    propagated_x: torch.Tensor,
    reference_x: torch.Tensor,
    config: QuantConfig,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    """Project accumulated upstream error onto the current Linear's weights.

    For reference inputs X and propagated inputs Xhat, the cumulative objective

        min_A ||X W - Xhat A||_F^2

    has A* = W + Hhat^{-1} Xhat^T (X-Xhat) W.  The finite-cache estimator uses
    the paired-cache Hessian and a trajectory-specific ridge.  A closed-form
    line search and norm trust region guarantee that the accepted correction
    does not increase paired teacher MSE merely because the empirical Hessian
    is ill-conditioned.  The target outputs are kept separately so candidate
    selection still measures the requested XW target, including the
    irreducible component outside span(Xhat).
    """

    device = layer.weight.device
    xhat = propagated_x.to(device=device, dtype=torch.float32)
    xref = reference_x.to(device=device, dtype=torch.float32)
    if xhat.shape != xref.shape:
        raise ValueError(
            f"paired cumulative caches must have equal shapes, got {tuple(xhat.shape)} and {tuple(xref.shape)}"
        )
    weight = layer.weight.detach().T.float()
    fit_xhat, fit_xref, holdout_xhat, holdout_xref = _split_fit_holdout(
        xhat,
        xref,
        config.trajectory_holdout_fraction if config.trajectory_diagnostics else 0.0,
    )
    fit_target_outputs = fit_xref @ weight
    fit_upstream_delta = fit_target_outputs - fit_xhat @ weight
    direction, metric_diagnostics = _solve_trajectory_direction(
        fit_xhat,
        fit_upstream_delta,
        config.trajectory_damp,
        config.trajectory_spectral_floor,
    )
    direction_output = fit_xhat @ direction
    denominator = direction_output.square().sum().clamp_min(1e-30)
    line_scale = (
        (fit_upstream_delta * direction_output).sum() / denominator
    ).clamp(0.0, 1.0)
    line_scale = line_scale * float(config.trajectory_scale)
    raw_line_correction = direction * line_scale

    full_target_outputs = xref @ weight
    full_upstream_delta = full_target_outputs - xhat @ weight
    upstream_mse = float(full_upstream_delta.square().mean().item())
    line_projected_mse = _mse(xhat @ (weight + raw_line_correction), full_target_outputs)

    holdout_upstream_mse = None
    holdout_projected_mse = None
    holdout_best_scale = 1.0
    if holdout_xhat is not None and holdout_xref is not None:
        holdout_target_outputs = holdout_xref @ weight
        holdout_upstream_delta = holdout_target_outputs - holdout_xhat @ weight
        holdout_upstream_mse = float(holdout_upstream_delta.square().mean().item())
        scale_errors = []
        for scale in config.trajectory_backtrack_scales:
            candidate = weight + raw_line_correction * float(scale)
            scale_errors.append((float(scale), _mse(holdout_xhat @ candidate, holdout_target_outputs)))
        if config.trajectory_holdout_backtracking:
            holdout_best_scale, holdout_projected_mse = min(scale_errors, key=lambda item: item[1])
        else:
            holdout_projected_mse = _mse(
                holdout_xhat @ (weight + raw_line_correction),
                holdout_target_outputs,
            )
            holdout_best_scale = 1.0
    correction = raw_line_correction * holdout_best_scale

    raw_norm_ratio = correction.norm().div(weight.norm().clamp_min(1e-20))
    trust_scale = min(
        1.0,
        float(config.trajectory_max_norm_ratio) / float(raw_norm_ratio.clamp_min(1e-30)),
    )
    correction = correction * trust_scale
    target_weight = weight + correction
    projected_mse = _mse(xhat @ target_weight, full_target_outputs)
    accepted_holdout_mse = None
    if holdout_xhat is not None and holdout_xref is not None:
        accepted_holdout_mse = _mse(holdout_xhat @ target_weight, holdout_xref @ weight)

    split_direction_cosine = 1.0
    split_norm_ratio = 1.0
    if config.trajectory_diagnostics and fit_xhat.shape[0] >= 4:
        even_xhat = fit_xhat[0::2]
        odd_xhat = fit_xhat[1::2]
        even_xref = fit_xref[0::2]
        odd_xref = fit_xref[1::2]
        if even_xhat.shape[0] >= 2 and odd_xhat.shape[0] >= 2:
            even_target = even_xref @ weight
            odd_target = odd_xref @ weight
            even_dir, _ = _solve_trajectory_direction(
                even_xhat,
                even_target - even_xhat @ weight,
                config.trajectory_damp,
                config.trajectory_spectral_floor,
            )
            odd_dir, _ = _solve_trajectory_direction(
                odd_xhat,
                odd_target - odd_xhat @ weight,
                config.trajectory_damp,
                config.trajectory_spectral_floor,
            )
            split_direction_cosine = _direction_cosine(even_dir, odd_dir)
            split_norm_ratio = float(
                min(even_dir.norm(), odd_dir.norm()).div(
                    max(even_dir.norm(), odd_dir.norm()).clamp_min(1e-30)
                ).item()
            )

    if projected_mse > upstream_mse * (1.0 + 1e-5):
        # Numerical guard: W (zero correction) is always a feasible candidate.
        correction.zero_()
        target_weight = weight
        projected_mse = upstream_mse
        if holdout_upstream_mse is not None:
            accepted_holdout_mse = holdout_upstream_mse

    holdout_gain = (
        _safe_gain(holdout_upstream_mse, accepted_holdout_mse)
        if holdout_upstream_mse is not None and accepted_holdout_mse is not None
        else 0.0
    )
    cap_headroom = float(raw_norm_ratio.item()) / max(float(config.trajectory_max_norm_ratio), 1e-30)
    diagnostics = {
        "upstream_mse": upstream_mse,
        "projected_mse": projected_mse,
        "line_projected_mse": line_projected_mse,
        "stabilization_gap": (line_projected_mse - projected_mse) / max(upstream_mse, 1e-30),
        "holdout_upstream_mse": float(holdout_upstream_mse) if holdout_upstream_mse is not None else 0.0,
        "holdout_projected_mse": float(accepted_holdout_mse) if accepted_holdout_mse is not None else 0.0,
        "holdout_gain": holdout_gain,
        "holdout_best_scale": float(holdout_best_scale),
        "correction_norm_ratio": float(
            correction.norm().div(weight.norm().clamp_min(1e-20)).item()
        ),
        "raw_correction_norm_ratio": float(raw_norm_ratio.item()),
        "cap_headroom": cap_headroom,
        "correction_line_scale": float(line_scale.item()),
        "correction_trust_scale": float(trust_scale),
        "correction_relative_gain": (upstream_mse - projected_mse) / max(upstream_mse, 1e-30),
        "split_direction_cosine": split_direction_cosine,
        "split_norm_ratio": split_norm_ratio,
        **metric_diagnostics,
    }
    if config.trajectory_oracle_diagnostics:
        try:
            oracle = torch.linalg.lstsq(xhat, full_target_outputs).solution
            oracle_mse = _mse(xhat @ oracle, full_target_outputs)
            diagnostics.update(
                {
                    "oracle_projected_mse": oracle_mse,
                    "j_irr_mse": oracle_mse,
                    "oracle_projection_gain": _safe_gain(upstream_mse, oracle_mse),
                    "oracle_to_projected_gap": (projected_mse - oracle_mse) / max(upstream_mse, 1e-30),
                }
            )
        except RuntimeError:
            diagnostics.update(
                {
                    "oracle_projected_mse": 0.0,
                    "j_irr_mse": 0.0,
                    "oracle_projection_gain": 0.0,
                    "oracle_to_projected_gap": 0.0,
                }
            )
    return target_weight, full_target_outputs, diagnostics


def _state_error(
    original_weight: torch.Tensor,
    hessian: torch.Tensor,
    cached_x: torch.Tensor | None,
    d: torch.Tensor,
    l1: torch.Tensor,
    l2: torch.Tensor,
    quantized_residual: torch.Tensor,
    activation_bits: int,
    activation_group_size: int = 0,
    target_outputs: torch.Tensor | None = None,
) -> float:
    if cached_x is not None:
        x = cached_x.to(device=original_weight.device, dtype=torch.float32)
        smoothed = x / d
        prediction = _quantize_activation(smoothed, activation_bits, activation_group_size) @ quantized_residual
        if l1.shape[1]:
            prediction = prediction + (smoothed @ l1) @ l2
        target = (
            x @ original_weight
            if target_outputs is None
            else target_outputs.to(device=original_weight.device, dtype=torch.float32)
        )
        return float((prediction - target).square().mean().item())
    effective = (l1 @ l2 + quantized_residual) / d[:, None]
    error = original_weight - effective
    return float(((hessian @ error) * error).sum().div(error.shape[1]).item())


def _factor_branch_matrix(
    hessian: torch.Tensor,
    branch: torch.Tensor,
    config: QuantConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rank-r factorization of a branch matrix in the H^{1/2} gauge."""

    return weighted_low_rank(
        hessian,
        branch,
        config,
        rank=config.rank,
        beta=0.5,
    )


def reducible_fixed_code_branch_correction(
    cached_x: torch.Tensor,
    d: torch.Tensor,
    current_l1: torch.Tensor,
    current_l2: torch.Tensor,
    quantized_residual: torch.Tensor,
    teacher_outputs: torch.Tensor,
    config: QuantConfig,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    """Absorb only the reducible residual into the FP branch at frozen codes.

    Unlike a full teacher re-fit of L1/L2, this solves for an additive low-rank
    correction B_delta matching (teacher - current_uniform_output), then
    re-factors baseline_branch + B_delta.  That keeps F_irr / already-correct
    mass out of the calibration update and is the operational form of eq. (9).
    """

    x = cached_x.to(device=current_l1.device, dtype=torch.float32)
    xtilde = x / d
    teacher = teacher_outputs.to(device=current_l1.device, dtype=torch.float32)
    current = _quantized_state_prediction(
        x, d, current_l1, current_l2, quantized_residual, config
    )
    delta = teacher - current
    rows = max(1, xtilde.shape[0])
    empirical_hessian = _sym(xtilde.T @ xtilde / float(rows))
    cross = xtilde.T @ delta / float(rows)
    unconstrained = torch.linalg.solve(
        _regularize_hessian(empirical_hessian, config.damp),
        cross,
    )
    # Rank-1 is the natural match to the scalar projection alpha used when
    # constructing the reducible teacher; allow up to rank when it helps fit.
    corr_rank = 1 if config.rank <= 1 else min(2, config.rank)
    delta_l1, delta_l2 = weighted_low_rank(
        empirical_hessian,
        unconstrained,
        config,
        rank=corr_rank,
        beta=0.5,
    )
    proposal_branch = current_l1 @ current_l2 + delta_l1 @ delta_l2
    proposal_l1, proposal_l2 = _factor_branch_matrix(
        empirical_hessian, proposal_branch, config
    )
    current_error = (teacher - current).square().mean()
    proposal = _quantized_state_prediction(
        x, d, proposal_l1, proposal_l2, quantized_residual, config
    )
    proposal_error = (teacher - proposal).square().mean()
    accepted = bool(proposal_error <= current_error * (1.0 + 1e-6))
    gain = float(((current_error - proposal_error) / current_error.clamp_min(1e-30)).item())
    diagnostics = {
        "rotation_fixed_code_mse_before": float(current_error.item()),
        "rotation_fixed_code_mse_after": float(proposal_error.item()),
        "rotation_fixed_code_gain": gain,
        "rotation_fixed_code_accepted": float(accepted),
        "reducible_correction_rank": float(corr_rank),
    }
    if not accepted:
        return current_l1, current_l2, diagnostics
    return proposal_l1, proposal_l2, diagnostics


@torch.no_grad()
def _joint_quantize_reducible_from_v2(
    layer: nn.Linear,
    hessian: torch.Tensor,
    cached_x: torch.Tensor,
    config: QuantConfig,
    target_weight: torch.Tensor | None,
    target_outputs: torch.Tensor | None,
    objective_diagnostics: dict[str, float] | None,
) -> dict[str, Any]:
    """Use the validated V2 solution as fallback, then optimize only F_A^red.

    Matches the reformulation (eq. 9): keep the uniform forward path and the
    V2 codes/scales/D frozen, then search only over the FP branch (L1, L2) for
    held-out reducible-teacher mismatch under the F_W trust region
    F_W <= (1+epsilon) F_W^(0).  The irreducible floor never enters the
    objective; V2 remains the feasible fallback when no eligible branch update
    improves F_A^red.
    """

    baseline_config = replace(
        config,
        activation_objective="full",
        joint_rotation_mode="none",
    )
    baseline = joint_quantize_linear(
        layer,
        hessian,
        cached_x,
        baseline_config,
        target_weight=target_weight,
        target_outputs=target_outputs,
        objective_diagnostics=objective_diagnostics,
    )
    device = layer.weight.device
    original_dtype = layer.weight.dtype
    local_weight = layer.weight.detach().T.float()
    weight = local_weight if target_weight is None else target_weight.to(device=device, dtype=torch.float32)
    hessian = hessian.to(device=device, dtype=torch.float32)
    d = baseline["d"].to(device=device, dtype=torch.float32)
    l1 = baseline["l1"].to(device=device, dtype=torch.float32)
    l2 = baseline["l2"].to(device=device, dtype=torch.float32)
    codes = baseline["codes"].to(device=device)
    scales = baseline["scales"].to(device=device, dtype=torch.float32)
    qres = _dequantize_codes(codes, scales, int(baseline["group_size"])).T.contiguous()
    (
        oracle_x,
        oracle_teacher,
        fit_index,
        test_index,
        teacher_diagnostics,
    ) = build_reducible_oracle_teacher(cached_x, d, l1, l2, qres, config)
    baseline_prediction = _quantized_state_prediction(
        oracle_x[test_index], d, l1, l2, qres, config
    )
    baseline_red_error = float(
        (baseline_prediction - oracle_teacher[test_index]).square().mean().item()
    )
    red_fit = float(teacher_diagnostics["reducible_fit_mse"])
    irr_test = float(teacher_diagnostics["irreducible_test_mse"])
    reliability = red_fit / max(red_fit + irr_test, 1e-30)
    fw_anchor = float(baseline["fw"])
    fw_limit = fw_anchor * (1.0 + config.joint_rotation_fw_epsilon)
    htilde = hessian / d[:, None] / d[None, :]
    wtilde = d[:, None] * weight
    baseline_branch = l1 @ l2
    best = dict(baseline)
    best.update(
        {
            "activation_objective": "reducible",
            "joint_rotation_mode": "empirical",
            "joint_rotation_fw_epsilon": config.joint_rotation_fw_epsilon,
            "error": baseline_red_error,
            "full_error": float(baseline.get("full_error", baseline.get("error", 0.0))),
            "fw_trust_anchor": fw_anchor,
            "fw_trust_limit": fw_limit,
            "fw_trust_eligible_rotations": 0,
            "fw_trust_total_rotations": 0,
            "reducible_teacher_diagnostics": dict(teacher_diagnostics),
            "reducible_reliability": float(reliability),
            "reducible_source": "v2_fallback",
            "reducible_accepted_updates": 0,
            "reducible_refine_history": [],
        }
    )
    proposal_l1, proposal_l2, rotation_diagnostics = reducible_fixed_code_branch_correction(
        oracle_x[fit_index],
        d,
        l1,
        l2,
        qres,
        oracle_teacher[fit_index],
        config,
    )
    proposal_branch = proposal_l1 @ proposal_l2
    # Closed-form additive correction, then trust-region line search toward it.
    # Codes/D stay frozen so F_A^red is not mixed with a fresh GPTQ residual.
    default_scales = (0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0)
    configured = tuple(
        float(scale) for scale in config.trajectory_backtrack_scales if float(scale) > 0
    )
    line_scales = tuple(sorted(set(configured or default_scales)))
    refine_history: list[dict[str, Any]] = []
    eligible = 0
    accepted = 0
    for scale in line_scales:
        branch = (1.0 - scale) * baseline_branch + scale * proposal_branch
        cand_l1, cand_l2 = _factor_branch_matrix(htilde, branch, config)
        terms = joint_surrogate_terms(
            htilde,
            wtilde,
            cand_l1,
            cand_l2,
            qres,
            torch.zeros_like(d),
            0.0,
        )
        prediction = _quantized_state_prediction(
            oracle_x[test_index], d, cand_l1, cand_l2, qres, config
        )
        red_error = float(
            (prediction - oracle_teacher[test_index]).square().mean().item()
        )
        full_error = _state_error(
            weight,
            hessian,
            cached_x,
            d,
            cand_l1,
            cand_l2,
            qres,
            config.activation_bits,
            config.activation_group_size,
            target_outputs,
        )
        is_eligible = float(terms["fw"]) <= fw_limit * (1.0 + 1e-7)
        eligible += int(is_eligible)
        red_gain = (baseline_red_error - red_error) / max(baseline_red_error, 1e-30)
        diagnostics = {
            "line_scale": float(scale),
            "state_mse": red_error,
            "full_state_mse": full_error,
            "fw": float(terms["fw"]),
            "fw_ratio": float(terms["fw"]) / max(fw_anchor, 1e-30),
            "fw_eligible": float(is_eligible),
            "reducible_heldout_gain": float(red_gain),
            "reducible_reliability": float(reliability),
            **teacher_diagnostics,
            **rotation_diagnostics,
        }
        refine_history.append(diagnostics)
        if is_eligible and red_error < float(best["error"]):
            accepted += 1
            candidate = dict(best)
            candidate.update(
                {
                    "d": d.detach().to("cpu", dtype=torch.float32),
                    "l1": cand_l1.detach().to("cpu", dtype=original_dtype),
                    "l2": cand_l2.detach().to("cpu", dtype=original_dtype),
                    "codes": codes.detach().to("cpu"),
                    "scales": scales.detach().to("cpu", dtype=original_dtype),
                    "error": red_error,
                    "full_error": full_error,
                    "fw": float(terms["fw"]),
                    "outer_iteration": int(baseline.get("outer_iteration", 0)),
                    "joint_code_iteration": int(baseline.get("joint_code_iteration", 0)),
                    "joint_diagnostics": list(baseline.get("joint_diagnostics", []))
                    + list(refine_history),
                    "reducible_source": "fixed_code_refine",
                    "reducible_line_scale": float(scale),
                }
            )
            best = candidate

    best["fw_trust_eligible_rotations"] = eligible
    best["fw_trust_total_rotations"] = len(line_scales)
    best["reducible_accepted_updates"] = accepted
    best["reducible_refine_history"] = refine_history
    best["history"] = list(baseline.get("history", [])) + [
        row["state_mse"] for row in refine_history
    ]
    return best


@torch.no_grad()
def joint_quantize_linear(
    layer: nn.Linear,
    hessian: torch.Tensor,
    cached_x: torch.Tensor | None,
    config: QuantConfig,
    target_weight: torch.Tensor | None = None,
    target_outputs: torch.Tensor | None = None,
    objective_diagnostics: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Jointly update D, L1, Z, and L2 for one Linear layer.

    ``target_weight`` optionally replaces the local W target with the projected
    cumulative-error target.  Runtime dimensions, dtype, bias, and checkpoint
    metadata continue to come from ``layer``.
    """

    config.validate()
    if config.activation_objective == "reducible":
        if cached_x is None:
            raise ValueError("reducible activation objective requires cached activations")
        return _joint_quantize_reducible_from_v2(
            layer,
            hessian,
            cached_x,
            config,
            target_weight,
            target_outputs,
            objective_diagnostics,
        )
    device = layer.weight.device
    original_dtype = layer.weight.dtype
    local_weight = layer.weight.detach().T.float()  # [in, out]
    weight = (
        local_weight
        if target_weight is None
        else target_weight.to(device=device, dtype=torch.float32)
    )
    if weight.shape != local_weight.shape:
        raise ValueError(f"target_weight has shape {tuple(weight.shape)}, expected {tuple(local_weight.shape)}")
    hessian = hessian.to(device=device, dtype=torch.float32)
    cached_x_device = None if cached_x is None else cached_x

    previous_a: torch.Tensor | None = None
    previous_l2: torch.Tensor | None = None
    best: dict[str, Any] | None = None
    history: list[float] = []
    reducible_x: torch.Tensor | None = None
    reducible_teacher: torch.Tensor | None = None
    reducible_fit_index: torch.Tensor | None = None
    reducible_test_index: torch.Tensor | None = None
    reducible_teacher_diagnostics: dict[str, float] = {}

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
        if cached_x_device is not None and config.activation_bits < 16:
            sigma_a = activation_noise_diagonal(
                cached_x_device,
                d,
                config.activation_bits,
                config.activation_group_size,
                device,
            )
        else:
            sigma_a = torch.zeros_like(d)
        split: dict[str, Any] = {"rank_w": config.rank, "rank_a": 0}
        if config.rank_a > 0 and cached_x_device is not None and config.activation_bits < 16:
            l1_a, s_a = activation_aware_branch(wtilde, sigma_a, config.rank_a, config)
            g_a1 = float(s_a[0].square()) if s_a.numel() else 0.0
            eig_tail = torch.linalg.eigvalsh(htilde).clamp_min(0)
            g_w_next = (
                float(eig_tail[-(config.rank - config.rank_a + 1)])
                if eig_tail.numel() >= config.rank - config.rank_a + 1
                else 0.0
            )
            split.update({"g_a1": g_a1, "g_w_next": g_w_next})
            # Gate (Prop. split): spend a rank unit on the F_A block only when its
            # marginal gain beats the next F_W deflation direction; gated-out
            # modules keep the full budget on the W block.
            use_split = config.rank_a_mode == "fixed" or g_a1 > g_w_next
            split["gated_in"] = bool(use_split) if config.rank_a_mode == "gated" else None
            if use_split:
                l1_w, _ = weighted_low_rank(htilde, wtilde, config, rank=config.rank - config.rank_a)
                l1 = _whiten_h_gauge(torch.cat([l1_w, l1_a], dim=1), htilde)
                # Pre-GPTQ L2: joint refit in the combined metric Htilde + Sigma_A,
                # i.e. the branch is fitted against F_W and F_A simultaneously
                # before the residual is frozen for quantization.
                l2_initial = refit_l2(htilde + torch.diag(sigma_a), wtilde, torch.zeros_like(wtilde), l1)
                split.update({"rank_w": config.rank - config.rank_a, "rank_a": config.rank_a})
            else:
                l1, l2_initial = weighted_low_rank(htilde, wtilde, config)
        else:
            if config.rank_a > 0 and (cached_x_device is None or config.activation_bits >= 16):
                pass  # no activation channel to serve; full budget stays on W
            l1, l2_initial = weighted_low_rank(htilde, wtilde, config)

        inner_iters = config.joint_code_iters if config.code_objective == "joint" else 1
        # Keep the unrotated candidates so empirical rotation can be admitted
        # against the F_W value of the exact V2 solution selected without it.
        # This makes F_W^(0) a genuine per-module trust-region anchor rather
        # than approximating it with equal rank or a fixed-code comparison.
        if outer == 0:
            unrotated_candidates: list[dict[str, Any]] = []
            rotated_candidates: list[dict[str, Any]] = []
        inner_history: list[dict[str, Any]] = []
        incoming_rotation_diagnostics: dict[str, float] = {}
        for joint_iteration in range(inner_iters):
            hperp_tilde = _hessian_deflate(htilde, l1)
            if (
                config.activation_objective == "full"
                and config.code_objective == "joint"
                and config.activation_bits < 16
            ):
                target, code_metric, prepared_upper, metric_jitter = joint_code_target(
                    hperp_tilde,
                    wtilde,
                    sigma_a,
                    config,
                )
                quantized_residual, codes, scales = gptq_quantize_residual(
                    target,
                    code_metric,
                    config,
                    prepared_upper=prepared_upper,
                )
                target_norm_ratio = float(
                    target.norm().div(wtilde.norm().clamp_min(1e-20)).item()
                )
            else:
                residual = wtilde - l1 @ l2_initial
                quantized_residual, codes, scales = gptq_quantize_residual(
                    residual,
                    hperp_tilde,
                    config,
                )
                target_norm_ratio = 1.0
                metric_jitter = 0.0

            # Once the complete quantized tensor is frozen, the total objective's
            # exact L2 block is the original H-metric least-squares refit.
            l2 = refit_l2(htilde, wtilde, quantized_residual, l1)
            terms = joint_surrogate_terms(
                htilde,
                wtilde,
                l1,
                l2,
                quantized_residual,
                sigma_a,
                config.activation_weight if config.activation_objective == "full" else 0.0,
            )
            full_error = _state_error(
                weight,
                hessian,
                cached_x_device,
                d,
                l1,
                l2,
                quantized_residual,
                config.activation_bits,
                config.activation_group_size,
                target_outputs,
            )
            if config.activation_objective == "reducible":
                if cached_x_device is None:
                    raise ValueError("reducible activation objective requires cached activations")
                if reducible_teacher is None:
                    (
                        reducible_x,
                        reducible_teacher,
                        reducible_fit_index,
                        reducible_test_index,
                        reducible_teacher_diagnostics,
                    ) = build_reducible_oracle_teacher(
                        cached_x_device,
                        d,
                        l1,
                        l2,
                        quantized_residual,
                        config,
                    )
                assert reducible_x is not None
                assert reducible_test_index is not None
                prediction = _quantized_state_prediction(
                    reducible_x[reducible_test_index],
                    d,
                    l1,
                    l2,
                    quantized_residual,
                    config,
                )
                error = float(
                    (prediction - reducible_teacher[reducible_test_index]).square().mean().item()
                )
            else:
                error = full_error
            history.append(error)
            inner_history.append(
                {
                    "iteration": joint_iteration,
                    "state_mse": error,
                    "target_norm_ratio": target_norm_ratio,
                    "metric_jitter": metric_jitter,
                    **incoming_rotation_diagnostics,
                    **reducible_teacher_diagnostics,
                    **terms,
                }
            )
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
                "activation_group_size": config.activation_group_size,
                "code_objective": config.code_objective,
                "joint_rotation_mode": config.joint_rotation_mode,
                "joint_rotation_fw_epsilon": config.joint_rotation_fw_epsilon,
                "activation_objective": config.activation_objective,
                "joint_code_iteration": joint_iteration,
                "joint_diagnostics": list(inner_history),
                "sigma_a_mean": float(sigma_a.mean().item()),
                "sigma_a_max": float(sigma_a.max().item()),
                "error": error,
                "full_error": full_error,
                "outer_iteration": outer,
                "rank_split": dict(split),
                "linear_objective": config.linear_objective,
                "ablation_mode": config.ablation_mode,
                "objective_diagnostics": dict(objective_diagnostics or {}),
                "fw": float(terms["fw"]),
            }
            if config.joint_rotation_mode == "empirical":
                if joint_iteration == 0:
                    unrotated_candidates.append(candidate)
                else:
                    rotated_candidates.append(candidate)
            if best is None or error < best["error"]:
                best = candidate

            if joint_iteration + 1 < inner_iters:
                if config.joint_rotation_mode == "empirical":
                    if cached_x_device is None:
                        raise ValueError(
                            "empirical joint rotation requires --activation-cache-tokens > 0"
                        )
                    rotation_x = cached_x_device
                    rotation_target = target_outputs
                    if config.activation_objective == "reducible":
                        assert reducible_x is not None
                        assert reducible_fit_index is not None
                        assert reducible_teacher is not None
                        rotation_x = reducible_x[reducible_fit_index]
                        rotation_target = reducible_teacher[reducible_fit_index]
                    l1, l2_initial, incoming_rotation_diagnostics = empirical_joint_branch_update(
                        rotation_x,
                        d,
                        weight,
                        quantized_residual,
                        l1,
                        l2,
                        config,
                        rotation_target,
                    )
                else:
                    # Exact fixed-Q surrogate branch block: fit the rank-r
                    # branch to what the quantized path did not carry.
                    l1, l2_initial = weighted_low_rank(
                        htilde,
                        wtilde - quantized_residual,
                        config,
                        rank=config.rank,
                        beta=0.5,
                    )
                    incoming_rotation_diagnostics = {}

        previous_a = l1 / d[:, None]
        previous_l2 = l2

    if config.joint_rotation_mode == "empirical":
        if not unrotated_candidates:
            raise RuntimeError("empirical joint rotation produced no unrotated F_W anchor")
        if config.activation_objective == "reducible":
            # The first FW-only state defines both the frozen oracle teacher
            # and F_W^(0); every later D/L/Z candidate is compared to that one
            # cross-fitted objective and cannot move the trust-region anchor.
            baseline_best = unrotated_candidates[0]
            alternatives = [*unrotated_candidates[1:], *rotated_candidates]
        else:
            baseline_best = min(unrotated_candidates, key=lambda state: state["error"])
            alternatives = list(rotated_candidates)
        fw_anchor = float(baseline_best["fw"])
        fw_limit = fw_anchor * (1.0 + config.joint_rotation_fw_epsilon)
        eligible_rotated = [
            state for state in alternatives if float(state["fw"]) <= fw_limit * (1.0 + 1e-7)
        ]
        best = min([baseline_best, *eligible_rotated], key=lambda state: state["error"])
        best["fw_trust_anchor"] = fw_anchor
        best["fw_trust_limit"] = fw_limit
        best["fw_trust_eligible_rotations"] = len(eligible_rotated)
        best["fw_trust_total_rotations"] = len(rotated_candidates)

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
        self.activation_group_size = int(state.get("activation_group_size", 0))
        dtype = compute_dtype or state["l1"].dtype
        self.register_buffer("d", state["d"].to(dtype=dtype))
        self.register_buffer("l1", state["l1"].to(dtype=dtype))
        self.register_buffer("l2", state["l2"].to(dtype=dtype))
        self.register_buffer("codes", state["codes"].to(torch.int8))
        self.register_buffer("scales", state["scales"].to(dtype=dtype))
        bias = state.get("bias")
        self.register_buffer("bias", None if bias is None else bias.to(dtype=dtype))
        # Dense residual is rebuilt per forward and discarded. Caching it on every
        # linear after the first eval pass roughly doubles weight VRAM vs compact codes.
        self.register_buffer("_qweight", None, persistent=False)
        self.persist_qweight = False
        correction = state.get("correction") or {}
        dc = correction.get("dc_coeff")
        if dc is None and isinstance(correction.get("dc"), dict):
            dc = correction["dc"].get("coeff")
        lut = correction.get("lut_coeff")
        if lut is None and isinstance(correction.get("lut"), dict):
            lut = correction["lut"].get("coeff")
        sparse = correction.get("sparse") if isinstance(correction.get("sparse"), dict) else {}
        sparse_threshold = correction.get("sparse_threshold", sparse.get("threshold", None))
        generic = correction.get("generic") if isinstance(correction.get("generic"), dict) else {}
        generic_left = correction.get("generic_left", generic.get("left", None))
        generic_right = correction.get("generic_right", generic.get("right", None))
        if (generic_left is None) != (generic_right is None):
            raise ValueError("generic correction requires both left and right factors")
        self.register_buffer("correction_dc", None if dc is None else dc.to(dtype=dtype))
        self.register_buffer("correction_lut", None if lut is None else lut.to(dtype=dtype))
        self.register_buffer(
            "correction_generic_left",
            None if generic_left is None else generic_left.to(dtype=dtype),
        )
        self.register_buffer(
            "correction_generic_right",
            None if generic_right is None else generic_right.to(dtype=dtype),
        )
        self.correction_sparse_threshold = None if sparse_threshold is None else float(sparse_threshold)

    def _build_qweight(self, dtype: torch.dtype | None = None) -> torch.Tensor:
        dtype = dtype or self.scales.dtype
        return _dequantize_codes(self.codes, self.scales, self.group_size).to(dtype=dtype)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if (
            self.persist_qweight
            and self._qweight is not None
            and self._qweight.device == inputs.device
            and self._qweight.dtype == inputs.dtype
        ):
            qweight = self._qweight
        else:
            qweight = self._build_qweight(inputs.dtype).to(device=inputs.device, dtype=inputs.dtype)
            if self.persist_qweight:
                self._qweight = qweight
        d = self.d.to(dtype=inputs.dtype)
        smoothed = inputs / d
        quantized_inputs, activation_codes, _activation_scales = _quantize_activation_with_codes(
            smoothed,
            self.activation_bits,
            self.activation_group_size,
        )
        output = F.linear(quantized_inputs, qweight, None)
        if self.l1.shape[1]:
            output = output + (smoothed @ self.l1.to(inputs.dtype)) @ self.l2.to(inputs.dtype)
        if self.correction_dc is not None:
            centers = _activation_group_centers(smoothed, self.activation_group_size)
            output = output + centers.to(inputs.dtype) @ self.correction_dc.to(inputs.dtype)
        if self.correction_lut is not None:
            hist = _activation_code_histograms(
                activation_codes,
                self.activation_bits,
                self.activation_group_size,
            )
            coeff = self.correction_lut.to(inputs.dtype).reshape(-1, self.out_features)
            output = output + hist.to(inputs.dtype) @ coeff
        if self.correction_sparse_threshold is not None:
            mask = _normalized_activation_mask(
                smoothed,
                self.activation_group_size,
                self.correction_sparse_threshold,
            )
            sparse_residual = (smoothed - quantized_inputs) * mask.to(dtype=inputs.dtype)
            output = output + F.linear(sparse_residual, qweight, None)
        if self.correction_generic_left is not None:
            activation_residual = smoothed - quantized_inputs
            output = output + (
                activation_residual @ self.correction_generic_left.to(inputs.dtype)
            ) @ self.correction_generic_right.to(inputs.dtype)
        if self.bias is not None:
            output = output + self.bias.to(inputs.dtype)
        if not self.persist_qweight:
            self._qweight = None
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


def _trajectory_module_allowed(name: str, config: QuantConfig) -> bool:
    if config.trajectory_module_filter == "all":
        return True
    if config.trajectory_module_filter == "down_proj":
        return name.endswith("down_proj")
    if config.trajectory_module_filter == "mlp":
        return any(name.endswith(kind) for kind in ("gate_proj", "up_proj", "down_proj"))
    if config.trajectory_module_filter == "attention":
        return any(name.endswith(kind) for kind in ("q_proj", "k_proj", "v_proj", "o_proj"))
    return True


def _state_error_from_state(
    original_weight: torch.Tensor,
    hessian: torch.Tensor,
    cached_x: torch.Tensor | None,
    state: dict[str, Any],
    target_outputs: torch.Tensor | None,
) -> float:
    device = original_weight.device
    d = state["d"].to(device=device, dtype=torch.float32)
    l1 = state["l1"].to(device=device, dtype=torch.float32)
    l2 = state["l2"].to(device=device, dtype=torch.float32)
    codes = state["codes"].to(device=device)
    scales = state["scales"].to(device=device, dtype=torch.float32)
    quantized_residual = _dequantize_codes(codes, scales, int(state["group_size"])).T.contiguous()
    return _state_error(
        original_weight,
        hessian,
        cached_x,
        d,
        l1,
        l2,
        quantized_residual,
        int(state["activation_bits"]),
        int(state.get("activation_group_size", 0)),
        target_outputs,
    )


class _StopForward(RuntimeError):
    pass


def _decoder_layers(model: nn.Module) -> nn.ModuleList:
    """Return the decoder stack for causal LMs or Qwen2.5-VL language towers."""
    if hasattr(model, "model"):
        inner = model.model
        if hasattr(inner, "layers"):
            return inner.layers
        if hasattr(inner, "language_model") and hasattr(inner.language_model, "layers"):
            return inner.language_model.layers
    if hasattr(model, "language_model") and hasattr(model.language_model, "layers"):
        return model.language_model.layers
    raise TypeError("expected a Qwen/Llama causal LM or Qwen2.5-VL language_model.layers architecture")


def _decoder_layer_prefix(model: nn.Module) -> str:
    if hasattr(model, "model"):
        inner = model.model
        if hasattr(inner, "layers"):
            return "model.layers"
        if hasattr(inner, "language_model") and hasattr(inner.language_model, "layers"):
            return "model.language_model.layers"
    if hasattr(model, "language_model") and hasattr(model.language_model, "layers"):
        return "language_model.layers"
    raise TypeError("expected a Qwen/Llama causal LM or Qwen2.5-VL language_model.layers architecture")


def _model_kind(model: nn.Module) -> str:
    model_type = getattr(getattr(model, "config", None), "model_type", "")
    if model_type == "qwen2_5_vl" or _decoder_layer_prefix(model).endswith("language_model.layers"):
        return "qwen2_5_vl"
    return "causal_lm"


@torch.no_grad()
def capture_first_layer_inputs(
    model: nn.Module,
    batches: Iterable[dict[str, torch.Tensor]],
    device: torch.device,
    *,
    stage_minimal: bool = False,
) -> tuple[list[torch.Tensor], list[dict[str, Any]]]:
    layers = _decoder_layers(model)
    hidden_batches: list[torch.Tensor] = []
    layer_kwargs: list[dict[str, Any]] = []
    staged: list[nn.Module] = []
    if stage_minimal:
        # Avoid putting the full FP16/BF16 model on GPU just to grab layer-0 inputs.
        staged = _stage_first_layer_for_capture(model, device)

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
        if stage_minimal:
            for module in staged:
                module.to("cpu")
            _cuda_empty_cache(device)
    if not hidden_batches:
        raise RuntimeError("failed to capture the first decoder-layer inputs")
    return hidden_batches, layer_kwargs


def _decoder_hidden(output: Any) -> torch.Tensor:
    """Handle Transformers decoder layers that return either a tensor or tuple."""
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (tuple, list)) and output and isinstance(output[0], torch.Tensor):
        return output[0]
    raise TypeError(f"unexpected decoder layer output type: {type(output)!r}")


def _paired_batch_stats(
    student: list[torch.Tensor], teacher: list[torch.Tensor]
) -> dict[str, float]:
    """Streaming teacher/student trajectory metrics without concatenating caches."""
    squared_error = 0.0
    teacher_energy = 0.0
    student_energy = 0.0
    inner_product = 0.0
    elements = 0
    for student_batch, teacher_batch in zip(student, teacher, strict=True):
        lhs = student_batch.float()
        rhs = teacher_batch.float()
        delta = lhs - rhs
        squared_error += float(delta.square().sum().item())
        teacher_energy += float(rhs.square().sum().item())
        student_energy += float(lhs.square().sum().item())
        inner_product += float((lhs * rhs).sum().item())
        elements += delta.numel()
    mse = squared_error / float(max(1, elements))
    teacher_mse = teacher_energy / float(max(1, elements))
    return {
        "mse": mse,
        "teacher_energy": teacher_mse,
        "student_energy": student_energy / float(max(1, elements)),
        "nmse": mse / max(teacher_mse, 1e-30),
        "student_normalized_mse": mse / max(student_energy / float(max(1, elements)), 1e-30),
        "cosine": inner_product / max(math.sqrt(student_energy * teacher_energy), 1e-30),
    }


def _cuda_empty_cache(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.empty_cache()


def _offload_decoder_layers_except(
    model: nn.Module,
    keep_index: int | None,
    compute_device: torch.device,
) -> None:
    """Keep at most one decoder layer on the compute device; park the rest on CPU."""
    layers = _decoder_layers(model)
    for index, layer in enumerate(layers):
        target = compute_device if keep_index is not None and index == keep_index else torch.device("cpu")
        layer.to(target)


def _language_tower(model: nn.Module) -> nn.Module | None:
    if hasattr(model, "model"):
        inner = model.model
        if hasattr(inner, "language_model") and hasattr(inner.language_model, "layers"):
            return inner.language_model
        return inner
    if hasattr(model, "language_model"):
        return model.language_model
    return None


def _drop_reconstructed_qweights(root: nn.Module) -> None:
    for module in root.modules():
        if isinstance(module, HSVQuantLinear) and module._qweight is not None:
            module._qweight = None


def enable_eval_cpu_offload(model: nn.Module, device: torch.device) -> None:
    """Keep embeddings / norm / lm_head on GPU; page decoder blocks in for each forward.

    Numerically identical to a full-GPU eval: each block runs in the same dtype on
    ``device``, then its parameters are parked on CPU so peak VRAM is one block plus
    activations and the language-model head.
    """
    if getattr(model, "_eval_cpu_offload", False):
        return
    inner = _language_tower(model)
    if inner is not None:
        for attr in ("embed_tokens", "rotary_emb", "norm"):
            module = getattr(inner, attr, None)
            if isinstance(module, nn.Module):
                module.to(device)
    if hasattr(model, "lm_head") and isinstance(model.lm_head, nn.Module):
        model.lm_head.to(device)
    layers = _decoder_layers(model)
    for layer in layers:
        layer.to("cpu")

        def _pre_hook(mod: nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any], compute=device):
            mod.to(compute)
            return args, kwargs

        def _post_hook(mod: nn.Module, _args: tuple[Any, ...], output: Any) -> Any:
            _drop_reconstructed_qweights(mod)
            mod.to("cpu")
            return output

        layer.register_forward_pre_hook(_pre_hook, with_kwargs=True)
        layer.register_forward_hook(_post_hook)
    model.config.use_cache = False
    model.eval()
    model._eval_cpu_offload = True
    _cuda_empty_cache(device)
    gc.collect()


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
    cpu_offload_layers: bool = False,
    layer_checkpoint_dir: Path | None = None,
) -> dict[str, dict[str, Any]]:
    layers = _decoder_layers(model)
    layer_prefix = _decoder_layer_prefix(model)
    states: dict[str, dict[str, Any]] = {}
    layer_count = len(layers) if max_layers < 0 else min(max_layers, len(layers))
    propagated_hidden_batches = hidden_batches
    reference_hidden_batches = hidden_batches
    diagnostic_reference_hidden_batches = hidden_batches
    need_reference_path = (
        config.linear_objective == "cumulative"
        or config.block_input_mode == "reference"
        or config.trajectory_diagnostics
    )
    if cpu_offload_layers:
        print(
            "[H-SVDQuant] cpu_offload_layers=on: only the active decoder block stays on "
            f"{device}; finished blocks are parked on CPU / optionally snapshotted to disk",
            flush=True,
        )
        _offload_decoder_layers_except(model, keep_index=None, compute_device=device)
        _cuda_empty_cache(device)

    for layer_index in range(layer_count):
        layer = layers[layer_index]
        print(f"[H-SVDQuant] layer {layer_index + 1}/{layer_count}", flush=True)
        if cpu_offload_layers:
            _offload_decoder_layers_except(model, keep_index=layer_index, compute_device=device)
            _cuda_empty_cache(device)
        sequential_groups = _qwen_sequential_groups(layer)
        trajectory_active = (
            config.linear_objective == "cumulative"
            and layer_index >= config.trajectory_start_layer
        )
        if (
            config.trajectory_rebase
            and config.linear_objective == "cumulative"
            and layer_index == config.trajectory_start_layer
        ):
            # Start a new FP target trajectory from the current student state.
            # The independent diagnostic teacher below remains the original FP path.
            reference_hidden_batches = propagated_hidden_batches
        reference_collectors: dict[str, ActivationCache] = {}
        next_reference_hidden: list[torch.Tensor] | None = None
        next_diagnostic_reference_hidden: list[torch.Tensor] | None = None

        if config.trajectory_diagnostics:
            next_diagnostic_reference_hidden = []
            for hidden, kwargs in zip(
                diagnostic_reference_hidden_batches, layer_kwargs, strict=True
            ):
                output = _decoder_hidden(layer(hidden.to(device), **_move_tree(kwargs, device)))
                next_diagnostic_reference_hidden.append(output.detach().to("cpu"))

        # Preserve the full-precision path before replacing any module in this
        # block.  In cumulative mode these paired caches provide X while the
        # group-wise walk below provides Xhat with identical reservoir indices.
        if need_reference_path:
            handles = []
            if trajectory_active:
                module_offset = 0
                for group in sequential_groups:
                    for offset, name in enumerate(group):
                        module = _get_submodule(layer, name)
                        cache_seed = seed + layer_index * 100 + (
                            module_offset
                            if config.intra_block_mode == "fp_independent"
                            else offset
                        )
                        cache = ActivationCache(
                            module.in_features,
                            cache_tokens,
                            cache_seed,
                        )
                        reference_collectors[name] = cache
                        if config.intra_block_mode == "fp_independent":
                            module_offset += 1

                        def reference_hook(
                            _module: nn.Module,
                            args: tuple[Any, ...],
                            _output: Any,
                            target=cache,
                        ) -> None:
                            target.add_batch(args[0])

                        handles.append(module.register_forward_hook(reference_hook))

            next_reference_hidden = []
            for hidden, kwargs in zip(reference_hidden_batches, layer_kwargs, strict=True):
                output = _decoder_hidden(layer(hidden.to(device), **_move_tree(kwargs, device)))
                next_reference_hidden.append(output.detach().to("cpu"))
            for handle in handles:
                handle.remove()

        block_hidden_batches = (
            propagated_hidden_batches
            if config.block_input_mode == "quantized"
            else reference_hidden_batches
        )
        block_input_stats = (
            _paired_batch_stats(
                propagated_hidden_batches,
                diagnostic_reference_hidden_batches
                if config.trajectory_diagnostics
                else reference_hidden_batches,
            )
            if need_reference_path
            else None
        )

        def quantize_module(
            layer_index: int,
            name: str,
            module: nn.Linear,
            hessian: torch.Tensor,
            cached_x: torch.Tensor | None,
        ) -> dict[str, Any]:
            started = time.time()
            target_weight = None
            target_outputs = None
            objective_diagnostics = None
            module_trajectory_active = trajectory_active and _trajectory_module_allowed(name, config)
            if trajectory_active and not module_trajectory_active:
                objective_diagnostics = {
                    "trajectory_skipped_by_filter": 1.0,
                    "trajectory_quantized_acceptance": 0.0,
                }
            if module_trajectory_active:
                if cached_x is None:
                    raise ValueError("cumulative linear objective requires --activation-cache-tokens > 0")
                reference_x = reference_collectors[name].finalize()
                target_weight, target_outputs, objective_diagnostics = cumulative_target_weight(
                    module,
                    hessian,
                    cached_x,
                    reference_x,
                    config,
                )
                reliable = (
                    objective_diagnostics.get("holdout_gain", 0.0)
                    >= config.trajectory_min_holdout_gain
                    and objective_diagnostics.get("split_direction_cosine", 1.0)
                    >= config.trajectory_min_direction_cosine
                )
                if not reliable:
                    objective_diagnostics["trajectory_reliability_reject"] = 1.0
                    target_weight = None
                    target_outputs = None
                else:
                    objective_diagnostics["trajectory_reliability_reject"] = 0.0
            local_state = None
            if (
                module_trajectory_active
                and config.trajectory_quantized_gate
                and target_outputs is not None
            ):
                local_state = joint_quantize_linear(
                    module,
                    hessian,
                    cached_x,
                    config,
                )
            state = joint_quantize_linear(
                module,
                hessian,
                cached_x,
                config,
                target_weight=target_weight,
                target_outputs=target_outputs,
                objective_diagnostics=objective_diagnostics,
            )
            if local_state is not None and target_outputs is not None:
                local_teacher_error = _state_error_from_state(
                    module.weight.detach().T.float(),
                    hessian.to(device=device, dtype=torch.float32),
                    cached_x,
                    local_state,
                    target_outputs,
                )
                state["objective_diagnostics"]["local_teacher_error"] = float(local_teacher_error)
                state["objective_diagnostics"]["trajectory_teacher_error"] = float(state["error"])
                if state["error"] <= local_teacher_error * (1.0 - config.trajectory_min_holdout_gain):
                    state["objective_diagnostics"]["trajectory_quantized_acceptance"] = 1.0
                else:
                    local_state["objective_diagnostics"] = dict(state["objective_diagnostics"])
                    local_state["objective_diagnostics"]["trajectory_quantized_acceptance"] = 0.0
                    local_state["objective_diagnostics"]["trajectory_quantized_reverted"] = 1.0
                    local_state["error"] = float(local_teacher_error)
                    state = local_state
            full_name = f"{layer_prefix}.{layer_index}.{name}"
            states[full_name] = state
            print(
                f"  {name}: best_mse={state['error']:.6e}, "
                f"iter={state['outer_iteration']}, time={time.time() - started:.1f}s",
                flush=True,
            )
            return state

        if config.intra_block_mode == "fp_independent":
            collectors: dict[str, ActivationStats] = {}
            handles = []
            module_offset = 0
            calibration_batches = (
                propagated_hidden_batches
                if config.linear_objective == "cumulative"
                else block_hidden_batches
            )
            for group in sequential_groups:
                for name in group:
                    module = _get_submodule(layer, name)
                    stats = ActivationStats(
                        module.in_features,
                        device,
                        cache_tokens,
                        hessian_block_size,
                        seed + layer_index * 100 + module_offset,
                    )
                    collectors[name] = stats
                    module_offset += 1

                    def hook(_module: nn.Module, args: tuple[Any, ...], _output: Any, target=stats) -> None:
                        target.add_batch(args[0])

                    handles.append(module.register_forward_hook(hook))
            for hidden, kwargs in zip(calibration_batches, layer_kwargs, strict=True):
                layer(hidden.to(device), **_move_tree(kwargs, device))
            for handle in handles:
                handle.remove()

            pending: list[tuple[str, nn.Linear, dict[str, Any]]] = []
            for group in sequential_groups:
                for name in group:
                    module = _get_submodule(layer, name)
                    hessian, cached_x = collectors[name].finalize()
                    state = quantize_module(layer_index, name, module, hessian, cached_x)
                    pending.append((name, module, state))
                    collectors[name].free()
                    if name in reference_collectors:
                        reference_collectors[name].free()
            for name, module, state in pending:
                replacement = HSVQuantLinear(state, compute_dtype=module.weight.dtype).to(device)
                _set_submodule(layer, name, replacement)
        else:
            for group in sequential_groups:
                collectors = {}
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

                for hidden, kwargs in zip(block_hidden_batches, layer_kwargs, strict=True):
                    layer(hidden.to(device), **_move_tree(kwargs, device))
                for handle in handles:
                    handle.remove()

                for name in group:
                    module = _get_submodule(layer, name)
                    hessian, cached_x = collectors[name].finalize()
                    state = quantize_module(layer_index, name, module, hessian, cached_x)
                    replacement = HSVQuantLinear(state, compute_dtype=module.weight.dtype).to(device)
                    _set_submodule(layer, name, replacement)
                    collectors[name].free()
                    if name in reference_collectors:
                        reference_collectors[name].free()
                    del hessian, cached_x, module
                    _cuda_empty_cache(device)

        if config.block_input_mode == "quantized":
            next_hidden: list[torch.Tensor] = []
            for hidden, kwargs in zip(propagated_hidden_batches, layer_kwargs, strict=True):
                output = _decoder_hidden(layer(hidden.to(device), **_move_tree(kwargs, device)))
                next_hidden.append(output.detach().to("cpu"))
            propagated_hidden_batches = next_hidden
        else:
            assert next_reference_hidden is not None
            propagated_hidden_batches = next_reference_hidden
        if next_reference_hidden is not None:
            reference_hidden_batches = next_reference_hidden
        if next_diagnostic_reference_hidden is not None:
            diagnostic_reference_hidden_batches = next_diagnostic_reference_hidden
        if need_reference_path:
            assert next_reference_hidden is not None
            diagnostic_teacher = (
                next_diagnostic_reference_hidden
                if next_diagnostic_reference_hidden is not None
                else next_reference_hidden
            )
            block_output_stats = _paired_batch_stats(propagated_hidden_batches, diagnostic_teacher)
            target_output_stats = _paired_batch_stats(
                propagated_hidden_batches, next_reference_hidden
            )
            input_mse = float(block_input_stats["mse"] if block_input_stats else 0.0)
            input_nmse = float(block_input_stats["nmse"] if block_input_stats else 0.0)
            output_mse = float(block_output_stats["mse"])
            output_nmse = float(block_output_stats["nmse"])
            block_diagnostics = {
                "input_mse": input_mse,
                "output_mse": output_mse,
                "input_nmse": input_nmse,
                "output_nmse": output_nmse,
                "input_teacher_energy": float(block_input_stats["teacher_energy"] if block_input_stats else 0.0),
                "input_student_energy": float(block_input_stats["student_energy"] if block_input_stats else 0.0),
                "output_teacher_energy": float(block_output_stats["teacher_energy"]),
                "output_student_energy": float(block_output_stats["student_energy"]),
                "input_student_normalized_mse": float(
                    block_input_stats["student_normalized_mse"] if block_input_stats else 0.0
                ),
                "output_student_normalized_mse": float(block_output_stats["student_normalized_mse"]),
                "input_cosine": float(block_input_stats["cosine"] if block_input_stats else 1.0),
                "output_cosine": float(block_output_stats["cosine"]),
                "target_output_nmse": float(target_output_stats["nmse"]),
                "target_output_cosine": float(target_output_stats["cosine"]),
                "trajectory_active": float(trajectory_active),
                "trajectory_rebased": float(
                    config.trajectory_rebase
                    and layer_index >= config.trajectory_start_layer
                ),
                "error_delta": float(output_mse - input_mse),
                "nmse_delta": float(output_nmse - input_nmse),
                "correction_gain": (
                    1.0 - output_nmse / input_nmse
                    if input_nmse > 1e-30
                    else 0.0
                ),
            }
            for group in sequential_groups:
                for name in group:
                    states[f"{layer_prefix}.{layer_index}.{name}"][
                        "block_trajectory_diagnostics"
                    ] = block_diagnostics
        if layer_checkpoint_dir is not None:
            layer_checkpoint_dir.mkdir(parents=True, exist_ok=True)
            layer_states = {
                name: state
                for name, state in states.items()
                if name.startswith(f"{layer_prefix}.{layer_index}.")
            }
            torch.save(layer_states, layer_checkpoint_dir / f"layer_{layer_index:03d}.pt")
        if cpu_offload_layers:
            # Drop rebuilt dense residuals before parking the finished block.
            for module in layer.modules():
                if isinstance(module, HSVQuantLinear) and module._qweight is not None:
                    module._qweight = None
            layer.to("cpu")
            _cuda_empty_cache(device)
        else:
            _cuda_empty_cache(device)
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
        local_root = os.environ.get("HSVDQ_WIKITEXT2_DIR", "")
        if local_root:
            from datasets import load_from_disk

            dataset = load_from_disk(local_root)["train"]
        else:
            local_parquet = os.environ.get("HSVDQ_WIKITEXT2_TRAIN_PARQUET", "")
            if local_parquet:
                dataset = load_dataset("parquet", data_files=local_parquet, split="train")
            else:
                dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        encoded = tokenizer("\n\n".join(dataset["text"]), return_tensors="pt").input_ids[0]
        if encoded.numel() <= sequence_length:
            raise RuntimeError("calibration corpus is shorter than sequence_length")
        for _ in range(nsamples):
            start = rng.randint(0, encoded.numel() - sequence_length - 1)
            samples.append(encoded[start : start + sequence_length])
    elif dataset_name == "c4":
        local_c4 = os.environ.get("HSVDQ_C4_TRAIN", "")
        endpoint = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
        url = f"{endpoint}/datasets/allenai/c4/resolve/main/en/c4-train.00000-of-01024.json.gz"
        source = open(local_c4, "rb") if local_c4 else urllib.request.urlopen(url, timeout=120)
        with source as response:
            rows = list(gzip.GzipFile(fileobj=response))
        rng.shuffle(rows)
        for line in rows:
            ids = tokenizer(json.loads(line)["text"], return_tensors="pt", truncation=False).input_ids[0]
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
    model_kind: str = "causal_lm",
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(output_dir)
    torch.save(states, output_dir / "hsvdquant.pt")
    metadata = {
        "format": "hsvdquant-v1",
        "base_model": model_name,
        "model_kind": model_kind,
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
    *,
    cpu_offload_layers: bool = False,
    runtime_backend: str = "eager",
    allow_activation_group_remap: bool = False,
) -> tuple[nn.Module, Any, dict[str, Any]]:
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    if runtime_backend not in {"eager", "nunchaku"}:
        raise ValueError(f"unsupported runtime backend: {runtime_backend}")
    if runtime_backend == "nunchaku" and dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("Nunchaku runtime requires float16 or bfloat16")
    metadata = json.loads((checkpoint_dir / "hsvdquant_config.json").read_text(encoding="utf-8"))
    model_kind = metadata.get("model_kind", "")
    if not model_kind:
        try:
            model_type = AutoConfig.from_pretrained(metadata["base_model"]).model_type
        except Exception:
            model_type = ""
        model_kind = "qwen2_5_vl" if model_type == "qwen2_5_vl" else "causal_lm"
    if model_kind == "qwen2_5_vl":
        from transformers import Qwen2_5_VLForConditionalGeneration

        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            metadata["base_model"], torch_dtype=dtype
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(metadata["base_model"], torch_dtype=dtype)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir, use_fast=True)
    try:
        states = torch.load(checkpoint_dir / "hsvdquant.pt", map_location="cpu", weights_only=True)
    except TypeError:
        states = torch.load(checkpoint_dir / "hsvdquant.pt", map_location="cpu")
    if runtime_backend == "nunchaku":
        from hsvdquant_int4 import build_nunchaku_linear

    activation_group_remapped = False
    for name, state in states.items():
        if runtime_backend == "nunchaku":
            source_activation_group = int(state.get("activation_group_size", 0))
            activation_group_remapped |= source_activation_group != 64
            replacement = build_nunchaku_linear(
                state,
                dtype,
                allow_activation_group_remap=allow_activation_group_remap,
            )
        else:
            replacement = HSVQuantLinear(state, compute_dtype=dtype)
        _set_submodule(model, name, replacement)
    del states
    gc.collect()
    model.eval()
    model.config.use_cache = False
    metadata["runtime_backend"] = runtime_backend
    metadata["activation_group_remap"] = activation_group_remapped
    if cpu_offload_layers:
        enable_eval_cpu_offload(model, device)
    else:
        model.to(device)
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


def _merge_lm_eval_payload(dst: dict[str, Any], src: dict[str, Any]) -> None:
    for key in (
        "results",
        "groups",
        "group_subtasks",
        "configs",
        "versions",
        "n-samples",
        "n-shot",
        "higher_is_better",
    ):
        value = src.get(key)
        if isinstance(value, dict):
            dst.setdefault(key, {})
            dst[key].update(value)
    for key in ("config", "date", "pretty_env_info", "git_hash"):
        if key in src:
            dst[key] = src[key]


def run_lm_eval(
    model: nn.Module,
    tokenizer: Any,
    tasks: Sequence[str],
    batch_size: int,
    limit: float | None,
    output_path: Path | None,
    *,
    device: torch.device | None = None,
    resume: bool = True,
) -> dict[str, Any]:
    from lm_eval import simple_evaluate
    from lm_eval.models.huggingface import HFLM
    from lm_eval.models.utils import Collator
    from lm_eval.models.utils_hf import pad_and_concat
    from tqdm import tqdm

    model.eval()
    torch.set_grad_enabled(False)

    class LowPeakHFLM(HFLM):
        """Same loglikelihood as HFLM, but only materializes continuation logits."""

        def _backbone_hidden(self, inps: torch.Tensor) -> torch.Tensor:
            inner = self.model.model
            if hasattr(inner, "language_model") and not hasattr(inner, "layers"):
                inner = inner.language_model
            return inner(input_ids=inps, use_cache=False).last_hidden_state

        def _output_head(self) -> nn.Module:
            if hasattr(self.model, "lm_head"):
                return self.model.lm_head
            return self.model.get_output_embeddings()

        def _loglikelihood_tokens(
            self,
            requests,
            disable_tqdm: bool = False,
            override_bs: int | None = None,
        ):
            if self.backend != "causal":
                with torch.no_grad():
                    return super()._loglikelihood_tokens(
                        requests, disable_tqdm=disable_tqdm, override_bs=override_bs
                    )
            with torch.no_grad():
                return self._loglikelihood_tokens_impl(
                    requests, disable_tqdm=disable_tqdm, override_bs=override_bs
                )

        def _loglikelihood_tokens_impl(
            self,
            requests,
            disable_tqdm: bool = False,
            override_bs: int | None = None,
        ):
            res = []

            def _collate(req):
                toks = req[1] + req[2]
                return -len(toks), tuple(toks)

            def _lookup_one_token_cont(req):
                return req[-2] + req[-1][:-1]

            re_ord = Collator(
                requests,
                sort_fn=_collate,
                group_by="contexts" if self.logits_cache else None,
                group_fn=_lookup_one_token_cont,
            )
            n_reordered_requests = len(re_ord)
            batch_size_local = (
                self.batch_size
                if self.batch_size != "auto"
                else override_bs
                if override_bs is not None
                else 0
            )
            batch_fn = (
                self._batch_scheduler
                if self.batch_size == "auto" and n_reordered_requests > 0 and not override_bs
                else None
            )
            if batch_fn is not None:
                self.batch_sizes = {}
            chunks = re_ord.get_batched(n=batch_size_local, batch_fn=batch_fn)
            pbar = tqdm(
                total=len(requests),
                disable=(disable_tqdm or (self.rank != 0)),
                desc="Running loglikelihood requests",
            )
            lm_head = self._output_head()
            for chunk in chunks:
                inps = []
                cont_toks_list = []
                inplens = []
                padding_len_inp = None
                for _, context_enc, continuation_enc in chunk:
                    total_length = len(context_enc) + len(continuation_enc)
                    if total_length > self.max_length + 1:
                        inp = torch.tensor(
                            (context_enc + continuation_enc)[-(self.max_length + 1) :][:-1],
                            dtype=torch.long,
                            device=self.device,
                        )
                    else:
                        inp = torch.tensor(
                            (context_enc + continuation_enc)[:-1],
                            dtype=torch.long,
                            device=self.device,
                        )
                    (inplen,) = inp.shape
                    padding_len_inp = max(padding_len_inp or inplen, inplen)
                    inps.append(inp)
                    cont_toks_list.append(continuation_enc)
                    inplens.append(inplen)
                batched_inps = pad_and_concat(padding_len_inp, inps, padding_side="right")
                hidden = self._backbone_hidden(batched_inps)
                del batched_inps
                for batch_index, ((request_str, ctx_tokens, _), inplen, cont_toks) in enumerate(
                    zip(chunk, inplens, cont_toks_list, strict=True)
                ):
                    contlen = len(cont_toks)
                    ctx_len = inplen + (hidden.shape[1] - padding_len_inp)
                    logits = lm_head(hidden[batch_index, ctx_len - contlen : ctx_len])
                    logits = F.log_softmax(logits, dim=-1, dtype=self.softmax_dtype).unsqueeze(0)
                    greedy_tokens = logits.argmax(dim=-1)
                    for request_str, cont_toks, logits in re_ord.get_cache(
                        req_str=request_str,
                        cxt_toks=ctx_tokens,
                        cont_toks=cont_toks,
                        logits=logits,
                    ):
                        cont_toks = torch.tensor(
                            cont_toks, dtype=torch.long, device=self.device
                        ).unsqueeze(0)
                        max_equal = (greedy_tokens[:, -cont_toks.shape[1] :] == cont_toks).all()
                        logits = torch.gather(logits, 2, cont_toks.unsqueeze(-1)).squeeze(-1)
                        answer = (float(logits.sum()), bool(max_equal))
                        res.append(answer)
                        if request_str is not None:
                            self.cache_hook.add_partial("loglikelihood", request_str, answer)
                        pbar.update(1)
                del hidden
            pbar.close()
            return re_ord.get_original(res)

    device_str = str(device) if device is not None else "cuda"
    wrapped = LowPeakHFLM(
        pretrained=model,
        tokenizer=tokenizer,
        batch_size=batch_size,
        backend="causal",
        device=device_str,
        logits_cache=True,
    )
    if device is not None:
        wrapped._device = device
    task_list = [task.strip() for task in tasks if task.strip()]
    merged: dict[str, Any] = {"results": {}}
    for task in task_list:
        shard_path = None
        if output_path is not None:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            shard_path = output_path.parent / f"{output_path.stem}_{task}.json"
            if resume and shard_path.exists():
                print(f"[eval] resume {task} from {shard_path}")
                _merge_lm_eval_payload(merged, json.loads(shard_path.read_text(encoding="utf-8")))
                continue
        print(f"[eval] task={task} batch_size={batch_size}")
        results = simple_evaluate(
            model=wrapped,
            tasks=[task],
            batch_size=batch_size,
            limit=limit,
            log_samples=False,
        )
        if shard_path is not None:
            shard_path.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
        _merge_lm_eval_payload(merged, results)
        if device is not None:
            _cuda_empty_cache(device)
        gc.collect()
    if output_path is not None:
        output_path.write_text(json.dumps(merged, indent=2, default=str), encoding="utf-8")
    return merged


def _load_model(
    model_name: str,
    device: torch.device,
    dtype: torch.dtype,
    *,
    keep_on_device: bool = True,
) -> nn.Module:
    from transformers import AutoConfig, AutoModelForCausalLM

    model_type = AutoConfig.from_pretrained(model_name).model_type
    if model_type == "qwen2_5_vl":
        from transformers import Qwen2_5_VLForConditionalGeneration

        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_name, torch_dtype=dtype)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype)
    _decoder_layers(model)
    # Large models with layer CPU-offload must stay on CPU after load; the first
    # capture pass stages only embed + layer0 onto the compute device.
    target = device if keep_on_device else torch.device("cpu")
    model.to(target).eval()
    model.config.use_cache = False
    return model


def _stage_first_layer_for_capture(model: nn.Module, device: torch.device) -> list[nn.Module]:
    """Move embed (+ optional rotary) and decoder layer0 to ``device``; return moved modules."""
    moved: list[nn.Module] = []
    layers = _decoder_layers(model)
    inner = None
    if hasattr(model, "model"):
        inner = model.model
        if hasattr(inner, "language_model") and hasattr(inner.language_model, "layers"):
            inner = inner.language_model
    elif hasattr(model, "language_model"):
        inner = model.language_model
    if inner is not None:
        for attr in ("embed_tokens", "rotary_emb"):
            if hasattr(inner, attr):
                module = getattr(inner, attr)
                if isinstance(module, nn.Module):
                    module.to(device)
                    moved.append(module)
    layers[0].to(device)
    moved.append(layers[0])
    return moved


def _resolve_ablation_mode(args: argparse.Namespace) -> dict[str, Any]:
    """Resolve named ablations into mutually isolated solver settings.

    V1 is the original same-cache local reconstruction.  V2 adds only the
    explicit F_W + lambda F_A joint code/D objective.  V3 adds only the paired
    FP-teacher target and propagated student trajectory.  V2+V3 composes both.
    The activation quantizer itself remains enabled according to
    ``activation_bits`` in every mode, so A4 inference is identical.
    """

    resolved: dict[str, Any] = {
        "rank_a": args.rank_a,
        "code_objective": args.code_objective,
        "joint_code_iters": args.joint_code_iters,
        "block_input_mode": args.block_input_mode,
        "intra_block_mode": args.intra_block_mode,
        "linear_objective": args.linear_objective,
        "activation_weight": args.activation_weight,
    }
    mode = args.ablation_mode
    if mode == "custom":
        return resolved
    resolved.update(rank_a=0, block_input_mode="quantized")
    if mode == "v1":
        resolved.update(
            code_objective="fw",
            joint_code_iters=1,
            linear_objective="local",
            activation_weight=0.0,
        )
    elif mode == "v2":
        resolved.update(code_objective="joint", linear_objective="local")
    elif mode == "v3":
        resolved.update(
            code_objective="fw",
            joint_code_iters=1,
            linear_objective="cumulative",
            activation_weight=0.0,
        )
    elif mode == "v2v3":
        resolved.update(code_objective="joint", linear_objective="cumulative")
    if mode in {"v2", "v2v3"} and resolved["activation_weight"] <= 0:
        raise ValueError(f"ablation mode {mode} requires --activation-weight > 0")
    return resolved


def quantize_command(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    dtype = _dtype_from_name(args.dtype)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    ablation = _resolve_ablation_mode(args)
    config = QuantConfig(
        bits=args.bits,
        activation_bits=args.activation_bits,
        activation_group_size=args.activation_group_size,
        d_fa_group_size=args.d_fa_group_size,
        rank=args.rank,
        rank_a=ablation["rank_a"],
        rank_a_mode=args.rank_a_mode,
        code_objective=ablation["code_objective"],
        joint_code_iters=ablation["joint_code_iters"],
        joint_rotation_mode=args.joint_rotation_mode,
        joint_rotation_fw_epsilon=args.joint_rotation_fw_epsilon,
        activation_objective=args.activation_objective,
        reducible_oracle_tokens=args.reducible_oracle_tokens,
        reducible_oracle_iters=args.reducible_oracle_iters,
        block_input_mode=ablation["block_input_mode"],
        intra_block_mode=args.intra_block_mode,
        linear_objective=ablation["linear_objective"],
        ablation_mode=args.ablation_mode,
        trajectory_damp=args.trajectory_damp,
        trajectory_max_norm_ratio=args.trajectory_max_norm_ratio,
        trajectory_scale=args.trajectory_scale,
        trajectory_diagnostics=args.trajectory_diagnostics,
        trajectory_start_layer=args.trajectory_start_layer,
        trajectory_rebase=args.trajectory_rebase,
        trajectory_holdout_fraction=args.trajectory_holdout_fraction,
        trajectory_holdout_backtracking=args.trajectory_holdout_backtracking,
        trajectory_backtrack_scales=tuple(args.trajectory_backtrack_scales),
        trajectory_spectral_floor=args.trajectory_spectral_floor,
        trajectory_min_holdout_gain=args.trajectory_min_holdout_gain,
        trajectory_min_direction_cosine=args.trajectory_min_direction_cosine,
        trajectory_quantized_gate=args.trajectory_quantized_gate,
        trajectory_module_filter=args.trajectory_module_filter,
        trajectory_oracle_diagnostics=args.trajectory_oracle_diagnostics,
        beta=args.beta,
        p=args.p,
        group_size=args.group_size,
        block_size=args.block_size,
        outer_iters=args.outer_iters,
        d_mode=args.d_mode,
        d_steps=args.d_steps,
        d_lr=args.d_lr,
        d_clip=args.d_clip,
        activation_weight=ablation["activation_weight"],
        damp=args.damp,
        svd_mode=args.svd_mode,
        svd_oversample=args.svd_oversample,
        svd_niter=args.svd_niter,
    )
    config.validate()
    print(
        "[H-SVDQuant] resolved ablation "
        f"mode={config.ablation_mode}, code={config.code_objective}, "
        f"rotation={config.joint_rotation_mode}, "
        f"activation_objective={config.activation_objective}, "
        f"trajectory={config.linear_objective}, lambda={config.activation_weight:g}, "
        f"block_input={config.block_input_mode}, intra_block={config.intra_block_mode}, "
        f"W{config.bits}A{config.activation_bits}, "
        f"trajectory_damp={config.trajectory_damp:g}, "
        f"trajectory_clip={config.trajectory_max_norm_ratio:g}",
        flush=True,
    )

    tokenizer, batches = _make_calibration_batches(
        args.model,
        args.calib_dataset,
        args.nsamples,
        args.sequence_length,
        args.calib_batch_size,
        args.seed,
    )
    model = _load_model(
        args.model,
        device,
        dtype,
        keep_on_device=not args.cpu_offload_layers,
    )
    hidden, kwargs = capture_first_layer_inputs(
        model,
        batches,
        device,
        stage_minimal=args.cpu_offload_layers,
    )
    del batches
    model_kind = _model_kind(model)
    layer_ckpt = Path(args.output) / "layer_checkpoints" if args.layer_checkpoints else None
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
        cpu_offload_layers=args.cpu_offload_layers,
        layer_checkpoint_dir=layer_ckpt,
    )
    if args.cpu_offload_layers:
        if args.eval_tasks:
            enable_eval_cpu_offload(model, device)
        _cuda_empty_cache(device)
    save_quant_checkpoint(Path(args.output), args.model, tokenizer, states, config, args, model_kind)

    if args.eval_tasks:
        results = run_lm_eval(
            model,
            tokenizer,
            args.eval_tasks.split(","),
            args.eval_batch_size,
            args.eval_limit,
            Path(args.output) / "lm_eval_results.json",
            device=device,
        )
        print(json.dumps(results.get("results", {}), indent=2, default=str))
    if args.export_dense:
        export_dense_model(model, tokenizer, Path(args.export_dense))


def eval_command(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    dtype = _dtype_from_name(args.dtype)
    model, tokenizer, _ = load_quant_checkpoint(
        Path(args.checkpoint),
        device,
        dtype,
        cpu_offload_layers=args.cpu_offload_layers,
        runtime_backend=args.runtime_backend,
        allow_activation_group_remap=args.allow_activation_group_remap,
    )
    results = run_lm_eval(
        model,
        tokenizer,
        args.tasks.split(","),
        args.batch_size,
        args.limit,
        None if not args.output else Path(args.output),
        device=device,
        resume=not args.no_resume,
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
        activation_group_size=4,
        rank=3,
        code_objective="joint",
        joint_code_iters=2,
        joint_rotation_mode="empirical",
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
    correction_state = dict(state)
    correction_groups = math.ceil(in_features / config.activation_group_size)
    correction_levels = 2 * (2 ** (config.activation_bits - 1) - 1) + 1
    correction_state["correction"] = {
        "dc_coeff": torch.zeros(correction_groups, out_features),
        "lut_coeff": torch.zeros(correction_groups, correction_levels, out_features),
        "sparse_threshold": 99.0,
        "generic_left": torch.zeros(in_features, 2),
        "generic_right": torch.zeros(2, out_features),
    }
    corrected = HSVQuantLinear(correction_state, compute_dtype=torch.float32)
    assert torch.allclose(output, corrected(x), atol=1e-6, rtol=1e-5)
    assert state["codes"].dtype == torch.int8
    assert len(state["history"]) == 4
    assert state["activation_group_size"] == 4
    assert state["code_objective"] == "joint"
    assert state["joint_rotation_mode"] == "empirical"
    assert state["joint_diagnostics"]
    assert quantized.activation_group_size == 4
    propagated_x = x + 0.05 * torch.randn_like(x)
    propagated_hessian = propagated_x.T @ propagated_x / propagated_x.shape[0]
    target_weight, target_outputs, diagnostics = cumulative_target_weight(
        layer,
        propagated_hessian,
        propagated_x,
        x,
        config,
    )
    cumulative_state = joint_quantize_linear(
        layer,
        propagated_hessian,
        propagated_x,
        config,
        target_weight=target_weight,
        target_outputs=target_outputs,
        objective_diagnostics=diagnostics,
    )
    assert cumulative_state["objective_diagnostics"]["correction_norm_ratio"] > 0
    reducible_config = replace(
        config,
        activation_bits=4,
        activation_group_size=4,
        activation_objective="reducible",
        joint_rotation_mode="empirical",
        joint_code_iters=2,
        joint_rotation_fw_epsilon=0.05,
        trajectory_backtrack_scales=(0.125, 0.25, 0.5, 1.0),
        reducible_oracle_tokens=64,
        reducible_oracle_iters=3,
    )
    reducible_state = joint_quantize_linear(layer, hessian, x, reducible_config)
    assert reducible_state["activation_objective"] == "reducible"
    assert reducible_state["reducible_source"] in {"v2_fallback", "fixed_code_refine"}
    assert "reducible_teacher_diagnostics" in reducible_state
    assert "reducible_refine_history" in reducible_state
    assert reducible_state["fw"] <= reducible_state["fw_trust_limit"] * (1.0 + 1e-6)
    reducible_module = HSVQuantLinear(reducible_state, compute_dtype=torch.float32)
    assert torch.isfinite(reducible_module(x)).all()
    print(
        json.dumps(
            {
                "status": "ok",
                "history": state["history"],
                "dense_weight_shape": list(quantized.dense_weight().shape),
                "reducible_source": reducible_state["reducible_source"],
                "reducible_accepted_updates": reducible_state["reducible_accepted_updates"],
                "reducible_oracle_gain": reducible_state["reducible_teacher_diagnostics"][
                    "reducible_oracle_gain"
                ],
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
        activation_group_size=8,
        rank=2,
        code_objective="joint",
        joint_code_iters=1,
        block_input_mode="quantized",
        linear_objective="cumulative",
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
    assert all(state["linear_objective"] == "cumulative" for state in states.values())
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
    quantize.add_argument(
        "--cpu-offload-layers",
        action="store_true",
        help="keep only the active decoder block on GPU; park finished blocks on CPU to cut peak VRAM",
    )
    quantize.add_argument(
        "--layer-checkpoints",
        action="store_true",
        help="snapshot each finished decoder block's quantized state under output/layer_checkpoints/",
    )
    quantize.add_argument("--bits", type=int, default=4)
    quantize.add_argument("--activation-bits", type=int, default=4)
    quantize.add_argument(
        "--activation-group-size",
        type=int,
        default=0,
        help="per-token activation quantization group size over input channels (0 = global max)",
    )
    quantize.add_argument(
        "--d-fa-group-size",
        type=int,
        default=-1,
        help="group size used in the D-block F_A surrogate (-1 = same as --activation-group-size, 0 = global)",
    )
    quantize.add_argument("--rank", type=int, default=8)
    quantize.add_argument(
        "--rank-a",
        type=int,
        default=0,
        help="rank units allocated to the Sigma_A-aware branch block (0 = pure F_W branch)",
    )
    quantize.add_argument(
        "--rank-a-mode",
        choices=["fixed", "gated"],
        default="fixed",
        help="fixed: rank-a everywhere; gated: per module, only when g_A(1) > g_W(rank-rank_a+1)",
    )
    quantize.add_argument(
        "--code-objective",
        choices=["fw", "joint"],
        default="fw",
        help="fw: legacy H_perp GPTQ; joint: GPTQ on H_perp+Sigma_A with the shrunken pseudo-target",
    )
    quantize.add_argument(
        "--joint-code-iters",
        type=int,
        default=1,
        help="alternating joint code / exact fixed-code rank-r branch updates (joint mode only)",
    )
    quantize.add_argument(
        "--joint-rotation-mode",
        choices=["none", "empirical"],
        default="none",
        help=(
            "none: legacy H-metric fixed-code branch update; empirical: rotate the rank-r branch "
            "against the exact cached W4A4 output before requantizing residual codes"
        ),
    )
    quantize.add_argument(
        "--joint-rotation-fw-epsilon",
        type=float,
        default=0.0,
        help="admit a rotated/requantized candidate only if F_W <= (1+epsilon) F_W^(0)",
    )
    quantize.add_argument(
        "--activation-objective",
        choices=["full", "reducible"],
        default="full",
        help="full: V2 total activation penalty; reducible: cross-fitted non-uniform projection teacher",
    )
    quantize.add_argument("--reducible-oracle-tokens", type=int, default=512)
    quantize.add_argument("--reducible-oracle-iters", type=int, default=5)
    quantize.add_argument(
        "--block-input-mode",
        choices=["quantized", "reference"],
        default="quantized",
        help="quantized: propagate each quantized block output as the next block cache; reference: use FP block inputs",
    )
    quantize.add_argument(
        "--intra-block-mode",
        choices=["sequential", "fp_independent"],
        default="sequential",
        help="sequential: quantize/replace within block in qkv->o, gate/up->down order; "
        "fp_independent: calibrate every linear on one FP block forward, then replace all",
    )
    quantize.add_argument(
        "--linear-objective",
        choices=["local", "cumulative"],
        default="local",
        help="local: reconstruct Xhat W; cumulative: reconstruct paired FP target X W from propagated Xhat",
    )
    quantize.add_argument(
        "--ablation-mode",
        choices=["custom", "v1", "v2", "v3", "v2v3"],
        default="custom",
        help=(
            "named isolated preset: v1=local baseline; v2=local F_W+lambda F_A; "
            "v3=teacher-student correction only; v2v3=joint objective plus trajectory correction"
        ),
    )
    quantize.add_argument(
        "--trajectory-damp",
        type=float,
        default=0.1,
        help="ridge coefficient for the paired-cache trajectory projection",
    )
    quantize.add_argument(
        "--trajectory-max-norm-ratio",
        type=float,
        default=0.25,
        help="trust-region cap ||A*-W||_F / ||W||_F for cumulative correction",
    )
    quantize.add_argument(
        "--trajectory-scale",
        type=float,
        default=1.0,
        help="additional scale in (0,1] after the paired-cache line search",
    )
    quantize.add_argument(
        "--trajectory-diagnostics",
        action="store_true",
        help="record paired FP-teacher/student hidden-state gap after every decoder block",
    )
    quantize.add_argument(
        "--trajectory-start-layer",
        type=int,
        default=0,
        help="zero-based first decoder layer that enables cumulative trajectory correction",
    )
    quantize.add_argument(
        "--trajectory-rebase",
        action="store_true",
        help="at the start layer, seed the optimization teacher with the current student hidden state",
    )
    quantize.add_argument(
        "--trajectory-holdout-fraction",
        type=float,
        default=0.0,
        help="fraction of paired cache rows held out for trajectory correction validation",
    )
    quantize.add_argument(
        "--trajectory-holdout-backtracking",
        action="store_true",
        help="choose the trajectory correction scale on the holdout split instead of taking the full line-search step",
    )
    quantize.add_argument(
        "--trajectory-backtrack-scales",
        type=float,
        nargs="+",
        default=[0.0, 0.125, 0.25, 0.5, 1.0],
        help="candidate correction scales for holdout backtracking",
    )
    quantize.add_argument(
        "--trajectory-spectral-floor",
        type=float,
        default=0.0,
        help="drop trajectory correction components below this relative Hessian eigenvalue floor",
    )
    quantize.add_argument(
        "--trajectory-min-holdout-gain",
        type=float,
        default=0.0,
        help="minimum held-out teacher-MSE gain required to accept a trajectory correction",
    )
    quantize.add_argument(
        "--trajectory-min-direction-cosine",
        type=float,
        default=-1.0,
        help="minimum split-half correction cosine required to accept a trajectory correction",
    )
    quantize.add_argument(
        "--trajectory-quantized-gate",
        action="store_true",
        help="quantize both local and trajectory targets, then keep trajectory only if its teacher MSE improves",
    )
    quantize.add_argument(
        "--trajectory-module-filter",
        choices=["all", "attention", "mlp", "down_proj"],
        default="all",
        help="limit cumulative trajectory correction to a subset of linear module types",
    )
    quantize.add_argument(
        "--trajectory-oracle-diagnostics",
        action="store_true",
        help="compute expensive least-squares projection lower-bound diagnostics on paired caches",
    )
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
    evaluate.add_argument("--batch-size", type=int, default=1)
    evaluate.add_argument("--limit", type=float, default=None)
    evaluate.add_argument("--output", default="")
    evaluate.add_argument(
        "--runtime-backend",
        choices=["eager", "nunchaku"],
        default="eager",
        help="eager reference path or packed Nunchaku W4A4 CUDA kernels",
    )
    evaluate.add_argument(
        "--allow-activation-group-remap",
        action="store_true",
        help="explicitly run non-g64 checkpoints with Nunchaku's A-group 64 quantizer",
    )
    evaluate.add_argument(
        "--cpu-offload-layers",
        action="store_true",
        default=False,
        help="page decoder blocks through GPU during eval (lower peak, slower)",
    )
    evaluate.add_argument(
        "--no-cpu-offload-layers",
        action="store_false",
        dest="cpu_offload_layers",
    )
    evaluate.add_argument(
        "--no-resume",
        action="store_true",
        help="recompute per-task shards even if output_task.json already exists",
    )
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
