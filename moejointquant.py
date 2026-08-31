#!/usr/bin/env python3
"""Reference implementation and toy verification for joint MoE PTQ.

The first target is correctness, not scale.  For a routed MoE down projection,
the weighted expert activations are concatenated into one augmented feature
matrix Phi.  The exact block reconstruction problem is then

    min_Q ||Phi (W - Q)||_F^2,

where W stacks all expert down-projection weights.  The script implements:

* RTN, expert-local GPTQ, affinity-weighted expert GPTQ, and full-joint GPTQ;
* finite discrete coordinate descent on the full joint objective, initialized
  from the affinity GPTQ solution, with a monotonic objective guarantee;
* exhaustive joint and independent optima for very small verification cases;
* a synthetic routed MoE block and calibration-set selection experiments.

This is deliberately a full-Hessian oracle.  Grouping, slicing, low-rank
approximations, and production model integration belong to the next stage and
should only be attempted after this verifier demonstrates a real joint gain.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

import torch
import torch.nn.functional as F


Tensor = torch.Tensor


@dataclass(frozen=True)
class QuantConfig:
    bits: int = 2
    damp: float = 1e-3
    act_order: bool = True
    joint_sweeps: int = 12
    improvement_tol: float = 1e-10

    def validate(self) -> None:
        if not 2 <= self.bits <= 8:
            raise ValueError("bits must be in [2, 8]")
        if self.damp < 0:
            raise ValueError("damp must be non-negative")
        if self.joint_sweeps < 1:
            raise ValueError("joint_sweeps must be positive")


@dataclass(frozen=True)
class ToyConfig:
    num_experts: int = 5
    top_k: int = 2
    d_model: int = 12
    d_ff: int = 7
    num_classes: int = 7
    input_noise: float = 0.55

    def validate(self) -> None:
        if self.num_experts < 2:
            raise ValueError("num_experts must be at least two")
        if not 1 <= self.top_k <= self.num_experts:
            raise ValueError("top_k must lie in [1, num_experts]")
        if self.d_model < self.num_experts:
            raise ValueError("d_model must be at least num_experts")
        if self.d_ff < 1:
            raise ValueError("d_ff must be positive")


@dataclass
class QuantizedWeights:
    name: str
    weights: Tensor
    calibration_objective: float
    history: list[float]


@dataclass
class ToyMoE:
    router: Tensor
    up: Tensor
    gate: Tensor
    down: Tensor
    task_pairs: list[tuple[int, int]]
    task_means: Tensor
    next_router: Tensor
    suffix_one: Tensor
    suffix_two: Tensor

    @property
    def num_experts(self) -> int:
        return self.router.shape[0]

    @property
    def d_ff(self) -> int:
        return self.down.shape[1]

    @property
    def d_model(self) -> int:
        return self.router.shape[1]


def _sym(matrix: Tensor) -> Tensor:
    return (matrix + matrix.T) * 0.5


def _qmax(bits: int) -> int:
    return 2 ** (bits - 1) - 1


def _fixed_grid_quantize(values: Tensor, scales: Tensor, bits: int) -> Tensor:
    qmax = _qmax(bits)
    safe = scales.clamp_min(torch.finfo(values.dtype).eps)
    codes = torch.round(values / safe).clamp(-qmax, qmax)
    return codes * safe


def expert_scales(weights: Tensor, bits: int) -> Tensor:
    """Return fixed per-expert, per-output symmetric scales.

    Args:
        weights: [experts, d_ff, d_out].
    """

    qmax = _qmax(bits)
    maxima = weights.abs().amax(dim=1, keepdim=True)
    fallback = torch.full_like(maxima, torch.finfo(weights.dtype).eps)
    scales = torch.where(maxima > 0, maxima / qmax, fallback)
    return scales.expand_as(weights).clone()


def augmented_features(expert_activations: Tensor, gates: Tensor) -> Tensor:
    """Build Phi_n = [g_n1 a_n1; ...; g_nE a_nE]."""

    if expert_activations.ndim != 3 or gates.ndim != 2:
        raise ValueError("expected activations [N,E,D] and gates [N,E]")
    if expert_activations.shape[:2] != gates.shape:
        raise ValueError("activation and gate shapes do not agree")
    return (expert_activations * gates[..., None]).reshape(expert_activations.shape[0], -1)


def joint_hessian(phi: Tensor) -> Tensor:
    return _sym(phi.T @ phi / float(max(1, phi.shape[0])))


def quadratic_objective(weight: Tensor, quantized: Tensor, hessian: Tensor) -> float:
    error = weight - quantized
    return float(((hessian @ error) * error).sum().item())


def direct_reconstruction_objective(phi: Tensor, weight: Tensor, quantized: Tensor) -> float:
    error = phi @ (weight - quantized)
    return float(error.square().sum().div(max(1, phi.shape[0])).item())


def _regularized_hessian(hessian: Tensor, damp: float) -> Tensor:
    hessian = _sym(hessian.double())
    diagonal = hessian.diagonal().clamp_min(0)
    reference = diagonal[diagonal > 0].mean() if bool((diagonal > 0).any()) else hessian.new_tensor(1.0)
    return hessian + torch.eye(hessian.shape[0], dtype=hessian.dtype) * (damp * reference + 1e-12)


def gptq_quantize(
    weight: Tensor,
    hessian: Tensor,
    scales: Tensor,
    config: QuantConfig,
) -> Tensor:
    """Quantize rows of W using a dense GPTQ/OBS reference update."""

    if weight.shape != scales.shape:
        raise ValueError("weight and scales must have identical shape")
    if hessian.shape != (weight.shape[0], weight.shape[0]):
        raise ValueError("hessian shape is incompatible with weight")

    order = torch.arange(weight.shape[0])
    if config.act_order:
        order = torch.argsort(hessian.diagonal(), descending=True)
    inverse_order = torch.argsort(order)
    work = weight[order].double().clone()
    local_scales = scales[order].double()
    metric = _regularized_hessian(hessian[order][:, order], config.damp)

    eigenvalues, eigenvectors = torch.linalg.eigh(metric)
    floor = eigenvalues.max().clamp_min(1.0) * 1e-10
    metric = (eigenvectors * eigenvalues.clamp_min(floor)) @ eigenvectors.T
    inverse = torch.linalg.inv(metric)
    upper = torch.linalg.cholesky(_sym(inverse), upper=True)
    quantized = torch.empty_like(work)

    for index in range(work.shape[0]):
        quantized[index] = _fixed_grid_quantize(work[index], local_scales[index], config.bits)
        pivot = upper[index, index].clamp_min(1e-12)
        error = (work[index] - quantized[index]) / pivot
        work[index:] -= upper[index, index:, None] * error[None, :]
    return quantized[inverse_order].to(weight.dtype)


def local_expert_hessians(expert_activations: Tensor, gates: Tensor, affinity: bool) -> list[Tensor]:
    metrics: list[Tensor] = []
    for expert in range(expert_activations.shape[1]):
        active = gates[:, expert] > 0
        rows = expert_activations[active, expert]
        if affinity:
            rows = rows * gates[active, expert, None]
        if rows.shape[0] == 0:
            metrics.append(torch.eye(expert_activations.shape[2], dtype=expert_activations.dtype))
        else:
            metrics.append(_sym(rows.T @ rows / float(rows.shape[0])))
    return metrics


def independent_gptq(
    weights: Tensor,
    expert_activations: Tensor,
    gates: Tensor,
    scales: Tensor,
    config: QuantConfig,
    affinity: bool,
) -> Tensor:
    metrics = local_expert_hessians(expert_activations, gates, affinity=affinity)
    pieces = [
        gptq_quantize(weights[e], metrics[e], scales[e], config)
        for e in range(weights.shape[0])
    ]
    return torch.stack(pieces)


def joint_coordinate_descent(
    weight: Tensor,
    hessian: Tensor,
    scales: Tensor,
    initial: Tensor,
    config: QuantConfig,
) -> tuple[Tensor, list[float]]:
    """Coordinate-wise exact minimization on the finite joint quantization grid.

    Every coordinate update minimizes the exact full quadratic while all other
    codes are fixed.  Therefore the history is monotone and initialization from
    an independent solution certifies a no-worse calibration objective.
    """

    qmax = _qmax(config.bits)
    levels = torch.arange(-qmax, qmax + 1, dtype=weight.dtype)
    quantized = initial.clone()
    error = weight - quantized
    order = torch.argsort(hessian.diagonal(), descending=True)
    history = [quadratic_objective(weight, quantized, hessian)]

    for _sweep in range(config.joint_sweeps):
        previous = history[-1]
        for coordinate in order.tolist():
            diagonal = hessian[coordinate, coordinate]
            if float(diagonal.abs()) <= 1e-18:
                continue
            cross = hessian[coordinate] @ error - diagonal * error[coordinate]
            candidates = levels[:, None] * scales[coordinate][None, :]
            candidate_error = weight[coordinate][None, :] - candidates
            costs = diagonal * candidate_error.square() + 2.0 * candidate_error * cross[None, :]
            best = torch.argmin(costs, dim=0)
            quantized[coordinate] = candidates[best, torch.arange(weight.shape[1])]
            error[coordinate] = weight[coordinate] - quantized[coordinate]
        current = quadratic_objective(weight, quantized, hessian)
        history.append(current)
        if current > previous + 1e-8 * max(1.0, abs(previous)):
            raise RuntimeError("joint coordinate descent violated monotonicity")
        if previous - current <= config.improvement_tol * max(1.0, abs(previous)):
            break
    return quantized, history


def _enumerated_codes(length: int, bits: int, dtype: torch.dtype) -> Tensor:
    qmax = _qmax(bits)
    values = list(itertools.product(range(-qmax, qmax + 1), repeat=length))
    return torch.tensor(values, dtype=dtype)


def exact_joint_quantize(weight: Tensor, hessian: Tensor, scales: Tensor, bits: int) -> Tensor:
    """Exhaustively solve the fixed-grid joint objective, for tiny cases only."""

    if weight.shape[0] > 10:
        raise ValueError("exact enumeration is restricted to at most ten coordinates")
    codes = _enumerated_codes(weight.shape[0], bits, weight.dtype)
    result = torch.empty_like(weight)
    for output in range(weight.shape[1]):
        candidates = codes * scales[:, output][None, :]
        errors = weight[:, output][None, :] - candidates
        costs = torch.einsum("bi,ij,bj->b", errors, hessian, errors)
        result[:, output] = candidates[int(torch.argmin(costs))]
    return result


def exact_independent_quantize(
    weight: Tensor,
    hessian: Tensor,
    scales: Tensor,
    expert_slices: Sequence[slice],
    bits: int,
) -> Tensor:
    """Solve each expert diagonal block exactly, ignoring cross blocks."""

    result = torch.empty_like(weight)
    for expert_slice in expert_slices:
        indices = torch.arange(weight.shape[0])[expert_slice]
        block = hessian[indices][:, indices]
        result[expert_slice] = exact_joint_quantize(
            weight[expert_slice], block, scales[expert_slice], bits
        )
    return result


def quantize_all_methods(
    weights: Tensor,
    expert_activations: Tensor,
    gates: Tensor,
    config: QuantConfig,
    fixed_scales: Tensor | None = None,
) -> dict[str, QuantizedWeights]:
    config.validate()
    phi = augmented_features(expert_activations, gates)
    hessian = joint_hessian(phi)
    flat_weight = weights.reshape(-1, weights.shape[-1])
    if fixed_scales is None:
        fixed_scales = expert_scales(weights, config.bits)
    elif fixed_scales.shape != weights.shape:
        raise ValueError("fixed_scales must have the same shape as weights")
    scales = fixed_scales.reshape_as(flat_weight)

    rtn = _fixed_grid_quantize(flat_weight, scales, config.bits)
    independent = independent_gptq(
        weights, expert_activations, gates, scales.reshape_as(weights), config, affinity=False
    ).reshape_as(flat_weight)
    affinity = independent_gptq(
        weights, expert_activations, gates, scales.reshape_as(weights), config, affinity=True
    ).reshape_as(flat_weight)
    joint_gptq = gptq_quantize(flat_weight, hessian, scales, config)
    joint_cd, history = joint_coordinate_descent(
        flat_weight, hessian, scales, affinity, config
    )

    candidates = {
        "rtn": (rtn, []),
        "independent_gptq": (independent, []),
        "affinity_gptq": (affinity, []),
        "joint_gptq": (joint_gptq, []),
        "joint_cd": (joint_cd, history),
    }
    return {
        name: QuantizedWeights(
            name=name,
            weights=value.reshape_as(weights),
            calibration_objective=quadratic_objective(flat_weight, value, hessian),
            history=objective_history,
        )
        for name, (value, objective_history) in candidates.items()
    }


def make_toy_moe(config: ToyConfig, seed: int) -> ToyMoE:
    config.validate()
    generator = torch.Generator().manual_seed(seed)
    dtype = torch.float64
    router = torch.zeros(config.num_experts, config.d_model, dtype=dtype)
    router[:, : config.num_experts] = torch.eye(config.num_experts, dtype=dtype)

    pairs = list(itertools.combinations(range(config.num_experts), 2))
    means = torch.zeros(len(pairs), config.d_model, dtype=dtype)
    means[:, : config.num_experts] = -1.25
    for task, pair in enumerate(pairs):
        means[task, pair[0]] = 2.5
        means[task, pair[1]] = 2.5
    if config.d_model > config.num_experts:
        tail = torch.randn(
            len(pairs), config.d_model - config.num_experts, generator=generator, dtype=dtype
        )
        means[:, config.num_experts :] = tail * 0.8

    scale = 1.0 / math.sqrt(config.d_model)
    up = torch.randn(
        config.num_experts, config.d_model, config.d_ff, generator=generator, dtype=dtype
    ) * scale
    gate = torch.randn(
        config.num_experts, config.d_model, config.d_ff, generator=generator, dtype=dtype
    ) * scale
    down = torch.randn(
        config.num_experts, config.d_ff, config.d_model, generator=generator, dtype=dtype
    ) / math.sqrt(config.d_ff)

    # A shared component makes co-routed expert errors correlated without making
    # the experts identical.
    shared = torch.randn(config.d_ff, config.d_model, generator=generator, dtype=dtype)
    shared /= math.sqrt(config.d_ff)
    down = 0.72 * down + 0.28 * shared[None, :, :]

    next_router = torch.randn(
        config.num_experts, config.d_model, generator=generator, dtype=dtype
    ) / math.sqrt(config.d_model)
    suffix_one = torch.randn(
        config.d_model, config.d_model * 2, generator=generator, dtype=dtype
    ) / math.sqrt(config.d_model)
    suffix_two = torch.randn(
        config.d_model * 2, config.num_classes, generator=generator, dtype=dtype
    ) / math.sqrt(config.d_model * 2)
    return ToyMoE(router, up, gate, down, pairs, means, next_router, suffix_one, suffix_two)


def skewed_task_probabilities(num_tasks: int) -> Tensor:
    ranks = torch.arange(num_tasks, dtype=torch.float64)
    probabilities = torch.exp(-0.45 * ranks)
    return probabilities / probabilities.sum()


def sample_toy_inputs(
    model: ToyMoE,
    count: int,
    probabilities: Tensor,
    noise: float,
    seed: int,
) -> tuple[Tensor, Tensor]:
    generator = torch.Generator().manual_seed(seed)
    labels = torch.multinomial(probabilities, count, replacement=True, generator=generator)
    inputs = model.task_means[labels].clone()
    inputs += torch.randn(inputs.shape, generator=generator, dtype=inputs.dtype) * noise
    return inputs, labels


def toy_forward_features(model: ToyMoE, inputs: Tensor, top_k: int) -> tuple[Tensor, Tensor, Tensor]:
    logits = inputs @ model.router.T
    top_values, top_indices = torch.topk(logits, k=top_k, dim=-1)
    top_weights = torch.softmax(top_values, dim=-1)
    gates = torch.zeros_like(logits)
    gates.scatter_(1, top_indices, top_weights)
    up = torch.einsum("nd,edf->nef", inputs, model.up)
    gate = torch.einsum("nd,edf->nef", inputs, model.gate)
    activations = F.silu(gate) * up
    return activations, gates, top_indices


def toy_block_output(activations: Tensor, gates: Tensor, weights: Tensor) -> Tensor:
    return augmented_features(activations, gates) @ weights.reshape(-1, weights.shape[-1])


def route_signatures(top_indices: Tensor) -> Tensor:
    sorted_indices = torch.sort(top_indices, dim=-1).values
    base = int(sorted_indices.max().item()) + 1
    signature = torch.zeros(sorted_indices.shape[0], dtype=torch.long)
    for column in range(sorted_indices.shape[1]):
        signature = signature * base + sorted_indices[:, column]
    return signature


def _balanced_indices(labels: Tensor, budget: int, seed: int) -> Tensor:
    generator = torch.Generator().manual_seed(seed)
    groups = torch.unique(labels).tolist()
    buckets: list[Tensor] = []
    per_group = max(1, budget // max(1, len(groups)))
    used = torch.zeros(labels.shape[0], dtype=torch.bool)
    for group in groups:
        candidates = torch.nonzero(labels == group).flatten()
        candidates = candidates[torch.randperm(candidates.numel(), generator=generator)]
        chosen = candidates[:per_group]
        buckets.append(chosen)
        used[chosen] = True
    selected = torch.cat(buckets) if buckets else torch.empty(0, dtype=torch.long)
    if selected.numel() < budget:
        remaining = torch.nonzero(~used).flatten()
        remaining = remaining[torch.randperm(remaining.numel(), generator=generator)]
        selected = torch.cat((selected, remaining[: budget - selected.numel()]))
    return selected[:budget]


def _feature_pivot_indices(features: Tensor, budget: int, seed: int) -> Tensor:
    """A fast pivoted-feature approximation to D-optimal row selection."""

    generator = torch.Generator().manual_seed(seed)
    rows = features.double().clone()
    norms = rows.norm(dim=1)
    cap = torch.quantile(norms, 0.95).clamp_min(1e-12)
    rows *= (cap / norms.clamp_min(cap))[:, None]
    residual = rows.clone()
    selected: list[int] = []
    available = torch.ones(rows.shape[0], dtype=torch.bool)
    rank_budget = min(budget, rows.shape[1])
    for _ in range(rank_budget):
        scores = residual.square().sum(dim=1)
        scores[~available] = -1
        index = int(torch.argmax(scores))
        if float(scores[index]) <= 1e-14:
            break
        selected.append(index)
        available[index] = False
        direction = residual[index] / scores[index].sqrt()
        residual -= (residual @ direction)[:, None] * direction[None, :]
    if len(selected) < budget:
        remaining = torch.nonzero(available).flatten()
        remaining = remaining[torch.randperm(remaining.numel(), generator=generator)]
        selected.extend(remaining[: budget - len(selected)].tolist())
    return torch.tensor(selected[:budget], dtype=torch.long)


def calibration_selectors(
    labels: Tensor,
    top_indices: Tensor,
    phi: Tensor,
    budget: int,
    seed: int,
) -> dict[str, Tensor]:
    generator = torch.Generator().manual_seed(seed)
    return {
        "random": torch.randperm(labels.numel(), generator=generator)[:budget],
        "task_balanced_oracle": _balanced_indices(labels, budget, seed + 11),
        "route_balanced": _balanced_indices(route_signatures(top_indices), budget, seed + 23),
        "feature_pivot": _feature_pivot_indices(phi, budget, seed + 37),
    }


def _topk_overlap(left: Tensor, right: Tensor) -> float:
    matches = []
    for row in range(left.shape[0]):
        lset = set(left[row].tolist())
        rset = set(right[row].tolist())
        matches.append(len(lset & rset) / max(1, len(lset)))
    return float(sum(matches) / max(1, len(matches)))


def evaluate_quantized(
    model: ToyMoE,
    inputs: Tensor,
    weights: Tensor,
    top_k: int,
) -> dict[str, float]:
    activations, gates, _ = toy_forward_features(model, inputs, top_k)
    reference = toy_block_output(activations, gates, model.down)
    prediction = toy_block_output(activations, gates, weights)
    error = prediction - reference
    block_nmse = error.square().mean() / reference.square().mean().clamp_min(1e-12)

    reference_state = inputs + reference
    quantized_state = inputs + prediction
    reference_suffix = torch.tanh(reference_state @ model.suffix_one) @ model.suffix_two
    quantized_suffix = torch.tanh(quantized_state @ model.suffix_one) @ model.suffix_two
    suffix_error = quantized_suffix - reference_suffix
    suffix_nmse = suffix_error.square().mean() / reference_suffix.square().mean().clamp_min(1e-12)
    agreement = (reference_suffix.argmax(dim=-1) == quantized_suffix.argmax(dim=-1)).double().mean()

    reference_route = torch.topk(reference_state @ model.next_router.T, top_k, dim=-1).indices
    quantized_route = torch.topk(quantized_state @ model.next_router.T, top_k, dim=-1).indices
    return {
        "block_nmse": float(block_nmse),
        "suffix_nmse": float(suffix_nmse),
        "teacher_agreement": float(agreement),
        "next_route_overlap": _topk_overlap(reference_route, quantized_route),
    }


def exact_oracle_demo(seed: int, bits: int = 2) -> dict[str, float | bool | list[float]]:
    toy_config = ToyConfig(num_experts=3, top_k=2, d_model=7, d_ff=2, num_classes=4)
    model = make_toy_moe(toy_config, seed)
    inputs, _ = sample_toy_inputs(
        model,
        count=96,
        probabilities=torch.ones(len(model.task_pairs), dtype=torch.float64) / len(model.task_pairs),
        noise=toy_config.input_noise,
        seed=seed + 1,
    )
    activations, gates, _ = toy_forward_features(model, inputs, toy_config.top_k)
    phi = augmented_features(activations, gates)
    hessian = joint_hessian(phi)
    weight = model.down.reshape(-1, model.d_model)
    scales = expert_scales(model.down, bits).reshape_as(weight)
    slices = [slice(e * toy_config.d_ff, (e + 1) * toy_config.d_ff) for e in range(toy_config.num_experts)]

    exact_independent = exact_independent_quantize(weight, hessian, scales, slices, bits)
    exact_joint = exact_joint_quantize(weight, hessian, scales, bits)
    cd_config = QuantConfig(bits=bits, joint_sweeps=30, improvement_tol=0.0)
    coordinate, history = joint_coordinate_descent(
        weight, hessian, scales, exact_independent, cd_config
    )
    independent_objective = quadratic_objective(weight, exact_independent, hessian)
    joint_objective = quadratic_objective(weight, exact_joint, hessian)
    coordinate_objective = quadratic_objective(weight, coordinate, hessian)
    return {
        "exact_independent_objective": independent_objective,
        "exact_joint_objective": joint_objective,
        "coordinate_objective": coordinate_objective,
        "exact_joint_gain": (independent_objective - joint_objective) / max(independent_objective, 1e-12),
        "coordinate_gain": (independent_objective - coordinate_objective) / max(independent_objective, 1e-12),
        "dominance_pass": joint_objective <= independent_objective + 1e-10,
        "coordinate_no_worse_pass": coordinate_objective <= independent_objective + 1e-10,
        "coordinate_history": history,
    }


def self_test() -> None:
    torch.set_default_dtype(torch.float64)
    config = ToyConfig(num_experts=3, top_k=2, d_model=7, d_ff=2, num_classes=4)
    model = make_toy_moe(config, seed=0)
    inputs, _ = sample_toy_inputs(
        model,
        64,
        torch.ones(len(model.task_pairs), dtype=torch.float64) / len(model.task_pairs),
        config.input_noise,
        1,
    )
    activations, gates, _ = toy_forward_features(model, inputs, config.top_k)
    phi = augmented_features(activations, gates)
    hessian = joint_hessian(phi)
    weight = model.down.reshape(-1, model.d_model)
    scales = expert_scales(model.down, 2).reshape_as(weight)
    initial = _fixed_grid_quantize(weight, scales, 2)
    direct = direct_reconstruction_objective(phi, weight, initial)
    quadratic = quadratic_objective(weight, initial, hessian)
    if not math.isclose(direct, quadratic, rel_tol=1e-9, abs_tol=1e-10):
        raise AssertionError(f"objective identity failed: {direct} vs {quadratic}")
    result = exact_oracle_demo(seed=3)
    if not bool(result["dominance_pass"]):
        raise AssertionError("exact joint optimum did not dominate independent optimum")
    if not bool(result["coordinate_no_worse_pass"]):
        raise AssertionError("coordinate descent did not preserve its baseline guarantee")
    history = result["coordinate_history"]
    if any(float(history[i + 1]) > float(history[i]) + 1e-10 for i in range(len(history) - 1)):
        raise AssertionError("coordinate history is not monotone")
    quant_config = QuantConfig(bits=2, joint_sweeps=2)
    default_grid = quantize_all_methods(model.down, activations, gates, quant_config)
    fixed_grid = quantize_all_methods(
        model.down,
        activations,
        gates,
        quant_config,
        fixed_scales=expert_scales(model.down, quant_config.bits),
    )
    for method in default_grid:
        if not torch.equal(default_grid[method].weights, fixed_grid[method].weights):
            raise AssertionError(f"explicit fixed scales changed {method}")
    print("moejointquant self-test: PASS")


def _mean(rows: Iterable[dict[str, object]], key: str) -> float:
    values = [float(row[key]) for row in rows]
    return sum(values) / max(1, len(values))


def _write_report(output: Path, rows: list[dict[str, object]], oracle: list[dict[str, object]]) -> None:
    methods = ["rtn", "independent_gptq", "affinity_gptq", "joint_gptq", "joint_cd"]
    selectors = ["random", "task_balanced_oracle", "route_balanced", "feature_pivot"]
    lines = [
        "# MoEJointQuant Toy Verification",
        "",
        "The experiment uses the exact dense co-routing Hessian. No grouping, slicing, or low-rank acceleration is used.",
        "",
        "## Exact tiny oracle",
        "",
        "| metric | mean |",
        "|---|---:|",
        f"| exact joint gain over exact independent | {_mean(oracle, 'exact_joint_gain'):.4f} |",
        f"| coordinate gain over exact independent | {_mean(oracle, 'coordinate_gain'):.4f} |",
        f"| dominance pass rate | {_mean(oracle, 'dominance_pass'):.3f} |",
        f"| coordinate guarantee pass rate | {_mean(oracle, 'coordinate_no_worse_pass'):.3f} |",
        "",
        "## Quantizer comparison with random calibration (skewed IID test)",
        "",
        "| method | calibration objective | IID block NMSE | suffix NMSE | teacher agreement | next-route overlap |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for method in methods:
        subset = [row for row in rows if row["selector"] == "random" and row["method"] == method]
        lines.append(
            f"| {method} | {_mean(subset, 'calibration_objective'):.6f} | "
            f"{_mean(subset, 'id_block_nmse'):.6f} | {_mean(subset, 'id_suffix_nmse'):.6f} | "
            f"{_mean(subset, 'id_teacher_agreement'):.4f} | {_mean(subset, 'id_next_route_overlap'):.4f} |"
        )
    lines.extend([
        "",
        "## Quantizer comparison with route-balanced calibration (balanced test)",
        "",
        "| method | calibration objective | balanced block NMSE | suffix NMSE | teacher agreement | next-route overlap |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for method in methods:
        subset = [row for row in rows if row["selector"] == "route_balanced" and row["method"] == method]
        lines.append(
            f"| {method} | {_mean(subset, 'calibration_objective'):.6f} | "
            f"{_mean(subset, 'balanced_block_nmse'):.6f} | {_mean(subset, 'balanced_suffix_nmse'):.6f} | "
            f"{_mean(subset, 'balanced_teacher_agreement'):.4f} | {_mean(subset, 'balanced_next_route_overlap'):.4f} |"
        )
    lines.extend([
        "",
        "## Calibration selection for joint coordinate descent",
        "",
        "| selector | balanced-test block NMSE | suffix NMSE | teacher agreement | next-route overlap |",
        "|---|---:|---:|---:|---:|",
    ])
    for selector in selectors:
        subset = [row for row in rows if row["selector"] == selector and row["method"] == "joint_cd"]
        lines.append(
            f"| {selector} | {_mean(subset, 'balanced_block_nmse'):.6f} | "
            f"{_mean(subset, 'balanced_suffix_nmse'):.6f} | {_mean(subset, 'balanced_teacher_agreement'):.4f} | "
            f"{_mean(subset, 'balanced_next_route_overlap'):.4f} |"
        )
    lines.extend([
        "",
        "## Interpretation rule",
        "",
        "The joint framework passes stage one only if the exact oracle dominates independently optimized experts and the monotone joint solver transfers a meaningful fraction of that gain to held-out block and suffix metrics. Calibration selection is secondary and should not rescue a failed joint model.",
        "",
    ])
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")


def _write_plots(output: Path, rows: list[dict[str, object]]) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    methods = ["rtn", "independent_gptq", "affinity_gptq", "joint_gptq", "joint_cd"]
    random_means = [
        _mean([r for r in rows if r["selector"] == "random" and r["method"] == method], "balanced_block_nmse")
        for method in methods
    ]
    fig, axis = plt.subplots(figsize=(8.2, 4.5))
    axis.bar(range(len(methods)), random_means, color=["#777777", "#C45A35", "#D8993A", "#39949E", "#087E8B"])
    axis.set_xticks(range(len(methods)), methods, rotation=18, ha="right")
    axis.set_ylabel("Balanced-test block NMSE")
    axis.set_title("Full-joint verification with random calibration")
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output / "method_comparison.png", dpi=180)
    plt.close(fig)

    selectors = ["random", "task_balanced_oracle", "route_balanced", "feature_pivot"]
    selector_means = [
        _mean([r for r in rows if r["selector"] == selector and r["method"] == "joint_cd"], "balanced_block_nmse")
        for selector in selectors
    ]
    fig, axis = plt.subplots(figsize=(8.2, 4.5))
    axis.bar(range(len(selectors)), selector_means, color=["#777777", "#4F7D5B", "#087E8B", "#C45A35"])
    axis.set_xticks(range(len(selectors)), selectors, rotation=18, ha="right")
    axis.set_ylabel("Balanced-test block NMSE")
    axis.set_title("Calibration selection after fixing the quantizer")
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output / "calibration_comparison.png", dpi=180)
    plt.close(fig)


def run_toy_experiment(args: argparse.Namespace) -> dict[str, object]:
    started = time.perf_counter()
    torch.set_default_dtype(torch.float64)
    random.seed(args.seed)
    quant_config = QuantConfig(
        bits=args.bits,
        damp=args.damp,
        act_order=not args.no_act_order,
        joint_sweeps=args.joint_sweeps,
    )
    toy_config = ToyConfig(
        num_experts=args.num_experts,
        top_k=args.top_k,
        d_model=args.d_model,
        d_ff=args.d_ff,
        num_classes=args.num_classes,
        input_noise=args.input_noise,
    )
    quant_config.validate()
    toy_config.validate()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    oracle_rows: list[dict[str, object]] = []
    for seed in args.seeds:
        print(f"[seed {seed}] building toy MoE", flush=True)
        model = make_toy_moe(toy_config, seed)
        probabilities = skewed_task_probabilities(len(model.task_pairs))
        balanced_probabilities = torch.ones_like(probabilities) / probabilities.numel()
        pool_x, pool_labels = sample_toy_inputs(
            model, args.pool_size, probabilities, toy_config.input_noise, seed * 101 + 1
        )
        id_test_x, _ = sample_toy_inputs(
            model, args.test_size, probabilities, toy_config.input_noise, seed * 101 + 2
        )
        balanced_test_x, _ = sample_toy_inputs(
            model, args.test_size, balanced_probabilities, toy_config.input_noise, seed * 101 + 3
        )
        pool_activations, pool_gates, pool_top = toy_forward_features(model, pool_x, toy_config.top_k)
        pool_phi = augmented_features(pool_activations, pool_gates)
        selectors = calibration_selectors(
            pool_labels, pool_top, pool_phi, args.calib_size, seed * 101 + 4
        )

        for selector, indices in selectors.items():
            print(f"[seed {seed}] selector={selector}", flush=True)
            calibration_activations = pool_activations[indices]
            calibration_gates = pool_gates[indices]
            quantized = quantize_all_methods(
                model.down, calibration_activations, calibration_gates, quant_config
            )
            affinity_objective = quantized["affinity_gptq"].calibration_objective
            if quantized["joint_cd"].calibration_objective > affinity_objective + 1e-8:
                raise RuntimeError("joint-CD baseline guarantee failed in experiment")
            for method, result in quantized.items():
                id_metrics = evaluate_quantized(model, id_test_x, result.weights, toy_config.top_k)
                balanced_metrics = evaluate_quantized(
                    model, balanced_test_x, result.weights, toy_config.top_k
                )
                rows.append({
                    "seed": seed,
                    "selector": selector,
                    "method": method,
                    "calibration_objective": result.calibration_objective,
                    "joint_cd_sweeps": max(0, len(result.history) - 1),
                    "id_block_nmse": id_metrics["block_nmse"],
                    "id_suffix_nmse": id_metrics["suffix_nmse"],
                    "id_teacher_agreement": id_metrics["teacher_agreement"],
                    "id_next_route_overlap": id_metrics["next_route_overlap"],
                    "balanced_block_nmse": balanced_metrics["block_nmse"],
                    "balanced_suffix_nmse": balanced_metrics["suffix_nmse"],
                    "balanced_teacher_agreement": balanced_metrics["teacher_agreement"],
                    "balanced_next_route_overlap": balanced_metrics["next_route_overlap"],
                    "guarantee_pass": (
                        result.calibration_objective <= affinity_objective + 1e-8
                        if method == "joint_cd" else True
                    ),
                })
        oracle_rows.append({"seed": seed, **exact_oracle_demo(seed + 1000, bits=args.bits)})

    payload: dict[str, object] = {
        "quant_config": asdict(quant_config),
        "toy_config": asdict(toy_config),
        "args": vars(args),
        "oracle": oracle_rows,
        "metrics": rows,
        "seconds": time.perf_counter() - started,
    }
    (output / "results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    with (output / "metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    _write_report(output, rows, oracle_rows)
    _write_plots(output, rows)
    print((output / "report.md").read_text(encoding="utf-8"), flush=True)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="hsvdquant/toy/results/moejointquant")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--bits", type=int, default=2)
    parser.add_argument("--damp", type=float, default=1e-3)
    parser.add_argument("--joint-sweeps", type=int, default=12)
    parser.add_argument("--no-act-order", action="store_true")
    parser.add_argument("--num-experts", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--d-model", type=int, default=12)
    parser.add_argument("--d-ff", type=int, default=7)
    parser.add_argument("--num-classes", type=int, default=7)
    parser.add_argument("--input-noise", type=float, default=0.55)
    parser.add_argument("--pool-size", type=int, default=768)
    parser.add_argument("--calib-size", type=int, default=96)
    parser.add_argument("--test-size", type=int, default=1024)
    parser.add_argument("--self-test", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.self_test:
        self_test()
        return
    run_toy_experiment(args)


if __name__ == "__main__":
    main()
