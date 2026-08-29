"""Local V3-OAR calibration primitives.

The module is deliberately eager and CPU/GPU agnostic.  It implements the
algorithmic pieces that must be validated before changing the native W4A4
kernel: static output-aware activation grouping, tail-aware smoothing, the
exact F_W/cross/F_A decomposition, and a fixed-code weighted rank-r FP branch
update.  A permutation is represented by an index vector ``order`` such that
``x_permuted = x[:, order]`` and ``w_permuted = w[order]``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class V3OARConfig:
    activation_bits: int = 4
    activation_group_size: int = 128
    weight_bits: int = 4
    weight_group_size: int = 128
    rank: int = 8
    damp: float = 1e-5
    tail_fraction: float = 0.01
    tail_weight: float = 1.0
    grouping_passes: int = 2
    grouping_candidates: int = 2048
    smoothing_steps: int = 80
    smoothing_lr: float = 0.04
    smoothing_clip: float = 16.0
    activation_weight: float = 1.0
    fw_epsilon: float = 0.05

    def validate(self, channels: int | None = None) -> None:
        if self.activation_bits < 2 or self.weight_bits < 2:
            raise ValueError("quantization bit widths must be >= 2")
        if self.activation_group_size <= 0 or self.weight_group_size <= 0:
            raise ValueError("V3-OAR requires positive fixed group sizes")
        if channels is not None and channels % self.activation_group_size:
            raise ValueError("channels must be divisible by activation_group_size")
        if self.rank < 0:
            raise ValueError("rank must be non-negative")
        if not 0 < self.tail_fraction <= 1:
            raise ValueError("tail_fraction must be in (0, 1]")
        if self.tail_weight < 0 or self.activation_weight < 0:
            raise ValueError("loss weights must be non-negative")
        if self.grouping_passes < 0 or self.grouping_candidates < 0:
            raise ValueError("grouping search budgets must be non-negative")
        if self.smoothing_steps < 0 or self.smoothing_lr <= 0:
            raise ValueError("invalid smoothing optimizer settings")
        if self.smoothing_clip < 1:
            raise ValueError("smoothing_clip must be >= 1")
        if self.fw_epsilon < 0:
            raise ValueError("fw_epsilon must be non-negative")


def _sym(matrix: torch.Tensor) -> torch.Tensor:
    return (matrix + matrix.T) * 0.5


def _metric_root(matrix: torch.Tensor, inverse: bool = False, floor: float = 1e-9) -> torch.Tensor:
    values, vectors = torch.linalg.eigh(_sym(matrix.float()))
    values = values.clamp_min(floor)
    powers = values.rsqrt() if inverse else values.sqrt()
    return (vectors * powers[None, :]) @ vectors.T


def _token_diagonal(
    rows: int,
    token_weights: torch.Tensor | None,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if token_weights is None:
        return torch.ones(rows, device=device, dtype=dtype)
    weights = token_weights.to(device=device, dtype=dtype).reshape(-1)
    if weights.numel() != rows:
        raise ValueError(f"expected {rows} token weights, got {weights.numel()}")
    return weights / weights.mean().clamp_min(1e-12)


def _output_metric(
    outputs: int,
    metric: torch.Tensor | None,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if metric is None:
        return torch.eye(outputs, device=device, dtype=dtype)
    result = metric.to(device=device, dtype=dtype)
    if result.ndim == 1:
        if result.numel() != outputs:
            raise ValueError(f"expected {outputs} output weights, got {result.numel()}")
        result = torch.diag(result)
    if result.shape != (outputs, outputs):
        raise ValueError(f"expected output metric {(outputs, outputs)}, got {tuple(result.shape)}")
    return _sym(result)


def quantize_activations(
    inputs: torch.Tensor,
    bits: int,
    group_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-token symmetric quantization in the current channel order."""

    if inputs.ndim != 2:
        raise ValueError("inputs must have shape [tokens, channels]")
    if bits >= 16:
        scales = torch.ones(
            inputs.shape[0], 1, device=inputs.device, dtype=inputs.dtype
        )
        return inputs, torch.zeros_like(inputs, dtype=torch.int16), scales
    rows, channels = inputs.shape
    if group_size <= 0 or channels % group_size:
        raise ValueError("group_size must be positive and divide channels")
    qmax = float(2 ** (bits - 1) - 1)
    groups = channels // group_size
    grouped = inputs.reshape(rows, groups, group_size)
    scales = grouped.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8) / qmax
    codes = torch.round(grouped / scales).clamp(-qmax, qmax)
    quantized = (codes * scales).reshape_as(inputs)
    return quantized, codes.reshape_as(inputs).to(torch.int16), scales.squeeze(-1)


def quantize_weights(
    weight: torch.Tensor,
    bits: int,
    group_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """RTN weight quantization over input-channel groups for each output."""

    if weight.ndim != 2:
        raise ValueError("weight must have shape [input_channels, outputs]")
    channels, outputs = weight.shape
    if group_size <= 0 or channels % group_size:
        raise ValueError("group_size must be positive and divide input channels")
    qmax = float(2 ** (bits - 1) - 1)
    groups = channels // group_size
    grouped = weight.T.reshape(outputs, groups, group_size)
    scales = grouped.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8) / qmax
    codes = torch.round(grouped / scales).clamp(-qmax, qmax)
    quantized = (codes * scales).reshape(outputs, channels).T.contiguous()
    return quantized, codes.reshape(outputs, channels).to(torch.int8), scales.squeeze(-1)


def _require_power_of_two(value: int) -> None:
    if value <= 0 or value & (value - 1):
        raise ValueError("block Hadamard group_size must be a positive power of two")


def block_hadamard_right(
    inputs: torch.Tensor,
    group_size: int,
    signs: torch.Tensor | None = None,
) -> torch.Tensor:
    """Multiply channel blocks on the right by R = D H.

    ``H`` is the normalized Walsh--Hadamard matrix and ``D`` is an optional
    static random-sign diagonal.  The transform preserves shape and L2 norm.
    """

    _require_power_of_two(group_size)
    channels = inputs.shape[-1]
    if channels % group_size:
        raise ValueError("channels must be divisible by the Hadamard group size")
    original_shape = inputs.shape
    transformed = inputs
    if signs is not None:
        sign_values = signs.to(device=inputs.device, dtype=inputs.dtype).reshape(-1)
        if sign_values.numel() != channels:
            raise ValueError("Hadamard signs length must equal the channel count")
        transformed = transformed * sign_values
    transformed = transformed.reshape(-1, channels // group_size, group_size)
    width = 1
    while width < group_size:
        blocks = transformed.reshape(
            *transformed.shape[:-1], group_size // (2 * width), 2, width
        )
        left = blocks[..., 0, :]
        right = blocks[..., 1, :]
        transformed = torch.cat((left + right, left - right), dim=-1).reshape(
            *transformed.shape[:-1], group_size
        )
        width *= 2
    transformed = transformed / float(group_size) ** 0.5
    transformed = transformed.reshape(original_shape)
    return transformed


def block_hadamard_weight(
    weight: torch.Tensor,
    group_size: int,
    signs: torch.Tensor | None = None,
) -> torch.Tensor:
    """Map residual rows by R^T = H D for ``xR @ R^T w == x @ w``."""

    transformed = weight
    if signs is not None:
        sign_values = signs.to(device=weight.device, dtype=weight.dtype).reshape(-1)
        if sign_values.numel() != weight.shape[0]:
            raise ValueError("Hadamard signs length must equal the weight row count")
        transformed = transformed * sign_values[:, None]
    return block_hadamard_right(transformed.T, group_size).T


def block_hadamard_weight_inverse(
    transformed_weight: torch.Tensor,
    group_size: int,
    signs: torch.Tensor | None = None,
) -> torch.Tensor:
    """Map transformed residual rows back by R = D H.

    This is the inverse of :func:`block_hadamard_weight` and is used when an
    eager checkpoint is materialized as a conventional dense weight.
    """

    weight = block_hadamard_right(transformed_weight.T, group_size).T
    if signs is not None:
        sign_values = signs.to(
            device=transformed_weight.device, dtype=transformed_weight.dtype
        ).reshape(-1)
        if sign_values.numel() != transformed_weight.shape[0]:
            raise ValueError("Hadamard signs length must equal the weight row count")
        weight = weight * sign_values[:, None]
    return weight


def verify_block_hadamard_equivalence(
    inputs: torch.Tensor,
    weight: torch.Tensor,
    group_size: int,
    signs: torch.Tensor | None = None,
) -> float:
    transformed_inputs = block_hadamard_right(inputs, group_size, signs)
    transformed_weight = block_hadamard_weight(weight, group_size, signs)
    error = transformed_inputs @ transformed_weight - inputs @ weight
    return float(error.abs().max().item())


def weighted_output_loss(
    error: torch.Tensor,
    token_weights: torch.Tensor | None = None,
    output_metric: torch.Tensor | None = None,
) -> torch.Tensor:
    if error.ndim != 2:
        raise ValueError("error must have shape [tokens, outputs]")
    omega = _token_diagonal(
        error.shape[0], token_weights, device=error.device, dtype=error.dtype
    )
    gamma = _output_metric(
        error.shape[1], output_metric, device=error.device, dtype=error.dtype
    )
    return torch.sum((error * omega[:, None]) * (error @ gamma))


def cvar(values: torch.Tensor, tail_fraction: float) -> torch.Tensor:
    """Empirical upper-tail mean; ``tail_fraction=.01`` is CVaR99."""

    flat = values.reshape(-1)
    count = max(1, int(round(flat.numel() * float(tail_fraction))))
    return torch.topk(flat, count, largest=True).values.mean()


def tail_token_weights(
    activation_output_error: torch.Tensor,
    tail_fraction: float,
    tail_weight: float,
) -> torch.Tensor:
    """IRLS weights corresponding to mean plus an empirical CVaR emphasis."""

    energy = activation_output_error.float().square().mean(dim=1)
    count = max(1, int(round(energy.numel() * float(tail_fraction))))
    threshold = torch.topk(energy, count, largest=True).values.min()
    weights = torch.ones_like(energy)
    weights[energy >= threshold] += float(tail_weight) / max(float(tail_fraction), 1e-12)
    return weights / weights.mean().clamp_min(1e-12)


def exact_error_terms(
    inputs: torch.Tensor,
    quantized_inputs: torch.Tensor,
    residual_target: torch.Tensor,
    quantized_residual: torch.Tensor,
    token_weights: torch.Tensor | None = None,
    output_metric: torch.Tensor | None = None,
) -> dict[str, float]:
    """Evaluate Eq. (2): F_W, cross, F_A, and the direct paired loss."""

    if inputs.shape != quantized_inputs.shape:
        raise ValueError("inputs and quantized_inputs must have the same shape")
    if residual_target.shape != quantized_residual.shape:
        raise ValueError("residual_target and quantized_residual must have the same shape")
    omega = _token_diagonal(
        inputs.shape[0], token_weights, device=inputs.device, dtype=inputs.dtype
    )
    gamma = _output_metric(
        residual_target.shape[1], output_metric, device=inputs.device, dtype=inputs.dtype
    )
    activation_error = inputs - quantized_inputs
    delta = residual_target - quantized_residual
    hessian = inputs.T @ (inputs * omega[:, None])
    cross_covariance = inputs.T @ (activation_error * omega[:, None])
    activation_covariance = activation_error.T @ (activation_error * omega[:, None])
    fw = torch.trace(gamma @ delta.T @ hessian @ delta)
    cross = 2.0 * torch.trace(gamma @ delta.T @ cross_covariance @ quantized_residual)
    fa = torch.trace(
        gamma @ quantized_residual.T @ activation_covariance @ quantized_residual
    )
    direct_error = inputs @ delta + activation_error @ quantized_residual
    direct = weighted_output_loss(direct_error, omega, gamma)
    scale = direct.abs().clamp_min(1e-30)
    return {
        "fw": float(fw.item()),
        "cross": float(cross.item()),
        "fa": float(fa.item()),
        "decomposed": float((fw + cross + fa).item()),
        "direct": float(direct.item()),
        "relative_identity_error": float(((fw + cross + fa - direct).abs() / scale).item()),
    }


def _group_output_cost(
    inputs: torch.Tensor,
    weight: torch.Tensor,
    indices: torch.Tensor,
    bits: int,
    token_weights: torch.Tensor | None,
    output_metric: torch.Tensor | None,
) -> torch.Tensor:
    group_inputs = inputs.index_select(1, indices)
    quantized, _codes, _scales = quantize_activations(
        group_inputs, bits, group_inputs.shape[1]
    )
    group_weight = weight.index_select(0, indices)
    return weighted_output_loss(
        (group_inputs - quantized) @ group_weight,
        token_weights,
        output_metric,
    )


def amax_permutation(inputs: torch.Tensor) -> torch.Tensor:
    """Staircase warm start: channels with comparable robust tails share groups."""

    if inputs.ndim != 2:
        raise ValueError("inputs must have shape [tokens, channels]")
    # A high quantile is stable enough for small calibration reservoirs, while
    # the max term keeps genuinely massive but rare channels in the top tier.
    robust = torch.quantile(inputs.abs().float(), 0.995, dim=0)
    massive = inputs.abs().amax(dim=0).float()
    score = robust.clamp_min(1e-12).log() + 0.25 * massive.clamp_min(1e-12).log()
    return torch.argsort(score, stable=True)


@torch.no_grad()
def refine_output_aware_permutation(
    inputs: torch.Tensor,
    residual_weight: torch.Tensor,
    initial_order: torch.Tensor,
    config: V3OARConfig,
    token_weights: torch.Tensor | None = None,
    output_metric: torch.Tensor | None = None,
    seed: int = 0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Balanced random-swap descent on the group cost in Eq. (3)."""

    config.validate(inputs.shape[1])
    channels = inputs.shape[1]
    group_size = config.activation_group_size
    groups = channels // group_size
    order = initial_order.to(device=inputs.device, dtype=torch.long).clone()
    if torch.unique(order).numel() != channels:
        raise ValueError("initial_order must be a permutation")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)

    def group_indices(group: int) -> torch.Tensor:
        start = group * group_size
        return order[start : start + group_size]

    costs = [
        _group_output_cost(
            inputs,
            residual_weight,
            group_indices(group),
            config.activation_bits,
            token_weights,
            output_metric,
        )
        for group in range(groups)
    ]
    initial_cost = torch.stack(costs).sum()
    accepted = 0
    attempted = 0
    per_pass = max(1, config.grouping_candidates // max(1, config.grouping_passes))
    for _ in range(config.grouping_passes):
        for _ in range(per_pass):
            if groups < 2:
                break
            pair = torch.randperm(groups, generator=generator)[:2]
            left_group, right_group = int(pair[0]), int(pair[1])
            left_offset = int(torch.randint(group_size, (1,), generator=generator))
            right_offset = int(torch.randint(group_size, (1,), generator=generator))
            left_position = left_group * group_size + left_offset
            right_position = right_group * group_size + right_offset
            old_pair = costs[left_group] + costs[right_group]
            saved_left = order[left_position].clone()
            order[left_position] = order[right_position]
            order[right_position] = saved_left
            new_left = _group_output_cost(
                inputs,
                residual_weight,
                group_indices(left_group),
                config.activation_bits,
                token_weights,
                output_metric,
            )
            new_right = _group_output_cost(
                inputs,
                residual_weight,
                group_indices(right_group),
                config.activation_bits,
                token_weights,
                output_metric,
            )
            attempted += 1
            if new_left + new_right < old_pair:
                costs[left_group] = new_left
                costs[right_group] = new_right
                accepted += 1
            else:
                saved_left = order[left_position].clone()
                order[left_position] = order[right_position]
                order[right_position] = saved_left
    final_cost = torch.stack(costs).sum()
    return order, {
        "surrogate_cost_before": float(initial_cost.item()),
        "surrogate_cost_after": float(final_cost.item()),
        "surrogate_gain": float(
            ((initial_cost - final_cost) / initial_cost.clamp_min(1e-30)).item()
        ),
        "swaps_attempted": float(attempted),
        "swaps_accepted": float(accepted),
    }


def activation_output_loss_for_order(
    inputs: torch.Tensor,
    residual_weight: torch.Tensor,
    order: torch.Tensor,
    bits: int,
    group_size: int,
    token_weights: torch.Tensor | None = None,
    output_metric: torch.Tensor | None = None,
) -> torch.Tensor:
    permuted_inputs = inputs.index_select(1, order)
    permuted_weight = residual_weight.index_select(0, order)
    quantized, _codes, _scales = quantize_activations(
        permuted_inputs, bits, group_size
    )
    return weighted_output_loss(
        (permuted_inputs - quantized) @ permuted_weight,
        token_weights,
        output_metric,
    )


def weighted_rank_projection(
    matrix: torch.Tensor,
    hessian: torch.Tensor,
    output_metric: torch.Tensor | None,
    rank: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Best rank-r approximation in the H-left/Gamma-right metric."""

    outputs = matrix.shape[1]
    gamma = _output_metric(
        outputs, output_metric, device=matrix.device, dtype=matrix.dtype
    )
    hroot = _metric_root(hessian)
    hinvroot = _metric_root(hessian, inverse=True)
    groot = _metric_root(gamma)
    ginvroot = _metric_root(gamma, inverse=True)
    transformed = hroot @ matrix.float() @ groot
    left, values, right_t = torch.linalg.svd(transformed, full_matrices=False)
    actual_rank = min(rank, values.numel())
    if actual_rank == 0:
        return (
            torch.zeros_like(matrix),
            matrix.new_zeros((matrix.shape[0], 0)),
            matrix.new_zeros((0, matrix.shape[1])),
        )
    left = left[:, :actual_rank]
    values = values[:actual_rank]
    right_t = right_t[:actual_rank]
    root_values = values.sqrt()
    l1 = hinvroot @ (left * root_values[None, :])
    l2 = (root_values[:, None] * right_t) @ ginvroot
    return l1 @ l2, l1.to(matrix.dtype), l2.to(matrix.dtype)


@torch.no_grad()
def fixed_code_fp_route(
    inputs: torch.Tensor,
    quantized_inputs: torch.Tensor,
    teacher_weight: torch.Tensor,
    quantized_residual: torch.Tensor,
    rank: int,
    token_weights: torch.Tensor | None = None,
    output_metric: torch.Tensor | None = None,
    baseline_branch: torch.Tensor | None = None,
    damp: float = 1e-5,
    fw_epsilon: float = 0.05,
    backtrack_scales: tuple[float, ...] = (0.0, 0.125, 0.25, 0.5, 1.0),
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, float]]:
    """Eq. (8)--(11) with a fixed-code A16/F_W trust region."""

    rows, channels = inputs.shape
    outputs = teacher_weight.shape[1]
    if teacher_weight.shape != quantized_residual.shape:
        raise ValueError("teacher_weight and quantized_residual must have the same shape")
    omega = _token_diagonal(
        rows, token_weights, device=inputs.device, dtype=inputs.dtype
    )
    gamma = _output_metric(
        outputs, output_metric, device=inputs.device, dtype=inputs.dtype
    )
    hessian = inputs.T @ (inputs * omega[:, None]) / float(max(1, rows))
    reference = hessian.diagonal().mean().clamp_min(1e-12)
    hessian = _sym(hessian) + torch.eye(
        channels, device=inputs.device, dtype=inputs.dtype
    ) * (float(damp) * reference)
    target = inputs @ teacher_weight - quantized_inputs @ quantized_residual
    cross = inputs.T @ (target * omega[:, None]) / float(max(1, rows))
    unconstrained = torch.linalg.solve(hessian, cross)
    proposal, _proposal_l1, _proposal_l2 = weighted_rank_projection(
        unconstrained, hessian, gamma, rank
    )
    if baseline_branch is None:
        baseline_branch = torch.zeros_like(proposal)
    baseline_branch = baseline_branch.to(device=proposal.device, dtype=proposal.dtype)

    def a4_loss(branch: torch.Tensor) -> torch.Tensor:
        return weighted_output_loss(
            target - inputs @ branch, omega, gamma
        )

    def fw_loss(branch: torch.Tensor) -> torch.Tensor:
        error = inputs @ (teacher_weight - branch - quantized_residual)
        return weighted_output_loss(error, omega, gamma)

    baseline_a4 = a4_loss(baseline_branch)
    baseline_fw = fw_loss(baseline_branch)
    fw_limit = baseline_fw * (1.0 + float(fw_epsilon))
    best_branch = baseline_branch
    best_l1_l2 = weighted_rank_projection(best_branch, hessian, gamma, rank)[1:]
    best_a4 = baseline_a4
    best_fw = baseline_fw
    best_scale = 0.0
    eligible = 0
    for scale in sorted(set(float(value) for value in backtrack_scales)):
        mixed = baseline_branch + scale * (proposal - baseline_branch)
        candidate, candidate_l1, candidate_l2 = weighted_rank_projection(
            mixed, hessian, gamma, rank
        )
        candidate_fw = fw_loss(candidate)
        if candidate_fw > fw_limit * (1.0 + 1e-7):
            continue
        eligible += 1
        candidate_a4 = a4_loss(candidate)
        if candidate_a4 < best_a4:
            best_branch = candidate
            best_l1_l2 = (candidate_l1, candidate_l2)
            best_a4 = candidate_a4
            best_fw = candidate_fw
            best_scale = scale
    best_l1, best_l2 = best_l1_l2
    return best_branch, best_l1, best_l2, {
        "a4_loss_before": float(baseline_a4.item()),
        "a4_loss_after": float(best_a4.item()),
        "a4_gain": float(
            ((baseline_a4 - best_a4) / baseline_a4.clamp_min(1e-30)).item()
        ),
        "fw_before": float(baseline_fw.item()),
        "fw_after": float(best_fw.item()),
        "fw_limit": float(fw_limit.item()),
        "accepted_scale": float(best_scale),
        "eligible_scales": float(eligible),
    }


def _geometric_normalize(values: torch.Tensor, clip: float) -> torch.Tensor:
    logs = values.clamp_min(1e-12).log()
    logs = logs - logs.mean()
    if clip > 1:
        bound = float(torch.tensor(clip).log().item())
        logs = logs.clamp(-bound, bound)
        logs = logs - logs.mean()
    return logs.exp()


def _tail_mean(values: torch.Tensor, fraction: float) -> torch.Tensor:
    count = max(1, int(round(values.numel() * float(fraction))))
    return torch.topk(values.reshape(-1), count, largest=True).values.mean()


def optimize_tail_smoothing(
    inputs: torch.Tensor,
    residual_unsmoothed: torch.Tensor,
    order: torch.Tensor,
    config: V3OARConfig,
    output_metric: torch.Tensor | None = None,
    initial: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Optimize D on the grouped mean+CVaR surrogate in Eq. (5)--(6)."""

    config.validate(inputs.shape[1])
    channels, outputs = residual_unsmoothed.shape
    if inputs.shape[1] != channels:
        raise ValueError("activation and residual channel counts differ")
    gamma = _output_metric(
        outputs, output_metric, device=inputs.device, dtype=inputs.dtype
    )
    hdiag = inputs.float().square().mean(dim=0).clamp_min(1e-12)
    residual_energy = torch.sum((residual_unsmoothed.float() @ gamma) * residual_unsmoothed.float(), dim=1)
    if initial is None:
        log_d = 0.25 * (hdiag.log() - residual_energy.clamp_min(1e-12).log())
        initial = _geometric_normalize(log_d.exp(), config.smoothing_clip)
    initial = initial.to(device=inputs.device, dtype=torch.float32)
    parameter = torch.nn.Parameter(initial.log())
    optimizer = torch.optim.Adam([parameter], lr=config.smoothing_lr)
    qmax_a = float(2 ** (config.activation_bits - 1) - 1)
    qmax_w = float(2 ** (config.weight_bits - 1) - 1)
    order = order.to(device=inputs.device, dtype=torch.long)
    groups = channels // config.activation_group_size
    if channels % config.weight_group_size:
        raise ValueError("channels must be divisible by weight_group_size")

    def terms(d: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        d_ordered = d.index_select(0, order)
        x_ordered = inputs.float().index_select(1, order) / d_ordered
        residual_ordered = residual_unsmoothed.float().index_select(0, order) * d_ordered[:, None]
        activation_tokens = torch.zeros(inputs.shape[0], device=inputs.device)
        for group in range(groups):
            start = group * config.activation_group_size
            end = start + config.activation_group_size
            peak2 = x_ordered[:, start:end].abs().amax(dim=1).square()
            block = residual_ordered[start:end]
            mass = torch.sum((block @ gamma) * block)
            activation_tokens = activation_tokens + peak2 * mass / (12.0 * qmax_a * qmax_a)
        fw = inputs.new_zeros((), dtype=torch.float32)
        weight_groups = channels // config.weight_group_size
        h_ordered = hdiag.index_select(0, order)
        for group in range(weight_groups):
            start = group * config.weight_group_size
            end = start + config.weight_group_size
            block = residual_ordered[start:end]
            step2 = block.abs().amax(dim=0).square() / (qmax_w * qmax_w)
            sensitivity = (h_ordered[start:end] / d_ordered[start:end].square()).sum()
            fw = fw + step2.sum() * sensitivity / 12.0
        fa_mean = activation_tokens.mean()
        fa_tail = _tail_mean(activation_tokens, config.tail_fraction)
        return fw, fa_mean, fa_tail

    with torch.enable_grad():
        for _ in range(config.smoothing_steps):
            optimizer.zero_grad(set_to_none=True)
            centered = parameter - parameter.mean()
            if config.smoothing_clip > 1:
                bound = float(torch.tensor(config.smoothing_clip).log().item())
                centered = centered.clamp(-bound, bound)
            d = centered.exp()
            fw, fa_mean, fa_tail = terms(d)
            loss = (
                fw
                + config.activation_weight
                * (fa_mean + config.tail_weight * fa_tail)
                + 1e-20
            ).log()
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                parameter.sub_(parameter.mean())
                if config.smoothing_clip > 1:
                    parameter.clamp_(-bound, bound)
    result = _geometric_normalize(parameter.detach().exp(), config.smoothing_clip)
    before = terms(initial)
    after = terms(result)
    return result.to(inputs.dtype), {
        "fw_before": float(before[0].item()),
        "fw_after": float(after[0].item()),
        "fa_mean_before": float(before[1].item()),
        "fa_mean_after": float(after[1].item()),
        "fa_tail_before": float(before[2].item()),
        "fa_tail_after": float(after[2].item()),
        "d_min": float(result.min().item()),
        "d_max": float(result.max().item()),
    }


def state_metrics(
    inputs: torch.Tensor,
    teacher_weight: torch.Tensor,
    branch: torch.Tensor,
    quantized_inputs: torch.Tensor,
    quantized_residual: torch.Tensor,
    tail_fraction: float = 0.01,
    output_metric: torch.Tensor | None = None,
    quantizer_inputs: torch.Tensor | None = None,
) -> dict[str, float]:
    if quantizer_inputs is None:
        quantizer_inputs = inputs
    if quantizer_inputs.shape != quantized_inputs.shape:
        raise ValueError("quantizer_inputs and quantized_inputs must have the same shape")
    teacher = inputs @ teacher_weight
    prediction = inputs @ branch + quantized_inputs @ quantized_residual
    error = teacher - prediction
    activation_error = (quantizer_inputs - quantized_inputs) @ quantized_residual
    per_token = error.square().mean(dim=1)
    teacher_energy = teacher.square().mean().clamp_min(1e-30)
    return {
        "mse": float(error.square().mean().item()),
        "nmse": float((error.square().mean() / teacher_energy).item()),
        "cvar": float(_tail_mean(per_token, tail_fraction).item()),
        "activation_mse": float(activation_error.square().mean().item()),
        "a16_mse": float(
            (
                teacher
                - inputs @ branch
                - quantizer_inputs @ quantized_residual
            ).square().mean().item()
        ),
        "weighted_loss": float(
            weighted_output_loss(error, None, output_metric).div(max(1, error.numel())).item()
        ),
    }


def serialize_diagnostics(values: dict[str, Any]) -> dict[str, Any]:
    """Convert tensors in experiment diagnostics into JSON-friendly values."""

    result: dict[str, Any] = {}
    for key, value in values.items():
        if isinstance(value, torch.Tensor):
            result[key] = value.detach().cpu().tolist()
        elif isinstance(value, dict):
            result[key] = serialize_diagnostics(value)
        else:
            result[key] = value
    return result
