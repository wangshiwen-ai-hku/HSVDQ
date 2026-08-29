#!/usr/bin/env python3
"""CPU checks for the identities used by the local V3-OAR design.

This is intentionally independent of model checkpoints and CUDA.  It verifies
the exact output-error decomposition and the weighted reduced-rank branch
solution in hsvdquant/V3_OUTLIER_ROUTING.md.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from v3_outlier_routing import (  # noqa: E402
    block_hadamard_right,
    block_hadamard_weight,
    block_hadamard_weight_inverse,
    exact_error_terms,
    verify_block_hadamard_equivalence,
    weighted_rank_projection,
)


def symmetric_root(matrix: torch.Tensor, inverse: bool = False) -> torch.Tensor:
    values, vectors = torch.linalg.eigh((matrix + matrix.T) * 0.5)
    values = values.clamp_min(1e-10)
    powers = values.rsqrt() if inverse else values.sqrt()
    return vectors @ torch.diag(powers) @ vectors.T


def weighted_loss(
    error: torch.Tensor,
    token_weight: torch.Tensor,
    output_metric: torch.Tensor,
) -> torch.Tensor:
    return torch.trace(output_metric @ error.T @ token_weight @ error)


def main() -> None:
    torch.manual_seed(7)
    dtype = torch.float64
    rows, inputs, outputs, rank = 37, 12, 9, 3

    x = torch.randn(rows, inputs, dtype=dtype)
    smooth = torch.rand(inputs, dtype=dtype) + 0.5
    u = x / smooth
    weight = torch.randn(inputs, outputs, dtype=dtype)
    branch = torch.randn(inputs, rank, dtype=dtype) @ torch.randn(
        rank, outputs, dtype=dtype
    )

    order = torch.randperm(inputs)
    permutation = torch.eye(inputs, dtype=dtype)[:, order]
    v = u @ permutation
    residual_permuted = permutation.T @ (weight - branch)

    equivalence_error = (v @ residual_permuted - u @ (weight - branch)).abs().max()
    torch.testing.assert_close(equivalence_error, torch.zeros_like(equivalence_error))

    hadamard_signs = torch.where(
        torch.rand(inputs) < 0.5,
        -torch.ones(inputs, dtype=dtype),
        torch.ones(inputs, dtype=dtype),
    )
    hadamard_error = verify_block_hadamard_equivalence(
        v, residual_permuted, 4, hadamard_signs
    )
    if hadamard_error > 1e-10:
        raise AssertionError(f"block Hadamard equivalence drifted: {hadamard_error:.3e}")
    transformed_v = block_hadamard_right(v, 4, hadamard_signs)
    transformed_residual = block_hadamard_weight(
        residual_permuted, 4, hadamard_signs
    )
    recovered_residual = block_hadamard_weight_inverse(
        transformed_residual, 4, hadamard_signs
    )
    torch.testing.assert_close(transformed_v.norm(), v.norm(), rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(
        recovered_residual, residual_permuted, rtol=1e-12, atol=1e-12
    )

    # A deterministic stand-in for the deployed activation and weight quantizers.
    activation_scale = v.abs().amax(dim=1, keepdim=True).clamp_min(1e-8) / 7.0
    qv = torch.round(v / activation_scale).clamp(-7, 7) * activation_scale
    weight_scale = residual_permuted.abs().amax(dim=0, keepdim=True).clamp_min(1e-8) / 7.0
    q = torch.round(residual_permuted / weight_scale).clamp(-7, 7) * weight_scale

    omega_diag = torch.rand(rows, dtype=dtype) + 0.2
    omega = torch.diag(omega_diag / omega_diag.mean())
    gamma_seed = torch.randn(outputs, outputs, dtype=dtype)
    gamma = gamma_seed.T @ gamma_seed / float(outputs) + torch.eye(outputs, dtype=dtype) * 0.1

    activation_error = v - qv
    delta = residual_permuted - q
    direct_error = v @ delta + activation_error @ q
    direct = weighted_loss(direct_error, omega, gamma)

    hessian = v.T @ omega @ v
    cross = v.T @ omega @ activation_error
    sigma = activation_error.T @ omega @ activation_error
    weight_term = torch.trace(gamma @ delta.T @ hessian @ delta)
    cross_term = 2.0 * torch.trace(gamma @ delta.T @ cross @ q)
    activation_term = torch.trace(gamma @ q.T @ sigma @ q)
    decomposed = weight_term + cross_term + activation_term
    torch.testing.assert_close(direct, decomposed, rtol=1e-10, atol=1e-10)
    module_terms = exact_error_terms(
        v,
        qv,
        residual_permuted,
        q,
        token_weights=omega_diag,
        output_metric=gamma,
    )
    torch.testing.assert_close(
        direct.float(),
        torch.tensor(module_terms["direct"]),
        rtol=2e-6,
        atol=2e-5,
    )
    if module_terms["relative_identity_error"] > 2e-6:
        raise AssertionError(f"implementation decomposition drifted: {module_terms}")

    # Fixed-code FP target T = V W_P - Qa(V) Q.
    weight_permuted = permutation.T @ weight
    target = v @ weight_permuted - qv @ q
    hessian_regularized = hessian + torch.eye(inputs, dtype=dtype) * 1e-8
    unconstrained = torch.linalg.solve(hessian_regularized, v.T @ omega @ target)

    hroot = symmetric_root(hessian_regularized)
    hinvroot = symmetric_root(hessian_regularized, inverse=True)
    groot = symmetric_root(gamma)
    ginvroot = symmetric_root(gamma, inverse=True)
    transformed = hroot @ unconstrained @ groot
    left, values, right_t = torch.linalg.svd(transformed, full_matrices=False)
    truncated = (left[:, :rank] * values[:rank]) @ right_t[:rank]
    optimal_branch = hinvroot @ truncated @ ginvroot
    module_branch, _module_l1, _module_l2 = weighted_rank_projection(
        unconstrained.float(),
        hessian_regularized.float(),
        gamma.float(),
        rank,
    )

    optimal_error = target - v @ optimal_branch
    optimal_loss = weighted_loss(optimal_error, omega, gamma)
    module_loss = weighted_loss(
        target - v @ module_branch.to(dtype), omega, gamma
    )
    torch.testing.assert_close(module_loss, optimal_loss, rtol=2e-5, atol=2e-4)

    old_branch_permuted = permutation.T @ branch
    old_error = target - v @ old_branch_permuted
    old_loss = weighted_loss(old_error, omega, gamma)
    if optimal_loss > old_loss * (1.0 + 1e-8):
        raise AssertionError(
            f"rank-r optimum regressed: optimal={optimal_loss.item():.8e}, "
            f"old={old_loss.item():.8e}"
        )

    print("V3-OAR identities verified")
    print(f"permutation equivalence max error: {equivalence_error.item():.3e}")
    print(f"block Hadamard equivalence max error: {hadamard_error:.3e}")
    print(f"exact loss decomposition error: {abs(direct - decomposed).item():.3e}")
    print(f"fixed-code rank-{rank} loss: {optimal_loss.item():.6e}")
    print(f"old branch loss: {old_loss.item():.6e}")
    print(f"cross-term / total: {(cross_term / direct).item():.3%}")


if __name__ == "__main__":
    main()
