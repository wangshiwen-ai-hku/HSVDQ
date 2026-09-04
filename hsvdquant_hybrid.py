"""Hybrid H-SVDQuant runtime: Nunchaku W4A4 prefill plus W4A16 decode."""

from __future__ import annotations

import importlib.metadata
import os
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from hsvdquant_cuda import pack_signed_codes
from hsvdquant_int4 import NunchakuHSVQuantLinear, prepare_nunchaku_state


HYBRID_POLICIES = ("auto", "force_w4a4", "force_w4a16")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def select_hybrid_kernel(policy: str, threshold: int, rows: int) -> str:
    _require(policy in HYBRID_POLICIES, f"hybrid policy must be one of {HYBRID_POLICIES}")
    _require(threshold > 0, "hybrid threshold must be positive")
    _require(rows > 0, "hybrid dispatch requires at least one row")
    if policy == "force_w4a4":
        return "w4a4"
    if policy == "force_w4a16":
        return "w4a16"
    return "w4a16" if rows < threshold else "w4a4"


def nunchaku_version() -> str:
    try:
        return importlib.metadata.version("nunchaku")
    except importlib.metadata.PackageNotFoundError:
        return "unavailable"


@torch.no_grad()
def prepare_w4a16_state(state: dict[str, Any], compute_dtype: torch.dtype) -> dict[str, Any]:
    """Build row-major packed W4 state for the small-M decode kernel."""
    _require(int(state["bits"]) == 4, "hybrid decode requires W4 checkpoints")
    _require(compute_dtype in (torch.float16, torch.bfloat16), "hybrid runtime requires float16 or bfloat16")
    _require(state.get("activation_permutation") is None, "hybrid runtime does not support activation permutations")
    _require(
        not int(state.get("activation_hadamard_group_size", 0)),
        "hybrid runtime does not support block Hadamard transforms",
    )
    correction = state.get("correction")
    if isinstance(correction, dict):
        runtime_keys = {
            "dc_coeff",
            "lut_coeff",
            "sparse_threshold",
            "generic_left",
            "generic_right",
            "dc",
            "lut",
            "sparse",
            "generic",
        }
        _require(
            not any(key in correction and correction[key] is not None for key in runtime_keys),
            "hybrid runtime does not support correction modules",
        )

    in_features = int(state["in_features"])
    out_features = int(state["out_features"])
    group_size = int(state.get("group_size", 0)) or in_features
    rank = int(state["l1"].shape[1])
    _require(in_features % group_size == 0, "in_features must be divisible by weight group_size")
    _require(in_features % 2 == 0, "in_features must be even for W4 packing")
    _require(0 < rank <= 8, "hybrid runtime supports logical rank 1..8")

    codes = state["codes"].to(device="cpu", dtype=torch.int8).contiguous()
    scales = state["scales"].to(device="cpu", dtype=compute_dtype).contiguous()
    _require(tuple(codes.shape) == (out_features, in_features), "weight code shape mismatch")
    _require(
        tuple(scales.shape) == (out_features, in_features // group_size),
        "weight scale shape mismatch",
    )
    bias = torch.zeros(out_features, dtype=compute_dtype)
    if state.get("bias") is not None:
        bias.copy_(state["bias"].to(device="cpu", dtype=compute_dtype))

    return {
        "in_features": in_features,
        "out_features": out_features,
        "group_size": group_size,
        "rank": rank,
        "qweight_decode": pack_signed_codes(codes, 4),
        "wscales_decode": scales,
        "smooth_decode": state["d"].to(device="cpu", dtype=compute_dtype).contiguous(),
        "l1_decode": state["l1"].to(device="cpu", dtype=compute_dtype).contiguous(),
        "l2_decode": state["l2"].to(device="cpu", dtype=compute_dtype).contiguous(),
        "bias_decode": bias,
    }


_DECODE_EXTENSION: Any | None = None


def _load_decode_extension() -> Any:
    global _DECODE_EXTENSION
    if _DECODE_EXTENSION is not None:
        return _DECODE_EXTENSION
    if not torch.cuda.is_available():
        raise RuntimeError("hybrid W4A16 decode requires a CUDA device")
    from torch.utils.cpp_extension import load

    source_dir = Path(__file__).resolve().parent / "csrc" / "hsvdq_hybrid"
    _DECODE_EXTENSION = load(
        name="hsvdq_hybrid_ext_v1",
        sources=[str(source_dir / "bindings.cpp"), str(source_dir / "w4a16_gemv.cu")],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "--use_fast_math", "-lineinfo"],
        verbose=os.environ.get("HSVDQ_CUDA_VERBOSE", "0") == "1",
    )
    return _DECODE_EXTENSION


class W4A16HSVQuantLinear(nn.Module):
    """Small-M packed W4 residual plus FP16/BF16 activation Linear."""

    def __init__(self, state: dict[str, Any], compute_dtype: torch.dtype) -> None:
        super().__init__()
        packed = prepare_w4a16_state(state, compute_dtype)
        self.in_features = int(packed["in_features"])
        self.out_features = int(packed["out_features"])
        self.group_size = int(packed["group_size"])
        self.rank = int(packed["rank"])
        self.bits = 4
        self.activation_bits = 16
        self.runtime_backend = "w4a16"
        for source, target in (
            ("qweight_decode", "qweight"),
            ("wscales_decode", "wscales"),
            ("smooth_decode", "smooth"),
            ("l1_decode", "l1"),
            ("l2_decode", "l2"),
            ("bias_decode", "bias"),
        ):
            self.register_buffer(target, packed[source], persistent=True)
        self.register_buffer("_smoothed_workspace", None, persistent=False)
        self.register_buffer("_lowrank_workspace", None, persistent=False)

    def _get_workspaces(self, inputs: torch.Tensor, rows: int) -> tuple[torch.Tensor, torch.Tensor]:
        smoothed = self._smoothed_workspace
        lowrank = self._lowrank_workspace
        if (
            smoothed is None
            or smoothed.device != inputs.device
            or smoothed.dtype != inputs.dtype
            or smoothed.shape[0] < rows
        ):
            workspace_rows = max(rows, 1 if smoothed is None else int(smoothed.shape[0]))
            smoothed = torch.empty(
                (workspace_rows, self.in_features),
                device=inputs.device,
                dtype=inputs.dtype,
            )
            lowrank = torch.empty(
                (workspace_rows, self.rank), device=inputs.device, dtype=torch.float32
            )
            self._smoothed_workspace = smoothed
            self._lowrank_workspace = lowrank
        assert lowrank is not None
        return smoothed, lowrank

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.device.type != "cuda":
            raise RuntimeError("W4A16 inference requires CUDA activations")
        if inputs.dtype not in (torch.float16, torch.bfloat16):
            raise TypeError("W4A16 inference requires float16 or bfloat16 activations")
        leading_shape = inputs.shape[:-1]
        flat = inputs.reshape(-1, self.in_features).contiguous()
        extension = _load_decode_extension()
        smoothed_workspace, lowrank_workspace = self._get_workspaces(
            flat, int(flat.shape[0])
        )
        output = extension.w4a16_forward(
            flat,
            self.qweight,
            self.wscales,
            self.smooth,
            self.l1,
            self.l2,
            self.bias,
            smoothed_workspace,
            lowrank_workspace,
            self.group_size,
        )
        return output.reshape(*leading_shape, self.out_features)

    def packed_storage_bytes(self) -> int:
        return sum(tensor.numel() * tensor.element_size() for tensor in self.state_dict().values())


class HybridHSVQuantLinear(nn.Module):
    """Shape-dispatched W4A4 prefill and W4A16 decode Linear."""

    def __init__(
        self,
        state: dict[str, Any],
        compute_dtype: torch.dtype,
        *,
        policy: str = "auto",
        threshold: int = 128,
        allow_activation_group_remap: bool = False,
        profile_stats: bool = False,
    ) -> None:
        super().__init__()
        _require(policy in HYBRID_POLICIES, f"hybrid policy must be one of {HYBRID_POLICIES}")
        _require(threshold > 0, "hybrid threshold must be positive")
        prefill = prepare_nunchaku_state(
            state,
            compute_dtype,
            allow_activation_group_remap=allow_activation_group_remap,
        )
        self.decode = W4A16HSVQuantLinear(state, compute_dtype)

        self.in_features = self.decode.in_features
        self.out_features = self.decode.out_features
        self.group_size = self.decode.group_size
        self.rank = self.decode.rank
        self.bits = 4
        self.activation_bits = 4
        self.runtime_backend = "hybrid"
        self.policy = policy
        self.threshold = int(threshold)
        self.profile_stats = bool(profile_stats)
        self.prefill = NunchakuHSVQuantLinear(prefill, compute_dtype)
        self._kernel_calls = {"w4a4": 0, "w4a16": 0}
        self._kernel_rows = {"w4a4": 0, "w4a16": 0}

    def _select_kernel(self, rows: int) -> str:
        return select_hybrid_kernel(self.policy, self.threshold, rows)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.device.type != "cuda":
            raise RuntimeError("hybrid inference requires CUDA activations")
        if inputs.dtype not in (torch.float16, torch.bfloat16):
            raise TypeError("hybrid inference requires float16 or bfloat16 activations")
        leading_shape = inputs.shape[:-1]
        flat = inputs.reshape(-1, self.in_features).contiguous()
        rows = int(flat.shape[0])
        kernel = self._select_kernel(rows)
        if self.profile_stats:
            self._kernel_calls[kernel] += 1
            self._kernel_rows[kernel] += rows

        if kernel == "w4a4":
            return self.prefill(inputs)

        return self.decode(inputs)

    def runtime_stats(self) -> dict[str, Any]:
        return {
            "policy": self.policy,
            "threshold": self.threshold,
            "kernel_calls": dict(self._kernel_calls),
            "rows_by_kernel": dict(self._kernel_rows),
        }

    def packed_storage_bytes(self) -> int:
        return sum(tensor.numel() * tensor.element_size() for tensor in self.state_dict().values())

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"rank={self.rank}, policy={self.policy}, threshold={self.threshold}"
        )


def build_hybrid_linear(
    state: dict[str, Any],
    compute_dtype: torch.dtype,
    *,
    policy: str = "auto",
    threshold: int = 128,
    allow_activation_group_remap: bool = False,
    profile_stats: bool = False,
) -> HybridHSVQuantLinear:
    return HybridHSVQuantLinear(
        state,
        compute_dtype,
        policy=policy,
        threshold=threshold,
        allow_activation_group_remap=allow_activation_group_remap,
        profile_stats=profile_stats,
    )


def build_w4a16_linear(state: dict[str, Any], compute_dtype: torch.dtype) -> W4A16HSVQuantLinear:
    return W4A16HSVQuantLinear(state, compute_dtype)


def collect_hybrid_runtime_stats(model: nn.Module) -> dict[str, Any]:
    totals = {
        "layers": 0,
        "policy": None,
        "threshold": None,
        "weight_layout": "dual",
        "kernel_calls": {"w4a4": 0, "w4a16": 0},
        "rows_by_kernel": {"w4a4": 0, "w4a16": 0},
    }
    for module in model.modules():
        if not isinstance(module, HybridHSVQuantLinear):
            continue
        totals["layers"] += 1
        stats = module.runtime_stats()
        if totals["policy"] is None:
            totals["policy"] = stats["policy"]
            totals["threshold"] = stats["threshold"]
        for kernel in ("w4a4", "w4a16"):
            totals["kernel_calls"][kernel] += stats["kernel_calls"][kernel]
            totals["rows_by_kernel"][kernel] += stats["rows_by_kernel"][kernel]
    return totals
