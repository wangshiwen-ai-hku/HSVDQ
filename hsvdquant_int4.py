"""Packed INT4 runtime adapter for H-SVDQuant checkpoints.

The packing transforms follow the MMA layout used by Nunchaku's W4A4 kernel.
They are kept dependency-free so checkpoint conversion can be tested on CPU;
the optional Nunchaku extension is imported only when a runtime module is built.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn


NUNCHAKU_GROUP_SIZE = 64
NUNCHAKU_WARP_N = 128
NUNCHAKU_RANK_TILE = 16


def _require_shape(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def pack_nunchaku_weight(codes: torch.Tensor) -> torch.Tensor:
    """Pack signed INT4 [out, in] codes for Nunchaku's MMA weight operand."""
    _require_shape(codes.ndim == 2, "weight codes must be a 2D tensor")
    out_features, in_features = codes.shape
    _require_shape(out_features % NUNCHAKU_WARP_N == 0, "out_features must be divisible by 128")
    _require_shape(in_features % 128 == 0, "in_features must be divisible by 128")
    codes_i32 = codes.to(device="cpu", dtype=torch.int32).contiguous()
    if codes_i32.numel():
        _require_shape(int(codes_i32.min()) >= -8 and int(codes_i32.max()) <= 7, "INT4 codes must be in [-8, 7]")

    # Adapted from DeepCompressor's Apache-2.0 NunchakuWeightPacker.
    tiled = codes_i32.reshape(
        out_features // 128,
        8,
        2,
        8,
        1,
        in_features // 64,
        1,
        2,
        4,
        8,
    )
    tiled = tiled.permute(0, 5, 6, 1, 3, 8, 2, 7, 4, 9).contiguous()
    shifts = torch.arange(0, 32, 4, dtype=torch.int32)
    words = ((tiled & 0xF) << shifts).sum(dim=-1, dtype=torch.int32)
    return words.view(torch.int8).reshape(out_features, in_features // 2).contiguous()


def unpack_nunchaku_weight(packed: torch.Tensor, in_features: int) -> torch.Tensor:
    """Inverse of :func:`pack_nunchaku_weight`, used by CPU validation."""
    _require_shape(packed.ndim == 2 and packed.dtype == torch.int8, "packed weight must be 2D int8")
    out_features = packed.shape[0]
    _require_shape(packed.shape[1] * 2 == in_features, "packed weight has the wrong in_features")
    words = packed.contiguous().view(torch.int32)
    permuted = words.reshape(
        out_features // 128,
        in_features // 64,
        1,
        8,
        8,
        4,
        2,
        2,
        1,
    )
    shifts = torch.arange(0, 32, 4, dtype=torch.int32)
    nibbles = (permuted.unsqueeze(-1) >> shifts) & 0xF
    signed = torch.where(nibbles >= 8, nibbles - 16, nibbles)
    return signed.permute(0, 3, 6, 4, 8, 1, 2, 7, 5, 9).contiguous().reshape(
        out_features, in_features
    ).to(torch.int8)


def pack_nunchaku_scales(scales: torch.Tensor) -> torch.Tensor:
    """Pack [out, groups] FP scales into Nunchaku's [groups, out] layout."""
    _require_shape(scales.ndim == 2, "scales must be [out_features, groups]")
    out_features, groups = scales.shape
    _require_shape(out_features % NUNCHAKU_WARP_N == 0, "out_features must be divisible by 128")
    tiled = scales.contiguous().reshape(out_features // 128, 1, 8, 2, 4, 2, groups)
    return tiled.permute(0, 6, 1, 2, 4, 3, 5).contiguous().reshape(groups, out_features)


def unpack_nunchaku_scales(packed: torch.Tensor) -> torch.Tensor:
    """Inverse of :func:`pack_nunchaku_scales`, used by CPU validation."""
    _require_shape(packed.ndim == 2, "packed scales must be [groups, out_features]")
    groups, out_features = packed.shape
    tiled = packed.contiguous().reshape(out_features // 128, groups, 1, 8, 4, 2, 2)
    return tiled.permute(0, 2, 3, 5, 4, 6, 1).contiguous().reshape(out_features, groups)


def pack_nunchaku_lowrank(weight: torch.Tensor, *, down: bool) -> torch.Tensor:
    """Pack a physical-rank FP16/BF16 low-rank projection."""
    _require_shape(weight.ndim == 2, "low-rank weight must be 2D")
    _require_shape(weight.dtype in (torch.float16, torch.bfloat16), "low-rank weight must be FP16/BF16")
    pack_n = pack_k = 16
    if down:
        rank, channels = weight.shape
        _require_shape(rank % pack_n == 0 and channels % pack_k == 0, "down projection is not tile aligned")
        rank_packs, channel_packs = rank // pack_n, channels // pack_k
        tiled = weight.reshape(rank_packs, pack_n, channel_packs, pack_k).permute(2, 0, 1, 3)
    else:
        channels, rank = weight.shape
        _require_shape(channels % pack_n == 0 and rank % pack_k == 0, "up projection is not tile aligned")
        channel_packs, rank_packs = channels // pack_n, rank // pack_k
        tiled = weight.reshape(channel_packs, pack_n, rank_packs, pack_k).permute(0, 2, 1, 3)
    tiled = tiled.reshape(channel_packs, rank_packs, 2, 8, 1, 2, 4, 2)
    return tiled.permute(0, 1, 3, 6, 2, 5, 4, 7).contiguous().reshape(channels, rank)


def unpack_nunchaku_lowrank(packed: torch.Tensor, *, down: bool) -> torch.Tensor:
    """Inverse of :func:`pack_nunchaku_lowrank`, used by CPU validation."""
    _require_shape(packed.ndim == 2, "packed low-rank weight must be 2D")
    channels, rank = packed.shape
    channel_packs, rank_packs = channels // 16, rank // 16
    tiled = packed.contiguous().reshape(channel_packs, rank_packs, 8, 4, 2, 2, 1, 2)
    tiled = tiled.permute(0, 1, 4, 2, 6, 5, 3, 7).contiguous().reshape(
        channel_packs, rank_packs, 16, 16
    )
    if down:
        return tiled.permute(1, 2, 0, 3).contiguous().reshape(rank, channels)
    return tiled.permute(0, 2, 1, 3).contiguous().reshape(channels, rank)


def _expand_weight_scales_to_g64(
    scales: torch.Tensor,
    in_features: int,
    source_group_size: int,
) -> torch.Tensor:
    source_group_size = in_features if source_group_size <= 0 else source_group_size
    _require_shape(in_features % NUNCHAKU_GROUP_SIZE == 0, "in_features must be divisible by 64")
    _require_shape(
        source_group_size >= NUNCHAKU_GROUP_SIZE
        and source_group_size % NUNCHAKU_GROUP_SIZE == 0,
        "Nunchaku requires weight group size 64 or an integer multiple of 64",
    )
    expected = math.ceil(in_features / source_group_size)
    _require_shape(scales.shape[1] == expected, "weight scale shape does not match group_size")
    starts = torch.arange(0, in_features, NUNCHAKU_GROUP_SIZE)
    source_index = torch.div(starts, source_group_size, rounding_mode="floor")
    return scales.index_select(1, source_index).contiguous()


def _has_runtime_correction(state: dict[str, Any]) -> bool:
    correction = state.get("correction")
    if not isinstance(correction, dict):
        return False
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
    return any(key in correction and correction[key] is not None for key in runtime_keys)


@torch.no_grad()
def prepare_nunchaku_state(
    state: dict[str, Any],
    compute_dtype: torch.dtype,
    *,
    allow_activation_group_remap: bool = False,
) -> dict[str, Any]:
    """Convert one H-SVDQuant state to a packed-only Nunchaku runtime state."""
    _require_shape(int(state["bits"]) == 4, "Nunchaku backend requires W4 checkpoints")
    _require_shape(int(state["activation_bits"]) == 4, "Nunchaku backend requires A4 checkpoints")
    _require_shape(compute_dtype in (torch.float16, torch.bfloat16), "Nunchaku requires float16 or bfloat16")
    _require_shape(not _has_runtime_correction(state), "runtime correction modules are not supported by Nunchaku")

    in_features = int(state["in_features"])
    out_features = int(state["out_features"])
    activation_group_size = int(state.get("activation_group_size", 0))
    if activation_group_size != NUNCHAKU_GROUP_SIZE and not allow_activation_group_remap:
        raise ValueError(
            "Nunchaku quantizes activations in groups of 64, but this checkpoint uses "
            f"group {activation_group_size or in_features}; pass allow_activation_group_remap=True "
            "for an explicit A-g64 inference remap or recalibrate with --activation-group-size 64"
        )

    codes = state["codes"].to(device="cpu", dtype=torch.int8).contiguous()
    scales = _expand_weight_scales_to_g64(
        state["scales"].to(device="cpu", dtype=compute_dtype).contiguous(),
        in_features,
        int(state["group_size"]),
    )
    rank = int(state["l1"].shape[1])
    physical_rank = max(NUNCHAKU_RANK_TILE, math.ceil(max(rank, 1) / NUNCHAKU_RANK_TILE) * NUNCHAKU_RANK_TILE)
    down = torch.zeros((physical_rank, in_features), dtype=compute_dtype)
    up = torch.zeros((out_features, physical_rank), dtype=compute_dtype)
    d = state["d"].to(device="cpu", dtype=compute_dtype).contiguous()
    if rank:
        # Nunchaku computes the low-rank down projection before applying its
        # fused input smoothing. Absorb D^-1 into L1 to preserve
        # (X / D) @ L1 while avoiding a separate smoothed activation tensor.
        l1_input = state["l1"].to(device="cpu", dtype=compute_dtype) / d[:, None]
        down[:rank].copy_(l1_input.T)
        up[:, :rank].copy_(state["l2"].to(device="cpu", dtype=compute_dtype).T)

    bias = state.get("bias")
    bias_tensor = torch.zeros(out_features, dtype=compute_dtype)
    if bias is not None:
        bias_tensor.copy_(bias.to(device="cpu", dtype=compute_dtype))
    return {
        "in_features": in_features,
        "out_features": out_features,
        "logical_rank": rank,
        "physical_rank": physical_rank,
        "source_activation_group_size": activation_group_size,
        "runtime_activation_group_size": NUNCHAKU_GROUP_SIZE,
        "qweight": pack_nunchaku_weight(codes),
        "wscales": pack_nunchaku_scales(scales),
        "bias": pack_nunchaku_scales(bias_tensor[:, None]).reshape(-1),
        "smooth": pack_nunchaku_scales(d[:, None]).reshape(-1),
        "proj_down": pack_nunchaku_lowrank(down, down=True),
        "proj_up": pack_nunchaku_lowrank(up, down=False),
    }


class NunchakuHSVQuantLinear(nn.Module):
    """H-SVDQuant linear backed by Nunchaku's fused true W4A4 CUDA kernels."""

    def __init__(self, packed: dict[str, Any], compute_dtype: torch.dtype) -> None:
        super().__init__()
        try:
            from nunchaku.models.linear import SVDQW4A4Linear
        except Exception as exc:
            raise RuntimeError(
                "runtime_backend=nunchaku requires a Nunchaku wheel matching Python and PyTorch"
            ) from exc

        self.in_features = int(packed["in_features"])
        self.out_features = int(packed["out_features"])
        self.logical_rank = int(packed["logical_rank"])
        self.physical_rank = int(packed["physical_rank"])
        self.source_activation_group_size = int(packed["source_activation_group_size"])
        self.activation_group_size = int(packed["runtime_activation_group_size"])
        self.bits = 4
        self.activation_bits = 4
        self.runtime_backend = "nunchaku"
        kernel = SVDQW4A4Linear(
            self.in_features,
            self.out_features,
            rank=self.physical_rank,
            bias=True,
            precision="int4",
            act_unsigned=False,
            torch_dtype=compute_dtype,
            device="cpu",
        )
        tensors = {
            "qweight": packed["qweight"],
            "wscales": packed["wscales"],
            "bias": packed["bias"],
            "smooth_factor": packed["smooth"],
            "smooth_factor_orig": packed["smooth"],
            "proj_down": packed["proj_down"],
            "proj_up": packed["proj_up"],
        }
        for name, value in tensors.items():
            parameter = getattr(kernel, name)
            _require_shape(parameter.shape == value.shape, f"Nunchaku {name} shape mismatch: {parameter.shape} != {value.shape}")
            parameter.copy_(value)
            parameter.requires_grad_(False)
        self.kernel = kernel

    @classmethod
    def from_state(
        cls,
        state: dict[str, Any],
        compute_dtype: torch.dtype,
        *,
        allow_activation_group_remap: bool = False,
    ) -> "NunchakuHSVQuantLinear":
        packed = prepare_nunchaku_state(
            state,
            compute_dtype,
            allow_activation_group_remap=allow_activation_group_remap,
        )
        return cls(packed, compute_dtype)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.device.type != "cuda":
            raise RuntimeError("Nunchaku W4A4 inference requires a CUDA tensor")
        if inputs.dtype not in (torch.float16, torch.bfloat16):
            raise TypeError("Nunchaku W4A4 inference requires float16 or bfloat16 activations")
        rows = inputs.numel() // self.in_features
        output = self.kernel(inputs.reshape(1, rows, self.in_features))
        return output.reshape(*inputs.shape[:-1], self.out_features)

    def packed_storage_bytes(self) -> int:
        return sum(tensor.numel() * tensor.element_size() for tensor in self.state_dict().values())

    def extra_repr(self) -> str:
        remap = self.source_activation_group_size != self.activation_group_size
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"rank={self.logical_rank} (physical={self.physical_rank}), A-group=64, remap={remap}"
        )


def build_nunchaku_linear(
    state: dict[str, Any],
    compute_dtype: torch.dtype,
    *,
    allow_activation_group_remap: bool = False,
) -> NunchakuHSVQuantLinear:
    return NunchakuHSVQuantLinear.from_state(
        state,
        compute_dtype,
        allow_activation_group_remap=allow_activation_group_remap,
    )
