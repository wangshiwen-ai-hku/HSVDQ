"""Native packed inference backend for H-SVDQuant checkpoints.

The public packing format is deliberately independent from a particular CUDA
kernel.  New WxAx implementations can register against :class:`PackedQuantSpec`
without changing checkpoints or the model loader.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


SUPPORTED_STORAGE_BITS = (2, 4, 8)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


@dataclass(frozen=True)
class PackedQuantSpec:
    weight_bits: int
    activation_bits: int
    weight_group_size: int
    activation_group_size: int
    rank: int
    signed: bool = True

    @property
    def kernel_key(self) -> tuple[int, int, int, int]:
        return (
            self.weight_bits,
            self.activation_bits,
            self.weight_group_size,
            self.activation_group_size,
        )

    @property
    def name(self) -> str:
        return (
            f"w{self.weight_bits}a{self.activation_bits}_"
            f"gw{self.weight_group_size}_ga{self.activation_group_size}_r{self.rank}"
        )


@dataclass(frozen=True)
class KernelCapability:
    name: str
    min_compute_capability: tuple[int, int]
    max_compute_capability: tuple[int, int]
    max_rank: int


# W4A4 is implemented now. The registry and canonical bit packing are shared by
# later W8A8, W4A8, and W2A2 kernels rather than embedding policy in the loader.
_KERNEL_REGISTRY: dict[tuple[int, int, int, int], KernelCapability] = {
    (4, 4, 128, 128): KernelCapability(
        name="w4a4_g128_wmma_sm75_89",
        min_compute_capability=(7, 5),
        max_compute_capability=(8, 9),
        max_rank=8,
    ),
}


def available_kernel_keys() -> tuple[tuple[int, int, int, int], ...]:
    return tuple(sorted(_KERNEL_REGISTRY))


def resolve_kernel(spec: PackedQuantSpec) -> KernelCapability:
    capability = _KERNEL_REGISTRY.get(spec.kernel_key)
    if capability is None:
        supported = ", ".join(
            f"W{w}A{a}/gw{gw}/ga{ga}" for w, a, gw, ga in available_kernel_keys()
        )
        raise ValueError(
            f"no native kernel registered for {spec.name}; available kernels: {supported}"
        )
    _require(spec.signed, "the current native kernels require signed weights and activations")
    _require(0 < spec.rank <= capability.max_rank, f"{capability.name} supports rank 1..{capability.max_rank}")
    return capability


def pack_signed_codes(codes: torch.Tensor, bits: int) -> torch.Tensor:
    """Pack signed codes along the last dimension into canonical low-bit bytes."""
    _require(bits in SUPPORTED_STORAGE_BITS, f"storage bits must be one of {SUPPORTED_STORAGE_BITS}")
    _require(codes.ndim >= 1, "codes must have at least one dimension")
    values_per_byte = 8 // bits
    _require(
        codes.shape[-1] % values_per_byte == 0,
        f"last dimension must be divisible by {values_per_byte} for {bits}-bit packing",
    )
    values = codes.detach().to(device="cpu", dtype=torch.int16).contiguous()
    qmin, qmax = -(1 << (bits - 1)), (1 << (bits - 1)) - 1
    if values.numel():
        _require(int(values.min()) >= qmin and int(values.max()) <= qmax, f"{bits}-bit codes must be in [{qmin}, {qmax}]")
    grouped = values.reshape(*values.shape[:-1], -1, values_per_byte)
    packed = torch.zeros(grouped.shape[:-1], dtype=torch.int16)
    mask = (1 << bits) - 1
    for index in range(values_per_byte):
        packed.bitwise_or_((grouped[..., index] & mask) << (index * bits))
    return packed.to(torch.uint8).contiguous()


def unpack_signed_codes(packed: torch.Tensor, bits: int) -> torch.Tensor:
    """Unpack canonical bytes to signed int8 codes."""
    _require(bits in SUPPORTED_STORAGE_BITS, f"storage bits must be one of {SUPPORTED_STORAGE_BITS}")
    _require(packed.ndim >= 1, "packed codes must have at least one dimension")
    values_per_byte = 8 // bits
    raw = packed.detach().to(device="cpu", dtype=torch.int16).contiguous()
    mask = (1 << bits) - 1
    sign = 1 << (bits - 1)
    pieces = []
    for index in range(values_per_byte):
        value = (raw >> (index * bits)) & mask
        pieces.append(torch.where(value >= sign, value - (1 << bits), value))
    return torch.stack(pieces, dim=-1).reshape(*raw.shape[:-1], -1).to(torch.int8).contiguous()


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
def prepare_hsvdq_cuda_state(state: dict[str, Any], compute_dtype: torch.dtype) -> dict[str, Any]:
    """Convert one checkpoint layer to the backend's packed, device-agnostic state."""
    _require(compute_dtype in (torch.float16, torch.bfloat16), "hsvdq_cuda requires float16 or bfloat16")
    _require(not _has_runtime_correction(state), "hsvdq_cuda does not yet fuse runtime correction modules")

    in_features = int(state["in_features"])
    out_features = int(state["out_features"])
    weight_group_size = int(state.get("group_size", 0)) or in_features
    activation_group_size = int(state.get("activation_group_size", 0)) or in_features
    rank = int(state["l1"].shape[1])
    spec = PackedQuantSpec(
        weight_bits=int(state["bits"]),
        activation_bits=int(state["activation_bits"]),
        weight_group_size=weight_group_size,
        activation_group_size=activation_group_size,
        rank=rank,
    )
    capability = resolve_kernel(spec)
    _require(in_features % weight_group_size == 0, "in_features must be divisible by weight_group_size")
    _require(in_features % activation_group_size == 0, "in_features must be divisible by activation_group_size")
    _require(in_features % 128 == 0, f"{capability.name} requires in_features divisible by 128")
    _require(out_features % 8 == 0, f"{capability.name} requires out_features divisible by 8")

    codes = state["codes"].to(device="cpu", dtype=torch.int8).contiguous()
    scales = state["scales"].to(device="cpu", dtype=compute_dtype).contiguous()
    expected_groups = in_features // weight_group_size
    _require(tuple(codes.shape) == (out_features, in_features), "weight code shape does not match layer dimensions")
    _require(tuple(scales.shape) == (out_features, expected_groups), "weight scale shape does not match group_size")
    bias = torch.zeros(out_features, dtype=compute_dtype)
    if state.get("bias") is not None:
        bias.copy_(state["bias"].to(device="cpu", dtype=compute_dtype))

    return {
        "in_features": in_features,
        "out_features": out_features,
        "spec": spec,
        "kernel_name": capability.name,
        "qweight": pack_signed_codes(codes, spec.weight_bits),
        "wscales": scales,
        "smooth": state["d"].to(device="cpu", dtype=compute_dtype).contiguous(),
        "l1": state["l1"].to(device="cpu", dtype=compute_dtype).contiguous(),
        "l2": state["l2"].to(device="cpu", dtype=compute_dtype).contiguous(),
        "bias": bias,
    }


_EXTENSION: Any | None = None


def _load_extension() -> Any:
    global _EXTENSION
    if _EXTENSION is not None:
        return _EXTENSION
    if not torch.cuda.is_available():
        raise RuntimeError("hsvdq_cuda requires a CUDA device")
    from torch.utils.cpp_extension import load

    source_dir = Path(__file__).resolve().parent / "csrc" / "hsvdq_cuda"
    _EXTENSION = load(
        name="hsvdq_cuda_ext_v1",
        sources=[str(source_dir / "bindings.cpp"), str(source_dir / "w4a4_wmma.cu")],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "--use_fast_math", "-lineinfo"],
        verbose=os.environ.get("HSVDQ_CUDA_VERBOSE", "0") == "1",
    )
    return _EXTENSION


class HSVDCudaLinear(nn.Module):
    """Packed H-SVDQuant linear using the native CUDA kernel registry."""

    def __init__(self, packed: dict[str, Any], compute_dtype: torch.dtype) -> None:
        super().__init__()
        self.in_features = int(packed["in_features"])
        self.out_features = int(packed["out_features"])
        self.spec: PackedQuantSpec = packed["spec"]
        self.kernel_name = str(packed["kernel_name"])
        self.bits = self.spec.weight_bits
        self.activation_bits = self.spec.activation_bits
        self.group_size = self.spec.weight_group_size
        self.activation_group_size = self.spec.activation_group_size
        self.rank = self.spec.rank
        self.runtime_backend = "hsvdq_cuda"
        self.compute_dtype = compute_dtype
        for name in ("qweight", "wscales", "smooth", "l1", "l2", "bias"):
            self.register_buffer(name, packed[name], persistent=True)

    @classmethod
    def from_state(cls, state: dict[str, Any], compute_dtype: torch.dtype) -> "HSVDCudaLinear":
        return cls(prepare_hsvdq_cuda_state(state, compute_dtype), compute_dtype)

    def _validate_device(self, inputs: torch.Tensor) -> None:
        if inputs.device.type != "cuda":
            raise RuntimeError("hsvdq_cuda inference requires CUDA activations")
        if inputs.dtype not in (torch.float16, torch.bfloat16):
            raise TypeError("hsvdq_cuda requires float16 or bfloat16 activations")
        if inputs.dtype != self.wscales.dtype:
            raise TypeError(f"activation dtype {inputs.dtype} does not match packed scales {self.wscales.dtype}")
        capability = resolve_kernel(self.spec)
        compute_capability = torch.cuda.get_device_capability(inputs.device)
        if not capability.min_compute_capability <= compute_capability <= capability.max_compute_capability:
            raise RuntimeError(
                f"{capability.name} supports SM {capability.min_compute_capability} through "
                f"{capability.max_compute_capability}, got SM {compute_capability}"
            )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        self._validate_device(inputs)
        leading_shape = inputs.shape[:-1]
        flat = inputs.reshape(-1, self.in_features).contiguous()
        extension = _load_extension()
        output = extension.w4a4_forward(
            flat,
            self.qweight,
            self.wscales,
            self.smooth,
            self.l1,
            self.l2,
            self.bias,
            self.group_size,
        )
        return output.reshape(*leading_shape, self.out_features)

    def packed_storage_bytes(self) -> int:
        return sum(tensor.numel() * tensor.element_size() for tensor in self.buffers())

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"{self.spec.name}, kernel={self.kernel_name}"
        )


def build_hsvdq_cuda_linear(state: dict[str, Any], compute_dtype: torch.dtype) -> HSVDCudaLinear:
    return HSVDCudaLinear.from_state(state, compute_dtype)
