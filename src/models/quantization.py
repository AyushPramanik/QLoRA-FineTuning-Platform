"""BitsAndBytes quantization configuration factory.

Centralises the construction of ``BitsAndBytesConfig`` objects so that all
training and inference code uses the same validated quantization settings.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class QuantType(str, Enum):
    """Supported 4-bit quantization data types."""

    NF4 = "nf4"    # NormalFloat4 — recommended for QLoRA
    FP4 = "fp4"    # FloatingPoint4


@dataclass
class QuantizationSpec:
    """User-facing specification for model quantization.

    Attributes:
        bits:               Number of bits (4 or 8).
        quant_type:         4-bit quantisation type (NF4 or FP4).
        compute_dtype:      dtype used for the dequantized matmul computation.
        use_double_quant:   Enable double quantisation to further reduce memory.
    """

    bits: int = 4
    quant_type: QuantType = QuantType.NF4
    compute_dtype: str = "bfloat16"
    use_double_quant: bool = True


def build_bnb_config(spec: QuantizationSpec) -> Any:
    """Build a ``BitsAndBytesConfig`` from a ``QuantizationSpec``.

    Args:
        spec: The quantization specification.

    Returns:
        A ``transformers.BitsAndBytesConfig`` object ready for use in
        ``AutoModelForCausalLM.from_pretrained``.

    Raises:
        ValueError: If ``spec.bits`` is not 4 or 8.
    """
    # TODO: Import and construct transformers.BitsAndBytesConfig.
    # TODO: Add support for AWQ and GPTQ quantization backends.
    # TODO: Validate that bitsandbytes is installed and its CUDA extensions are compiled.

    if spec.bits not in (4, 8):
        raise ValueError(f"Only 4-bit and 8-bit quantization supported, got {spec.bits}.")

    raise NotImplementedError("build_bnb_config is not yet implemented.")


def estimate_vram_gb(
    num_parameters: int,
    bits: int = 4,
    overhead_factor: float = 1.2,
) -> float:
    """Estimate VRAM usage for a quantized model.

    Simple formula: ``(num_params * bits / 8) * overhead_factor`` bytes,
    converted to gigabytes.

    Args:
        num_parameters:  Total number of model parameters.
        bits:            Quantization bits (4 or 8).
        overhead_factor: Multiplier to account for activations and optimizer state.

    Returns:
        Estimated VRAM in gigabytes.
    """
    bytes_per_param = bits / 8.0
    total_bytes     = num_parameters * bytes_per_param * overhead_factor
    return total_bytes / (1024**3)
