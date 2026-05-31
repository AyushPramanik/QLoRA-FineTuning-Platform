"""LoRA adapter construction and management.

Wraps PEFT's ``LoraConfig`` and ``get_peft_model`` with validation,
parameter counting helpers, and adapter merge utilities.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class AdapterSpec:
    """Specification for a LoRA adapter.

    Attributes:
        r:               LoRA rank. Higher rank = more parameters and capacity.
        lora_alpha:      Scaling factor. Typically set to 2 * r.
        lora_dropout:    Dropout probability applied inside LoRA layers.
        target_modules:  Names of attention/linear modules to attach adapters to.
        bias:            Which biases to train. One of ``"none"``, ``"all"``, or
                         ``"lora_only"``.
        task_type:       PEFT task type string (e.g., ``"CAUSAL_LM"``).
    """

    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: list[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"]
    )
    bias: str = "none"
    task_type: str = "CAUSAL_LM"


def build_peft_config(spec: AdapterSpec) -> Any:
    """Construct a PEFT ``LoraConfig`` from an ``AdapterSpec``.

    Args:
        spec: The adapter specification.

    Returns:
        A ``peft.LoraConfig`` instance.
    """
    # TODO: Implement using peft.LoraConfig.
    # TODO: Add DoRA (Weight-Decomposed Low-Rank Adaptation) toggle.
    # TODO: Support RSLoRA (Rank-Stabilised LoRA) scaling.
    raise NotImplementedError("build_peft_config is not yet implemented.")


def apply_lora(base_model: Any, spec: AdapterSpec) -> Any:
    """Attach LoRA adapters to a base model.

    Args:
        base_model: A loaded ``PreTrainedModel`` (possibly quantized).
        spec:       The adapter spec describing which modules to adapt.

    Returns:
        A ``PeftModel`` wrapping the base model.
    """
    # TODO: Call peft.get_peft_model with the built config.
    # TODO: Call model.print_trainable_parameters() and log the result.
    raise NotImplementedError("apply_lora is not yet implemented.")


def count_trainable_parameters(model: Any) -> tuple[int, int]:
    """Count trainable vs total parameters in a model.

    Args:
        model: A PyTorch ``nn.Module``.

    Returns:
        A tuple of ``(trainable_params, total_params)``.
    """
    # TODO: Implement via model.named_parameters() iteration.
    raise NotImplementedError("count_trainable_parameters is not yet implemented.")


def merge_and_unload(peft_model: Any, *, safe_serialization: bool = True) -> Any:
    """Merge LoRA weights into the base model and remove the adapter.

    Call this before saving a final model for inference.  The merged model
    behaves identically to the original base model but includes the fine-tuned
    weights in full precision.

    Args:
        peft_model:         The ``PeftModel`` to merge.
        safe_serialization: Save with safetensors format (recommended).

    Returns:
        The merged base ``PreTrainedModel`` without adapter wrappers.
    """
    # TODO: Implement via peft_model.merge_and_unload().
    # TODO: Call model.save_pretrained() with the merged weights.
    raise NotImplementedError("merge_and_unload is not yet implemented.")
