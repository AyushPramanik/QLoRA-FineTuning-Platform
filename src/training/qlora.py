"""QLoRA-specific setup: 4-bit loading + LoRA attachment pipeline.

Combines quantization and LoRA adapter construction into a single,
atomic ``prepare_qlora_model`` function that training code calls once.
"""

from __future__ import annotations

from typing import Any

from src.models.lora_adapter import AdapterSpec, apply_lora
from src.models.quantization import QuantizationSpec, build_bnb_config
from src.utils.logging import get_logger

logger = get_logger(__name__)


def prepare_qlora_model(
    base_model: Any,
    *,
    adapter_spec: AdapterSpec | None = None,
    quant_spec: QuantizationSpec | None = None,
) -> Any:
    """Prepare a base model for QLoRA fine-tuning.

    Applies the following steps in order:
      1. Enable ``prepare_model_for_kbit_training`` for gradient-checkpointing
         compatibility with quantized layers.
      2. Attach LoRA adapters via ``apply_lora``.

    Args:
        base_model:   A model already loaded with ``BitsAndBytesConfig``.
        adapter_spec: LoRA adapter specification. Falls back to defaults.
        quant_spec:   Quantization spec (informational only at this stage).

    Returns:
        A ``PeftModel`` ready for training.
    """
    # TODO: Call peft.prepare_model_for_kbit_training(base_model).
    # TODO: Pass adapter_spec (or build a default one) to apply_lora.
    # TODO: Enable gradient checkpointing on the returned model.
    # TODO: Disable the KV-cache (incompatible with gradient checkpointing).

    _adapter_spec = adapter_spec or AdapterSpec()
    logger.info(
        "qlora_prepare",
        lora_r=_adapter_spec.r,
        lora_alpha=_adapter_spec.lora_alpha,
        target_modules=_adapter_spec.target_modules,
    )
    raise NotImplementedError("prepare_qlora_model is not yet implemented.")


def get_qlora_default_modules(architecture: str) -> list[str]:
    """Return the recommended LoRA target modules for a model architecture.

    Args:
        architecture: Model family string (e.g., ``"llama"``, ``"mistral"``).

    Returns:
        List of module name patterns to attach adapters to.

    Raises:
        KeyError: If the architecture is not in the lookup table.
    """
    # TODO: Expand this lookup table for Phi-3, Gemma, Qwen, etc.
    _MODULE_MAP: dict[str, list[str]] = {
        "llama":   ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "mistral": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "falcon":  ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"],
        "gpt_neox":["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"],
    }

    if architecture not in _MODULE_MAP:
        raise KeyError(
            f"Architecture '{architecture}' not in module map. "
            f"Known: {sorted(_MODULE_MAP.keys())}"
        )
    return _MODULE_MAP[architecture]
