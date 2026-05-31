"""Model loading with optional 4-bit / 8-bit quantization.

This module provides a clean interface for loading pretrained language models
from the Hugging Face Hub or local paths, with full support for bitsandbytes
quantization and device mapping.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class LoadedModel:
    """Container for a loaded model and its associated metadata.

    Attributes:
        model:      The underlying ``PreTrainedModel`` instance.
        config:     Raw HF model config dict.
        device_map: The resolved device map used during loading.
        is_quantized: Whether the model was loaded with quantization.
    """

    model: Any  # transformers.PreTrainedModel
    config: dict[str, Any]
    device_map: str | dict[str, Any]
    is_quantized: bool


class ModelLoader:
    """Loads pretrained models with configurable quantization and device mapping.

    Usage::

        loader = ModelLoader(cache_dir="/data/huggingface_cache")
        result = loader.load("meta-llama/Meta-Llama-3-8B", quantization_config=...)

    Args:
        cache_dir: Directory for caching downloaded model weights.
        token:     Hugging Face API token for gated models.
    """

    def __init__(
        self,
        cache_dir: str | Path | None = None,
        token: str | None = None,
    ) -> None:
        self._cache_dir = Path(cache_dir) if cache_dir else None
        self._token = token

        # TODO: Initialize disk-space check to warn on low cache capacity.

    def load(
        self,
        model_name_or_path: str | Path,
        *,
        quantization_config: Any | None = None,
        device_map: str | dict[str, Any] = "auto",
        torch_dtype: Any | None = None,
        attn_implementation: str = "eager",
        trust_remote_code: bool = False,
    ) -> LoadedModel:
        """Load a pretrained model.

        Args:
            model_name_or_path: HF Hub model ID or local path.
            quantization_config: A ``BitsAndBytesConfig`` or similar quantization spec.
            device_map:          Device placement strategy (``"auto"``, ``"cuda:0"``, etc.).
            torch_dtype:         Compute dtype override (e.g., ``torch.bfloat16``).
            attn_implementation: Attention kernel to use (``"eager"``, ``"flash_attention_2"``).
            trust_remote_code:   Whether to allow remote code execution.

        Returns:
            A ``LoadedModel`` dataclass.

        Raises:
            ValueError: If the model path does not exist and the HF token is missing.
        """
        # TODO: Implement actual model loading using transformers.AutoModelForCausalLM.from_pretrained.
        # TODO: Validate VRAM availability before loading and warn if insufficient.
        # TODO: Support multi-node tensor parallelism via device_map=split.

        logger.info(
            "model_load_requested",
            model=str(model_name_or_path),
            quantized=quantization_config is not None,
            device_map=device_map,
        )
        raise NotImplementedError("ModelLoader.load is not yet implemented.")

    def load_adapter(
        self,
        base_model: Any,
        adapter_path: str | Path,
        *,
        is_trainable: bool = False,
    ) -> Any:
        """Attach a PEFT LoRA adapter to an already-loaded base model.

        Args:
            base_model:   A loaded ``PreTrainedModel``.
            adapter_path: Path to the saved PEFT adapter directory.
            is_trainable: Whether the adapter weights should require gradients.

        Returns:
            The model with the adapter attached.
        """
        # TODO: Implement via peft.PeftModel.from_pretrained.
        # TODO: Validate adapter compatibility with the base model architecture.
        raise NotImplementedError("ModelLoader.load_adapter is not yet implemented.")
