"""Tokenizer loading and chat-template management.

Handles the subtleties of padding tokens, EOS/BOS token alignment, and
prompt formatting across different model families.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.utils.logging import get_logger

logger = get_logger(__name__)

# Prompt templates supported by the platform.
PROMPT_TEMPLATES: dict[str, str] = {
    "alpaca": (
        "Below is an instruction that describes a task, paired with an input that provides "
        "further context. Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n{output}"
    ),
    "alpaca_no_input": (
        "Below is an instruction that describes a task. Write a response that appropriately "
        "completes the request.\n\n### Instruction:\n{instruction}\n\n### Response:\n{output}"
    ),
    "chatml": "<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n{assistant}<|im_end|>",
}


@dataclass
class TokenizerOutput:
    """Tokenized batch ready for training or inference.

    Attributes:
        input_ids:      Token ID tensor of shape ``(batch, seq_len)``.
        attention_mask: Binary mask tensor of shape ``(batch, seq_len)``.
        labels:         Target token IDs with padding positions set to ``-100``.
    """

    input_ids: Any       # torch.Tensor
    attention_mask: Any  # torch.Tensor
    labels: Any          # torch.Tensor


class TokenizerManager:
    """Loads and configures a Hugging Face tokenizer for training or inference.

    Ensures correct padding-side, pad token assignment, and chat-template
    registration for models that don't ship one by default.

    Args:
        model_name_or_path: HF Hub model ID or local directory.
        max_length:         Hard cap on sequence length.
        padding_side:       Which side to pad (``"right"`` for training, ``"left"`` for generation).
        cache_dir:          Local cache directory for the tokenizer files.
        token:              HF API token for gated models.
    """

    def __init__(
        self,
        model_name_or_path: str | Path,
        *,
        max_length: int = 2048,
        padding_side: str = "right",
        cache_dir: str | Path | None = None,
        token: str | None = None,
    ) -> None:
        self._model_name_or_path = str(model_name_or_path)
        self._max_length          = max_length
        self._padding_side        = padding_side
        self._cache_dir           = str(cache_dir) if cache_dir else None
        self._token               = token
        self._tokenizer: Any      = None  # Lazy-loaded.

        # TODO: Validate that padding_side is one of "left" or "right".

    def load(self) -> Any:
        """Load the tokenizer from the Hub or local path.

        Returns:
            A ``PreTrainedTokenizerFast`` instance.
        """
        # TODO: Implement using transformers.AutoTokenizer.from_pretrained.
        # TODO: Handle missing pad tokens (set to eos_token if absent).
        # TODO: Warn if the tokenizer vocabulary size differs from the model's embed matrix.
        raise NotImplementedError("TokenizerManager.load is not yet implemented.")

    def encode(
        self,
        texts: list[str],
        *,
        truncation: bool = True,
        return_tensors: str = "pt",
    ) -> TokenizerOutput:
        """Tokenize a batch of text strings.

        Args:
            texts:          List of raw text strings to tokenize.
            truncation:     Whether to truncate to ``max_length``.
            return_tensors: Framework string (``"pt"`` for PyTorch).

        Returns:
            A ``TokenizerOutput`` with ``input_ids``, ``attention_mask``, and ``labels``.
        """
        # TODO: Implement and mask labels at padding positions (set to -100).
        raise NotImplementedError("TokenizerManager.encode is not yet implemented.")

    def apply_template(self, sample: dict[str, str], template_name: str = "alpaca") -> str:
        """Apply a named prompt template to a dataset sample.

        Args:
            sample:        Dictionary of field values (e.g., instruction, input, output).
            template_name: Key into ``PROMPT_TEMPLATES``.

        Returns:
            A formatted prompt string.

        Raises:
            KeyError: If ``template_name`` is not found in ``PROMPT_TEMPLATES``.
        """
        # TODO: Add Jinja2 template support for more complex prompt structures.
        if template_name not in PROMPT_TEMPLATES:
            raise KeyError(f"Unknown template '{template_name}'. Available: {list(PROMPT_TEMPLATES)}")
        return PROMPT_TEMPLATES[template_name].format(**sample)
