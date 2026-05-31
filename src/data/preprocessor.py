"""Dataset preprocessing — tokenisation, prompt formatting, and packing.

Transforms raw dataset dictionaries into tokenized tensors ready for the
training loop.  Supports instruction-following and conversational formats.
"""

from __future__ import annotations

from typing import Any, Callable

from src.utils.logging import get_logger

logger = get_logger(__name__)


class DataPreprocessor:
    """Applies tokenization and prompt formatting to a HF dataset.

    Args:
        tokenizer:       A loaded ``PreTrainedTokenizer``.
        prompt_template: A callable that converts a raw sample dict into a
                         formatted prompt string.
        max_length:      Maximum token sequence length (sequences are truncated).
        num_workers:     Number of CPU workers for ``.map()`` calls.
        pack_sequences:  If ``True``, pack multiple short examples into a single
                         ``max_length`` sequence to improve GPU utilisation.
    """

    def __init__(
        self,
        tokenizer: Any,
        prompt_template: Callable[[dict[str, Any]], str],
        *,
        max_length: int = 2048,
        num_workers: int = 4,
        pack_sequences: bool = False,
    ) -> None:
        self._tokenizer        = tokenizer
        self._prompt_template  = prompt_template
        self._max_length       = max_length
        self._num_workers      = num_workers
        self._pack_sequences   = pack_sequences

    def process(self, dataset: Any) -> Any:
        """Apply full preprocessing pipeline to a ``Dataset``.

        Steps:
          1. Apply prompt template to produce a ``text`` column.
          2. Tokenize the ``text`` column.
          3. Create ``labels`` by copying ``input_ids`` and masking padding.
          4. (Optional) Pack short sequences.

        Args:
            dataset: A HF ``Dataset`` with raw feature columns.

        Returns:
            A tokenized HF ``Dataset`` with ``input_ids``, ``attention_mask``,
            and ``labels`` columns.
        """
        # TODO: Implement using dataset.map with batched=True.
        # TODO: Remove columns that are not needed after tokenization.
        # TODO: Log the final dataset size and average sequence length.
        raise NotImplementedError("DataPreprocessor.process is not yet implemented.")

    def _tokenize_batch(self, batch: dict[str, list[Any]]) -> dict[str, list[Any]]:
        """Tokenize a batch of formatted text strings.

        Args:
            batch: A dict with a ``"text"`` key containing a list of strings.

        Returns:
            A dict with ``input_ids``, ``attention_mask``, and ``labels``.
        """
        # TODO: Implement tokenizer call with padding and truncation.
        # TODO: Set label positions corresponding to padding tokens to -100.
        raise NotImplementedError("DataPreprocessor._tokenize_batch is not yet implemented.")
