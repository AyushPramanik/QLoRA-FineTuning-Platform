"""Perplexity evaluation on held-out text corpora.

Perplexity = exp(average negative log-likelihood per token).  Lower values
indicate that the model assigns higher probability to the reference text.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class PerplexityResult:
    """Result of a perplexity evaluation run.

    Attributes:
        perplexity:   Final perplexity score.
        loss:         Mean negative log-likelihood (natural log).
        num_tokens:   Total number of tokens evaluated.
        dataset_name: Name of the evaluation dataset.
    """

    perplexity: float
    loss: float
    num_tokens: int
    dataset_name: str


class PerplexityEvaluator:
    """Computes perplexity of a model over a text dataset.

    Args:
        model:        A loaded causal LM.
        tokenizer:    The corresponding tokenizer.
        stride:       Sliding-window stride for long-document evaluation.
                      A stride equal to ``max_length`` gives non-overlapping chunks.
        max_length:   Maximum chunk size in tokens.
        batch_size:   Batch size for forward passes.
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        *,
        stride: int = 512,
        max_length: int = 1024,
        batch_size: int = 4,
    ) -> None:
        self._model      = model
        self._tokenizer  = tokenizer
        self._stride     = stride
        self._max_length = max_length
        self._batch_size = batch_size

    def evaluate(self, texts: list[str], dataset_name: str = "custom") -> PerplexityResult:
        """Compute perplexity over a list of text strings.

        Args:
            texts:        List of raw text strings to evaluate.
            dataset_name: Human-readable name for logging.

        Returns:
            A ``PerplexityResult`` with the computed perplexity and supporting stats.
        """
        # TODO: Tokenize texts into overlapping windows using stride.
        # TODO: Batch forward passes through the model with torch.no_grad().
        # TODO: Accumulate the cross-entropy loss weighted by non-padding tokens.
        # TODO: Exponentiate the mean loss to get perplexity.

        logger.info("perplexity_eval_start", dataset=dataset_name, num_texts=len(texts))
        raise NotImplementedError("PerplexityEvaluator.evaluate is not yet implemented.")

    def evaluate_dataset(self, dataset_name: str = "wikitext-103") -> PerplexityResult:
        """Convenience method: load a standard HF dataset and evaluate perplexity.

        Args:
            dataset_name: HF Hub dataset identifier (e.g., ``"wikitext-103"``).

        Returns:
            A ``PerplexityResult``.
        """
        # TODO: Load dataset via datasets.load_dataset and extract the text column.
        raise NotImplementedError("PerplexityEvaluator.evaluate_dataset is not yet implemented.")
