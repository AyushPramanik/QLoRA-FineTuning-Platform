"""Metrics registry and primitive metric computations.

Provides a central ``MetricsRegistry`` for discovering available evaluation
tasks, and standalone functions for common NLP metrics.
"""

from __future__ import annotations

from typing import Any, Callable


class MetricsRegistry:
    """Registry of named evaluation metric functions.

    Metric functions have the signature::

        fn(predictions: list[str], references: list[str]) -> dict[str, float]

    Usage::

        registry = MetricsRegistry()
        registry.register("bleu", compute_bleu)
        scores = registry.compute("bleu", predictions=[...], references=[...])
    """

    def __init__(self) -> None:
        self._registry: dict[str, Callable[..., dict[str, float]]] = {}

    def register(self, name: str, fn: Callable[..., dict[str, float]]) -> None:
        """Register a metric function under a given name.

        Args:
            name: Unique metric identifier.
            fn:   Callable implementing the metric computation.

        Raises:
            ValueError: If ``name`` is already registered.
        """
        if name in self._registry:
            raise ValueError(f"Metric '{name}' is already registered.")
        self._registry[name] = fn

    def compute(self, name: str, **kwargs: Any) -> dict[str, float]:
        """Compute a registered metric.

        Args:
            name:    The registered metric name.
            **kwargs: Arguments forwarded to the metric function.

        Returns:
            A dictionary of metric name → score.

        Raises:
            KeyError: If ``name`` is not registered.
        """
        if name not in self._registry:
            raise KeyError(f"Metric '{name}' not found. Available: {sorted(self._registry)}")
        return self._registry[name](**kwargs)

    def available(self) -> list[str]:
        """Return the list of registered metric names."""
        return sorted(self._registry.keys())


# ── Primitive metric functions (stubs) ────────────────────────────────────────


def compute_rouge(
    predictions: list[str],
    references: list[str],
) -> dict[str, float]:
    """Compute ROUGE-1, ROUGE-2, and ROUGE-L F1 scores.

    Args:
        predictions: Model-generated strings.
        references:  Ground-truth reference strings.

    Returns:
        Dict with keys ``rouge1``, ``rouge2``, ``rougeL``.
    """
    # TODO: Implement using the ``rouge_score`` library.
    raise NotImplementedError("compute_rouge is not yet implemented.")


def compute_bleu(
    predictions: list[str],
    references: list[str],
) -> dict[str, float]:
    """Compute corpus-level BLEU score.

    Args:
        predictions: Model-generated strings (tokenized).
        references:  Reference strings (tokenized).

    Returns:
        Dict with key ``bleu``.
    """
    # TODO: Implement using sacrebleu.
    raise NotImplementedError("compute_bleu is not yet implemented.")


def compute_exact_match(
    predictions: list[str],
    references: list[str],
) -> dict[str, float]:
    """Compute exact-match accuracy.

    Args:
        predictions: Model predictions (normalised).
        references:  Gold references (normalised).

    Returns:
        Dict with key ``exact_match`` (fraction of correct answers).
    """
    if len(predictions) != len(references):
        raise ValueError("predictions and references must have the same length.")
    correct = sum(p.strip() == r.strip() for p, r in zip(predictions, references))
    return {"exact_match": correct / len(predictions)}
