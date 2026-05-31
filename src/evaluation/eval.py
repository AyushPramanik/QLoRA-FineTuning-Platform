"""Main evaluation runner — orchestrates evaluation tasks over a model.

Provides ``EvaluationRunner`` which wires together a model, tokenizer, and
one or more evaluation tasks (perplexity, MT-Bench, custom benchmarks).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class EvaluationResult:
    """Aggregated results across all evaluation tasks.

    Attributes:
        model_name:  Identifier of the evaluated model.
        task_scores: Mapping from task name to its score dictionary.
        total_time_sec: Wall-clock time for the full evaluation run.
    """

    model_name: str
    task_scores: dict[str, dict[str, float]] = field(default_factory=dict)
    total_time_sec: float = 0.0


class EvaluationRunner:
    """Orchestrates evaluation across multiple tasks.

    Args:
        model:      A loaded (and optionally fine-tuned) causal LM.
        tokenizer:  The matching tokenizer.
        output_dir: Directory to write evaluation reports.
        tasks:      List of task names to evaluate (``"perplexity"``, ``"mt_bench"``).
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        output_dir: str | Path,
        tasks: list[str] | None = None,
    ) -> None:
        self._model      = model
        self._tokenizer  = tokenizer
        self._output_dir = Path(output_dir)
        self._tasks      = tasks or ["perplexity"]

        # TODO: Validate that all task names are registered in MetricsRegistry.

    def run(self) -> EvaluationResult:
        """Run all configured evaluation tasks.

        Returns:
            An ``EvaluationResult`` with per-task scores.
        """
        # TODO: Dispatch each task to the appropriate evaluator class.
        # TODO: Collect results and write a JSON report to output_dir.
        # TODO: Log results to W&B as summary metrics.

        logger.info("evaluation_run_start", tasks=self._tasks)
        raise NotImplementedError("EvaluationRunner.run is not yet implemented.")

    def main(self) -> None:
        """CLI entrypoint for running evaluation from the command line."""
        # TODO: Parse CLI args (model_path, tasks, output_dir) with typer.
        raise NotImplementedError


def main() -> None:
    """Module-level CLI entrypoint used by the ``qlora-eval`` script."""
    EvaluationRunner(model=None, tokenizer=None, output_dir="/tmp").main()
