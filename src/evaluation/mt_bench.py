"""MT-Bench evaluation — multi-turn instruction following benchmark.

MT-Bench uses GPT-4 as a judge to score model responses on 80 multi-turn
questions across 8 categories.

References:
    - Zheng et al., 2023. "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena."
    - https://huggingface.co/spaces/lmsys/mt-bench
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.utils.logging import get_logger

logger = get_logger(__name__)

MT_BENCH_CATEGORIES = [
    "writing",
    "roleplay",
    "reasoning",
    "math",
    "coding",
    "extraction",
    "stem",
    "humanities",
]


@dataclass
class MTBenchResult:
    """Results from an MT-Bench evaluation run.

    Attributes:
        overall_score:      Mean score across all questions and turns (0–10 scale).
        category_scores:    Per-category mean scores.
        turn1_score:        Mean score on first-turn responses.
        turn2_score:        Mean score on second-turn responses (follow-up).
        num_questions:      Number of questions evaluated.
    """

    overall_score: float = 0.0
    category_scores: dict[str, float] = field(default_factory=dict)
    turn1_score: float = 0.0
    turn2_score: float = 0.0
    num_questions: int = 0


class MTBenchEvaluator:
    """Runs MT-Bench evaluation using GPT-4 as a judge.

    Args:
        model:          The model under evaluation (for generating responses).
        tokenizer:      The matching tokenizer.
        judge_model:    Name of the judge LLM (defaults to ``"gpt-4"``).
        questions_path: Path to the MT-Bench questions JSONL file.
        output_dir:     Directory for writing raw responses and scores.
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        judge_model: str = "gpt-4",
        questions_path: str | Path | None = None,
        output_dir: str | Path = "/tmp/mt_bench",
    ) -> None:
        self._model          = model
        self._tokenizer      = tokenizer
        self._judge_model    = judge_model
        self._questions_path = Path(questions_path) if questions_path else None
        self._output_dir     = Path(output_dir)

    def generate_responses(self) -> list[dict[str, Any]]:
        """Generate model responses for all MT-Bench questions.

        Returns:
            List of response dicts with ``question_id``, ``turn``, and ``choices``.
        """
        # TODO: Load questions from questions_path or the bundled default.
        # TODO: Run multi-turn generation for each question.
        # TODO: Save responses to output_dir/model_responses.jsonl.
        raise NotImplementedError("MTBenchEvaluator.generate_responses is not yet implemented.")

    def judge_responses(self, responses: list[dict[str, Any]]) -> MTBenchResult:
        """Use the judge LLM to score the generated responses.

        Args:
            responses: Response dicts produced by ``generate_responses``.

        Returns:
            An ``MTBenchResult`` with category and overall scores.
        """
        # TODO: For each response, call the judge LLM with the official MT-Bench prompt.
        # TODO: Parse the judge's numeric score from its response.
        # TODO: Aggregate per-category and overall scores.
        raise NotImplementedError("MTBenchEvaluator.judge_responses is not yet implemented.")

    def run(self) -> MTBenchResult:
        """Full pipeline: generate responses then judge them.

        Returns:
            An ``MTBenchResult``.
        """
        logger.info("mt_bench_start", judge=self._judge_model)
        responses = self.generate_responses()
        return self.judge_responses(responses)
