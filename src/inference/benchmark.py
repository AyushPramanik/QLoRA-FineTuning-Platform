"""Inference benchmarking — throughput and latency measurement.

Measures tokens-per-second, time-to-first-token (TTFT), and latency
percentiles for a given model and serving backend.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from statistics import mean, median, stdev
from typing import Any, Callable

from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class BenchmarkResult:
    """Aggregated results from an inference benchmark run.

    Attributes:
        total_requests:       Total number of inference requests issued.
        successful_requests:  Number of requests that completed without error.
        total_output_tokens:  Sum of generated tokens across all requests.
        total_wall_time_sec:  Wall-clock duration of the benchmark.
        throughput_tps:       Tokens per second (output tokens / wall time).
        latency_mean_ms:      Mean end-to-end latency in milliseconds.
        latency_p50_ms:       Median latency in milliseconds.
        latency_p95_ms:       95th-percentile latency in milliseconds.
        latency_p99_ms:       99th-percentile latency in milliseconds.
        ttft_mean_ms:         Mean time-to-first-token in milliseconds.
    """

    total_requests: int = 0
    successful_requests: int = 0
    total_output_tokens: int = 0
    total_wall_time_sec: float = 0.0
    throughput_tps: float = 0.0
    latency_mean_ms: float = 0.0
    latency_p50_ms: float = 0.0
    latency_p95_ms: float = 0.0
    latency_p99_ms: float = 0.0
    ttft_mean_ms: float = 0.0
    errors: list[str] = field(default_factory=list)


@dataclass
class BenchmarkConfig:
    """Configuration for a benchmark run.

    Attributes:
        num_requests:     Total requests to send.
        concurrency:      Number of concurrent requests in flight.
        max_new_tokens:   Tokens to generate per request.
        prompt_length:    Approximate input prompt length in tokens.
        warmup_requests:  Requests to discard before recording measurements.
    """

    num_requests: int = 100
    concurrency: int = 1
    max_new_tokens: int = 256
    prompt_length: int = 128
    warmup_requests: int = 5


class InferenceBenchmark:
    """Runs latency and throughput benchmarks against an inference backend.

    Args:
        generate_fn: A callable ``(prompt: str) -> str`` representing the backend.
        config:      Benchmark parameters.
    """

    def __init__(
        self,
        generate_fn: Callable[[str], str],
        config: BenchmarkConfig | None = None,
    ) -> None:
        self._generate_fn = generate_fn
        self._config      = config or BenchmarkConfig()

    def run(self) -> BenchmarkResult:
        """Execute the benchmark and return aggregated results.

        Returns:
            A ``BenchmarkResult`` with all measured statistics.
        """
        # TODO: Implement concurrent request dispatch using asyncio or ThreadPoolExecutor.
        # TODO: Record per-request latency and token counts.
        # TODO: Compute percentiles using statistics.quantiles.
        # TODO: Log progress every N requests.

        logger.info(
            "benchmark_start",
            num_requests=self._config.num_requests,
            concurrency=self._config.concurrency,
        )
        raise NotImplementedError("InferenceBenchmark.run is not yet implemented.")

    def _synthetic_prompt(self) -> str:
        """Generate a synthetic prompt of approximately the configured length."""
        # TODO: Use a proper synthetic text generator rather than repetition.
        return ("The quick brown fox jumped over the lazy dog. " * self._config.prompt_length)[:500]
