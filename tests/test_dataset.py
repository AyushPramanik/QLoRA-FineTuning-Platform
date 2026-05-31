"""Unit tests for data loading and preprocessing scaffolding."""

from __future__ import annotations

import pytest

from src.data.loader import DatasetLoader
from src.data.preprocessor import DataPreprocessor
from src.evaluation.metrics import (
    MetricsRegistry,
    compute_exact_match,
)
from src.utils.config import DatasetConfig


class TestDatasetConfig:
    """Tests for DatasetConfig defaults."""

    def test_default_name(self) -> None:
        config = DatasetConfig()
        assert config.name == "tatsu-lab/alpaca"

    def test_default_val_split(self) -> None:
        config = DatasetConfig()
        assert 0 < config.val_split_size < 1.0

    def test_default_preprocessing_workers(self) -> None:
        config = DatasetConfig()
        assert config.preprocessing_num_workers > 0


class TestDatasetLoaderInit:
    """Tests for DatasetLoader construction."""

    def test_init_with_config(self) -> None:
        config = DatasetConfig(name="tatsu-lab/alpaca")
        loader = DatasetLoader(config)
        assert loader._config.name == "tatsu-lab/alpaca"

    def test_load_raises_not_implemented(self) -> None:
        loader = DatasetLoader(DatasetConfig())
        with pytest.raises(NotImplementedError):
            loader.load()

    def test_load_local_unsupported_format_raises(self) -> None:
        loader = DatasetLoader(DatasetConfig())
        with pytest.raises(ValueError, match="Unsupported format"):
            loader.load_local("/tmp/data.csv", fmt="csv")

    def test_load_local_valid_formats_accepted(self) -> None:
        loader = DatasetLoader(DatasetConfig())
        for fmt in ("jsonl", "json", "parquet", "arrow"):
            with pytest.raises(NotImplementedError):
                loader.load_local("/tmp/data", fmt=fmt)


class TestMetricsRegistry:
    """Tests for the MetricsRegistry."""

    def test_register_and_list(self) -> None:
        registry = MetricsRegistry()
        registry.register("dummy", lambda **kw: {"score": 1.0})
        assert "dummy" in registry.available()

    def test_compute_registered_metric(self) -> None:
        registry = MetricsRegistry()
        registry.register("const", lambda **kw: {"value": 42.0})
        result = registry.compute("const")
        assert result["value"] == pytest.approx(42.0)

    def test_compute_unknown_metric_raises(self) -> None:
        registry = MetricsRegistry()
        with pytest.raises(KeyError):
            registry.compute("nonexistent")

    def test_double_register_raises(self) -> None:
        registry = MetricsRegistry()
        registry.register("m", lambda **kw: {})
        with pytest.raises(ValueError, match="already registered"):
            registry.register("m", lambda **kw: {})


class TestExactMatch:
    """Tests for the exact-match metric function."""

    def test_all_correct(self) -> None:
        result = compute_exact_match(["a", "b", "c"], ["a", "b", "c"])
        assert result["exact_match"] == pytest.approx(1.0)

    def test_none_correct(self) -> None:
        result = compute_exact_match(["x", "y"], ["a", "b"])
        assert result["exact_match"] == pytest.approx(0.0)

    def test_partial_correct(self) -> None:
        result = compute_exact_match(["a", "wrong"], ["a", "b"])
        assert result["exact_match"] == pytest.approx(0.5)

    def test_whitespace_stripped(self) -> None:
        result = compute_exact_match(["  a  "], ["a"])
        assert result["exact_match"] == pytest.approx(1.0)

    def test_mismatched_lengths_raise(self) -> None:
        with pytest.raises(ValueError):
            compute_exact_match(["a", "b"], ["a"])
