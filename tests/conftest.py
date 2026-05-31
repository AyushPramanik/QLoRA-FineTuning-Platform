"""Shared pytest fixtures for the QLoRA platform test suite."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest


# ── Path fixtures ──────────────────────────────────────────────────────────────


@pytest.fixture()
def tmp_output_dir(tmp_path: Path) -> Path:
    """A temporary directory for test outputs."""
    output = tmp_path / "outputs"
    output.mkdir()
    return output


@pytest.fixture()
def sample_config_path(tmp_path: Path) -> Path:
    """Write a minimal YAML config and return its path."""
    config = tmp_path / "config.yaml"
    config.write_text(
        "model:\n  name_or_path: test-model\n"
        "training:\n  num_train_epochs: 1\n"
        "dataset:\n  name: tatsu-lab/alpaca\n"
    )
    return config


# ── Mock model / tokenizer fixtures ───────────────────────────────────────────


@pytest.fixture()
def mock_tokenizer() -> MagicMock:
    """A minimal mock tokenizer for unit tests (no GPU required)."""
    tokenizer = MagicMock()
    tokenizer.pad_token_id = 0
    tokenizer.eos_token_id = 2
    tokenizer.model_max_length = 2048
    tokenizer.padding_side = "right"
    tokenizer.return_value = {
        "input_ids": [[1, 2, 3, 4, 5]],
        "attention_mask": [[1, 1, 1, 1, 1]],
    }
    return tokenizer


@pytest.fixture()
def mock_model() -> MagicMock:
    """A mock causal LM that records calls without loading GPU resources."""
    model = MagicMock()
    model.config.model_type = "llama"
    model.num_parameters.return_value = 8_000_000_000
    return model


@pytest.fixture()
def sample_texts() -> list[str]:
    """A small list of text strings for tokenizer/perplexity tests."""
    return [
        "The capital of France is Paris.",
        "Quantum computing uses qubits instead of classical bits.",
        "Fine-tuning large language models requires careful hyperparameter selection.",
    ]


@pytest.fixture()
def sample_dataset_records() -> list[dict[str, str]]:
    """A small list of Alpaca-format instruction records."""
    return [
        {
            "instruction": "What is the capital of France?",
            "input": "",
            "output": "The capital of France is Paris.",
        },
        {
            "instruction": "Summarize quantum computing in one sentence.",
            "input": "",
            "output": "Quantum computing uses qubits to perform computations exponentially faster than classical bits for certain problems.",
        },
    ]
