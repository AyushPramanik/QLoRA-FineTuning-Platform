"""Unit tests for the TokenizerManager and prompt template utilities."""

from __future__ import annotations

import pytest

from src.models.tokenizer import PROMPT_TEMPLATES, TokenizerManager


class TestPromptTemplates:
    """Tests for built-in prompt template formatting."""

    def test_alpaca_template_exists(self) -> None:
        assert "alpaca" in PROMPT_TEMPLATES

    def test_chatml_template_exists(self) -> None:
        assert "chatml" in PROMPT_TEMPLATES

    def test_alpaca_format_produces_string(self) -> None:
        template = PROMPT_TEMPLATES["alpaca"]
        result = template.format(
            instruction="What is 2+2?",
            input="",
            output="4",
        )
        assert "What is 2+2?" in result
        assert "### Response:" in result


class TestTokenizerManagerInit:
    """Tests for TokenizerManager construction and configuration."""

    def test_init_stores_model_name(self) -> None:
        manager = TokenizerManager("test/model", max_length=512)
        assert manager._model_name_or_path == "test/model"

    def test_init_stores_max_length(self) -> None:
        manager = TokenizerManager("test/model", max_length=1024)
        assert manager._max_length == 1024

    def test_default_padding_side_is_right(self) -> None:
        manager = TokenizerManager("test/model")
        assert manager._padding_side == "right"

    def test_left_padding_side_accepted(self) -> None:
        manager = TokenizerManager("test/model", padding_side="left")
        assert manager._padding_side == "left"


class TestTokenizerManagerApplyTemplate:
    """Tests for apply_template method."""

    def test_unknown_template_raises_key_error(self) -> None:
        manager = TokenizerManager("test/model")
        with pytest.raises(KeyError, match="no_such_template"):
            manager.apply_template(
                {"instruction": "hi", "input": "", "output": "hello"},
                template_name="no_such_template",
            )

    def test_apply_alpaca_template(self) -> None:
        manager = TokenizerManager("test/model")
        result = manager.apply_template(
            {"instruction": "Say hello.", "input": "", "output": "Hello!"},
            template_name="alpaca",
        )
        assert "Say hello." in result
        assert "Hello!" in result

    def test_load_raises_not_implemented(self) -> None:
        manager = TokenizerManager("test/model")
        with pytest.raises(NotImplementedError):
            manager.load()
