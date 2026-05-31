"""Unit tests for training module scaffolding.

These tests verify structural correctness (class existence, interface
signatures, config defaults) without requiring GPU resources.
"""

from __future__ import annotations

import pytest

from src.training.checkpointing import CheckpointManager, CheckpointMetadata
from src.training.deepspeed import DeepSpeedConfig, ZeroStage
from src.training.fsdp import FSDPConfig, ShardingStrategy
from src.training.qlora import get_qlora_default_modules
from src.utils.config import LoraConfig, TrainingConfig


class TestTrainingConfig:
    """Tests for TrainingConfig defaults and validation."""

    def test_default_values(self) -> None:
        config = TrainingConfig()
        assert config.num_train_epochs == 3
        assert config.learning_rate == pytest.approx(2e-4)
        assert config.bf16 is True
        assert config.gradient_checkpointing is True

    def test_report_to_includes_wandb(self) -> None:
        config = TrainingConfig()
        assert "wandb" in config.report_to


class TestLoraConfig:
    """Tests for LoRA adapter configuration defaults."""

    def test_default_rank(self) -> None:
        config = LoraConfig()
        assert config.r == 16

    def test_default_alpha_is_2x_rank(self) -> None:
        config = LoraConfig()
        assert config.lora_alpha == config.r * 2

    def test_default_target_modules_nonempty(self) -> None:
        config = LoraConfig()
        assert len(config.target_modules) > 0


class TestQLoRAModuleLookup:
    """Tests for the architecture → target-modules lookup table."""

    def test_llama_modules_returned(self) -> None:
        modules = get_qlora_default_modules("llama")
        assert "q_proj" in modules
        assert "v_proj" in modules

    def test_mistral_modules_returned(self) -> None:
        modules = get_qlora_default_modules("mistral")
        assert len(modules) > 0

    def test_unknown_architecture_raises(self) -> None:
        with pytest.raises(KeyError):
            get_qlora_default_modules("unknown_arch")


class TestFSDPConfig:
    """Tests for FSDP configuration defaults."""

    def test_default_strategy_is_full_shard(self) -> None:
        config = FSDPConfig()
        assert config.sharding_strategy == ShardingStrategy.FULL_SHARD

    def test_activation_checkpointing_enabled_by_default(self) -> None:
        config = FSDPConfig()
        assert config.activation_checkpointing is True


class TestDeepSpeedConfig:
    """Tests for DeepSpeed ZeRO configuration defaults."""

    def test_default_stage_is_3(self) -> None:
        config = DeepSpeedConfig()
        assert config.zero_stage == ZeroStage.STAGE_3

    def test_bf16_enabled_by_default(self) -> None:
        config = DeepSpeedConfig()
        assert config.bf16 is True


class TestCheckpointMetadata:
    """Tests for CheckpointMetadata serialisation helpers."""

    def test_metadata_construction(self) -> None:
        meta = CheckpointMetadata(
            global_step=100,
            epoch=1.5,
            eval_loss=1.23,
            wall_time_sec=3600.0,
            model_name="llama3",
        )
        assert meta.global_step == 100
        assert meta.eval_loss == pytest.approx(1.23)

    def test_metadata_none_eval_loss(self) -> None:
        meta = CheckpointMetadata(
            global_step=50,
            epoch=0.5,
            eval_loss=None,
            wall_time_sec=1800.0,
            model_name="mistral",
        )
        assert meta.eval_loss is None


class TestCheckpointManagerInit:
    """Tests for CheckpointManager initialisation."""

    def test_creates_with_defaults(self, tmp_output_dir: object) -> None:
        manager = CheckpointManager(root_dir=tmp_output_dir)  # type: ignore[arg-type]
        assert manager._max_keep == 3
        assert manager._save_best_only is False
