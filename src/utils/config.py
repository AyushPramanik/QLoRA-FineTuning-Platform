"""Configuration management using OmegaConf / Hydra.

Central place for loading, validating, and merging YAML configuration
files.  All config objects are typed dataclasses to ensure full IDE
support and runtime validation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf


@dataclass
class LoraConfig:
    """LoRA adapter hyper-parameters."""

    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: list[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    bias: str = "none"
    task_type: str = "CAUSAL_LM"


@dataclass
class QuantizationConfig:
    """BitsAndBytes quantization settings."""

    load_in_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_use_double_quant: bool = True


@dataclass
class ModelConfig:
    """Base model selection and loading options."""

    name_or_path: str = "meta-llama/Meta-Llama-3-8B"
    revision: str = "main"
    trust_remote_code: bool = False
    torch_dtype: str = "bfloat16"
    attn_implementation: str = "flash_attention_2"
    lora: LoraConfig = field(default_factory=LoraConfig)
    quantization: QuantizationConfig = field(default_factory=QuantizationConfig)


@dataclass
class TrainingConfig:
    """Training loop hyper-parameters."""

    output_dir: str = "/data/outputs"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-4
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.03
    weight_decay: float = 0.001
    max_grad_norm: float = 1.0
    max_seq_length: int = 2048
    save_steps: int = 200
    eval_steps: int = 200
    logging_steps: int = 10
    fp16: bool = False
    bf16: bool = True
    gradient_checkpointing: bool = True
    group_by_length: bool = True
    report_to: list[str] = field(default_factory=lambda: ["wandb"])
    run_name: str = "qlora-run"


@dataclass
class DatasetConfig:
    """Dataset loading and preprocessing settings."""

    name: str = "tatsu-lab/alpaca"
    split: str = "train"
    val_split_size: float = 0.05
    prompt_template: str = "alpaca"
    max_samples: int | None = None
    preprocessing_num_workers: int = 4


@dataclass
class PlatformConfig:
    """Top-level configuration that composes all sub-configs."""

    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)


# ── Loader helpers ─────────────────────────────────────────────────────────────


def load_config(path: str | Path) -> DictConfig:
    """Load a YAML config file into an OmegaConf ``DictConfig``.

    Args:
        path: Path to the YAML configuration file.

    Returns:
        A merged ``DictConfig`` object.

    Raises:
        FileNotFoundError: If the config file does not exist.
    """
    # TODO: Add Hydra composition for multi-file config merging.
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    return OmegaConf.load(config_path)


def merge_configs(*configs: DictConfig | dict[str, Any]) -> DictConfig:
    """Merge multiple OmegaConf configs (later configs override earlier).

    Args:
        *configs: One or more configs to merge in order.

    Returns:
        A single merged ``DictConfig``.
    """
    # TODO: Validate against the PlatformConfig schema after merge.
    return OmegaConf.merge(*configs)


def to_dict(config: DictConfig) -> dict[str, Any]:
    """Convert an OmegaConf config to a plain Python dict.

    Args:
        config: The config to convert.

    Returns:
        A plain dict representation.
    """
    return OmegaConf.to_container(config, resolve=True)  # type: ignore[return-value]
