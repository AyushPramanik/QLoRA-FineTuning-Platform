"""Fully Sharded Data Parallel (FSDP) configuration and utilities.

PyTorch FSDP shards model parameters, gradients, and optimizer state across
GPUs, enabling training of models that exceed single-GPU VRAM.

References:
    - https://pytorch.org/docs/stable/fsdp.html
    - https://engineering.fb.com/2021/07/15/open-source/fsdp/
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class ShardingStrategy(str, Enum):
    """FSDP sharding strategies mapped to PyTorch enum values."""

    FULL_SHARD      = "FULL_SHARD"        # Shard params + gradients + optimizer
    SHARD_GRAD_OP   = "SHARD_GRAD_OP"     # Shard gradients + optimizer only
    NO_SHARD        = "NO_SHARD"          # DDP-equivalent (no sharding)
    HYBRID_SHARD    = "HYBRID_SHARD"      # Shard within node, replicate across


@dataclass
class FSDPConfig:
    """Configuration for FSDP-based distributed training.

    Attributes:
        sharding_strategy:         How model state is distributed across ranks.
        cpu_offload:               Offload parameters and gradients to CPU RAM.
        mixed_precision:           Enable bf16 mixed precision (recommended).
        backward_prefetch:         Prefetch next layer's params during backward.
        activation_checkpointing:  Recompute activations during backward to save memory.
        auto_wrap_policy:          Module class pattern for automatic FSDP wrapping.
    """

    sharding_strategy: ShardingStrategy = ShardingStrategy.FULL_SHARD
    cpu_offload: bool = False
    mixed_precision: bool = True
    backward_prefetch: bool = True
    activation_checkpointing: bool = True
    auto_wrap_policy: str = "transformer_auto_wrap_policy"


def build_fsdp_plugin(config: FSDPConfig) -> Any:
    """Construct an Accelerate ``FullyShardedDataParallelPlugin`` from config.

    Args:
        config: ``FSDPConfig`` controlling sharding behaviour.

    Returns:
        An ``accelerate.utils.FullyShardedDataParallelPlugin`` instance.
    """
    # TODO: Implement using accelerate.utils.FullyShardedDataParallelPlugin.
    # TODO: Map ShardingStrategy enum to torch.distributed.fsdp.ShardingStrategy.
    # TODO: Set up activation checkpointing via apply_activation_checkpointing.
    raise NotImplementedError("build_fsdp_plugin is not yet implemented.")


def wrap_model_with_fsdp(model: Any, config: FSDPConfig) -> Any:
    """Wrap a model with FSDP for distributed training.

    Args:
        model:  A bare ``nn.Module`` or PEFT model.
        config: FSDP settings.

    Returns:
        The FSDP-wrapped model.
    """
    # TODO: Implement using torch.distributed.fsdp.FullyShardedDataParallel.
    # TODO: Apply transformer_auto_wrap_policy for decoder-layer granularity.
    # TODO: Validate that the process group is initialized before wrapping.
    raise NotImplementedError("wrap_model_with_fsdp is not yet implemented.")
