# Distributed Training

This platform supports two distributed training strategies: **FSDP** (for PyTorch-native sharding) and **DeepSpeed ZeRO** (for optimizer and parameter offloading).

---

## Choosing a Strategy

| | FSDP | DeepSpeed ZeRO-3 |
|--|------|-----------------|
| Framework | PyTorch native | Requires DeepSpeed |
| Sharding | Params + grads + optimizer | Params + grads + optimizer |
| CPU offload | Yes (limited) | Yes (extensive, NVMe) |
| Ease of setup | Moderate | Complex |
| Throughput | High | Slightly lower with offload |
| Best for | Multi-GPU single node | Very large models, multi-node |

**Rule of thumb**: Use FSDP for 70B models on 4–8 GPUs. Use DeepSpeed ZeRO-3 for 70B+ with CPU offload or for multi-node clusters.

---

## FSDP

### Quick Start

```bash
# 4-GPU training
bash scripts/launch_multi_gpu.sh 4 configs/training/fsdp.yaml

# Or manually with torchrun
torchrun --standalone --nproc_per_node=4 \
    -m src.training.train \
    --config configs/training/fsdp.yaml
```

### Configuration

Edit `configs/training/fsdp.yaml`:

```yaml
fsdp:
  sharding_strategy: FULL_SHARD      # Shard params + grads + optimizer
  cpu_offload: false
  mixed_precision: true              # BF16 mixed precision
  activation_checkpointing: true    # Recompute activations to save memory
```

### VRAM Requirements (LLaMA-3 70B, 8× A100-80G)

| Config | Per-GPU VRAM |
|--------|-------------|
| No sharding | OOM |
| FSDP FULL_SHARD + NF4 | ~38 GB |
| FSDP FULL_SHARD + NF4 + activation ckpt | ~22 GB |

---

## DeepSpeed ZeRO

### Quick Start

```bash
# Generate a DeepSpeed config
python -c "
from src.training.deepspeed import DeepSpeedConfig, ZeroStage, save_deepspeed_config
save_deepspeed_config(DeepSpeedConfig(zero_stage=ZeroStage.STAGE_3), 'configs/training/ds_config.json')
"

# Launch with DeepSpeed
deepspeed --num_gpus=8 \
    -m src.training.train \
    --config configs/training/qlora_default.yaml \
    --deepspeed configs/training/ds_config.json
```

### ZeRO Stage Comparison

```
ZeRO-1: Shard optimizer states only
         Memory savings: ~4× (optimizer)

ZeRO-2: Shard optimizer states + gradients
         Memory savings: ~8× (optimizer + gradients)

ZeRO-3: Shard everything (params + optimizer + gradients)
         Memory savings: ~64× theoretical
         Trade-off: communication overhead
```

---

## Multi-Node Training

For training across multiple nodes, set these environment variables on all nodes:

```bash
export MASTER_ADDR=<node-0-ip>
export MASTER_PORT=29500
export NCCL_IB_DISABLE=0         # Enable InfiniBand if available
export NCCL_SOCKET_IFNAME=eth0   # Network interface

# Node 0 (rank 0)
torchrun \
    --nnodes=4 \
    --nproc_per_node=8 \
    --node_rank=0 \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    -m src.training.train --config configs/training/fsdp.yaml
```

---

## Gradient Checkpointing

All training configs enable gradient checkpointing by default. This recomputes activations during the backward pass instead of storing them, trading compute for memory:

```
Memory reduction: ~60–70% for activations
Compute overhead: ~33% additional forward passes
```

Enable via config:

```yaml
gradient_checkpointing: true
```

---

## Troubleshooting

### OOM during training
1. Reduce `per_device_train_batch_size` and increase `gradient_accumulation_steps`.
2. Enable `gradient_checkpointing: true`.
3. Switch from SHARD_GRAD_OP to FULL_SHARD (FSDP).
4. Enable `cpu_offload: true` (DeepSpeed ZeRO-3).

### NCCL timeout errors
1. Check that all nodes can communicate on the MASTER_PORT.
2. Increase `NCCL_TIMEOUT` environment variable.
3. Disable InfiniBand with `NCCL_IB_DISABLE=1` if not available.

### Uneven GPU utilisation
1. Enable `group_by_length: true` to reduce padding waste.
2. Set `dataloader_num_workers` equal to the number of CPUs per GPU.
