#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# launch_multi_gpu.sh — Distributed multi-GPU training via torchrun
#
# Usage:
#   bash scripts/launch_multi_gpu.sh [NUM_GPUS] [CONFIG] [EXTRA_ARGS...]
#
# Examples:
#   bash scripts/launch_multi_gpu.sh 4
#   bash scripts/launch_multi_gpu.sh 8 configs/training/fsdp.yaml
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

NUM_GPUS="${1:-$(python -c "import torch; print(torch.cuda.device_count())")}"
CONFIG="${2:-configs/training/fsdp.yaml}"
shift 2 || true

MASTER_ADDR="${MASTER_ADDR:-localhost}"
MASTER_PORT="${MASTER_PORT:-29500}"

if [[ -f .env ]]; then
    set -a && source .env && set +a
fi

echo "[INFO] Launching distributed training on ${NUM_GPUS} GPUs"
echo "[INFO] Master: ${MASTER_ADDR}:${MASTER_PORT}"
echo "[INFO] Config: ${CONFIG}"

# ── torchrun launch ───────────────────────────────────────────────────────────
# TODO: Add --node_rank and --nnodes for multi-node training.
# TODO: Support deepspeed launcher as an alternative to torchrun.

torchrun \
    --standalone \
    --nproc_per_node="${NUM_GPUS}" \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    -m src.training.train \
    --config "${CONFIG}" \
    "$@"
