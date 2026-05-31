#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# train.sh — Single-GPU QLoRA training launcher
#
# Usage:
#   bash scripts/train.sh [CONFIG_PATH] [EXTRA_ARGS...]
#
# Examples:
#   bash scripts/train.sh
#   bash scripts/train.sh configs/training/qlora_default.yaml
#   bash scripts/train.sh configs/training/qlora_default.yaml training.lora_r=64
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

CONFIG="${1:-configs/training/qlora_default.yaml}"
shift || true   # Allow empty extra args

# ── Validate environment ──────────────────────────────────────────────────────
if [[ -z "${HF_TOKEN:-}" ]]; then
    echo "[WARN] HF_TOKEN is not set. Gated models will not be accessible."
fi

if ! python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    echo "[ERROR] CUDA is not available. Aborting."
    exit 1
fi

GPU_COUNT=$(python -c "import torch; print(torch.cuda.device_count())")
echo "[INFO] Found ${GPU_COUNT} GPU(s)."

# ── Load environment ──────────────────────────────────────────────────────────
if [[ -f .env ]]; then
    echo "[INFO] Loading .env"
    # shellcheck source=/dev/null
    set -a && source .env && set +a
fi

# ── Launch training ───────────────────────────────────────────────────────────
echo "[INFO] Starting training with config: ${CONFIG}"
echo "[INFO] Extra args: $*"

# TODO: Replace with Hydra-composed entrypoint once implemented.
python -m src.training.train \
    --config "${CONFIG}" \
    "$@"
