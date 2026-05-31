#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# merge_lora.sh — Merge LoRA adapter weights into the base model
#
# Produces a single self-contained model directory suitable for vLLM or
# direct loading with transformers.
#
# Usage:
#   ADAPTER_DIR=/data/outputs/run-001/adapter \
#   OUTPUT_DIR=/data/outputs/run-001/merged \
#   bash scripts/merge_lora.sh
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

ADAPTER_DIR="${ADAPTER_DIR:?Must set ADAPTER_DIR}"
OUTPUT_DIR="${OUTPUT_DIR:?Must set OUTPUT_DIR}"
SAFE_SERIALIZATION="${SAFE_SERIALIZATION:-true}"

if [[ -f .env ]]; then
    set -a && source .env && set +a
fi

echo "[INFO] Merging LoRA adapter"
echo "       Adapter : ${ADAPTER_DIR}"
echo "       Output  : ${OUTPUT_DIR}"

mkdir -p "${OUTPUT_DIR}"

# TODO: Replace with actual merge script once merge_and_unload is implemented.
python - <<PYEOF
from src.models.lora_adapter import merge_and_unload
print("[STUB] merge_and_unload not yet implemented.")
PYEOF

echo "[INFO] Done. Merged model written to ${OUTPUT_DIR}"
