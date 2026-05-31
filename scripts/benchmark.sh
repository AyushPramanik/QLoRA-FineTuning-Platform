#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# benchmark.sh — Inference throughput and latency benchmarking
#
# Usage:
#   API_URL=http://localhost:8000 bash scripts/benchmark.sh
#   NUM_REQUESTS=200 CONCURRENCY=8 bash scripts/benchmark.sh
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

API_URL="${API_URL:-http://localhost:8000}"
NUM_REQUESTS="${NUM_REQUESTS:-100}"
CONCURRENCY="${CONCURRENCY:-1}"
MAX_TOKENS="${MAX_TOKENS:-256}"
OUTPUT_DIR="${OUTPUT_DIR:-./benchmark_results}"

if [[ -f .env ]]; then
    set -a && source .env && set +a
fi

echo "[INFO] Starting benchmark"
echo "       Target       : ${API_URL}"
echo "       Requests     : ${NUM_REQUESTS}"
echo "       Concurrency  : ${CONCURRENCY}"
echo "       Max tokens   : ${MAX_TOKENS}"

mkdir -p "${OUTPUT_DIR}"

# ── Health check ──────────────────────────────────────────────────────────────
if ! curl -sf "${API_URL}/health" > /dev/null; then
    echo "[ERROR] API at ${API_URL} is not reachable. Start the server first."
    exit 1
fi

# ── Run benchmark ─────────────────────────────────────────────────────────────
# TODO: Replace with actual benchmark runner once InferenceBenchmark is implemented.
python - <<PYEOF
from src.inference.benchmark import BenchmarkConfig, InferenceBenchmark
config = BenchmarkConfig(
    num_requests=${NUM_REQUESTS},
    concurrency=${CONCURRENCY},
    max_new_tokens=${MAX_TOKENS},
)
print("[STUB] InferenceBenchmark.run() not yet implemented.")
PYEOF

echo "[INFO] Benchmark complete. Results in ${OUTPUT_DIR}"
