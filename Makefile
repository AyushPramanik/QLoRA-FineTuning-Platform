.PHONY: help install install-dev lint format typecheck test test-gpu test-cov \
        clean docker-build docker-up docker-down train serve benchmark \
        merge-lora pre-commit-install pre-commit-run docs-serve

PYTHON      := python3.12
PIP         := $(PYTHON) -m pip
PYTEST      := $(PYTHON) -m pytest
RUFF        := $(PYTHON) -m ruff
MYPY        := $(PYTHON) -m mypy
SRC_DIRS    := src tests
DOCKER_COMP := docker compose

# ─── Formatting ───────────────────────────────────────────────────────────────
BOLD   := \033[1m
RESET  := \033[0m
GREEN  := \033[32m
YELLOW := \033[33m

help: ## Show this help message
	@echo "$(BOLD)QLoRA Fine-Tuning Platform — Available Commands$(RESET)"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| sort \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  $(GREEN)%-25s$(RESET) %s\n", $$1, $$2}'
	@echo ""

# ─── Installation ─────────────────────────────────────────────────────────────
install: ## Install production dependencies
	$(PIP) install -e .

install-dev: ## Install all dependencies including dev extras
	$(PIP) install -e ".[all]"

install-inference: ## Install inference extras (vLLM)
	$(PIP) install -e ".[inference]"

# ─── Code Quality ─────────────────────────────────────────────────────────────
lint: ## Run ruff linter
	$(RUFF) check $(SRC_DIRS)

format: ## Auto-format code with ruff
	$(RUFF) format $(SRC_DIRS)
	$(RUFF) check --fix $(SRC_DIRS)

typecheck: ## Run mypy type checker
	$(MYPY) src/

check: lint typecheck ## Run all static analysis checks

# ─── Testing ──────────────────────────────────────────────────────────────────
test: ## Run fast unit tests (no GPU required)
	$(PYTEST) -m "not gpu and not slow" -v

test-gpu: ## Run GPU tests (requires CUDA)
	$(PYTEST) -m "gpu" -v

test-cov: ## Run tests with coverage report
	$(PYTEST) --cov=src --cov-report=html --cov-report=term-missing

test-all: ## Run the full test suite
	$(PYTEST) -v

# ─── Pre-commit ───────────────────────────────────────────────────────────────
pre-commit-install: ## Install pre-commit hooks
	pre-commit install
	pre-commit install --hook-type commit-msg

pre-commit-run: ## Run pre-commit on all files
	pre-commit run --all-files

# ─── Docker ───────────────────────────────────────────────────────────────────
docker-build: ## Build all Docker images
	$(DOCKER_COMP) build

docker-build-train: ## Build the training Docker image
	docker build -f deployment/docker/Dockerfile.train -t qlora-train:latest .

docker-build-inference: ## Build the inference Docker image
	docker build -f deployment/docker/Dockerfile.inference -t qlora-inference:latest .

docker-up: ## Start services with docker compose
	$(DOCKER_COMP) up -d

docker-down: ## Stop and remove docker compose services
	$(DOCKER_COMP) down

docker-logs: ## Tail docker compose logs
	$(DOCKER_COMP) logs -f

# ─── Training ─────────────────────────────────────────────────────────────────
train: ## Launch a training run (set CONFIG=configs/training/qlora_default.yaml)
	@CONFIG ?= configs/training/qlora_default.yaml; \
	bash scripts/train.sh $$CONFIG

train-multi-gpu: ## Launch distributed multi-GPU training
	bash scripts/launch_multi_gpu.sh

merge-lora: ## Merge LoRA adapters into base model (set ADAPTER_DIR=...)
	bash scripts/merge_lora.sh

# ─── Serving ──────────────────────────────────────────────────────────────────
serve: ## Start the FastAPI inference server
	uvicorn src.serving.api:app --host 0.0.0.0 --port 8000 --reload

serve-prod: ## Start the server in production mode
	uvicorn src.serving.api:app --host 0.0.0.0 --port 8000 --workers 4

# ─── Benchmarking ─────────────────────────────────────────────────────────────
benchmark: ## Run inference benchmarks
	bash scripts/benchmark.sh

# ─── Docs ─────────────────────────────────────────────────────────────────────
docs-serve: ## Serve documentation locally with mkdocs
	mkdocs serve

# ─── Cleanup ──────────────────────────────────────────────────────────────────
clean: ## Remove all generated artifacts
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache"   -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache"   -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "htmlcov"       -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "coverage.xml"  -delete 2>/dev/null || true
	@echo "$(GREEN)Cleaned up generated artifacts.$(RESET)"

clean-models: ## Remove downloaded model weights (irreversible!)
	@echo "$(YELLOW)WARNING: This will delete all cached model weights.$(RESET)"
	@read -p "Continue? [y/N] " confirm && [ "$$confirm" = "y" ] && \
		rm -rf $$HF_HOME || echo "Aborted."
