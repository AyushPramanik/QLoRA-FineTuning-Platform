# QLoRA Fine-Tuning Platform

A production-grade platform for fine-tuning large language models using QLoRA (Quantized Low-Rank Adaptation). Built for scalability, reproducibility, and operational excellence.

[![CI — Lint](https://github.com/ayushpramanik/qlora-finetuning-platform/actions/workflows/lint.yml/badge.svg)](https://github.com/ayushpramanik/qlora-finetuning-platform/actions/workflows/lint.yml)
[![CI — Test](https://github.com/ayushpramanik/qlora-finetuning-platform/actions/workflows/test.yml/badge.svg)](https://github.com/ayushpramanik/qlora-finetuning-platform/actions/workflows/test.yml)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

---

## Overview

This platform provides a complete, opinionated infrastructure for fine-tuning LLMs at scale. It handles the full lifecycle — from data preparation through training, evaluation, and serving — with first-class support for:

- **QLoRA** training with configurable rank, alpha, and target modules
- **Distributed training** via FSDP and DeepSpeed ZeRO-3
- **4-bit quantization** with bitsandbytes NF4 / FP4
- **Production serving** with FastAPI + vLLM continuous batching
- **Experiment tracking** with Weights & Biases
- **Kubernetes-native** deployment with autoscaling

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        QLoRA Platform                               │
├──────────────┬──────────────┬──────────────┬───────────────────────┤
│  Data Layer  │ Training     │  Inference   │   Serving             │
│              │              │              │                       │
│  datasets/   │  FSDP        │  vLLM        │  FastAPI              │
│  HF Hub      │  DeepSpeed   │  TGI         │  Autoscaling          │
│  S3          │  QLoRA       │  Benchmarks  │  Load Balancing       │
└──────────────┴──────────────┴──────────────┴───────────────────────┘
         │              │              │               │
         └──────────────┴──────────────┴───────────────┘
                              │
                    ┌─────────▼──────────┐
                    │   W&B | Prometheus │
                    │   Grafana          │
                    └────────────────────┘
```

See [docs/architecture.md](docs/architecture.md) for full Mermaid diagrams.

## Quick Start

### Prerequisites

- Python 3.12+
- CUDA 12.1+
- Docker & Docker Compose (for containerized workflows)
- `make`

### Installation

```bash
# Clone the repository
git clone https://github.com/ayushpramanik/qlora-finetuning-platform.git
cd qlora-finetuning-platform

# Set up environment
cp .env.example .env
# Edit .env with your HF_TOKEN, WANDB_API_KEY, etc.

# Install dependencies
make install-dev

# Install pre-commit hooks
make pre-commit-install
```

### Running a Fine-Tuning Job

```bash
# Single-GPU training
make train CONFIG=configs/training/qlora_default.yaml

# Multi-GPU with FSDP
make train-multi-gpu

# Check training status in W&B
# https://wandb.ai/<your-entity>/<your-project>
```

### Starting the Inference Server

```bash
# Local development
make serve

# Production (Docker)
docker compose up api
```

### Running Tests

```bash
make test         # fast unit tests
make test-cov     # with coverage report
make test-gpu     # GPU-required tests
```

## Repository Structure

```
qlora-finetuning-platform/
├── configs/                    # Hydra/OmegaConf configuration files
│   ├── model/                  #   Model architecture configs
│   ├── training/               #   Training hyperparameter configs
│   ├── datasets/               #   Dataset preprocessing configs
│   └── inference/              #   Inference server configs
├── datasets/                   # Dataset scripts and manifests
├── deployment/
│   ├── docker/                 # Dockerfiles (train + inference)
│   └── kubernetes/             # K8s manifests (job, deployment, HPA)
├── docs/                       # Architecture & operational docs
├── experiments/                # Per-experiment configs and notes
├── monitoring/                 # Prometheus + Grafana config
├── notebooks/                  # Exploratory analysis notebooks
├── scripts/                    # Shell entrypoints for common tasks
├── src/
│   ├── data/                   # Data loading and preprocessing
│   ├── evaluation/             # Evaluation harness (perplexity, MT-Bench)
│   ├── inference/              # Inference engines and benchmarking
│   ├── models/                 # Model loading, quantization, LoRA
│   ├── serving/                # FastAPI application
│   ├── training/               # Training loop, QLoRA, FSDP, DeepSpeed
│   └── utils/                  # Shared utilities (logging, config, seed)
├── tests/                      # Pytest test suite
└── .github/workflows/          # CI/CD pipelines
```

## Configuration

The platform uses [Hydra](https://hydra.cc) for configuration composition. Configs are located under `configs/` and can be overridden from the command line:

```bash
# Override training hyperparameters
python -m src.training.train \
    training=qlora_default \
    training.lora_r=64 \
    training.learning_rate=2e-4 \
    model=llama3_8b
```

## Supported Models

| Model | Parameters | VRAM (4-bit) | Status |
|-------|-----------|-------------|--------|
| LLaMA-3 8B | 8B | ~6 GB | Planned |
| LLaMA-3 70B | 70B | ~40 GB | Planned |
| Mistral 7B | 7B | ~5 GB | Planned |
| Mixtral 8x7B | 47B | ~26 GB | Planned |
| Gemma 7B | 7B | ~5 GB | Planned |

## Documentation

| Document | Description |
|----------|-------------|
| [Architecture](docs/architecture.md) | System design and component diagrams |
| [QLoRA Explained](docs/qlora-explained.md) | Theory behind QLoRA |
| [Deployment](docs/deployment.md) | Docker and Kubernetes deployment guide |
| [Distributed Training](docs/distributed-training.md) | FSDP and DeepSpeed setup |

## Development

```bash
# Lint and format
make lint
make format

# Type checking
make typecheck

# Run all checks
make check
```

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feat/your-feature`)
3. Commit your changes following [Conventional Commits](https://www.conventionalcommits.org/)
4. Push to the branch and open a Pull Request

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.
