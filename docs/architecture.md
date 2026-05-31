# Architecture

This document describes the high-level system design of the QLoRA Fine-Tuning Platform.

---

## System Overview

```mermaid
graph TB
    subgraph Clients
        CLI[CLI / Scripts]
        API_CLIENT[REST Client]
        NB[Notebooks]
    end

    subgraph Data Layer
        HF_HUB[(HF Hub)]
        S3[(S3 / Object Store)]
        LOCAL_DATA[(Local Datasets)]
        DATA_LOADER[DatasetLoader]
        PREPROCESSOR[DataPreprocessor]
    end

    subgraph Training Layer
        TRAIN_ENTRY[train.py entrypoint]
        QLORA_PREP[QLoRA Prepare]
        MODEL_LOADER[ModelLoader]
        BNB_QUANT[BitsAndBytes 4-bit]
        LORA_ADAPT[LoRA Adapter]
        TRAINER[QLoRATrainer]
        FSDP[FSDP Strategy]
        DS[DeepSpeed ZeRO]
        CKPT[CheckpointManager]
    end

    subgraph Experiment Tracking
        WANDB[Weights & Biases]
    end

    subgraph Inference Layer
        GEN_ENGINE[GenerationEngine]
        VLLM[vLLM Server]
        BATCHER[DynamicBatchingEngine]
    end

    subgraph Serving Layer
        FASTAPI[FastAPI App]
        HEALTH[GET /health]
        GENERATE[POST /generate]
    end

    subgraph Evaluation Layer
        EVAL_RUNNER[EvaluationRunner]
        PERPLEXITY[Perplexity]
        MT_BENCH[MT-Bench]
        METRICS[MetricsRegistry]
    end

    subgraph Monitoring
        PROMETHEUS[Prometheus]
        GRAFANA[Grafana]
    end

    CLI --> TRAIN_ENTRY
    API_CLIENT --> FASTAPI
    NB --> DATA_LOADER

    HF_HUB --> DATA_LOADER
    S3 --> DATA_LOADER
    LOCAL_DATA --> DATA_LOADER
    DATA_LOADER --> PREPROCESSOR
    PREPROCESSOR --> TRAINER

    TRAIN_ENTRY --> MODEL_LOADER
    MODEL_LOADER --> BNB_QUANT
    BNB_QUANT --> QLORA_PREP
    QLORA_PREP --> LORA_ADAPT
    LORA_ADAPT --> TRAINER
    TRAINER --> FSDP
    TRAINER --> DS
    TRAINER --> CKPT
    TRAINER --> WANDB

    CKPT --> GEN_ENGINE
    CKPT --> VLLM
    GEN_ENGINE --> BATCHER
    VLLM --> FASTAPI
    BATCHER --> FASTAPI
    FASTAPI --> HEALTH
    FASTAPI --> GENERATE

    TRAINER --> EVAL_RUNNER
    EVAL_RUNNER --> PERPLEXITY
    EVAL_RUNNER --> MT_BENCH
    EVAL_RUNNER --> METRICS

    FASTAPI --> PROMETHEUS
    PROMETHEUS --> GRAFANA
```

---

## Training Pipeline

```mermaid
sequenceDiagram
    participant CLI as CLI / train.sh
    participant Train as train.py
    participant Loader as ModelLoader
    participant Quant as BitsAndBytes
    participant LoRA as LoraAdapter
    participant Data as DatasetLoader
    participant Prep as DataPreprocessor
    participant Trainer as QLoRATrainer
    participant CKPT as CheckpointManager
    participant WB as W&B

    CLI->>Train: launch with config
    Train->>Loader: load base model
    Loader->>Quant: apply 4-bit quantization
    Quant-->>Loader: quantized model
    Loader-->>Train: LoadedModel

    Train->>LoRA: attach LoRA adapters
    LoRA-->>Train: PeftModel

    Train->>Data: load dataset
    Data-->>Train: DatasetSplit

    Train->>Prep: tokenize & format
    Prep-->>Train: tokenized dataset

    Train->>Trainer: train(model, dataset, config)
    loop Training loop
        Trainer->>WB: log step metrics
        Trainer->>CKPT: save checkpoint (every N steps)
    end
    Trainer-->>Train: TrainerState

    Train->>LoRA: merge_and_unload
    Train->>CKPT: save final model
```

---

## Inference Pipeline

```mermaid
sequenceDiagram
    participant Client as HTTP Client
    participant API as FastAPI
    participant Batcher as DynamicBatcher
    participant Engine as GenerationEngine / vLLM
    participant Model as Merged Model

    Client->>API: POST /generate {prompt, params}
    API->>Batcher: submit(request_id, prompt, params)
    Note over Batcher: Wait up to max_wait_ms<br/>or until batch is full
    Batcher->>Engine: batch_generate([prompts], params)
    Engine->>Model: forward pass
    Model-->>Engine: logits
    Engine-->>Batcher: [completions]
    Batcher-->>API: completion for request_id
    API-->>Client: GenerateResponse {text, tokens, latency_ms}
```

---

## Component Responsibilities

| Component | Package | Responsibility |
|-----------|---------|----------------|
| `ModelLoader` | `src.models` | Load base model from HF Hub with optional quantization |
| `TokenizerManager` | `src.models` | Load tokenizer, apply prompt templates |
| `LoraAdapter` | `src.models` | Attach and manage PEFT LoRA adapters |
| `DatasetLoader` | `src.data` | Load datasets from HF Hub or local files |
| `DataPreprocessor` | `src.data` | Tokenize and format datasets for training |
| `QLoRATrainer` | `src.training` | Orchestrate the training loop |
| `CheckpointManager` | `src.training` | Save, rotate, and restore checkpoints |
| `GenerationEngine` | `src.inference` | HF-native text generation |
| `VLLMServerManager` | `src.inference` | Manage vLLM subprocess lifecycle |
| `InferenceBenchmark` | `src.inference` | Throughput and latency measurement |
| `EvaluationRunner` | `src.evaluation` | Dispatch evaluation tasks |
| `FastAPI app` | `src.serving` | REST API for inference requests |
| `DynamicBatchingEngine` | `src.serving` | Group concurrent requests into GPU batches |

---

## Configuration Architecture

```mermaid
graph LR
    CLI_ARGS[CLI Overrides] --> HYDRA[Hydra Composer]
    MODEL_YAML[configs/model/*.yaml] --> HYDRA
    TRAIN_YAML[configs/training/*.yaml] --> HYDRA
    DATA_YAML[configs/datasets/*.yaml] --> HYDRA
    INFER_YAML[configs/inference/*.yaml] --> HYDRA
    ENV[.env variables] --> HYDRA
    HYDRA --> PLATFORM_CFG[PlatformConfig]
    PLATFORM_CFG --> TRAIN[Training]
    PLATFORM_CFG --> SERVE[Serving]
    PLATFORM_CFG --> EVAL[Evaluation]
```

---

## Deployment Architecture (Kubernetes)

```mermaid
graph TB
    subgraph Kubernetes Cluster
        subgraph Training Namespace
            JOB[trainer-job.yaml<br/>Kubernetes Job]
            GPU_NODE[GPU Node<br/>8× A100]
        end

        subgraph Inference Namespace
            DEPLOY[inference-deployment.yaml<br/>Deployment]
            HPA[autoscaler.yaml<br/>HorizontalPodAutoscaler]
            SVC[Service / LoadBalancer]
        end

        subgraph Monitoring Namespace
            PROM[Prometheus]
            GRAF[Grafana]
        end
    end

    S3_MODELS[(S3 Model Store)] --> JOB
    JOB --> GPU_NODE
    GPU_NODE --> S3_MODELS

    S3_MODELS --> DEPLOY
    DEPLOY --> HPA
    DEPLOY --> SVC
    SVC --> INTERNET[Internet]

    DEPLOY --> PROM
    JOB --> PROM
    PROM --> GRAF
```
