# Deployment Guide

This guide covers deploying the QLoRA platform with Docker (local / staging) and Kubernetes (production).

---

## Docker

### Building Images

```bash
# Training image
docker build -f deployment/docker/Dockerfile.train \
    -t qlora-train:latest .

# Inference image
docker build -f deployment/docker/Dockerfile.inference \
    -t qlora-inference:latest .
```

### Running the Inference API

```bash
# Start the inference server (requires a merged model at $MODEL_PATH)
docker run --gpus all \
    -e HF_TOKEN=$HF_TOKEN \
    -e MODEL_PATH=/models/llama3-8b-merged \
    -v /data/models:/models \
    -p 8000:8000 \
    qlora-inference:latest
```

### Docker Compose (Full Stack)

```bash
# Copy and configure environment
cp .env.example .env

# Start API + monitoring
docker compose --profile monitoring up -d

# Start a training job
docker compose --profile training up trainer
```

---

## Kubernetes

### Prerequisites

- Kubernetes 1.28+
- NVIDIA GPU Operator installed
- `kubectl` configured for your cluster
- A container registry (e.g., ECR, GCR, Docker Hub)

### Deploying the Inference Service

```bash
# Push the image to your registry
docker tag qlora-inference:latest $REGISTRY/qlora-inference:latest
docker push $REGISTRY/qlora-inference:latest

# Apply manifests
kubectl apply -f deployment/kubernetes/inference-deployment.yaml
kubectl apply -f deployment/kubernetes/autoscaler.yaml

# Verify rollout
kubectl rollout status deployment/qlora-inference
```

### Running a Training Job

```bash
# Create a ConfigMap from your training config
kubectl create configmap qlora-train-config \
    --from-file=configs/training/qlora_default.yaml

# Launch the training job
kubectl apply -f deployment/kubernetes/trainer-job.yaml

# Monitor progress
kubectl logs -f job/qlora-trainer
```

### Autoscaling

The `autoscaler.yaml` configures a `HorizontalPodAutoscaler` targeting:

- **Min replicas**: 1
- **Max replicas**: 10
- **Scale trigger**: GPU utilisation > 70%

Requires the [DCGM Exporter](https://github.com/NVIDIA/dcgm-exporter) for GPU metrics.

---

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `HF_TOKEN` | Yes | Hugging Face API token |
| `WANDB_API_KEY` | For training | Weights & Biases API key |
| `MODEL_PATH` | Yes (inference) | Path to the merged model directory |
| `API_PORT` | No | Server bind port (default: 8000) |
| `LOG_LEVEL` | No | Log verbosity (default: INFO) |

---

## Health Checks

The inference API exposes:

```
GET /health → 200 OK {"status": "ok", "version": "0.1.0", "model": "..."}
```

Kubernetes liveness and readiness probes are configured in `inference-deployment.yaml`.

---

## Production Checklist

- [ ] Restrict `CORS` origins in `src/serving/api.py`
- [ ] Set `ENVIRONMENT=production` in your deployment environment
- [ ] Enable TLS termination at the ingress layer
- [ ] Configure resource `requests` and `limits` in the K8s manifests
- [ ] Set up Prometheus scraping for the `/metrics` endpoint
- [ ] Configure PodDisruptionBudget for zero-downtime rolling updates
- [ ] Store `HF_TOKEN` and `WANDB_API_KEY` in Kubernetes Secrets (not ConfigMaps)
