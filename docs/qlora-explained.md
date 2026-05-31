# QLoRA Explained

A technical primer on Quantized Low-Rank Adaptation (QLoRA) — the technique underpinning this platform.

---

## Background: Full Fine-Tuning is Expensive

Fine-tuning a 70B-parameter language model in full precision (BF16) requires storing:

| Quantity | Memory |
|----------|--------|
| Model weights (BF16) | 140 GB |
| Optimizer states (AdamW, FP32) | 560 GB |
| Gradients (BF16) | 140 GB |
| **Total** | **~840 GB** |

This exceeds the capacity of even the largest single-node GPU clusters, making full fine-tuning inaccessible for most practitioners.

---

## LoRA: Parameter-Efficient Fine-Tuning

Low-Rank Adaptation (Hu et al., 2021) observes that the weight updates during fine-tuning have a low intrinsic rank. Instead of updating the full weight matrix **W** ∈ ℝ^(d×k), LoRA decomposes the update into two small matrices:

```
W' = W + ΔW = W + BA

where:
  B ∈ ℝ^(d×r)   (r << d, k)
  A ∈ ℝ^(r×k)
  ΔW ∈ ℝ^(d×k)
```

Only **A** and **B** are trained; **W** is frozen. The number of trainable parameters drops from `d×k` to `r×(d+k)`.

**Example (LLaMA-3 8B, r=16):**

| | Parameters |
|--|------------|
| Full attention weight | 4096 × 4096 = 16.8M |
| LoRA update (r=16) | 16 × (4096 + 4096) = 131K |
| Reduction | ~128× |

During inference, the adapter is merged: `W' = W + αBA/r`, where `α` (lora_alpha) is a scaling factor.

---

## QLoRA: Adding Quantization

Dettmers et al. (2023) introduced QLoRA, combining LoRA with two innovations:

### 1. NormalFloat 4-bit (NF4)

NF4 is an information-theoretically optimal 4-bit dtype for normally-distributed weights. Quantization maps each weight to the nearest of 16 pre-defined quantization levels, chosen to minimize quantization error for zero-mean Gaussian distributions.

```
Memory per parameter:
  FP32: 32 bits = 4 bytes
  BF16: 16 bits = 2 bytes
  INT8:  8 bits = 1 byte
  NF4:   4 bits = 0.5 bytes   ← 8× reduction vs FP32
```

### 2. Double Quantization

Quantization constants (one per block of weights) are themselves quantized from FP32 to FP8, saving an additional ~0.37 bits per parameter on average.

### 3. Paged Optimizers

GPU memory for optimizer states is managed with NVIDIA's unified memory, allowing overflow to CPU RAM without OOM errors during batch processing spikes.

---

## Memory Savings Summary

For a 7B-parameter model:

| Method | Training VRAM |
|--------|--------------|
| Full fine-tune (BF16) | ~60 GB |
| LoRA (BF16 base) | ~28 GB |
| QLoRA (NF4 base + BF16 adapters) | ~10 GB |

QLoRA brings 7B models within reach of a single consumer GPU (RTX 3090/4090 with 24 GB VRAM).

---

## Key Hyperparameters

| Parameter | Description | Typical Values |
|-----------|-------------|----------------|
| `lora_r` | Adapter rank — higher = more capacity, more parameters | 8, 16, 32, 64 |
| `lora_alpha` | Scaling factor, typically `2 × lora_r` | 16, 32, 64, 128 |
| `lora_dropout` | Dropout within LoRA layers | 0.0 – 0.1 |
| `target_modules` | Which layers to adapt | q_proj, v_proj, … |
| `bnb_4bit_quant_type` | Quantization dtype | nf4 (recommended), fp4 |
| `bnb_4bit_compute_dtype` | Compute dtype for dequantized ops | bfloat16 |
| `bnb_4bit_use_double_quant` | Double quantization for constant compression | true |

---

## References

1. Hu, E. J. et al. (2021). *LoRA: Low-Rank Adaptation of Large Language Models*. [arXiv:2106.09685](https://arxiv.org/abs/2106.09685)
2. Dettmers, T. et al. (2023). *QLoRA: Efficient Finetuning of Quantized LLMs*. [arXiv:2305.14314](https://arxiv.org/abs/2305.14314)
3. Dettmers, T. et al. (2022). *LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale*. [arXiv:2208.07339](https://arxiv.org/abs/2208.07339)
