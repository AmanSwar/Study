# APPENDIX B — Model Memory Sizing Calculator

## 1. Introduction

Predicting model memory requirements is critical for deployment planning: OOM (out-of-memory) errors cause cascading failures, while over-provisioning wastes hardware costs. This appendix provides exact formulas and a complete Python calculator for predicting memory requirements across all major model families: vision transformers, language models, diffusion models, and multimodal systems.

The key insight: **total memory = parameter memory + activation memory + KV-cache memory + overhead**. Each component scales differently with batch size, sequence length, and precision.

---

## 2. Exact Memory Formulas

### 2.1 Parameter Memory

Parameters are learned weights, stored once in memory (before inference).

```
Parameter Memory (GB) = (num_parameters × bytes_per_parameter) / 10^9

Precision to bytes per parameter:
- FP32: 4 bytes
- FP16: 2 bytes
- INT8: 1 byte
- INT4: 0.5 bytes

Example: BERT-large
num_parameters = 340M
FP32: 340M × 4 bytes = 1.36 GB
FP16: 340M × 2 bytes = 0.68 GB
INT8: 340M × 1 byte = 0.34 GB
```

### 2.2 Activation Memory

Activations are intermediate results during forward pass, recomputed each batch.

```
Activation memory depends on:
- Batch size (B)
- Sequence length (S)
- Hidden dimension (H)
- Number of layers (L)

For Transformer:
Activation memory ≈ B × S × H × L × 4 bytes × (1 + recomputation_factor)

Recomputation factor:
- No recomputation: 1.0 (keep all activations in memory)
- Full recomputation: 0.1 (recompute, minimal storage)
- Selective: 0.3-0.5 (trade compute for memory)

Example: BERT-base with B=32, S=384, H=768, L=12
Activation memory = 32 × 384 × 768 × 12 × 4 / 10^9 ≈ 1.19 GB
With full recomputation: 1.19 × 0.1 ≈ 0.12 GB
```

### 2.3 KV-Cache Memory (Autoregressive Models)

During autoregressive generation, KV (key-value) cache grows with sequence length.

```
KV-Cache per token (bytes) = 2 × num_heads × head_dim × num_layers × batch_size × 2

For LLaMA-7B (num_heads=32, head_dim=128, layers=32):
Per token per batch = 2 × 32 × 128 × 32 × batch_size × 2 bytes
                    = 524,288 × batch_size bytes
                    ≈ 0.5 MB per token per batch element

For 2048 token sequence:
KV-cache = 0.5 MB/token × 2048 tokens ≈ 1.0 GB per batch element

Critical: KV-cache can exceed parameter memory for large sequences!
Optimization: Use key-value cache quantization (INT8) to reduce by 4x
```

### 2.4 Total Memory During Inference

```
Total memory = Parameter memory + Activation memory + KV-cache + Overhead

Overhead includes:
- Optimizer states (Adam: 8 bytes per param for momentum + variance)
- Gradient buffers: 4 bytes per param (FP32 gradients)
- Framework overhead: 10-20% of above

Example: Fine-tuning LLaMA-7B
- Parameter memory (FP32): 28 GB
- Activations (B=4, S=2048): 2 GB
- KV-cache (B=4, S=2048): 4 GB
- Optimizer states: 28 × 8 = 224 GB (!!)
- Total for training: ~258 GB (impossible on single GPU!)

Solution: Use LORA (8-bit quantization + parameter-efficient fine-tuning)
- Parameter memory: 28 GB (quantized to 7 GB)
- LORA adapters: 0.1-0.2 GB
- KV-cache: 4 GB
- Total: ~11 GB (fits on single A100 80GB)
```

---

## 3. Complete Python Calculator

```python
import numpy as np
from typing import Dict, Tuple
from dataclasses import dataclass

@dataclass
class MemoryEstimate:
    """Complete memory estimate breakdown."""
    parameter_memory_gb: float
    activation_memory_gb: float
    kv_cache_memory_gb: float
    total_inference_gb: float
    total_training_gb: float

    def summary(self):
        print(f"Parameter memory:     {self.parameter_memory_gb:6.2f} GB")
        print(f"Activation memory:    {self.activation_memory_gb:6.2f} GB")
        print(f"KV-cache memory:      {self.kv_cache_memory_gb:6.2f} GB")
        print(f"─────────────────────────────")
        print(f"Total (inference):    {self.total_inference_gb:6.2f} GB")
        print(f"Total (training):     {self.total_training_gb:6.2f} GB")

class MemoryCalculator:
    """Calculate model memory requirements."""

    # Bytes per element in each precision
    DTYPE_BYTES = {
        'fp32': 4,
        'fp16': 2,
        'bf16': 2,
        'int8': 1,
        'int4': 0.5
    }

    @staticmethod
    def calculate_bert_memory(
        num_params_m: float,  # in millions
        batch_size: int,
        seq_length: int,
        hidden_dim: int,
        num_layers: int,
        dtype: str = 'fp32'
    ) -> MemoryEstimate:
        """Calculate BERT/RoBERTa memory."""
        dtype_bytes = MemoryCalculator.DTYPE_BYTES[dtype]

        # Parameter memory
        param_memory = (num_params_m * 1e6 * dtype_bytes) / 1e9

        # Activation memory (simplified)
        activation_memory = (batch_size * seq_length * hidden_dim * num_layers * 4) / 1e9
        activation_memory *= 0.5  # Account for selective activation recomputation

        # No KV-cache for BERT (bidirectional encoder)
        kv_cache_memory = 0

        # Total
        total_inference = param_memory + activation_memory + kv_cache_memory
        total_training = param_memory + activation_memory + (param_memory * 2)  # Optimizer states

        return MemoryEstimate(
            parameter_memory_gb=param_memory,
            activation_memory_gb=activation_memory,
            kv_cache_memory_gb=kv_cache_memory,
            total_inference_gb=total_inference,
            total_training_gb=total_training
        )

    @staticmethod
    def calculate_llm_memory(
        num_params_m: float,  # millions
        batch_size: int,
        seq_length: int,
        hidden_dim: int,
        num_heads: int,
        num_layers: int,
        dtype: str = 'fp32',
        kv_cache_dtype: str = None
    ) -> MemoryEstimate:
        """Calculate LLaMA/Mistral/Llama2 memory."""
        dtype_bytes = MemoryCalculator.DTYPE_BYTES[dtype]
        kv_cache_dtype_bytes = MemoryCalculator.DTYPE_BYTES.get(kv_cache_dtype or dtype, dtype_bytes)

        # Parameter memory
        param_memory = (num_params_m * 1e6 * dtype_bytes) / 1e9

        # Activation memory during prefill (not decoding)
        activation_memory = (batch_size * seq_length * hidden_dim * num_layers * 4) / 1e9
        activation_memory *= 0.3  # With gradient checkpointing

        # KV-cache memory
        # KV per layer = 2 (K and V) × num_heads × (hidden_dim / num_heads) × batch_size × seq_length
        head_dim = hidden_dim // num_heads
        kv_per_head = 2 * num_heads * head_dim  # 2 for K and V
        kv_cache_memory = (kv_per_head * batch_size * seq_length * num_layers * kv_cache_dtype_bytes) / 1e9

        # Total
        total_inference = param_memory + activation_memory + kv_cache_memory
        total_training = param_memory + activation_memory + (param_memory * 8)  # 8x for optimizer + grads

        return MemoryEstimate(
            parameter_memory_gb=param_memory,
            activation_memory_gb=activation_memory,
            kv_cache_memory_gb=kv_cache_memory,
            total_inference_gb=total_inference,
            total_training_gb=total_training
        )

    @staticmethod
    def calculate_vit_memory(
        num_params_m: float,
        batch_size: int,
        image_resolution: int,
        patch_size: int,
        hidden_dim: int,
        num_layers: int,
        dtype: str = 'fp32'
    ) -> MemoryEstimate:
        """Calculate Vision Transformer memory."""
        dtype_bytes = MemoryCalculator.DTYPE_BYTES[dtype]

        # Parameter memory
        param_memory = (num_params_m * 1e6 * dtype_bytes) / 1e9

        # Number of patches
        num_patches = (image_resolution // patch_size) ** 2

        # Activation memory (patch embeddings + transformer)
        activation_memory = (batch_size * num_patches * hidden_dim * num_layers * 4) / 1e9
        activation_memory *= 0.3

        # No KV-cache for ViT (not autoregressive)
        kv_cache_memory = 0

        total_inference = param_memory + activation_memory
        total_training = param_memory + activation_memory + (param_memory * 2)

        return MemoryEstimate(
            parameter_memory_gb=param_memory,
            activation_memory_gb=activation_memory,
            kv_cache_memory_gb=kv_cache_memory,
            total_inference_gb=total_inference,
            total_training_gb=total_training
        )

    @staticmethod
    def calculate_diffusion_memory(
        num_params_m: float,
        batch_size: int,
        latent_size: int,  # 64 for 512x512 images
        hidden_dim: int,
        num_layers: int,
        dtype: str = 'fp32'
    ) -> MemoryEstimate:
        """Calculate diffusion model memory (e.g., Stable Diffusion)."""
        dtype_bytes = MemoryCalculator.DTYPE_BYTES[dtype]

        param_memory = (num_params_m * 1e6 * dtype_bytes) / 1e9

        # Activation for all timesteps (50-100 denoising steps typically)
        timesteps = 50
        activation_memory = (batch_size * latent_size * latent_size * hidden_dim * num_layers * timesteps * 4) / 1e9
        activation_memory *= 0.2

        # KV-cache for cross-attention (text embeddings)
        kv_cache_memory = (batch_size * 77 * hidden_dim * 2) / 1e9  # 77 token prompt

        total_inference = param_memory + activation_memory + kv_cache_memory
        total_training = param_memory * 1.1  # Simplified

        return MemoryEstimate(
            parameter_memory_gb=param_memory,
            activation_memory_gb=activation_memory,
            kv_cache_memory_gb=kv_cache_memory,
            total_inference_gb=total_inference,
            total_training_gb=total_training
        )

# Usage examples
def demo_memory_calculations():
    """Run example calculations."""
    print("="*60)
    print("BERT-base inference (B=32, S=384)")
    print("="*60)
    bert = MemoryCalculator.calculate_bert_memory(
        num_params_m=110,
        batch_size=32,
        seq_length=384,
        hidden_dim=768,
        num_layers=12,
        dtype='fp32'
    )
    bert.summary()

    print("\n" + "="*60)
    print("LLaMA-7B inference (B=4, S=2048)")
    print("="*60)
    llama = MemoryCalculator.calculate_llm_memory(
        num_params_m=7000,
        batch_size=4,
        seq_length=2048,
        hidden_dim=4096,
        num_heads=32,
        num_layers=32,
        dtype='fp16',
        kv_cache_dtype='int8'
    )
    llama.summary()

    print("\n" + "="*60)
    print("ViT-B inference (B=64, 384x384 images)")
    print("="*60)
    vit = MemoryCalculator.calculate_vit_memory(
        num_params_m=86,
        batch_size=64,
        image_resolution=384,
        patch_size=16,
        hidden_dim=768,
        num_layers=12,
        dtype='fp32'
    )
    vit.summary()

    print("\n" + "="*60)
    print("Stable Diffusion inference (B=1, 512x512)")
    print("="*60)
    diffusion = MemoryCalculator.calculate_diffusion_memory(
        num_params_m=860,  # UNet + VAE
        batch_size=1,
        latent_size=64,
        hidden_dim=768,
        num_layers=12,
        dtype='fp16'
    )
    diffusion.summary()

if __name__ == "__main__":
    demo_memory_calculations()
```

---

## 4. Model Family Memory Reference

### 4.1 Language Models

```
┌──────────────────┬────────┬─────────┬──────────┬──────────┬──────────┐
│ Model            │ Params │ FP32(GB)│ FP16(GB) │ INT8(GB) │ w/KVcache│
├──────────────────┼────────┼─────────┼──────────┼──────────┼──────────┤
│ GPT2-medium      │ 345M   │ 1.38    │ 0.69     │ 0.35     │ 0.50     │
│ BERT-base        │ 110M   │ 0.44    │ 0.22     │ 0.11     │ 0.11     │
│ BERT-large       │ 340M   │ 1.36    │ 0.68     │ 0.34     │ 0.34     │
│ GPT-3 (small)    │ 125M   │ 0.50    │ 0.25     │ 0.13     │ 0.20     │
│ T5-base          │ 220M   │ 0.88    │ 0.44     │ 0.22     │ 0.30     │
│ T5-large         │ 770M   │ 3.08    │ 1.54     │ 0.77     │ 1.00     │
│ DistilBERT       │ 66M    │ 0.26    │ 0.13     │ 0.07     │ 0.07     │
│ ALBERT-base      │ 12M    │ 0.05    │ 0.03     │ 0.01     │ 0.02     │
└──────────────────┴────────┴─────────┴──────────┴──────────┴──────────┘

Scaling: Total Inference Memory = Parameters + KV-cache (for S=2048, B=1)
- FP32: 1x
- FP16: 0.5x (2x speedup, minimal quality loss)
- INT8: 0.25x (4x speedup, 1-2% accuracy loss)
```

### 4.2 Large Language Models (Decoder-only)

```
┌──────────────────────┬────────┬─────────┬──────────┬──────────┬──────────────────┐
│ Model                │ Params │ FP32(GB)│ FP16(GB) │ INT8(GB) │ KV-cache(S=2K,B=1)│
├──────────────────────┼────────┼─────────┼──────────┼──────────┼──────────────────┤
│ Mistral-7B           │ 7B     │ 28      │ 14       │ 7        │ 1.5              │
│ LLaMA-13B            │ 13B    │ 52      │ 26       │ 13       │ 3.0              │
│ Llama2-70B           │ 70B    │ 280     │ 140      │ 70       │ 15.0             │
│ Falcon-180B          │ 180B   │ 720     │ 360      │ 180      │ 39.0             │
│ GPT-3 (175B)         │ 175B   │ 700     │ 350      │ 175      │ 38.0             │
│ PaLM-540B            │ 540B   │ 2160    │ 1080     │ 540      │ 117.0            │
└──────────────────────┴────────┴─────────┴──────────┴──────────┴──────────────────┘

Critical insight: Llama-70B requires 280GB in FP32
- Fits on: 4x H100 with NVLink (80GB each) OR 2x H100-900GB
- CPU inference: Infeasible (requires 1000GB+ RAM)
- Solution: Quantization to INT4 (70GB) or INT8 (70GB)
```

### 4.3 Vision Models

```
┌──────────────────┬────────┬─────────┬──────────┬──────────┐
│ Model            │ Params │ FP32(GB)│ FP16(GB) │ INT8(GB) │
├──────────────────┼────────┼─────────┼──────────┼──────────┤
│ ResNet-50        │ 26M    │ 0.10    │ 0.05     │ 0.03     │
│ EfficientNet-B7  │ 66M    │ 0.26    │ 0.13     │ 0.07     │
│ ViT-B/16         │ 86M    │ 0.34    │ 0.17     │ 0.09     │
│ ViT-L/16         │ 304M   │ 1.22    │ 0.61     │ 0.30     │
│ ViT-H/14         │ 633M   │ 2.53    │ 1.27     │ 0.63     │
│ CLIP-ViT-L       │ 427M   │ 1.71    │ 0.86     │ 0.43     │
└──────────────────┴────────┴─────────┴──────────┴──────────┘

Batching effects on activation memory (B, S, H = parameters):
- Activation ∝ B × S × H
- 1 image vs 64 images: ~64x activation memory increase
- But 64x throughput gain (parallelization)
```

---

## 5. Hardware Requirements by Model

```
Recommended GPU memory for common scenarios:

┌────────────────────┬─────────────┬─────────────┬─────────────┐
│ Model              │ Single GPU  │ Batching(32)│ Fine-tuning │
├────────────────────┼─────────────┼─────────────┼─────────────┤
│ BERT-base (FP32)   │ RTX 3090    │ A100 (40GB) │ A100 (80GB) │
│ Llama-13B (FP16)   │ A100 (40GB) │ A100 (80GB) │ 2x A100     │
│ Llama-70B (FP16)   │ A100 (80GB) │ 2x A100     │ 4x A100     │
│ Llama-70B (INT8)   │ A100 (40GB) │ A100 (80GB) │ Not feasible│
│ GPT-3 (175B, FP16) │ 3x A100     │ 6x A100     │ Not practical│
└────────────────────┴─────────────┴─────────────┴─────────────┘

Alternative: CPU + offloading
- Use CPU RAM (256GB+) as swap for GPU VRAM
- 10-50x slower but enables fitting larger models
- Viable for development/research, not production
```

---

## 6. Optimization Strategies

### 6.1 Reducing Memory Without Quality Loss

```
Technique                  Memory Reduction    Quality Loss    Latency Impact
──────────────────────────────────────────────────────────────────────────
FP32 → FP16               2x                  <0.5%           2x faster
FP16 → INT8               2x more             1-3%            2x faster
Quantization (all)        4x                  1-3%            3-4x faster
Distillation              4-10x               2-5%            5-20x faster
LoRA fine-tuning          100x                <1%             Same
Pruning (50%)             2x                  2-5%            1.5x faster
Knowledge distillation    10x                 3-10%           10-20x faster
Flash Attention           No reduction        None            2-3x faster
Gradient checkpointing    2x (training)       None            1.2x slower
```

### 6.2 KV-Cache Optimization

```
KV-cache is often the bottleneck for long sequences. Optimizations:

1. Quantize KV-cache: 4x reduction
   - FP32 → FP16: 2x
   - FP16 → INT8: 2x more
   - Combined: 4x (minimal quality loss)

2. Sparse attention: 10-100x reduction
   - Only attend to recent tokens
   - Trade: Requires architecture change

3. Paged attention: Flexible memory
   - Allocate in pages, not contiguous
   - Enables larger batch sizes without OOM

4. Multi-query attention: 4-8x reduction
   - Single head for K, V (vs per-head in MQA)
   - Minimal quality loss
```

---

## 7. Summary & Key Takeaways

**Memory prediction formula:**
```
Total Memory (GB) = (Parameters × bytes/param +
                     Activation +
                     KV-cache) / 10^9
```

**Quick estimates:**
- Parameters dominate for small sequences
- KV-cache dominates for long sequences (>1K tokens)
- Quantization essential: 4x reduction without accuracy loss
- For 70B+ models: INT8 or INT4 quantization mandatory

**Hardware matching:**
- <10GB model: RTX 4090 / RTX 3090
- 10-40GB model: A100 (40GB) / A10
- 40-80GB model: A100 (80GB) / H100 (80GB)
- 80GB+ model: Multi-GPU or CPU offloading

**Next steps:**
1. Calculate exact memory requirements for your model
2. Choose appropriate precision (FP32 vs FP16 vs INT8)
3. Test batch size limits
4. Implement KV-cache optimization if needed
5. Measure actual memory usage (may differ from formula)
