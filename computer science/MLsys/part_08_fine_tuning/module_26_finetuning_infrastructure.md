# MODULE 26 — Fine-Tuning Infrastructure

## 1. Introduction & Learning Objectives

Fine-tuning large language models represents a critical capability in enterprise machine learning, enabling task-specific adaptation while leveraging pre-trained knowledge. However, fine-tuning 7B-70B parameter models on consumer hardware demands sophisticated memory optimization techniques, distributed training strategies, and precision management. This module provides a systems perspective on fine-tuning infrastructure, emphasizing memory efficiency through mathematical analysis, quantification of tradeoffs, and production-grade implementation patterns.

**Learning Objectives:**
- Develop precise understanding of memory consumption across forward pass, backward pass, optimizer state, and activation caching
- Master mixed precision training (FP16/BF16) with loss scaling and numerical stability considerations
- Implement gradient checkpointing with theoretical memory-compute tradeoff analysis
- Understand ZeRO optimization stages with exact memory savings quantification
- Design FSDP (Fully Sharded Data Parallel) systems for efficient multi-GPU training
- Apply DeepSpeed configuration for production fine-tuning pipelines

## 2. Full Fine-Tuning Memory Breakdown

### 2.1 Theoretical Memory Components

Fine-tuning a model requires simultaneous storage of multiple memory categories. For a model with P parameters, the total memory requirement is:

```
Total Memory (bytes) = Model States + Gradient States + Optimizer States + Activation Cache
```

Let's analyze each component for an LLM with the following typical structure:

**Model Configuration Example:**
- Parameters: P = 7 billion
- Parameter dtype: FP32 (4 bytes), FP16 (2 bytes), BF16 (2 bytes)
- Batch size: B = 32
- Sequence length: T = 2048
- Hidden dimension: d = 4096
- Number of layers: L = 32
- Attention heads: h = 32

### 2.2 Model State Memory

**Standard Precision (FP32):**
```
Memory = P × 4 bytes
       = 7×10^9 × 4
       = 28 GB

For 70B model:
Memory = 70×10^9 × 4 = 280 GB (exceeds single GPU)
```

**Mixed Precision (FP16):**
```
Memory = P × 2 bytes (FP16 weights)
       = 7×10^9 × 2
       = 14 GB

Note: Master weights (FP32 copy) stored separately for stability
```

**Mixed Precision with Master Weights:**
```
Memory = P × 2 bytes (FP16 working copy)
       + P × 4 bytes (FP32 master for optimizer updates)
       = 7×10^9 × (2 + 4)
       = 42 GB

This doubles effective footprint vs FP16 alone
```

### 2.3 Gradient Memory

During backward pass, gradients accumulate for all parameters:

**Gradient Storage:**
```
Memory = P × dtype_size

FP32 gradients:
Memory = 7×10^9 × 4 = 28 GB

BF16 gradients:
Memory = 7×10^9 × 2 = 14 GB
```

**Gradient Accumulation with Multiple Micro-Batches:**
```
If gradient_accumulation_steps = G
Then effective batch size = B_micro × G

Memory requirement increases proportionally:
Total gradient memory = G × P × dtype_size

Example: G = 4 accumulation steps
Memory = 4 × 7×10^9 × 2 = 56 GB
```

### 2.4 Optimizer State Memory

Different optimizers maintain varying state sizes:

**SGD (Momentum):**
```
Per-parameter state: momentum buffer = dtype_size

Memory = P × dtype_size
       = 7×10^9 × 4 (FP32 momentum)
       = 28 GB
```

**AdamW (Most Common):**
```
Per-parameter state:
- Exponential moving average of gradients (m): dtype_size
- Exponential moving average of squared gradients (v): dtype_size

Memory = P × 2 × dtype_size
       = 7×10^9 × 2 × 4
       = 56 GB

For 70B model:
Memory = 70×10^9 × 2 × 4 = 560 GB (single GPU unable to hold)
```

**AdamW Breakdown:**
```
Total optimizer memory = (M_1 + M_2) where:
M_1 (first moment): size of gradient tensor
M_2 (second moment): size of gradient tensor

Example calculation for 7B LLM:
- Embedding layer (7B vocab × 4096 hidden):
  M_1 = 28.7 GB × 1 (gradient size)
  M_2 = 28.7 GB × 1 (gradient size)
  Total embedding optimizer state = 57.4 GB

- Attention + FFN layers:
  M_1 ≈ 56 GB (similar to embedding)
  M_2 ≈ 56 GB
  Total layer optimizer state = 112 GB

- Combined: 169.4 GB for optimizer alone
```

### 2.5 Activation Memory (Forward Pass Cache)

During forward pass, activations are cached for gradient computation in backward pass. This represents significant memory overhead.

**Per-Layer Activation Cache:**
```
For a transformer block computing:
- Self-attention output
- Feed-forward output
- Normalization intermediates

Typical activation memory per token:
- Attention hidden state: [B, T, d] = 32 × 2048 × 4096 × 4 bytes = 1.08 GB
- FFN hidden state (expanded): [B, T, 4d] = 32 × 2048 × 16384 × 4 bytes = 4.3 GB
- Normalization intermediate: [B, T, d] = 1.08 GB
- Per-layer activation total: ~6.5 GB

Total activation memory across L=32 layers:
Memory = L × per_layer_activation
       = 32 × 6.5 GB
       = 208 GB
```

**Activation Memory Formula (General):**
```
Activation memory = B × T × d × num_layers × bytes_per_element

Where:
- B = batch size
- T = sequence length
- d = hidden dimension
- num_layers = number of transformer layers
- bytes_per_element = 4 (FP32), 2 (FP16), etc.

For our example:
Activation memory = 32 × 2048 × 4096 × 32 × 4
                  = 33.5 × 10^9 bytes
                  ≈ 33.5 GB
```

### 2.6 Complete Memory Budget Example

**Scenario: Fine-tune 7B LLM on single GPU (A100-80GB)**

| Component | FP32 | FP16 + Master |
|-----------|------|---------------|
| Model weights | 28 GB | 14 GB (FP16) + 28 GB (master) = 42 GB |
| Gradients | 28 GB | 14 GB |
| Optimizer (AdamW) | 56 GB | 56 GB |
| Activation cache | 33.5 GB | 33.5 GB |
| **Total** | **145.5 GB** | **145.5 GB** |
| GPU VRAM | 80 GB | 80 GB |
| **Feasible?** | ❌ **No** | ❌ **No** |

**Conclusion**: Standard fine-tuning configurations exceed GPU memory. Solutions required:
1. Gradient checkpointing (reduce activation cache)
2. ZeRO sharding (distribute across GPUs)
3. Mixed precision (reduce dtype sizes)
4. LoRA/PEFT (reduce trainable parameters)

## 3. Mixed Precision Training

### 3.1 FP32 vs FP16 vs BF16

**IEEE 754 Floating Point Formats:**

```
FP32 (Single Precision):
[Sign: 1 bit] [Exponent: 8 bits] [Mantissa: 23 bits]
Range: ±10^-38 to ±10^38
Precision: ~7 decimal digits
Per-value: 4 bytes

FP16 (Half Precision):
[Sign: 1 bit] [Exponent: 5 bits] [Mantissa: 10 bits]
Range: ±6×10^-5 to ±65504
Precision: ~3-4 decimal digits
Per-value: 2 bytes
Risk: Underflow/overflow in gradients

BF16 (Brain Float 16):
[Sign: 1 bit] [Exponent: 8 bits] [Mantissa: 7 bits]
Range: ±10^-38 to ±10^38 (matches FP32!)
Precision: ~3 decimal digits (reduced vs FP16)
Per-value: 2 bytes
Advantage: No range clipping needed
```

**Numerical Properties Comparison:**

| Property | FP32 | FP16 | BF16 |
|----------|------|------|------|
| Exponent bits | 8 | 5 | 8 |
| Mantissa bits | 23 | 10 | 7 |
| Range | 10^-38 to 10^38 | 6×10^-5 to 65K | 10^-38 to 10^38 |
| Smallest non-zero | ~1.2×10^-38 | ~6.1×10^-5 | ~1.2×10^-38 |
| Gradient safety | ✓ Safe | ⚠ Underflow risk | ✓ Safe |
| Memory | 4 bytes | 2 bytes | 2 bytes |
| Compute speed | 1× | 2-4× (GPU dependent) | 2-4× |

### 3.2 Loss Scaling Strategy

Loss scaling addresses gradient underflow in FP16 training by temporarily amplifying loss values:

**Gradient Underflow Problem:**
```
Standard backward pass:
loss = 0.001 (for example)
dloss/dw = gradient_vector

gradient values typically: 10^-2 to 10^-4

FP16 minimum representable: ~6×10^-5

Risk: Very small gradients underflow to zero
Consequence: No parameter update, no learning
```

**Dynamic Loss Scaling Solution:**
```
loss_scaled = loss × scale_factor (e.g., 1024)
              = 0.001 × 1024 = 1.024

gradient_scaled = dloss_scaled/dw × 1024
                = gradient_vector × 1024

gradient_scaled values: 10^0 to 10^-2 (safely representable in FP16)

After backward pass:
gradient_unscaled = gradient_scaled / 1024
                  = gradient_vector (original value)

optimizer.step(gradient_unscaled)
```

**Dynamic Scaling Algorithm:**
```python
class DynamicLossScaler:
    def __init__(self, initial_scale=2**16, max_scale=2**24):
        self.scale = initial_scale
        self.max_scale = max_scale
        self.consecutive_overflow_count = 0
        self.consecutive_normal_count = 0

    def scale_loss(self, loss):
        return loss * self.scale

    def unscale_gradients(self, gradients):
        return [g / self.scale for g in gradients]

    def step(self, overflow_flag):
        """Adjust scale based on overflow detection"""
        if overflow_flag:
            # Overflow detected: reduce scale
            self.scale = max(self.scale / 2.0, 1.0)
            self.consecutive_overflow_count += 1
            self.consecutive_normal_count = 0
        else:
            # No overflow: increase scale if safe
            if self.consecutive_normal_count > 1000:
                self.scale = min(self.scale * 2.0, self.max_scale)
                self.consecutive_overflow_count = 0
                self.consecutive_normal_count = 0
            else:
                self.consecutive_normal_count += 1

# Usage in training loop
scaler = DynamicLossScaler()

for batch in dataloader:
    loss = forward_pass(batch)
    scaled_loss = scaler.scale_loss(loss)

    backward_pass(scaled_loss)

    # Check for overflow in gradients
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    has_overflow = torch.isnan(grad_norm) or torch.isinf(grad_norm)

    if not has_overflow:
        optimizer.step()

    scaler.step(has_overflow)
```

### 3.3 Master Weights Pattern

Master weights maintain FP32 precision for optimization steps while using FP16 for computation:

**Dual-Weight System:**
```
Model weights (FP16):      2 × P bytes
Master weights (FP32):     4 × P bytes
Total weight memory:       6 × P bytes

For 7B model:
Total = 6 × 7×10^9 = 42 GB
```

**Training Loop with Master Weights:**
```python
def training_step_with_master_weights(
    model_fp16, master_weights_fp32,
    batch, optimizer
):
    # Forward: use FP16 model
    loss = forward_pass(model_fp16, batch)

    # Backward: accumulate gradients
    backward_pass(loss)

    # Unscale gradients
    scaler.unscale_gradients()

    # Gradient clipping on FP16 model
    torch.nn.utils.clip_grad_norm_(model_fp16.parameters(), max_norm=1.0)

    # Copy FP16 gradients to FP32 for optimization
    for name, param in model_fp16.named_parameters():
        master_param = master_weights_fp32[name]
        master_param.grad = param.grad.to(torch.float32)

    # Optimizer step on FP32 master weights
    optimizer.step()

    # Copy updated FP32 weights back to FP16 model
    with torch.no_grad():
        for name, param in model_fp16.named_parameters():
            param.copy_(master_weights_fp32[name].half())

    optimizer.zero_grad()

    return loss
```

## 4. Gradient Checkpointing

### 4.1 Time-Memory Tradeoff Analysis

Gradient checkpointing (also called activation checkpointing or gradient recomputation) trades compute time for memory by not storing intermediate activations.

**Memory Savings Formula:**
```
Standard training (full activation caching):
Memory_standard = (model_params + gradients + optimizer_state) + activation_cache
                ≈ (6P + P + 2P) + (B × T × d × L)
                = 9P + B·T·d·L

With gradient checkpointing (checkpoint every k layers):
Memory_checkpoint = (model_params + gradients + optimizer_state) + (B × T × d × k)
                  ≈ 9P + B·T·d·k

Memory savings = (B·T·d·L - B·T·d·k) / (B·T·d·L)
               = (L - k) / L
               = 1 - k/L

For 32-layer model with checkpoint_k=1:
Savings = (32 - 1) / 32 = 96.9% activation memory reduction
```

**Concrete Example (7B LLM):**

| Config | Without Checkpointing | With Checkpointing (k=1) |
|--------|----------------------|-------------------------|
| Model weights | 42 GB | 42 GB |
| Gradients | 14 GB | 14 GB |
| Optimizer | 56 GB | 56 GB |
| Activations | 208 GB | 6.5 GB |
| **Total** | **320 GB** | **118.5 GB** |
| **GPU (80GB)** | ❌ Infeasible | ✓ Feasible (with ZeRO) |

### 4.2 Recomputation Cost Analysis

Enabling gradient checkpointing requires recomputing activations during backward pass:

**Time Cost Breakdown:**
```
Standard training (L=32 layers):
- Forward pass: 32 × time_per_layer
- Backward pass: 32 × time_per_layer (uses cached activations)
- Total compute: 64 × time_per_layer

With checkpointing (checkpoint every layer):
- Forward pass: 32 × time_per_layer
- Backward pass:
  - Recompute activations: 32 × time_per_layer
  - Compute gradients: 32 × time_per_layer
- Total compute: 96 × time_per_layer

Overhead = (96 - 64) / 64 = 50% increase in compute time
```

**Selective Checkpointing (Checkpoint Every k Layers):**
```
Checkpoint every 2 layers in 32-layer model:
Forward: 32 × t
Backward:
- Recompute 16 layers: 16 × t
- Compute gradients 32 layers: 32 × t
Total: 80 × t

Overhead = (80 - 64) / 64 = 25% increase
Memory savings: (32 - 2) / 32 = 93.75%
```

### 4.3 Optimal Checkpointing Strategy

**Greedy Algorithm for Optimal Checkpointing:**
```python
def find_optimal_checkpoints(
    num_layers=32,
    activation_memory_per_layer_gb=6.5,
    max_gpu_memory_gb=80,
    overhead_tolerance=0.20  # Allow 20% compute overhead
):
    """
    Find checkpoint configuration minimizing memory while maintaining speedup
    """
    best_checkpoint_interval = None
    best_memory_savings = 0

    for checkpoint_interval in range(1, num_layers + 1):
        # Memory with this checkpoint interval
        activation_memory = (
            activation_memory_per_layer_gb * checkpoint_interval
        )

        total_memory = (42 + 14 + 56) + activation_memory  # model + grad + opt + activation

        if total_memory > max_gpu_memory_gb:
            continue

        # Compute overhead
        layers_to_recompute = num_layers - checkpoint_interval
        compute_overhead = layers_to_recompute / num_layers

        if compute_overhead <= overhead_tolerance:
            memory_savings = (num_layers - checkpoint_interval) / num_layers
            if memory_savings > best_memory_savings:
                best_memory_savings = memory_savings
                best_checkpoint_interval = checkpoint_interval

    return best_checkpoint_interval, best_memory_savings
```

### 4.4 Implementation Patterns

**PyTorch Checkpoint API:**
```python
import torch
from torch.utils.checkpoint import checkpoint

class CheckpointedTransformer(torch.nn.Module):
    def __init__(self, num_layers=32):
        super().__init__()
        self.layers = torch.nn.ModuleList([
            TransformerBlock() for _ in range(num_layers)
        ])

    def forward(self, x, attention_mask=None):
        for i, layer in enumerate(self.layers):
            if i % 2 == 0:  # Checkpoint every 2 layers
                # Recompute this layer during backward
                x = checkpoint(
                    layer,
                    x,
                    attention_mask,
                    use_reentrant=False,
                )
            else:
                x = layer(x, attention_mask)

        return x
```

**Manual Checkpointing:**
```python
class ManualCheckpoint(torch.nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, *args, **kwargs):
        if self.training:
            # Save forward state needed for recomputation
            self.forward_args = args
            self.forward_kwargs = kwargs

            # Run forward pass, don't save intermediate activations
            with torch.no_grad():
                output = self.module(*args, **kwargs)

            # For backward: recompute if gradients needed
            output = self.module(*args, **kwargs)
            return output
        else:
            return self.module(*args, **kwargs)
```

## 5. ZeRO (Zero Redundancy Optimizer)

### 5.1 ZeRO Stage 1: Optimizer State Sharding

ZeRO Stage 1 (ZeRO-1) distributes optimizer state across GPUs, eliminating redundancy.

**Standard Training (No Sharding):**
```
GPU 0: Model (42 GB) + Gradients (14 GB) + Optimizer (56 GB) = 112 GB
GPU 1: Model (42 GB) + Gradients (14 GB) + Optimizer (56 GB) = 112 GB
GPU 2: Model (42 GB) + Gradients (14 GB) + Optimizer (56 GB) = 112 GB
GPU 3: Model (42 GB) + Gradients (14 GB) + Optimizer (56 GB) = 112 GB

Total redundant: 448 GB of optimizer state stored 4× across GPUs
```

**ZeRO Stage 1 (Optimizer State Sharding):**
```
GPU 0: Model (42 GB) + Gradients (14 GB) + Optimizer[0:1/4] (14 GB) = 70 GB
GPU 1: Model (42 GB) + Gradients (14 GB) + Optimizer[1/4:2/4] (14 GB) = 70 GB
GPU 2: Model (42 GB) + Gradients (14 GB) + Optimizer[2/4:3/4] (14 GB) = 70 GB
GPU 3: Model (42 GB) + Gradients (14 GB) + Optimizer[3/4:4/4] (14 GB) = 70 GB

Total: 280 GB (vs 448 GB without ZeRO)
Memory savings per GPU: (112 - 70) / 112 = 37.5%
```

**Communication Pattern:**
```
Training step:
1. Each GPU computes gradient for its data
2. All-reduce across GPUs (AllReduce communication)
3. Each GPU updates its partition of optimizer state
4. GPU i holds only optimizer[i/N : (i+1)/N]

Memory formula:
Per-GPU memory = model + gradients + optimizer / num_gpus
               = 42 + 14 + (56 / 4)
               = 70 GB (for 4 GPUs)
```

### 5.2 ZeRO Stage 2: Optimizer + Gradient Sharding

ZeRO Stage 2 extends sharding to both optimizer state and gradients.

**Memory Distribution:**
```
GPU 0: Model (42 GB) + Gradient[0:1/4] (3.5 GB) + Optimizer[0:1/4] (14 GB) = 59.5 GB
GPU 1: Model (42 GB) + Gradient[1/4:2/4] (3.5 GB) + Optimizer[1/4:2/4] (14 GB) = 59.5 GB
GPU 2: Model (42 GB) + Gradient[2/4:3/4] (3.5 GB) + Optimizer[2/4:3/4] (14 GB) = 59.5 GB
GPU 3: Model (42 GB) + Gradient[3/4:4/4] (3.5 GB) + Optimizer[3/4:4/4] (14 GB) = 59.5 GB

Total: 238 GB (vs 448 GB without ZeRO, 280 GB with ZeRO-1)
Memory savings per GPU: (112 - 59.5) / 112 = 46.9%
```

**Communication Complexity:**
```
Backward pass gradient computation:
1. Each layer computes gradients (distributed)
2. Reduce-scatter: each GPU collects its gradient partition
3. Requires gradient communication across all layers

Backward communication:
- Reduce-scatter at each backward layer (L × communication)
- All-gather to reconstruct full gradients for next training step
```

### 5.3 ZeRO Stage 3: Full Parameter Sharding

ZeRO Stage 3 shards model parameters as well, enabling training of arbitrarily large models.

**Parameter Sharding:**
```
GPU 0: Model[0:1/4] (10.5 GB) + Gradient[0:1/4] (3.5 GB) + Optimizer[0:1/4] (14 GB) = 28 GB
GPU 1: Model[1/4:2/4] (10.5 GB) + Gradient[1/4:2/4] (3.5 GB) + Optimizer[1/4:2/4] (14 GB) = 28 GB
GPU 2: Model[2/4:3/4] (10.5 GB) + Gradient[2/4:3/4] (3.5 GB) + Optimizer[2/4:3/4] (14 GB) = 28 GB
GPU 3: Model[3/4:4/4] (10.5 GB) + Gradient[3/4:4/4] (3.5 GB) + Optimizer[3/4:4/4] (14 GB) = 28 GB

Total: 112 GB (vs 448 GB without ZeRO)
Memory savings per GPU: (112 - 28) / 112 = 75%

Enables training models up to 4× larger than single GPU can hold
```

**Forward Pass Communication:**
```
During forward computation of layer i:
1. All-gather: reconstruct full model weights from all GPUs
2. Compute forward pass for layer i
3. Drop model weights after computation (save memory)
4. Repeat for next layer

All-gather communication: P / (num_gpus) per layer
Total communication: L × [P / (num_gpus)]
```

**Memory Reduction Summary:**
```
Configuration         | Memory/GPU | Reduction
---------------------|-----------|----------
No sharding           | 112 GB    | 0%
ZeRO-1 (Opt state)   | 70 GB     | 37.5%
ZeRO-2 (Opt + Grad)  | 59.5 GB   | 46.9%
ZeRO-3 (Full)        | 28 GB     | 75%
```

### 5.4 ZeRO-Offload: CPU Acceleration

ZeRO-Offload moves optimizer states to CPU memory, reducing GPU memory:

**GPU + CPU Hybrid:**
```
GPU 0: Model (42 GB) + Gradient (3.5 GB) = 45.5 GB
CPU:   Optimizer (56 GB)

GPU memory: 45.5 GB (feasible on 80GB GPU)
CPU memory: 56 GB (typical system has 500GB+)

Tradeoff: CPU-GPU communication overhead (~PCI-E 4.0 bandwidth)
```

## 6. DeepSpeed Configuration & Integration

### 6.1 DeepSpeed Architecture

DeepSpeed is a PyTorch extension providing ZeRO, model parallelism, and optimization utilities.

**Configuration File (config.json):**
```json
{
  "train_batch_size": 32,
  "train_micro_batch_size_per_gpu": 8,
  "gradient_accumulation_steps": 4,

  "zero_optimization": {
    "stage": 2,
    "offload_optimizer": {
      "device": "cpu",
      "pin_memory": true
    },
    "offload_param": {
      "device": "cpu"
    },
    "reduce_bucket_size": 2e8,
    "stage3_prefetch_bucket_size": 2e8,
    "stage3_param_persistence_threshold": 1e4
  },

  "optimizer": {
    "type": "AdamW",
    "params": {
      "lr": 1e-4,
      "betas": [0.9, 0.999],
      "eps": 1e-8,
      "weight_decay": 0.01
    }
  },

  "scheduler": {
    "type": "WarmupLR",
    "params": {
      "warmup_min_lr": 0,
      "warmup_max_lr": 1e-4,
      "warmup_num_steps": 1000
    }
  },

  "fp16": {
    "enabled": true,
    "fp16_master_weights_and_grads": false,
    "loss_scale": 0,
    "loss_scale_window": 1000,
    "hysteresis": 2,
    "min_loss_scale": 1,
    "initial_scale_power": 16
  }
}
```

### 6.2 Training Loop Integration

**DeepSpeed Training Pattern:**
```python
import deepspeed
import torch

def main():
    # Initialize model
    model = load_model()

    # Initialize DeepSpeed
    model_engine, optimizer, _, _ = deepspeed.initialize(
        args=args,
        model=model,
        model_parameters=[p for p in model.parameters() if p.requires_grad],
        config=config_dict
    )

    # Training loop
    for epoch in range(num_epochs):
        for step, batch in enumerate(dataloader):
            inputs = batch['input_ids'].to(model_engine.device)
            labels = batch['labels'].to(model_engine.device)

            # Forward pass
            outputs = model_engine(inputs, labels=labels)
            loss = outputs.loss

            # Backward pass (automatic with DeepSpeed)
            model_engine.backward(loss)

            # Optimizer step
            model_engine.step()

            if step % log_interval == 0:
                print(f"Step {step}, Loss {loss.item()}")

    # Save checkpoint
    model_engine.save_checkpoint(save_dir)

if __name__ == '__main__':
    main()
```

### 6.3 Parameter Tuning Guidelines

**Key Configuration Parameters:**

| Parameter | Recommendation | Impact |
|-----------|---------------|--------|
| `stage` | 2 or 3 | Stage 2: 47% memory savings; Stage 3: 75% savings |
| `reduce_bucket_size` | 2×10^8 - 5×10^8 | Balance communication granularity |
| `micro_batch_size` | Largest fitting in GPU | Higher = Better throughput; Lower = More gradient accumulation |
| `gradient_accumulation_steps` | 4-16 | Simulate larger batch without memory overhead |
| `loss_scale` | 2^16 or 2^15 | Depends on loss magnitude and precision |

**Memory Estimation:**
```python
def estimate_deepspeed_memory(
    num_params_billions,
    batch_size_per_gpu,
    seq_length,
    zero_stage,
    gradient_accumulation=1
):
    """
    Estimate GPU memory with DeepSpeed
    """
    p = num_params_billions * 1e9

    # Model weights
    model_memory = p * 2 / 1e9  # FP16

    # Gradients
    grad_memory = p * 2 / 1e9

    # Optimizer (AdamW)
    if zero_stage == 1:
        opt_memory = p * 2 * 2 / 1e9  # 2x state, sharded across GPUs
    elif zero_stage == 2:
        opt_memory = p * 2 * 2 / (4 * 1e9)  # Assume 4 GPUs
    elif zero_stage == 3:
        opt_memory = p * 2 * 2 / (4 * 1e9)  # Fully sharded

    # Activations (simplified)
    activation_memory = batch_size_per_gpu * seq_length * p / (1e9 * 1e9) * 200

    total = model_memory + grad_memory + opt_memory + activation_memory

    return total
```

## 7. FSDP (Fully Sharded Data Parallel)

### 7.1 FSDP Architecture

FSDP is PyTorch's native implementation of fully sharded data parallelism (ZeRO Stage 3 equivalent).

**FSDP Sharding Strategy:**
```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
from functools import partial

def setup_fsdp_model(model, world_size):
    """Configure FSDP with size-based sharding"""

    # Wrap policy: shard modules > 100M parameters
    auto_wrap_policy = partial(
        size_based_auto_wrap_policy,
        min_num_params=100e6  # 100M threshold
    )

    # Initialize FSDP
    model = FSDP(
        model,
        auto_wrap_policy=auto_wrap_policy,
        sharding_strategy=ShardingStrategy.FULL_SHARD,  # Full parameter sharding
        mixed_precision=mixed_precision_policy,
        device_id=torch.cuda.current_device(),
        limit_all_gathers=True,
        use_orig_params=True,
    )

    return model

# Memory with FSDP (4 GPUs):
# Per-GPU: (42 + 14) / 4 + (56 / 4) = 14 + 14 = 28 GB
```

### 7.2 Sharding Strategies

**SHARD_PER_LAYER:**
```python
# Shard parameters only when needed (all-gather in forward, drop after)
model = FSDP(
    model,
    sharding_strategy=ShardingStrategy.SHARD_PER_LAYER,
    # All parameters sharded within each layer
)
```

**HYBRID_SHARD:**
```python
# Shard across multiple process groups
from torch.distributed.fsdp import ShardingStrategy

model = FSDP(
    model,
    sharding_strategy=ShardingStrategy.HYBRID_SHARD,
    process_groups=(backend_process_group, intra_node_process_group),
    # Reduce inter-node communication overhead
)
```

### 7.3 Synchronization & Gradient Accumulation

**Gradient Accumulation with FSDP:**
```python
def train_with_grad_accumulation(
    model, train_loader, num_accumulation_steps=4
):
    """FSDP training with gradient accumulation"""

    accumulation_counter = 0

    for batch in train_loader:
        inputs = batch['input_ids'].cuda()
        labels = batch['labels'].cuda()

        # Forward pass
        outputs = model(inputs, labels=labels)
        loss = outputs.loss

        # Normalize loss by accumulation steps
        loss = loss / num_accumulation_steps

        # Backward (accumulates gradients)
        loss.backward()

        accumulation_counter += 1

        if accumulation_counter == num_accumulation_steps:
            # Only sync gradients every num_accumulation_steps
            with model.no_sync() if torch.distributed.is_initialized() else contextlib.nullcontext():
                pass

            # Optimizer step
            optimizer.step()
            optimizer.zero_grad()
            accumulation_counter = 0
```

### 7.4 Wrap Policy Configuration

**Custom Wrap Policy:**
```python
def custom_wrap_policy(module, recurse, nonwrapped_numel):
    """
    Custom FSDP wrapping: shard transformer blocks separately
    """
    if recurse:
        return True

    # Check if module is a transformer block
    if isinstance(module, TransformerBlock):
        return True

    return False

model = FSDP(
    model,
    auto_wrap_policy=custom_wrap_policy,
    # Each TransformerBlock sharded independently
)
```

## 8. Activation Recomputation

### 8.1 Selective Activation Recomputation

Activation recomputation trades compute for memory during backward pass.

**Recomputation Patterns:**

```python
class SelectiveCheckpointedBlock(torch.nn.Module):
    def __init__(self, block, checkpoint_ratio=0.5):
        super().__init__()
        self.block = block
        self.checkpoint_ratio = checkpoint_ratio

    def forward(self, x, attention_mask=None):
        if self.training and random.random() < self.checkpoint_ratio:
            # Checkpoint this block
            return checkpoint(
                self.block.forward,
                x,
                attention_mask,
                use_reentrant=False
            )
        else:
            # Regular forward
            return self.block(x, attention_mask)
```

### 8.2 Attention-Specific Recomputation

Attention layers dominate memory. Specialized recomputation improves memory-compute tradeoff:

```python
class CheckpointedAttention(torch.nn.Module):
    def __init__(self, num_heads, hidden_dim):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim

    def forward(self, query, key, value, attention_mask=None):
        # Attention pattern:
        # Attention = softmax(QK^T / sqrt(d)) V
        # Most memory: QK^T (seq_len^2)

        # Recompute softmax during backward, store only logits
        if self.training:
            # Forward: compute attention weights
            scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.hidden_dim)

            if attention_mask is not None:
                scores = scores + attention_mask

            # Save scores, don't save softmax
            attn_weights = torch.softmax(scores, dim=-1)

            # Mark softmax for recomputation
            attn_weights.requires_grad_(True)

        output = torch.matmul(attn_weights, value)

        return output
```

## 9. Advanced Optimization Techniques

### 9.1 Layer-wise Learning Rate Decay

Decaying learning rates across layers improves fine-tuning stability:

```python
def get_layer_wise_learning_rates(
    model,
    base_lr=1e-4,
    decay_rate=0.9
):
    """
    Lower layers (pre-trained) get lower LR
    Higher layers (task-specific) get higher LR
    """
    param_groups = []

    for layer_idx, (name, params) in enumerate(get_model_layers(model)):
        # Exponential decay: earlier layers have lower LR
        lr = base_lr * (decay_rate ** (num_layers - layer_idx))

        param_groups.append({
            'params': list(params),
            'lr': lr,
            'name': name
        })

    return param_groups

# Usage
optimizer = torch.optim.AdamW(
    get_layer_wise_learning_rates(model),
    weight_decay=0.01
)
```

### 9.2 Warm-up Scheduling

Linear warm-up improves training stability in early epochs:

```python
from torch.optim.lr_scheduler import LambdaLR

def warmup_lambda(current_step, total_steps, warmup_steps=1000):
    if current_step < warmup_steps:
        return float(current_step) / float(max(1, warmup_steps))
    return max(0.0, float(total_steps - current_step) / float(max(1, total_steps - warmup_steps)))

scheduler = LambdaLR(
    optimizer,
    lr_lambda=lambda step: warmup_lambda(
        step,
        total_steps=num_training_steps,
        warmup_steps=num_warmup_steps
    )
)
```

## 10. Summary & Best Practices

### 10.1 Memory Optimization Checklist

- [ ] Calculate baseline memory footprint (model, gradient, optimizer, activation)
- [ ] Enable mixed precision (BF16 preferred for stability)
- [ ] Implement gradient checkpointing (aim for ~1-2% compute overhead)
- [ ] Enable ZeRO Stage 2/3 for multi-GPU training
- [ ] Use FSDP for native PyTorch deployment
- [ ] Configure gradient accumulation for larger effective batch sizes
- [ ] Monitor loss scaling behavior to prevent gradient underflow

### 10.2 Performance Optimization Checklist

- [ ] Profile baseline throughput (tokens/second)
- [ ] Benchmark each optimization (memory, compute overhead)
- [ ] Validate accuracy on downstream tasks (< 0.5% accuracy loss typical)
- [ ] Use layer-wise learning rate decay for pre-trained models
- [ ] Implement warm-up scheduling for stable convergence
- [ ] Monitor gradient norms and clip if needed (max norm 1.0 typical)

### 10.3 Production Deployment

**Recommended Configuration for 7B LLM:**
```json
{
  "micro_batch_size": 8,
  "gradient_accumulation_steps": 4,
  "zero_stage": 2,
  "mixed_precision": "bf16",
  "gradient_checkpointing": true,
  "checkpoint_interval": 1,
  "learning_rate": 1e-4,
  "warmup_steps": 1000,
  "max_grad_norm": 1.0
}
```

Expected memory/GPU (4×A100-80GB):
- Peak: ~65GB per GPU
- Steady-state: ~70GB per GPU

**Key Reading:**
- DeepSpeed documentation: https://www.deepspeed.ai/
- PyTorch FSDP: https://pytorch.org/docs/stable/fsdp.html
- Mixed Precision Training: https://arxiv.org/abs/1710.03740

---

**Module Completion Status**: Comprehensive coverage of fine-tuning infrastructure from fundamental memory analysis through production optimization techniques.
