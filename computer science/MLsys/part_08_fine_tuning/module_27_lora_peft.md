# MODULE 27 — LoRA & Parameter-Efficient Fine-Tuning: Systems Perspective

## 1. Introduction & Learning Objectives

Parameter-Efficient Fine-Tuning (PEFT) enables task-specific adaptation of large language models while updating only a small fraction of parameters. Low-Rank Adaptation (LoRA), introduced by Hu et al. (ICLR 2022), has emerged as the dominant PEFT technique in production systems, enabling efficient multi-model serving with minimal overhead. This module provides systems-level analysis of LoRA and variants, with emphasis on mathematical foundations, implementation mechanics, memory analysis, and production deployment patterns.

**Learning Objectives:**
- Understand LoRA's low-rank decomposition mathematics and rank selection strategies
- Master LoRA implementation including weight injection, forward/backward pass mechanics
- Analyze LoRA merging for zero-overhead inference
- Comprehend QLoRA's quantization strategies and memory savings
- Compare LoRA variants (DoRA, LoRA+, VeRA, GaLore) with systems implications
- Design multi-LoRA serving systems (S-LoRA, Punica)
- Implement LoRA on CPU with memory-efficient inference

## 2. LoRA Fundamentals

### 2.1 Low-Rank Decomposition Mathematics

LoRA decomposes weight updates into low-rank matrices, reducing trainable parameters:

**Standard Fine-Tuning:**
```
Weight matrix W (d_out × d_in) requires d_out × d_in parameters
Example: 4096 × 4096 = 16,777,216 parameters

Gradient computation:
dW = learning_rate × gradient_tensor  (same shape as W)
Memory for gradients: 16M parameters × dtype_size
```

**LoRA Decomposition:**
```
W' = W_0 + BA

Where:
- W_0: Original frozen weight matrix (4096 × 4096)
- B: Low-rank decomposition matrix (4096 × r)
- A: Low-rank decomposition matrix (r × 4096)
- r: Rank (typically 8, 16, 32, or 64)

Parameter count with LoRA:
= B parameters + A parameters
= (4096 × r) + (r × 4096)
= 2 × 4096 × r
= 8192 × r

For r=8: 65,536 parameters (99.6% reduction from 16.7M)
For r=64: 524,288 parameters (96.9% reduction)
```

**Mathematical Formulation:**
```
Forward pass output:
y = f(x; W_0 + BA)
  = f(x; W_0) + f'(x; BA)  (approximation)

Constraint: rank(BA) << min(d_out, d_in)
This constraint enables parameter efficiency

Gradient computation during training:
dB = gradient_for_B_matrix  (4096 × r)
dA = gradient_for_A_matrix  (r × 4096)

Frozen gradients:
dW_0 = 0  (not updated)
```

### 2.2 Rank Selection Strategy

Choosing rank r involves fundamental tradeoff between parameter efficiency and expressiveness:

**Empirical Rank Selection:**
```python
def analyze_optimal_rank(
    model_dim=4096,
    task_complexity='moderate',  # 'simple', 'moderate', 'complex'
):
    """
    Heuristic rank selection based on task
    """

    # Baseline: rank proportional to model dimension
    # Typical range: 1-5% of model dimension

    if task_complexity == 'simple':
        # Domain-specific but related task
        optimal_rank = int(model_dim * 0.01)  # ~1%
        # Example: 4096 × 0.01 = 40

    elif task_complexity == 'moderate':
        # Standard fine-tuning task
        optimal_rank = int(model_dim * 0.02)  # ~2%
        # Example: 4096 × 0.02 = 82 (round to 64)

    else:  # 'complex'
        # Multi-task, significant distribution shift
        optimal_rank = int(model_dim * 0.05)  # ~5%
        # Example: 4096 × 0.05 = 204

    return optimal_rank

# Typical rank values for different model sizes:
# 7B model (4096 hidden): rank 64-256
# 13B model (5120 hidden): rank 64-256
# 70B model (8192 hidden): rank 128-512
```

**Theoretical Analysis - Intrinsic Dimension:**
```
LoRA assumes: Δ W ≈ BA (low-rank approximation)

Definition (Intrinsic Dimension):
The minimum rank r such that ||W - W_0 - BA||_F ≤ ε

Theorem (Li et al., 2018):
Most natural language tasks have intrinsic dimension << model dimension
- Simple tasks: intrinsic_dim ≈ 1-10
- Moderate tasks: intrinsic_dim ≈ 10-100
- Complex tasks: intrinsic_dim ≈ 100-1000

Therefore:
- r should be > intrinsic dimension to fit task
- r << min(d_out, d_in) to preserve parameter efficiency
- Practical: r in [8, 64, 128, 256] covers most tasks
```

### 2.3 Alpha Scaling & Initialization

LoRA includes scaling factor α to control adaptation magnitude:

**Scaled Update Rule:**
```
y = f(x; W_0 + (α/r) × BA)

Where α/r normalizes update magnitude

Effect of α:
- Small α: Weak adaptation, close to original model
- Large α: Strong adaptation, may overfit
- Typical α: 16 (scales to α/r ≈ 1 for r=16)
```

**Initialization Strategy:**
```python
def initialize_lora_weights(
    B_matrix,
    A_matrix,
    rank,
    scaling_alpha=16
):
    """
    Standard LoRA initialization
    """

    # B matrix: Gaussian initialization
    torch.nn.init.kaiming_uniform_(B_matrix, a=math.sqrt(5))

    # A matrix: Zero initialization
    torch.nn.init.zeros_(A_matrix)

    # Effect: At initialization, LoRA updates are near-zero
    # Update magnitude: (α/r) × (0) = 0
    # This preserves original model at start of training

    # Practical implementation
    scaling_factor = scaling_alpha / rank

    with torch.no_grad():
        B_matrix.mul_(scaling_factor)

    return B_matrix, A_matrix
```

**Scaling Impact on Training:**
```
Training dynamics:
dA ∝ input_activation × (output_gradient)
dB ∝ output_gradient × A

Effect of α/r scaling:
- Smaller α/r: Smaller gradient magnitude, slower learning
- Larger α/r: Larger gradient magnitude, faster learning but risk of instability

Empirical evidence:
α/r in [0.1, 1.0] works well for most tasks
Default: α = 16, r = 16 → α/r = 1.0
```

## 3. LoRA Implementation Details

### 3.1 Weight Injection Patterns

LoRA weights must be integrated into model forward pass. Two main approaches:

**Direct Injection (Modifying Model Architecture):**
```python
class LoRALinear(torch.nn.Module):
    def __init__(self, in_features, out_features, rank=8, alpha=16):
        super().__init__()

        # Original weight matrix
        self.original_linear = torch.nn.Linear(in_features, out_features)
        self.original_linear.weight.requires_grad = False
        self.original_linear.bias.requires_grad = False

        # LoRA adaptation matrices
        self.lora_rank = rank
        self.lora_alpha = alpha
        self.scaling = alpha / rank

        # B matrix (out_features × rank)
        self.lora_B = torch.nn.Parameter(
            torch.randn(out_features, rank) * (1.0 / math.sqrt(rank))
        )

        # A matrix (rank × in_features)
        self.lora_A = torch.nn.Parameter(torch.zeros(rank, in_features))

    def forward(self, x):
        # Original forward pass
        y_original = self.original_linear(x)

        # LoRA forward pass: x @ A^T @ B^T
        # Shape: (..., in_features) @ (in_features, rank) @ (rank, out_features)
        #      = (..., out_features)
        y_lora = (x @ self.lora_A.T @ self.lora_B.T) * self.scaling

        return y_original + y_lora
```

**Parameter Wrapper (Non-Invasive Adaptation):**
```python
class LoRAWrapper(torch.nn.Module):
    def __init__(self, original_layer, rank=8, alpha=16):
        super().__init__()

        self.original_layer = original_layer
        self.rank = rank
        self.alpha = alpha

        # Extract dimensions from original layer
        if isinstance(original_layer, torch.nn.Linear):
            in_features = original_layer.in_features
            out_features = original_layer.out_features

            # Create LoRA matrices
            self.lora_B = torch.nn.Parameter(
                torch.randn(out_features, rank) / math.sqrt(rank)
            )
            self.lora_A = torch.nn.Parameter(torch.zeros(rank, in_features))

    def forward(self, x):
        # Original forward
        y_original = self.original_layer(x)

        # Add LoRA adaptation
        y_lora = (x @ self.lora_A.T @ self.lora_B.T) * (self.alpha / self.rank)

        return y_original + y_lora

# Usage
def add_lora_to_model(model, rank=8):
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            # Replace linear layers with LoRA-wrapped versions
            parent_name = name.rsplit('.', 1)[0]
            child_name = name.rsplit('.', 1)[1]

            parent_module = dict(model.named_modules())[parent_name]
            setattr(parent_module, child_name, LoRAWrapper(module, rank=rank))
```

### 3.2 Forward Pass Mechanics

**Computation Graph:**
```
Input x: [batch, seq_len, in_features]

Path 1 (Original):
x → W_0 → y_original: [batch, seq_len, out_features]

Path 2 (LoRA):
x → A → [batch, seq_len, rank]
    ↓
    B → [batch, seq_len, out_features]
    ↓
    (scale by α/r)
    ↓
    y_lora: [batch, seq_len, out_features]

Output:
y = y_original + y_lora

Memory footprint during forward:
- Input buffer: batch × seq_len × in_features × dtype_size
- Intermediate (rank): batch × seq_len × rank × dtype_size (smaller)
- Output buffer: batch × seq_len × out_features × dtype_size
```

**Computational Complexity:**
```
Standard forward (W_0 only):
FLOPs = batch × seq_len × in_features × out_features
      = B × T × d × d (typically d=4096)
      = 32 × 2048 × 4096 × 4096
      ≈ 1.1 × 10^12 FLOPS

LoRA additional forward:
FLOPs = B × T × (in_features × rank + rank × out_features)
      = B × T × rank × (in_features + out_features)
      = 32 × 2048 × 16 × 8192
      ≈ 8.6 × 10^9 FLOPS

LoRA overhead: (8.6 / 1100) ≈ 0.78% (negligible for r=16)
```

### 3.3 Backward Pass & Gradient Flow

**Gradient Computation:**
```
Forward:
y = W_0 @ x + (α/r) × (B @ (A @ x))

Backward (chain rule):
∂L/∂A = ∂L/∂y @ ∂y/∂(B @ A @ x) @ ∂(B @ A @ x)/∂A
      = (∂L/∂y) @ (α/r) @ B^T @ x^T

∂L/∂B = (∂L/∂y) @ (α/r) @ (A @ x)^T

∂L/∂x = ∂L/∂y @ W_0^T + ∂L/∂y @ (α/r) @ B @ A

Frozen gradients:
∂L/∂W_0 = 0  (not updated)
```

**Memory-Efficient Gradient Accumulation:**
```python
class LoRABackwardOptimization:
    """
    Gradient computation optimized for LoRA
    """

    @staticmethod
    def compute_gradients_lora(x, output_grad, rank, alpha):
        """
        Compute ∂L/∂A and ∂L/∂B efficiently
        """

        # Compute B^T @ output_grad first (reuse)
        # Shape: [rank, out_features] @ [out_features, batch*seq]
        #      = [rank, batch*seq]
        grad_common = output_grad @ alpha / rank

        # Gradient for A: reshape for batch processing
        # ∂L/∂A = grad_common @ x^T
        grad_A = grad_common @ x.T  # [rank, in_features]

        # Gradient for B: implicit in grad_common
        # ∂L/∂B = output_grad @ (A@x)^T
        # Delay this computation to avoid intermediate storage
        grad_B_needs_A_output = True

        return grad_A, grad_B_needs_A_output
```

### 3.4 Multi-Head Adaptation

Applying LoRA to multiple layers with shared computation:

```python
class MultiHeadLoRA(torch.nn.Module):
    def __init__(self, module_list, rank=8, alpha=16):
        super().__init__()

        self.modules = module_list
        self.rank = rank
        self.alpha = alpha / rank

        # Shared LoRA A matrix (can be task-specific)
        self.lora_A = torch.nn.ParameterList([
            torch.nn.Parameter(torch.zeros(rank, module.in_features))
            for module in module_list
        ])

        # Per-module B matrices
        self.lora_B = torch.nn.ParameterList([
            torch.nn.Parameter(
                torch.randn(module.out_features, rank) / math.sqrt(rank)
            )
            for module in module_list
        ])

    def forward(self, x_list):
        """
        x_list: List of inputs for each module

        Memory efficiency:
        - Share A matrices (single computation)
        - Separate B matrices (necessary for different output shapes)
        """

        outputs = []
        for x, module, lora_A, lora_B in zip(
            x_list, self.modules, self.lora_A, self.lora_B
        ):
            y_original = module(x)
            y_lora = (x @ lora_A.T @ lora_B.T) * self.alpha
            outputs.append(y_original + y_lora)

        return outputs
```

## 4. LoRA Merging for Zero-Overhead Inference

### 4.1 Weight Merging Mechanics

Merging LoRA weights into original weights eliminates inference overhead:

**Merge Operation:**
```
W_merged = W_0 + (α/r) × B @ A

This is exact (no approximation):
W_merged is simply a new weight matrix with same shape as W_0

Forward with merged weights:
y = W_merged @ x
  = (W_0 + (α/r) × B @ A) @ x
  = W_0 @ x + (α/r) × B @ (A @ x)
  = y_original + y_lora (identical to unmerged)
```

**Merge Algorithm:**
```python
def merge_lora_to_weights(original_weight, lora_B, lora_A, alpha, rank):
    """
    Merge LoRA into original weight matrix

    Args:
        original_weight: [out_features, in_features]
        lora_B: [out_features, rank]
        lora_A: [rank, in_features]
        alpha: scaling factor
        rank: LoRA rank

    Returns:
        merged_weight: [out_features, in_features] (same shape as original)
    """

    # Compute BA (element-wise same as GEMM)
    BA = lora_B @ lora_A  # [out_features, in_features]

    # Scale and add to original
    merged_weight = original_weight + (alpha / rank) * BA

    return merged_weight

# After merging:
# 1. Replace original weights with merged_weight
# 2. Delete lora_A and lora_B parameters (save memory)
# 3. Inference uses standard weight matrix (no LoRA overhead)

# Memory after merge:
# Before: original_weight + lora_A + lora_B
#       = out*in + in*r + out*r
#       = out*in + r*(in+out)
#
# After: merged_weight only
#      = out*in (exact same size as original!)
#
# Savings: r*(in+out) parameters no longer needed
```

### 4.2 Inference-Time Optimization

**Merged Weight Storage:**
```python
class MergedLoRAModel(torch.nn.Module):
    def __init__(self, original_model, lora_weights_dict):
        super().__init__()

        # Copy original model
        self.model = copy.deepcopy(original_model)

        # Merge LoRA weights into model
        for name, (lora_B, lora_A, alpha, rank) in lora_weights_dict.items():
            # Get original weight from model
            original_weight = self._get_weight_by_name(name)

            # Compute merged weight
            merged = merge_lora_to_weights(
                original_weight, lora_B, lora_A, alpha, rank
            )

            # Replace in model
            self._set_weight_by_name(name, merged)

        # Freeze all parameters
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, x):
        # Standard inference, no LoRA computation
        return self.model(x)

    def get_model_size(self):
        """Merged model has no LoRA overhead"""
        return sum(p.numel() * p.element_size() for p in self.parameters())
```

### 4.3 Production Deployment Pattern

**Multi-Task Serving with Merged LoRA:**
```python
class MultiTaskLoRAServer:
    def __init__(self, base_model, task_configs):
        super().__init__()

        self.base_model = base_model
        self.task_models = {}  # task_name -> merged_model

        # Merge LoRA for each task
        for task_name, lora_weights in task_configs.items():
            merged_model = MergedLoRAModel(base_model, lora_weights)
            self.task_models[task_name] = merged_model

    def forward(self, x, task_name):
        """
        Inference for specific task (no LoRA overhead)
        """
        model = self.task_models[task_name]
        return model(x)

    def benchmark_inference(self, batch_size=32, seq_length=512):
        """
        Verify zero LoRA overhead in merged models
        """
        x = torch.randn(batch_size, seq_length, 4096).cuda()

        # Time base model
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(100):
            self.base_model(x)
        torch.cuda.synchronize()
        base_time = time.time() - start

        # Time task-specific model (should be identical)
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(100):
            self.task_models['task_1'](x)
        torch.cuda.synchronize()
        task_time = time.time() - start

        # Overhead should be negligible (< 1%)
        overhead = (task_time - base_time) / base_time * 100
        print(f"LoRA overhead after merging: {overhead:.2f}%")
```

## 5. QLoRA: Quantized LoRA

### 5.1 Quantization Components

QLoRA (Dettmers et al., NeurIPS 2023) combines quantization with LoRA for extreme efficiency:

**QLoRA Pipeline:**
```
Full-precision original model (7B LLM = 28 GB)
    ↓
Quantize to NF4 (nearly 4-bit)
    ↓
Store quantized weights (7B × 0.25 bytes ≈ 1.75 GB)
    ↓
During training:
  - Load quantized weights to GPU as NF4
  - Dequantize on-the-fly to higher precision (BF16)
  - Compute LoRA updates only
    ↓
Gradient storage: LoRA matrices only (negligible)
    ↓
Total training memory: ~6 GB (vs. 145 GB for full fine-tuning)
```

### 5.2 NF4 Quantization Format

NF4 (Normalized Float 4) is a specialized 4-bit format for LLM weights:

**NF4 Format:**
```
Standard INT4 quantization:
[Sign: 1 bit] [Exponent: 2 bits] [Mantissa: 1 bit]
Issue: Symmetrical around zero, wastes sign bit for weights (mostly negative)

NF4 approach:
- Analyze weight distributions (typically long-tailed)
- Create optimal quantization grid based on empirical distribution
- Non-uniform quantization levels (more precision in dense regions)

Example quantization grid for weight range [-2.0, 2.0]:
{-1.0, -0.8, -0.5, -0.2, 0.0, 0.2, 0.5, 0.8, 1.0, ...}
(16 levels total = 2^4)

Inverse quantization:
x_fp8 = quantized_value × scale_factor
```

**NF4 Quantization Code:**
```python
def quantize_to_nf4(weight_tensor):
    """
    Quantize weight to NF4 format
    """

    # Analyze distribution
    shape = weight_tensor.shape
    weight_flat = weight_tensor.reshape(-1)

    # Compute quantization scale (per-channel)
    absmax = weight_flat.abs().max()
    scale = absmax / 7.0  # NF4 range = [-7, 7]

    # Round to nearest NF4 level
    NF4_LEVELS = [-1.0, -0.8, -0.5, -0.2, 0.0, 0.2, 0.5, 0.8, 1.0, ...]

    normalized = weight_flat / scale
    quantized = torch.round(normalized * 7.0).clamp(-7, 7)

    # Pack into 4-bit representation
    # 2 values per byte
    packed = torch.zeros(len(quantized) // 2, dtype=torch.uint8)
    packed = (quantized[0::2].int() + 7) << 4
    packed |= (quantized[1::2].int() + 7)

    return packed.reshape(shape[0], -1), scale

def dequantize_from_nf4(quantized_tensor, scale):
    """
    Dequantize NF4 back to floating point
    """

    # Unpack from 4-bit
    high_nibble = (quantized_tensor >> 4).float() - 7.0
    low_nibble = (quantized_tensor & 0x0F).float() - 7.0

    # Map to NF4 values
    NF4_VALUES = [...list of NF4 grid points...]
    dequantized = torch.stack([
        torch.tensor([NF4_VALUES[int(x)] for x in high_nibble.flatten()]),
        torch.tensor([NF4_VALUES[int(x)] for x in low_nibble.flatten()])
    ])

    return dequantized * scale
```

### 5.3 Double Quantization & Paged Optimizer

**Double Quantization (Quantizing Quantization Statistics):**
```
Standard quantization:
Weight matrix W (fp32) → scale_factor (fp32)
Problem: scale_factor itself is fp32 (20 bits overhead for fp32)

Double quantization:
Weight matrix W → scale_1 (fp8)
               → scale_of_scale_2 (fp8)

Memory savings:
- Weight: 4 bits per value (vs 32 bits)
- Scale: 8 bits (vs 32 bits per weight)
- Overhead reduction: 24 bits → 8 bits
```

**Paged Optimizer State (QLoRA specific):**
```
Problem: AdamW optimizer requires 2× model parameters for state
QLoRA solution: Page optimizer state to CPU memory, access on-demand

CPU memory:
- First moment (m): 2 bytes per parameter (BF16)
- Second moment (v): 2 bytes per parameter (BF16)
- Total: 4 bytes per parameter

GPU memory:
- Only current batch's optimizer states
- Page-in/page-out via PCIe

Trade-off: CPU-GPU communication overhead << memory savings
```

**Paged Optimizer Implementation:**
```python
class PagedAdamWOptimizer(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-4, page_size=10000):
        self.page_size = page_size
        self.optimizer_state_cpu = {}
        self.param_to_page_idx = {}

    def step(self, closure=None):
        """
        Optimizer step with paged state
        """

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                # Get page from CPU memory
                page_idx = self.param_to_page_idx[id(p)]
                m, v = self.load_page_from_cpu(page_idx)

                # Standard Adam update
                m.mul_(0.9).add_(p.grad, alpha=0.1)
                v.mul_(0.999).add_(p.grad ** 2, alpha=0.001)

                # Update parameter
                p.sub_(m / (v.sqrt() + 1e-8))

                # Save page back to CPU
                self.save_page_to_cpu(page_idx, m, v)
```

### 5.4 QLoRA Memory Calculation

**Detailed Memory Breakdown (7B Model):**

| Component | Size | Notes |
|-----------|------|-------|
| Quantized weights (NF4) | 1.75 GB | 7B params × 0.25 bytes |
| LoRA-A matrices (r=64) | 0.256 GB | 32 layers × 4096 × 64 × 2 bytes |
| LoRA-B matrices (r=64) | 0.256 GB | 32 layers × 4096 × 64 × 2 bytes |
| Optimizer state (CPU) | 56 GB | 7B params × 8 bytes (m, v) |
| Activations cache | 3-5 GB | Batch=4, seq_len=512 |
| **GPU Total** | **~6 GB** | Quantized model + LoRA + activations |
| **CPU Total** | **~56 GB** | Optimizer state only |

**QLoRA vs Full Fine-tuning Comparison:**

| Metric | Full FT | QLoRA | Reduction |
|--------|---------|-------|-----------|
| GPU Memory | 145 GB | 6 GB | 95.9% |
| Trainable Params | 7B | 0.5B | 92.9% |
| Training Speed | 1.0× | 0.95× | -5% (minimal) |
| Final Accuracy | 100% | 99.8% | -0.2% |

## 6. LoRA Variants

### 6.1 DoRA (Decomposed Low-Rank Adaptation)

DoRA (Kyathanahally et al., 2024) decomposes LoRA adaptation into magnitude and direction:

**DoRA Formulation:**
```
Standard LoRA:
W' = W_0 + BA

DoRA decomposition:
W' = W_0 + m × (BA / ||BA||)

Where:
- m: magnitude scaling vector (d_out,)
- BA / ||BA||: direction vector (normalized)

Effect: Decouple magnitude adaptation from direction
- Magnitude m: learned per-output
- Direction BA: low-rank shared adaptation
```

**DoRA Implementation:**
```python
class DoRA(torch.nn.Module):
    def __init__(self, original_layer, rank=8, alpha=16):
        super().__init__()

        self.original_layer = original_layer
        self.rank = rank
        self.alpha = alpha

        # Low-rank direction
        self.lora_B = torch.nn.Parameter(
            torch.randn(original_layer.out_features, rank) / math.sqrt(rank)
        )
        self.lora_A = torch.nn.Parameter(torch.zeros(rank, original_layer.in_features))

        # Magnitude scaling
        self.magnitude = torch.nn.Parameter(torch.ones(original_layer.out_features))

    def forward(self, x):
        y_original = self.original_layer(x)

        # Compute low-rank update
        y_lora_unnormalized = (x @ self.lora_A.T @ self.lora_B.T) * (self.alpha / self.rank)

        # Normalize by direction, scale by magnitude
        norm = torch.norm(y_lora_unnormalized, p=2, dim=-1, keepdim=True)
        y_lora_direction = y_lora_unnormalized / (norm + 1e-8)

        # Apply magnitude scaling
        y_lora = self.magnitude.unsqueeze(0) * y_lora_direction

        return y_original + y_lora
```

**DoRA Advantages:**
- Better convergence speed (empirical)
- Lower per-rank-step loss (improved gradient flow)
- Typical: 0.5-1% accuracy improvement over LoRA

### 6.2 LoRA+

LoRA+ (Zhang et al., 2024) uses different learning rates for A and B matrices:

**Motivation:**
```
Observation: A and B matrices learn at different rates
- A matrix: larger gradients, converges quickly
- B matrix: smaller gradients, benefits from longer training

Standard LoRA: same learning rate for both
Issue: Sub-optimal convergence
```

**LoRA+ Approach:**
```python
def get_lora_plus_param_groups(model, base_lr=1e-4, rank=64):
    """
    Different learning rates for A and B matrices
    """

    param_groups = []

    for name, param in model.named_parameters():
        if 'lora_A' in name:
            # A matrix: standard learning rate
            param_groups.append({'params': [param], 'lr': base_lr})
        elif 'lora_B' in name:
            # B matrix: higher learning rate (2x typical)
            param_groups.append({'params': [param], 'lr': base_lr * 2.0})

    return param_groups

# Typical settings:
# A learning rate: 1e-4
# B learning rate: 2e-4 (2x higher)

# Improvement: 0.5-1.5% accuracy gain
```

### 6.3 VeRA (Vector-based LoRA)

VeRA (Ahmadi-Asl et al., 2024) reduces LoRA parameters by sharing B across layers:

**VeRA Formulation:**
```
Standard LoRA per layer:
B_i (layer-specific): d_out × rank
A_i (layer-specific): rank × d_in

Total for 32 layers: 32 × (d_out × rank + rank × d_in)

VeRA (Shared B):
B (shared): d_out × rank (single matrix for all layers!)
A_i (layer-specific): rank × d_in

Total: 1 × (d_out × rank) + 32 × (rank × d_in)
     = d_out × rank + 32 × rank × d_in
     ≈ d_out × rank (dominant term)

Comparison:
- Standard LoRA: 32 × 2 × d × rank parameters
- VeRA: (1 + 32) × d × rank parameters
- Ratio: (1 + 32) / (32 × 2) ≈ 0.516x (50% fewer parameters)
```

**VeRA Implementation:**
```python
class VeRA(torch.nn.Module):
    def __init__(self, model, rank=8, alpha=16):
        super().__init__()

        self.rank = rank
        self.alpha = alpha

        # Single shared B matrix
        self.shared_B = torch.nn.Parameter(
            torch.randn(4096, rank) / math.sqrt(rank)
        )

        # Per-layer A matrices
        self.layer_A = torch.nn.ParameterList([
            torch.nn.Parameter(torch.zeros(rank, 4096))
            for _ in range(32)
        ])

    def forward(self, x, layer_idx):
        """
        Compute LoRA with shared B
        """
        y_original = self.original_forward(x)

        # Use shared B with layer-specific A
        y_lora = (x @ self.layer_A[layer_idx].T @ self.shared_B.T) * (self.alpha / self.rank)

        return y_original + y_lora
```

**VeRA Trade-offs:**
- Parameter reduction: 50% vs standard LoRA
- Empirical accuracy: -0.5-1% vs standard LoRA
- Use case: Memory-constrained fine-tuning on small devices

### 6.4 GaLore (Gradient-Aligned Low Rank)

GaLore (Zhang et al., 2024) decomposes gradients rather than weights:

**GaLore Approach:**
```
Instead of:
W' = W_0 + BA

GaLore:
- Decompose gradient updates into low-rank form
- Update low-rank gradient representation
- Reconstruct full gradient periodically

Key insight: Gradient updates often have intrinsic low rank
- Can apply to any optimizer
- Compatible with other techniques (quantization, etc.)
```

## 7. Multi-LoRA Serving Systems

### 7.1 S-LoRA (Serving Many LoRAs)

S-LoRA (Sheng et al., MLSys 2024) enables efficient serving of many LoRA adapters simultaneously:

**Multi-LoRA Architecture:**
```
Base model: 7B parameters (shared across all tasks)
LoRA adapter 1: 0.5M parameters (task_1)
LoRA adapter 2: 0.5M parameters (task_2)
...
LoRA adapter N: 0.5M parameters (task_N)

Traditional approach:
- Load base model once
- Load adapter for request
- Inference
- Unload adapter
- Load next adapter
Overhead: Adapter load/unload delays

S-LoRA approach:
- Load base model once
- Cache all adapters in GPU memory
- Dynamic batching across requests with different adapters
```

**S-LoRA Serving System:**
```python
class SLoRAServer:
    def __init__(self, base_model, max_adapters=32):
        self.base_model = base_model
        self.adapter_cache = {}
        self.request_queue = []
        self.max_adapters = max_adapters

    def register_adapter(self, task_id, lora_B, lora_A, alpha, rank):
        """Register LoRA adapter for task"""
        if len(self.adapter_cache) >= self.max_adapters:
            # Evict least-recently-used adapter
            lru_task = min(self.adapter_cache, key=lambda t: self.adapter_cache[t]['last_used'])
            del self.adapter_cache[lru_task]

        self.adapter_cache[task_id] = {
            'lora_B': lora_B,
            'lora_A': lora_A,
            'alpha': alpha,
            'rank': rank,
            'last_used': time.time()
        }

    def forward(self, x, task_ids, batch_size):
        """
        Batch forward across multiple tasks
        task_ids: [task_1, task_2, task_1, task_3, ...]
        """

        # Group requests by task for batching
        task_groups = {}
        for idx, task_id in enumerate(task_ids):
            if task_id not in task_groups:
                task_groups[task_id] = []
            task_groups[task_id].append(idx)

        outputs = torch.zeros_like(x)

        # Process each task group
        for task_id, indices in task_groups.items():
            adapter = self.adapter_cache[task_id]
            x_task = x[indices]

            # Forward with adapter
            y_base = self.base_model(x_task)
            y_adapter = (x_task @ adapter['lora_A'].T @ adapter['lora_B'].T) * (adapter['alpha'] / adapter['rank'])

            outputs[indices] = y_base + y_adapter
            adapter['last_used'] = time.time()

        return outputs
```

**Memory Analysis:**
```
Base model: 28 GB
32 LoRA adapters (rank=64 each):
- Per-adapter: 2 × 4096 × 64 × 2 bytes = 1 MB (negligible)
- All adapters: 32 MB

Total GPU memory: 28 GB + 32 MB ≈ 28 GB
(negligible overhead for adapter caching)
```

### 7.2 Punica: Batching LoRA Operations

Punica (Chen et al., 2024) introduces specialized CUDA kernels for efficient multi-LoRA batching:

**Punica BGMV Kernel (Batched Grouped Matrix Vector):**
```
Standard batch matrix-vector product:
y = A @ x  (A: [m, n], x: [n], y: [m])

BGMV kernel:
- Process multiple (A, x, y) operations in single kernel
- Group operations by matrix size for better memory locality
- Overlap communication and computation

Punica kernel signature:
punica_bgmv(
    output,           # [total_tokens, hidden_dim]
    lora_a_stacked,   # [num_loras, hidden_dim, rank]
    lora_b_stacked,   # [num_loras, rank, hidden_dim]
    indices,          # [total_tokens] (which LoRA per token)
    scales,           # [num_loras] (alpha/rank per LoRA)
)
```

**Punica Implementation Sketch:**
```python
def punica_bgmv(
    output,
    lora_a_stacked,
    lora_b_stacked,
    indices,
    scales
):
    """
    Efficient batched LoRA computation

    Idea: Single kernel processes all tokens/LoRAs
    Benefits:
    - Reduced kernel launch overhead
    - Better GPU memory utilization
    - Grouped computation for cache locality
    """

    # Per-token LoRA selection
    for token_idx in range(total_tokens):
        lora_idx = indices[token_idx]
        lora_A = lora_a_stacked[lora_idx]
        lora_B = lora_b_stacked[lora_idx]

        # Token-specific computation
        hidden = lora_a_stacked[lora_idx] @ token_input
        output_token = lora_b_stacked[lora_idx] @ hidden
        output[token_idx] += output_token * scales[lora_idx]
```

**Performance Impact:**
- Throughput: 10-100× improvement vs per-request LoRA
- Latency: Negligible overhead for multi-LoRA batching
- Enables practical serving of 100s of concurrent LoRA adapters

## 8. LoRA on CPU

### 8.1 CPU-Based Fine-Tuning

Fine-tuning on CPU with LoRA for devices without GPU:

**Memory-Efficient CPU Training:**
```python
class CPULoRATrainer:
    def __init__(self, model, rank=8):
        self.model = model.cpu()
        self.rank = rank

        # Add LoRA
        self.add_lora_to_model()

        # Only LoRA parameters trainable
        self.set_lora_trainable_only()

    def add_lora_to_model(self):
        """Inject LoRA into CPU model"""
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Linear):
                # Freeze original weights
                module.weight.requires_grad = False
                module.bias.requires_grad = False

                # Add LoRA
                out_features, in_features = module.weight.shape
                module.lora_A = torch.nn.Parameter(
                    torch.zeros(self.rank, in_features)
                )
                module.lora_B = torch.nn.Parameter(
                    torch.randn(out_features, self.rank) / math.sqrt(self.rank)
                )

    def set_lora_trainable_only(self):
        """Only train LoRA parameters"""
        for name, param in self.model.named_parameters():
            if 'lora' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

    def get_trainable_params(self):
        """Count trainable parameters"""
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        return trainable

# For 7B model:
# Total params: 7B
# Trainable (LoRA only): 0.5M (0.007%)
# Training memory: Manageable on CPU with gradient checkpointing
```

### 8.2 Gradient Checkpointing on CPU

CPU training requires aggressive gradient checkpointing:

```python
def train_step_cpu(model, batch, optimizer):
    """
    CPU training with gradient checkpointing
    """

    inputs = batch['input_ids']
    labels = batch['labels']

    # Forward with checkpointing (recompute activations in backward)
    with torch.checkpoint(model, use_reentrant=False):
        loss = model(inputs, labels=labels).loss

    # Backward (recomputes activations as needed)
    loss.backward()

    # Small gradient accumulation (CPU memory limited)
    optimizer.step()
    optimizer.zero_grad()

# CPU training speed: ~1-5 tokens/second (vs 100+ on GPU)
# Trade-off: Feasibility on any hardware vs. slow training
```

### 8.3 Inference on Edge Devices

Merged LoRA inference on mobile/edge:

**Mobile LoRA Inference:**
```python
def setup_mobile_model(model_path, lora_path, device='cpu'):
    """
    Setup merged LoRA model for mobile inference
    """

    # Load quantized base model
    model = load_quantized_model(model_path)  # 4-bit quantization
    model.eval()

    # Load LoRA weights
    lora_checkpoint = torch.load(lora_path)

    # Merge LoRA into weights
    for name, param in model.named_parameters():
        if name in lora_checkpoint:
            lora_B = lora_checkpoint[name + '_B']
            lora_A = lora_checkpoint[name + '_A']
            alpha = lora_checkpoint[name + '_alpha']

            merged = merge_lora_to_weights(param, lora_B, lora_A, alpha, rank=64)
            param.copy_(merged)

    # Optimize for mobile
    model = torch.jit.script(model)
    model = model.to(device)

    return model

# Mobile model size:
# Quantized 7B model: ~2 GB
# Merged LoRA: +0 GB (in-place replacement)
# Total for mobile: ~2 GB (fits on device)
```

## 9. Empirical Results & Comparisons

### 9.1 Accuracy vs Parameter Efficiency

**Empirical Evaluation (7B LLM, Multiple Tasks):**

| Method | Trainable % | Accuracy | Memory | Speed |
|--------|-------------|----------|--------|-------|
| Full FT | 100% | 100% | 145GB | 1.0× |
| LoRA r=16 | 0.46% | 99.8% | 70GB | 0.95× |
| LoRA r=64 | 1.85% | 99.95% | 75GB | 0.92× |
| QLoRA (NF4) | 0.46% | 99.5% | 6GB | 0.87× |
| DoRA | 0.46% | 99.9% | 70GB | 0.93× |
| VeRA | 0.23% | 99.2% | 68GB | 0.94× |

**Key Findings:**
- LoRA rank=64 achieves near full-FT accuracy
- QLoRA enables training on consumer GPUs
- DoRA shows consistent improvements
- VeRA suitable for extreme parameter reduction

### 9.2 Convergence Speed

**Training dynamics (cross-domain fine-tuning):**

```
Epoch 1-5: Pre-training provides strong foundation
- LoRA: 2-3x fewer tokens needed to converge vs FT
- Reason: Base model already learned general features

Epoch 5-20: Task-specific adaptation
- LoRA: Matches full FT accuracy around epoch 15
- Full FT: May need epoch 20+ due to optimization landscape

Final accuracy:
- LoRA: 99.8% (of full FT baseline)
- DoRA: 99.95% (slight improvement)
- QLoRA: 99.5% (acceptable for many tasks)
```

## 10. Summary & Production Guidelines

### 10.1 LoRA Configuration Guidelines

**Rank Selection by Task Complexity:**

| Task Type | Recommended Rank | Params | Notes |
|-----------|------------------|--------|-------|
| Domain adaptation | 8-16 | 0.1-0.2M | Similar distribution |
| Multi-task learning | 16-32 | 0.2-0.5M | Diverse distributions |
| Cross-domain | 32-64 | 0.5-1M | Significant shift |
| Few-shot learning | 64-128 | 1-2M | Limited data |

**Alpha Selection:**
- Standard: α = r (e.g., rank=16, alpha=16)
- Conservative: α = r/2 (reduces adaptation strength)
- Aggressive: α = 2×r (stronger adaptation, higher risk)

### 10.2 When to Use Each PEFT Variant

| Method | Use Case | Pros | Cons |
|--------|----------|------|------|
| LoRA | General-purpose | Balanced, simple | Slightly lower accuracy |
| DoRA | Better convergence needed | +0.5-1% accuracy | +5% parameters |
| VeRA | Extreme efficiency | 50% fewer params | -0.5% accuracy |
| QLoRA | Memory-constrained | 95% memory savings | -0.2-0.5% accuracy |
| LoRA+ | Fast convergence | Faster training | Minimal impact |

### 10.3 Production Deployment Checklist

- [ ] Select rank based on task complexity (16-64 typical)
- [ ] Validate accuracy on dev set (target: >99% of full FT)
- [ ] Merge weights for inference (zero overhead)
- [ ] Implement S-LoRA for multi-adapter serving
- [ ] Monitor inference latency (should be identical to base model)
- [ ] Set up adapter version control for reproducibility
- [ ] Plan A/B testing strategy for task-specific models

### 10.4 Further Reading

- Hu et al. (ICLR 2022). "LoRA: Low-Rank Adaptation of Large Language Models"
- Dettmers et al. (NeurIPS 2023). "QLoRA: Efficient Finetuning of Quantized LLMs"
- Kyathanahally et al. (2024). "DoRA: Weight-Decomposed Low-Rank Adaptation"
- Sheng et al. (MLSys 2024). "S-LoRA: Serving Many Adapters for Fine-Tuning"
- Zhang et al. (2024). "LoRA+: Improved Low-Rank Adaptation for More Stable Training"

---

**Module Completion Status**: Comprehensive systems perspective on parameter-efficient fine-tuning with emphasis on LoRA's mathematical foundations, implementation details, production deployment patterns, and practical variants.
