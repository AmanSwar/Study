# MODULE 17 — CoreML & ANE Programming

## 1. SYSTEMS OVERVIEW

CoreML is Apple's machine learning inference framework spanning from high-level model conversion tools (coremltools) through intermediate representation (MIL—Model Intermediate Language) to backend dispatch on ANE, GPU, or CPU. Unlike PyTorch or TensorFlow which handle both training and inference, CoreML is inference-only, allowing aggressive optimization at each compilation stage.

The compilation pipeline transforms a PyTorch model into a machine code binary optimized for Apple Silicon's hardware accelerators. This process involves type inference, operator mapping to hardware-specific implementations, memory allocation planning, and quantization. Success requires understanding both the abstract computation graph and concrete hardware constraints (e.g., ANE's 512-token sequence limit, no dynamic shapes).

CoreML's strength is transparent hardware abstraction: developers write high-level code, and the framework automatically dispatches to ANE, GPU, or CPU based on operator support and performance characteristics. This abstraction hides significant complexity but requires developers to understand when dispatch decisions break down—e.g., when a model unexpectedly falls through to CPU due to unsupported operators, resulting in 10× slowdown.

### 1.1 CoreML Stack Layering

```
┌─────────────────────────────────────────────┐
│  User Code (Swift/Objective-C)              │
│  - Load MLModel from disk                   │
│  - Create predictor instance                │
│  - Invoke prediction with inputs            │
├─────────────────────────────────────────────┤
│  CoreML Runtime API                         │
│  - MLModel, MLSession, MLFeatureProvider    │
│  - Async prediction, batch prediction       │
├─────────────────────────────────────────────┤
│  Backend Dispatch Layer                     │
│  - Neural Network Backend (NeuralNet)       │
│  - ML Program Backend (newer)                │
│  - Preference ranking: ANE > GPU > CPU      │
├─────────────────────────────────────────────┤
│  Hardware Accelerators                      │
│  - ANE (Neural Engine, 16-core systolic)    │
│  - GPU (Metal compute pipeline)             │
│  - CPU (NEON SIMD, multithreaded)           │
└─────────────────────────────────────────────┘
```

### 1.2 Conversion Pipeline: PyTorch → CoreML

```
PyTorch Model (FP32)
  ↓ [coremltools.convert()]
MIL IR (Intermediate Language, graph representation)
  ↓ [Apply passes: constant folding, dead code elimination, op fusion]
Optimized MIL Graph
  ↓ [Backend selection: ANE/GPU/CPU per op]
Neural Network / ML Program format
  ↓ [Quantization: FP16, INT8, W4A16, etc.]
Quantized Model
  ↓ [JIT compilation: MIL → Metal/NEON bytecode]
MLModel Binary (.mlmodel)
  ↓ [Loaded at runtime]
Neural Engine / GPU Execution
```

### 1.3 Two Backend Architectures

**Neural Network Backend (NeuralNet):**
- Older architecture, deprecated in favor of ML Program
- Fixed operators, graph-level optimization only
- Cannot express dynamic shapes or control flow
- Pros: Stable, well-tested on production models
- Cons: Limited operator set, no ANE for newer models

**ML Program Backend:**
- New architecture (iOS 17+, Sonoma+)
- Function-based model, supports control flow (if/while)
- Better operator coverage, more flexible quantization
- Enables eager execution for debugging
- Preferred for new models

---

## 2. THEORETICAL FOUNDATION

### 2.1 MIL Intermediate Representation

MIL represents computation as a **directed acyclic graph (DAG)** of typed operations, with explicit data flow and control flow edges.

**MIL IR Structure:**
```
Program:
  - Name: "my_model"
  - Inputs: {
      input_0: (BATCH, 512) float32,
      input_1: (BATCH,) int64
    }
  - Outputs: {
      output: (BATCH, 1024) float16
    }

  Function main:
    block0(x: (BATCH, 512), idx: (BATCH,)):
      # Operator: embedding lookup
      embed = mb.embedding(x=x, weight=weight_var)  # (BATCH, 512, 768)

      # Operator: matrix multiply
      matmul_out = mb.matmul(x=embed, y=proj_weight)  # (BATCH, 512, 1024)

      # Operator: add bias
      output = mb.add(x=matmul_out, y=bias_var)  # (BATCH, 512, 1024)

      # Operator: activation (ReLU)
      relu_out = mb.relu(x=output)  # (BATCH, 512, 1024)

      return relu_out
```

**Type System:**
MIL enforces strict typing:
```
Scalar types: int32, int64, fp32 (float32), fp16, bool
Tensor types: Tensor[shape, dtype]
  - shape: (D1, D2, ..., Dn) where Di ∈ {literal, variable, "symbolic"}
  - dtype: fp32, fp16, int8, int32, etc.
Optional types: Optional[T]
```

**Shape Inference:**
```
% Input: x: (BATCH, 512, 768)
%        weight: (768, 1024)
% Operation: matmul(x, weight)
% Output shape: (BATCH, 512, 1024)
%   (BATCH, 512, 768) × (768, 1024) → (BATCH, 512, 1024)
```

### 2.2 Operator Support Matrix: ANE vs GPU vs CPU

Not all operators are supported on all backends. The dispatch decision is critical for performance.

**Common Operators and Support:**

| Operator | ANE | GPU | CPU | Notes |
|---|---|---|---|---|
| **linear** (fully connected) | ✓ | ✓ | ✓ | ANE: FP16/INT8 only, no FP32 |
| **conv2d** | ✓ | ✓ | ✓ | ANE: 1×1 conv, no strided |
| **batch_norm** | ✓ | ✓ | ✓ | ANE: inference mode, fused to linear |
| **add**, **mul** | ✓ | ✓ | ✓ | Broadcast: ANE limited to <= 4D |
| **relu**, **sigmoid** | ✓ | ✓ | ✓ | Element-wise, native on all |
| **softmax** | ✗ | ✓ | ✓ | Not ANE native; dispatch to GPU |
| **layer_norm** | ✗ | ✓ | ✓ | Requires custom kernel or CPU |
| **attention** (full) | ✗ | ✓ | ✓ | ANE cannot express QK^T matmul |
| **embedding** | ✓ | ✓ | ✓ | ANE: lookup table, batch size matters |
| **gelu**, **swiglu** | ✗ | ✓ | ✓ | Custom activation, need GPU kernel |
| **scatter**, **gather** | ✗ | ✓ | ✓ | Dynamic indexing, not ANE |

### 2.3 Quantization Mechanics

**Asymmetric Quantization (Most Common):**

For tensor x ∈ ℝ^{n×m}, asymmetric quantization maps to q-bit integer range [0, 2^q-1]:

```
x_int8 = round((x - zero_point) / scale)
  where scale = (x_max - x_min) / (2^q - 1)
        zero_point = round(-x_min / scale)
```

**CoreML Quantization Schemes:**

1. **W8A8 (Weights INT8, Activations INT8):**
   - Calibration: Collect min/max statistics on representative data
   - Inference: `out = (W_int8 @ A_int8) * (scale_w * scale_a)`
   - Throughput: Same as FP32 (same memory size)
   - Accuracy: 1-2% drop typical, 3-5% for aggressive quantization

2. **W4A16 (Weights INT4, Activations FP16):**
   - Weights: 2 bits per value (4-bit = 2×2-bit packed)
   - Activations: Full FP16 (no quantization loss)
   - Memory: 50% weight reduction vs FP16
   - Inference: `out = (W_int4_dequant @ A_fp16) * scale_w`
   - Throughput: Limited by weight dequantization (requires unpacking)

3. **Palettization (Vector Quantization):**
   - Cluster weights into k centroids (e.g., k=16 for 4-bit)
   - Store indices (4-bit) instead of FP16 values
   - Lookup table: palette[index] → FP16 value
   - Accuracy: 0.5-1% drop, extreme compression (16:1 for k=16)

**ANE Quantization Constraints:**
- ANE natively supports FP16, INT8
- W4A16 requires software dequantization loop (GPU or CPU), breaking ANE efficiency
- Recommended: FP16 (minimal accuracy loss) or INT8 (if accuracy permits)

### 2.4 Backend Dispatch Logic

Compiler implements heuristic dispatch:

```
for each operator Op in graph:
  supported_backends = [ANE, GPU, CPU]

  # Remove unsupported backends
  if Op not in ANE_whitelist:
    supported_backends -= ANE
  if Op not in GPU_supported_ops:
    supported_backends -= GPU

  # Cost-based selection
  cost_ANE = cost_memory(Op on ANE) + cost_latency(Op on ANE)
  cost_GPU = cost_memory(Op on GPU) + cost_latency(Op on GPU)
  cost_CPU = cost_memory(Op on CPU) + cost_latency(Op on CPU)

  selected_backend = argmin(cost_ANE, cost_GPU, cost_CPU)

  # If selected_backend = ANE and constraints violated:
  if not (batch_size in {1,8,16,32} and
          sequence_len <= 512 and
          quantization in {FP16, INT8}):
    selected_backend = GPU
```

---

## 3. HARDWARE MAPPING

### 3.1 Memory Layout for ANE

ANE requires specific weight layouts (not standard row-major or column-major). CoreML compiler applies layout transformation automatically.

**Standard PyTorch Linear Layer Weight Layout:**
```
weight shape: (out_features, in_features) = (1024, 768)
Layout: Row-major (C-contiguous)
Memory:
  weight[0] = [w00, w01, ..., w0,767, padding...]  ← 1024 bytes per row
  weight[1] = [w10, w11, ..., w1,767, padding...]
  ...
```

**ANE-Optimized Layout (Systolic Tiling):**
```
Weight shape after tiling: (out_features // 16, in_features // 16, 16, 16)
                          = (64, 48, 16, 16) for 1024×768

Layout: Systolic interleaving (16×16 tiles)
Tile[i,j]: 16×16 block starting at position (i*16, j*16)
Memory organization: Tiles packed linearly
  Tile[0,0] (16×16) | Tile[0,1] (16×16) | ... | Tile[1,0] | ...

Advantage: Systolic array processes one tile per clock cycle
           Sequential memory accesses within tile, minimal cache misses
```

**CoreML Transformation (Automatic):**
```
PyTorch model:
  nn.Linear(768, 1024)

Converted to CoreML (ANE backend):
  % Apply weight layout transformation
  weight_layout_transform = reshape + transpose + tile_reorder
  weight_ane = weight_layout_transform(weight_pytorch)

Compilation:
  % Fuse with layout transform, embed in MLModel binary
  % No runtime overhead (layout transform baked in)
```

### 3.2 Batch Size Constraints

ANE systolic array is optimized for specific batch sizes (1, 8, 16, 32). Other sizes require padding or reshaping.

**Batch Size 1 (Single Sample):**
- Systolic dataflow: A[0] × B[0] → C[0]
- Throughput: ~1 GEMM per clock cycle (pipelined)
- ANE utilization: 90%+
- Latency: ~100 μs (for 512×512 GEMM)

**Batch Size 8:**
- Systolic dataflow: A[0..7] × B[0..7] → C[0..7] (parallel in rows)
- Throughput: ~8 GEMMs per cycle
- ANE utilization: 95%+
- Latency: ~110 μs (overlapped with A fetches)

**Batch Size 7 (Non-optimal):**
- Systolic array configured for 8-wide, 1 row idle
- Padding required: [A[0..7] with A[7] duplicated] × B → C (then extract C[0..7])
- ANE utilization: 90% (one row wasted)
- Latency: Same ~110 μs, throughput reduced by ~12%

**Batch Size 32:**
- Requires two systolic passes (batch size must be <= 32 for current ANE)
- First pass: Batch 1-16, latency ~200 μs
- Second pass: Batch 17-32, latency ~200 μs
- Total: ~400 μs (not parallelizable, sequential)
- Throughput: 32 ÷ 400 μs = 80 Msamples/s (lower throughput/sample due to overhead)

### 3.3 Sequence Length Constraints

**ANE Sequence Length Limit (Hard Constraint):**

ANE uses fixed-size SRAM (1024 KB total) for buffering. For sequence-parallel operations (e.g., layer norm over sequence dimension):

```
Sequence length L, hidden dimension D, batch size B
SRAM budget: 1024 KB = 2^20 bytes

Layer Norm requires: L × D × 2 bytes (FP16)
  L × 1024 × 2 ≤ 1024 KB
  L ≤ 512

Attention:
  Query: L × D FP16
  Key:   L × D FP16
  Value: L × D FP16
  Total: 3 × L × D × 2 bytes
  L ≤ 170 (practical limit due to other allocations)
```

**CoreML Compiler Decision:**
```
if sequence_length > 512:
  dispatch_to = GPU (ANE cannot buffer)
elif sequence_length in [1, 256, 512]:
  dispatch_to = ANE (normal case)
else:
  # Intermediate (e.g., 300):
  # ANE can handle, but non-optimal tiling
  dispatch_to = ANE  # with padding to 512 if beneficial
```

---

## 4. IMPLEMENTATION DEEP DIVE

### 4.1 PyTorch to CoreML Conversion Pipeline

**Complete Example: LLaMA-like Transformer Conversion**

```python
import torch
import coremltools as ct
from coremltools.models import neural_network as nn
from coremltools.proto import FeatureTypes_pb2

# Step 1: Load PyTorch model
class SimpleLLaMA(torch.nn.Module):
    def __init__(self, vocab_size=50257, hidden_dim=768, num_layers=2):
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, hidden_dim)
        self.layers = torch.nn.ModuleList([
            torch.nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=8,
                dim_feedforward=3072,
                batch_first=True,
                activation='gelu'
            )
            for _ in range(num_layers)
        ])
        self.head = torch.nn.Linear(hidden_dim, vocab_size)

    def forward(self, input_ids, attention_mask=None):
        x = self.embedding(input_ids)  # (batch, seq_len, hidden)
        for layer in self.layers:
            x = layer(x, src_key_padding_mask=~attention_mask.bool())
        logits = self.head(x)  # (batch, seq_len, vocab)
        return logits

model = SimpleLLaMA()
model.load_state_dict(torch.load('llama_weights.pt'))
model.eval()

# Step 2: Trace model (convert to JIT)
# Note: coremltools requires symbolic shapes for variable dimensions
batch_size = 1
seq_length = 512
example_input = torch.randint(0, 50257, (batch_size, seq_length))
example_mask = torch.ones((batch_size, seq_length), dtype=torch.bool)

# Trace with example inputs
traced_model = torch.jit.trace(model, (example_input, example_mask))

# Step 3: Convert to MIL representation
# Note: Flexible inputs require specifying RangeDim
inputs = [
    ct.TensorType(shape=(ct.RangeDim(1, 4), ct.RangeDim(1, 512))),  # input_ids: batch 1-4, seq 1-512
    ct.TensorType(shape=(ct.RangeDim(1, 4), ct.RangeDim(1, 512)))   # mask: same shape
]

converted_model = ct.convert(
    traced_model,
    inputs=inputs,
    outputs=[ct.TensorType(name="logits")],
    target_opset=ct.target.iOS17,  # Use ML Program (iOS 17+) for better operator support
    compute_units=ct.ComputeUnit.ALL,  # Allow ANE, GPU, CPU dispatch
    minimum_deployment_target=ct.target.iOS17,
    debug=True,  # Verbose logging for dispatch decisions
)

# Step 4: Inspect MIL representation before saving
# Optional: print MIL IR to debug operator support
print(converted_model._mil_program)

# Step 5: Apply optimizations (coremltools v6+)
# Remove redundant layers, fuse operations
optimized_model = ct.optimize.remove_unused(converted_model)

# Step 6: Set metadata
converted_model.short_description = "LLaMA-2 7B Inference"
converted_model.author = "ML Researcher"

# Step 7: Save as MLModel
converted_model.save("llama_2.mlmodel")

print("Conversion complete!")
print(f"Model size: {converted_model.weights_dir.__sizeof__() / 1e6:.2f} MB")
```

### 4.2 Inspection and Debugging of MIL Graph

**Extracting MIL IR for Analysis:**

```python
import coremltools as ct

# Load converted model
model = ct.models.MLModel("llama_2.mlmodel")

# Access MIL program
mil_program = model._mil_program

# Inspect main function
main_func = mil_program.functions['main']

# Print all operations
for block in main_func.blocks:
    for op in block.operations:
        print(f"Op: {op.op_type}")
        print(f"  Inputs: {[inp.name for inp in op.inputs.values()]}")
        print(f"  Output shape: {op.outputs[0].shape}")
        print(f"  Placement: {op.placement}")  # ANE, GPU, CPU
```

**Analyzing Backend Dispatch:**

```python
def analyze_dispatch(model):
    """Print which backend executes each operation."""
    mil_program = model._mil_program
    main_func = mil_program.functions['main']

    dispatch_stats = {'ANE': 0, 'GPU': 0, 'CPU': 0, 'UNKNOWN': 0}

    for block in main_func.blocks:
        for op in block.operations:
            placement = getattr(op, 'placement', 'UNKNOWN')
            if placement in dispatch_stats:
                dispatch_stats[placement] += 1
            else:
                dispatch_stats['UNKNOWN'] += 1

            # Warn if operation forced to CPU
            if placement == 'CPU' and op.op_type in ['mul', 'add', 'relu']:
                print(f"WARNING: {op.op_type} on CPU (expect GPU)")

    print("Dispatch Summary:")
    for backend, count in dispatch_stats.items():
        print(f"  {backend}: {count} ops")

    return dispatch_stats

analyze_dispatch(model)
```

### 4.3 Quantization Pipeline: Post-Training Quantization (PTQ)

```python
import coremltools as ct
import numpy as np

# Load baseline model (FP32)
baseline_model = ct.models.MLModel("llama_baseline.mlmodel")

# Step 1: Collect representative data for calibration
def get_calibration_data():
    """Generate diverse calibration inputs to represent real workloads."""
    calibration_data = []
    for i in range(100):  # 100 samples
        input_ids = np.random.randint(0, 50257, (1, 512), dtype=np.int32)
        mask = np.ones((1, 512), dtype=np.float32)
        calibration_data.append({
            'input_ids': input_ids,
            'attention_mask': mask
        })
    return calibration_data

calibration_data = get_calibration_data()

# Step 2: Apply INT8 quantization
from coremltools.quantization_utils import quantize_weights

# Create quantization config
quantization_config = ct.QuantizationConfig(
    mode="int8",  # INT8 weights and activations
    weight_threshold=100,  # Quantize weights with threshold
)

# Quantize
quantized_model = ct.quantize_weights(
    baseline_model,
    mode='int8',
    nbits=8,
)

# Step 3: (Optional) Fine-grained control with per-layer quantization
from coremltools.models import neural_network

quantized_per_layer = ct.quantize_weights(
    baseline_model,
    mode='int8_linear_quantization',
    nbits=8,
)

# Step 4: Validate quantized model accuracy
# (On CPU/GPU, measure accuracy drop)
def validate_quantization(baseline_model, quantized_model, test_data):
    """Compare outputs of baseline vs quantized."""
    baseline_predictor = ct.models.MLModel(baseline_model)
    quantized_predictor = ct.models.MLModel(quantized_model)

    max_error = 0.0
    for sample in test_data[:10]:  # Spot check first 10 samples
        baseline_out = baseline_predictor.predict(sample)
        quantized_out = quantized_predictor.predict(sample)

        error = np.abs(baseline_out - quantized_out).max()
        max_error = max(max_error, error)

    print(f"Max quantization error: {max_error:.6f}")
    return max_error < 0.01  # Allow 0.01 max error

quantized_model.save("llama_quantized_int8.mlmodel")
```

### 4.4 Profiling and Dispatch Analysis at Runtime

**Swift Code for Runtime Inspection:**

```swift
import CoreML

func profileCoreMLInference() {
    // Load model
    guard let modelURL = Bundle.main.url(forResource: "llama_2", withExtension: "mlmodel") else {
        print("Model not found")
        return
    }

    let model = try! MLModel(contentsOf: modelURL)

    // Create session with profiling enabled
    let config = MLModelConfiguration()
    config.computeUnits = .all  // ANE + GPU + CPU

    let session = try! MLSession(model: model, configuration: config)

    // Prepare input
    let inputTokens = MLMultiArray(shape: [1, 512], dataType: .int32)
    let inputMask = MLMultiArray(shape: [1, 512], dataType: .float32)

    // Create input provider
    let input = [
        "input_ids": inputTokens,
        "attention_mask": inputMask
    ]

    // Profile prediction
    let startTime = Date()

    let output = try! session.run(inputs: input)

    let elapsed = Date().timeIntervalSince(startTime)
    print("Inference latency: \(elapsed * 1000) ms")

    // Note: CoreML does not expose backend dispatch info at runtime
    // Dispatch determined statically during compilation
}
```

### 4.5 ANE Constraints Workaround: Sequence Length > 512

**Problem:** Model with 4096 token context; ANE cannot buffer in SRAM.

**Solution: Chunked Inference with GPU Fallback**

```python
import torch
import coremltools as ct

class ChunkedLLaMA(torch.nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        self.chunk_size = 512  # ANE max sequence length

    def forward(self, input_ids):
        # input_ids shape: (batch, seq_len)
        seq_len = input_ids.shape[1]

        if seq_len <= self.chunk_size:
            # Within ANE range, use ANE backend
            return self.base_model(input_ids)
        else:
            # Chunk and process (note: attention over chunk boundaries is lost)
            outputs = []
            for i in range(0, seq_len, self.chunk_size):
                chunk = input_ids[:, i:i+self.chunk_size]
                outputs.append(self.base_model(chunk))
            # Concatenate outputs
            return torch.cat(outputs, dim=1)

# Alternatively: Use GPU only for long sequences
class GPUFallbackLLaMA(torch.nn.Module):
    def __init__(self, base_model, ane_threshold=512):
        super().__init__()
        self.base_model = base_model
        self.ane_threshold = ane_threshold

    def forward(self, input_ids):
        seq_len = input_ids.shape[1]
        if seq_len > self.ane_threshold:
            # Force GPU by adding dynamic shape dimension
            input_ids_padded = torch.nn.functional.pad(
                input_ids,
                (0, (seq_len - 1) % 512 + 1)  # Pad to next power of 512
            )
            # GPU processes padded sequence (ANE dispatch fails due to size)
            return self.base_model(input_ids_padded)[:, :seq_len]
        else:
            return self.base_model(input_ids)
```

---

## 5. KEY PAPERS

1. **"coremltools: Towards Efficient Machine Learning Inference on Apple Silicon" (Apple ML Research, 2023)**
   - Architecture of MIL IR and backend dispatch
   - Operator support and constraints
   - Quantization algorithms and ANE mapping

2. **"Unified Memory Model for Heterogeneous Execution on Apple Platforms" (OSDI 2021 adjacent work)**
   - Memory semantics for GPU-ANE coordination
   - Zero-copy weight loading implications for model switching

3. **"Neural Network Compiler for Apple Silicon" (LLVM-based optimizations)**
   - Operator fusion strategies
   - Quantization-aware compilation
   - Reference: LLVM documentation on Apple targets

4. **"ANE Quantization and Constraints" (Internal Apple documentation, reverse-engineered by community)**
   - Batch size requirements
   - Sequence length limits
   - Data layout specifications

5. **"Efficient LLM Inference on Mobile Devices" (MLSys 2024 adjacent work)**
   - Quantization strategies for mobile hardware
   - Dispatch heuristics for heterogeneous backends

---

## 6. SYSTEMS TRADEOFFS

### 6.1 Conversion Stability vs Optimization Aggressiveness

| Tradeoff | Conservative | Aggressive |
|---|---|---|
| **Operator Support** | Older ops only (NeuralNet backend) | Latest operators (ML Program) |
| **Dispatch Heuristics** | All ops on GPU (safe, slow) | ANE when possible (faster, risk dispatch errors) |
| **Quantization** | None (FP32, large models) | W4A16 (fast, accuracy risk) |
| **Accuracy Drop** | <0.1% | 1-3% |
| **Model Size** | 100% baseline | 25-50% baseline |
| **Latency** | 2-5× slower | 3-10× faster |

**When Conservative:** Deployment in regulated industry (healthcare, finance) where accuracy is critical.
**When Aggressive:** Mobile/edge deployment where latency dominates, some accuracy loss acceptable.

### 6.2 ANE vs GPU for Batch Inference

**ANE Strengths:**
- Systolic array perfectly matches GEMM patterns
- 11 TOPS peak, but low energy (~2W sustained)
- Deterministic throughput (FIFO queue)

**GPU Strengths:**
- Flexible operator support (softmax, attention, custom kernels)
- Better utilization for non-dense ops
- Handles dynamic shapes

**Decision Heuristic:**
```
if model is purely dense GEMM-based (like BERT):
  use ANE (5-10× energy improvement)
elif model has attention or softmax:
  use GPU (10-50% performance advantage)
elif batch_size >= 32 and model is vision-based:
  use ANE (parallelism over spatial dims)
else:
  use GPU (flexibility outweighs throughput gap)
```

### 6.3 Quantization Precision vs Accuracy Tradeoff

| Quantization | Bit Width | Model Size | Accuracy Drop | ANE Support | Throughput |
|---|---|---|---|---|---|
| FP16 | 16 | 100% | <0.2% | ✓ | 1.0× |
| INT8 Symmetric | 8 | 50% | 0.5-1.5% | ✓ | 0.95× |
| INT8 Asymmetric | 8 | 50% | 0.3-1.0% | ✓ | 0.95× |
| W4A16 | 4 | 25% | 0.5-2% | ✗ (GPU only) | 0.7× |
| Palettization | 4-6 | 6-12% | 0.3-2% | ✗ | 0.8× |

**Tradeoff:** INT8 is sweet spot (50% size reduction, <1% accuracy drop, ANE native).

### 6.4 Compilation Time vs Runtime Performance

**Eager Compilation (First Load):**
- MIL optimization: 2-5 sec per 100MB model
- Quantization: 30-60 sec per GB (if post-training)
- JIT to Metal: 10-30 sec per 100MB model
- **Total:** 1-2 minutes for 7B model (happens once, then cached)

**Runtime (After Loading):**
- Memory load: 100-500 ms (from disk to unified memory)
- First prediction: 50-100 ms (cache warmup, kernel launch overhead)
- Subsequent predictions: Minimal overhead (10-20 ms)

**Optimization:** Pre-compile models during app release, ship as MLModel binary. Avoids runtime compilation overhead.

---

## 7. EXPERT INSIGHT

### 7.1 Debugging Unexpected CPU Fallback

**Symptom:** Model runs 10× slower than expected (GPU bound instead of ANE).

**Root Cause Analysis:**

```python
def find_cpu_ops(model):
    """Find operations dispatched to CPU (usually indicates dispatch failure)."""
    mil_program = model._mil_program
    main_func = mil_program.functions['main']

    cpu_ops = []
    for block in main_func.blocks:
        for op in block.operations:
            if hasattr(op, 'placement') and op.placement == 'CPU':
                cpu_ops.append({
                    'name': op.op_type,
                    'output_shape': op.outputs[0].shape,
                    'reason': infer_dispatch_reason(op)
                })

    return cpu_ops

def infer_dispatch_reason(op):
    """Heuristically determine why op was dispatched to CPU."""
    if op.op_type == 'softmax':
        return "softmax not ANE-native"
    elif op.op_type == 'gelu':
        return "custom activation requires GPU/CPU"
    elif op.op_type == 'mul' and op.outputs[0].dtype == 'fp32':
        return "FP32 not supported by ANE, requires GPU"
    elif 'dynamic' in str(op.outputs[0].shape):
        return "dynamic shape not ANE-compatible"
    else:
        return "operator not in ANE whitelist"

# Usage:
model = ct.models.MLModel("llama_2.mlmodel")
cpu_ops = find_cpu_ops(model)
for op in cpu_ops:
    print(f"CPU dispatch: {op['name']}, reason: {op['reason']}")
```

### 7.2 Batch Size Optimization for ANE

**Scenario:** Running inference on 100 independent samples (e.g., batch translation).

**Naive Approach:** Batch all 100 together.
```
Batch size 100 → ANE pads to 128 → Run 128 samples (28 wasted)
Latency: 100 / ANE_throughput × overhead
```

**Optimized Approach:** Multiple passes with batch size 32.
```
Pass 1: Batch size 32 → ANE runs 32 samples
Pass 2: Batch size 32 → ANE runs 32 samples
Pass 3: Batch size 32 → ANE runs 32 samples
Pass 4: Batch size 4 → ANE pads to 8 (4 wasted)
Total: 100 + 4 wasted = 104 (vs 28 wasted above)
Latency: Same due to pipelining
```

**Batching Policy:**
```python
def optimal_batch_size(num_samples, ane_supported_batch_sizes=[1, 8, 16, 32]):
    """Partition samples into batches minimizing wasted compute."""
    best_waste = float('inf')
    best_partition = []

    # Greedy: use largest batch size first
    remaining = num_samples
    partition = []
    for batch_size in sorted(ane_supported_batch_sizes, reverse=True):
        while remaining >= batch_size:
            partition.append(batch_size)
            remaining -= batch_size

    if remaining > 0:
        partition.append(remaining)  # Final partial batch

    total_compute = sum(
        max(b, min(ane_supported_batch_sizes))  # ANE pads to nearest supported size
        for b in partition
    )
    waste = total_compute - num_samples

    return partition, waste
```

### 7.3 Quantization Sensitivity Analysis

Not all layers have equal quantization impact.

```python
def sensitivity_analysis(model, test_data, layer_names):
    """Quantize each layer individually, measure accuracy drop."""
    baseline_model = ct.models.MLModel(model)
    baseline_output = baseline_model.predict(test_data)

    sensitivity = {}
    for layer_name in layer_names:
        quantized_model = ct.quantize_weights(
            model,
            nbits=8,
            # Quantize only this layer (hypothetical API)
            layer_names=[layer_name]
        )
        quantized_output = quantized_model.predict(test_data)

        error = np.mean(np.abs(baseline_output - quantized_output))
        sensitivity[layer_name] = error

    # Identify sensitive layers
    sensitive_layers = sorted(
        sensitivity.items(),
        key=lambda x: x[1],
        reverse=True
    )

    print("Layer quantization sensitivity:")
    for layer, error in sensitive_layers[:10]:
        print(f"  {layer}: {error:.6f}")

    return sensitivity
```

---

## 8. BENCHMARKING METHODOLOGY

### 8.1 CoreML Load-Time vs Inference-Time Profiling

**Setup: Measure Complete Pipeline**

```swift
import CoreML
import os.log

let logger = os.Logger(subsystem: "CoreML-Profiling", category: "Inference")

func profileCoreMLComplete() {
    // Measure 1: Model loading from disk
    let startLoad = Date()
    let modelURL = Bundle.main.url(forResource: "llama_2", withExtension: "mlmodel")!
    let model = try! MLModel(contentsOf: modelURL)
    let loadTime = Date().timeIntervalSince(startLoad)
    logger.info("Model load time: \(loadTime * 1000)ms")

    // Measure 2: Compilation (if not cached)
    let startCompile = Date()
    let config = MLModelConfiguration()
    config.computeUnits = .all
    let session = try! MLSession(model: model, configuration: config)
    let compileTime = Date().timeIntervalSince(startCompile)
    logger.info("Compilation time: \(compileTime * 1000)ms")

    // Measure 3: First prediction (cache warmup)
    let inputTokens = MLMultiArray(shape: [1, 512], dataType: .int32)
    let inputMask = MLMultiArray(shape: [1, 512], dataType: .float32)
    let input = ["input_ids": inputTokens, "attention_mask": inputMask]

    let startWarmup = Date()
    let _ = try! session.run(inputs: input)
    let warmupTime = Date().timeIntervalSince(startWarmup)
    logger.info("Warmup prediction time: \(warmupTime * 1000)ms")

    // Measure 4: Steady-state latency (average of 10 runs)
    let startSteady = Date()
    for _ in 0..<10 {
        let _ = try! session.run(inputs: input)
    }
    let steadyTime = (Date().timeIntervalSince(startSteady)) / 10.0
    logger.info("Steady-state latency: \(steadyTime * 1000)ms")

    // Summary
    print("""
    CoreML Timing:
      Model load:    \(loadTime * 1000)ms
      Compilation:   \(compileTime * 1000)ms
      Warmup:        \(warmupTime * 1000)ms
      Steady-state:  \(steadyTime * 1000)ms
      Total (init):  \(loadTime + compileTime + warmupTime)ms
    """)
}
```

### 8.2 Comparing Backends: ANE vs GPU vs CPU

**Methodology: Run Same Model on Each Backend**

```python
import coremltools as ct
import time
import numpy as np

def benchmark_backends(model_path, num_runs=100):
    """Benchmark model on ANE, GPU, CPU backends separately."""

    backends = [
        (ct.ComputeUnit.ANE_ONLY, "ANE"),
        (ct.ComputeUnit.GPU_ONLY, "GPU"),
        (ct.ComputeUnit.CPU_ONLY, "CPU"),
        (ct.ComputeUnit.ALL, "Mixed (ANE+GPU+CPU)")
    ]

    results = {}

    for compute_unit, name in backends:
        config = MLModelConfiguration()
        config.computeUnits = compute_unit

        try:
            model = ct.models.MLModel(model_path, configuration=config)
        except:
            print(f"Skipping {name} (not supported for this model)")
            continue

        # Prepare input
        input_data = np.random.randint(0, 50257, (1, 512), dtype=np.int32)

        # Warmup
        for _ in range(5):
            _ = model.predict({"input_ids": input_data})

        # Benchmark
        times = []
        for _ in range(num_runs):
            start = time.time()
            _ = model.predict({"input_ids": input_data})
            times.append(time.time() - start)

        # Statistics
        times = np.array(times)
        results[name] = {
            'mean': np.mean(times),
            'median': np.median(times),
            'p95': np.percentile(times, 95),
            'p99': np.percentile(times, 99),
            'stdev': np.std(times)
        }

    # Print comparison
    print("\nBackend Comparison:")
    print(f"{'Backend':<25} {'Mean (ms)':<12} {'P95 (ms)':<12} {'Stdev':<8}")
    for backend, stats in results.items():
        print(f"{backend:<25} {stats['mean']*1000:<12.2f} {stats['p95']*1000:<12.2f} {stats['stdev']*1000:<8.2f}")

    return results

# Usage:
benchmark_results = benchmark_backends("llama_2.mlmodel")
```

### 8.3 Quantization Accuracy Validation

```python
def validate_quantization_accuracy(baseline_model_path, quantized_model_path, test_dataset, task='classification'):
    """Compare accuracy between baseline and quantized models."""

    import torch.nn.functional as F

    baseline_model = ct.models.MLModel(baseline_model_path)
    quantized_model = ct.models.MLModel(quantized_model_path)

    baseline_correct = 0
    quantized_correct = 0
    max_logit_diff = 0.0

    for sample, label in test_dataset:
        baseline_output = baseline_model.predict(sample)
        quantized_output = quantized_model.predict(sample)

        # Extract logits (assuming output is 'logits')
        baseline_logits = baseline_output['logits']
        quantized_logits = quantized_output['logits']

        # Compare predictions
        baseline_pred = np.argmax(baseline_logits)
        quantized_pred = np.argmax(quantized_logits)

        baseline_correct += (baseline_pred == label)
        quantized_correct += (quantized_pred == label)

        # Compare logit distributions
        logit_diff = np.max(np.abs(baseline_logits - quantized_logits))
        max_logit_diff = max(max_logit_diff, logit_diff)

    baseline_acc = baseline_correct / len(test_dataset)
    quantized_acc = quantized_correct / len(test_dataset)
    accuracy_drop = baseline_acc - quantized_acc

    print(f"""
    Quantization Validation:
      Baseline accuracy:  {baseline_acc:.4f}
      Quantized accuracy: {quantized_acc:.4f}
      Accuracy drop:      {accuracy_drop:.4f} ({accuracy_drop*100:.2f}%)
      Max logit diff:     {max_logit_diff:.6f}
    """)

    return {
        'baseline_acc': baseline_acc,
        'quantized_acc': quantized_acc,
        'accuracy_drop': accuracy_drop,
        'max_logit_diff': max_logit_diff
    }
```

---

## 9. OPEN PROBLEMS

1. **Dynamic Shape Support in ANE:** Current ANE dispatch requires static shapes known at compile time. Supporting dynamic shapes (e.g., variable sequence length) forces GPU fallback. Challenge: modify systolic array to handle variable-length inputs without padding overhead.

2. **Mixed-Precision Quantization:** CoreML forces uniform precision per layer (all FP16 or all INT8). Optimal models might use FP16 for attention, INT8 for FFN. Support matrix: requires per-operator precision specification in MIL.

3. **Custom Operator Registration:** Limited to Apple's operator whitelist. Custom operators (e.g., specialized attention kernels) cannot be registered without shipping custom Metal shaders. Framework limitation: no user-defined operator API (unlike TorchScript).

4. **Caching and Compilation Distribution:** Each device compiles models independently, wasting computation. Ideal: compile once, distribute as sealed binary. Challenge: device-specific optimization (frequency, cache sizes) makes universal binary suboptimal.

5. **ANE Operator Expansion:** ANE remains unchanged since M1 (2020). Community requests: softmax, layer norm, attention primitives. Apple's roadmap unclear (rumored to expand in future chips, but unconfirmed).

6. **Quantization-Aware Training Integration:** CoreML quantization is post-training. QAT (fine-tuning with quantization-aware loss) is standard in TensorFlow/PyTorch ecosystems but underutilized for CoreML. Integration: could significantly improve INT8 accuracy.

---

## 10. PHD QUALIFIER QUESTIONS

**Q1 (Core Concepts):** "Explain how CoreML's MIL intermediate representation enables ANE dispatch while maintaining GPU fallback. What information must MIL retain to make dispatch decisions? Provide a concrete example of an operator that cannot dispatch to ANE and why."

**A1 Outline:**
- MIL retains operator identity, input/output shapes, data types, and dynamic attributes (batch size, sequence length)
- Dispatch logic: Check if operator in ANE whitelist AND constraints satisfied (dtype in {FP16, INT8}, shapes within limits)
- Example: Softmax operator
  - ANE dispatch: softmax requires reduction over feature dimension, systolic array cannot express variable reduction pattern
  - GPU fallback: Metal provides atomic operations and shared memory for parallel reduction
  - Constraint violation: ANE cannot maintain running sum of exponentials during matrix multiplication

**Q2 (Practical Engineering):** "You're deploying a 13B LLaMA model on M2 Pro (100 GB/s bandwidth). After conversion to CoreML, inference latency is 200ms (target: 150ms). Describe your optimization strategy: which of quantization, operator fusion, or dispatch tuning would you prioritize, and why? Include quantitative analysis."

**A2 Outline:**
- Baseline analysis: 13 GB model, 100 GB/s bandwidth → 130 ms bandwidth bound
- 200 ms observed means: 200 - 130 = 70 ms overhead (kernel launch, dispatch, synchronization)
- Quantization: 13 GB × 0.5 = 6.5 GB (INT8) → 65 ms bandwidth time, 70 ms overhead → 135 ms (exceeds target!)
  - Benefit: 65 ms, not primary bottleneck
- Dispatch tuning: Identify GPU-only operators (softmax, layer norm) dispatching to CPU
  - CPU softmax: 10 ms per operation × 64 layers = 640 ms (likely culprit for 200 ms)
  - Solution: Implement custom Metal kernel for softmax (5 ms)
  - Result: 200 - 640 + 5 = -435 ms (fix underestimated)
- Conclusion: Profile with powermetrics to locate actual bottleneck before optimization

**Q3 (Advanced Architecture):** "CoreML dispatch logic chooses ANE only if sequence length ≤ 512. Design an inference system that supports 4096-token context while maximizing ANE utilization. What are the tradeoffs?"

**A3 Outline:**
- Constraint: ANE SRAM (1 MB) cannot buffer 4096 × 1024 FP16 activations
- Strategy 1: Chunked inference (process 512 tokens at a time)
  - Benefit: Each chunk uses ANE
  - Tradeoff: Cross-chunk attention lost (QK^T between chunks = 0, breaks semantics)
  - Result: Broken for decoder-only LLMs (attention to all previous tokens required)

- Strategy 2: Hybrid GPU+ANE (GPU for full attention, ANE for FFN)
  - Attention (Q, K, V matmuls): GPU (not ANE-native) → 50 ms
  - Layer norm: GPU → 5 ms
  - FFN (two dense layers): ANE → 30 ms
  - Result: 85 ms per layer, less improvement

- Strategy 3: Speculative decoding (use smaller model with ANE, verify with full model)
  - Small 2B model: 4096 tokens @ 20 tokens/sec (ANE) → draft 10 tokens in 500 ms
  - Verify with 13B GPU: 10 tokens in 150 ms
  - Combined: 650 ms for 10 tokens ≈ 15.4 tokens/sec (baseline GPU: 7.3 tokens/sec)
  - Benefit: 2× throughput with hybrid strategy

- Conclusion: Speculative decoding is practical, others sacrifice correctness or throughput

**Q4 (Systems Design):** "Quantization introduces 1-2% accuracy drop, but ANE dispatch requires INT8/FP16. Design a system that allows models to exceed this accuracy constraint. Consider mixed-precision strategies, PTQ calibration methods, and ANE operator coverage as variables."

**A4 Outline:**
- Problem formulation:
  - Constraint: ANE requires uniform INT8/FP16
  - Objective: Maximize accuracy while minimizing latency
  - Variables: precision per layer, calibration strategy, operator dispatch

- Solution 1: Layer-wise precision tuning
  - Identify sensitive layers (attention QK^T projections) via sensitivity analysis
  - Keep sensitive layers FP32 (CPU), quantize others to INT8 (ANE)
  - Implementation: Custom dispatch logic in MIL compiler
  - Tradeoff: Cross-layer data type conversions (overhead), some ops fallback to CPU

- Solution 2: Advanced calibration for INT8
  - Min-max calibration: crude (1-2% drop)
  - KL-divergence calibration: reduces error to 0.5-1%
  - Learned step size quantization (LSQ): fine-tuning approach, 0.2-0.5% drop
  - Implementation: coremltools with custom calibration hooks

- Solution 3: Operator coverage expansion
  - Implement custom Metal shaders for layer norm, softmax
  - Dispatch to GPU instead of CPU (still ANE-native precision)
  - Hybrid: ANE for GEMM, GPU for elementwise, maintains INT8 precision

- Conclusion: Combination of LSQ quantization (0.5% drop) + operator fusion (reduced dispatch overhead) likely achieves target accuracy while maintaining ANE throughput

---

## CONCLUSION

CoreML represents a mature inference framework leveraging Apple Silicon's heterogeneous architecture through intelligent dispatch and quantization. Mastery requires understanding three abstraction levels: (1) MIL IR and operator semantics, (2) ANE/GPU/CPU capabilities and constraints, and (3) runtime performance implications of dispatch decisions. The field's open problems revolve around expanding ANE operator coverage, supporting dynamic shapes, and integrating quantization-aware training for mobile models. Next module explores Metal programming for direct GPU kernel authorship, bypassing CoreML when maximum control is necessary.

