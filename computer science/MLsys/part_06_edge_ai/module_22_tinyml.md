# MODULE 22: TinyML — Inference on Microcontrollers

## Abstract

Inference on resource-constrained microcontroller units (MCUs) represents the extreme frontier of edge AI deployment. With 256 KB to 2 MB of SRAM, no operating system, no virtual memory, and no dynamic heap, MCU-based inference demands radically different system architecture than mobile or server inference. This module examines TensorFlow Lite Micro (TFLite Micro), the reference framework for MCU inference, alongside emerging approaches like MCUNet and MCUNetV2 that co-design neural architecture with memory-aware compilation. We explore interpreter-based execution, memory arena management, operator fusion, INT8/INT4 quantization strategies, and novel techniques like patch-based inference for overcoming SRAM constraints. Special attention to CMSIS-NN (ARM's DSP library for MCU), on-device training capabilities, and emerging RISC-V alternatives rounds out the landscape. This module synthesizes low-level hardware knowledge (Module 21) into practical deployment pipelines.

---

## 1. Introduction: The MCU Inference Challenge

Microcontroller units occupy a unique niche in AI deployment:

**Defining constraints**:
- **Memory**: 256 KB – 2 MB SRAM (vs. 8-12 GB phone)
- **Compute**: 100-500 MMACS INT8 (vs. 5-10 TOPS phone)
- **Execution model**: Bare metal, no OS, no heap allocator
- **Typical models**: Speech recognition (1-5 MB), anomaly detection (100 KB)
- **Latency target**: 10-1000 ms (not real-time like mobile)

**Traditional approach**: Store model weights in external flash, stream activations through SRAM.

**New challenge (2020+)**: Deploy 1-10 MB models (RNNs, small CNNs) on 256-512 KB MCU.

### 1.1 Why MCU Inference Matters

1. **Scale**: 20+ billion MCUs deployed globally (automotive, industrial, IoT)
2. **Cost**: $0.50 – $5 per unit (phones are $500–$1500)
3. **Power**: <1 mW in sleep, 10-50 mW active (months/years on battery)
4. **Latency tolerance**: 100 ms acceptable (voice wakeword detection, motion sensing)

### 1.2 Execution Model Fundamentals

**MCU program layout**:

```
┌─────────────────────────┐
│    Flash Memory (2 MB)  │
├─────────────────────────┤
│  Program code (100 KB)  │  ← Firmware executable
│  Model weights (1 MB)   │  ← Neural network (quantized)
│  Lookup tables (50 KB)  │  ← Activation functions
└─────────────────────────┘

┌─────────────────────────┐
│     SRAM (512 KB)       │
├─────────────────────────┤
│  Runtime state (10 KB)  │  ← Current layer activations
│  Input buffer (50 KB)   │  ← Spectrogram / image
│  Output buffer (10 KB)  │  ← Model predictions
│  Inference arena (430 KB)│  ← Layer outputs, reused
└─────────────────────────┘
```

**Key insight**: Weights **static** in flash, activations **dynamic** in SRAM (reused across layers).

---

## 2. TensorFlow Lite Micro Architecture

### 2.1 Design Philosophy

TFLite Micro (released 2019, led by Google Brain) is purpose-built for MCUs:

**Core principles**:
1. **Statically allocated memory**: No dynamic malloc (real-time guarantee)
2. **Operator registry**: Selective inclusion of ops (only conv, fc, relu needed)
3. **Flat buffer format**: Model metadata embedded in linear buffer (no parsing overhead)
4. **Minimal dependencies**: C++11, no STL, no exceptions

**Code footprint**:
- Core interpreter: ~50 KB (stripped)
- CMSIS-NN ops: ~20 KB
- Total executable: <100 KB on typical MCU

### 2.2 Flat Buffer Model Format

TFLite uses **flatbuffer** encoding (Google standard for serialization):

**Structure**:

```
┌──────────────────────────────────────────┐
│       TFLite Flatbuffer (.tflite)        │
├──────────────────────────────────────────┤
│  Header (4 bytes)                        │
│  Root table offset → (4 bytes)           │
├──────────────────────────────────────────┤
│  Model (root table)                      │
│  ├─ subgraphs: [Subgraph]                │
│  │  ├─ tensors: [Tensor]                 │
│  │  │  ├─ name, shape, type, buffer_idx  │
│  │  └─ operators: [Operator]             │
│  │     ├─ opcode_index, inputs, outputs  │
│  │     └─ builtin_options (op-specific)  │
│  └─ buffers: [Buffer]                    │
│     ├─ weights (indexed)                 │
│     └─ constants                         │
└──────────────────────────────────────────┘
```

**Advantages for MCU**:
- **Zero-copy loading**: Model pointer into flash, no parsing required
- **Alignment safety**: Flatbuffer aligns weights to cache line (32 bytes)
- **Metadata compact**: Shapes/types in-line, minimal overhead

### 2.3 Interpreter Execution Model

TFLite Micro uses a **simple tree-walk interpreter** (not graph optimization, not XLA):

```c
// Pseudo-code: TFLite Micro interpreter loop
struct Interpreter {
    Tensor* tensors;           // All tensor metadata
    Operator* operators;       // All ops in order
    uint8_t* arena;            // Activation buffer
    const Model* model;        // Flatbuffer model (in flash)
};

void Interpreter::Invoke() {
    for (Operator* op : operators) {
        // 1. Validate inputs exist (check tensor buffer_idx)
        for (int input_idx : op->inputs) {
            Tensor* input = &tensors[input_idx];
            ASSERT(input->buffer != NULL, "uninitialized input");
        }

        // 2. Allocate output buffer (in arena)
        for (int output_idx : op->outputs) {
            Tensor* output = &tensors[output_idx];
            output->buffer = arena.allocate(output->bytes);
        }

        // 3. Execute kernel
        BuiltinOperator op_type = op->builtin_code;
        switch (op_type) {
            case kConv2D:
                conv2d_kernel(op, tensors);
                break;
            case kFullyConnected:
                fc_kernel(op, tensors);
                break;
            // ... more ops
        }

        // 4. Mark output tensor as ready
        tensors[output_idx].buffer_valid = true;
    }
}
```

**Characteristics**:
- **Single-threaded**: Interpreter runs sequentially
- **No optimization**: Operators executed in model-defined order (no fusion)
- **Arena allocation**: Outputs allocated/deallocated in LIFO stack order

### 2.4 Memory Arena Management

TFLite Micro uses a **linear arena allocator** (essentially a stack):

```c
struct Arena {
    uint8_t* base;              // Start of SRAM buffer
    size_t size;                // Total size
    size_t position;            // Current allocation pointer
};

uint8_t* Arena::allocate(size_t num_bytes) {
    if (position + num_bytes > size) {
        return NULL;  // Out of memory
    }
    uint8_t* ptr = base + position;
    position += num_bytes;
    return ptr;
}

void Arena::reset() {
    position = 0;  // Reset for next inference
}
```

**Tensor allocation sequence**:

```
Inference begins:
┌──────────────────────────────┐
│ Input tensor (50 KB)         │ ← arena[0:50KB]
│ Layer 1 output (100 KB)      │ ← arena[50KB:150KB]
│ Layer 2 output (80 KB)       │ ← arena[150KB:230KB]
│ Layer 3 output (60 KB)       │ ← arena[230KB:290KB]
│ (deallocate Layer 1)         │ ← arena[50KB:290KB]
│ Layer 2 reused (100 KB)      │ ← arena[290KB:390KB]
│ ...                          │
│ Final output (10 KB)         │ ← arena[390KB:400KB]
└──────────────────────────────┘

Peak SRAM = 400 KB (max allocation extent)
```

### 2.5 Operator Registration

TFLite Micro requires **explicit registration** of each operator kernel:

```c
namespace {
const TFLMRegistration* const micro_builtin_data[] = {
    &tflite::micro::Register_ABS(),
    &tflite::micro::Register_ADD(),
    &tflite::micro::Register_CONV_2D(),
    &tflite::micro::Register_DEPTHWISE_CONV_2D(),
    &tflite::micro::Register_FULLY_CONNECTED(),
    &tflite::micro::Register_RELU(),
    // Only registered ops can execute
};
}
```

**Benefit**: Linker removes unused ops → binary ~20% smaller.

### 2.6 MCU Porting Guide

Porting TFLite Micro to new MCU platform:

```c
// 1. Implement platform-specific I/O
namespace tflite {
namespace micro {
    void LogString(const char* s) {
        // Write to UART, log buffer, etc.
    }
}
}

// 2. Implement memory allocation (if no stdlib)
void* malloc(size_t size) {
    static uint8_t heap[256 * 1024];
    static size_t used = 0;
    if (used + size > sizeof(heap)) return NULL;
    void* ptr = heap + used;
    used += size;
    return ptr;
}

// 3. Link against CMSIS-NN (if ARM MCU)
// ...

// 4. Main inference loop
int main() {
    // Load model from flash
    const unsigned char* model_data = model_tflite;
    tflite::Model* model = tflite::GetModel(model_data);

    // Create interpreter
    static uint8_t arena[256 * 1024];
    tflite::MicroInterpreter interpreter(model, resolver, arena, 256*1024);

    // Allocate tensors
    interpreter.AllocateTensors();

    // Fill input
    float* input = interpreter.typed_input_tensor<float>(0);
    memcpy(input, sensor_data, input_size);

    // Run inference
    interpreter.Invoke();

    // Read output
    float* output = interpreter.typed_output_tensor<float>(0);
    for (int i = 0; i < output_size; i++) {
        printf("Output[%d] = %f\n", i, output[i]);
    }

    return 0;
}
```

---

## 3. MCUNet: Neural Architecture Co-Design for Tiny Devices

### 3.1 The Co-Design Problem (Lin et al., NeurIPS 2020)

Standard models (MobileNetV2, EfficientNet) designed for **mobile** (8 GB DRAM, 5W power) are oversized for **MCU** (256 KB SRAM, 50 mW power).

**MCUNet's insight**: Joint optimization of **model architecture** and **runtime execution** for extreme memory constraint.

### 3.2 TinyNAS: Automated Architecture Search

**Search space** targets MCU-friendly architectures:

```
Macro-architecture search:
├─ Number of layers: [4, 6, 8, 10]
├─ Per-layer channels: [8, 16, 32, 64, 128]
└─ Kernel sizes: [1×1, 3×3, 5×5]

Micro-architecture search:
├─ Depthwise-separable conv ratio: [0.25, 0.5, 1.0]
├─ Skip connections (residual): [yes, no]
└─ Activation functions: [ReLU, ReLU6, Swish]

Constraints:
- Peak SRAM during inference < 256 KB
- Model weights < 1 MB
- Latency < 100 ms on Cortex-M4F
```

**Search algorithm**: DNAS (Differentiable NAS) + hardware-aware loss:

```
Loss = accuracy_loss + λ₁ * peak_memory + λ₂ * latency
```

### 3.3 TinyEngine: Memory-Aware Execution

**Core innovation**: **In-place operations** and **layer fusion** to minimize peak memory.

**Traditional execution** (layer-by-layer):

```
Input (50 KB) → Conv1 (100 KB output) → Relu1 (100 KB) → Conv2 (80 KB output) → ...
Peak memory = 50 + 100 + 80 = 230 KB
```

**TinyEngine execution** (fused, in-place):

```
Input (50 KB)
├─ Conv1 kernel (in SRAM): output written to temporary buffer (100 KB)
├─ Relu1 kernel: reads from temp, overwrites in-place (no copy)
├─ Conv2 kernel: reads fused (relu) output, writes to new temp (80 KB)
└─ Deallocate Conv1 temp buffer

Peak memory = 50 + max(100, 80) = 150 KB
```

**Graph coloring for buffer reuse**:

```c
// MCUNet's buffer allocation strategy
struct TensorAllocation {
    Tensor* tensor;
    int first_use;      // Operator index
    int last_use;       // Operator index
    int memory_offset;  // Position in arena
    int size;
};

// Build lifetime intervals
for (Operator op : model) {
    for (Tensor output : op.outputs) {
        allocation[output].last_use = op.index;
    }
}

// Greedy allocation (offline)
std::vector<TensorAllocation> arena_plan = GreedyColoringAllocate(allocations);

// At runtime, follow plan (no dynamic allocation)
```

### 3.4 Example: MCUNet Results

**Target**: ImageNet classification on STM32H745 (Cortex-M7, 1.3 MB SRAM)

| Model | Peak SRAM | Model Size | Latency (M7) | Accuracy (ImageNet) |
|-------|-----------|-----------|--------------|---------------------|
| MobileNetV2 (0.5×) | 1.2 MB | 2.2 MB | 500 ms | 70.6% |
| EfficientNet-Lite0 | 2.5 MB | 4.8 MB | 800 ms | 75.1% |
| **MCUNet-5** | 256 KB | 350 KB | 800 ms | 60.3% |
| **MCUNet-160** | 1.0 MB | 1.5 MB | 400 ms | 74.4% |

**Key result**: MCUNet-160 achieves 74.4% accuracy (near EfficientNet-Lite0 at 75.1%) while fitting in 1 MB model + 1 MB SRAM (vs. EfficientNet 4.8 MB model alone).

---

## 4. MCUNetV2: Patch-Based Inference

### 4.1 Motivation: Overcoming SRAM Bottleneck

MCUNet-160's 1 MB SRAM requirement is still 4× typical MCU capacity (256 KB). **MCUNetV2** (Lin et al., NeurIPS 2021) introduces **patch-based inference** to eliminate this barrier.

**Key idea**: Process images **patch by patch**, not all-at-once.

### 4.2 Patch-Based Processing

**Traditional convolution** (full image):

```
Input: 224×224×3  →  Conv5×5  →  Output: 220×220×64
Memory: 224×224×3 (150 KB) + 220×220×64 (3 MB) = huge!
```

**Patch-based** (16×16 patches):

```
Image: 224×224 → tile into overlapping 16×16 patches (overlap = receptive field)
For each patch:
    ├─ Load patch (256 bytes)
    ├─ Process through model
    └─ Write output patch

Peak memory: 16×16×3 (0.768 KB input) + layer outputs (~50 KB) = 50 KB total
```

### 4.3 Architecture Constraints for Patch Processing

**Design requirement**: Model must support arbitrary input resolutions (no global pooling in middle of network).

**Typical MCUNetV2 architecture**:

```
Input: (H, W, 3) [variable H, W for different patches]
  ↓
Conv layers: stride=1, padding maintains spatial dims
  ↓
No global pooling (instead: adaptive spatial pooling later)
  ↓
Dense layers: operate on flattened spatial feature maps
```

**Example architecture string**:
```
block_1: [depthwise_conv 3×3 stride 1, channels 32]
block_2: [conv 1×1 stride 1, channels 16, then expand to 64]
block_3: [depthwise_conv 5×5 stride 1, channels 64]
...
final_conv: [conv 1×1 channels C]  ← per-patch output
global_pool: [adaptive 2D pool]     ← patch outputs pooled
dense: [FC to num_classes]
```

### 4.4 Patch Tiling Implementation

```c
// MCUNetV2 inference loop
int patch_size = 16;
int overlap = 8;  // Receptive field padding

for (int y = 0; y < 224; y += patch_size - overlap) {
    for (int x = 0; x < 224; x += patch_size - overlap) {
        // 1. Extract patch from input image
        uint8_t patch[16*16*3];
        ExtractPatch(input_image, 224, 224, x, y, patch_size, patch);

        // 2. Set interpreter input
        uint8_t* model_input = interpreter.input(0)->data.uint8;
        memcpy(model_input, patch, 16*16*3);

        // 3. Run inference
        interpreter.Invoke();

        // 4. Collect output patch
        float* model_output = interpreter.output(0)->data.f;
        int out_size = patch_size - overlap;  // Receptive field crop
        AccumulateOutputPatch(output_feature_map, model_output, out_size);
    }
}

// 5. Final global pooling
FinalPrediction = GlobalAveragePool(output_feature_map);
```

### 4.5 MCUNetV2 Results

**Target**: ImageNet on STM32L476 (Cortex-M4, **96 KB** SRAM)

| Model | Peak SRAM | Model Size | Latency | Accuracy |
|-------|-----------|-----------|---------|----------|
| **MCUNetV2-10** | 96 KB | 150 KB | 2.5 s | 50.3% |
| **MCUNetV2-20** | 96 KB | 350 KB | 8 s | 63.1% |
| **MCUNetV2-40** | 192 KB | 700 KB | 30 s | 69.2% |

**Breakthrough**: Runs ImageNet classification on 96 KB MCU (256× smaller SRAM than MCUNet-160)!

---

## 5. Quantization for MCU: INT8 and INT4

### 5.1 INT8 Quantization Strategies

**MCU quantization goals** (different from mobile):
1. Reduce model size (weights in flash)
2. Enable fixed-point arithmetic (no floating-point hardware on many MCUs)
3. Improve energy per operation

**Symmetric INT8 quantization** (common for MCU):

```
x_quantized = clamp(round(x_float / scale), -128, 127)
x_float ≈ scale * x_quantized

scale = max(|x|) / 128
```

**Example**: Weights for fully-connected layer:

```python
import numpy as np

weights_fp32 = np.random.randn(1024, 2048)  # FC layer
scale = np.max(np.abs(weights_fp32)) / 128.0
weights_int8 = np.round(weights_fp32 / scale).astype(np.int8)

# Verify: max error
max_error = np.max(np.abs(weights_fp32 - weights_int8 * scale))
print(f"Quantization scale: {scale:.6f}")
print(f"Max error: {max_error:.6f}")
```

### 5.2 INT4 Quantization: Further Compression

For extreme memory constraints, **INT4 packing** reduces model size by 2×:

```
4 weights packed per byte:
┌──────────────────────────────────────────┐
│ Byte 0: [w0(4-bit) | w1(4-bit)]          │
│ Byte 1: [w2(4-bit) | w3(4-bit)]          │
└──────────────────────────────────────────┘

Unpacking on the fly:
int4_t w0 = (byte >> 0) & 0x0F;  // Extract lower 4 bits
int4_t w1 = (byte >> 4) & 0x0F;  // Extract upper 4 bits
```

**Cost**: Unpacking adds ~10-20% computational overhead (bit manipulation).

**Trade-off**: 2× model size reduction vs. ~15% latency increase.

### 5.3 CMSIS-NN: ARM DSP Library

**CMSIS-NN** (Cortex Microcontroller Software Interface Standard) provides **hand-optimized INT8 kernels**:

**Architecture**:
- **Intrinsics for ARM Cortex-M7/M55**: SMLAD, SMLAWB (4-lane INT8 MAC in single instruction)
- **Fully-connected**: `arm_fully_connected_q7()` (INT8 input/output, INT32 accumulator)
- **Convolution**: `arm_convolve_HWC_q7_fast()` (uses CMSIS-NN im2col + GEMM)

**CMSIS-NN INT8 FC kernel** (conceptual):

```c
void arm_fully_connected_q7(
    const q7_t *pV,              // Input vector (Q7: INT8)
    const q7_t *pM,              // Weight matrix (Q7)
    uint16_t dim_vec,             // Input vector length
    uint16_t num_of_rows,         // Number of output neurons
    uint16_t bias_shift,          // Bias scaling
    uint16_t out_shift,           // Output scaling
    const q7_t *bias_c,           // Bias vector
    q7_t *pOut,                   // Output vector
    q15_t *vec_buffer             // Temporary buffer
) {
    // 1. Convert Q7 to Q15 (sign-extend)
    for (int i = 0; i < dim_vec; i++) {
        vec_buffer[i] = (q15_t)pV[i] << 7;  // Sign-extend INT8 → INT16
    }

    // 2. Matrix-vector multiply (GEMV)
    for (int i = 0; i < num_of_rows; i++) {
        int sum = 0;
        const q7_t *pW = &pM[i * dim_vec];

        // Unroll 2: SMLAD instruction (4-lane INT16 MAC per cycle)
        for (int j = 0; j < dim_vec; j += 2) {
            q31_t v_input = *(q31_t*)&vec_buffer[j];      // 2 INT16 values
            q31_t v_weight = arm_nn_read_q7x4(&pW[j]);    // 4 INT8 values (sign-ext)
            sum = __smlad(v_input, v_weight, sum);        // 4-lane multiply-add
        }

        // 3. Bias and scaling
        sum += (int)bias_c[i] << bias_shift;
        sum = (sum + (1 << (out_shift - 1))) >> out_shift;

        // 4. Clamp to INT8 range
        pOut[i] = (q7_t)__SSAT(sum, 8);
    }
}
```

**Performance** (STM32H745 Cortex-M7):
- Scalar INT8 MAC: ~1 cycle per MAC (theoretical 200 MMACS at 200 MHz)
- CMSIS-NN INT8 GEMV: ~0.6 cycles per MAC (due to SMLAD 4-lane operations)
- Speedup: ~1.7× vs. scalar

---

## 6. Memory Planning and Graph Coloring

### 6.1 Tensor Lifetime Analysis

**Goal**: Compute minimum SRAM needed by analyzing when each tensor is live.

```c
struct TensorInfo {
    int tensor_idx;
    int first_op;      // First operator that reads this tensor
    int last_op;       // Last operator that produces this tensor
    size_t size_bytes;
    bool is_weight;    // True if static (in flash)
    bool is_input;     // True if network input
};

// Build lifetime intervals
std::vector<TensorInfo> GetTensorLifetimes(const Model& model) {
    std::vector<TensorInfo> lifetimes;

    for (size_t op_idx = 0; op_idx < model.operators.size(); op_idx++) {
        const Operator& op = model.operators[op_idx];

        // Mark input tensors' first use
        for (int tensor_idx : op.inputs) {
            if (lifetimes[tensor_idx].first_op == -1) {
                lifetimes[tensor_idx].first_op = op_idx;
            }
        }

        // Mark output tensors' last use
        for (int tensor_idx : op.outputs) {
            lifetimes[tensor_idx].last_op = op_idx;
        }
    }

    return lifetimes;
}
```

### 6.2 Graph Coloring Algorithm

**Interval graph coloring** assigns SRAM addresses to tensors such that non-overlapping lifetimes reuse same memory:

```python
def graph_coloring_allocate(tensor_lifetimes, arena_size):
    """
    Assign SRAM offsets to tensors to minimize peak memory.
    """
    # Sort tensors by first use (earliest first)
    tensor_lifetimes.sort(key=lambda t: t.first_op)

    allocations = {}
    active_intervals = []

    for tensor in tensor_lifetimes:
        if tensor.is_weight:
            continue  # Skip weights (in flash)

        # Remove expired intervals (tensors no longer needed)
        active_intervals = [t for t in active_intervals
                           if t.last_op >= tensor.first_op]

        # Find earliest available offset that doesn't conflict
        offset = 0
        for active_tensor in sorted(active_intervals,
                                    key=lambda t: allocations[t.tensor_idx]):
            active_offset = allocations[active_tensor.tensor_idx]
            if offset == active_offset:
                # Conflict, try next offset
                offset = active_offset + active_tensor.size_bytes
                offset = (offset + 31) & ~31  # Align to 32-byte cache line
            else:
                break

        allocations[tensor.tensor_idx] = offset
        active_intervals.append(tensor)

    # Compute peak memory (max allocation extent)
    peak_memory = max(allocations[t] + t.size_bytes
                     for t in tensor_lifetimes
                     if not t.is_weight)
    return allocations, peak_memory
```

### 6.3 Example: Tensor Lifetime Analysis

**Simple 3-layer model**:

```
Input (50 KB, lifetime [0, 0])
  → Conv1 output (100 KB, lifetime [0, 1])
    → ReLU1 (inplace, no new buffer, lifetime [0, 1])
      → Conv2 output (80 KB, lifetime [1, 2])
        → ReLU2 (inplace, lifetime [1, 2])
          → Conv3 output (60 KB, lifetime [2, 3])
            → Output (10 KB, lifetime [3, 3])

Memory allocation (greedy coloring):
┌─────────────────────────────────┐
│ Offset 0: Input (50 KB)         │  [0, 0]
├─────────────────────────────────┤
│ Offset 50K: Conv1 output (100 KB)│ [0, 1]
├─────────────────────────────────┤
│ Offset 150K: Conv2 output (80 KB)│ [1, 2]
├─────────────────────────────────┤
│ Offset 230K: Conv3 output (60 KB)│ [2, 3]
└─────────────────────────────────┘

At operator 1: Input + Conv1 live (150 KB)
At operator 2: Conv1 (deallocate) + Conv2 live (80 KB)
At operator 3: Conv2 (deallocate) + Conv3 live (60 KB)

Peak memory = 150 KB
```

---

## 7. On-Device Training on MCU

### 7.1 Motivation and Feasibility

**On-device training** (not just inference) on MCU enables:
- **Personalization**: Fine-tune model on user data without cloud
- **Federated learning**: Train across devices, aggregate gradients
- **Low power**: Local training uses less bandwidth than cloud round-trip

**Feasibility**: Extremely limited (shallow networks, INT8 precision, single pass).

### 7.2 Quantization-Aware Fine-Tuning

**Approach**: Pre-trained model (via normal QAT on desktop), then fine-tune on MCU.

```c
// MCU-based fine-tuning (pseudo-code)
struct MicroTrainer {
    Model* model;
    Optimizer optimizer;  // SGD (simplest), no Adam (requires 2 buffers per param)
};

void FinetuneOneStep(MicroTrainer& trainer, float* input_data) {
    // 1. Forward pass (inference)
    float* logits = ForwardPass(trainer.model, input_data);

    // 2. Compute loss (cross-entropy simplified)
    float loss = 0.0;
    for (int i = 0; i < num_classes; i++) {
        float p_i = softmax(logits[i]);
        loss -= label[i] * log(p_i);  // Cross-entropy
    }

    // 3. Backward pass (backprop through layers)
    float* grad = BackwardPass(trainer.model, logits, label);

    // 4. Update weights (SGD: w ← w - lr * grad)
    for (int i = 0; i < num_params; i++) {
        int8_t* w = &trainer.model->weights[i];
        float w_float = dequantize(w);
        w_float -= learning_rate * grad[i];
        w[i] = quantize(w_float);
    }
}
```

### 7.3 Limitations and Current State

**Practical constraints**:
- **Forward-backward**: ~2× memory of inference (gradients must be stored)
- **Precision loss**: INT8 gradients lose significant information
- **No optimizer state**: SGD only (Adam needs 2 buffers per parameter)
- **Convergence slow**: Very small learning rates required for INT8

**Current practice**: Companies like Qualcomm, ARM prototype on-device training but rely on off-device pre-training + on-device fine-tuning (not training from scratch).

---

## 8. RISC-V for Edge AI

### 8.1 RISC-V Opportunity in MCU Space

**RISC-V advantages**:
- Open ISA (no ARM licensing cost)
- Modular: base I + extensions (M, F, D, V for ML)
- Growing ecosystem (SiFive, Alibaba, Western Digital)

**Market position** (2024): ~5-10% MCU share, growing.

### 8.2 RVV (RISC-V Vector Extension)

**RVV design**:
- **Scalable vector length** (like ARM SVE)
- **Predicated operations**
- Typical mobile RISC-V: 128-256 bit (same as ARM NEON)

**RVV vector multiply (pseudo-assembly)**:

```asm
vle8.v      v0, (x0)         # Load 16 INT8 values (128-bit RVV)
vle8.v      v1, (x1)         # Load 16 INT8 weights

vmul.vv     v2, v0, v1       # Element-wise multiply (16× int8)
vadd.vv     v3, v3, v2       # Accumulate

add         x0, x0, x2       # Next input
add         x1, x1, x2       # Next weight
blt         x3, x4, loop     # Loop if more data
```

**Performance parity**: RVV achieves similar throughput to ARM NEON for same vector length.

### 8.3 RISC-V ML Ecosystem

**TFLite Micro support**: Partial (interpreter works, but CMSIS-NN equivalent missing).

**Emerging solutions**:
- **Alibaba TVM + RISC-V**: LLVM backend generates RVV instructions
- **SiFive specific optimizations**: Hand-tuned kernels for SiFive HiFive1 Rev B

---

## 9. Example: Complete MCU Inference Pipeline

### 9.1 From Keras Model to Deployed MCU Binary

**Step 1: Train and quantize on desktop (TensorFlow)**:

```python
# Train model
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
model.fit(train_images, train_labels, epochs=10)

# Quantize to INT8
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]

def representative_dataset():
    for image in test_images[:100]:
        yield [image[np.newaxis, ...].astype(np.float32)]

converter.representative_dataset = representative_dataset
quantized_model = converter.convert()

with open('model.tflite', 'wb') as f:
    f.write(quantized_model)
```

**Step 2: Convert to C array (embedding model in firmware)**:

```bash
xxd -i model.tflite > model_data.h
```

**Step 3: MCU firmware (STM32H745)**:

```c
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "model_data.h"

// Global tensors and arena
static uint8_t tensor_arena[256 * 1024];
static tflite::MicroErrorReporter micro_error_reporter;
static tflite::MicroInterpreter* interpreter = nullptr;

void setup_inference() {
    // Get model from flash
    const tflite::Model* model = tflite::GetModel(model_tflite);

    // Create resolver with only needed ops
    static tflite::MicroMutableOpResolver<5> resolver;
    resolver.AddBuiltin(tflite::BuiltinOperator_CONV_2D,
                       tflite::ops::micro::Register_CONV_2D());
    resolver.AddBuiltin(tflite::BuiltinOperator_MAX_POOL_2D,
                       tflite::ops::micro::Register_MAX_POOL_2D());
    resolver.AddBuiltin(tflite::BuiltinOperator_FULLY_CONNECTED,
                       tflite::ops::micro::Register_FULLY_CONNECTED());
    resolver.AddBuiltin(tflite::BuiltinOperator_SOFTMAX,
                       tflite::ops::micro::Register_SOFTMAX());

    static tflite::MicroInterpreter static_interpreter(
        model, resolver, tensor_arena, 256 * 1024, &micro_error_reporter);

    interpreter = &static_interpreter;
    interpreter->AllocateTensors();
}

void run_inference(uint8_t* image_data) {
    // Fill input
    uint8_t* input = interpreter->typed_input_tensor<uint8_t>(0);
    memcpy(input, image_data, 28 * 28);

    // Invoke
    interpreter->Invoke();

    // Read output
    float* output = interpreter->typed_output_tensor<float>(0);
    int predicted_class = argmax(output, 10);

    printf("Predicted class: %d\n", predicted_class);
}

int main() {
    setup_inference();

    // Capture image from camera
    uint8_t image_buffer[28 * 28];
    capture_image_from_sensor(image_buffer);

    // Run inference
    run_inference(image_buffer);

    return 0;
}
```

---

## 10. Conclusion and Future Directions

### 10.1 State-of-the-Art (2024)

**Achievable on STM32H745 (1.3 MB SRAM)**:
- Small CNNs: MNIST-scale (60% ImageNet-equivalent)
- Speech: Keyword spotting (wake word detection)
- Anomaly detection: Sensor-based (accelerometer, temperature)
- Latency: 100-500 ms typical

**Not yet feasible**:
- Large language models (even 1B parameter models)
- Real-time video processing
- Multi-model ensembles

### 10.2 Emerging Techniques

1. **Sparse computation**: Exploiting model sparsity for 2-3× speedup
2. **Heterogeneous execution**: Offload convolution to DSP, other ops to CPU
3. **Neural architecture search**: AutoML specifically targeting MCU constraints
4. **Post-training compression**: Further quantization + pruning after QAT

### 10.3 Key Insights

- **Memory footprint dominates**, not compute (opposite of datacenter)
- **Arena-based allocation** enables predictable, bounded memory usage
- **Quantization to INT8 is mandatory** for any non-trivial model
- **Architecture co-design** (MCUNet) essential for extreme constraints
- **RISC-V opportunity**: Open ISA could disrupt ARM monopoly in MCU ML

### 10.4 Design Principles for MCU Inference

1. **Profile first**: Measure memory, latency on target before optimizing
2. **Quantize early**: INT8 from start, INT4 if desperate
3. **Reuse buffers**: Offline compute tensor lifetimes, allocate greedily
4. **Fuse operations**: Conv+ReLU, ReLU+Add eliminate intermediate writes
5. **Avoid allocators**: Static memory planning only
6. **Benchmark relentlessly**: Cycle counts ≠ wall-clock time on MCU (cache effects)

---

## Further Reading

- Lin et al.: "MCUNet: Tiny Deep Learning on IoT Devices" (NeurIPS 2020)
- Lin et al.: "MCUNetV2: Memory-Efficient Patch-based Inference for Tiny Deep Learning" (NeurIPS 2021)
- Google TensorFlow Lite Micro: https://www.tensorflow.org/lite/microcontrollers
- ARM CMSIS-NN: https://github.com/ARM-software/CMSIS-NN
- Qualcomm Hexagon DSP ML Benchmarks (2023)
