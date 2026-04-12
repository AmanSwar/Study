# MODULE 12 — CPU Inference Engine Internals

## Table of Contents
1. Introduction: The CPU Inference Stack
2. llama.cpp: The Dominant Open-Source Framework
3. IPEX: Intel's PyTorch Extension for CPU
4. OpenVINO: Intel's Production-Grade Inference
5. oneDNN: The Foundation Library
6. DeepSparse: Sparsity-Optimized CPU Inference
7. Comparative Framework Analysis
8. Format Standards & Interoperability
9. Production Deployment Considerations
10. Conclusion: Framework Selection Guide

---

## 1. Introduction: The CPU Inference Stack

### 1.1 The CPU Inference Ecosystem

The CPU inference landscape in 2026 consists of a distinct stack architecture, fundamentally different from GPU inference tooling:

```
┌─────────────────────────────────────────────────────────┐
│         Application Layer (Web Servers, APIs)          │
├─────────────────────────────────────────────────────────┤
│  Inference Engine Layer                                 │
│  ├─ llama.cpp (open-source dominant)                   │
│  ├─ IPEX (Intel PyTorch)                               │
│  ├─ OpenVINO Model Server (production)                 │
│  └─ DeepSparse (sparsity-optimized)                    │
├─────────────────────────────────────────────────────────┤
│  Format/Model Layer                                     │
│  ├─ GGUF (llama.cpp ecosystem)                         │
│  ├─ ONNX (cross-framework)                             │
│  ├─ OpenVINO IR (Intel proprietary)                    │
│  └─ Custom binary formats (DeepSparse)                 │
├─────────────────────────────────────────────────────────┤
│  Optimization Libraries                                 │
│  ├─ oneDNN (Intel's deep learning primitives)          │
│  ├─ XNNPACK (efficient neural nets)                    │
│  └─ QNNPACK (quantized networks)                       │
├─────────────────────────────────────────────────────────┤
│  CPU Hardware Abstraction                              │
│  ├─ AVX-512 / AVX2 path selection                      │
│  ├─ AMX tile engine initialization                     │
│  ├─ NUMA topology querying                             │
│  └─ CPU feature detection (CPUID)                      │
└─────────────────────────────────────────────────────────┘
```

Each layer has distinct responsibilities and optimization strategies. Unlike GPU frameworks (which mostly abstract hardware behind CUDA), **CPU frameworks expose hardware details as first-class optimization concerns**.

### 1.2 Key Architectural Principles

**Principle 1: Hardware-Aware Kernel Selection**
```c
// Pseudo-code: Runtime path selection
if (cpu_has_avx512_bf16) {
    use_bfloat16_kernel();  // AMX-optimized
} else if (cpu_has_avx512) {
    use_int8_kernel();       // VNNI + AVX-512
} else if (cpu_has_avx2) {
    use_fallback_kernel();   // Portable C++
}
```

**Principle 2: Memory Layout Optimization**
CPU inference frameworks pre-pack weights in hardware-specific layouts to maximize cache efficiency and vectorization.

**Principle 3: Thread-Level Parallelism First**
Unlike GPUs (which use thousands of threads), CPUs rely on careful thread distribution across cores and NUMA domains.

---

## 2. llama.cpp: The Dominant Open-Source Framework

### 2.1 Architecture Overview

llama.cpp is the de facto standard for CPU LLM inference, powering Ollama, LM Studio, and countless applications. Its architecture centers on two concepts:

**GGML (Georgiou's GPU Meta Language)**: A tensor library designed for efficient inference
**GGUF (GGML Universal Format)**: A quantization-friendly binary format

```
llama.cpp Architecture:

┌──────────────────────────────────────────────────────┐
│              Application Interface                   │
│  (main.cpp, server.cpp, Python bindings)            │
└──────────────────────────────────────────────────────┘
                        │
┌──────────────────────────────────────────────────────┐
│            llama.cpp Inference Loop                  │
│  ├─ Tokenization (built-in BPE)                     │
│  ├─ Prompt processing                                │
│  ├─ Sampling (nucleus, temperature)                 │
│  └─ Generation loop management                       │
└──────────────────────────────────────────────────────┘
                        │
┌──────────────────────────────────────────────────────┐
│              GGML Tensor Operations                  │
│  ├─ ggml_mul_mat (GEMM/GEMV)                        │
│  ├─ ggml_rope (RoPE positional encoding)            │
│  ├─ ggml_soft_max (attention softmax)               │
│  ├─ ggml_view (tensor views, zero-copy)            │
│  └─ ggml_quantize (dynamic quantization)            │
└──────────────────────────────────────────────────────┘
                        │
┌──────────────────────────────────────────────────────┐
│         Backend Abstraction Layer                    │
│  ├─ CPU backend (ggml-cpu.c)                        │
│  ├─ CUDA backend (ggml-cuda.cu)                     │
│  ├─ Metal backend (ggml-metal.m for macOS)         │
│  ├─ Vulkan backend (GPU-agnostic)                   │
│  └─ SYCL backend (Intel Arc)                        │
└──────────────────────────────────────────────────────┘
                        │
┌──────────────────────────────────────────────────────┐
│    CPU-Specific Optimized Kernels                    │
│  ├─ ggml_gemm_q4_0_q8_0 (INT4×INT8 GEMM)           │
│  ├─ ggml_gemv_q4_0_q8_0 (INT4×INT8 GEMV)           │
│  ├─ ggml_mul_mat_f32 (FP32 baseline)                │
│  └─ AVX-512 / AVX2 / NEON variants                  │
└──────────────────────────────────────────────────────┘
                        │
           CPU Hardware (AVX-512, AMX)
```

### 2.2 GGML Tensor Library Deep Dive

GGML is essentially a **minimal, inference-optimized tensor library**. Key design principles:

**Principle 1: Immutability**
All tensors are immutable. Computation is expressed as a DAG (directed acyclic graph).

```c
// GGML tensor creation example
struct ggml_tensor {
    enum ggml_type type;      // GGML_TYPE_Q4_0, Q4_1, Q5_0, Q8_0, F32, etc.
    int32_t ne[GGML_MAX_DIMS]; // Tensor shape
    size_t nb[GGML_MAX_DIMS];  // Strides in bytes
    void * data;               // Quantized data buffer

    // Computation graph
    struct ggml_tensor * src0, * src1;
    enum ggml_op op;           // Operation type
};
```

**Principle 2: Quantization-Native**

Unlike PyTorch (which quantizes during export), GGML quantizes **at load time and keeps data quantized** throughout inference:

```c
// GGML quantization types
enum ggml_type {
    GGML_TYPE_F32 = 0,        // 32-bit float, 4 bytes/value
    GGML_TYPE_F16 = 1,        // 16-bit float, 2 bytes/value
    GGML_TYPE_Q4_0 = 2,       // 4-bit quantized, 0.5 bytes/value + scale
    GGML_TYPE_Q4_1 = 3,       // 4-bit with separate min/max
    GGML_TYPE_Q5_0 = 6,       // 5-bit quantized
    GGML_TYPE_Q8_0 = 7,       // 8-bit quantized, 1 byte/value
};
```

Each quantization type includes:
- **Block structure**: Weights grouped into 32-value blocks
- **Inline scaling**: Per-block FP32 scale factor (8 bytes overhead per 32 values = 25%)
- **Activation quantization**: Int8 quantization of activations on-the-fly

Example: **Q4_0 format**
```
Block (32 weights):
┌──────┬──────┬──────┬──────┬──────┬──────┐
│ scale│ min  │  4b  │  4b  │  4b  │  4b  │  (16 values shown, actually 32)
│ f32  │ i8   │ val0 │ val1 │ val2 │ val3 │
└──────┴──────┴──────┴──────┴──────┴──────┘
  4B    1B      2B      2B      2B      2B    = 13B total for 32 values
                                            = 0.40625 bytes/value (vs 4 for FP32)
```

**Principle 3: Zero-Copy Views**

Tensors can be viewed without copying memory:

```c
// Create a transposed view (no copy)
struct ggml_tensor * t_transposed = ggml_transpose(ctx, tensor);

// Create a slice (no copy)
struct ggml_tensor * t_slice = ggml_view_1d(ctx, tensor, len, offset);
```

This is crucial for efficiency: **attention operations can manipulate K/V caches by reshaping without allocating new memory**.

### 2.3 Backend Abstraction

GGML's backend abstraction allows the **same computation graph to execute on different hardware**:

```c
// ggml_backend.h API
typedef struct ggml_backend * ggml_backend_t;

ggml_backend_t ggml_backend_cpu_init(void);
ggml_backend_t ggml_backend_cuda_init(int device);
ggml_backend_t ggml_backend_metal_init(void);

// Execute on any backend
ggml_backend_graph_compute(backend, &graph);
```

**CPU Backend Implementation** (`ggml-cpu.c`):

```c
// CPU backend state
struct ggml_backend_cpu_context {
    int n_threads;                  // Thread pool size
    ggml_threadpool_t threadpool;   // Work queue

    // Temporary buffers for computation
    void * work_buffer;
    size_t work_buffer_size;

    // NUMA topology
    int n_numa_nodes;
    int * numa_node_mapping;  // core -> NUMA node
};
```

### 2.4 Quantized Kernel Implementation

The core optimization in llama.cpp is **efficient quantized GEMM**. Here's how Q4_0 × Q8_0 multiplication works:

```c
// Simplified ggml_gemm_q4_0_q8_0 (from ggml.c)
void ggml_gemm_q4_0_q8_0(
    const ggml_tensor * src0,  // Q4_0 weights [n_rows, n_cols]
    const ggml_tensor * src1,  // Q8_0 activations [n_cols, batch_size]
    ggml_tensor * dst,         // Output [n_rows, batch_size]
) {
    const int n_rows = src0->ne[0];   // Hidden dimension
    const int n_cols = src0->ne[1];   // Weight rows
    const int batch_size = src1->ne[1];

    // Q4_0 format: 2 x (scale_f32, block[16 x 2bits])
    const block_q4_0 * x = (block_q4_0 *) src0->data;
    const block_q8_0 * y = (block_q8_0 *) src1->data;

    float * output = (float *) dst->data;

    // For each output element [row, col]
    for (int col = 0; col < batch_size; col++) {
        for (int row = 0; row < n_rows; row++) {
            float sum = 0.0f;

            // Dot product of single row × column
            // Q4_0 weights: 32 values per block
            for (int block_idx = 0; block_idx < (n_cols + 31) / 32; block_idx++) {
                const block_q4_0 * x_block = &x[row * n_cols_blocks + block_idx];
                const block_q8_0 * y_block = &y[col * n_cols_blocks + block_idx];

                float d_x = x_block->d;      // Scale factor (1.0 / (max - min))
                float d_y = y_block->d;      // Quantization scale

                int8_t  m_x = x_block->m;   // Offset/minimum
                int8_t  m_y = y_block->m;   // Offset

                const uint8_t * q_x = x_block->qs;
                const int8_t  * q_y = y_block->qs;

                // Unpack 4-bit values and accumulate
                for (int i = 0; i < 32; i++) {
                    // Unpack two 4-bit values from one byte
                    uint8_t v0 = q_x[i/2] & 0x0F;      // Low 4 bits
                    uint8_t v1 = (q_x[i/2] >> 4) & 0x0F; // High 4 bits
                    uint8_t v = (i & 1) ? v1 : v0;

                    // Dequantize and multiply
                    float dq_x = d_x * (v - m_x);
                    sum += dq_x * (d_y * q_y[i]);
                }
            }

            output[row + col * n_rows] = sum;
        }
    }
}
```

**Performance Notes:**
- This naive implementation is O(N²) in naive loops
- Real implementation uses AVX-512 to process multiple values in parallel
- With VNNI (Multiply-Add Native Integer Instructions), a single `VPDPBUSD` does 16× (4-bit × 8-bit) → 32-bit accumulations
- Achieves ~400 GFLOP on modern CPUs for Q4_0 × Q8_0 GEMV

### 2.5 GGUF Format Specification

GGUF (GGML Universal Format) is a binary format specifically designed for quantized inference models.

**Format Structure:**
```
GGUF File:
┌─────────────────────────────────┐
│ Magic ("GGUF")                  │
│ Version (uint32)                │
│ Tensor Count (uint64)           │
├─────────────────────────────────┤
│ Key-Value Metadata              │
│ ├─ model.name                   │
│ ├─ model.type ("llama")         │
│ ├─ llama.context_length         │
│ ├─ llama.embedding_length       │
│ ├─ llama.block_count            │
│ └─ quantization.method          │
├─────────────────────────────────┤
│ Tensor Definitions              │
│ ├─ Token Embeddings (Q4_0)      │
│ ├─ Layer 0 Q (Q4_0)             │
│ ├─ Layer 0 K (Q4_0)             │
│ ├─ Layer 0 V (Q4_0)             │
│ ├─ Layer 0 Dense (Q4_0)         │
│ └─ ... for 40+ layers           │
├─────────────────────────────────┤
│ Aligned Tensor Data             │
│ (32-byte alignment for SIMD)    │
└─────────────────────────────────┘
```

**Key advantages over SafeTensors:**
1. **Native quantization support**: Doesn't require conversion
2. **Flexible padding**: Aligns tensors for SIMD operations
3. **Metadata rich**: Includes all model configuration in file
4. **Streaming-friendly**: Can deserialize incrementally

**Loading a GGUF model:**

```c
// gguf.h API (simplified)
struct gguf_context {
    FILE * file;
    size_t file_size;

    // Metadata
    uint32_t version;
    uint64_t n_tensors;
    struct gguf_kv * kv_data;  // Key-value pairs

    // Tensor info
    struct gguf_tensor_info * tensors;
};

gguf_context * gguf_init_from_file(const char * fname) {
    FILE * f = fopen(fname, "rb");

    // Read header
    uint32_t magic;
    fread(&magic, sizeof(uint32_t), 1, f);  // Verify "GGUF"

    // Read version, tensor count
    // ...load metadata...

    // Tensor offsets are stored, data loaded on-demand
    return gguf_create();
}

// Access weights
ggml_tensor * layer_k = gguf_get_tensor(ctx, "blk.0.attn_k");
```

---

## 3. IPEX: Intel's PyTorch Extension for CPU

### 3.1 Overview and Integration

**IPEX (Intel PyTorch Extension)** is Intel's production framework for optimizing PyTorch models on Intel CPUs.

**Key features:**
- **In-place optimization**: Works with existing PyTorch code
- **Graph optimization**: Fuses operations across layers
- **Quantization**: INT8 and BF16 optimizations
- **AMX integration**: Automatic use of Advanced Matrix Extensions

```python
import torch
import intel_extension_for_pytorch as ipex

# Load model
model = torch.load('llama-7b.pth')
model.eval()

# Convert to optimized IPEX variant
with torch.no_grad():
    ipex_model = ipex.optimize(
        model,
        dtype=torch.int8,  # Or torch.bfloat16
        inplace=True
    )

# Inference
with torch.no_grad():
    output = ipex_model(input_ids)
```

### 3.2 Optimization Techniques

**Technique 1: Graph Capture & Fusion**

IPEX captures the computation graph and fuses operations:

```python
# Before optimization
class TransformerBlock(nn.Module):
    def forward(self, x):
        q, k, v = self.qkv(x)                    # Linear
        scores = torch.matmul(q, k.t()) / sqrt_d  # Attention score
        attn_weights = F.softmax(scores, dim=-1) # Softmax
        context = torch.matmul(attn_weights, v)  # Weighted sum
        output = self.out_proj(context)          # Linear
        return output

# After IPEX optimization
# The 5 kernels become 1 fused kernel:
#  LinearFusedSoftmaxLinear (with QK matmul, softmax, VC matmul, out linear)
```

**Technique 2: AMX-Aware GEMM Selection**

```c
// In IPEX's gemm dispatch code
if (cpu_has_amx() && dtype == int8) {
    call_amx_gemm_int8();       // 1024-bit tiles
} else if (cpu_has_avx512_vnni()) {
    call_vnni_gemm_int8();      // 512-bit vectors
} else if (cpu_has_avx2()) {
    call_avx2_gemm_int8();      // 256-bit vectors
}
```

### 3.3 Quantization with IPEX

IPEX provides **static quantization** (weights pre-quantized) and **dynamic quantization** (activations quantized at runtime):

```python
from torch.quantization import quantize_dynamic

# Static quantization (weights only, recommended for inference)
qconfig = {
    'int8': ipex.quantization.QConfig(
        weight_observer=MinMaxObserver(),
        act_observer=MinMaxObserver(),
    ),
    'fp32': None
}

# Quantize specific layers
quantized_model = ipex.quantize(
    model,
    qconfig_dict=qconfig,
    inplace=True
)

# Save quantized model
torch.save(quantized_model.state_dict(), 'model-int8.pth')
```

---

## 4. OpenVINO: Intel's Production-Grade Inference

### 4.1 Architecture

OpenVINO is Intel's comprehensive inference platform, designed for production deployments.

```
OpenVINO Stack:

┌─────────────────────────────────┐
│  Model Zoo & Pre-trained Models │
│  (Object detection, NLP, etc.)  │
└─────────────────────────────────┘
                │
┌─────────────────────────────────┐
│   Model Optimizer (MO)          │
│   ├─ PyTorch → IR conversion    │
│   ├─ Quantization calibration   │
│   └─ Graph optimization         │
└─────────────────────────────────┘
                │
┌─────────────────────────────────┐
│   Intermediate Representation    │
│   (IR: .xml + .bin)             │
│   ├─ Optimized graph            │
│   ├─ Quantization info          │
│   └─ Hardware hints             │
└─────────────────────────────────┘
                │
┌─────────────────────────────────┐
│   OpenVINO Runtime              │
│   ├─ IR Loader                  │
│   ├─ Graph Compiler             │
│   └─ Multi-device orchestration  │
└─────────────────────────────────┘
                │
┌─────────────────────────────────┐
│   CPU Plugin                    │
│   ├─ Primitive Library          │
│   ├─ Kernel Generator           │
│   └─ Execution Graph            │
└─────────────────────────────────┘
                │
         CPU Hardware
```

### 4.2 Intermediate Representation (IR)

OpenVINO converts models to an **Intermediate Representation** optimized for inference:

**Example: Linear layer + ReLU fused to single operation**

```xml
<!-- OpenVINO IR (simplified) -->
<net name="model" version="11">
    <layers>
        <layer id="0" name="input" type="Parameter">
            <output>
                <port id="0" precision="f32"/>
            </output>
        </layer>

        <layer id="1" name="fc1" type="MatMul">
            <data m="1" n="4096" k="12288"/>
            <input>
                <port id="0" node="input" socket="0"/>
                <port id="1" node="weights"/>
            </input>
            <output>
                <port id="0" precision="f32"/>
            </output>
        </layer>

        <layer id="2" name="relu1" type="ReLU">
            <input>
                <port id="0" node="fc1" socket="0"/>
            </input>
            <output>
                <port id="0" precision="f32"/>
            </output>
        </layer>
    </layers>

    <edges>
        <edge from-layer="0" from-port="0" to-layer="1" to-port="0"/>
        <edge from-layer="1" from-port="0" to-layer="2" to-port="0"/>
    </edges>
</net>
```

### 4.3 POT (Post-Training Optimization Tool)

OpenVINO's **POT** performs INT8 quantization with minimal accuracy loss:

```python
from openvino.tools import pot

# 1. Prepare calibration dataset
calibration_dataset = create_calibration_data(model, 100)

# 2. Define quantization algorithm
quantization_algorithm = pot.quantization.DefaultQuantization(
    preset=pot.quantization.QuantizationPreset.PERFORMANCE,
    stat_subset_size=300,
    target_device="CPU",
    metric=["accuracy"],  # Metric to maximize
)

# 3. Quantize
quantized_model = quantization_algorithm.run(
    original_model,
    calibration_dataset,
    metric_fn=accuracy_fn
)

# 4. Save quantized model
ov_config = pot.graph.serialize_model(
    quantized_model,
    'model-int8.xml'
)
```

**Result**: INT8 model with <1% accuracy loss (typically), 4× model size reduction.

---

## 5. oneDNN: The Foundation Library

### 5.1 Architecture

oneDNN (Deep Neural Network Library) is the **low-level primitive library** that both IPEX and OpenVINO build upon.

```
oneDNN Execution Model:

User Code
    │
    ├─ Create primitive descriptor
    │  (operation, data types, memory formats)
    │
    ├─ Query performance characteristics
    │  (expected throughput, memory usage)
    │
    ├─ Create primitive (compile kernel)
    │  (may JIT-compile CPU-specific code)
    │
    └─ Execute primitive
       (with input/output memory handles)
```

### 5.2 Primitive API

oneDNN provides primitives for all neural network operations:

```cpp
#include <dnnl/dnnl.hpp>
using namespace dnnl;

// Create engine (CPU)
engine eng(engine::kind::cpu, 0);
stream s(eng);

// Tensors
memory::dims src_dims = {batch, seq_len, d_model};  // Input
memory::dims weight_dims = {d_model, d_ff};          // Weight matrix
memory::dims dst_dims = {batch, seq_len, d_ff};      // Output

// Create memory objects
auto src_mem = memory(
    {{src_dims}, memory::data_type::f32, memory::format_tag::abc},
    eng,
    src_buffer
);

auto weight_mem = memory(
    {{weight_dims}, memory::data_type::int8, memory::format_tag::cb},
    eng,
    weight_buffer
);

// Create matmul descriptor
auto matmul_d = matmul::desc(
    src_mem.get_desc(),      // Input
    weight_mem.get_desc(),   // Weights (int8)
    memory::desc(dst_dims, memory::data_type::f32, memory::format_tag::abc)
);

// Create primitive
auto matmul_p = matmul::primitive_desc(matmul_d, eng);

// Execute
matmul(matmul_p).execute(s, {
    {DNNL_ARG_SRC, src_mem},
    {DNNL_ARG_WEIGHTS, weight_mem},
    {DNNL_ARG_DST, dst_mem}
});

s.wait();
```

### 5.3 JIT Kernel Generation

oneDNN uses **Just-In-Time compilation** to generate CPU-specific kernels:

```cpp
// oneDNN detects:
// 1. Available instruction sets (AVX-512, AMX, VNNI, etc.)
// 2. CPU cache topology
// 3. Memory characteristics

// Then generates optimized code for:
// - Register blocking (GEMM register tile sizes)
// - Cache blocking (L1, L2, L3)
// - Vectorization strategy
// - Memory access patterns
```

### 5.4 AMX (Advanced Matrix Extensions) Support

For Intel Xeon Sapphire/Granite Rapids CPUs with AMX:

```cpp
// oneDNN automatically selects AMX kernels when available
// AMX provides 1024-bit tiles for:
// - BF16 (16-bit float) GEMM
// - INT8 GEMM

// Example: BF16 GEMM with AMX
auto gemm_d = matmul::desc(
    memory::desc({M, K}, data_type::bf16, format_tag::ab),
    memory::desc({K, N}, data_type::bf16, format_tag::ab),
    memory::desc({M, N}, data_type::f32, format_tag::ab)
);

// oneDNN recognizes bf16 + CPU has AMX, selects AMX kernel
auto gemm_p = matmul::primitive_desc(gemm_d, eng);
```

---

## 6. DeepSparse: Sparsity-Optimized CPU Inference

### 6.1 Motivation: Exploiting Model Sparsity

Modern large models are **highly sparse** after pruning/distillation. DeepSparse exploits this.

```
Weight matrix (7B model layer):
Dense (4,096 × 11,008):
    ████████████████████████████ (100% operations needed)

Sparse (80% pruned):
    ██ █ ██ ███ █  ██ ███ █ ██ (20% operations needed)

Speedup with sparse GEMM: up to 5×
```

### 6.2 Sparse GEMM Format

DeepSparse uses **block-structured sparsity** for efficient execution:

```
Dense GEMM: C = A × B
    A: M × K
    B: K × N
    C: M × N

Sparse GEMM (block sparsity):
    A: M × K (with 2:4 sparsity: 2 non-zero per 4-value block)
    Represented as:
    ├─ indices: which blocks are non-zero
    ├─ values: non-zero block data
    └─ metadata: sparsity pattern
```

### 6.3 DeepSparse Architecture

```
DeepSparse Stack:

┌──────────────────────────────┐
│  Sparsity-aware Model       │
│  (trained with pruning)      │
└──────────────────────────────┘
         │
         ├─ Neural Magic's Sparsification ────┐
         │  (SparseML, SparseZoo)             │
         │                                     │
         ├─ Model Card: sparsity metadata    │
         │  (2:4 sparse, 87% pruned, etc.)   │
         └──────────────────────────────────┘
                     │
┌──────────────────────────────┐
│  DeepSparse Engine           │
│  ├─ Graph compilation        │
│  ├─ Sparse kernel selection  │
│  └─ NUMA-aware scheduling    │
└──────────────────────────────┘
```

### 6.4 Sparse VNNI Instruction

The key to sparse inference on CPU is **VNNI sparse instructions** (VPDPBUSDS for sparse).

```asm
; Regular VNNI (Multiply-Add Native Integer Instructions)
vpdpbusd zmm0, zmm1, [rdi]   ; zmm0 += 16 × (int8 × uint8)

; Sparse variant (coming in Granite Rapids)
vpdpbusds zmm0, zmm1, [rdi]  ; zmm0 += sparse (int8 × uint8)
                             ; Hardware skips zero blocks
```

---

## 7. Comparative Framework Analysis

### 7.1 Framework Comparison Matrix

| Feature | llama.cpp | IPEX | OpenVINO | DeepSparse |
|---------|-----------|------|----------|-----------|
| **Setup effort** | 5 min (binary) | 30 min (compile) | 1 hour (full setup) | 45 min |
| **Model support** | LLaMA, Mistral, Phi | Any PyTorch | Any framework (ONNX) | Sparse models only |
| **Quantization** | INT4, INT8 (native) | INT8, BF16 | INT8, FP16 | 2:4, 80%+ sparsity |
| **Performance (batch=1)** | ★★★★★ | ★★★★☆ | ★★★★☆ | ★★★★★ (sparse) |
| **Throughput (batch=64)** | ★★☆☆☆ | ★★★☆☆ | ★★★☆☆ | ★★★☆☆ |
| **Production-ready** | Limited* | Yes | Yes | Yes |
| **Cost** | Free | Free | Free | $$ (commercial) |
| **Community** | Huge | Growing | Moderate | Niche |

*llama.cpp is production-ready for embedding/small LLM workloads, but lacks robust serving infrastructure.

### 7.2 Selection Guide

**Use llama.cpp if:**
- Building open-source inference server
- Targeting embedded/edge devices
- Need rapid prototyping
- Model is LLaMA-family

**Use IPEX if:**
- Have PyTorch trained models
- Want in-place optimization
- Targeting Intel infrastructure
- Need fine-grained control

**Use OpenVINO if:**
- Enterprise deployment required
- Need multi-device orchestration
- Standardizing on Intel tooling
- Require formal support

**Use DeepSparse if:**
- Models are heavily pruned/sparse
- Can afford 80%+ model reduction
- Throughput (not latency) critical

---

## 8. Format Standards & Interoperability

### 8.1 ONNX: The Cross-Framework Standard

ONNX (Open Neural Network Exchange) enables model portability:

```python
# Convert PyTorch to ONNX
import torch.onnx

model = get_model()
dummy_input = torch.randn(1, 128)

torch.onnx.export(
    model,
    dummy_input,
    "model.onnx",
    input_names=['input'],
    output_names=['output'],
    opset_version=14,
    dynamic_axes={'input': {0: 'batch'}}
)
```

**ONNX→OpenVINO IR conversion:**
```bash
mo.py --input_model model.onnx --output_dir ./ir
# Produces: model.xml (graph) + model.bin (weights)
```

### 8.2 Format Interoperability Challenges

```
PyTorch model
    └─→ ONNX export
        └─→ OpenVINO conversion
            └─→ Potential accuracy loss
                (due to quantization/optimization)

Alternative (more direct):
PyTorch model
    └─→ IPEX optimization
        └─→ Save as .pth
            └─→ Load with IPEX runtime (no conversion loss)
```

---

## 9. Production Deployment Considerations

### 9.1 Model Serving Architecture

```
Production Inference Deployment:

┌────────────────────────────────────┐
│  Load Balancer (nginx)             │
│  ├─ HTTP/2 support                 │
│  ├─ gRPC support                   │
│  └─ Health checks                  │
└────────────────────────────────────┘
        │
        ├─────────────────────────────────┐
        │                                 │
┌───────▼──────────┐          ┌───────────▼────────┐
│ llama.cpp Server │          │ OpenVINO Model     │
│ (8 workers)      │          │ Server (4 workers) │
│ ├─ Single model  │          │ ├─ Multi-model     │
│ ├─ Shared pool   │          │ ├─ Request queue   │
│ └─ Avg latency   │          │ └─ Batch window    │
│    2ms batch=1   │          └────────────────────┘
└──────────────────┘
```

### 9.2 NUMA-Aware Deployment

For dual-socket servers:

```bash
# Core affinity for optimal NUMA performance
numactl -N 0 -m 0 ./llama-cpp-server \
    --model model.gguf \
    --threads 64 \
    --n-gpu-layers 0 \
    --port 8000 &

numactl -N 1 -m 1 ./llama-cpp-server \
    --model model.gguf \
    --threads 64 \
    --n-gpu-layers 0 \
    --port 8001 &
```

---

## 10. Conclusion: Framework Selection Guide

### 10.1 Decision Tree

```
START: Which CPU inference framework?
│
├─ Need production multi-model serving?
│  └─ YES → OpenVINO Model Server
│
├─ Have sparse models (80%+ pruned)?
│  └─ YES → DeepSparse
│
├─ Building from PyTorch?
│  ├─ Need fine-grained control?
│  │  └─ YES → IPEX
│  └─ Otherwise → OpenVINO (via ONNX)
│
└─ Need embedding/small LLM server?
   └─ YES → llama.cpp (fastest setup)
```

### 10.2 Summary Table

| Scenario | Recommended | Rationale |
|----------|-------------|-----------|
| Embedding service | llama.cpp | Simplest, proven at scale |
| LLM inference | llama.cpp or OpenVINO | Trade-off: setup vs enterprise needs |
| Batch processing | IPEX or OpenVINO | Better batch scaling |
| Sparse models | DeepSparse | Purpose-built for sparsity |
| Multi-model | OpenVINO Model Server | Native multi-model support |
| Legacy PyTorch | IPEX | Minimal code changes |

---

## References & Further Reading

1. **llama.cpp**: Georgiou et al., GitHub repository (2024)
2. **IPEX**: Intel PyTorch Extension documentation
3. **OpenVINO**: Intel OpenVINO toolkit documentation
4. **oneDNN**: Intel Deep Neural Network Library documentation
5. **GGML**: Tensor library design patterns
6. **ONNX Standard**: Open Neural Network Exchange specification
7. **DeepSparse**: Neural Magic's sparsity research papers

---

**End of Module 12**

*Total word count: 3,800 words*
