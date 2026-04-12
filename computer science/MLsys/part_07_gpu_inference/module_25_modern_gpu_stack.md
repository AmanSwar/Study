# MODULE 25 — Modern GPU Inference Stack

## 1. Introduction & Learning Objectives

The inference phase of large language models represents the production workload that directly impacts user experience, operational cost, and system sustainability. Unlike training, where throughput-optimized batching amortizes overhead, inference demands extreme efficiency across multiple orthogonal dimensions: latency (time-to-first-token), throughput (tokens/second), memory footprint, and energy efficiency. This module synthesizes the complete modern GPU inference stack—from hardware innovations (NVIDIA's Hopper architecture) through specialized software frameworks (TensorRT-LLM) to cutting-edge algorithmic optimizations (FlashAttention-3, FP8 quantization).

**Learning Objectives:**
- Master NVIDIA Hopper GPU architecture including Tensor Memory Accelerator (TMA), FP8 precision, persistent kernels, and Thread Block Clusters
- Understand TensorRT-LLM's architecture, in-flight batching mechanisms, and plugin extensibility
- Analyze FlashAttention-3's implementation leveraging TMA for async copies, warpgroup GEMM, and FP8 attention computation
- Comprehend FP8 inference quantization strategies including E4M3/E5M2 formats, scaling mechanisms, and accuracy preservation
- Develop practical expertise in Triton language for GPU kernel development with autotuning and torch.compile integration

## 2. Hardware Foundation: NVIDIA Hopper Architecture

### 2.1 Hopper GPU Overview

The NVIDIA Hopper architecture (H100) represents a paradigm shift in GPU design, specifically optimized for inference and specialized workloads. Key specifications include:

- **Compute Density**: 132 SMs (Streaming Multiprocessors) with 128 CUDA cores each = 16,896 CUDA cores (FP32)
- **Memory Hierarchy**: 96GB HBM3, 20GB L2 cache, 228KB per-SM L1/Shared Memory
- **Tensor Core Performance**:
  - FP32 matrix operations: 16 TFLOPS per SM
  - TF32 (Tensor Float 32): 32 TFLOPS per SM
  - FP16/BF16: 64 TFLOPS per SM
  - FP8: 128 TFLOPS per SM (double FP16 throughput)
- **Peak Memory Bandwidth**: 3.35 TB/s (HBM3)
- **Interconnect**: PCIe 5.0, NVLink-C2C for multi-GPU

The architecture emphasizes sustained performance through reduced memory access overhead and enhanced compute density.

### 2.2 Tensor Memory Accelerator (TMA)

The Tensor Memory Accelerator (TMA) is Hopper's most significant innovation for inference workloads. Traditional GPU kernels require cooperative thread blocks to explicitly load data from global memory into shared memory, consuming precious registers and synchronization resources.

**TMA Architecture:**
- Hardware-managed asynchronous memory transfer engine
- Independent from SM execution pipelines
- Capable of copying data from global memory to shared memory without thread participation
- Supports strided, tiled, and multi-dimensional memory patterns natively

**Operational Model:**
```
Traditional approach (Ampere):
[Thread block executes load instructions]
    ↓
[Data loaded into registers]
    ↓
[Data copied to shared memory]
    ↓
[Synchronization barriers]
    ↓
[Computation begins]

TMA approach (Hopper):
[Kernel initiates TMA async copy (non-blocking)]
    ↓
[TMA hardware manages transfer independently]
    ↓
[Threads continue computation on other data]
    ↓
[TMA completion notification]
    ↓
[Synchronization (minimal overhead)]
```

**Memory Patterns Supported:**
- 1D/2D/3D block copies with configurable tile sizes
- Stride patterns for non-contiguous memory layouts
- Transposed copies for layout-changing operations
- Multi-dimensional tensor slicing with native support

**Performance Impact:**
For a typical transformer attention head computation:
- Traditional: Load time overhead ≈ 15-20% of execution
- TMA-optimized: Load time overhead ≈ 3-5% (overlapped with computation)
- Effective throughput improvement: 20-30% for memory-bound kernels

### 2.3 Thread Block Clusters (TBCs)

Thread Block Clusters enable fine-grained synchronization and communication between thread blocks without global synchronization.

**TBC Architecture:**
- Up to 16 thread blocks can form a cluster (Hopper supports 2×2×4 configurations)
- Cluster-level synchronization via `barrier.cluster` instructions
- Shared memory accessible across entire cluster (cluster shared memory)
- Efficient producer-consumer patterns within clusters

**Memory Hierarchy with TBCs:**
```
Global Memory
    ↓ (3.35 TB/s bandwidth)
Cluster Memory (224 KB per SM × cluster)
    ↓ (within-cluster latency: ~10 cycles)
Per-SM Shared Memory (228 KB)
    ↓ (per-thread access latency: ~3-5 cycles)
L1 Cache / Registers
```

**Use Cases:**
- Ring-allreduce patterns for multi-GPU NCCL primitives
- Tile matrix multiply with column blocks computed by separate TBCs
- Distributed reduction operations
- Asynchronous pipeline stages with producer-consumer synchronization

### 2.4 FP8 Tensor Cores

Hopper doubles FP16 tensor throughput via FP8 precision, with two complementary formats:

**E4M3 Format (Exponent 4, Mantissa 3):**
```
[Sign: 1 bit] [Exponent: 4 bits] [Mantissa: 3 bits]
Range: [-2.0, 2.0] × (1 ± 1/8)
Minimum non-zero: 2^-6 ≈ 0.0156
Maximum: 240
Suitable for: Weight storage, forward activations
```

**E5M2 Format (Exponent 5, Mantissa 2):**
```
[Sign: 1 bit] [Exponent: 5 bits] [Mantissa: 2 bits]
Range: [-2^16, 2^16] (much wider dynamic range)
Minimum non-zero: 2^-14 ≈ 6e-5
Maximum: 57344
Suitable for: Gradients, backward computations
```

**Tensor Core Implications:**
- D = A (m×n, FP8) × B (n×k, FP8) → C (m×k, FP32 accumulation)
- Output accumulated in FP32 before potential conversion back to FP8
- Per-token/per-channel quantization supported via scaling factors

### 2.5 Persistent Kernels & Long-Running Kernels

Hopper supports kernels that span entire GPU execution lifetime, maintaining state across millions of iterations.

**Motivation for Persistent Kernels:**
- Traditional kernel: Launch overhead ≈ 1-2 microseconds per iteration
- LLM inference: Generating 128 tokens requires 128+ kernel launches in token-by-token mode
- Total overhead: 128-256 microseconds ≈ 0.5-1ms per sequence
- For 10ms target latency, this represents 5-10% of execution

**Persistent Kernel Pattern:**
```python
# Traditional approach
for token_idx in range(max_tokens):
    attn_out = kernel_attention(query, key, value)
    ffn_out = kernel_ffn(attn_out, weights)

# Persistent approach
kernel_persistent_llm():
    initialize_persistent_state()
    while not finished():
        load_current_token_embedding()
        # All layers computed in single kernel
        for layer in range(num_layers):
            attn_out = compute_attention_block()
            ffn_out = compute_ffn_block()
        write_output_logits()
        barrier_sync_across_blocks()
        wait_for_new_input()
```

**Technical Considerations:**
- Shared memory persists across token iterations (must be manually managed)
- Thread block synchronization within persistent kernel via atomics and grid-level synchronization
- Occupancy requirements: Sufficient thread blocks to keep GPU saturated during sync points
- Memory footprint: KV-cache remains in device memory throughout sequence

## 3. TensorRT-LLM: Production Inference Framework

### 3.1 Architecture Overview

TensorRT-LLM is NVIDIA's end-to-end optimization compiler and runtime for LLM inference, built on top of TensorRT. It provides:

1. **Model Compilation Layer**: Converting PyTorch/ONNX models to optimized kernel graphs
2. **Runtime Execution Engine**: Managing batching, scheduling, and memory
3. **Plugin Extensibility**: Custom CUDA kernels integrated seamlessly
4. **Quantization Pipeline**: FP8, INT8, INT4 with per-layer calibration

**Design Philosophy:**
```
PyTorch Model
    ↓
TensorRT-LLM Parser (C++ API)
    ↓
Intermediate Representation (IR)
    ↓
Optimizations:
    - Kernel fusion
    - Memory optimization
    - Quantization aware rewrites
    - Plugin substitution
    ↓
CUDA Kernel Graph
    ↓
Runtime Executor
    ↓
Inference Results
```

### 3.2 In-Flight Batching (Dynamic Batching)

Traditional batching requires synchronizing all requests to completion before processing next batch. In-flight batching (continuous batching) processes requests asynchronously:

**Traditional Batching:**
```
Time →
Batch 1: [Request A (3 tokens), Request B (5 tokens), Request C (2 tokens)]
         Wait for C to complete
         ↓ (actual compute: max(3,5,2)=5 steps)
Batch 2: [Request D (4 tokens), Request E (6 tokens)]
         Wait for D to complete
         ↓ (actual compute: 6 steps)

Efficiency: (3+5+2+4+6)/(5+6) = 20/11 ≈ 1.82x vs single-token processing
```

**In-Flight Batching:**
```
Time →
Step 1: [A₁, B₁, C₁]
Step 2: [A₂, B₂, C₂] (C completes, new request D added)
Step 3: [A₃, B₃, D₁, E₁]
Step 4: [A₄, B₄, D₂, E₂] (A completes)
Step 5: [B₅, D₃, E₃]
Step 6: [B₆, D₄, E₄] (B completes)
Step 7: [D₅, E₅, F₁]
...continues until all complete

Efficiency: Approaches theoretical maximum of (sum of all tokens) / (max tokens)
In this example: 25 tokens / 7 steps ≈ 3.57x improvement
```

**Implementation Details:**

TensorRT-LLM maintains per-request metadata:
```cpp
struct RequestState {
    int32_t request_id;
    int32_t num_generated_tokens;
    int32_t max_new_tokens;

    // Embedding/KV-cache pointers
    void* embedding_cache;
    void* kv_cache_base;

    // Dynamic memory management
    int32_t kv_cache_offset;

    // Beam search / sampling parameters
    BeamSearchConfig beam_config;
    SamplingConfig sample_config;
};

// Kernel signature with variable batch size
__global__ void kernel_mha_varlength(
    float* q, float* k, float* v,
    int32_t* cu_seqlens,  // Cumulative sequence lengths
    int32_t* cu_token_counts,  // Cumulative token counts
    float* output,
    int32_t batch_size,
    int32_t num_requests
);
```

**Memory Efficiency Mechanics:**
- Requests occupy fixed KV-cache allocations determined at runtime
- Freed KV-cache blocks returned to memory pool
- Block-granular allocation (e.g., 16 tokens per block) reduces fragmentation
- Requests added to batch queue asynchronously

**Scheduling Algorithm:**
```python
class DynamicBatchScheduler:
    def schedule(self, queue, gpu_memory_available):
        selected_requests = []
        cumulative_kv_cache = 0

        for request in queue.sorted_by_priority():
            kv_cache_required = (
                request.num_generated_tokens +
                request.max_new_tokens
            ) * sizeof(KV_cache_per_token)

            if (cumulative_kv_cache + kv_cache_required
                <= gpu_memory_available):
                selected_requests.append(request)
                cumulative_kv_cache += kv_cache_required
            else:
                break

        return selected_requests
```

### 3.3 Plugin System & Custom Kernels

TensorRT-LLM provides C++ and CUDA APIs for integrating custom kernels:

**Plugin Registration:**
```cpp
class CustomAttentionPlugin : public nvinfer1::IPluginV2DynamicExt {
    // Forward computation
    int32_t enqueue(
        int32_t batch_size,
        void const* const* inputs,
        void* const* outputs,
        void* workspace,
        cudaStream_t stream) override;

    // Kernel launch
    nvinfer1::DimsExprs getOutputDimensions(
        int32_t output_index,
        nvinfer1::DimsExprs const* inputs,
        int32_t num_inputs,
        nvinfer1::IExprBuilder& expr_builder) override;
};
```

**Memory Management:**
- Plugins request workspace memory from runtime
- Runtime allocates contiguous blocks to minimize fragmentation
- Plugin responsible for all temporary allocations within workspace

**Optimization Opportunities:**
1. **Kernel Fusion**: Combine attention + add normalization in single kernel
2. **Arithmetic Intensity Enhancement**: Reduce memory bandwidth bottlenecks
3. **Custom Data Layouts**: Exploit tensor-specific properties
4. **Algorithm-Specific Optimizations**: E.g., grouped query attention (GQA)

## 4. FlashAttention-3: Modern Attention Implementation

### 4.1 Algorithmic Foundation

FlashAttention-3 (Shah et al., 2024) represents the latest evolution of IO-efficient attention, explicitly designed for Hopper's TMA and warpgroup capabilities.

**Attention Complexity Analysis:**

Standard attention: Softmax(Q K^T / √d) V requires O(N²) FLOPs and O(N) memory accesses where N = sequence length.

```
Memory Access Patterns (per attention head):

Standard Implementation:
- Load Q (N × d floats): N·d reads
- Load K (N × d floats): N·d reads
- Compute Q·K^T (N² × d FLOPs): N² writes → N² reads
- Load V (N × d floats): N·d reads
- Total: O(N²) memory IO to compute O(N²) FLOPs
- Arithmetic Intensity: (N²) / (N²) = 1 FLOP per byte (extremely memory-bound)
```

**Flash Algorithm Approach:**
```
Memory Access Patterns (FlashAttention-3):

Block-wise Computation:
- Tile Q into blocks of size B_r = 64
- Tile K, V into blocks of size B_c = 64
- For each Q-block:
    - Load Q once (B_r × d reads)
    - Stream across all K, V blocks
    - For each (K-block, V-block) pair:
        - Load K and V (2 × B_c × d reads)
        - Compute attention contribution (B_r × B_c FLOPs)
        - Update running softmax statistics
    - Write output (B_r × d writes)

Total Memory Reads: O(N·d) instead of O(N²)
Arithmetic Intensity: (N²) / (N·d) = N/d FLOPs per byte
For d=128, N=1024: 8 FLOPs per byte (compute-bound range)
```

### 4.2 TMA-Based Async Memory Copies

FlashAttention-3 leverages TMA for asynchronous block copies:

**Traditional Approach (FlashAttention-2):**
```cuda
// Synchronous copy by threads
#pragma unroll
for (int i = threadIdx.x; i < block_size; i += blockDim.x) {
    int idx = threadIdx.y * block_size + i;
    shared_mem[idx] = global_mem[idx];
}
__syncthreads();  // Wait for copy completion
```

**TMA Approach (FlashAttention-3):**
```cuda
// Asynchronous copy by hardware
tma::memcpy_async<Layout>(
    shared_mem_ptr,
    global_mem_ptr,
    thread_block.sync
);
// Overlap copy with previous computation
// No explicit sync needed by threads
cp_async_wait_all();
__syncthreads();
```

**Performance Impact:**
- Frees ~32 threads that previously handled data movement
- These threads now contribute to compute
- TMA operates in parallel with SM execution
- Achieved speedup: 1.3-1.5× for attention computation

### 4.3 Warpgroup GEMM (Matrix Multiply)

Hopper introduces warpgroup-level tensor operations coordinating 4 warps (128 threads) as a unit.

**Warpgroup GEMM Operation:**
```
Input Dimensions:
- Matrix A: 64 × 128 elements (per-warpgroup)
- Matrix B: 128 × 64 elements (per-warpgroup)
- Output C: 64 × 64 elements (per-warpgroup)

Execution Model:
- 4 warps (128 threads) cooperate
- Each warp handles 16×32 portion
- Data layout optimized for warpgroup distribution
- Latency: ~256-512 cycles for 64×64×128 GEMM

Compared to Standard Warp GEMM (32 threads):
- Throughput: 4× improvement (128 vs 32 threads)
- Latency: reduced from ~512 cycles to ~256 cycles
- Occupancy: Requires fewer registers per warpgroup
```

**FlashAttention-3 Usage:**
```cuda
// Compute attention scores: Q·K^T
// Q: [B_r × d], K: [B_c × d]
// Output: [B_r × B_c]

warpgroup::gemm<float>(
    Q.ptr,        // [64 × 128]
    K.ptr,        // [128 × 64] transposed layout
    scores.ptr,   // [64 × 64] output
    scale_factor
);
```

### 4.4 FP8 Attention Computation

FlashAttention-3 extends to FP8 precision for both forward and backward passes:

**Forward Pass (FP8 Storage, FP32 Computation):**
```
Input: Q, K, V in FP8 (E4M3 format)
Scale Factors: s_q, s_k, s_v (FP32 per-token or per-batch)

Computation Flow:
1. Dequantize Q, K: (FP8 × scale) → FP32
2. Compute S = (Q/√d) · K^T (FP32)
3. Apply softmax (FP32)
4. Dequantize V: (FP8 × scale) → FP32
5. Compute O = softmax(S) · V (FP32)
6. Quantize output: O → FP8 (E4M3 format)

Accuracy Preservation:
- Key insight: Softmax produces values in [0, 1]
  → E4M3 format with max 2.0 is sufficient
- Attention weights naturally fit FP8 range
```

**Scaling Strategy:**
```python
def compute_attention_fp8(Q_fp8, K_fp8, V_fp8,
                         s_q, s_k, s_v, d):
    # Dequantization with rescaling
    Q_fp32 = Q_fp8.to(torch.float32) * s_q.unsqueeze(-1)
    K_fp32 = K_fp8.to(torch.float32) * s_k.unsqueeze(-1)
    V_fp32 = V_fp8.to(torch.float32) * s_v.unsqueeze(-1)

    # Attention with proper scaling
    scores = torch.matmul(Q_fp32, K_fp32.transpose(-2, -1)) / math.sqrt(d)
    attn_weights = torch.softmax(scores, dim=-1)
    output = torch.matmul(attn_weights, V_fp32)

    # Quantize with computed scale
    s_out = compute_scale(output)
    output_fp8 = quantize_to_fp8(output, s_out)

    return output_fp8, s_out
```

## 5. FP8 Quantization for Inference

### 5.1 Quantization Fundamentals

Quantization maps floating-point values to lower-precision formats, reducing memory bandwidth and enabling faster computation.

**Affine Quantization Formula:**
```
x_quantized = round(clip(x / scale, min_int, max_int))

For FP8 E4M3:
- min_int = -127, max_int = 127
- scale = max(|x|) / 127

Dequantization:
x_reconstructed = x_quantized × scale
```

**Quantization Error Analysis:**
```
For uniform quantization with scale s:
Quantization error per value: max error = s/2
Per-layer impact depends on:
1. Weight magnitudes distribution
2. Activation magnitudes distribution
3. Layer position in network
```

### 5.2 E4M3 vs E5M2 Format Selection

**E4M3 Characteristics:**
```
Range: [-240, 240]
Precision: 1 / 8 = 0.125 (fixed fractional part)

Best for:
- Weight matrices (typically bounded)
- Activations (typically bounded)
- Forward pass intermediate values

Example distribution fitting:
Weights: μ=0, σ=0.01, 99.9th percentile=0.03
  → Scale = 0.03 / 127 ≈ 2.4e-4
  → Quantization error ≈ 1.2e-4 (relative error: 0.4%)
```

**E5M2 Characteristics:**
```
Range: [-57344, 57344]
Precision: Variable (wider exponent range)

Best for:
- Gradients (wider dynamic range)
- Backward pass (large value variations)

Example gradient distribution:
Gradients: μ=0, σ=0.005, range=[-0.5, 0.5]
  → E5M2 handles wider range with minimal clipping
  → E4M3 would require much smaller scale
  → Risk of underflow in small gradients
```

### 5.3 Scaling Strategies

**Per-Token Scaling (Recommended for Inference):**
```python
def per_token_quantize(tensor, dtype_fp8):
    # tensor shape: [batch, seq_len, hidden]
    # Compute max per token (along hidden dimension)

    max_abs_val = tensor.abs().max(dim=-1, keepdim=True)[0]  # [batch, seq_len, 1]

    if dtype_fp8 == E4M3:
        scale = max_abs_val / 127.0  # E4M3 max value
    else:  # E5M2
        scale = max_abs_val / 448.0  # E5M2 max value

    scale = scale.clamp(min=1e-6)  # Avoid division by zero
    quantized = (tensor / scale).round().clamp(-127, 127).to(torch.int8)

    return quantized, scale

# In attention:
Q_q, scale_q = per_token_quantize(Q, E4M3)
K_q, scale_k = per_token_quantize(K, E4M3)

# Dequantization preserves original range
Q_dq = Q_q.to(torch.float32) * scale_q
K_dq = K_q.to(torch.float32) * scale_k
```

**Per-Channel Scaling (For Weight Matrices):**
```python
def per_channel_quantize_weights(weight, dtype_fp8):
    # weight shape: [out_features, in_features]
    # Compute max per output channel

    max_abs_val = weight.abs().max(dim=1, keepdim=True)[0]  # [out_features, 1]
    scale = max_abs_val / 127.0
    scale = scale.clamp(min=1e-6)

    quantized = (weight / scale).round().clamp(-127, 127).to(torch.int8)

    return quantized, scale

# Per-channel is more granular, preserves activation-specific ranges
```

**Per-Group Scaling (Grouped Quantization):**
```python
def per_group_quantize(tensor, group_size=32):
    # Divide hidden dimension into groups
    batch, seq, hidden = tensor.shape

    # Reshape to [batch, seq, hidden // group_size, group_size]
    tensor_grouped = tensor.reshape(batch, seq, hidden // group_size, group_size)

    # Compute max per group
    max_abs_val = tensor_grouped.abs().max(dim=-1, keepdim=True)[0]
    scale = max_abs_val / 127.0

    quantized = (tensor_grouped / scale).round().clamp(-127, 127)

    return quantized, scale

# Balance between per-token (coarse) and per-element (too fine-grained)
```

### 5.4 Accuracy Preservation Techniques

**1. Calibration-Based Scaling:**
```python
def calibrate_quantization_scale(activation_samples, num_calib_batches=32):
    """
    Compute quantization scales using calibration data
    """
    percentile_vals = []

    for batch in activation_samples[:num_calib_batches]:
        # Use 99.9th percentile instead of max for robustness
        per_token_max = batch.abs().quantile(0.999, dim=-1)
        percentile_vals.append(per_token_max)

    # Statistics across calibration set
    percentile_vals = torch.cat(percentile_vals)
    scale = percentile_vals.mean() / 127.0

    return scale
```

**2. Clipping & Outlier Handling:**
```python
def quantize_with_clipping(tensor, scale, clip_ratio=0.95):
    """
    Clip extreme outliers before quantization
    """
    max_val = scale * 127.0
    clipped = torch.clamp(tensor, -max_val, max_val)
    quantized = (clipped / scale).round().clamp(-127, 127)

    return quantized

# Motivation: Single large outlier can scale entire matrix
# Solution: Clip outliers, accept small accuracy loss
# Impact: Typically < 0.1% accuracy loss vs. 1-3% without clipping
```

**3. Mixed-Precision Inference:**
```python
def mixed_precision_forward(x_fp8, weight_q, scale_q,
                           layer_idx, critical_layers):
    """
    Keep critical layers in higher precision
    """
    if layer_idx in critical_layers:
        # Dequantize for computation
        x_fp32 = x_fp8.to(torch.float32) * scale_fp32
        weight_fp32 = weight_q.to(torch.float32) * scale_q
        output = torch.matmul(x_fp32, weight_fp32.T)
    else:
        # FP8 computation
        x_fp32 = x_fp8.to(torch.float32) * scale_x
        output_fp32 = torch.matmul(x_fp32, weight_q.to(torch.float32) * scale_q)

    return output
```

## 6. Triton Language for GPU Kernels

### 6.1 Triton Programming Model

Triton is a Python-based DSL for writing efficient GPU kernels without explicit CUDA coding. It provides:

1. **Block-Level Abstraction**: Program at tile level, not thread level
2. **Automatic Optimization**: Compiler handles register allocation, vectorization, memory coalescing
3. **Portability**: Code runs on different GPU architectures with minimal changes
4. **Composability**: Kernels combine seamlessly with PyTorch through torch.compile

**Triton vs CUDA Comparison:**
```
CUDA:
- 1000+ lines per custom kernel
- Manual memory management
- Explicit synchronization
- Register/memory tuning per architecture
- Months to master

Triton:
- 50-200 lines per kernel
- Block-level memory management
- Implicit synchronization within block
- Autotuning at compile time
- Days to learn basics
```

### 6.2 Tile-Based Programming

**Basic Tile Concept:**
```python
import triton
import triton.language as tl

@triton.jit
def kernel_tile_add(x_ptr, y_ptr, output_ptr,
                   n_elements,
                   BLOCK_SIZE: tl.constexpr):
    # Each program instance handles one block of data
    pid = tl.program_id(axis=0)  # Program ID (which block)
    block_start = pid * BLOCK_SIZE

    # Generate indices for this block
    offsets = block_start + tl.arange(0, BLOCK_SIZE)  # [0, BLOCK_SIZE)

    # Bounds checking (handles remainder)
    mask = offsets < n_elements

    # Load block of data
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)

    # Element-wise computation
    z = x + y

    # Store result
    tl.store(output_ptr + offsets, z, mask=mask)
```

**2D Tile Example (Matrix Operations):**
```python
@triton.jit
def kernel_gemm(
    a_ptr, b_ptr, c_ptr,
    M, N, K, stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Compute C = A @ B (simplified)
    A: [M, K], B: [K, N], C: [M, N]
    """
    # Get 2D program ID
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    # Initialize accumulator for this tile
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension
    for k in range(0, K, BLOCK_K):
        # Load A block [BLOCK_M, BLOCK_K]
        offsets_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offsets_k = k + tl.arange(0, BLOCK_K)
        a_ptrs = a_ptr + offsets_m[:, None] * stride_am + offsets_k[None, :] * stride_ak
        a = tl.load(a_ptrs)

        # Load B block [BLOCK_K, BLOCK_N]
        offsets_k = k + tl.arange(0, BLOCK_K)
        offsets_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        b_ptrs = b_ptr + offsets_k[:, None] * stride_bk + offsets_n[None, :] * stride_bn
        b = tl.load(b_ptrs)

        # Accumulate: block matrix multiply
        accumulator += tl.dot(a, b)

    # Store result [BLOCK_M, BLOCK_N]
    offsets_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offsets_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + offsets_m[:, None] * stride_cm + offsets_n[None, :] * stride_cn
    mask = (offsets_m[:, None] < M) & (offsets_n[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=mask)
```

### 6.3 Autotuning & Performance Tuning

Triton's autotuner explores hyperparameter combinations to find optimal kernel configurations:

**Autotuning Definition:**
```python
@triton.autotune(
    configs=[
        triton.Config(kwargs={'BLOCK_SIZE': 128}, num_warps=4),
        triton.Config(kwargs={'BLOCK_SIZE': 256}, num_warps=8),
        triton.Config(kwargs={'BLOCK_SIZE': 512}, num_warps=16),
        triton.Config(kwargs={'BLOCK_SIZE': 1024}, num_warps=32),
    ],
    key=['n_elements'],  # Autotune based on input size
)
@triton.jit
def kernel_optimized_add(x_ptr, y_ptr, output_ptr,
                        n_elements,
                        BLOCK_SIZE: tl.constexpr):
    # Kernel implementation (same as before)
    ...
```

**Autotuning Process:**
1. Compile kernel with each config
2. Run on representative data
3. Measure execution time
4. Select best performer
5. Cache result for future use

**Custom Metrics:**
```python
@triton.autotune(
    configs=[...],
    key=['n_elements'],
    prune_by_latency=False,  # Optimize for throughput
    reset_to_zero=['output_ptr'],  # Zero output between runs
)
@triton.jit
def kernel_custom_metric(...):
    ...
```

### 6.4 torch.compile Integration

PyTorch 2.0's torch.compile can optimize Triton kernels automatically:

**Integration Pattern:**
```python
import torch
import triton
import triton.language as tl

@triton.jit
def triton_add(x_ptr, y_ptr, output_ptr, n_elements,
              BLOCK_SIZE: tl.constexpr):
    # Triton kernel definition
    ...

def add_function(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Wrapper function for torch.compile"""
    output = torch.empty_like(x)
    assert x.is_cuda and y.is_cuda
    assert x.shape == y.shape

    n_elements = x.numel()
    grid = (triton.cdiv(n_elements, 1024),)
    triton_add[grid](x, y, output, n_elements, BLOCK_SIZE=1024)

    return output

# Compile for first 3 calls (warm-up), optimize after
compiled_add = torch.compile(add_function, mode='reduce-overhead')

x = torch.randn(1000000, device='cuda')
y = torch.randn(1000000, device='cuda')
result = compiled_add(x, y)  # First call: trace + compile
result = compiled_add(x, y)  # Second/third: cached
result = compiled_add(x, y)  # Subsequent: fully compiled
```

**Optimization Benefits:**
- Elimination of Python interpreter overhead
- Fusion of multiple Triton kernels
- Memory layout optimization across kernel boundaries
- Automatic dtype conversions

## 7. Production Deployment Considerations

### 7.1 Batching Strategies

**Token-per-Request Scheduling:**
```python
class TokenLevelScheduler:
    def __init__(self, max_batch_size=256):
        self.max_batch_size = max_batch_size
        self.active_requests = {}
        self.token_queue = []

    def step(self):
        """Schedule one token for all active requests"""
        selected_ids = list(self.active_requests.keys())[:self.max_batch_size]

        if selected_ids:
            # Process one token for each selected request
            outputs = forward_model(selected_ids)

            # Update request states
            for req_id, output in zip(selected_ids, outputs):
                self.active_requests[req_id].increment_token()
                if self.active_requests[req_id].is_complete():
                    del self.active_requests[req_id]

        return outputs
```

**Sequence-Length Balanced Batching:**
```python
def balanced_batch_scheduler(requests, max_gpu_memory):
    """Group sequences by length to reduce padding"""
    requests_by_length = {}

    for req in requests:
        length = req.max_new_tokens
        bucket = (length // 100) * 100  # Group by 100-token buckets
        if bucket not in requests_by_length:
            requests_by_length[bucket] = []
        requests_by_length[bucket].append(req)

    batches = []
    for length_bucket, batch_reqs in requests_by_length.items():
        # Fit as many sequences as possible
        current_batch = []
        cumulative_tokens = 0

        for req in sorted(batch_reqs, key=lambda r: r.length, reverse=True):
            tokens_needed = req.length * len(current_batch) + req.length
            if tokens_needed <= max_gpu_memory:
                current_batch.append(req)
            else:
                if current_batch:
                    batches.append(current_batch)
                current_batch = [req]

        if current_batch:
            batches.append(current_batch)

    return batches
```

### 7.2 Memory Optimization

**KV-Cache Compression:**
```python
def kv_cache_memory_analysis(num_layers, hidden_dim,
                            max_seq_len, batch_size,
                            dtype=torch.float16):
    """
    Analyze KV-cache memory requirements
    """
    # Per-layer KV-cache size
    bytes_per_token = 2 * hidden_dim * (2 if dtype == torch.float16 else 4)

    total_cache_bytes = (
        num_layers *
        max_seq_len *
        batch_size *
        bytes_per_token
    )

    # Example calculation
    # 80 layers, 8192 hidden dim, 2048 seq len, 32 batch
    example = (
        80 *
        2048 *
        32 *
        (2 * 8192 * 2)  # 2 for K,V; 2 for FP16
    )
    # = 21 GB

    return total_cache_bytes
```

**Attention Backend Selection:**
```python
def select_attention_backend(seq_len, batch_size, hidden_dim):
    """Choose optimal attention implementation"""

    # FlashAttention-3: TMA-based, Hopper-optimized
    if supports_hopper() and seq_len >= 512:
        return "flash_attention_3"

    # FlashAttention-2: Memory-efficient, most GPUs
    elif seq_len >= 256:
        return "flash_attention_2"

    # Standard PyTorch attention: small sequences
    else:
        return "pytorch_sdpa"
```

## 8. Advanced Topics & Research Directions

### 8.1 Speculative Decoding

Speculative decoding uses a small draft model to propose tokens, verified by large model:

```python
class SpeculativeDecode:
    def __init__(self, large_model, draft_model, num_speculative_tokens=4):
        self.large_model = large_model
        self.draft_model = draft_model
        self.num_spec_tokens = num_speculative_tokens

    def forward(self, input_ids, max_length):
        generated = input_ids.clone()

        for _ in range(max_length - len(input_ids)):
            # Draft model: propose next k tokens
            with torch.no_grad():
                draft_logits = self.draft_model(generated)
            draft_tokens = draft_logits[:, -1, :].argmax(dim=-1)

            # Create speculation sequence
            spec_ids = torch.cat([
                generated,
                draft_tokens.unsqueeze(1).repeat(1, self.num_spec_tokens)
            ], dim=1)

            # Large model: verify all at once
            large_logits = self.large_model(spec_ids)

            # Acceptance check
            for spec_idx in range(self.num_spec_tokens):
                token_logits = large_logits[:, -self.num_spec_tokens + spec_idx, :]
                accepted_token = self.acceptance_check(
                    token_logits,
                    draft_tokens[spec_idx]
                )

                if accepted_token:
                    generated = torch.cat([generated, accepted_token.unsqueeze(1)], dim=1)
                else:
                    # Rejection: resample from large model
                    resample_token = sample_from_logits(token_logits)
                    generated = torch.cat([generated, resample_token.unsqueeze(1)], dim=1)
                    break

        return generated
```

### 8.2 Multi-Token Prediction

Recent models predict multiple tokens simultaneously to reduce iterations:

```python
def multi_token_forward(model, input_ids, num_output_tokens=4):
    """
    Model outputs multiple tokens in single forward pass
    Reduces latency from N×T to T (for N tokens predicted)
    """
    # Input: [batch, seq_len]
    # Output: [batch, seq_len + num_output_tokens]

    output_ids = model.generate_multiple(input_ids, num_output_tokens)

    return output_ids
```

## 9. Performance Analysis & Benchmarking

### 9.1 Profiling Tools

**NVIDIA Nsight Compute:**
```bash
# Profile kernel
ncu --set full -o profile.ncu-rep python inference.py

# Analyze results
ncu --import profile.ncu-rep

# Key metrics:
# - SM Efficiency: % of peak SM throughput
# - Memory Efficiency: % of peak memory bandwidth
# - Warp Efficiency: % of threads active
```

**PyTorch Profiler:**
```python
from torch.profiler import profile, record_function, ProfilerActivity

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True) as prof:
    output = model(input_ids)

prof.key_averages().table(sort_by="cuda_time_total")

# Output shows:
# - Kernel name
# - CUDA time (cumulative)
# - CPU time
# - Shapes
```

### 9.2 Performance Metrics

**Throughput Metrics:**
```python
def measure_throughput(model, batch_size, seq_length, num_iterations=100):
    """Measure tokens/second"""

    input_ids = torch.randint(0, 32000, (batch_size, seq_length)).cuda()

    torch.cuda.synchronize()
    start = time.time()

    for _ in range(num_iterations):
        with torch.no_grad():
            output = model(input_ids)
        torch.cuda.synchronize()

    elapsed = time.time() - start

    total_tokens = batch_size * seq_length * num_iterations
    tokens_per_second = total_tokens / elapsed

    return tokens_per_second
```

**Latency Breakdown:**
```python
def profile_latency(model, input_ids):
    """Profile component latencies"""

    components = {
        'embedding': [],
        'attention': [],
        'ffn': [],
        'normalization': [],
    }

    # Instrument model
    hooks = []

    def attention_hook(module, input, output):
        components['attention'].append(time.time())

    for name, module in model.named_modules():
        if 'attention' in name.lower():
            hooks.append(module.register_forward_hook(attention_hook))

    # Run inference
    output = model(input_ids)

    # Analyze timings
    print("Latency Breakdown:")
    for component, times in components.items():
        total_time = sum(times)
        print(f"  {component}: {total_time:.2f}ms")
```

## 10. Summary & Key Takeaways

### 10.1 Critical Insights

1. **Hardware-Algorithm Co-Design**: Modern GPU inference requires simultaneous optimization of both hardware (TMA, warpgroups) and algorithms (FlashAttention-3, FP8)

2. **Memory Bandwidth Dominance**: Inference is memory-bound (arithmetic intensity 1-10 FLOPS/byte). Optimization priorities:
   - Reduce memory access (TMA async copies)
   - Increase compute/memory ratio (block-wise processing)
   - Maximize bandwidth utilization (continuous batching)

3. **Quantization Necessity**: FP8 inference requires careful scaling strategies:
   - Per-token quantization for activations
   - Per-channel for weights
   - Empirical validation on representative data

4. **Software Abstraction Value**: Triton enables 10× productivity improvement over raw CUDA while maintaining performance within 10-15% of hand-optimized kernels

### 10.2 Engineering Best Practices

**Checklist for Production Inference:**
- [ ] Profile baseline performance (latency percentiles, throughput)
- [ ] Evaluate quantization impact on accuracy (>99% retention typical)
- [ ] Implement continuous batching (2-3× throughput improvement)
- [ ] Select attention backend per sequence length
- [ ] Monitor KV-cache memory fragmentation
- [ ] Enable speculative decoding if draft model available
- [ ] Set up periodic profiling to catch regressions

### 10.3 Research Frontiers

- **Mixture-of-Experts (MoE) Inference**: Load balancing across sparse expert networks
- **Multi-Modal Inference**: Efficient handling of image/text sequences
- **Heterogeneous Deployment**: Inference across CPU/GPU/TPU
- **Energy-Efficient Inference**: Reducing power consumption with precision optimization

### 10.4 Further Reading

- Shah et al. (2024). "FlashAttention-3: Fast and Accurate Attention with Asynchronous I/O"
- Triton documentation: https://triton-lang.org/
- NVIDIA TensorRT-LLM: https://github.com/NVIDIA/TensorRT-LLM
- PyTorch GPU Performance Tuning: https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html

---

**Module Completion Status**: Comprehensive coverage of modern GPU inference stack from hardware through software frameworks to cutting-edge optimizations. Students should be able to design, implement, and optimize inference pipelines for production LLM deployment.
