# MODULE 19 — llama.cpp on Apple Silicon

## 1. SYSTEMS OVERVIEW

llama.cpp is an optimized C++ inference engine for large language models, originally designed for LLaMA but now supporting any GGUF-quantized model (Mistral, Vicuña, Hermes, etc.). It achieves remarkable efficiency through:

1. **Quantization-First Design:** Operates on GGUF format (weights pre-quantized to INT8, INT4, INT5, etc.)
2. **Metal Backend:** Uses Metal compute shaders for GPU acceleration on Apple Silicon
3. **Memory Mapping (mmap):** Loads model weights directly from disk into address space, exploiting unified memory
4. **Minimal Dependencies:** C++11 standard library, no GPU SDK requirements (unlike CUDA)

On Apple Silicon, llama.cpp combines GGUF quantization, Metal acceleration, and mmap to achieve unprecedented efficiency: **7B LLaMA generates 7-10 tokens/sec on M2 with <15W power draw**, compared to 1-2 tokens/sec on CPU or 20W+ on discrete GPU.

### 1.1 GGUF Format Overview

GGUF (Georgi's Unified Format, originally Good Game Until Friday) is a binary format storing quantized models with metadata. Unlike PyTorch's `.pt` format or ONNX, GGUF is designed for inference with explicit weight quantization.

**GGUF File Structure:**

```
┌──────────────────────────────────────────────────┐
│  GGUF Header                                     │
│  - Magic number: 0x46465547 (little-endian)     │
│  - Version: 3 (current)                          │
│  - Size of tensor names section                  │
│  - Number of tensors                             │
├──────────────────────────────────────────────────┤
│  Key-Value Metadata                              │
│  - Model name: "LLaMA-7B"                        │
│  - Architecture: "llama" or "mistral"            │
│  - Token embedding dimension: 4096               │
│  - Number of layers: 32                          │
│  - Quantization type: "Q4_K_M"                   │
│  - (Custom metadata for model-specific tuning)   │
├──────────────────────────────────────────────────┤
│  Tensor Data (one tensor per layer)              │
│  Each tensor:                                     │
│  - Name (e.g., "blk.0.attn.wq")                 │
│  - Shape: (4096, 4096) for projection matrix    │
│  - Data type: GGML_TYPE_Q4_K (quantized)        │
│  - Offset in file: position of weight data      │
│  - Actual bytes: compressed/quantized weights   │
│                                                  │
│  Layout: 1st layer weights, 2nd layer, ...      │
│  Last: Embedding table, output projection       │
└──────────────────────────────────────────────────┘
```

### 1.2 Quantization Types in GGUF

llama.cpp supports multiple quantization methods, each trading off accuracy vs model size/speed:

| Format | Bits/Weight | Model Size (7B) | Accuracy Loss | Speed | ANE Support? |
|---|---|---|---|---|---|
| F32 (no quant) | 32 | 26 GB | 0% | 1.0× | ✗ (too large) |
| F16 | 16 | 13 GB | <0.2% | 0.95× | ✗ (CoreML native) |
| Q8_0 | 8 | 7 GB | <0.5% | 0.95× | Possible (via dequant) |
| Q5_K_M | 5 | 4.3 GB | 0.5-1% | 0.85× | ✗ (dequant overhead) |
| Q4_K_M | 4-5 | 3.5 GB | 1-2% | 0.7× | ✗ (dequant overhead) |
| Q3_K | 3 | 2.6 GB | 2-3% | 0.5× | ✗ (quality too low) |
| Q2_K | 2 | 1.7 GB | 5%+ | 0.3× | ✗ (significant loss) |

**Recommended for Apple Silicon:**
- **Q4_K_M:** Best tradeoff (3.5 GB model, minimal accuracy loss, 70% speed vs F32)
- **Q5_K_M:** If RAM abundant (4.3 GB, better quality)
- **Q8_0:** If maximum accuracy needed (still 7 GB, no accuracy loss vs F16)

### 1.3 Metal Backend Architecture

llama.cpp's Metal backend replaces CPU tensor operations with GPU kernels for:
- Matrix multiply (Q4_K_M dequantized × FP32 activations)
- Attention softmax
- Layer normalization
- Elementwise operations (ReLU, GELU approximations)

**Dispatch Decision Logic:**

```
For each tensor operation:
  if tensor.element_count < 1M:
    execute_on_CPU  # Small tensors: kernel launch overhead > actual compute
  elif quantization_type == Q4_K:
    # Dequantize on GPU (Q4 → FP16) then multiply
    execute_on_GPU_with_dequant
  elif operation == "attention_softmax":
    execute_on_GPU  # Reduction ops parallelized well on GPU
  else:
    execute_on_CPU  # Default fallback
```

---

## 2. THEORETICAL FOUNDATION

### 2.1 GGUF Quantization Mechanics

**Q4_K_M (4-bit Quantization with K-quantization + Importance Masking):**

```
Input tensor: W ∈ ℝ^{m × n} (e.g., projection matrix 4096 × 4096)
Goal: Compress to 4-bit integers while preserving accuracy

K-means clustering approach:
  1. Divide W into blocks (e.g., 32×32 tiles)
  2. For each block: Find 16 representative values (centroids) via K-means
  3. Each weight maps to nearest centroid index (4 bits)
  4. Store: centroid values (FP16 × 16 = 256 bytes) + indices (4 bits × 1024 = 512 bytes)

Compressed size per block:
  Original: 32 × 32 × 2 bytes (FP16) = 2048 bytes
  Compressed: 256 (centroids) + 512 (indices) = 768 bytes
  Compression ratio: 2048 ÷ 768 = 2.67× (or 6 bits/weight, rounded to 4-5)

Dequantization (at inference):
  For each block:
    indices = load_indices_from_storage()  # (32, 32) array of 4-bit values
    centroids = load_centroids()            # 16 FP16 values
    W_reconstructed = centroids[indices]    # Lookup table
```

**Accuracy Analysis:**

Standard K-quantization loses information; Q4_K_M adds **importance masking**:

```
For sensitive weights (e.g., attention QK projection):
  - Use finer quantization (5-6 bits)
For less sensitive weights (e.g., FFN intermediate):
  - Use coarser quantization (3-4 bits)

Importance score: Based on gradient magnitude during QAT
  importance = |∂loss/∂w|  (higher magnitude = more sensitive)

Mixed-bit strategy:
  - Top 20% of weights: 6 bits (highest importance)
  - Next 30%: 5 bits
  - Bottom 50%: 3-4 bits

Result: Effective 4-5 bits per weight on average, <1% accuracy loss
```

### 2.2 Memory Bandwidth Analysis for Quantized GEMM

For quantized matrix multiply on Apple Silicon:

```
GEMM: C = A @ B
  A: (m, k) FP32 activations
  B: (k, n) Q4_K_M quantized weights

Dequantization overhead:
  Load 4-bit weights: k × n × 0.5 bytes
  Dequantization: lookup centroids, expand to FP32 (k × n × 4 bytes output)
  Latency: 50-100 μs (GPU dequant kernel)

Memory bandwidth required:
  Input: A (m × k × 4) + B_quantized (k × n × 0.5)
  Output: C (m × n × 4)
  Total: 4.5 × k × n bytes per GEMM

Arithmetic Intensity (AI):
  FLOPs: 2 × m × k × n
  AI = (2 × m × k × n) / (4.5 × k × n) = 0.44 × m FLOPS/byte

For m=1 (single token generation):
  AI = 0.44 FLOPS/byte (extremely memory-bound)

For m=32 (batch of 32 tokens):
  AI = 14 FLOPS/byte (still memory-bound, but less severe)

Roofline ceiling:
  Throughput = min(peak_flops, bandwidth × AI)
           = min(600 GFLOPS, 100 GB/s × 0.44 × 1)
           = min(600 GFLOPS, 44 GFLOPS)
           = 44 GFLOPS (bandwidth bound, as expected)
```

### 2.3 Speculative Decoding for LLMs

Speculative decoding accelerates generation by using a smaller, faster model to draft tokens, then verifying with the full model.

**Algorithm (Leviathan et al., 2023):**

```
Given:
  - Draft model: Small 2B LLaMA (fast)
  - Target model: Full 7B LLaMA (accurate)
  - Temperature: Sampling parameter τ

Process:
  1. Use draft model to generate K candidate tokens (e.g., K=5)
     - Draft latency: 5 × 50 ms = 250 ms
  2. Feed all K candidates through target model in batch
     - Target latency: batch_processing = 60 ms (vs 500 ms sequential)
  3. Verify each candidate:
     - If P_target(token) > P_draft(token) * threshold:
       Accept token (early exit)
     - Else:
       Sample from (P_target - P_draft) / (1 - P_draft)
  4. Once any candidate rejected, start new draft round

Speedup:
  Sequential generation: K tokens × 100 ms = 500 ms
  Speculative generation: K × 50 ms (draft) + 1 × 60 ms (verify) = 310 ms
  Speedup: 500 ÷ 310 = 1.6×

Requirement:
  - Draft model must be << target model size (2B vs 7B works)
  - Prediction accuracy: Draft should predict 70%+ of target tokens correctly
```

---

## 3. HARDWARE MAPPING

### 3.1 GGUF Model Loading via mmap

llama.cpp uses memory mapping (mmap) to load GGUF files efficiently:

```c++
// llama.cpp mmap_impl.cpp (simplified)

struct llama_mmap {
    void* addr;      // Mapped memory address
    size_t size;     // Total size
    int fd;          // File descriptor
};

// Load model with mmap
llama_mmap* mmap_open_model(const char* fname) {
    int fd = open(fname, O_RDONLY);
    struct stat st;
    fstat(fd, &st);  // Get file size

    void* addr = mmap(
        NULL,                // Any address
        st.st_size,          // Full file size (e.g., 3.5 GB)
        PROT_READ,           // Read-only
        MAP_SHARED,          // Share with other processes
        fd,                  // File descriptor
        0                    // Offset in file
    );

    // addr now points to model weights in virtual address space
    // But physical pages are on disk until accessed (demand paging)
    return new llama_mmap{addr, (size_t)st.st_size, fd};
}

// Compute forward pass
void llama_forward(llama_context* ctx, const llama_tensor* weights) {
    // weights->data is pointer into mmap'd region
    // Accessing weights[i] triggers page fault (if not in memory)
    // OS loads page from disk to physical RAM

    // Apple Silicon UMA: Loaded page is visible to both CPU and GPU
    // No PCIe transfer needed
}
```

**Performance Implications:**

1. **Initial Load:** First inference triggers 3.5 GB model load into physical RAM (~5-10 sec on SSD)
2. **Subsequent Requests:** If model stays in RAM, zero reload latency
3. **Model Switching:** Two models A (3.5 GB) and B (3.5 GB) totaling 7 GB
   - If 16 GB RAM: Both fit, instant switching
   - If 8 GB RAM: One at a time, 5 sec reload penalty
4. **Unified Memory Benefit:** Loaded pages automatically accessible to GPU kernels (no PCIe transfer)

### 3.2 Token Generation on Metal GPU

**Prefill (Prompt Processing):**
```
Input sequence: [BOS, "Explain", "quantum", "computing", PAD, ..., PAD]  (512 tokens)
Operation: Forward pass with batch matmul through all layers
  - Layer 0: (1, 512, 4096) @ (4096, 4096) → (1, 512, 4096)
  - Layer 1: (1, 512, 4096) @ (4096, 4096) → (1, 512, 4096)
  - ...
  - Layer 31: (1, 512, 4096) @ (4096, 4096) → (1, 512, 4096)

Arithmetic Intensity (prefill):
  AI = (2 × 1 × 512 × 4096 × 4096) / (512 × 4096 × 2 + 4096 × 4096 × 2)
     ≈ 512 FLOPS/byte (compute-bound!)

Throughput: min(600 GFLOPS, 100 GB/s × 512) = 600 GFLOPS
Latency: 2 × 512 × 4096^2 / 600 GFLOPS ≈ 28 ms (prefill)
```

**Decode (Token-by-Token Generation):**
```
Input: Last hidden state (1, 1, 4096)  [one new token, not sequence]
Operation: Single forward pass (batch size 1)
  - Layer 0: (1, 1, 4096) @ (4096, 4096) → (1, 1, 4096)

Arithmetic Intensity (decode):
  AI = (2 × 1 × 1 × 4096 × 4096) / (1 × 4096 × 2 + 4096 × 4096 × 2)
     ≈ 1 FLOPS/byte (memory-bound!)

Roofline: min(600 GFLOPS, 100 GB/s × 1) = 100 GFLOPS
Latency: 2 × 4096^2 / 100 GFLOPS ≈ 336 μs (per token)
         vs 28 ms prefill → token latency dominates

Token generation rate: 1 / 336 μs ≈ 2976 tokens/sec (theoretical)
Practical (50% GPU util + overhead): 1000-1500 tokens/sec
But typical measurement: 7-10 tokens/sec ← why?
```

**Why Decode Latency is Higher (Not Theoretical Peak):**

1. **Attention KV Cache Management:** Each token requires fetching/updating KV cache (256 MB for 4K context, impacts bandwidth)
2. **Sampling + Logits Processing:** argmax/softmax on output probabilities (CPU-bound, 1-2 ms)
3. **Metal Kernel Launch Overhead:** Each layer dispatches separate kernel (0.5 ms × 32 layers = 16 ms)
4. **Memory Pressure:** Model weights + KV cache + activations compete for memory bandwidth

**Realistic Bottleneck Analysis (7B LLaMA, decode):**

```
Per-token latency breakdown:
  32 layers × 0.5 ms (kernel dispatch) = 16 ms
  32 layers × 2 GEMM (attn + FFN) × 5 μs (compute) = 320 μs
  KV cache read: 512 × 4096 × 2 bytes / 100 GB/s = 40 μs
  Output softmax: (4096 + 32 argmax) / 1 GFLOPS = 1 ms
  ────────────────
  Total: ~17 ms per token → 58 tokens/sec (theoretical)

Measured (actual): 7-10 tokens/sec
Gap (~6-8× slower):
  - GPU kernel not reaching peak utilization (memory stalls)
  - CPU/GPU synchronization overhead
  - Model precision (INT4 dequant adds 5-10 ms)
  - Memory pressure from unified memory shared with OS
```

---

## 4. IMPLEMENTATION DEEP DIVE

### 4.1 Building llama.cpp with Metal Backend

```bash
# Clone and build with Metal support
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp

# Build with Metal GPU support (macOS only)
make clean
LLAMA_METAL=1 make -j$(sysctl -n hw.ncpu)

# Test with a GGUF model (download from HuggingFace)
# Example: LLaMA-7B-Q4_K_M.gguf (3.5 GB)

./main -m LLaMA-7B-Q4_K_M.gguf \
    -n 256 \
    -c 2048 \
    -t 8 \
    --gpu-layers 33 \
    -p "Once upon a time"

# Parameters:
#   -m: Model file (GGUF format)
#   -n: Number of tokens to generate
#   -c: Context size (max sequence length for KV cache)
#   -t: Number of threads (CPU backend)
#   --gpu-layers: How many layers to offload to GPU (33 = all for 32-layer model)
#   -p: Prompt text
```

### 4.2 Metal GPU Kernels in llama.cpp

**Key Metal Kernels:**

```cpp
// ggml_metal.m (Metal shading language kernels)

// 1. Quantized Matrix Multiply Kernel
kernel void ggml_metal_kernel_matmul_f32_q4_k(
    device float* src0 [[buffer(0)]],      // A: (m, k) FP32
    device void* src1 [[buffer(1)]],       // B: (k, n) Q4_K_M quantized
    device float* dst [[buffer(2)]],       // C: (m, n) FP32

    constant uint3& ne [[buffer(3)]],      // Shape (m, k, batch)
    constant uint& nb01 [[buffer(4)]],     // Row stride A

    uint2 tgpig [[thread_position_in_grid]],
    uint2 tgpg [[threadgroup_position_in_grid]],
    uint2 tpitg [[thread_position_in_threadgroup]]
) {
    // Each thread computes one output element C[i, j]
    uint i = tgpig.x;  // Row of A
    uint j = tgpig.y;  // Column of B

    if (i >= ne[0] || j >= ne[1]) return;  // Bounds check

    float sum = 0.0f;

    // Inner loop over K dimension
    for (uint k = 0; k < ne[1]; k++) {
        float a_val = src0[i * nb01 + k];  // A[i, k] (FP32)

        // Dequantize B[k, j] from Q4_K_M
        // b_val = dequantize_q4_k(src1, k, j)
        // (Simplified; actual dequant is more complex)

        sum += a_val * b_val;
    }

    dst[i * ne[1] + j] = sum;
}

// 2. Softmax Kernel
kernel void ggml_metal_kernel_soft_max_f32(
    device float* src [[buffer(0)]],
    device float* dst [[buffer(1)]],
    constant uint3& ne [[buffer(2)]],     // Shape (n_rows, n_cols)

    uint thread_idx [[thread_index_in_grid]],
    uint threads_per_threadgroup [[threads_per_threadgroup]],
    threadgroup float* shmem [[threadgroup(0)]]
) {
    uint row = thread_idx / ne[1];  // Which row
    uint col = thread_idx % ne[1];  // Which column

    if (row >= ne[0]) return;

    // Parallel reduction: Find max element in row (for numerical stability)
    float max_val = src[row * ne[1] + col];
    for (uint i = col + threads_per_threadgroup; i < ne[1]; i += threads_per_threadgroup) {
        max_val = max(max_val, src[row * ne[1] + i]);
    }

    // Parallel reduction across SIMD group
    max_val = simd_max(max_val);

    // Compute exp(x - max) and sum
    float exp_sum = 0.0f;
    for (uint i = col; i < ne[1]; i += threads_per_threadgroup) {
        float exp_val = exp(src[row * ne[1] + i] - max_val);
        dst[row * ne[1] + i] = exp_val;
        exp_sum += exp_val;
    }

    // Final normalization
    exp_sum = simd_sum(exp_sum);
    for (uint i = col; i < ne[1]; i += threads_per_threadgroup) {
        dst[row * ne[1] + i] /= exp_sum;
    }
}

// 3. Rope Frequency Embedding (for attention scaling)
kernel void ggml_metal_kernel_rope_f32(
    device float* src [[buffer(0)]],
    device float* dst [[buffer(1)]],
    constant uint3& ne [[buffer(2)]],
    constant float& freq_base [[buffer(3)]],

    uint idx [[thread_index_in_grid]]
) {
    // Rotary Position Embedding (RoPE) application
    // This is critical for positional encoding in transformers

    uint pos = idx / ne[2];    // Token position
    uint dim = idx % ne[2];    // Dimension within hidden state

    float freq = pow(freq_base, float(dim) / float(ne[2]));
    float theta = float(pos) * freq;

    // Apply rotation: [x, y] -> [x cos θ - y sin θ, x sin θ + y cos θ]
    float cos_theta = cos(theta);
    float sin_theta = sin(theta);

    float x = src[idx * 2];
    float y = src[idx * 2 + 1];

    dst[idx * 2] = x * cos_theta - y * sin_theta;
    dst[idx * 2 + 1] = x * sin_theta + y * cos_theta;
}
```

### 4.3 Prompt Processing Optimization (Batch Matmul)

llama.cpp optimizes prompt processing with **chunked prefill**:

```python
# High-level algorithm (Python pseudocode)
def forward_pass_optimized(model, tokens):
    """
    Process tokens in chunks to maximize GPU throughput
    """
    chunk_size = 256  # Process 256 tokens at a time

    # Phase 1: Chunk prefill (process prompt in chunks)
    for chunk in chunks(tokens, chunk_size):
        # Forward pass: (1, chunk_size, hidden_dim) through all layers
        activations = model.forward(chunk)
        # Save attention KV cache for this chunk

    # Phase 2: Token-by-token decode (single token generation)
    # Only last hidden state is needed; reuse KV cache
    for _ in range(num_tokens_to_generate):
        next_token_logits = model.forward(last_hidden_state)
        sampled_token = sample(next_token_logits)
        yield sampled_token
```

**Performance Impact:**

```
Prefill bottleneck (prompt processing):
  - Naive: Process 512-token prompt token-by-token: 512 × 100 ms = 51 sec
  - Chunked: Process 256-token chunks: (512/256) × 15 ms = 30 ms (1700× faster!)

Decode (token generation):
  - Single token: 100 ms
  - But must happen sequentially (autoregressive)
  - For 256 generated tokens: 256 × 100 ms = 25 sec
```

### 4.4 Speculative Decoding Implementation

```cpp
// llama-cpp-speculative.cpp (simplified)

struct SpeculativeSampler {
    llama_context* draft_ctx;     // Small 2B model
    llama_context* target_ctx;    // Full 7B model
    float acceptance_threshold = 0.8;
};

std::vector<llama_token> sample_with_speculation(
    SpeculativeSampler& spec,
    const std::vector<llama_token>& prompt,
    int num_tokens_to_generate
) {
    std::vector<llama_token> generated;

    // Initial prefill with full model
    llama_eval(spec.target_ctx, prompt.data(), prompt.size(), 0);

    for (int i = 0; i < num_tokens_to_generate; ) {
        // Phase 1: Draft with small model (fast)
        std::vector<llama_token> candidates;
        llama_eval(spec.draft_ctx, &generated.back(), 1, 0);

        // Sample K tokens from draft model
        const int K = 5;
        for (int k = 0; k < K; k++) {
            float* draft_logits = llama_get_logits(spec.draft_ctx);
            llama_token draft_token = sample_from_logits(draft_logits);
            candidates.push_back(draft_token);

            llama_eval(spec.draft_ctx, &draft_token, 1, 0);
        }

        // Phase 2: Verify with target model (batch all candidates)
        std::vector<llama_token> target_candidates;
        for (const auto& cand : candidates) {
            llama_eval(spec.target_ctx, &cand, 1, 0);

            float* target_logits = llama_get_logits(spec.target_ctx);
            float* draft_logits = llama_get_logits(spec.draft_ctx);

            // Get probabilities
            float p_target = softmax_prob(target_logits, cand);
            float p_draft = softmax_prob(draft_logits, cand);

            // Acceptance test
            if (p_target > p_draft * acceptance_threshold) {
                generated.push_back(cand);
                target_candidates.push_back(cand);
                i++;
            } else {
                // Rejection sampling
                // Sample from (p_target - p_draft) / (1 - p_draft)
                break;
            }
        }
    }

    return generated;
}

// Results:
// - Draft model: 50 ms per token
// - Target model batch: 60 ms per 5 tokens (if all accepted)
// - Speedup: (5 × 100) / (250 + 60) = 1.6×
```

### 4.5 Benchmark: llama.cpp on M2

```bash
# Benchmark command
./main -m llama-7b-q4_k_m.gguf \
    -n 128 \
    --gpu-layers 33 \
    --temp 0.0 \  # Deterministic sampling for consistent benchmarks
    -p "The meaning of life is" \
    2>&1 | grep -E "prompt eval|eval|total"

# Expected output:
# prompt eval time = 28.43 ms / 32 tokens ( 1125.52 tokens/sec)
# eval time = 93.75 ms / 128 tokens (   1365.33 tokens/sec)
# total time = 7734.89 ms

# Analysis:
# - Prompt eval: 32 tokens in 28 ms → 1125 tokens/sec (GPU batched)
# - Token gen: 128 tokens in 93 ms → 1365 tokens/sec (aggregate, includes overhead)
# - Effective throughput: 128 tokens / 122 ms = 1049 tokens/sec
# - But this is aggregate; true decode latency: 122 ms / 128 = 0.95 ms per token
#   (Includes sampling, KV cache, etc.)

# With no GPU:
# ./main -m llama-7b-q4_k_m.gguf \
#     -n 128 --gpu-layers 0 \
#     ...
# eval time = 5245.36 ms / 128 tokens (   24.39 tokens/sec)  [45× slower!]
```

---

## 5. KEY PAPERS

1. **"Efficient LLM Inference on Apple Silicon" (llama.cpp GitHub, Ggerganov, 2023)**
   - Architecture of Metal backend
   - Quantization strategies and performance tradeoffs
   - Practical benchmarks on M1/M2/M3

2. **"Speculative Decoding with Smaller Language Models" (Leviathan et al., 2023)**
   - Algorithm for 1.5-2× speedup in token generation
   - Acceptance criteria for draft model predictions
   - Statistical guarantees (identical distribution to autoregressive)

3. **"GGUF: A Simple Standard for Large Language Model Checkpoints" (GGML community)**
   - File format specification
   - Quantization schemes (Q3-Q8)
   - Metadata requirements for inference

4. **"Memory-Efficient Attention in Linear Time" (Rabe & Staats, 2021)**
   - Foundational work on attention optimization for inference
   - Attention KV cache compression techniques

5. **"mmap: An Efficient Way to Load Models" (Linux Kernel documentation)**
   - OS-level memory mapping semantics
   - Page fault handling and demand paging
   - Implications for unified memory systems

---

## 6. SYSTEMS TRADEOFFS

### 6.1 Quantization Quality vs Speed

| Scheme | Size (7B) | Prefill (tok/s) | Decode (tok/s) | Accuracy Drop |
|---|---|---|---|---|
| Q8_0 (No quant) | 7 GB | 850 | 8 | <0.2% |
| Q5_K_M | 4.3 GB | 1100 | 9 | 0.5-1% |
| Q4_K_M | 3.5 GB | 1200 | 10 | 1-2% |
| Q3_K | 2.6 GB | 1400 | 12 | 2-3% |

**Recommendation:**
- **Quality-First (research, fine-tuning):** Q8_0
- **Production (accuracy + speed):** Q4_K_M (best tradeoff)
- **Edge devices (limited RAM):** Q3_K if accuracy loss acceptable

### 6.2 GPU vs CPU for Decode

**GPU (Metal Backend):**
- Prefill: 1000+ tokens/sec (batched, compute-bound)
- Decode: 10 tokens/sec (memory-bound, single token)
- Energy: ~10W sustained

**CPU (NEON SIMD):**
- Prefill: 50-100 tokens/sec (single-threaded bottleneck)
- Decode: 2-3 tokens/sec (less efficient for GEMM)
- Energy: ~5W sustained

**Tradeoff:** GPU is 3-5× faster for decode but uses 2× power. For throughput-oriented serving, GPU preferred. For single-user latency-sensitive, marginal.

### 6.3 Chunk Size for Prefill

Larger chunks improve GPU utilization but increase prompt latency:

| Chunk Size | Latency | Throughput | GPU Util |
|---|---|---|---|
| 32 | 3 ms | 10.7 ktok/s | 60% |
| 64 | 6 ms | 10.7 ktok/s | 70% |
| 256 | 20 ms | 12.8 ktok/s | 90% |
| 512 | 40 ms | 12.8 ktok/s | 95% |

**Tradeoff:** Larger chunks (256-512) maximize throughput but increase perceived latency. For streaming output, smaller chunks (64) better for user experience.

---

## 7. EXPERT INSIGHT

### 7.1 Model Size Selection for Apple Silicon

**Question:** "I have 8 GB RAM MacBook Air. Which models fit?"

**Analysis:**

```
Available RAM: 8 GB
OS + System: ~2 GB
Available for model: ~6 GB

Model sizes (GGUF Q4_K_M):
  - 3B model: ~1 GB ✓ (plenty of headroom)
  - 7B model: ~3.5 GB ✓ (can fit, little room for KV cache)
  - 13B model: ~7 GB ✗ (exceeds available memory)
  - 34B model: ~18 GB ✗ (2× memory available)

For 7B on 8 GB:
  - Model weights: 3.5 GB
  - KV cache (4K context): 1 GB (= 4000 tokens × 4096 hidden × 2 × 8 bytes)
  - Runtime activations: 500 MB
  - Total: 5 GB (fits, leaves 3 GB for OS)

Practical issue: Page swapping if model + KV cache exceeds physical RAM
  - Solution: Reduce context size (max_seq_len = 1024 instead of 4096)
  - KV cache: 1000 × 4096 × 2 × 8 / 1024^3 = 62 MB (manageable)
```

**Recommendation:** 3B or small 7B (3.5B parameter variant) for 8 GB MacBook.

### 7.2 KV Cache Memory Pressure

**Problem:** 7B model with 4K context requires 1 GB KV cache (25% of total model memory).

**Solutions:**

1. **Attention KV Cache Compression:**
   - Store KV cache in INT8 (1/4 size)
   - Trade: Small accuracy loss, requires dequant on access
   - Impact: 4K context → 256 MB KV cache (major savings)

2. **Multi-Query Attention (MQA):**
   - Share key/value projections across multiple query heads
   - Reduces KV cache by 8× (if 8 attention heads)
   - Trade: Slight attention quality loss
   - Models: Mistral uses MQA variant (cheaper inference)

3. **Sliding Window Attention:**
   - Only attend to last N tokens (e.g., N=2048)
   - KV cache: 2048 × 4096 × 2 × 8 / 1024^3 = 128 MB
   - Trade: Cannot attend to tokens beyond window

**Typical Setup for M2 (16 GB):**
- 7B model (3.5 GB) + INT8 KV cache (250 MB) + activations (500 MB) = 4.25 GB
- Leaves 12 GB for OS and other processes (comfortable)

---

## 8. BENCHMARKING METHODOLOGY

### 8.1 Profiling llama.cpp with powermetrics

```bash
# Terminal 1: Start powermetrics sampling
sudo powermetrics --samplers cpu_power,gpu_power,core_power -n 60 > power.log &

# Terminal 2: Run inference
./main -m llama-7b-q4_k_m.gguf \
    -n 256 \
    --gpu-layers 33 \
    -p "Explain machine learning in one paragraph"

# Wait for completion (~30 seconds)

# Analyze results
grep "GPU Power:" power.log | awk '{sum += $NF; count++} END {print "Avg GPU Power: " sum/count " W"}'
grep "CPU Power:" power.log | awk '{sum += $NF; count++} END {print "Avg CPU Power: " sum/count " W"}'

# Typical output:
# Avg GPU Power: 10.5 W
# Avg CPU Power: 3.2 W
# Total: 13.7 W (vs 25W+ for discrete GPU)
```

### 8.2 Latency Analysis via Timestamps

```bash
# Modify llama.cpp to add detailed timing
# In main.cpp, add:

#include <chrono>

auto start_prompt = std::chrono::high_resolution_clock::now();
llama_eval(ctx, prompt.data(), prompt.size(), 0);
auto end_prompt = std::chrono::high_resolution_clock::now();
auto prompt_time = std::chrono::duration<double>(end_prompt - start_prompt).count();

for (int i = 0; i < num_tokens; i++) {
    auto start_token = std::chrono::high_resolution_clock::now();
    // Sample one token
    auto end_token = std::chrono::high_resolution_clock::now();
    auto token_time = std::chrono::duration<double>(end_token - start_token).count();

    printf("Token %d: %.2f ms\n", i, token_time * 1000);
}

// Output:
// Prompt eval: 28.45 ms (32 tokens → 1125 tokens/sec aggregate)
// Token 0: 93.2 ms
// Token 1: 92.8 ms
// Token 2: 93.1 ms
// ...
// Average token latency: 93 ms → 10.7 tokens/sec
```

### 8.3 Comparing Models (Accuracy vs Latency Tradeoff)

```python
import subprocess
import json
import time

models = [
    ("llama-3b-q4_k_m.gguf", "3B"),
    ("llama-7b-q4_k_m.gguf", "7B"),
    ("mistral-7b-q4_k_m.gguf", "7B Mistral"),
]

prompt = "The capital of France is"
expected_output = "Paris"

results = []

for model_file, model_name in models:
    start = time.time()
    output = subprocess.run(
        ["./main", "-m", model_file, "-n", "10", "-p", prompt],
        capture_output=True,
        text=True,
        timeout=30
    ).stdout
    elapsed = time.time() - start

    # Extract latency from output
    for line in output.split('\n'):
        if 'eval time' in line:
            tokens = int(line.split()[4])
            latency = float(line.split()[3])
            tok_per_sec = tokens / (latency / 1000)
            break

    # Check if output contains expected token
    accuracy = expected_output in output

    results.append({
        'model': model_name,
        'latency_ms': latency,
        'throughput': tok_per_sec,
        'accuracy': accuracy,
        'total_time_sec': elapsed
    })

print("Model Comparison:")
for r in results:
    print(f"{r['model']:20} {r['latency_ms']:8.1f} ms  {r['throughput']:6.1f} tok/s  Accuracy: {r['accuracy']}")
```

---

## 9. OPEN PROBLEMS

1. **Dynamic Batch Dispatch:** llama.cpp processes tokens sequentially. Supporting dynamic batching (multiple independent requests) would require GPU kernel modifications to handle ragged tensor shapes. Challenge: Maintaining latency guarantees while improving throughput.

2. **INT4 Arithmetic on Metal:** Current INT4 quantization requires dequantization to FP32/FP16 before compute. True INT4 arithmetic (like NVIDIA's in H100) would reduce bandwidth by 4×. Challenge: Metal doesn't expose sub-byte operations (unlike NVIDIA's IMMA units).

3. **Attention KV Cache Optimization:** KV cache grows with context size (O(N) memory). Techniques like KV cache compression (INT8 or sparse) are implemented in some models but not universally supported. Challenge: Maintaining accuracy while compressing by 4-8×.

4. **Model-Specific Optimization:** llama.cpp is generic; each model variant (LLaMA, Mistral, Phi, Qwen) has slightly different architectures. Hand-tuned kernels per architecture would improve performance by 20-30%. Challenge: Maintenance burden.

5. **Inference Serving with Multiple Models:** Switching between models (e.g., multilingual LLMs) incurs model load overhead. Current mmap approach requires loading entire model. Solution: Shared weight caching across models with same architecture.

---

## 10. PHD QUALIFIER QUESTIONS

**Q1 (Quantization):** "Explain Q4_K_M quantization in detail. For a 4096×4096 weight matrix, calculate the compression ratio and number of K-means clusters. How does the accuracy loss depend on the number of clusters?"

**A1 Outline:**
- Q4_K_M: 4-bit quantization with K-means clustering per block
- Block size: 32×32 (standard)
- K-means clusters: 16 (2^4 = 16 possible 4-bit values)
- Original size: 32 × 32 × 2 bytes = 2048 bytes per block
- Compressed: 16 centroids × 2 bytes + 32×32 × 4 bits = 32 + 512 = 544 bytes per block
- Compression: 2048 / 544 = 3.76× per block
- Total (4096×4096): 4M blocks × 544 bytes = 2.1 GB (vs 32 GB FP32)
- Compression ratio: 15.2× (very aggressive)

- Accuracy loss:
  - Each weight approximated by nearest centroid
  - Error per weight: σ = (max_weight - min_weight) / sqrt(16) ≈ range / 4
  - Relative error: σ / mean(|W|) ≈ 1-2% for typical models
  - Task-level accuracy: 1-2% drop in perplexity (acceptable)

- With K=256 (8 bits, no quantization):
  - Compression: Only 256× instead of 3.76×
  - Accuracy: 0% loss (but defeats quantization purpose)

**Q2 (Systems Design):** "Design an optimized llama.cpp inference system for a 16-core M3 Max with 96 GB RAM, supporting concurrent requests from 10 users. Each user generates 128 tokens with 2K token context. Propose KV cache allocation, batch processing strategy, and expected throughput."

**A2 Outline:**
- Hardware: M3 Max (dual GPU, 200 GB/s memory)
- Model: 7B Q4_K_M (3.5 GB)
- Per-user requirements:
  - Model: 3.5 GB (shared)
  - KV cache (2K context): 2000 × 4096 × 2 × 8 bytes = 128 MB per user
  - Runtime activations: 256 MB per user

- Memory allocation:
  - Model: 3.5 GB (once)
  - 10 users × 128 MB KV cache = 1.28 GB
  - 10 users × 256 MB activations = 2.56 GB
  - Total: 7.34 GB (plenty of headroom in 96 GB)

- Batch processing strategy:
  - Prefill (batch): Process all 10 prompts together in single batch
    - Input shape: (10, 1024) [1K token context per user]
    - Latency: 1024 tokens / 1200 tok/s = 0.85 sec
  - Decode (sequential): Generate tokens for each user in round-robin
    - User 0 token 1: 100 ms
    - User 1 token 1: 100 ms
    - ... (10 users × 100 ms = 1 sec per round)
    - Rounds needed: 128 tokens
    - Total decode: 128 rounds × 1 sec = 128 sec

- Expected throughput:
  - Total time: 0.85 + 128 = 128.85 sec
  - Tokens/sec: (10 users × 128 tokens) / 128.85 = 9.9 tok/sec aggregate
  - Per-user: 1.0 tok/sec (expected from M3 Max specs)

- Optimization: Batch decode with speculative decoding
  - If draft model (2B) can generate 5-token drafts in 200 ms
  - Verify with 7B in batch: 200 + 60 = 260 ms per 5 tokens = 1.6× speedup
  - New throughput: 15 tokens/sec aggregate

**Q3 (Quantization Trade-offs):** "A model achieves 8.5 tokens/sec on M2 with Q4_K_M (3.5 GB). If quantized to Q3_K (2.6 GB), latency increases to 11 tokens/sec due to extra dequantization overhead. Explain this paradox and propose a solution."

**A3 Outline:**
- Paradox: Q3_K is smaller (2.6 GB), so why slower (11 tokens/sec vs 8.5)?

- Root cause: Dequantization overhead
  - Q4_K_M: Direct Metal matmul on (FP32 activations @ Q4 weights)
  - Q3_K: Extra dequantization step (3-bit → FP16) → then matmul
  - Dequant kernel: 2.6 GB × 2 (FP16 output) = 5.2 GB data
  - Bandwidth: 5.2 GB / (1/100 GB/s) = 52 ms per layer × 32 = 1664 ms per token (vs 118 ms Q4_K_M)

- Counter-intuitive: Smaller model slower due to extra operation

- Solutions:
  1. **Fused Dequant+GEMM Kernel:** Avoid writing intermediate FP16 weights
     - Keep Q3 weights in registers, expand on-the-fly during GEMM
     - Saves bandwidth: 5.2 GB → 2.6 GB
     - Expected: 11 tok/sec → 9 tok/sec (closer to Q4_K_M)

  2. **Bit-Serial GEMM:** Compute dot products in 3-bit fixed-point
     - Very complex, low adoption
     - Potential: 11 tok/sec → 12+ tok/sec (if implemented well)

  3. **Hybrid Q3/Q4:** Use Q4_K_M for critical layers, Q3_K for others
     - Q4: attention projections (32 layers) → 3.5 GB
     - Q3: FFN layers (32 layers) → 0.8 GB
     - Total: 4.3 GB
     - Speed: ~10 tok/sec (sweet spot)

**Q4 (Production System):** "Design a llama.cpp-based inference server for a MacBook Pro cluster (4 M3 Max machines). Support 50 concurrent users, each with 10-minute sessions generating text. Include model loading strategy, fault tolerance, and expected latency distribution."

**A4 Outline:**
- Cluster: 4 × M3 Max = 4 × 200 GB/s = 800 GB/s aggregate
- Model: 7B Q4_K_M (3.5 GB)
- Sessions: 50 users × 128 tokens (assumption) = 6400 tokens total

- Architecture:
  ```
  Load Balancer (distribute users to machines)
    ↓
  4 llama.cpp instances (one per M3 Max)
    ↓
  Shared Model Storage (NFS or local SSD)
  ```

- Model loading:
  - First user on each machine: Load 3.5 GB model
  - Time: ~5 sec (mmap from local SSD)
  - Subsequent users: Already in RAM (instant)
  - Optimization: Pre-load model at startup, avoid repeated loads

- Fault tolerance:
  - If one machine fails: Rebalance users to remaining 3 machines
  - User latency increase: From 128 sec to 170 sec (1.33× degradation)
  - No user disconnection (session-aware load balancing)

- Expected latency distribution:
  - Median: 1.0 tok/sec (expected from M3 Max)
  - P95: 0.8 tok/sec (under load, some contention)
  - P99: 0.5 tok/sec (heavy contention, multiple users same machine)

- QoS guarantee:
  - SLA: 99% of tokens < 1 sec latency
  - Meets requirement only if batch processing optimized

---

## CONCLUSION

llama.cpp demonstrates that inference-optimized systems can achieve remarkable efficiency on consumer hardware through quantization, memory mapping, and targeted GPU acceleration. The combination of GGUF quantization, Metal kernels, and mmap-based loading represents best practices for on-device LLM serving. Open problems center on expanding operator coverage (INT4 arithmetic, sparse attention), supporting dynamic shapes, and scaling to multiple devices. Next module synthesizes all prior knowledge into full-stack inference system design, balancing accuracy, latency, and energy across all available hardware accelerators.

