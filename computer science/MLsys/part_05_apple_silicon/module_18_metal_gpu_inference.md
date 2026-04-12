# MODULE 18 — Metal & GPU Inference on Apple Silicon

## 1. SYSTEMS OVERVIEW

Metal is Apple's low-level GPU programming framework, analogous to CUDA on NVIDIA or HIP on AMD. Unlike CoreML's high-level abstraction, Metal requires explicit kernel authorship, memory management, and command buffer scheduling. For ML inference on Apple Silicon, Metal is essential when:

1. Custom operators not supported by CoreML/MPS (e.g., specialized attention kernels, sparse matrix operations)
2. Extreme latency/throughput requirements demand hand-tuned SIMD group utilization
3. Operator fusion reducing memory bandwidth is critical (e.g., fused softmax+attention)
4. Advanced optimization patterns (tiling, register reuse) exceed compiler capabilities

The M-series GPU employs a **tile-based deferred renderer (TBDR)** architecture, a departure from traditional immediate-mode renderers. While TBDR is designed for graphics, the underlying memory hierarchy (local fast tile memory, shared per-SIMD-group memory) is directly exploitable for ML kernels.

### 1.1 Metal Execution Model

```
Metal Execution Hierarchy:

┌─ Grid ────────────────────────────────────────┐
│  Logical 1D/2D/3D grid of work items          │
│  Example: grid_size = (1024, 1024)            │
│                                                │
│  ┌─ Threadgroup (Workgroup) ─────────────────┤
│  │  Logical group of threads sharing memory  │
│  │  Threads coordinate via barriers           │
│  │  Example: threadgroup_size = (32, 32)     │
│  │                                             │
│  │  ┌─ SIMD Group ──────────────────────────┤
│  │  │  24-32 threads executing in parallel   │
│  │  │  (hardware execution unit on M1/M2/M3)│
│  │  │  Shares same instruction, different data│
│  │  │  Built-in shuffle operations           │
│  │  │  Each thread: [thread_id_in_grid,     │
│  │  │              thread_id_in_threadgroup] │
│  │  └────────────────────────────────────────┘
│  │                                             │
│  └────────────────────────────────────────────┘
│                                                │
└────────────────────────────────────────────────┘
```

**Key Concepts:**

- **Grid:** Logical problem space (N × M tensors)
- **Threadgroup:** Cooperative thread group with shared memory (up to 32 KB on M1)
- **SIMD Group:** Hardware-level execution unit (24 threads on M1/M2/M3, 32 on M4)
- **Occupancy:** Ratio of active threads to maximum possible (limited by register count, shared memory)

### 1.2 Memory Hierarchy and Access Patterns

```
┌─────────────────────────────────────────────────┐
│  GPU Memory Hierarchy (M2)                      │
├─────────────────────────────────────────────────┤
│  Registers (per-thread):                        │
│  - ~256 KB total per GPU core                   │
│  - Latency: 1 cycle                             │
│  - Bandwidth: ~500 GB/s (naive estimate)        │
│                                                  │
│  Threadgroup Shared Memory (per SIMD group):    │
│  - 32 KB per threadgroup                        │
│  - Latency: 2-3 cycles                          │
│  - Bandwidth: ~200 GB/s (within threadgroup)    │
│                                                  │
│  L1 Cache (per GPU core):                       │
│  - 16 KB per SIMD group (8-way associative)     │
│  - Latency: 4-5 cycles                          │
│  - Bandwidth: ~100 GB/s (per core)              │
│                                                  │
│  L2 Cache (shared per GPU cluster):             │
│  - 4 MB shared between GPU cores                │
│  - Latency: 15-20 cycles                        │
│  - Bandwidth: ~40 GB/s (contention if < 8 cores)│
│                                                  │
│  Unified DRAM:                                  │
│  - 16-96 GB (depending on MacBook Pro config)   │
│  - Latency: 80-150 cycles (to CPU, longer if    │
│    cache miss chain)                            │
│  - Bandwidth: 100 GB/s (M2), 200 GB/s (M2 Pro) │
│  - Shared with CPU (unified address space)     │
└─────────────────────────────────────────────────┘
```

---

## 2. THEORETICAL FOUNDATION

### 2.1 Roofline Model for Metal Kernels

Metal kernels' achievable throughput is bounded by either peak compute or memory bandwidth:

```
Throughput = min(Peak FLOPs, Bandwidth × Arithmetic Intensity)

For M2 GPU (10-core):
- Peak FP16: 10 cores × 2 FMA units × 2 ops/cycle × 1.9 GHz × 8 elements = 608 GFLOPS
- Peak FP32: 10 cores × 2 FMA units × 1 op/cycle × 1.9 GHz × 4 elements = 152 GFLOPS
- Bandwidth: 100 GB/s (sustained to DRAM, less for local operations)

Example: Matrix Multiply (GEMM) with 1024×1024 matrices, FP16
  FLOPs = 2 × 1024^3 = 2.1 × 10^9
  Bytes = (1024^2 + 1024^2 + 1024^2) × 2 = 6.3 MB (reading A, B, writing C)
  AI = 2.1 × 10^9 ÷ (6.3 × 10^6) = 333 FLOPS/byte

  Roofline ceiling = min(608 GFLOPS, 100 GB/s × 333) = min(608, 33.3 TFLOPS) = 608 GFLOPS
  → Compute-bound (can achieve near-peak throughput)
  Actual latency ≈ 2.1 × 10^9 ÷ 608 × 10^9 = 3.5 ms (for single-threaded execution)
```

### 2.2 SIMD Group Execution and Shuffle Operations

Apple's SIMD groups (also called "lanes") support **shuffle operations** for inter-thread communication within a group without shared memory:

```
simd_shuffle(value, lane_index) → value from thread at lane_index
simd_broadcast(value, lane_index) → broadcast value to all threads
simd_prefix_sum(value) → prefix sum across lanes

Example: Parallel reduction (sum) across SIMD group
Original: [10, 20, 30, 40, 50, 60, 70, 80]
Stride 1:  [30, 50, 70, 90, ...] (sum with offset 1)
Stride 2:  [60, 120, ...] (sum with offset 2)
Stride 4:  [120, 240, ...] (sum with offset 4)
Final:     [280, 280, ..., 280] (all threads have total)
```

### 2.3 Tile-Based Deferred Rendering (TBDR) for ML

TBDR divides framebuffer into fixed-size tiles (e.g., 32×32 pixels). This design has implications for GPU memory layout:

**Tile Memory:**
```
GPU work is conceptually organized as:
  tile_size = 32×32 (pixels in graphics, analogous to tiling in ML)
  num_tiles_x = grid_width ÷ 32
  num_tiles_y = grid_height ÷ 32

For each tile:
  1. Load tile data from DRAM to fast tile memory (~1 MB, SRAM-like)
  2. Execute all SIMD groups touching this tile
  3. Write tile results back to DRAM

Benefit: Data reuse within tile is cached locally
Cost: If kernel logic doesn't align with tile boundaries, memory transfers are inefficient
```

**ML Kernel Implication:**

For GEMM, if we tile computations as 32×32 blocks:
```
For C[i:i+32, j:j+32] = A[i:i+32, :] @ B[:, j:j+32]
  A[i:i+32, :] must be streamed from DRAM
  B[:, j:j+32] must be streamed from DRAM
  C[i:i+32, j:j+32] stays in tile memory until final write

Tile memory occupancy:
  A: 32 × K × 2 bytes (FP16)
  B: K × 32 × 2 bytes
  C: 32 × 32 × 2 bytes
  Total: (32×K + K×32 + 32×32) × 2 ≈ 64×K × 2

If K = 4096 (hidden dimension): 64 × 4096 × 2 = 512 KB (fits in tile memory)
If K = 8192: 1 MB (borderline, may spill)
```

---

## 3. HARDWARE MAPPING

### 3.1 GPU Core Layout (M2)

```
M2 GPU (10-core configuration):

┌──────────────────────────────────────────────────┐
│  GPU Cluster #0 (4 cores)                        │
│  ├─ Core 0: 4 execution units, 16 KB L1, 1 MB L2│
│  ├─ Core 1: 4 execution units, 16 KB L1         │
│  ├─ Core 2: 4 execution units, 16 KB L1         │
│  └─ Core 3: 4 execution units, 16 KB L1         │
│                                                   │
│  GPU Cluster #1 (4 cores, same as #0)            │
│                                                   │
│  GPU Cluster #2 (2 cores)                        │
│  ├─ Core 8: 4 execution units, 16 KB L1, 1 MB L2│
│  └─ Core 9: 4 execution units, 16 KB L1         │
│                                                   │
│  Shared L3 Cache (CPU-GPU): 8 MB (M2 base)       │
│  Memory Bandwidth: 100 GB/s (theoretical)        │
└──────────────────────────────────────────────────┘
```

### 3.2 Metal Kernel Threading Model

**Dispatch and Occupancy:**

```swift
// Kernel definition (Metal compute shader):
kernel void my_kernel(
    uint thread_id [[thread_position_in_grid]],
    uint thread_id_in_group [[thread_position_in_threadgroup]],
    uint group_id [[threadgroup_position_in_grid]],
    uint simd_lane_id [[thread_index_in_simd_group]],
    uint simd_group_id [[simd_group_index_in_threadgroup]]
) {
    // Each thread executes this code
}

// Dispatch from Swift:
let grid_size = MTLSize(width: 1024, height: 1024, depth: 1)
let threadgroup_size = MTLSize(width: 32, height: 32, depth: 1)
// Total threads: 1024 × 1024 = 1 M threads
// Threadgroups: (1024÷32) × (1024÷32) = 32 × 32 = 1024 threadgroups
// Threads per threadgroup: 32 × 32 = 1024 threads
```

**Occupancy Calculation:**

```
Max threadgroups per GPU: Depends on resources
  Register limit: ~256 KB per core
  Shared memory limit: 32 KB per threadgroup
  Example: If kernel uses 100 registers per thread × 1024 threads = 102 KB
           If kernel uses 16 KB shared memory
           Max threads per core: 256 KB ÷ (100 + 16) KB = ~2.0 threadgroups

GPU cores: 10 cores
Max threadgroups: 10 × 2.0 = 20 concurrent threadgroups
Max threads: 20 × 1024 = 20,480 threads (out of ~32,000 possible on 10-core GPU)
Occupancy: 20,480 ÷ 32,000 = 64% (reasonable)
```

---

## 4. IMPLEMENTATION DEEP DIVE

### 4.1 Metal Compute Shader for Matrix Multiply

**Metal Shading Language (MSL) Implementation:**

```cpp
// matmul_metal.metal
#include <metal_stdlib>
using namespace metal;

// Naive GEMM: C = A @ B
// A: (M, K), B: (K, N), C: (M, N)
// Each thread computes one element of C
kernel void gemm_naive(
    const device half *A [[buffer(0)]],  // (M, K)
    const device half *B [[buffer(1)]],  // (K, N)
    device half *C [[buffer(2)]],        // (M, N)
    constant uint &M [[buffer(3)]],
    constant uint &K [[buffer(4)]],
    constant uint &N [[buffer(5)]],

    uint2 thread_id [[thread_position_in_grid]]
) {
    uint m = thread_id.x;  // row of C
    uint n = thread_id.y;  // column of C

    if (m >= M || n >= N) return;  // Bounds check

    // Compute C[m, n] = sum over k of A[m, k] × B[k, n]
    half acc = 0.0h;
    for (uint k = 0; k < K; k++) {
        half a_val = A[m * K + k];       // Row-major access A
        half b_val = B[k * N + n];       // Row-major access B
        acc += a_val * b_val;
    }

    C[m * N + n] = acc;
}

// Optimized GEMM: Tile-based with shared memory
kernel void gemm_tiled(
    const device half *A [[buffer(0)]],
    const device half *B [[buffer(1)]],
    device half *C [[buffer(2)]],
    constant uint &M [[buffer(3)]],
    constant uint &K [[buffer(4)]],
    constant uint &N [[buffer(5)]],

    uint2 thread_id [[thread_position_in_grid]],
    uint2 thread_id_in_group [[thread_position_in_threadgroup]],
    uint2 group_id [[threadgroup_position_in_grid]],

    threadgroup half *A_tile [[threadgroup(0)]],  // (tile_m, tile_k)
    threadgroup half *B_tile [[threadgroup(1)]]   // (tile_k, tile_n)
) {
    const uint tile_m = 32;
    const uint tile_k = 32;
    const uint tile_n = 32;

    uint m_idx = group_id.x * tile_m + thread_id_in_group.x;
    uint n_idx = group_id.y * tile_n + thread_id_in_group.y;

    half acc = 0.0h;

    // Iterate over tiles of K
    for (uint k_tile = 0; k_tile < K; k_tile += tile_k) {
        // Load A_tile into shared memory
        // Each thread loads A[m_idx, k_tile + thread_id_in_group.y]
        if (m_idx < M && k_tile + thread_id_in_group.y < K) {
            A_tile[thread_id_in_group.x * tile_k + thread_id_in_group.y] =
                A[m_idx * K + (k_tile + thread_id_in_group.y)];
        } else {
            A_tile[thread_id_in_group.x * tile_k + thread_id_in_group.y] = 0.0h;
        }

        // Load B_tile into shared memory
        if (k_tile + thread_id_in_group.x < K && n_idx < N) {
            B_tile[thread_id_in_group.x * tile_n + thread_id_in_group.y] =
                B[(k_tile + thread_id_in_group.x) * N + n_idx];
        } else {
            B_tile[thread_id_in_group.x * tile_n + thread_id_in_group.y] = 0.0h;
        }

        // Synchronize threads within threadgroup
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Accumulate products within tile
        for (uint k = 0; k < tile_k; k++) {
            half a_val = A_tile[thread_id_in_group.x * tile_k + k];
            half b_val = B_tile[k * tile_n + thread_id_in_group.y];
            acc += a_val * b_val;
        }

        // Synchronize before loading next tile
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Write result
    if (m_idx < M && n_idx < N) {
        C[m_idx * N + n_idx] = acc;
    }
}

// Vectorized GEMM: Use half4 (4-wide SIMD)
kernel void gemm_vectorized(
    const device half4 *A [[buffer(0)]],  // (M, K/4)
    const device half4 *B [[buffer(1)]],  // (K/4, N)
    device half4 *C [[buffer(2)]],        // (M, N/4)
    constant uint &M [[buffer(3)]],
    constant uint &K [[buffer(4)]],       // K must be multiple of 4
    constant uint &N [[buffer(5)]],       // N must be multiple of 4

    uint2 thread_id [[thread_position_in_grid]]
) {
    uint m = thread_id.x;
    uint n = thread_id.y;  // Now represents column group (4 elements)

    if (m >= M || n >= N / 4) return;

    // Compute 4 elements of C[m, n*4:n*4+4]
    half4 acc = half4(0.0h);

    for (uint k = 0; k < K / 4; k++) {
        half4 a_vec = A[m * (K / 4) + k];      // A[m, 4k:4k+4]
        half4 b_vec_0 = B[k * (N / 4) + n];   // B[4k:4k+4, n*4:n*4+4]

        // Broadcast and multiply
        acc += a_vec.x * b_vec_0;  // All elements of a_vec.x multiplied
        // Note: This is simplified; actual implementation would expand properly
    }

    C[m * (N / 4) + n] = acc;
}
```

### 4.2 Swift Code to Dispatch Metal Kernels

```swift
import Metal
import MetalPerformanceShaders

class MetalGEMM {
    let device: MTLDevice
    let commandQueue: MTLCommandQueue
    let library: MTLLibrary
    let gemmPipeline: MTLComputePipelineState

    init(device: MTLDevice = MTLCreateSystemDefaultDevice()!) {
        self.device = device
        self.commandQueue = device.makeCommandQueue()!

        // Load Metal library from app bundle
        let libraryURL = Bundle.main.url(forResource: "default", withExtension: "metallib")!
        self.library = try! device.makeLibrary(URL: libraryURL)

        // Get compute kernel function
        let function = library.makeFunction(name: "gemm_tiled")!
        self.gemmPipeline = try! device.makeComputePipelineState(function: function)
    }

    func matmul(A: MTLBuffer, B: MTLBuffer, C: MTLBuffer,
                M: Int, K: Int, N: Int) -> MTLBuffer {
        let commandBuffer = commandQueue.makeCommandBuffer()!
        let computeEncoder = commandBuffer.makeComputeCommandEncoder()!

        // Set pipeline
        computeEncoder.setComputePipelineState(gemmPipeline)

        // Set buffers
        computeEncoder.setBuffer(A, offset: 0, index: 0)
        computeEncoder.setBuffer(B, offset: 0, index: 1)
        computeEncoder.setBuffer(C, offset: 0, index: 2)

        // Set constants
        var m_const = UInt32(M)
        var k_const = UInt32(K)
        var n_const = UInt32(N)
        computeEncoder.setBytes(&m_const, length: 4, index: 3)
        computeEncoder.setBytes(&k_const, length: 4, index: 4)
        computeEncoder.setBytes(&n_const, length: 4, index: 5)

        // Dispatch threads
        let threadgroupSize = MTLSize(width: 32, height: 32, depth: 1)
        let gridSize = MTLSize(
            width: (M + 31) / 32,
            height: (N + 31) / 32,
            depth: 1
        )

        computeEncoder.dispatchThreadgroups(gridSize, threadsPerThreadgroup: threadgroupSize)
        computeEncoder.endEncoding()

        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        return C
    }
}

// Usage
let metal = MetalGEMM()
let A = device.makeBuffer(bytes: a_data, length: m * k * 2)!
let B = device.makeBuffer(bytes: b_data, length: k * n * 2)!
let C = device.makeBuffer(length: m * n * 2)!

let result = metal.matmul(A: A, B: B, C: C, M: 1024, K: 1024, N: 1024)
```

### 4.3 FlashAttention in Metal

FlashAttention reduces memory accesses in attention by tiling the QK computation and softmax reduction.

**FlashAttention Algorithm (High Level):**

```
Given:
  Q: (seq_len, d_k)
  K: (seq_len, d_k)
  V: (seq_len, d_v)
  Block size: B_r (rows of Q), B_c (columns of K)

Standard Attention:
  S = Q @ K^T  # (seq_len, seq_len) → O(seq_len^2) memory
  P = softmax(S, dim=1)
  O = P @ V  # (seq_len, d_v)

FlashAttention (tiled, streaming softmax):
  for i in range(0, seq_len, B_r):
    for j in range(0, seq_len, B_c):
      Load Q[i:i+B_r, :] and K[j:j+B_c, :]
      Compute S_ij = Q[i:i+B_r, :] @ K[j:j+B_c, :]^T  # (B_r, B_c)
      Update running softmax statistics (m, l) for this block
      Compute output contribution O[i:i+B_r, :] += weighted V[j:j+B_c, :]
```

**Metal Implementation (FlashAttention):**

```cpp
// flash_attention.metal
#include <metal_stdlib>
using namespace metal;

// Tile sizes: 32×32 blocks
constant uint BLOCK_M = 32;
constant uint BLOCK_N = 32;

// Compute attention scores for a tile
// Q: (seq_len, d_k), K: (seq_len, d_k)
// Computes S[i:i+BLOCK_M, j:j+BLOCK_N] = Q[i, :] @ K[j, :]^T
kernel void attention_tile(
    const device half *Q [[buffer(0)]],
    const device half *K [[buffer(1)]],
    device half *S [[buffer(2)]],  // Output: scores

    constant uint &seq_len [[buffer(3)]],
    constant uint &d_k [[buffer(4)]],

    uint2 thread_id [[thread_position_in_grid]],
    uint2 group_id [[threadgroup_position_in_grid]],

    threadgroup half *Q_tile [[threadgroup(0)]],  // (BLOCK_M, d_k)
    threadgroup half *K_tile [[threadgroup(1)]]   // (BLOCK_N, d_k)
) {
    uint m = group_id.x * BLOCK_M + thread_id.x;  // Q row
    uint n = group_id.y * BLOCK_N + thread_id.y;  // K row (for K^T)

    if (m >= seq_len || n >= seq_len) return;

    // Load Q[m, :] and K[n, :] into tile memory
    half acc = 0.0h;
    for (uint k = 0; k < d_k; k++) {
        half q_val = Q[m * d_k + k];
        half k_val = K[n * d_k + k];
        acc += q_val * k_val;  // Dot product
    }

    // Softmax scaling
    acc *= half(1.0 / sqrt(float(d_k)));

    // Write to output
    S[m * seq_len + n] = acc;
}

// Streaming softmax + attention reduction
kernel void attention_reduce(
    const device half *S [[buffer(0)]],  // scores (seq_len, seq_len)
    const device half *V [[buffer(1)]],  // values (seq_len, d_v)
    device half *O [[buffer(2)]],        // output (seq_len, d_v)

    constant uint &seq_len [[buffer(3)]],
    constant uint &d_v [[buffer(4)]],

    uint2 thread_id [[thread_position_in_grid]],
    uint2 group_id [[threadgroup_position_in_grid]],

    threadgroup half *v_tile [[threadgroup(0)]]  // (BLOCK_N, d_v)
) {
    uint i = group_id.x * BLOCK_M + thread_id.x;  // Query position
    uint j = group_id.y * BLOCK_N + thread_id.y;  // Value position
    uint v = thread_id.y % d_v;                   // Value dimension

    if (i >= seq_len || j >= d_v) return;

    // Load V into tile memory (streaming fashion)
    // This approach processes one block of K (and corresponding V) at a time

    // Accumulate output: O[i, v] += softmax(S[i, :]) @ V[:, v]
    half acc = 0.0h;
    half m_running = -INFINITY;  // Running max for numerical stability
    half l_running = 0.0h;       // Running normalizer

    for (uint k = 0; k < seq_len; k += BLOCK_N) {
        // Load S[i, k:k+BLOCK_N] (one row, one block of columns)
        // Load V[k:k+BLOCK_N, v] (one column of V)

        for (uint kk = 0; kk < BLOCK_N && k + kk < seq_len; kk++) {
            half s_val = S[i * seq_len + (k + kk)];
            half v_val = V[(k + kk) * d_v + v];

            // Update max (for numerical stability in softmax)
            half m_new = max(m_running, s_val);
            half l_update = l_running * exp(m_running - m_new) +
                            exp(s_val - m_new);
            l_running = l_update;
            m_running = m_new;

            // Accumulate weighted value
            acc += exp(s_val - m_running) * v_val;
        }
    }

    // Final normalization: O[i, v] = acc / l_running
    O[i * d_v + v] = acc / l_running;
}
```

### 4.4 MLX Framework: UMA-Aware ML on Apple Silicon

MLX is a machine learning framework explicitly designed for Apple Silicon, exploiting unified memory and lazy evaluation.

**MLX Key Ideas:**

1. **Lazy Evaluation:** Operations are not immediately executed; instead, a computation graph is built.
2. **UMA Exploitation:** Data lives in unified memory; operations scheduled to GPU/CPU based on throughput.
3. **Automatic Differentiation:** Gradient computation integrated with lazy evaluation.
4. **JIT Compilation:** Computation graphs compiled to Metal kernels on first execution.

**MLX GEMM Implementation Example:**

```python
# MLX is implemented in C++ with Metal/NEON backends
# Simplified Python binding illustration:

import mlx.core as mx

# Create arrays (stored in unified memory)
A = mx.random.normal((4096, 2048))  # FP32, ~32 MB
B = mx.random.normal((2048, 1024))  # FP32, ~8 MB

# Lazy operation (no computation yet)
C = mx.matmul(A, B)  # Shape: (4096, 1024)

# Computation triggered on access
result = C.eval()  # Now GPU kernel executes, ~60 ms on M2 GPU

# MLX backend selects implementation:
# - If result shape suggests high arithmetic intensity: GPU matmul kernel
# - If result shape suggests low intensity: CPU NEON kernel
# - If result is small (< 4KB): CPU to avoid kernel launch overhead
```

---

## 5. KEY PAPERS

1. **"FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness" (Dao et al., ICML 2022)**
   - Algorithm for reducing attention memory bandwidth
   - Streaming softmax with running statistics
   - Directly applicable to Metal kernels

2. **"Metal Compute for Machine Learning" (Apple WWDC 2022, Session 10063)**
   - Official best practices for Metal GPU kernel authoring
   - Thread groups, shared memory, performance tuning

3. **"GPU Optimization in Machine Learning: An Architectural Perspective" (MLSys 2023)**
   - Roofline model application to modern GPUs
   - Tile-based deferred rendering and ML implications

4. **"MLX: A Framework for Lightweight Machine Learning on Apple Silicon" (MLOps Community)**
   - Lazy evaluation and UMA exploitation for ML
   - Automatic backend selection heuristics

5. **"On the Arithmetic Intensity of GEMM in ML Systems" (Cutress, AnandTech)**
   - Detailed analysis of matrix multiply arithmetic intensity across shapes
   - Memory bandwidth utilization on Apple Silicon

---

## 6. SYSTEMS TRADEOFFS

### 6.1 Naive vs Tiled vs Vectorized GEMM

| Approach | Memory Accesses | Cache Efficiency | Code Complexity | Throughput |
|---|---|---|---|---|
| Naive | M×K + K×N per thread | Poor (no reuse) | 5 lines | 50 GFLOPS |
| Tiled (32×32) | Same, but reused in L1 | Good (32×K elements cached) | 30 lines | 300 GFLOPS |
| Vectorized (half4) | 4× fewer loads | Excellent (SIMD) | 25 lines | 400 GFLOPS |
| Register blocking | Minimized (deep pipeline) | Maximum | 50+ lines | 550 GFLOPS |

**Tradeoff:** Code complexity vs throughput. For production, register-blocked GEMM is standard (e.g., MPS uses this).

### 6.2 Shared Memory vs Register Pressure

**Shared Memory Option:**
- Allocate 32 KB per threadgroup for A_tile + B_tile
- Reduce register pressure (fewer intermediate values stored in registers)
- Trade-off: Synchronization overhead (barrier between tile loads)
- Suitable when: Kernel is memory-bound, SIMD groups are small (8-16)

**Register Blocking Option:**
- Keep A_tile in registers, B_tile streamed from L2 cache
- Higher register usage (256+ per thread), reduces occupancy
- Trade-off: Minimizes synchronization barriers
- Suitable when: Kernel is compute-bound, high occupancy essential

### 6.3 CPU vs GPU Dispatch for Attention

For attention with small sequence length (< 1024):
- CPU (scalar attention): Simple loop, low latency, but single-threaded (50 ms)
- GPU (tile-based FlashAttention): Parallel, ~10 ms but kernel launch overhead (5-10 ms setup)

**Threshold:** seq_len > 256 suggests GPU, < 256 suggests CPU.

---

## 7. EXPERT INSIGHT

### 7.1 Metal Kernel Performance Debugging

**Symptom:** Metal kernel achieves 100 GFLOPS instead of expected 300 GFLOPS.

**Debugging Workflow:**

```swift
// Step 1: Profile with Xcode Instruments
// Xcode → Instruments → System Trace, GPU timeline
// Look for:
//   - GPU utilization (ideal: 85%+)
//   - Kernel launch latency (ideal: < 5 μs)
//   - Memory stalls (L2 cache misses, DRAM latency)

// Step 2: Measure memory bandwidth efficiency
let bandwidthMeasured = (totalBytesTransferred / totalTimeSeconds) / 1e9  // GB/s
let bandwidthTheoretical = 100.0  // GB/s for M2

if bandwidthMeasured < bandwidthTheoretical * 0.5 {
    // Memory bandwidth bottleneck
    // Solutions: Improve spatial locality, use tiling, vectorize
}

// Step 3: Measure compute efficiency
let computeUtilization = (actualGFLOPS / peakGFLOPS) * 100
if computeUtilization < 50 {
    // Compute pipeline not saturated
    // Solutions: Reduce branches, improve instruction parallelism, ILP
}
```

### 7.2 MLX vs Metal: When to Use Which

**MLX Advantages:**
- Automatic backend selection (user doesn't think about CPU/GPU)
- Lazy evaluation enables graph-level optimizations
- Seamless integration with Python ML ecosystem

**MLX Disadvantages:**
- Black-box performance (hard to understand where cycles go)
- Limited to MLX's pre-written kernels
- Lazy evaluation adds overhead for frequent small ops

**Metal Advantages:**
- Full control over execution (understand every cycle)
- Custom kernels for domain-specific ops (e.g., sparse matmul)
- Predictable performance for real-time systems

**Metal Disadvantages:**
- Requires GPU kernel programming (error-prone)
- No automatic differentiation (must write backward kernels)
- Development overhead (compile-debug cycle)

**Decision Heuristic:**
```
if using_established_ML_model:
    use MLX (or CoreML for higher abstraction)
elif have_custom_operators:
    use Metal
elif latency_is_critical_SLA (< 10 ms):
    use Metal (direct control)
elif throughput_is_critical:
    use MLX (automatic optimization)
```

---

## 8. BENCHMARKING METHODOLOGY

### 8.1 Metal Kernel Profiling with Xcode Instruments

```swift
import os.log

let logger = os.Logger(subsystem: "Metal-Profiling", category: "GEMM")

func profileMetalGEMM() {
    let device = MTLCreateSystemDefaultDevice()!
    let commandQueue = device.makeCommandQueue()!

    // Create GEMM instance
    let gemm = MetalGEMM(device: device)

    let M = 1024, K = 1024, N = 1024

    // Create buffers
    let A = device.makeBuffer(length: M * K * 2)!
    let B = device.makeBuffer(length: K * N * 2)!
    let C = device.makeBuffer(length: M * N * 2)!

    // Warmup
    for _ in 0..<5 {
        _ = gemm.matmul(A: A, B: B, C: C, M: M, K: K, N: N)
    }

    // Benchmark
    var times: [Double] = []
    for _ in 0..<50 {
        let start = Date()
        _ = gemm.matmul(A: A, B: B, C: C, M: M, K: K, N: N)
        times.append(Date().timeIntervalSince(start))
    }

    let avgTime = times.reduce(0, +) / Double(times.count)
    let p95 = times.sorted()[Int(Double(times.count) * 0.95)]

    let flops = Double(2 * M * K * N)
    let gflops = flops / (avgTime * 1e9)

    logger.info("GEMM (\(M)×\(K)×\(N))")
    logger.info("Avg latency: \(avgTime * 1000)ms")
    logger.info("P95 latency: \(p95 * 1000)ms")
    logger.info("Throughput: \(gflops)GFLOPS")

    // Expected: ~300 GFLOPS for tiled kernel
}
```

### 8.2 MLX Benchmark Example

```python
import mlx.core as mx
import time

# Benchmark MLX GEMM
def benchmark_mlx_gemm(m, k, n, num_runs=100):
    A = mx.random.normal((m, k), dtype=mx.float16)
    B = mx.random.normal((k, n), dtype=mx.float16)

    # Warmup (triggers JIT compilation)
    for _ in range(5):
        C = mx.matmul(A, B)
        mx.eval(C)

    # Benchmark
    times = []
    for _ in range(num_runs):
        start = time.time()
        C = mx.matmul(A, B)
        mx.eval(C)
        times.append(time.time() - start)

    avg_time = sum(times) / len(times)
    flops = 2 * m * k * n
    gflops = flops / (avg_time * 1e9)

    print(f"MLX GEMM ({m}×{k}×{n})")
    print(f"Avg latency: {avg_time*1000:.2f}ms")
    print(f"Throughput: {gflops:.1f}GFLOPS")

    return gflops

# Test on M2 Pro
benchmark_mlx_gemm(1024, 1024, 1024)
# Expected: ~600-800 GFLOPS (GPU) or ~50-100 GFLOPS (CPU)
```

---

## 9. OPEN PROBLEMS

1. **Sparse Tensor Support in Metal:** Most ML workloads use dense operations. Sparse matmul (irregular access patterns) poorly suited to tile-based architecture. Challenge: Custom Metal kernels for sparse formats (CSR, COO), but no framework integration.

2. **Dynamic Shape Execution:** Metal kernels are compiled for fixed shapes. Variable batch sizes or sequence lengths require recompilation or padding. Solution: JIT compilation per shape (expensive) or multi-version kernels (code bloat).

3. **Automatic Vectorization:** Metal compiler doesn't automatically vectorize (half4) operations. Manual half4 programming is error-prone. Solution: Higher-level abstraction (e.g., Metal compute shaders with vector type annotations).

4. **Memory Coalescing Visibility:** No direct visibility into memory access patterns for coalescing analysis. Developers must reason about cache hierarchy manually. Solution: Metal profiler enhanced with memory access tracing.

5. **Cross-Threadgroup Communication:** Threadgroups cannot directly synchronize across different groups. Requires global memory + barriers. Limits kernel fusion opportunities.

---

## 10. PHD QUALIFIER QUESTIONS

**Q1 (Foundation):** "Explain how TBDR (tile-based deferred rendering) architecture in Apple GPUs influences optimal tile sizes for ML kernels. For a 1024×1024 GEMM with M2 GPU, calculate optimal tile dimensions (tile_m × tile_k) to maximize L1 cache utilization."

**A1 Outline:**
- TBDR processes fixed-size tiles (32×32 pixels analogy)
- For ML, tile size should maximize data reuse within tile memory
- Tile memory: ~32 KB (threadgroup shared memory)
- Data layout for tile (M, K) dimensions:
  ```
  Tile size = (tile_m, tile_k) FP16
  Memory = tile_m × tile_k × 2 bytes + tile_k × tile_n × 2 bytes
  32 KB = 16,384 bytes = tile_m × tile_k × 2 + tile_k × tile_n × 2
  ```
- If tile_m = tile_n = 32 (square tiles):
  ```
  16,384 = 32 × tile_k × 2 + tile_k × 32 × 2
  16,384 = 64 × tile_k + 64 × tile_k = 128 × tile_k
  tile_k = 128 (unrealistic, exceeds shared memory)
  ```
- Practical: tile_m = 32, tile_k = 64, tile_n = 16:
  ```
  Memory = 32 × 64 × 2 + 64 × 16 × 2 = 4096 + 2048 = 6144 bytes (fits)
  ```
- Roofline analysis: Arithmetic intensity = (2 × 32 × 64 × 16) / (32×64 + 64×16 + 32×16 × 2) = ~2 FLOPS/byte
  - Roofline ceiling = 100 GB/s × 2 = 200 GFLOPS (still memory-bound)

**Q2 (Systems Design):** "Design a Metal kernel for FlashAttention on M2 GPU. Your model has 4K context length, 128 hidden dim, 8 attention heads. Calculate memory bandwidth requirements and estimate latency. Compare against standard attention (no tiling)."

**A2 Outline:**
- Standard Attention:
  - Compute S = Q @ K^T: (4096, 128) @ (128, 4096) = 4096×4096 = 16 M elements
  - Memory: Q (512 KB) + K (512 KB) + S (32 MB) + temp = ~34 MB
  - Latency: 34 MB ÷ 100 GB/s = 340 μs (memory)
  - Compute S @ V: (4096, 4096) @ (4096, 128) = 2×4096×4096×128 = 4.3 TOPS
  - Latency: 4.3 TOPS ÷ (80 GFLOPS) = 53.7 ms (compute-bound)
  - **Total: ~54 ms**

- FlashAttention:
  - Tile Q and K: tile_size = 64 rows
  - Num tiles: 4096 ÷ 64 = 64 tiles
  - Per tile: Q (8 KB) + K (8 KB) + running stats (1 KB) = 17 KB (fits in L1)
  - Per tile latency: 64×64×128 = 0.5 M ops ÷ 80 GFLOPS = 6 μs (compute)
  - Num tiles: 64 → 64 × 6 μs = 384 μs (compute)
  - Memory bandwidth (streaming): 2×64×64×64×2 bytes per tile = 1 MB per tile → 64 MB total
  - Bandwidth latency: 64 MB ÷ 100 GB/s = 640 μs (overlapped with compute)
  - **Total: ~1 ms (55× faster!)**

**Q3 (Advanced ML Systems):** "MLX chooses between CPU and GPU based on arithmetic intensity. For a sequence of operations (embedding → attention → FFN → softmax), estimate arithmetic intensity of each operation and predict backend assignment on M2. Propose an optimization to reduce data movement."

**A3 Outline:**
- Embedding (vocab_size=50K, seq_len=512, hidden=1024):
  - FLOPs: 512 × 50K lookups ≈ 26 M ops (simple memory ops)
  - Bytes: 50K × 1024 × 2 = 102 MB (embedding table load)
  - AI = 26M / 102M = 0.25 FLOPS/byte
  - Assignment: **CPU** (low AI, acceptable latency for lookup)

- Attention (seq_len=512, d_k=1024):
  - FLOPs: 2 × 512 × 512 × 1024 = 512 M ops
  - Bytes: 512×1024×2 + 512×1024×2 + 512×512×2 = 3.1 MB
  - AI = 512M / 3.1M = 165 FLOPS/byte
  - Assignment: **GPU** (high AI, compute-bound)

- FFN (hidden=1024, ffn_dim=4096, seq_len=512):
  - FLOPs: 2 × 512 × 1024 × 4096 = 4.3 TOPS
  - Bytes: 512×1024×2 + 1024×4096×2 + (accumulated) = 10 MB
  - AI = 4.3 TOPS / 10MB = 430 FLOPS/byte
  - Assignment: **GPU** (high AI)

- Softmax (seq_len=512, hidden=1024):
  - FLOPs: ~3 × 512 × 1024 = 1.5 M ops (exp, sum, div)
  - Bytes: 512×1024×2 = 1 MB
  - AI = 1.5M / 1M = 1.5 FLOPS/byte
  - Assignment: **CPU or GPU** (borderline, depends on kernel launch overhead)

- Optimization (Kernel Fusion):
  - Fuse attention + softmax: GPU kernel computes S = Q@K^T, applies softmax, multiplies V in single kernel
  - Fused memory: Q (512 KB) + K (512 KB) + V (512 KB) = 1.5 MB (vs 3.1 MB unfused)
  - AI improvement: 165 → 250 FLOPS/byte
  - Expected speedup: 1.5× (fusion eliminates S write/read cycle)

**Q4 (Open Problem):** "Metal kernels must be pre-compiled for fixed shapes. Propose a JIT compilation strategy for dynamic attention sequences while maintaining sub-1ms latency for inference. What metadata must be extracted during model loading?"

**A4 Outline:**
- Problem: Recompiling Metal kernel per sequence length incurs 50-100 ms overhead
- Solution: Hierarchical template library
  - Compile templates for seq_len in {128, 256, 512, 1024, 2048, 4096}
  - At runtime, select template ≥ actual seq_len, pad to template size
  - Trade: Some wasted compute (padding) vs compilation overhead

- Alternative: Multi-version kernels
  - Single kernel supports variable seq_len via dynamic loop unrolling
  - Metadata extracted during model load:
    - Expected seq_len distribution (histogram from training data)
    - Frequency of each seq_len
  - Compile N versions optimized for top-N seq_len values
  - Route execution to best-matching version at inference time

- Metadata requirements:
  - Model architecture (seq_len dependency graph)
  - Typical seq_len values (from deployment telemetry)
  - GPU hardware capabilities (thread count, shared memory size)

- Implementation outline:
  ```
  Upon model load:
    seq_len_stats = analyze_model_seq_length_usage()
    top_k_seq_lens = seq_len_stats.most_common(k=5)

    for seq_len in top_k_seq_lens:
      compile_attention_kernel(seq_len)  # 50-100ms, done once

  At inference:
    actual_seq_len = input_shape[-1]
    selected_kernel = find_closest_compiled_kernel(actual_seq_len)
    execute_kernel(selected_kernel, pad_to_size(input, selected_kernel.seq_len))
  ```

- Expected performance: <1 ms overhead (kernel selection is O(1) hash lookup)

---

## CONCLUSION

Metal programming on Apple Silicon enables hand-optimized kernels for ML inference, particularly for custom operators or extreme latency requirements. Mastery requires understanding SIMD groups, shared memory utilization, tile-based architecture implications, and roofline analysis for performance prediction. While CoreML provides convenient abstraction, Metal empowers developers to achieve 2-3× speedups through careful kernel design. MLX framework bridges this gap, providing automatic backend selection without explicit GPU programming. Next module explores llama.cpp's integration with Metal, demonstrating these concepts in production LLM serving.

