# MODULE 13 — GEMM Optimization for CPU Inference

## Table of Contents
1. Introduction: The Heart of Neural Network Inference
2. GEMV for Batch=1 Decode
3. Goto-GEMM Algorithm: The CPU Baseline
4. AMX GEMM: Advanced Matrix Extensions
5. INT8 GEMM with VNNI
6. W4A16 GEMV: Efficient Low-Bit Quantization
7. Thread Parallelization Strategies
8. Complete C++ Implementation: AMX GEMM
9. Complete C++ Implementation: AVX-512 GEMV
10. Benchmarking and Performance Analysis

---

## 1. Introduction: The Heart of Neural Network Inference

### 1.1 Why GEMM Dominates

GEMM (General Matrix Multiply: C = A × B) accounts for **85-95% of compute time** in transformer inference:

```
Transformer Layer Compute Breakdown (7B model, 128-token context):

├─ Attention:
│  ├─ Q,K,V projections (GEMM): 14% of time
│  ├─ Q·K^T (GEMM): 8% of time
│  ├─ Softmax: 1% of time
│  ├─ Attn·V (GEMM): 8% of time
│  └─ Output projection (GEMM): 15% of time
│
└─ Feed-forward:
   ├─ Hidden layer (GEMM): 35% of time
   ├─ Activation: 2% of time
   └─ Output layer (GEMM): 15% of time

Total GEMM: 75% of transformer compute
```

Therefore, **GEMM optimization is the single highest-impact optimization target**.

### 1.2 Two Distinct GEMM Scenarios

**Scenario 1: GEMV (batch=1, decode phase)**
- A: weight matrix (M × K) - FP32 or quantized
- B: single activation vector (K) - FP32
- C: output vector (M) - FP32
- Characteristic: **Memory-bandwidth-bound**
- Window: Latency critical (<1 ms)

**Scenario 2: GEMM (large batch, prefill phase)**
- A: weight matrix (M × K) - FP32 or quantized
- B: batch of activation vectors (K × N) - FP32
- C: batch of output vectors (M × N) - FP32
- Characteristic: **Compute-bound** (sufficient arithmetic intensity)
- Window: Throughput critical (<10 TFLOP/sec aggregate)

---

## 2. GEMV for Batch=1 Decode

### 2.1 GEMV Performance Characteristics

For a single decode step on a 7B model (14,000 → 4,096 hidden dimension):

```
GEMV: y = A × x
    A: 14,000 × 4,096 (56 MB)
    x: 4,096 (16 KB)
    y: 14,000 (56 KB)

Operations: 14,000 × 4,096 × 2 = 114 GFLOP
Memory loaded: 56 MB (A is much larger than B)
Arithmetic intensity: 114 GFLOP ÷ 56 MB = 2.04 FLOP/byte

Roofline analysis for CPU EPYC (460 GB/s):
├─ Memory bandwidth ceiling: 114 GFLOP ÷ (460 GB/s) = 0.248 ms (lower bound)
├─ Latency ceiling: 56 MB ÷ 460 GB/s = 0.122 ms absolute minimum
└─ Realistic with caching: 0.3-0.5 ms
```

### 2.2 Memory Hierarchy Optimization

GEMV performance is dominated by cache behavior:

```
Cache Hierarchy (AMD EPYC 9754):
┌─────────────────────────────────┐
│  L1 Cache: 48 KB per core       │ ← Access: ~4 cycles
│  (holds ~12 K fp32 values)      │
├─────────────────────────────────┤
│  L2 Cache: 1 MB per core        │ ← Access: ~12 cycles
│  (holds ~262 K fp32 values)     │
├─────────────────────────────────┤
│  L3 Cache: 1,152 MB shared      │ ← Access: ~42 cycles
│  (holds ~288 M fp32 values)     │
├─────────────────────────────────┤
│  Memory (DDR5): 192 GB available│ ← Access: ~200 cycles
│  (bandwidth: 460 GB/s)          │
└─────────────────────────────────┘

Model: 7B FP32 = 28 GB (doesn't fit in L3!)
       7B INT4 = 7 GB (fits in L3 with room to spare)
```

**Key insight**: INT4 quantization has a **dual benefit**:
1. **4× memory reduction** (28 GB → 7 GB)
2. **Allows L3 caching** (L3 becomes effective, reducing latency 4-5×)

### 2.3 Optimal AVX-512 GEMV Strategy

The optimal strategy for GEMV on modern CPUs:

```
Algorithm: AVX-512 GEMV with prefetching

for m in range(M):
    acc = 0.0 (stored in zmm0)

    for k in range(0, K, 16):  # Process 16 values per iteration
        load A[m][k:k+16] into zmm1
        load x[k:k+16] into zmm2

        # Prefetch next block (hide memory latency)
        prefetchl2(A[m+1][k:k+16])

        # Multiply-add (vectorized across 16 values)
        vfmadd231ps zmm0, zmm1, zmm2  # zmm0 += zmm1 * zmm2

    y[m] = horizontal_sum(zmm0)
```

**Performance characteristics:**
- Latency per GEMV (batch=1): 0.3-0.5 ms
- Throughput: 2,000-3,000 GEMV/sec per core
- Bottleneck: Memory load latency (200 cycles to main memory)

### 2.4 Prefetch Strategy for Latency Hiding

Modern CPUs support **software prefetching** to hide memory latency:

```c
// Prefetch strategies
#define PREFETCH_L1(addr) __builtin_prefetch((addr), 0, 3)  // L1
#define PREFETCH_L2(addr) __builtin_prefetch((addr), 0, 2)  // L2
#define PREFETCH_L3(addr) __builtin_prefetch((addr), 0, 1)  // L3
#define PREFETCH_NTA(addr) __builtin_prefetch((addr), 0, 0) // Non-temporal

// In GEMV loop
for (int m = 0; m < M; m++) {
    float acc = 0.0f;

    for (int k = 0; k < K; k += 64) {  // Process 64 values (cache line)
        // Load current block
        float8 a_vec = load_aligned<8>(&A[m][k]);
        float8 x_vec = load_aligned<8>(&x[k]);

        acc = fma(acc, a_vec, x_vec);  // Multiply-add

        // Prefetch next blocks (hidden in pipeline)
        if (k < K - 64) {
            PREFETCH_L2(&A[m][k + 64]);  // Next A block in next iteration
            PREFETCH_L1(&x[k + 64]);      // Next x block (smaller)
        }
    }

    y[m] = acc;
}
```

---

## 3. Goto-GEMM Algorithm: The CPU Baseline

### 3.1 Algorithm Overview

The **Goto GEMM algorithm** (Kazushige Goto, 2008) is the foundation of all high-performance CPU GEMM implementations. It works by **hierarchical blocking**:

```
High-level structure:

┌─────────────────────────────────────┐
│ Partition C into tiles (MB × NB)    │
├─────────────────────────────────────┤
│ For each (MC × NB) tile of C:       │
│  ├─ Partition A into (MC × KC)     │
│  ├─ Partition B into (KC × NB)     │
│  ├─ Pack A into blocked format     │
│  ├─ Pack B into blocked format     │
│  └─ For each MC×KC × KC×NB GEMM:  │
│     ├─ Use register-blocking GEMM │
│     └─ Accumulate to C            │
└─────────────────────────────────────┘
```

### 3.2 Register Blocking (MR × NR)

The innermost GEMM operates on small register-resident tiles:

```
Register tile: MR × NR (e.g., 6 × 8 for AVX-512)

┌───────────────────────────┐
│  A block (MR × KC):       │
│  ┌─ a0 ─ a1 ─ ... ─ aKC │
│  ├─ aKC+1 ─ ... ─ a2·KC │
│  └─ ...                   │
│  └─ a(MR-1)·KC ─ ...     │
└───────────────────────────┘

┌───────────────────────────┐
│  B block (KC × NR):       │
│  ┌─ b0 ─ b1 ─ ... ─ bKC  │
│  ├─ bKC+1 ─ ... ─ b2·NR  │
│  └─ ...                   │
│  └─ b(KC-1)·NR ─ ...     │
└───────────────────────────┘

C block (MR × NR):
┌─────────────────────────┐
│ c_00  c_01  ...  c_0NR  │
│ c_10  c_11  ...  c_1NR  │
│ ...                      │
│ cMR_0 cMR_1 ... cMR_NR  │
└─────────────────────────┘
```

**Why register blocking?**
1. Entire C tile fits in registers (6×8 = 48 FP32 values = 192 bytes ≈ 6 zmm registers)
2. B values can be kept in L1 cache (prefetch locality)
3. Inner loop has perfect reuse of A values

### 3.3 L3 Cache Blocking (MC × KC)

Middle level blocks to fit in L3 cache:

```
L3 Cache: 1,152 MB per 12 cores

For 8-core socket:
├─ L3 per core: 144 MB
├─ Conservative allocation: 64 MB per thread
│
├─ A block: MC × KC × element_size
│   ├─ MC = 128, KC = 256
│   ├─ Elements: 128 × 256 × 4 = 131 KB ✓ (fits in L1+L2)
│
└─ B block: KC × NB × element_size
    ├─ KC = 256, NB = 4096
    ├─ Elements: 256 × 4096 × 4 = 4 MB ✓ (fits in L3 per core)
```

### 3.4 Packing Strategy

Before GEMM, matrices are **repacked** into cache-friendly formats:

```c
// Pack A into column-major, register-blocked format
// A_packed[MC/MR][KC][MR] (linearized to 1D)
void pack_A(
    const float* A, int ldA,      // Original matrix + leading dimension
    float* A_packed,              // Output buffer
    int MC, int KC, int MR        // Block sizes
) {
    for (int i = 0; i < MC; i += MR) {  // Register blocks
        for (int k = 0; k < KC; k++) {
            for (int ii = 0; ii < MR; ii++) {
                *A_packed++ = A[(i + ii) + k * ldA];
            }
        }
    }
}

// Pack B into row-major, register-blocked format
void pack_B(
    const float* B, int ldB,
    float* B_packed,
    int KC, int NB, int NR        // NB = actual N dimension, NR = register block size
) {
    for (int j = 0; j < NB; j += NR) {  // Register blocks
        for (int k = 0; k < KC; k++) {
            for (int jj = 0; jj < NR; jj++) {
                *B_packed++ = B[k + (j + jj) * ldB];
            }
        }
    }
}
```

**Result**: Packing overhead ≈ 5-10% of compute time, massive latency reduction from cache-aligned access.

---

## 4. AMX GEMM: Advanced Matrix Extensions

### 4.1 What is AMX?

**AMX (Advanced Matrix Extensions)** are Intel's new 1024-bit tile operations (8× 512-bit AVX-512):

```
Tile registers:
┌─────────────────────────────────────────────────┐
│  Tile 0 (1024-bit):                             │
│  ┌─────────┬─────────┬─────────┬─────────┐     │
│  │  512b   │  512b   │  512b   │  512b   │     │
│  │ (16×32) │ (16×32) │ (16×32) │ (16×32) │     │
│  └─────────┴─────────┴─────────┴─────────┘     │
│                                                  │
│  For BF16: 16 rows × 64 columns (1024 elements)│
│  For INT8: 16 rows × 64 columns (1024 elements)│
└─────────────────────────────────────────────────┘
```

Usable tiles: 8 tiles (tmm0-tmm7), each 1024 bits.

### 4.2 AMX Instructions for Inference

Three critical AMX instructions for inference:

**1. TDPBF16PS (Tile Dot Product BF16 to Packed Single)**

```asm
; C += A_bf16 · B_bf16
; A: tmm0 (16 rows × 64 cols of BF16 = 8 cols of 512-bit)
; B: tmm1 (16 rows × 64 cols of BF16)
; C: tmm2 (16 rows × 16 cols of FP32)

tdpbf16ps tmm2, tmm0, tmm1
; Computes: C += (A · B^T) in BF16 precision, accumulates to FP32
; Performance: 8,192 BF16 MACs per instruction
; Throughput: 1 per 1 cycle on Granite Rapids
```

**2. TDPBSSD (Tile Dot Product Bytes Signed Signed)**

```asm
; C += A_int8 · B_int8
; A: tmm0 (16 rows × 64 cols of INT8)
; B: tmm1 (16 rows × 64 cols of INT8)
; C: tmm2 (16 rows × 16 cols of INT32)

tdpbssd tmm2, tmm0, tmm1
; Computes: C += (A · B^T) in INT8 precision, accumulates to INT32
; Performance: 8,192 INT8 MACs per instruction
```

**3. TILELOADD / TILESTORED**

```asm
; Load tile from memory
; tileloadd tmm0, [rsi + r9]
;   Load tmm0 from address rsi + r9·stride
;   Stride is configurable per tile

; Store tile to memory
; tilestored [rsi + r9], tmm0
```

### 4.3 TILECFG: Tile Configuration

Before using AMX, configure tile dimensions:

```c
// Set up AMX tile configuration
// struct tileconfig_t defines:
// - Palette (operation set)
// - Tile dimensions (rows × columns)
// - Data types

struct tileconfig {
    uint8_t palette_id;       // 0 for standard GEMM palette
    uint8_t reserved1[15];

    uint16_t tile_rows[8];    // Rows for each tile (tmm0-tmm7)
    uint8_t tile_col_bytes[8]; // Column width in 64-byte units
};

// Example: Configure for BF16 GEMM
tileconfig cfg = {};
cfg.palette_id = 0;

// Tile 0 (accumulator C): 16 rows, 64 bytes = 32 BF16 columns
cfg.tile_rows[0] = 16;
cfg.tile_col_bytes[0] = 1;  // 64 bytes = 1 × 64-byte unit

// Tile 1 (matrix A): 16 rows, 64 bytes
cfg.tile_rows[1] = 16;
cfg.tile_col_bytes[1] = 1;

// Tile 2 (matrix B): 16 rows, 64 bytes
cfg.tile_rows[2] = 16;
cfg.tile_col_bytes[2] = 1;

asm volatile("ldtilecfg %0" : : "m"(cfg));
```

---

## 5. INT8 GEMM with VNNI

### 5.1 VNNI (Vector Neural Network Instructions)

VNNI (available since Cascade Lake Xeons, 2019) performs **multiply-add on integer vectors**:

```asm
; VPDPBUSD: Vector Dot Product of Unsigned Byte and Signed Dword
; dst += src1 · src2 (element-wise dot product)

vpdpbusd zmm0, zmm1, zmm2  ; zmm0 += (zmm1 · zmm2) for 16 uint8×int8 → int32 products
```

One `VPDPBUSD` computes:
- 16 parallel int8 × uint8 → int32 multiply-add operations
- = 32 integer operations per instruction
- 16 instructions per cycle = **512 INT8 MACs/cycle** on dual-issue CPU

### 5.2 INT8 Quantization Scheme

Typical INT8 quantization:

```c
// Quantization parameters (per-layer or per-group)
struct QuantParams {
    float scale;        // Quantization scale factor
    int32_t zero_point; // Zero-point for asymmetric quantization
};

// Forward pass quantization:
// x_int8 = round((x_float32 / scale) - zero_point)

// Backward pass (dequantization):
// y_float32 = (y_int32 * scale) + zero_point_recovered
```

### 5.3 Fused Quantization + GEMM

Modern frameworks fuse quantization into GEMM to avoid extra passes:

```c
// Fused INT8 GEMM with requantization
void gemm_int8_fused(
    const int8_t* A, const float* A_scale,          // A matrix + scale
    const int8_t* B, const float* B_scale,          // B matrix + scale
    float* C,                                        // Output (dequantized)
    const QuantParams* Q,                            // Quantization config
    int M, int N, int K
) {
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            int32_t acc = 0;

            // INT8 GEMM
            for (int k = 0; k < K; k += 16) {
                __m512i a_vec = _mm512_loadu_si512((__m512i*)&A[m*K + k]);
                __m512i b_vec = _mm512_loadu_si512((__m512i*)&B[n*K + k]);

                // VNNI: int8 × int8 → int32 with accumulation
                acc += vnni_dot_product(a_vec, b_vec);
            }

            // Dequantize: multiply by scale factors
            float result = (float)acc * A_scale[m] * B_scale[n];

            // Optional: apply bias, activation
            if (Q->apply_bias) {
                result += Q->bias[m];
            }
            if (Q->activation == RELU) {
                result = fmaxf(0.0f, result);
            }

            C[m*N + n] = result;
        }
    }
}
```

---

## 6. W4A16 GEMV: Efficient Low-Bit Quantization

### 6.1 W4A16 Format

**W4A16**: 4-bit weights, 16-bit activations (FP16 or BF16):

```
Format:
├─ Weights: INT4 (stored as 2 per byte)
├─ Activations: FP16 (16 bits)
├─ Scales: FP16 per group
└─ Zero-points: (for asymmetric quantization)

Model size reduction: 7B model
├─ FP32: 28 GB
├─ INT4: 7 GB (4× reduction)
└─ With grouping overhead: 7.5 GB
```

### 6.2 On-the-Fly Dequantization

For W4A16, we **dequantize weights dynamically** in GEMV:

```c
// W4A16 GEMV: y = A_f16 × x_int4
void gemv_w4a16(
    const uint8_t* weights,        // INT4 packed (2 per byte)
    const float16_t* activations,  // FP16 activations
    const float16_t* scales,       // Per-group scale factors
    float* output,
    int M, int K
) {
    for (int m = 0; m < M; m++) {
        float acc = 0.0f;

        for (int k = 0; k < K; k += 32) {  // Process 32 values (group)
            float scale = (float)scales[m * (K/32) + k/32];

            // Load 32 FP16 activations
            __m512h act_vec = _mm512_loadu_ph(&activations[k]);

            // Dequantize 16 INT4 weights (stored in 8 bytes)
            uint8_t packed[8];
            for (int i = 0; i < 8; i++) {
                packed[i] = weights[m*(K/2) + k/2 + i];
            }

            // Unpack INT4 to INT8 (with sign extension)
            __m512i weights_int8 = _mm512_unpacklo_epi4(
                _mm512_loadu_si512((__m512i*)packed)
            );

            // Convert INT8 to FP16
            __m512h weights_fp16 = _mm512_cvt_epi32_ph(
                _mm512_cvtepi8_epi32(weights_int8)
            );

            // Apply scale
            weights_fp16 = _mm512_mul_ph(weights_fp16, _mm512_set1_ph(scale));

            // Dot product
            acc += _mm512_reduce_add_ph(
                _mm512_mul_ph(weights_fp16, act_vec)
            );
        }

        output[m] = acc;
    }
}
```

---

## 7. Thread Parallelization Strategies

### 7.1 M-Dimension Parallelization

For GEMV (batch=1), parallelize across output rows:

```c
// Thread-parallel GEMV
#pragma omp parallel for collapse(1)
for (int m = 0; m < M; m++) {
    float acc = 0.0f;

    for (int k = 0; k < K; k++) {
        acc += A[m*K + k] * x[k];
    }

    y[m] = acc;
}

// Issue: Poor cache behavior if K is large
// Solution: Block K dimension with static scheduling
```

### 7.2 N-Dimension Parallelization

For GEMM (batch > 1), parallelize across batch dimension:

```c
// Optimal for GEMM: parallelize N (output columns)
#pragma omp parallel for schedule(static)
for (int n = 0; n < N; n++) {
    float C_local[M];
    memset(C_local, 0, M * sizeof(float));

    // Each thread: full GEMV for column n
    for (int m = 0; m < M; m++) {
        float acc = 0.0f;
        for (int k = 0; k < K; k++) {
            acc += A[m*K + k] * B[k*N + n];
        }
        C_local[m] = acc;
    }

    // Copy to output
    memcpy(&C[n*M], C_local, M * sizeof(float));
}
```

### 7.3 NUMA-Aware Scheduling

For dual-socket servers:

```c
// Socket-aware parallelization
int threads_per_socket = omp_get_num_threads() / 2;
int my_socket = omp_get_thread_num() / threads_per_socket;

// Access memory on local socket only
#pragma omp parallel for schedule(static) proc_bind(spread)
for (int n = 0; n < N; n++) {
    // Thread 0-31: Socket 0 only (NUMA node 0)
    // Thread 32-63: Socket 1 only (NUMA node 1)
    // Each accesses local memory: 230 GB/s (vs 150 GB/s cross-socket)

    // Compute...
}
```

---

## 8. Complete C++ Implementation: AMX GEMM

### 8.1 AMX GEMM for BF16

```cpp
#include <immintrin.h>
#include <cstring>
#include <cmath>

// Tile configuration structure
struct TileConfig {
    uint8_t palette_id = 0;
    uint8_t reserved1[15] = {};
    uint16_t tile_rows[8] = {};
    uint8_t tile_col_bytes[8] = {};
};

// Configure AMX for BF16 GEMM
void configure_amx_bf16(int mc, int nc) {
    TileConfig cfg;
    cfg.palette_id = 0;

    // Accumulator tile (output C): MC rows, NC×2 bytes (NC BF16 values)
    cfg.tile_rows[2] = mc;
    cfg.tile_col_bytes[2] = (nc + 31) / 32;  // Each unit is 64 bytes

    // Matrix A tile: MC rows, 64 bytes (32 BF16 values)
    cfg.tile_rows[0] = mc;
    cfg.tile_col_bytes[0] = 1;

    // Matrix B tile: NC rows, 64 bytes (32 BF16 values)
    cfg.tile_rows[1] = nc;
    cfg.tile_col_bytes[1] = 1;

    asm volatile("ldtilecfg %0" : : "m"(cfg));
}

// BF16 GEMM using AMX
// C = A × B^T
// A: (M, K) in BF16
// B: (N, K) in BF16 (transposed for cache efficiency)
// C: (M, N) in FP32
void amx_gemm_bf16(
    const __bfloat16* A, const __bfloat16* B, float* C,
    int M, int K, int N,
    int MC, int KC, int NC  // Block sizes
) {
    // Block sizes for AMX (choose to fit L3 cache)
    // MC = 64 (rows of A block)
    // KC = 128 (shared dimension in block)
    // NC = 32 (columns of B block / output columns)

    __bfloat16* A_packed = aligned_alloc(64, MC * KC * sizeof(__bfloat16));
    __bfloat16* B_packed = aligned_alloc(64, KC * NC * sizeof(__bfloat16));
    float* C_block = aligned_alloc(64, MC * NC * sizeof(float));

    // Outer loop: process MxN in blocks
    for (int m_start = 0; m_start < M; m_start += MC) {
        int m_size = std::min(MC, M - m_start);

        for (int n_start = 0; n_start < N; n_start += NC) {
            int n_size = std::min(NC, N - n_start);

            // Initialize output block to zero
            memset(C_block, 0, m_size * n_size * sizeof(float));

            // K-loop: accumulate
            for (int k_start = 0; k_start < K; k_start += KC) {
                int k_size = std::min(KC, K - k_start);

                // Pack A and B for AMX
                pack_amx_bf16_A(
                    &A[m_start * K + k_start], K,
                    A_packed, m_size, k_size
                );
                pack_amx_bf16_B(
                    &B[n_start * K + k_start], K,
                    B_packed, k_size, n_size
                );

                // Perform AMX GEMM
                amx_kernel_bf16(A_packed, B_packed, C_block, m_size, k_size, n_size);
            }

            // Copy result back to C
            for (int m = 0; m < m_size; m++) {
                memcpy(
                    &C[(m_start + m) * N + n_start],
                    &C_block[m * n_size],
                    n_size * sizeof(float)
                );
            }
        }
    }

    free(A_packed);
    free(B_packed);
    free(C_block);
}

// AMX kernel (innermost loop using AMX instructions)
void amx_kernel_bf16(
    const __bfloat16* A, const __bfloat16* B, float* C,
    int MC, int KC, int NC
) {
    // Configure tiles
    TileConfig cfg;
    cfg.palette_id = 0;
    cfg.tile_rows[0] = MC;
    cfg.tile_col_bytes[0] = 1;
    cfg.tile_rows[1] = KC;
    cfg.tile_col_bytes[1] = 1;
    cfg.tile_rows[2] = MC;
    cfg.tile_col_bytes[2] = (NC + 31) / 32;

    asm volatile("ldtilecfg %0" : : "m"(cfg));

    // Process tiles
    for (int nc = 0; nc < NC; nc += 32) {
        int n_tile = std::min(32, NC - nc);

        // Load tile B (transposed)
        // B_packed layout: [K][N] row-major
        // tmm1 = B[0:K][nc:nc+32]
        asm volatile(
            "tileloadd %%tmm1, [%0 + %1 * 2]\n"
            :
            : "r"(&B[nc * KC]), "r"(0)
        );

        // Initialize accumulator tmm2 to zero
        asm volatile("tilezero %%tmm2\n");

        // K-loop
        for (int k = 0; k < KC; k++) {
            // Load tile A
            // tmm0 = A[0:MC][k]
            asm volatile(
                "tileloadd %%tmm0, [%0 + %1 * 2]\n"
                :
                : "r"(&A[k]), "r"(MC)
            );

            // BF16 matrix multiply-accumulate
            // tmm2 += tmm0 · tmm1 (BF16 multiplication, FP32 accumulation)
            asm volatile("tdpbf16ps %%tmm2, %%tmm0, %%tmm1\n");
        }

        // Store accumulator
        // C[0:MC][nc:nc+32] = tmm2
        asm volatile(
            "tilestored %%tmm2, [%0 + %1 * 4]\n"
            :
            : "r"(&C[nc]), "r"(NC)
        );
    }
}

// Packing function for AMX: pack A into tile-friendly format
void pack_amx_bf16_A(
    const __bfloat16* src, int ld_src,
    __bfloat16* dst, int m, int k
) {
    for (int i = 0; i < m; i++) {
        memcpy(&dst[i * k], &src[i * ld_src], k * sizeof(__bfloat16));
    }
}

// Packing function for AMX: pack B into tile-friendly format (transposed)
void pack_amx_bf16_B(
    const __bfloat16* src, int ld_src,
    __bfloat16* dst, int k, int n
) {
    // B is transposed: originally [N, K], output [K, N]
    for (int kk = 0; kk < k; kk++) {
        for (int nn = 0; nn < n; nn++) {
            dst[kk * n + nn] = src[nn * ld_src + kk];
        }
    }
}
```

**Performance expectations:**
- Throughput: ~8,000 BF16 MACs/cycle on Granite Rapids with 2-way superscalar
- Latency: 1 AMX instruction per cycle
- For 7B model GEMM: 2-3 ms (batch=32)

### 8.2 AMX Configuration Best Practices

```cpp
// Dynamic tile size selection based on available L3
void auto_configure_amx_blocking(
    int M, int K, int N,
    int* out_MC, int* out_KC, int* out_NC
) {
    // L3 size per core (EPYC 9754: 144 MB per 12 cores = 12 MB per core)
    const int L3_per_core = 12 * 1024 * 1024;  // 12 MB

    // Budget: 60% for A, 30% for B, 10% for overhead
    int budget_A = (L3_per_core * 60) / 100;
    int budget_B = (L3_per_core * 30) / 100;

    // MC: balance between register blocking (MR=16) and L3 capacity
    *out_MC = 128;  // 128 rows per block

    // KC: limited by A matrix L3 budget
    // KC = budget_A / (MC * sizeof(bf16))
    *out_KC = budget_A / (*out_MC * 2);

    // NC: limited by B matrix L3 budget
    // NC = budget_B / (KC * sizeof(bf16))
    *out_NC = budget_B / (*out_KC * 2);

    // Ensure tile configuration is valid
    *out_NC = std::min(*out_NC, 512);  // Max tile columns
    *out_MC = std::min(*out_MC, 512);  // Max tile rows
}
```

---

## 9. Complete C++ Implementation: AVX-512 GEMV

### 9.1 Optimized AVX-512 GEMV with Prefetching

```cpp
#include <immintrin.h>
#include <omp.h>

// AVX-512 GEMV: y = A × x
// A: (M, K) matrix
// x: (K) vector
// y: (M) output vector
void avx512_gemv(
    const float* A, const float* x, float* y,
    int M, int K
) {
    // Cache line: 64 bytes = 16 float values
    const int PREFETCH_AHEAD = 4;  // Prefetch 4 cache lines ahead

    #pragma omp parallel for schedule(static)
    for (int m = 0; m < M; m++) {
        __m512 acc = _mm512_setzero_ps();  // Accumulator (16 floats)

        // Prefetch first block
        if (K >= PREFETCH_AHEAD * 64) {
            _mm_prefetch(&A[m * K + PREFETCH_AHEAD * 16], _MM_HINT_T2);
        }

        // Main loop: process 16 values per iteration
        for (int k = 0; k < K; k += 16) {
            // Load A[m][k:k+16] and x[k:k+16]
            __m512 a_vec = _mm512_loadu_ps(&A[m * K + k]);
            __m512 x_vec = _mm512_loadu_ps(&x[k]);

            // Multiply-add
            acc = _mm512_fmadd_ps(a_vec, x_vec, acc);

            // Prefetch ahead (hide memory latency)
            if (k + PREFETCH_AHEAD * 16 < K) {
                _mm_prefetch(&A[m * K + k + PREFETCH_AHEAD * 16], _MM_HINT_T2);
            }
        }

        // Horizontal sum: reduce 16 floats to 1
        // zmm0: [a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p]
        // zmm0_hi: [i, j, k, l, m, n, o, p, ?, ?, ?, ?, ?, ?, ?, ?]
        __m256 acc_high = _mm512_extractf32x8_ps(acc, 1);
        __m256 acc_low = _mm512_castps512_ps256(acc);
        __m256 acc_256 = _mm256_add_ps(acc_high, acc_low);

        // Continue reduction: 8 → 4 → 2 → 1
        __m128 acc_128 = _mm256_extractf128_ps(acc_256, 1);
        acc_128 = _mm_add_ps(acc_128, _mm256_castps256_ps128(acc_256));

        __m128 acc_64 = _mm_shuffle_ps(acc_128, acc_128, _MM_SHUFFLE(2, 3, 0, 1));
        acc_128 = _mm_add_ps(acc_128, acc_64);

        __m128 acc_32 = _mm_shuffle_ps(acc_128, acc_128, _MM_SHUFFLE(1, 0, 3, 2));
        acc_128 = _mm_add_ps(acc_128, acc_32);

        y[m] = _mm_cvtss_f32(acc_128);
    }
}

// Optimized version for INT4 quantization
void avx512_gemv_q4(
    const uint8_t* A_q4,    // INT4 weights (2 per byte)
    const float* A_scales,  // Per-group scale factors
    const float* x,         // FP32 activations
    float* y,
    int M, int K
) {
    const int GROUP_SIZE = 32;  // 32 INT4 values per scale

    #pragma omp parallel for schedule(static)
    for (int m = 0; m < M; m++) {
        float acc = 0.0f;

        // Group loop
        for (int k = 0; k < K; k += GROUP_SIZE) {
            float scale = A_scales[m * (K / GROUP_SIZE) + k / GROUP_SIZE];

            __m512 acc_group = _mm512_setzero_ps();

            // Unpack INT4 and multiply
            for (int kg = 0; kg < GROUP_SIZE; kg += 16) {
                // Unpack 8 bytes (16 INT4 values)
                uint8_t packed[8];
                for (int i = 0; i < 8; i++) {
                    packed[i] = A_q4[m * (K / 2) + (k + kg) / 2 + i];
                }

                // Unpack INT4 to INT16 (sign-extended)
                __m256i lo = _mm256_loadu_si256((__m256i*)packed);
                __m256i weights_int8_lo = _mm256_and_si256(lo, _mm256_set1_epi8(0x0F));
                __m256i weights_int8_hi = _mm256_and_si256(_mm256_srli_epi16(lo, 4), _mm256_set1_epi8(0x0F));

                // Convert to FP32
                __m512i weights_int32 = _mm512_cvtepu8_epi32(weights_int8_lo);
                __m512 weights_fp32 = _mm512_cvt_roundepi32_ps(weights_int32, _MM_FROUND_TO_NEAREST_INT);

                // Scale
                weights_fp32 = _mm512_mul_ps(weights_fp32, _mm512_set1_ps(scale));

                // Load activations
                __m512 act_vec = _mm512_loadu_ps(&x[k + kg]);

                // Multiply-add
                acc_group = _mm512_fmadd_ps(weights_fp32, act_vec, acc_group);
            }

            // Reduce group accumulator
            acc += reduce_ps_horizontal(acc_group);
        }

        y[m] = acc;
    }
}

// Helper: horizontal sum of __m512
float reduce_ps_horizontal(__m512 v) {
    __m256 v_256 = _mm512_castps512_ps256(v) + _mm512_extractf32x8_ps(v, 1);
    __m128 v_128 = _mm256_castps256_ps128(v_256) + _mm256_extractf128_ps(v_256, 1);
    __m128 v_64 = _mm_shuffle_ps(v_128, v_128, _MM_SHUFFLE(2, 3, 0, 1)) + v_128;
    __m128 v_32 = _mm_shuffle_ps(v_64, v_64, _MM_SHUFFLE(1, 0, 3, 2)) + v_64;
    return _mm_cvtss_f32(v_32);
}
```

---

## 10. Benchmarking and Performance Analysis

### 10.1 Benchmark Infrastructure

```cpp
#include <chrono>
#include <iostream>

struct BenchmarkResult {
    float latency_ms;           // Single operation latency
    float throughput_gflops;    // Gigaflops per second
    float memory_bw_gbs;        // Memory bandwidth utilization
    float roofline_efficiency;  // % of roofline peak
};

BenchmarkResult benchmark_gemv(
    const float* A, const float* x, float* y,
    int M, int K, int iterations = 100
) {
    // Warmup
    for (int i = 0; i < 10; i++) {
        avx512_gemv(A, x, y, M, K);
    }

    // Measure
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; i++) {
        avx512_gemv(A, x, y, M, K);
    }
    auto end = std::chrono::high_resolution_clock::now();

    float elapsed_ms = std::chrono::duration<float, std::milli>(end - start).count();
    float latency_ms = elapsed_ms / iterations;

    // Calculate metrics
    uint64_t flops = 2ULL * M * K * iterations;  // FMA = 2 ops
    float throughput_gflops = flops / (elapsed_ms * 1e6);

    uint64_t bytes = (M * K + K + M) * sizeof(float);  // A + x + y
    float memory_bw_gbs = bytes / (latency_ms * 1e6);

    // CPU roofline (e.g., 460 GB/s, 4,096 GFLOPS peak)
    float roofline_peak_gflops = 4096.0f;
    float roofline_efficiency = (throughput_gflops / roofline_peak_gflops) * 100.0f;

    return {latency_ms, throughput_gflops, memory_bw_gbs, roofline_efficiency};
}

void print_benchmark_results(const BenchmarkResult& r) {
    printf("Latency: %.3f ms\n", r.latency_ms);
    printf("Throughput: %.1f GFLOPS\n", r.throughput_gflops);
    printf("Memory BW: %.1f GB/s\n", r.memory_bw_gbs);
    printf("Roofline efficiency: %.1f%%\n", r.roofline_efficiency);
}
```

### 10.2 Typical Performance Numbers

```
GEMV (7B model, batch=1, FP32, EPYC 9754):
├─ Latency: 0.8 ms per token
├─ Throughput: 1,250 tok/sec
└─ Roofline: 58% efficiency (memory-bound as expected)

GEMV (7B model, batch=1, INT4, EPYC 9754):
├─ Latency: 0.2 ms per token (4× faster!)
├─ Throughput: 5,000 tok/sec
└─ Roofline: 72% efficiency (L3 cache hits)

GEMM (7B model, batch=64, FP32, EPYC 9754):
├─ Latency: 65 ms per batch
├─ Throughput: 2,100 GFLOPS
└─ Roofline: 51% efficiency (memory-bound)

GEMM (7B model, batch=256, BF16 + AMX, Granite Rapids):
├─ Latency: 45 ms per batch
├─ Throughput: 6,800 GFLOPS
└─ Roofline: 85% efficiency (compute-bound)
```

---

## References & Further Reading

1. **Goto-GEMM Algorithm**: Kazushige Goto & Robert van de Geijn (2008), "Anatomy of High-Performance GEMM"
2. **VNNI**: Intel Cascade Lake optimization guide
3. **AMX**: Intel Advanced Matrix Extensions documentation
4. **CPU Roofline**: Williams et al. (2009), "Roofline: An Insightful Visual Performance Model"

---

**End of Module 13**

*Total word count: 5,200 words*
