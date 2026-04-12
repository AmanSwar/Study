# MODULE 9 — SIMD & Vector Computing: AVX-512 Mastery

## 1. CONCEPTUAL FOUNDATION

### 1.1 Evolution: SSE → AVX → AVX2 → AVX-512

The progression of x86 SIMD from 128-bit SSE (introduced Pentium III, 1999) through 256-bit AVX/AVX2 to 512-bit AVX-512 represents fundamental scaling in *parallelism within a single core*. This is distinct from thread-level parallelism: a single instruction now operates on 16 × 32-bit floats or 64 × 8-bit integers in parallel.

**SSE (128-bit, 4 × 32-bit floats):**
- xmm0–xmm15 (8 additional regs on x64)
- Fixed register width throughout x86-64 history
- Three operand forms (SSE3+): `movdqa xmm0, [rsi]`

**AVX (256-bit, 8 × 32-bit floats, Sandy Bridge 2011):**
- ymm0–ymm31 (register renaming, still 128-bit physical)
- Non-destructive 3-operand encoding: `vaddps ymm0, ymm1, ymm2` (ymm0 = ymm1 + ymm2)
- Lane width: operations still operate on independent 128-bit lanes (AVX weakness)

**AVX2 (256-bit, but true 256-bit operations, Haswell 2013):**
- Removed 128-bit lane boundary for most operations
- Gather instructions (variable stride loads): `vpgatherdd`, `vgatherqpd`
- Permutation instructions without lane crossing: `vpermq`
- Bit manipulation: `vpext`, `vpins` at 256-bit width

**AVX-512 (512-bit, Skylake-X 2017, Xeon Scalable 2019):**
- zmm0–zmm31 (32 full 512-bit registers)
- Mask registers k0–k7 (8 64-bit mask registers)
- Native 512-bit operations across all lanes
- Predication: per-element conditional execution via masks

See: Agner Fog, *Instruction tables: Lists of instruction latencies, throughputs and micro-operation breakdowns for Intel, AMD and VIA CPUs*, Vol 3 (for exact latency tables); Intel Optimization Reference Manual, Chapter 2 (Sandy Bridge through Sapphire Rapids).

### 1.2 AVX-512 Register Architecture

**zmm0–zmm31:** 512-bit registers. Physical layout on Sapphire Rapids:
- 4 × 256-bit segments (for L2 bandwidth constraints)
- Aliasing: zmm0 contains ymm0 (lower 256), xmm0 (lower 128), mm0 (lower 64)

**Mask Registers k0–k7:**
- Each 64 bits; one bit per 8-bit, 16-bit, 32-bit, or 64-bit element
- k0 is special: cannot be written (writes ignored, reads return all-1s)
- Merge semantics: `vpaddd zmm0{k1}, zmm1, zmm2` → zmm0[i] = (k1[i] ? zmm1[i] + zmm2[i] : zmm0[i])
- Zero semantics: `vpaddd zmm0{k1}{z}, zmm1, zmm2` → zmm0[i] = (k1[i] ? zmm1[i] + zmm2[i] : 0)

**Instruction Forms (using vadd as example):**
```
vpaddd zmm0, zmm1, zmm2          // no masking, zmm0 = zmm1 + zmm2
vpaddd zmm0{k1}, zmm1, zmm2      // merge mask: zmm0[i] = k1[i] ? zmm1[i]+zmm2[i] : zmm0[i]
vpaddd zmm0{k1}{z}, zmm1, zmm2   // zero mask: zmm0[i] = k1[i] ? zmm1[i]+zmm2[i] : 0
```

See: Intel Intrinsics Guide (https://www.intel.com/content/www/en/en/docs/intrinsics-guide/index.html), filter by AVX-512 and platform.

### 1.3 AVX-512 Family: Instruction Extensions

**AVX-512 Foundation (F):** Baseline 512-bit ops on all Skylake-X and Xeon Scalable.
- 32-bit and 64-bit int/float, basic arithmetic, permutation

**AVX-512 Byte & Word (BW):** 8-bit and 16-bit operations.
- Available on Skylake-X, Cascade Lake, Ice Lake, Sapphire Rapids
- `vpaddb zmm0, zmm1, zmm2` (64 × 8-bit adds), `vpaddw` (32 × 16-bit adds)
- Critical for NLP token processing, audio

**AVX-512 Doubleword & Quadword (DQ):** 32-bit and 64-bit specializations.
- Included with Foundation; extends with gather/scatter for those widths

**AVX-512 Vector Neural Network Instructions (VNNI):** INT8 matrix multiply.
- `vpdpbusd zmm0, zmm1, zmm2` → dot product of unsigned bytes → signed dword
- `vpdpwssd zmm0, zmm1, zmm2` → dot product of signed words → signed dword
- **This is the key instruction for INT8 inference.** One instruction computes 4 × 4-element dot product.
- Each 512-bit instruction: 4 zmm lanes × (4 dotproducts per lane) = 16 INT8 dot products per cycle

**AVX-512 BF16 (AVX-512 BF16):** Brain Float 16-bit format.
- `vcvtne2ps2bf16` (convert 2 × FP32 to 2 × BF16), `vdpbf16ps` (dot product)
- Available on Sapphire Rapids, Cooper Lake

**AVX-512 FP16 (AVX-512 FP16):** IEEE 754 half-precision (16-bit).
- `vcvtps2phx` (FP32 → FP16), `vfmadd...ph` (FP16 multiply-accumulate)
- Available on Sapphire Rapids onwards

**AVX-512 Integer Fused Multiply-Add (IFMA):** 52-bit product + 64-bit accumulator.
- `vpmadd52luq`, `vpmadd52huq` (52-bit×52-bit→104-bit)
- Rarely used in practice; only on Skylake-X and Cascade Lake

**AVX-512 Vector Byte Manipulation Instructions (VBMI):** Advanced byte permutation.
- `vpermi2b` (3-way permute), `vperm...` (variable permutation)

**AVX-512 Vector Byte Manipulation Instructions 2 (VBMI2):** Bit-level operations.
- `vpexpandb` (expand from compact representation), `vpcompressd` (compact)

**AVX-512 VP2INTERSECT:** Two-input permute-based intersection (newer SKUs only).

**SKU Availability:**
- Skylake-X (2017): F, DQ, BW (consumer/HEDT)
- Cascade Lake (2019): F, DQ, BW, VNNI (Xeon)
- Ice Lake (2021): F, DQ, BW, VNNI, BF16 (Xeon)
- Sapphire Rapids (2023): F, DQ, BW, VNNI, BF16, FP16, IFMA, VBMI, VBMI2, VP2INTERSECT

See: Intel Software Development Manual, Vol 2, Instruction Set Reference; Agner Fog Vol 3, Instruction tables for latency/throughput per extension.

### 1.4 Intel AMX (Advanced Matrix Extensions) on Sapphire Rapids

AMX introduces tile registers: conceptually 2D arrays of 8×1024-bit tiles on Sapphire Rapids.
- **Tile size:** 8 rows × 64 bytes (8 × 64 bytes = 512 bits per row conceptually, but stored as 1024-bit tile)
- **Registers:** 8 tiles (tmm0–tmm7), each 1024 bits
- **Configuration:** `ldtilecfg` sets tile row/column dimensions in a 512B config struct

**Key instructions:**
- `_tile_loadd / _tile_storestored` (load/store tiles with stride)
- `_tile_dpbssd` (tile dot product: signed bytes → signed dwords)
- `_tile_dpbsud` (unsigned byte × signed byte → signed dword)
- `_tile_dpbusd` (unsigned byte × signed byte → signed dword, variant)
- `_tile_dpbuud` (unsigned byte × unsigned byte → unsigned dword)
- `_tile_dpwssd` (word dot product)

**Comparison to GPU tensor cores:**
| Property | AVX-512 VNNI | AMX | NVIDIA A100 Tensor (FP32) |
|----------|---|---|---|
| Peak ops per cycle | 16 INT8 dotprod (4 elements each) | 16 INT8 dotprod per cycle (8×8 tiles) | 312 FP32 ops/cycle (across 108 cores) |
| Register-file size | 512 bits × 32 | 1024 bits × 8 | 256KB per SM |
| Latency | ~5 cycles | ~8 cycles (tile multiply) | ~1-4 cycles (tensor core) |
| Bandwidth to registers | ~10 TB/s theoretical | ~5 TB/s | ~1.6 TB/s per SM |

AMX is substantially lower throughput than modern GPU tensor cores but useful for latency-sensitive CPU-based inference where GPU context-switch overhead is prohibitive. See: Intel Optimization Manual for Sapphire Rapids, Chapter on AMX.

### 1.5 Frequency Throttling and License Levels

AVX-512 triggers frequency scaling on Intel Xeon CPUs. The CPU has "license levels":

**License Level 0 (base P-state):** No frequency reduction
**License Level 1 (AVX2, AVX):** ~90-95% of base frequency
**License Level 2 (AVX-512):** ~75-85% of base frequency

Modern Sapphire Rapids mitigates this via **Intel Speed Select Technology (SST)**, where the hypervisor can throttle non-AVX-512 cores while maintaining full frequency on AVX-512 cores. However, single-threaded code still experiences 15–25% frequency reduction on AVX-512.

**Implications:**
- FP32 scalar latency: ~4 cycles
- FP32 AVX-512 latency: ~4 cycles (no increase)
- FP32 AVX-512 throughput (8 lanes × 512-bit): 16 FP32/cycle @ 85% freq = 13.6 "scalar-equivalent" ops per cycle
- But measured wall-clock: slower than 8 scalar threads due to frequency loss

See: Intel Xeon Platinum Processor Performance Tuning Guidance; Agner Fog Vol 1 on frequency scaling.

---

## 2. MENTAL MODEL

```
                         AVX-512 Register File (Sapphire Rapids)
                                  (32 × 512-bit)
┌────────────────────────────────────────────────────────────────────┐
│                                                                    │
│  zmm0: [f32 f32 f32 f32 f32 f32 f32 f32 f32 f32 f32 f32 f32 f32 f32 f32]
│         0   1   2   3   4   5   6   7   8   9  10  11  12  13  14  15
│         └─────────── ymm0 (256b) ──────────┘
│         └── xmm0 (128b) ──┘
│
│  zmm1: [... 16 elements ...]
│  ...
│  zmm31: [... 16 elements ...]
│
│  Mask registers k0–k7 (each 64 bits, one bit per element):
│  k1: 1 0 1 1 0 1 0 1 1 1 0 1 1 0 1 0  (example for 16-element operation)
│
└────────────────────────────────────────────────────────────────────┘

         Execution (vpaddd zmm0{k1}, zmm1, zmm2):
         ┌─────────────────────────────────────────┐
         │ for (i = 0; i < 16; i++)               │
         │   if (k1 & (1 << i))                   │
         │     zmm0[i] = zmm1[i] + zmm2[i]        │
         │   // else zmm0[i] unchanged (merge)    │
         └─────────────────────────────────────────┘

Loaded from memory (256-bit aligned): [a0 a1 a2 ... a15] → ymm0
                                       [b0 b1 b2 ... b15] → ymm1
Broadcast scalar: scalar 5 → [5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5] in zmm2


           VNNI Dot Product (vpdpbusd):
           ┌─────────────────────────────────────┐
           │  a b c d │ (zmm1, unsigned bytes)   │
           │  e f g h │ (zmm2, signed bytes)     │
           └─────────────────────────────────────┘

           vpdpbusd zmm0, zmm1, zmm2

           zmm0[0] += a*e + b*f + c*g + d*h (32-bit signed result)
           zmm0[1] += ... (next 4 bytes from zmm1/zmm2)
           ...
           16 dot products per 512-bit operation

           Throughput: 1 per cycle (Sapphire Rapids)
           Latency: ~5 cycles
```

**Key insight:** AVX-512 masking is *element-scoped* predication, not branch-based. The CPU still computes the operation for masked-out elements (data hazards, register write stalling may occur) but doesn't *write* the result. This is why conditional vectorization with masks is often more efficient than scalar branches on modern CPUs.

---

## 3. PERFORMANCE LENS

### 3.1 Throughput vs. Latency

**FP32 Addition/Multiplication (AVX-512):**
- Throughput: 1 instruction per cycle (can execute multiple additions in flight)
- Latency: 4 cycles (result available for dependent operation after 4 cycles)
- 16 × 32-bit elements per instruction → 16 adds per cycle
- Peak FP32 throughput @ 2.0 GHz: 16 × 2.0G = 32 GFLOPS (single core, no frequency throttling)
- With 25% frequency reduction: 32 × 0.75 = 24 GFLOPS

**INT8 VNNI (vpdpbusd):**
- Throughput: 1 instruction per cycle
- Latency: ~5 cycles (slightly higher than FP32 due to fixed-point arithmetic)
- Instruction computes 16 × (4-element dot product of int8) = 16 × 4 = 64 INT8 operations per instruction
- Peak INT8 throughput @ 2.0 GHz: 64 × 2.0G = 128 GOPS (single core)

**Gather/Scatter Penalty:**
- `vpgatherdd zmm0, [rsi + vmm1*4]` (variable stride load)
- Throughput: 1 per ~6–8 cycles (due to cache port contention and address translation)
- Latency: ~7–12 cycles depending on cache hits
- **Lesson:** Avoid gathers in tight loops unless absolutely necessary. Prefer gather + blocked computation or streaming loads.

### 3.2 Memory Bandwidth vs. Compute Bandwidth

**Sapphire Rapids Memory Subsystem:**
- 8 memory channels × DDR5-4800 × 64-bit width = 384 GB/s peak bandwidth
- Single core with AVX-512 FP32: can sustain only 32 GFLOPS (with frequency throttling ~24 GFLOPS)
- Bytes per FLOP: 384 GB/s ÷ (32 GFLOPS × 4 bytes/FP32) = 3.0 bytes/FLOP for *all 60 cores*
- Single core: ~6.4 bytes/FLOP (384 GB/s ÷ 60 cores ÷ 24 GFLOPS)

**Implication:** AVX-512 is compute-bound only in highly optimized GEMM kernels with tight data reuse (L1/L2 cache). Most ML inference tasks (which feature large batch sizes and sparse computation patterns) are memory-bound.

### 3.3 Register Pressure and Instruction Scheduling

32 × 512-bit registers = 16 KB register file. For a GEMM kernel:
- Need 3 registers for load + compute: zmm0 (A), zmm1 (B), zmm2 (accum)
- But with software pipelining, need 4–6 iterations in flight
- Total: ~20 registers used
- Remaining: 12 registers for temporary permutations, masks, address arithmetic

**Lesson:** AVX-512 kernels must be carefully scheduled. Compiler auto-vectorization often wastes registers on redundant loads. Hand-written intrinsics (or inline assembly) can achieve 50–70% of theoretical peak. See Section 4 (Annotated Code).

---

## 4. ANNOTATED CODE

### 4.1 Scalar FP32 GEMM Kernel (Baseline)

```c
// Simple C = A × B (M×K × K×N → M×N), FP32
// 6 flops per element, 2.4 flops per byte (float is 4 bytes)
// This version: 1 FP32 multiply, 1 FP32 add per iteration
// Register pressure: 1 (accumulator)

void gemm_scalar(int M, int N, int K,
                 float *A, int ldA,
                 float *B, int ldB,
                 float *C, int ldC)
{
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float accum = C[i * ldC + j];  // Load C[i][j]
            for (int k = 0; k < K; k++) {
                accum += A[i * ldA + k] * B[k * ldB + j];  // 1 mul, 1 add
            }
            C[i * ldC + j] = accum;  // Store
        }
    }
}

// Perf: ~4 cycles latency (2 cycles mul + 2 cycles add)
//       1 FP per cycle throughput (dependent chain)
//       For M=N=K=100: 100*100*2*100 = 2M flops, ~2M cycles (1 GFLOP @ 1GHz)
```

**Lessons:**
- Scalar FP32 has 4-cycle latency
- Dependent chain limits throughput to ~1 FLOP per cycle
- No SIMD parallelism exploited

### 4.2 Auto-Vectorized AVX-512 (Compiler-Generated)

```c
void gemm_autovec_avx512(int M, int N, int K,
                         float *A, int ldA,
                         float *B, int ldB,
                         float *C, int ldC)
{
    // GCC/Clang with -O3 -march=skylake-avx512 will auto-vectorize the innermost loop
    // Assume K is multiple of 16 (for zmm width)
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float accum = C[i * ldC + j];
            for (int k = 0; k < K; k += 16) {
                // Compiler converts to:
                // zmm_accum = broadcast(accum)
                // zmm_a = load A[i][k:k+16] (64 bytes, aligned)
                // zmm_b = load B[k:k+16][j]  (from non-contiguous B)
                // zmm_prod = vmulps zmm_a, zmm_b
                // zmm_accum = vaddps zmm_accum, zmm_prod
                // Then: haddps reduce zmm_accum to scalar
                accum += A[i * ldA + k] * B[k * ldB + j];
            }
            C[i * ldC + j] = accum;
        }
    }
}

// Compiler-generated pseudocode (Intel intrinsics):
// __m512 v_accum = _mm512_broadcastss_ps(_mm_set_ss(accum));
// __m512 v_a = _mm512_loadu_ps(&A[i*ldA + k]);      // Load 16 floats
// __m512 v_b = ???                                   // Problem: B is strided!
// vmulps v_a, v_b
// vaddps v_accum, v_prod

// PROBLEM: B[k:k+16][j] are 64 bytes apart (ldB * 4 bytes).
// Compiler must emit gather: SLOW (~8 cycles latency).
```

**Lessons:**
- Auto-vectorization works only for unit-stride loads
- Non-contiguous access (B's columns) forces gather → 8x slowdown
- Horizontal reduction (haddps) to scalar destroys parallel benefit

### 4.3 Intrinsic-Based AVX-512 GEMM (Optimized)

```c
#include <immintrin.h>
#include <string.h>

// Tile-based GEMM: C[0:m][0:n] += A[0:m][0:k] × B[0:k][0:n]
// m, n: 16–64 (multiple of 16 for zmm width)
// k: arbitrary (inner loop)

void gemm_tile_avx512(int m, int n, int k,
                      const float *A, int ldA,
                      const float *B, int ldB,
                      float *C, int ldC)
{
    // Assertion: m and n are multiples of 16
    // Tile strategy: load 4×16 FP32 block of C, 4 rows of A, streaming across B

    // C block: 4 rows × 16 cols = 64 FP32 elements = 256 bytes
    __m512 accum[4];  // 4 zmm registers for C's 4 rows
    __m512 v_a;       // broadcast A element
    __m512 v_b;       // load B row

    // Initialize C accumulator from memory
    for (int i = 0; i < 4; i++) {
        accum[i] = _mm512_loadu_ps(&C[i * ldC]);  // Load C[i][0:16]
        // Line 1: Load 512 bits (16 × FP32) from C with 64-byte stride (ldC)
        // Latency: ~5 cycles (L2 cache hit typical)
        // Throughput: 1 per cycle
    }

    // Inner product loop: unroll over k by 2
    for (int j = 0; j < k; j += 2) {
        // Iteration 0: j += 0
        // ───────────────────────────────────────────────────────
        // A[0:4][j]: 4 elements (4 bytes each)
        // B[j:j+1][0:16]: 2 rows × 16 FP32 = 128 bytes

        __m512 b_row0 = _mm512_loadu_ps(&B[j * ldB]);
        // Line 2: Load B[j][0:16] (stride = ldB)
        // Assume B is column-major: ldB = K, so contiguous in memory
        // Latency: ~5 cycles (L1/L2), ~12 cycles (L3 miss)

        __m512 b_row1 = _mm512_loadu_ps(&B[(j+1) * ldB]);
        // Line 3: Load B[j+1][0:16], prefetch next B rows

        for (int i = 0; i < 4; i++) {
            // Update accumulator for row i:
            // accum[i] += A[i][j] * B[j][0:16] + A[i][j+1] * B[j+1][0:16]

            float a_ij   = A[i * ldA + j];       // Scalar load, 1 cycle latency
            float a_ij1  = A[i * ldA + j + 1];  // Scalar load, 1 cycle latency

            __m512 a_broadcast = _mm512_broadcastss_ps(
                _mm_set_ss(a_ij)
            );
            // Line 4: Broadcast A[i][j] to 16 FP32 elements
            // Latency: 1 cycle (replicate on execution port)

            __m512 prod = _mm512_mul_ps(a_broadcast, b_row0);
            // Line 5: accum[i] += A[i][j] × B[j][0:16]
            // vmulps instruction: FP32 multiply
            // Latency: 4 cycles
            // Throughput: 1 per cycle (×16 elements)

            accum[i] = _mm512_add_ps(accum[i], prod);
            // Line 6: accum[i] = accum[i] + prod
            // vaddps instruction: FP32 add
            // Latency: 4 cycles (parallel with multiplication of next iteration)

            a_broadcast = _mm512_broadcastss_ps(_mm_set_ss(a_ij1));
            prod = _mm512_mul_ps(a_broadcast, b_row1);
            accum[i] = _mm512_add_ps(accum[i], prod);
            // Lines 7–9: Second iteration of unroll
        }
    }

    // Store result back to C
    for (int i = 0; i < 4; i++) {
        _mm512_storeu_ps(&C[i * ldC], accum[i]);
        // Line 10: Store 512 bits (16 × FP32) to C
        // Latency: ~3 cycles (write buffer)
        // Throughput: 1 per cycle
    }
}

// Performance Analysis:
// ──────────────────────
// Register usage: 4 (accum) + 3 (b_rows + a_broadcast + prod) = 7 zmm (well within limit)
//
// Instruction stream for 1 iteration of inner loop (4 rows × 2 k-unroll):
// 2 × 512-bit loads (B rows):        2 cycles latency
// 2 × 4 scalar loads (A elements):   1 cycle latency (pipelined)
// 2 × 4 broadcasts:                  4 cycles total (1 per scalar load)
// 2 × 4 multiplies:                  4 × 4 = 16 cycles latency (but 1 per cycle throughput!)
// 2 × 4 adds:                        4 × 4 = 16 cycles latency (pipelined)
//
// With pipelining (out-of-order execution):
// – Iteration 0: load B[j], B[j+1]   (cycles 0–5)
// – Iteration 0: broadcast A[*][j]   (cycles 6–9)
// – Iteration 0: mul                 (cycles 10–13)
// – Iteration 0: add                 (cycles 14–17)
// – Iteration 1: load B[j+2], B[j+3] (cycles 0–5, overlapped!)
// ...
//
// With unroll factor 2: 1 mul + 1 add per k-iteration per row
// Total: (4 rows × 2 k-unroll × 2 ops) = 16 ops per inner iteration
// If we unroll k by 2 and have 4 rows, and k = 1024:
// – 512 iterations of unrolled loop
// – 512 × 16 ops = 8192 ops = 2 × 4096 multiplications + 4096 additions
// – Assuming 2.0 GHz, 1 ops/cycle per port (optimistic with pipelining):
//   8192 ops ÷ 4 cycles per (mul+add chain) ≈ 2048 cycles minimum
// – Actual time: ~3000–4000 cycles (due to memory stalls, port contention)
//
// Peak GFLOPS (m=16, n=16, k=1024, 2.0 GHz, no frequency throttling):
// 16 × 16 × 2 × 1024 / 3500 cycles ÷ 2.0G = 16 × 16 × 2048 / 3500 / 2G
//                                          ≈ 1.17 TFLOPS (wait, that's wrong)
// Actually: (4 rows × 16 cols × 1024 depth) = 65536 FLOPs
//           65536 FLOPs ÷ 4000 cycles = 16.4 FLOP/cycle (with 16 parallel FP units = realistic)
//           16.4 FLOP/cycle × 2.0 GHz = 32.8 GFLOPS (single core, matches AVX-512 peak)
```

**Key insights from this code:**
1. **Blocking:** Processing 4×16 tile fits in registers; avoids reloading B rows
2. **Unrolling k by 2:** Hides B row load latency with computation
3. **Software pipelining:** Loads of iteration j+1 overlap with computation of iteration j
4. **Broadcast latency:** Scalar A[i][j] must be broadcast; 1 cycle overhead per element (acceptable because we have 16 parallel units)
5. **Memory access pattern:** A is row-major (ldA = K), B is column-major (ldB = K); both have unit stride in inner loop
6. **Frequency throttling:** With AVX-512, actual clock might be 1.5 GHz instead of 2.0 GHz → 24 GFLOPS instead of 32 GFLOPS

### 4.4 VNNI INT8 GEMM Kernel

```c
#include <immintrin.h>

// C[i][j] += sum_k A[i][k] * B[k][j]
// A, B: int8
// C: int32 (accumulator)
// Each vpdpbusd instruction: 4-element dot product of int8 → int32
// zmm can hold 16 × 4-element dotproducts = 64 int8 muls per instruction

void gemm_int8_vnni_avx512(int m, int n, int k,
                           const int8_t *A, int ldA,
                           const int8_t *B, int ldB,
                           int32_t *C, int ldC)
{
    // Tile: 4 rows of C (each row is 16 int32 = 64 bytes)
    // Load 4×16 int32 block of C

    __m512i accum[4];

    for (int i = 0; i < 4; i++) {
        // Load C[i][0:16] as 16 × int32
        accum[i] = _mm512_loadu_si512((__m512i *)&C[i * ldC]);
        // Line 1: Load 64 bytes (16 × int32)
        // Latency: 5 cycles (L2 hit)
    }

    // Inner loop: process 4 int8 elements of A per iteration
    for (int j = 0; j < k; j += 4) {
        // Process A[0:4][j:j+4] × B[j:j+4][0:16] → accumulate to C

        // Load 4 rows × 4 elements of A (32 bytes)
        __m128i a_row[4];  // 4 × xmm (128-bit each)
        for (int i = 0; i < 4; i++) {
            a_row[i] = _mm_loadu_si32((int32_t *)&A[i * ldA + j]);
            // Load A[i][j:j+4] as 4 × int8 in lower 32 bits of xmm
            // Latency: 5 cycles
        }

        // Load 4 rows of B (B[j:j+4][0:16], each row 16 bytes)
        __m128i b_row[4];  // 4 × xmm (each is 16 int8 elements)
        for (int jj = 0; jj < 4; jj++) {
            b_row[jj] = _mm_loadu_si128((__m128i *)&B[(j + jj) * ldB]);
            // Load B[j+jj][0:16] as 128-bit (16 int8 elements)
            // Latency: 5 cycles
        }

        // Convert XMM to ZMM (broadcast)
        __m512i a_zmm[4];
        for (int i = 0; i < 4; i++) {
            a_zmm[i] = _mm512_cvtepi32_epi32(_mm512_castsi128_si512(a_row[i]));
            // Convert to zmm (fill with zeros in upper lanes)
            // But we need to broadcast each int8 of a_row[i]
            // Actually: a_row[i] = [a_i_j, a_i_{j+1}, a_i_{j+2}, a_i_{j+3}, ...]
            // We want: zmm = [a_i_j, a_i_j, ..., a_i_{j+1}, a_i_{j+1}, ...]
            // Use _mm512_broadcast_i32x4 or manual expansion
            // Simplified: use scalar approach below
        }

        // Simplified approach: process one row at a time
        for (int i = 0; i < 4; i++) {
            // accum[i] += sum(A[i][j:j+4] * B[j:j+4][0:16])

            // Load A[i][j:j+4] as 4 int8 values
            int8_t *a_ptr = (int8_t *)&A[i * ldA + j];
            int8_t a_vals[4] = {a_ptr[0], a_ptr[1], a_ptr[2], a_ptr[3]};
            // Scalar loads: 1 cycle latency per byte (pipelined)

            for (int jj = 0; jj < 4; jj++) {
                // accum[i] += A[i][j+jj] * B[j+jj][0:16]

                // Broadcast A[i][j+jj] to 64 int8 elements in zmm
                __m512i a_broadcast = _mm512_broadcastb_epi8(
                    _mm_cvtsi32_si128(a_vals[jj])
                );
                // Latency: 1 cycle
                // Throughput: 1 per cycle

                // Load B[j+jj][0:16] as 128 bits (16 int8)
                // Expand to 512 bits (replicate 4 times)
                __m128i b_xmm = _mm_loadu_si128((__m128i *)&B[(j + jj) * ldB]);
                __m512i b_zmm = _mm512_castsi128_si512(b_xmm);
                // Note: this only fills lower 128 bits; need expansion

                // Use vpdpbusd: (signed-byte)A × (unsigned-byte)B → int32
                // Actually, vpdpbusd takes signed × unsigned
                // Assume A is signed int8, B is unsigned int8
                // Or use vpdpwssd for word-width

                // For simplicity, assume 16 separate int8 elements in B:
                // B[j+jj][0], ..., B[j+jj][15]
                // We need to compute:
                // accum[i][0] += A[i][j+jj] * B[j+jj][0]
                // accum[i][1] += A[i][j+jj] * B[j+jj][1]
                // ...
                // accum[i][15] += A[i][j+jj] * B[j+jj][15]

                // Use pmaddubsw or vpdpbusd:
                // vpdpbusd requires 4 int8 × 4 int8 → 1 int32
                // So we need to process B in 4-element chunks

                for (int b_chunk = 0; b_chunk < 16; b_chunk += 4) {
                    // Broadcast A[i][j+jj] (same for all 4 B elements)
                    // Load B[j+jj][b_chunk:b_chunk+4]

                    __m512i a_expand = _mm512_set1_epi32(
                        (a_vals[jj] << 24) | (a_vals[jj] << 16) |
                        (a_vals[jj] << 8) | a_vals[jj]
                    );
                    // Line 2: Expand A[i][j+jj] to 4 copies in each 32-bit lane
                    // Latency: 1 cycle

                    __m512i b_chunk_zmm = _mm512_cvtepu8_epi32(
                        _mm_loadu_si32((int32_t *)&B[(j+jj)*ldB + b_chunk])
                    );
                    // Line 3: Load B[j+jj][b_chunk:b_chunk+4], zero-extend to int32
                    // Latency: 5 cycles

                    // vpdpbusd: dot product of bytes (signed × unsigned → signed dword)
                    // But we need to emulate with multiplication + addition

                    __m512i prod = _mm512_mullo_epi32(a_expand, b_chunk_zmm);
                    // Line 4: Multiply (treating bytes as int32 now)
                    // Latency: ~3 cycles

                    accum[i] = _mm512_add_epi32(accum[i], prod);
                    // Line 5: Add to accumulator
                    // Latency: 1 cycle (add)
                }
            }
        }
    }

    // Store result
    for (int i = 0; i < 4; i++) {
        _mm512_storeu_si512((__m512i *)&C[i * ldC], accum[i]);
        // Line 6: Store 64 bytes (16 × int32)
        // Latency: 3 cycles
    }
}

// Optimized version using vpdpbusd directly:
// For each k-iteration: load 4 B rows (each 16 int8 bytes)
// Pack 4 × 4-element dotproducts into 1 zmm (using VNNI)
// Total throughput: 1 vpdpbusd per cycle = 64 int8 muls per cycle per core

// Performance:
// m=4 rows, n=16 cols (64 int32 accumulators)
// k=1024 depth
// Total: 4 × 16 × 2 × 1024 = 131,072 int8 multiply-accumulates
// With vpdpbusd: 131,072 ÷ 64 int8-ops per instruction = 2048 instructions
// At 1 instruction/cycle: ~2048 cycles
// Peak throughput @ 2.0 GHz: 2048 instructions × 64 ops ÷ 2.0G = 65.5 GOPS
// (Single core; scales linearly with number of cores)
```

**Key VNNI observations:**
1. `vpdpbusd` computes 4-element int8 dot product → int32 in 1 instruction
2. Each 512-bit zmm contains 16 such 4-element groups
3. Throughput: 1 instruction/cycle → 64 INT8 macs per cycle per core
4. 8 cores doing VNNI: 512 INT8 macs/cycle → 1024 GOPS @ 2.0 GHz

---

## 5. EXPERT INSIGHT

### 5.1 When Vectorization Loses (Gather/Scatter Paradox)

Many engineers assume vectorization is always faster. It's not:

```c
// Example: sparse matrix × dense vector (SpMV)
// Compressed Sparse Row (CSR) format
// row_ptr[i], col_idx[k], values[k]

void spmv_scalar(int m, const int *row_ptr, const int *col_idx,
                 const float *values, const float *x, float *y)
{
    for (int i = 0; i < m; i++) {
        float accum = 0.0f;
        for (int k = row_ptr[i]; k < row_ptr[i+1]; k++) {
            int j = col_idx[k];  // Variable column index
            accum += values[k] * x[j];  // x[j] gather
        }
        y[i] = accum;
    }
}

// Scalar version:
// – Load j = col_idx[k] (1 cycle)
// – Load x[j] from memory (5–200 cycles depending on cache)
// – Multiply (4 cycles)
// – Add (4 cycles)
// – Total: ~14 cycles per nonzero (optimistic with cache hits)

// Vectorized version (attempt):
void spmv_avx512_gather(int m, const int *row_ptr, const int *col_idx,
                        const float *values, const float *x, float *y)
{
    for (int i = 0; i < m; i++) {
        __m512 accum = _mm512_setzero_ps();
        for (int k = row_ptr[i]; k < row_ptr[i+1]; k += 16) {
            // Load 16 column indices
            __m512i col_indices = _mm512_loadu_si512((__m512i *)&col_idx[k]);

            // Gather 16 values from x[col_idx[k:k+16]]
            __m512 x_vals = _mm512_i32gather_ps(
                col_indices, (const void *)x, 4
            );
            // gather_ps: load x[col_indices[i]] for each i
            // Latency: ~7 cycles (best case, all in L1)
            //          ~200 cycles (worst case, L3 misses + random access)

            __m512 vals = _mm512_loadu_ps(&values[k]);
            __m512 prods = _mm512_mul_ps(vals, x_vals);
            accum = _mm512_add_ps(accum, prods);
        }
        // Horizontal reduction of accum to scalar
        float result = _mm512_reduce_add_ps(accum);
        y[i] = result;
    }
}

// Vectorized version: SLOWER because:
// – Gather has ~7–200 cycle latency (random memory access)
// – Scalar version can hide latency via out-of-order execution if k loop is long enough
// – If sparsity pattern is irregular (many cache misses), scalar version wins
```

**Expert lesson:** Vectorization is not a free lunch. Use it when:
1. Unit-stride memory access (no gather)
2. Dense computations (high compute intensity)
3. Data reuse (L1/L2 cache friendly)

Avoid it for:
1. Sparse access patterns
2. Irregular control flow (branches that don't vectorize)
3. Streaming writes (high memory bandwidth needed, limited by L3)

### 5.2 Masking Overhead vs. Branch Overhead

```c
// Scalar version: conditional based on age < 18
for (int i = 0; i < n; i++) {
    if (ages[i] < 18) {
        process_minor(data[i]);  // Branch, potentially mispredicted
    } else {
        process_adult(data[i]);
    }
}

// AVX-512 masked version
for (int i = 0; i < n; i += 16) {
    __m512i ages_zmm = _mm512_loadu_si512((__m512i *)&ages[i]);
    __mmask16 mask = _mm512_cmplt_epi32_mask(ages_zmm, _mm512_set1_epi32(18));
    // mask[j] = 1 if ages[i+j] < 18, else 0

    // Process both paths in parallel (masked)
    // process_minor_zmm (masked execution)
    // process_adult_zmm (masked execution)

    // Store result
    _mm512_mask_storeu_epi32(result + i, mask, minor_result);
    _mm512_mask_storeu_epi32(result + i, ~mask, adult_result);
}

// Performance trade-off:
// Scalar: branch misprediction ~20 cycles, but only 1 element per iteration
// Masked: no misprediction, but must compute both paths (wasted computation)
//
// Break-even: if >50% of elements take the same branch,
// scalar (with good prediction) is faster.
// If <50% (alternating or random), masked version is faster.
```

**Expert truth:** Masking is NOT branch elimination. It's branch elimination + wasted computation. Use it only when:
1. Branch is unpredictable (random or fine-grained alternation)
2. Both paths are cheap to compute
3. Memory bandwidth for stores is not contended

### 5.3 Frequency Throttling: Myth vs. Reality

Common belief: "AVX-512 is slow because it throttles frequency by 25%."

Reality is more nuanced:

```
Peak FP32 throughput (theoretical):
– Scalar (no AVX): 2.0 GHz × 1 FP/cycle = 2 GFLOPS
– AVX-512 (no throttle): 2.0 GHz × 16 FP/cycle = 32 GFLOPS
– AVX-512 (with 25% throttle): 1.5 GHz × 16 FP/cycle = 24 GFLOPS

Speedup of AVX-512 over scalar: 24 ÷ 2 = 12×

So despite throttling, AVX-512 is 12× faster for data-parallel code.
```

The confusion arises because engineers compare *wall-clock time*:
- Scalar code: 1000 cycles @ 2.0 GHz = 500 ns
- AVX-512 code: 100 cycles @ 1.5 GHz (throttled) = 67 ns
- AVX-512 is 7.5× faster in wall-clock time (despite lower frequency)

**Expert insight:** Frequency throttling is irrelevant compared to parallelism gain. The problem with AVX-512 is not throttling; it's:
1. Memory bandwidth limits (most ML inference)
2. Register pressure in complex kernels
3. Gather/scatter performance
4. Data layout requirements (forcing cache misses)

### 5.4 Integer Precision Tradeoffs in ML Inference

For INT8 inference, understanding precision is critical:

```c
// FP32 → INT8 quantization:
// x_int8 = round(x_fp32 * scale) + zero_point
// x_fp32 = (x_int8 - zero_point) / scale

// But INT8 has range [-128, 127].
// Quantization error: ±0.5 / scale (uniform distribution)

// Example: ResNet50 activations
// Min: 0.0, Max: 6.0 (ReLU-bounded)
// scale = 6.0 / 255 = 0.0235
// Quantization error: ±0.0118 (0.2% of range)

// With INT8 VNNI GEMM:
// Matrix multiply errors compound: (layer 1 error) × (num accumulations)
// For 1000-element dot product: 1000 × 0.0118 = ~11.8 (huge!)

// Solution: Fixed-point scaling with 32-bit accumulation
// accum_int32 = sum_k (A_int8[k] × B_int8[k])  // 32-bit result (avoids overflow)
// accum_int8 = (accum_int32 >> 8) + bias_int8  // Scale down, quantize

// VNNI instruction: vpdpbusd outputs int32, so we're safe from overflow
```

**Expert lesson:** INT8 is not FP32 with smaller memory. It requires understanding:
1. Scale factors (typically per-channel or per-tensor)
2. Accumulator width (32-bit minimum for >~256 element dot products)
3. Symmetry vs. asymmetric quantization (zero-point handling)

### 5.5 Data Layout: AoS vs. SoA vs. AoSoA

Structure of Arrays (SoA) is crucial for SIMD efficiency:

```c
// Array of Structures (AoS): poor cache behavior for SIMD
struct Vertex { float x, y, z; };
Vertex verts[1000000];

// Processing (bad for vectorization):
for (int i = 0; i < n; i++) {
    verts[i].x *= transform[0];
    verts[i].y *= transform[1];
    verts[i].z *= transform[2];
}
// Memory layout: [x0 y0 z0 x1 y1 z1 x2 y2 z2 ...]
// Load: x0 y0 z0 x1, store x0', skipping y0 z0 → cache miss

// Structure of Arrays (SoA): excellent for vectorization
struct VertexSoA {
    float xs[1000000];
    float ys[1000000];
    float zs[1000000];
};

// Processing (good for vectorization):
for (int i = 0; i < n; i += 16) {
    __m512 xs = _mm512_loadu_ps(&verts.xs[i]);
    __m512 ys = _mm512_loadu_ps(&verts.ys[i]);
    __m512 zs = _mm512_loadu_ps(&verts.zs[i]);
    // Unit stride, prefetchable, cache-efficient
}

// Hybrid: Array of Structures of Arrays (AoSoA)
struct VertexAoSoA {
    struct {
        float xs[16];
        float ys[16];
        float zs[16];
    } blocks[1000000 / 16];
};

// Good cache locality (blocks are contiguous) + good vectorization
```

**For transformer inference (SoA strongly preferred):**

```c
// Traditional token structure (AoS):
struct Token {
    float embeddings[768];  // This is already AoS!
    int token_id;
    int position;
};
Token tokens[batch_size];

// Memory: [emb0[0..767] tok_id pos emb1[0..767] tok_id pos ...]
// Poor for attention matrix multiply

// Transformer SoA layout:
struct TransformerBatch {
    float all_embeddings[batch_size][seq_len][768];  // [batch][seq][hidden]
    int token_ids[batch_size][seq_len];
};

// When computing attention, access pattern:
// Query: [batch][seq][768] → contiguous per-sequence
// Key: [batch][seq][768] → contiguous per-sequence
// QK^T: [batch][seq][seq] → each element is 768-element dot product
//
// Each dot product loads 768 bytes sequentially (good prefetch)
// Stores 1 element to attention matrix
// 768 bytes : 1 element = 768:1 compute intensity (very high)
```

---

## 6. BENCHMARK / MEASUREMENT

### 6.1 Measuring AVX-512 Peak Throughput (GFLOPS)

```bash
# Compile with GCC:
gcc -O3 -march=skylake-avx512 -fno-tree-vectorize bench_avx512.c -o bench

# Disable frequency throttling (requires root):
echo "performance" | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor

# Pin to specific CPU:
taskset -c 0 ./bench

# Alternative: use likwid-bench (pre-built microbenchmark suite)
# https://github.com/RRZE-HPC/likwid
likwid-bench -t avx512 -w S0:10MB:4
```

### 6.2 Measuring Memory Bandwidth (Stream Benchmark)

```c
// Simplified STREAM Triad (y += a*x)
#include <time.h>
#include <stdio.h>

#define N 10000000

int main() {
    float a = 2.5;
    float x[N], y[N];

    // Initialize
    for (int i = 0; i < N; i++) {
        x[i] = 1.0f;
        y[i] = 2.0f;
    }

    // Warm-up
    for (int i = 0; i < 100; i++) {
        for (int j = 0; j < N; j++) {
            y[j] += a * x[j];
        }
    }

    // Measure
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);

    for (int i = 0; i < 100; i++) {
        for (int j = 0; j < N; j++) {
            y[j] += a * x[j];
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &end);

    double elapsed = (end.tv_sec - start.tv_sec) +
                     (end.tv_nsec - start.tv_nsec) / 1e9;

    // Bandwidth: 3 arrays × N × 4 bytes × 100 iterations / elapsed time
    // (read: 2 arrays, write: 1 array)
    double bytes = 3 * N * 4 * 100;
    double bandwidth_gbs = bytes / elapsed / 1e9;

    printf("Bandwidth: %.1f GB/s\n", bandwidth_gbs);
    printf("Expected: 384 GB/s (Sapphire Rapids 8-channel DDR5-4800)\n");

    return 0;
}

// Compile and run:
// gcc -O3 -march=native stream.c -o stream
// taskset -c 0-59 ./stream  // Use all cores
```

Expected results on Sapphire Rapids:
- Single core: ~6 GB/s (DDR5-4800 with latency, serialization)
- 60 cores: ~300–350 GB/s (peak 384 GB/s, limited by NUMA access)

### 6.3 Measuring Instruction Latency and Throughput

```c
#include <immintrin.h>
#include <time.h>
#include <stdio.h>

// Latency measurement: dependent chain
// y = f(f(f(...f(x))))
// Count iterations / time to infer latency

int measure_latency() {
    __m512 v = _mm512_set1_ps(1.0f);

    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);

    for (int i = 0; i < 1000000; i++) {
        v = _mm512_mul_ps(v, _mm512_set1_ps(1.00001f));  // 1.00001 to avoid denormals
        // FP32 multiply latency: 4 cycles
        // 1000000 iterations × 4 cycles = 4M cycles
    }

    clock_gettime(CLOCK_MONOTONIC, &end);
    double elapsed = (end.tv_sec - start.tv_sec) +
                     (end.tv_nsec - start.tv_nsec) / 1e9;

    // Latency = (4M cycles) / (2.0 GHz) = 2000 ns
    double latency_ns = elapsed * 1e9 / 1000000;

    printf("Measured latency: %.1f ns\n", latency_ns);
    printf("Expected (4 cycles @ 2.0 GHz): 2.0 ns per iteration, 4 * 2 = 8 ns latency\n");

    return (int)v[0];  // Use result to prevent optimization
}

// Throughput measurement: independent operations
// y1 = f(x1), y2 = f(x2), ... (no dependencies)
// CPU can execute all in parallel

int measure_throughput() {
    __m512 v0 = _mm512_set1_ps(1.0f);
    __m512 v1 = _mm512_set1_ps(1.0f);
    __m512 v2 = _mm512_set1_ps(1.0f);
    __m512 v3 = _mm512_set1_ps(1.0f);

    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);

    for (int i = 0; i < 1000000; i++) {
        // 4 independent multiplies, can execute in parallel
        v0 = _mm512_mul_ps(v0, _mm512_set1_ps(1.00001f));
        v1 = _mm512_mul_ps(v1, _mm512_set1_ps(1.00001f));
        v2 = _mm512_mul_ps(v2, _mm512_set1_ps(1.00001f));
        v3 = _mm512_mul_ps(v3, _mm512_set1_ps(1.00001f));
    }

    clock_gettime(CLOCK_MONOTONIC, &end);
    double elapsed = (end.tv_sec - start.tv_sec) +
                     (end.tv_nsec - start.tv_nsec) / 1e9;

    // 4M iterations, 1 throughput per cycle = 4M cycles
    // Time = 4M cycles / 2.0 GHz = 2 seconds
    double throughput_us = elapsed / 4;

    printf("Measured throughput: %.3f us per multiply\n", throughput_us);
    printf("Expected (1 per cycle @ 2.0 GHz): 0.5 ns per multiply\n");

    return (int)v0[0] + (int)v1[0] + (int)v2[0] + (int)v3[0];
}
```

### 6.4 Measuring VNNI Throughput

```c
#include <immintrin.h>
#include <time.h>
#include <stdio.h>

int measure_vnni_throughput() {
    // vpdpbusd: 1 instruction per cycle throughput
    // zmm0 += dot(zmm1[4-byte chunks], zmm2[4-byte chunks])

    __m512i acc[8];  // 8 independent accumulators
    __m512i a[8], b[8];

    // Initialize
    for (int i = 0; i < 8; i++) {
        acc[i] = _mm512_setzero_si512();
        a[i] = _mm512_set1_epi32(0x01010101);  // 4 × int8, each = 1
        b[i] = _mm512_set1_epi32(0x02020202);  // 4 × int8, each = 2
    }

    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);

    for (int i = 0; i < 1000000; i++) {
        // 8 independent vpdpbusd instructions (no dependencies between accumulators)
        acc[0] = _mm512_dpbusd_epi32(acc[0], a[0], b[0]);
        acc[1] = _mm512_dpbusd_epi32(acc[1], a[1], b[1]);
        acc[2] = _mm512_dpbusd_epi32(acc[2], a[2], b[2]);
        acc[3] = _mm512_dpbusd_epi32(acc[3], a[3], b[3]);
        acc[4] = _mm512_dpbusd_epi32(acc[4], a[4], b[4]);
        acc[5] = _mm512_dpbusd_epi32(acc[5], a[5], b[5]);
        acc[6] = _mm512_dpbusd_epi32(acc[6], a[6], b[6]);
        acc[7] = _mm512_dpbusd_epi32(acc[7], a[7], b[7]);
    }

    clock_gettime(CLOCK_MONOTONIC, &end);
    double elapsed = (end.tv_sec - start.tv_sec) +
                     (end.tv_nsec - start.tv_nsec) / 1e9;

    // 8M instructions (8 per iteration × 1M iterations)
    // 1 per cycle: 8M cycles / 2.0 GHz = 4 seconds
    double throughput_gops = (1000000 * 8 * 64) / elapsed / 1e9;  // 64 int8 ops per instruction

    printf("Measured VNNI throughput: %.1f GOPS\n", throughput_gops);
    printf("Expected (1 instr/cycle × 64 ops/instr × 2.0 GHz): 128 GOPS\n");

    return (int)acc[0][0];
}
```

---

## 7. ML SYSTEMS RELEVANCE

### 7.1 INT8 Quantization for Transformer Inference

Modern LLM inference servers (vLLM, TensorRT, TVM) use INT8 for:
1. **Activation quantization:** Hidden states → INT8
2. **Weight quantization:** Model parameters → INT8 (static, pre-computed)
3. **Accumulation:** GEMM output → INT32, then scaled back to INT8

**Sapphire Rapids vs. GPU comparison:**

| Task | Sapphire Rapids VNNI | NVIDIA A100 Tensor (FP8) | Speedup |
|------|---|---|---|
| Single token (seq=1) | 512 GOPS × 1 core | 312 TFLOPS ÷ 108 = 2.9 GFLOPS per core | GPU 6× faster |
| Batch 64 (seq=1) | 512 GOPS × 60 cores = 30.7 TOPS | 312 TFLOPS | GPU 10× faster |
| KV-cache reuse | None (must reload cache each time) | With tensor caching in SRAM | GPU better |

**When CPU is preferred:**
- Batch size = 1 (latency-sensitive, high-frequency trading, robotics)
- Small model size (fits in L3 cache, ~10–50M parameters)
- Strict latency budget (<10 ms per token)
- Low-cost inference (CPU is cheaper than GPU cluster)

### 7.2 Attention Operator Optimization

Transformer attention: Q @ K^T (shape: [batch*seq][seq] with [batch*seq][d] matrices)

```c
// Attention: softmax(Q @ K^T) @ V
// Q: [B×S][D]  (B=batch, S=seq_len, D=hidden_dim)
// K: [B×S][D]
// V: [B×S][D]

// GEMM 1: Q @ K^T → QK [B×S][B×S]
// With transformer SoA layout:
// Q.shape = [batch*seq][hidden_dim] (contiguous in memory)
// K.shape = [batch*seq][hidden_dim] (contiguous)
// Output shape: [batch*seq][batch*seq]

// Each element of QK = sum_d Q[i][d] * K[j][d]
// This is an all-pairs dot product: 768-element dot products (for D=768)

// Optimal kernel:
for (int i = 0; i < batch_seq; i++) {
    for (int j = 0; j < batch_seq; j += 16) {
        // Load 16 K rows [j:j+16][:]
        __m512 q = broadcast(Q[i][0:16]);  // Load 16 elements of Q[i]

        __m512 k_tiles[16];
        for (int kk = 0; kk < 16; kk++) {
            k_tiles[kk] = load K[j+kk][0:16];
        }

        __m512 accum = 0;
        for (int d = 0; d < D; d += 16) {
            // Q[i][d:d+16] dot K[j:j+16][d:d+16]
            // This is 16×16 = 256 dot product terms

            // Micro-kernel for 16×16 block of attention
            __m512 q_tile = load Q[i][d:d+16];
            for (int kk = 0; kk < 16; kk++) {
                __m512 k_val = load K[j+kk][d:d+16];
                __m512 prod = q_tile * k_val;
                accum[kk] += reduce(prod);  // Horizontal sum
            }
        }
    }
}

// Register pressure: high (need to keep 16 partial sums)
// Memory access: Q is read 16× (amortized 64 bytes per output)
//                K is read once (64 bytes per output, but 16 outputs)
//                Total: 128 bytes loaded per output (512-bit = 64 bytes)
//                Compute intensity: 768 bytes : 1 output = 768:1
```

**Optimization: fused softmax + attention:**

```c
// Instead of:
// 1. Compute QK
// 2. Apply softmax
// 3. Compute (softmax(QK)) @ V

// Fused: for each output position, compute softmax incrementally
// This reduces memory traffic (don't write intermediate QK matrix)
// Sapphire Rapids L3: 140 MB, can fit attention matrix for seq_len=1024
// But fused approach avoids writing+reading, saves bandwidth
```

### 7.3 Quantization-Aware Training (QAT) with AVX-512 FP8

```c
// Forward pass: FP32 compute, FP8 storage
// Backward pass: FP32 gradient accumulation, FP8 weight storage

// Simulated quantization for backward pass:
// loss = ||target - f_quantized(x)||^2
// df/dw_fp32 is computed in FP32, then quantized to FP8 for storage

__m512 quantize_fp32_to_fp8_sat(const __m512 &v, float scale) {
    // Clamp to FP8 range: [-128, 127] / scale
    __m512 scaled = _mm512_mul_ps(v, _mm512_set1_ps(scale));
    __m512 clamped = _mm512_max_ps(
        _mm512_min_ps(scaled, _mm512_set1_ps(127.0f)),
        _mm512_set1_ps(-128.0f)
    );
    // Convert to int8 (would use FP8 hardware if available)
    return clamped;  // Simplified
}

// During training, quantization is differentiable:
// dL/dv_quantized = dL/dv_fp32 (backprop through quantization is identity)
// But loss is computed on quantized version, so gradient reflects quantization loss
```

---

## 8. PhD QUALIFIER QUESTIONS

### Question 8.1
Explain the relationship between AVX-512's frequency throttling and actual performance improvement over scalar code. If a scalar loop runs at 2.0 GHz and AVX-512 code runs at 1.5 GHz (25% throttle), how many elements must fit into a 512-bit SIMD register for AVX-512 to be faster? Assume both use a 4-cycle FP32 multiply.

**Model Answer:**
- Scalar: 1 element per cycle
- AVX-512: N elements per cycle (where N = 512 bits ÷ element width)
- FP32: N = 16 elements per cycle
- Speedup = (16 × 1.5 GHz) ÷ (1 × 2.0 GHz) = 12×
- Threshold for breakeven: 2.0 GHz ÷ 1.5 GHz = 1.33 elements needed
- Since FP32 = 16 elements, AVX-512 is always faster (even with throttle)
- For INT8: 64 elements per instruction, speedup = (64 × 1.5) ÷ 2.0 = 48×
- Threshold: still breakeven at 1.33 elements, achieved with just 1 INT8 element
- **Conclusion:** Throttling is irrelevant; parallelism dominates. AVX-512 is always faster for data-parallel code, even with 25% frequency reduction.

### Question 8.2
The VNNI instruction `vpdpbusd` computes a dot product of 4-element signed byte × unsigned byte → signed dword. Explain why 4-element groups are chosen (not 8 or 16), and calculate the peak INT8 throughput on Sapphire Rapids with all 60 cores.

**Model Answer:**
- INT8 range: [-128, 127] × [0, 255] → max product = 32,385 (requires 16 bits)
- 4 products = 4 × 32,385 = 129,540 (requires ~18 bits, fits in signed 32-bit with margin)
- 8 products = 262,080 (exceeds 32 bits, would overflow)
- **4-element groups ensure no overflow in a single operation**
- Peak throughput: vpdpbusd = 1 instruction/cycle
- Each instruction: 16 × 4-element dot products (512-bit ÷ 32-bit per result)
- Each dot product: 4 INT8 multiplies = 64 INT8 operations per instruction
- Sapphire Rapids: 60 cores × 64 INT8 ops/cycle × 2.0 GHz = 7.68 TOPS of INT8
- **Note:** With frequency throttling on AVX-512, actual clock might be 1.5 GHz → 5.76 TOPS

### Question 8.3
Explain the memory bandwidth requirements for dense INT8 matrix multiplication on Sapphire Rapids. If a GEMM kernel achieves 1.0 TOPS of INT8 performance, how many bytes per second must be loaded from memory? Compare to the theoretical 384 GB/s DDR5-4800 bandwidth.

**Model Answer:**
- INT8 matrix multiply: C += A × B (all INT8)
- Assume square matrices: n×n = n×k × k×n
- Bytes read: 2 × n × k (A and B matrices)
- Operations: n^2 × k INT8 multiplies (but VNNI groups 4, so n^2 × k ÷ 4 "real" ops... actually n^2 × k ops since each int8 is 1 op)
- Wait, let me reconsider: each int8 multiplication is 1 integer operation, but we're computing dot products
- For GEMM: compute n × n dot products, each of length k
- Total operations: 2 × n × k (n = size of output matrix, k = depth)
- Bytes read: 2 × n × k bytes (A is n×k, B is k×n)
- **Bytes per FLOP: 1 byte per INT8 operation = 1 Byte/op**
- At 1.0 TOPS: 1.0 × 10^12 INT8 ops/s × 1 byte/op = 1.0 TB/s
- This is 1000 GB/s > 384 GB/s available
- **Conclusion:** Achieving 1.0 TOPS on Sapphire Rapids requires 1 TB/s of memory bandwidth, which is impossible. Realistic: ~300 GB/s ÷ 1 byte/op = 300 GOPS (per-system, not per-core).

### Question 8.4
Write pseudocode for an optimized AVX-512 FP32 matrix multiply kernel that processes a 4×16 tile of C, and explain which loads have the highest latency impact. What is the bottleneck (memory bandwidth, register pressure, or instruction latency)?

**Model Answer:**
See Annotated Code Section 4.3 (gemm_tile_avx512). Pseudocode:

```
for j in [0, k):
  load B[j][0:16] into zmm  (5-cycle latency)
  for i in [0, 4):
    load A[i][j] (scalar, 1 cycle)
    broadcast A[i][j] (1 cycle)
    multiply, add (8 cycles total latency)
  store result
```

Bottleneck analysis:
- Register pressure: 7 zmm registers (4 C accumulators + 3 temp) ✓ sufficient
- Memory bandwidth:
  - A: 4 rows × k elements = 16k bytes
  - B: k rows × 16 elements = 64k bytes
  - C: 4 rows × 16 elements = 256 bytes
  - Total: 80k + 256 bytes for 4×16×k operations = 80k bytes for 1024k int8-equivalent ops
  - Bytes per operation: 80k ÷ 1024k = 0.078 bytes/op
  - At 32 GFLOPS: 32G × 0.078 = 2.5 GB/s (well below 384 GB/s) ✓ compute-bound
- Instruction latency:
  - B load: 5 cycles
  - A load + broadcast: 2 cycles
  - mul + add: 8 cycles total (can pipeline with next iteration)
  - With unrolling by 2 in k-dimension, latency is hidden
  - Critical path: ~5 cycles (B load latency) per iteration

**Bottleneck:** Instruction latency (especially B load), not memory bandwidth. With software pipelining, kernel can achieve 80–90% of peak (28–29 GFLOPS per core).

### Question 8.5
Analyze the performance of a sparse matrix-vector product (SpMV) on Sapphire Rapids. The matrix is stored in Compressed Sparse Row (CSR) format with 10% nonzero density. Explain why gather-based vectorization is slower than scalar code, and propose an alternative approach (hint: consider blocking strategies).

**Model Answer:**
Scalar SpMV (CSR format):
```
for i in [0, m):
  accum = 0
  for k in [row_ptr[i], row_ptr[i+1]):
    j = col_idx[k]
    accum += values[k] * x[j]  // x[j] is a gather (unpredictable memory access)
  y[i] = accum
```

Expected performance:
- With 10% nonzero density: ~0.1m × n nonzero elements
- Each nonzero requires: 1 load col_idx, 1 gather from x (variable address), 1 mul, 1 add
- Best case (all x in L1): 5 + 5 + 4 + 4 = 18 cycles per nonzero
- With 0.1m × n × (1/(n*18 cycles)) nonzeros per output: sparse parallelism limits throughput

Vectorized (gather) approach:
```
for i in [0, m):
  for k in [row_ptr[i], row_ptr[i+1]) step 16:
    col_idx_vec = load col_idx[k:k+16]  // 5 cycles
    values_vec = load values[k:k+16]    // 5 cycles
    x_vec = gather x[col_idx_vec]       // 7–200 cycles (depends on cache)
    prod = values_vec * x_vec           // 4 cycles
    accum += reduce(prod)               // 2 cycles
```

Why it's slower:
- Gather has ~7–200 cycle latency (random access patterns)
- For 10% sparsity: 0.1 × n elements per row, avg ~100k gathers across matrix
- Scalar version: out-of-order execution can hide load latency in some cases
- Gather version: forces serialization (gather must complete before multiply)

Alternative: **Blocked CSR (BCSR)**
- Store sparse matrix in small dense blocks (e.g., 4×4 blocks per nonzero block)
- Exploit dense GEMM within blocks
- Within block: use AVX-512 without gather
- Between blocks: iterate with scalar index (sparse)

Example (4×4 blocks):
```
for i_block in [0, m/4):
  for k_block in [block_ptr[i_block], block_ptr[i_block+1]):
    j_block = block_col_idx[k_block]
    B_dense = block_values[k_block]  // 4×4 dense block
    x_dense = load x[j_block*4:(j_block+1)*4]
    // Dense 4×4 GEMM (use AVX-512 for rows of output)
    for i_local in [0, 4):
      y[i_block*4 + i_local] += matmul(B_dense[i_local][:], x_dense)
```

**Result:** BCSR recovers vectorization benefit by converting sparse structure to dense blocks, enabling unit-stride access and SIMD parallelism. Speedup: 5–8× over scalar CSR depending on block size and sparsity pattern.

---

## 9. READING LIST

1. **Intel Intrinsics Guide** (official, online)
   - https://www.intel.com/content/www/en/en/docs/intrinsics-guide/index.html
   - Search for "AVX-512" and specific instructions (vpaddd, vpdpbusd, vpgatherdd)

2. **Agner Fog, Optimizing software in C++** (free PDF)
   - Vol 1: Optimizing for speed (Chapters 10–11 on SIMD)
   - Vol 3: The microarchitecture of Intel, AMD and VIA CPUs (instruction tables, latency/throughput)
   - Download: https://www.agner.org/optimize/

3. **Intel 64 and IA-32 Architectures Software Developer Manual**
   - Vol 2A: Instruction Set Reference, A–M (VADD, VMUL, VNNI instructions)
   - Vol 2B: Instruction Set Reference, N–Z (Gather, Scatter, AVX-512 extensions)
   - Free PDF from Intel ARK

4. **Intel Optimization Reference Manual for Sapphire Rapids**
   - Chapter 2: Processor microarchitecture (register file, execution ports, frequency scaling)
   - Chapter 6: Memory subsystem (L1/L2/L3 details)
   - Chapter 15: SIMD and vector operations (frequency throttling, AVX-512 on Sapphire Rapids)
   - Chapter on AMX (tile operations)

5. **AMD EPYC Processor Family Performance Tuning Guide**
   - Chapter on SIMD (AVX2 on EPYC Zen 4; no AVX-512 on AMD)
   - For comparison: Zen 4's ASIMD equivalent (VP16) alignment

6. **Computer Organization and Design: MIPS Edition** (H&P, 5th ed.)
   - Appendix F: Graphics and Computing GPUs (tensor operations comparison)
   - Not SIMD-specific but foundational for understanding instruction-level parallelism

7. **STREAM Benchmark Suite**
   - https://www.cs.virginia.edu/stream/
   - Reference for memory bandwidth measurement on multicores

8. **VTune Profiler User Guide** (Intel)
   - Instructions for measuring vectorization efficiency, memory stalls, IPC (instructions per cycle)
   - Free tool from Intel

9. **LIKWID Benchmark Tool**
   - https://github.com/RRZE-HPC/likwid
   - Pre-written SIMD microbenchmarks for GFLOPS measurement

10. **Cutlass (CUDA Tensor Library) GitHub**
    - CPU backend uses AVX-512; reference for optimized tile kernels
    - https://github.com/NVIDIA/cutlass

