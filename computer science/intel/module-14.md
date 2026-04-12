# MODULE 14 — The Performance Engineering Framework

## 1. CONCEPTUAL FOUNDATION

Modern CPU performance analysis has moved from single-metric thinking to **multi-dimensional characterization**. The field has crystallized around three orthogonal frameworks that together provide complete insight into why a kernel executes at a particular speed.

### The Roofline Model

The **Roofline Model** (Williams et al., CACM 2009) defines achievable performance as the minimum of two physical ceilings:

```
Achievable GFLOPs = min(Peak_Compute, Peak_Bandwidth × Arithmetic_Intensity)
```

Where:
- **Peak_Compute** = theoretical peak FP operations per second (e.g., Sapphire Rapids: 1.5 TFLOPs for AVX-512 FP64)
- **Peak_Bandwidth** = memory bandwidth to L1/L2/L3/DRAM (e.g., Sapphire Rapids: 12.8 GB/s per core to L1)
- **Arithmetic_Intensity (AI)** = FLOPs per byte of data moved from a reference level (usually DRAM)

For a SpMM kernel loading a sparse matrix once and executing 2M FLOPs per M bytes:
```
AI = 2M FLOPs / M bytes = 2 FLOPs/byte
```

At 2 FLOPs/byte on a 12.8 GB/s DDR5 link:
```
Achievable = min(1500 GFLOPs, 12.8 GB/s × 2) = min(1500, 25.6) GFLOPs = 25.6 GFLOPs
```

This kernel is **memory-bandwidth-bound**, not compute-bound. No amount of vectorization helps; only algorithm changes (like dataflow tiling to reuse data) will improve performance.

Reference: Williams et al., *Roofline: An Insightful Visual Performance Model for Floating-Point Programs and Multicore Architectures*, CACM 52(4), April 2009. See also Dendibakh, *Performance Analysis and Tuning on Modern CPUs*, Ch. 6.

### Top-Down Microarchitecture Analysis (TMA)

Intel's **Top-Down Microarchitecture Analysis** (Yasin et al., ISPASS 2014) decomposes CPU stalls into a four-level hierarchy:

```
Total_Cycles = Frontend_Bound + Bad_Speculation + Backend_Bound + Retiring

Frontend_Bound:
  ├─ Front-End Latency (fetch stalls, iTLB misses, branch prediction)
  └─ Front-End Bandwidth (i-cache misses, decoding limits)

Bad_Speculation:
  ├─ Branch Misprediction (execution on wrong path)
  └─ Machine_Clears (pipeline flushes, unclear exception, etc.)

Backend_Bound:
  ├─ Memory_Bound (L1/L2/L3/DRAM stalls)
  └─ Core_Bound (compute resource contention, port pressure)

Retiring:
  └─ Useful_Instructions (actually retiring meaningful work)
```

The **critical insight**: a cycle not executing useful instructions is wasted. TMA tells you *where* those cycles go.

On Sapphire Rapids (and 12th-gen Intel), TMA 4.0 refines Backend_Bound further:
```
Backend_Bound:
  ├─ Memory_Bound
  │   ├─ L1_Bound (L1D cache or store buffer issues)
  │   ├─ L2_Bound (L2 cache misses)
  │   ├─ L3_Bound (L3 cache misses)
  │   └─ DRAM_Bound (off-socket memory latency/bandwidth)
  └─ Core_Bound
      ├─ Divider (long-latency divider/FP operations)
      ├─ Ports_Utilization (execution port saturation)
      ├─ Serializing_Operations
      └─ ALU_Op_Utilization
```

Reference: Yasin et al., *Top-Down Microarchitecture Analysis Methodology*, ISPASS 2014. Intel's official TMA documentation: https://www.intel.com/content/dam/develop/external/us/en/documents/tma-methodology.pdf

### Working-Set Analysis and Cache Residency

A critical but often-ignored analysis: **does your data fit in cache?**

Define:
- **Working Set Size (WSS)** = unique memory locations touched per unit time
- **Cache Residency** = fraction of accesses hitting a particular cache level

For an ML inference engine loading weight matrices:
- Batch size 1, attention heads with 64×64 key/query matrices
- If WSS × element_size < 32 MB (Sapphire Rapids L3), the entire attention head graph fits in L3
- Every access hits L3 in ~40 cycles, not DRAM in ~300 cycles

Practical calculation:
```c
// For transformer inference: estimate per-token working set
working_set_bytes = (hidden_dim * num_layers * 2) // weights + activations
                  + (seq_len * hidden_dim * 4)    // KV cache + attention
                  + (batch_size * seq_len * hidden_dim * 8) // activations

bool fits_in_l3 = (working_set_bytes < 51_200_000); // Sapphire Rapids L3
```

If true, your code should achieve ~3-4 GB/s per core (L3 bandwidth), not 12.8 GB/s (theoretical).

### Little's Law Applied to Pipelines

**Little's Law**: `Average_In_System = Arrival_Rate × Response_Time`

Applied to a CPU pipeline:
```
Occupancy = (Throughput_per_cycle) × (Latency_in_cycles)
```

Example: An AVX-512 FP64 add has **3-cycle latency**. If you dispatch adds to the same register every cycle (throughput = 1 add/cycle):
```
Occupancy = 1 add/cycle × 3 cycles = 3 independent adds in flight
```

The pipeline is "half-empty"—you need 6 independent adds to keep it full (throughput = 2 adds/cycle on Sapphire Rapids' dual FP pipes).

**Implication for ML inference**: if your operation graph is a long chain of sequential operations (like autoregressive decoding), you're limited by latency, not throughput. Parallelization strategies (batching, speculative decoding) combat this.

Reference: Dendibakh, Ch. 2.

### Latency-Bound vs Throughput-Bound Transformation

**Latency-bound code**: Limited by the latency of the longest dependency chain.
- Example: Autoregressive decoding where token t+1 depends on outputs of token t
- Instruction-level parallelism (ILP) is the solution

**Throughput-bound code**: Limited by execution port saturation or cache bandwidth.
- Example: Dense matrix multiply where many independent operations exist
- Data-level parallelism (vectorization) and thread-level parallelism (multithreading) solve it

**Transformation strategy**:
1. Measure: is your code latency-bound or throughput-bound? (TMA → Core_Bound → Ports_Utilization)
2. If latency-bound: unroll loops, expose ILP, use prefetching to hide memory latency
3. If throughput-bound: vectorize, multithread, increase data reuse (Roofline AI)

For ML systems: batch-size 1 inference is often latency-bound (long decoding chains). Batch-size 8+ becomes throughput-bound (matrix multiply bottleneck).

---

## 2. MENTAL MODEL

```
┌─────────────────────────────────────────────────────────────────┐
│                    ROOFLINE PERFORMANCE SPACE                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│   GFLOPs                                                         │
│      │        ╱─────── Compute Roof (1.5 TFLOPs peak)           │
│      │       ╱                                                   │
│ 1500 ├──────╱─────────────────────────────────────────────      │
│      │     ╱                                                     │
│      │    ╱  Kernel must fit under BOTH roofs                   │
│  400 ├───╱───────────────────────────────────────               │
│      │  ╱     Memory Roof: 12.8 GB/s bandwidth                 │
│   50 ├──────────────────────────                                │
│      │╱                                                          │
│    0 └────────────────────────────────────────── AI (FLOPs/B)   │
│      0      2      4      6      8     10                       │
│                                                                   │
│  Example kernels:                                               │
│  • SpMV (AI ≈ 0.5): hits memory roof @ 6.4 GFLOPs              │
│  • Dense MatMul (AI ≈ 6): hits compute roof @ 1500 GFLOPs      │
│  • L3-resident (AI ≈ 3): limited by L3 bandwidth @ 120 GFLOPs  │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│         TMA CYCLE DECOMPOSITION (Per 100 Cycles)                │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌─────────────┐    Example stall breakdown:                   │
│  │  Retiring   │    34 cycles: useful instructions              │
│  │    34%      │    12 cycles: frontend stalled (BTB miss)      │
│  ├─────────────┤    20 cycles: branch misprediction             │
│  │   Frontend  │    34 cycles: backend (L3 miss → DRAM wait)    │
│  │ Bad_Spec    │    ─────────                                   │
│  │    12%      │    100 cycles total (+ rounding)               │
│  ├─────────────┤                                                │
│  │   Backend   │  Action: Target L3 misses first (34% of       │
│  │   Bound     │  Backend), then BTB (12% of Frontend)         │
│  │    34%      │                                                │
│  └─────────────┘                                                │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│           WORKING SET vs CACHE HIERARCHY                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│ Sapphire Rapids: 6 cores, per-core cache                        │
│                                                                   │
│  Working Set Size:                                              │
│  ┌────────────────────┐                                         │
│  │  DRAM (512GB+)     │ Access time: 300ns, BW: 1 GB/s/core   │
│  │                    │                                         │
│  │ ┌────────────────┐ │                                         │
│  │ │ L3 (51.2MB)    │ │ Access time: 40ns, BW: 120 GB/s       │
│  │ │                │ │ Shared by 6 cores                      │
│  │ │ ┌────────────┐ │ │                                         │
│  │ │ │ L2 (1.25MB)│ │ │ Access time: 12ns, BW: 300 GB/s       │
│  │ │ │            │ │ │ Per-core L2                           │
│  │ │ │ ┌────────┐ │ │ │                                         │
│  │ │ │ │L1 (48KB)│ │ │ │ Access time: 4ns, BW: 2.5 TB/s       │
│  │ │ │ │        │ │ │ │ Per-core data cache                   │
│  │ │ │ └────────┘ │ │ │                                         │
│  │ │ └────────────┘ │ │                                         │
│  │ └────────────────┘ │                                         │
│  └────────────────────┘                                         │
│                                                                   │
│  Strategy: Maximize "hits per byte loaded from DRAM"           │
│  ⇒ Tile algorithm to fit in L3 when possible                   │
│  ⇒ Trade computation for memory reuse (cache-oblivious)        │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│      LATENCY vs THROUGHPUT DEPENDENCY CHAINS                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  LATENCY-BOUND (sequential dependency):                         │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ a[0] → compute_x → load_x → compute_y → load_y → ret y │   │
│  │ │       3 cyc    │ 40cyc │ 3 cyc   │ 40cyc │ 3 cyc    │   │
│  │ └─────────────────────────────────────────────────────┘   │
│  │ Total: ~89 cycles, limited by longest chain              │   │
│  │ Solution: unroll to execute multiple independent chains  │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                   │
│  THROUGHPUT-BOUND (many independent operations):               │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ for (int i = 0; i < N; i++) {                           │   │
│  │   a[i] = b[i] + c[i];  // independent FMA ops          │   │
│  │   d[i] = e[i] * f[i];  // 2 operations/cycle possible  │   │
│  │ }                                                       │   │
│  │                                                          │   │
│  │ Port pressure: both FMA units busy every cycle          │   │
│  │ Solution: vectorize (AVX-512 → 8× parallelism)         │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. PERFORMANCE LENS

Understanding these frameworks directly impacts engineering decisions:

**Roofline for Algorithm Selection:**
- Your quantized SpMM kernel computes 4 weights per 0.5 bytes (INT4 packed)
- AI = 8 FLOPs/byte on a 25.6 GB/s memory roof → achievable 200 GFLOPs
- Dense GEMM with AI > 8 is compute-bound at 1500 GFLOPs
- **Decision**: Use quantized SpMM for sparse models (>80% sparsity), dense GEMM for dense models

**TMA for Bottleneck Prioritization:**
- If TMA shows 45% Backend_Bound, 30% Memory_Bound (DRAM):
  - First, optimize memory access patterns (blocking, prefetch, data layout)
  - Only then optimize compute (vectorization, instruction-level parallelism)
- If TMA shows 35% Frontend_Bound:
  - Reduce code complexity, improve branch prediction, tune for i-cache

**Working-Set Analysis for Cache Tuning:**
- Transformer layer weights: 768 × 768 × 2 (fp16) = 1.15 MB per layer
- Batch 1, 12 layers → 14 MB of weights + activations < 51.2 MB L3
- **Implication**: all weights stay resident in L3, data reuse is dominated by on-chip bandwidth
- Batch 16 → 224 MB, exceeds L3 → must optimize DRAM prefetch

**Little's Law for Batching Decisions:**
- Single-token autoregressive: latency-bound, batch size 1 = 1 token in flight
- Add speculative decoding: 5 tokens in flight × 5 ns latency = higher throughput
- Add batching: 64 tokens × different sequences = 64× throughput improvement

---

## 4. ANNOTATED CODE

### Example 1: Measuring Roofline AI for a Kernel

```c
// File: roofline_measurement.c
// Compute peak theoretical roofline for a tensor operation
// Measure arithmetic intensity and compare to roofline ceiling

#include <immintrin.h>
#include <stdint.h>
#include <stdio.h>

// Sapphire Rapids: peak 1.5 TFLOPs FP64, 12.8 GB/s DRAM bandwidth
#define PEAK_COMPUTE_GFLOPs 1500.0  // GFLOPs
#define PEAK_BANDWIDTH_GBs 12.8     // GB/s

// Measure memory traffic using RAPL energy counters
typedef struct {
    uint64_t cycles;
    uint64_t instructions;
    uint64_t llc_loads;      // L3 cache line loads
    uint64_t llc_stores;     // L3 cache line stores
    uint64_t bytes_dram_rd;  // DRAM bytes read
    uint64_t bytes_dram_wr;  // DRAM bytes written
} perf_counters_t;

// Dense GEMM: C[M×N] += A[M×K] × B[K×N]
// Arithmetic intensity = 2MNK / (M*K + K*N + M*N) bytes, assuming all from DRAM
void gemm_f64(double *C, const double *A, const double *B,
              int M, int N, int K) {
    // Assuming optimal blocking where A, B fit in L3
    // But we count all loads from memory hierarchy

    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            double sum = C[i*N + j];  // L1 hit (reused 64 times)

            // Vectorized FMA: 4 FLOPs per iteration (AVX-512 would be 8)
            for (int k = 0; k < K; k += 4) {
                __m256d a = _mm256_loadu_pd(&A[i*K + k]);    // L1/L2 load
                __m256d b0 = _mm256_loadu_pd(&B[(k+0)*N + j]); // stride-N load
                __m256d b1 = _mm256_loadu_pd(&B[(k+1)*N + j]);
                __m256d b2 = _mm256_loadu_pd(&B[(k+2)*N + j]);
                __m256d b3 = _mm256_loadu_pd(&B[(k+3)*N + j]);

                sum = _mm256_fmadd_pd(a, b0, sum);  // 4 FMA = 8 FLOPs
                // ... (reuse A[i*K+k] across all j, high arithmetic intensity)
            }
            C[i*N + j] = sum;
        }
    }
}

// Analyze roofline for this GEMM
void analyze_roofline(int M, int N, int K, perf_counters_t *counters) {
    // FLOPs: 2*M*N*K (assuming fused multiply-add)
    double flops = 2.0 * M * N * K;

    // Bytes: count first-load from DRAM
    // Optimal case: A[M×K] loaded once, B[K×N] loaded once, C[M×N] loaded/stored
    // = (M*K + K*N + M*N) * 8 bytes (FP64)
    double bytes_estimate = (M*K + K*N + M*N) * 8.0;

    // Arithmetic intensity
    double ai = flops / bytes_estimate;

    // Achievable GFLOPs from roofline
    double gflops_memory_limited = PEAK_BANDWIDTH_GBs * ai;
    double achievable_gflops = (gflops_memory_limited < PEAK_COMPUTE_GFLOPs)
                               ? gflops_memory_limited
                               : PEAK_COMPUTE_GFLOPs;

    // Measure actual performance from counters
    double actual_gflops = flops / (counters->cycles / 3.0e9);  // assuming 3 GHz

    printf("Roofline Analysis:\n");
    printf("  FLOPs: %.2e\n", flops);
    printf("  Bytes (est.): %.2e\n", bytes_estimate);
    printf("  AI: %.2f FLOPs/byte\n", ai);
    printf("  Memory ceiling: %.0f GFLOPs (AI × 12.8 GB/s)\n", gflops_memory_limited);
    printf("  Compute ceiling: %.0f GFLOPs\n", PEAK_COMPUTE_GFLOPs);
    printf("  Roofline ceiling: %.0f GFLOPs\n", achievable_gflops);
    printf("  Actual measured: %.0f GFLOPs\n", actual_gflops);
    printf("  Efficiency: %.1f%% of roofline\n", 100.0 * actual_gflops / achievable_gflops);
}
```

### Example 2: TMA-Guided Optimization Loop

```c
// File: tma_optimization.c
// Read TMA output and decide optimization strategy

#include <stdio.h>

typedef struct {
    double frontend_bound;      // % of cycles
    double bad_speculation;
    double backend_bound;
    double retiring;

    // Backend breakdown
    double memory_bound;        // subset of backend_bound
    double l1_bound;            // subset of memory_bound
    double l2_bound;
    double l3_bound;
    double dram_bound;

    // Frontend breakdown
    double itlb_miss_bound;
    double icache_miss_bound;
    double branch_pred_miss;
} tma_metrics_t;

// Strategy: address highest bottleneck first
void optimize_based_on_tma(tma_metrics_t *tma) {
    printf("TMA Optimization Strategy:\n");

    // Rule 1: Fix speculation leaks first (they're parasitic)
    if (tma->bad_speculation > 15.0) {
        printf("1. BAD_SPECULATION = %.1f%% (high)\n", tma->bad_speculation);
        printf("   → Reduce branches: unroll loops, use branchless selection\n");
        printf("   → Improve branch predictor: align loop bodies\n");
        printf("   → Measure: perf record -e branch-misses\n");
        return;  // Fix this first before anything else
    }

    // Rule 2: Fix frontend latency (affects all downstream cycles)
    if (tma->frontend_bound > 20.0) {
        printf("2. FRONTEND_BOUND = %.1f%% (high)\n", tma->frontend_bound);
        if (tma->itlb_miss_bound > 5.0) {
            printf("   → iTLB misses: THP (transparent huge pages), reduce code size\n");
        } else if (tma->icache_miss_bound > 10.0) {
            printf("   → I-cache misses: reduce code bloat, improve locality\n");
        } else {
            printf("   → Fetch/decode bottleneck: simplify instruction stream\n");
        }
        return;
    }

    // Rule 3: Fix backend memory issues (largest opportunity)
    if (tma->backend_bound > 40.0) {
        printf("3. BACKEND_BOUND = %.1f%% (high)\n", tma->backend_bound);

        if (tma->dram_bound > tma->l3_bound && tma->dram_bound > 15.0) {
            printf("   ├─ DRAM_BOUND = %.1f%%\n", tma->dram_bound);
            printf("   │  → Reduce dataset size to fit L3 (51 MB per socket)\n");
            printf("   │  → Use cache-oblivious matrix multiplication\n");
            printf("   │  → Prefetch: _mm_prefetch(..., _MM_HINT_T0)\n");
        } else if (tma->l3_bound > 10.0) {
            printf("   ├─ L3_BOUND = %.1f%%\n", tma->l3_bound);
            printf("   │  → Temporal reuse: blocking/tiling\n");
            printf("   │  → Increase arithmetic intensity\n");
        } else if (tma->l2_bound > 10.0) {
            printf("   ├─ L2_BOUND = %.1f%%\n", tma->l2_bound);
            printf("   │  → Align data: 64-byte boundaries for cache lines\n");
            printf("   │  → Reduce strides: traverse data sequentially\n");
        } else if (tma->l1_bound > 10.0) {
            printf("   ├─ L1_BOUND = %.1f%%\n", tma->l1_bound);
            printf("   │  → Store forwarding: avoid load-after-store\n");
            printf("   │  → Organize data to fit 32 KB L1D cache\n");
        }
        return;
    }

    // Rule 4: Core-bound (compute resource contention)
    if (tma->backend_bound - tma->memory_bound > 30.0) {
        printf("4. CORE_BOUND = %.1f%%\n", tma->backend_bound - tma->memory_bound);
        printf("   → Insufficient parallelism: unroll loops, vectorize\n");
        printf("   → Port pressure: balance FP adds/multiplies (Sapphire Rapids: 4 FP ports)\n");
        printf("   → Measure: perf stat -e cpu/port_*/\n");
        return;
    }

    // Rule 5: If retiring > 40%, code is reasonably efficient
    printf("5. RETIRING = %.1f%% (excellent)\n", tma->retiring);
    printf("   → Code is compute-efficient, focus on scaling (threads, batches)\n");
}

// Example invocation with real measurements
int main() {
    tma_metrics_t example = {
        .frontend_bound = 8.5,
        .bad_speculation = 3.2,
        .backend_bound = 52.1,
        .retiring = 36.2,
        .memory_bound = 38.4,
        .l1_bound = 2.1,
        .l2_bound = 5.3,
        .l3_bound = 12.8,
        .dram_bound = 18.2,
        .itlb_miss_bound = 0.5,
        .icache_miss_bound = 2.1,
        .branch_pred_miss = 2.8,
    };

    optimize_based_on_tma(&example);
    return 0;
}
```

### Example 3: Cache Residency Analysis for Transformer Layers

```c
// File: cache_residency_transformer.c
// Compute working set size and verify L3 residency

#include <stdint.h>
#include <stdio.h>
#include <math.h>

typedef struct {
    int hidden_dim;          // 768, 1024, etc.
    int num_heads;           // 12, 16, etc.
    int num_layers;          // 12, 24, etc.
    int batch_size;          // token batch
    int seq_len;             // context length
} transformer_config_t;

// Compute memory footprint for forward pass
typedef struct {
    uint64_t weights_bytes;   // All W_q, W_k, W_v, W_o, FFN weights
    uint64_t kv_cache_bytes;  // KV cache for all layers
    uint64_t activation_bytes; // Layer outputs
    uint64_t total_bytes;
} memory_footprint_t;

void compute_footprint(transformer_config_t *cfg, memory_footprint_t *footprint) {
    // Weights per layer (fp16 = 2 bytes)
    // W_q, W_k, W_v, W_o: 4 × (hidden_dim × hidden_dim)
    // W_ff1, W_ff2: (hidden_dim × 4*hidden_dim) + (4*hidden_dim × hidden_dim)
    uint64_t weights_per_layer =
        (4 * cfg->hidden_dim * cfg->hidden_dim +  // attention matrices
         cfg->hidden_dim * 4 * cfg->hidden_dim +   // FFN first
         4 * cfg->hidden_dim * cfg->hidden_dim) * 2; // FFN second (fp16)

    footprint->weights_bytes = weights_per_layer * cfg->num_layers;

    // KV cache: per layer, per head: (seq_len × head_dim)
    // head_dim = hidden_dim / num_heads
    int head_dim = cfg->hidden_dim / cfg->num_heads;
    uint64_t kv_per_layer = cfg->num_heads * seq_len * head_dim * 2 * 2; // K and V
    footprint->kv_cache_bytes = kv_per_layer * cfg->num_layers;

    // Activations: hidden states + attention scores + intermediate states
    uint64_t activation_per_layer =
        (cfg->batch_size * cfg->seq_len * cfg->hidden_dim +  // hidden state
         cfg->batch_size * cfg->num_heads * cfg->seq_len * cfg->seq_len) * 2; // attention
    footprint->activation_bytes = activation_per_layer * cfg->num_layers;

    footprint->total_bytes = footprint->weights_bytes +
                            footprint->kv_cache_bytes +
                            footprint->activation_bytes;
}

#define L3_CACHE_BYTES (51 * 1024 * 1024)  // Sapphire Rapids, 51 MB

void analyze_residency(transformer_config_t *cfg) {
    memory_footprint_t footprint;
    compute_footprint(cfg, &footprint);

    printf("Transformer Memory Analysis:\n");
    printf("  Config: hidden=%d, heads=%d, layers=%d, batch=%d, seq_len=%d\n",
           cfg->hidden_dim, cfg->num_heads, cfg->num_layers,
           cfg->batch_size, cfg->seq_len);
    printf("  Weights: %.1f MB\n", footprint.weights_bytes / 1e6);
    printf("  KV Cache: %.1f MB\n", footprint.kv_cache_bytes / 1e6);
    printf("  Activations: %.1f MB\n", footprint.activation_bytes / 1e6);
    printf("  Total: %.1f MB\n", footprint.total_bytes / 1e6);
    printf("  L3 Cache: %.1f MB\n", L3_CACHE_BYTES / 1e6);

    if (footprint.total_bytes < L3_CACHE_BYTES) {
        printf("  ✓ FITS IN L3: expect L3 latency (~40ns)\n");
        printf("    Effective bandwidth: 120 GB/s (L3 bandwidth)\n");
    } else if (footprint.weights_bytes < L3_CACHE_BYTES) {
        printf("  ✗ Weights fit in L3, but activations spill\n");
        printf("    → Reuse: only weights are cached\n");
        printf("    → Bottleneck: KV cache DRAM bandwidth\n");
    } else {
        printf("  ✗ DOES NOT FIT IN L3: expect DRAM latency (~300ns)\n");
        printf("    Effective bandwidth: ~1 GB/s per core (DRAM)\n");
    }

    // Estimate memory bandwidth utilization
    // Assuming compute is fast (not memory-bound), bytes per useful FLOP
    double weights_reuse = 1.0;  // each weight loaded once per token
    double bytes_per_token = footprint.weights_bytes / cfg->batch_size +
                            footprint.activation_bytes / cfg->batch_size;
    double flops_per_token = 2.0 * cfg->hidden_dim * cfg->hidden_dim * cfg->num_layers;
    double arithmetic_intensity = flops_per_token / bytes_per_token;

    printf("  Arithmetic Intensity: %.2f FLOPs/byte\n", arithmetic_intensity);
    printf("  (Higher = more compute-bound, lower = more memory-bound)\n");
}

int main() {
    // Test case: BERT-base (768-dim, 12 layers, 12 heads)
    transformer_config_t bert_base = {
        .hidden_dim = 768,
        .num_heads = 12,
        .num_layers = 12,
        .batch_size = 1,
        .seq_len = 512,
    };

    analyze_residency(&bert_base);
    printf("\n");

    // Test case: GPT-3 style (1024-dim inference)
    transformer_config_t gpt3_small = {
        .hidden_dim = 1024,
        .num_heads = 16,
        .num_layers = 24,
        .batch_size = 8,   // batched inference
        .seq_len = 2048,
    };

    analyze_residency(&gpt3_small);

    return 0;
}
```

---

## 5. EXPERT INSIGHT

### Non-Obvious Truth #1: Roofline Ceilings Are Soft

Many junior engineers plot roofline and assume "I cannot do better than this line." **False.** Roofline assumes:
- All memory bandwidth is equally accessible
- Compute and memory overlap perfectly

In reality:
- **L3 bandwidth > DRAM bandwidth**: your code might be "DRAM-bound" on roofline but achieve 200 GB/s (L3 roofline) through prefetch tuning
- **Overlap incomplete**: if memory stalls block the pipeline before compute pipes fill, you hit compute ceiling before roofline predicts
- **Vectorization penalty**: moving from scalar (4-cycle latency) to AVX-512 (same latency) doesn't improve roofline, but *does* reduce achievable throughput if dependency chains are long

**Lesson**: Use roofline as a lower bound ("we *cannot* exceed this"), not an upper bound. Expert optimization often exceeds the original roofline by changing memory layout, blocking, or vectorization strategy.

### Non-Obvious Truth #2: TMA 4.0 Conceals High-Level Stalls

TMA tells you which hardware unit is stalled, but not *why* you stalled it.

Example: TMA says "30% DRAM_Bound". Possible causes:
1. **Suboptimal prefetch**: you're issuing loads sequentially (128-byte L2 prefetch latency)
2. **Too many pending requests**: hardware prefetch is overloaded, loads queue in port buffers
3. **LLC miss pattern**: your access pattern defeats the LLC's coherence predictor
4. **NUMA locality**: in multi-socket systems, remote NUMA accesses cost 2-3× more

**Fix**: Measure finer granularity: `perf stat -e cache-references,cache-misses,LLC-loads,LLC-stores,mem_load_l3_miss_retired.local,mem_load_l3_miss_retired.remote`

The senior engineer reads TMA as a hypothesis generator, not a diagnosis. They then measure specific counters to confirm.

### Non-Obvious Truth #3: Working-Set Analysis Predicts Cache Behavior, Not Performance

A kernel might have WSS < L3, but still be memory-bound if **access patterns are terrible**.

Example: Random access to 40 MB (fits in L3) with 64-bit strides:
- L3 hit rate: ~30% (many cache lines touched once, replaced)
- L3 effective bandwidth: 0.3 × 120 GB/s = 36 GB/s
- Still memory-bound on roofline, despite L3 residency

**Countermeasure**: Measure **cache reuse distance**. For each load, count how many unique cache lines are loaded before that line is touched again. High reuse distance = low effective bandwidth.

### Non-Obvious Truth #4: Little's Law Explains Batch Size Selection

Autoregressive decoding bottleneck for batch size 1:
```
Occupancy = 1 token/cycle × (5 layers × 40ns DRAM latency) = 200 tokens in flight
```

But only 1 token is present. So:
```
Pipeline Utilization = 1 / 200 = 0.5%
```

**Solution**: Speculative decoding adds 5 tokens in flight:
```
Pipeline Utilization = 5 / 200 = 2.5%
```

Speculative decoding doesn't change roofline, it changes **occupancy**, which is what matters when latency >> throughput. This is why speculatively decoding 5 tokens (at cost of some discarded compute) is worth it: you're using the pipeline capacity that's wasted anyway.

### Non-Obvious Truth #5: Latency-Bound ≠ Sequential Dependency Chain

A kernel can be latency-bound even with no data dependencies if:
- Register pressure is high (spill to L1, 40-cycle latency)
- Memory prefetch buffer is full (new loads must wait)
- Instruction window is shallow (Sapphire Rapids: 352-entry ROB)

Example: 1000-iteration loop with 20 independent memory loads:
```c
for (int i = 0; i < 1000; i++) {
    for (int j = 0; j < 20; j++) {
        x[i][j] = load(random_address[i*20+j]);  // 40-cycle DRAM latency
    }
}
```

Seems parallelizable (20 independent loads per i), but if DRAM memory controller can only issue 8 outstanding requests, the 9th load stalls. This is **latency-bound due to resource saturation**, not data dependency.

---

## 6. BENCHMARK / MEASUREMENT

### Procedure: Extract Roofline from Real Hardware

```bash
# Step 1: Install Intel VTune or Linux perf
# On Ubuntu:
sudo apt install linux-tools-generic

# Step 2: Run a known-compute kernel and measure
# Use STREAM benchmark to measure DRAM bandwidth:
git clone https://github.com/jeffhammond/STREAM.git
cd STREAM
gcc -O3 -march=native -fopenmp stream.c -o stream
./stream
# Output: Copy bandwidth, scale bandwidth, add bandwidth, triad bandwidth
# Sapphire Rapids typical: 12.8 GB/s per core

# Step 3: Measure compute ceiling
# Using OpenMP + AVX-512 FMA:
gcc -O3 -march=sapphirerapids -fopenmp flops.c -o flops
# Count FP64 operations per cycle using perf

# Step 4: Plot your kernel on roofline
# Measure arithmetic intensity:
perf stat -e cycles,instructions,LLC-load-misses,LLC-loads,dTLB-load-misses ./your_kernel
# Calculate: AI = instructions / (LLC-loads × 64 bytes)

# Step 5: Verify on the roofline plot
# AI × 12.8 GB/s = memory-bound ceiling
# 1500 GFLOPs = compute ceiling
# actual GFLOPs from (instructions / 2) / (cycles / 3e9)
```

### Procedure: Measure TMA and Interpret

```bash
# Step 1: Ensure TMA events are available (perf list | grep TMA_)
# Sapphire Rapids: yes. Older CPUs: may require Intel VTune

# Step 2: Run TMA level 1 (broadest breakdown)
perf stat -e cycles \
  -e 'cpu/event=0x3c,umask=0x00,name=CPU_CLK_UNHALTED.THREAD/' \
  -e 'cpu/event=0xa4,umask=0x01,name=TOPDOWN.SLOTS/' \
  -e 'cpu/event=0xc2,umask=0x02,name=UOPS_RETIRED.RETIRE_SLOTS/' \
  ./your_kernel

# Step 3: Calculate percentages
# TMA_Backend_Bound = (Backend_Cycles / Total_Cycles) × 100
# where Backend_Cycles = stalled cycles not retiring useful uops

# Step 4: Use Intel VTune for finer breakdown (Level 2)
vtune -collect uarch-exploration -knob pmu-event-sampling=true \
  -r results_dir -- ./your_kernel
vtune -report uarch-exploration -r results_dir

# Step 5: Interpret output
# If Backend_Bound > 40% && Memory_Bound > 20%: focus on memory
# If Frontend_Bound > 30%: focus on i-cache / branching
# If Bad_Speculation > 15%: focus on branch prediction
```

### Procedure: Cache Residency Measurement

```bash
# Step 1: Measure L3 hit rate
perf stat -e cache-references,cache-misses,LLC-loads,LLC-load-misses \
  ./your_kernel

# Step 2: Calculate metrics
# L3 Hit Rate = (LLC-loads - LLC-load-misses) / LLC-loads
# If > 95%, data is mostly cached in L3
# If < 50%, data is thrashing in DRAM

# Step 3: Measure working set size in real time
perf record -e cache-misses:u -g ./your_kernel
perf report --stdio

# Step 4: Cross-check with cachegrind
valgrind --tool=cachegrind --cachegrind-out-file=out.cgout ./your_kernel
cg_annotate out.cgout

# Output shows:
#  I1 refs:      instructions executed
#  LL miss:      last-level cache misses
#  D1 miss:      L1 data cache misses
```

---

## 7. ML SYSTEMS RELEVANCE

### Attention Layer Analysis via Roofline

For a transformer attention head (batch size B, sequence length N, head dimension D):

```
Operation: scores = Q @ K^T + softmax + weighted_sum(V)
```

Arithmetic intensity varies by phase:
- **Q @ K^T**: (B×N×D) × (D×N) → (B×N×N) matrix = 2BN²D / (2BND + BN²) FLOPs/byte
  - If N = 512, D = 64: AI ≈ 16 FLOPs/byte → compute-bound at 200 GFLOPs
  - If N = 512, D = 8: AI ≈ 2 FLOPs/byte → memory-bound at 25 GFLOPs

**Implication**: Long-context models (large N) shift attention from memory-bound to compute-bound. Flash Attention exploits this: by fusing operations and increasing AI, it moves attention from below-roofline to above-roofline performance.

### Batching Impact on Throughput-Bound Behavior

Batch size 1: 512 tokens per sequence, each token latency-bound by autoregressive dependency.
- Occupancy: 1 token in flight, 512-cycle latency → 0.2% pipeline utilization
- TMA: likely 50%+ Backend_Bound waiting on KV cache accesses

Batch size 64: 64 independent token sequences executing in parallel.
- Occupancy: 64 tokens × 512 layers = 32k logical tokens in flight
- All execution ports saturated; many independent operations
- TMA: retiring > 35%, backend bound drops to 20%

**Optimization strategy**: Start with small batches (B=1) and measure. If retiring < 30%, you're memory or speculation bound. Increase B until retiring plateaus → you've found the sweet spot for throughput.

### Quantization's AI Impact

FP32 weight matrix: 1 billion parameters × 4 bytes = 4 GB
INT8 quantized: 1 billion parameters × 1 byte = 1 GB
**AI improves 4×** (same FLOPs, 1/4 the bandwidth).

Example: SpMM with FP32, AI = 1 FLOP/byte (memory-bound at 12.8 GFLOPs).
With INT8: AI = 4 FLOPs/byte (memory-bound at 51.2 GFLOPs → compute-bound).

**Lesson**: Quantization shifts the roofline ceiling. Junior engineers sometimes ignore this; expert engineers recognize it's often the dominant factor in deployment performance.

---

## 8. PhD QUALIFIER QUESTIONS

**Q1**: Define the Roofline Model's two ceilings and explain when a kernel is limited by each. For a SpMV kernel with AI = 0.5 on hardware with 10 GB/s DRAM bandwidth and 500 GFLOPs peak compute, what is the maximum achievable performance? Draw the roofline and annotate your answer.

**Q2**: Explain the relationship between Little's Law (Occupancy = Throughput × Latency) and autoregressive language model inference bottlenecks. Why is speculative decoding beneficial even if it increases total FLOPs?

**Q3**: TMA reports 35% Frontend_Bound, 42% Backend_Bound (of which 22% is DRAM_Bound), and 23% Retiring. What is your optimization priority, and why? Propose one concrete optimization for the top bottleneck.

**Q4**: You measure a transformer attention layer with working-set size of 35 MB (weights + activations). Sapphire Rapids L3 cache is 51.2 MB per socket. However, L3 hit rate is only 60%. Explain possible causes and how you would diagnose each.

**Q5**: Compare the arithmetic intensity and roofline ceilings for (a) dense GEMM (C += A×B), (b) sparse MatMul with 90% sparsity, and (c) attention with sequence length 4096. Which is compute-bound? Which is memory-bound? Justify numerically.

---

## 9. READING LIST

1. **Williams, S. A., et al.** (2009). "Roofline: An Insightful Visual Performance Model for Floating-Point Programs and Multicore Architectures." *Communications of the ACM*, 52(4): 65-76.
   - See: Sections 2-3 (model definition), Section 4 (roofline plots)

2. **Dendibakh, D.** (2024). *Performance Analysis and Tuning on Modern CPUs*. O'Reilly Media.
   - See: Chapter 2 (CPU basics), Chapter 6 (Roofline), Chapter 7 (TMA)

3. **Yasin, A.** (2014). "Top-Down Microarchitecture Analysis Method." *ISPASS 2014*.
   - See: Sections 2-3 (TMA hierarchy), Section 4 (event formulas)

4. **Intel**. *64 Architecture Optimization Reference Manual*. Order #248966.
   - See: Chapter 2 (memory hierarchy), Section 3.5 (performance monitoring)

5. **Fog, A.** (2023). *Instruction Tables: Lists of Instruction Latencies, Throughputs, and Micro-operation Breakdowns for Intel, AMD, and VIA CPUs*. Vol. 4.
   - See: Appendix A (Sapphire Rapids latencies), Appendix B (vectorization throughput)

6. **Intel**. *VTune Profiler User Guide*. Online documentation.
   - See: Section "Microarchitecture Analysis", "Hot Spot Analysis"

7. **Gregg, B.** (2021). *Performance Testing Guide for Software Architects*. Addison-Wesley.
   - See: Chapter 4 (measurement methodology), Chapter 5 (profiling tools)

---

**End of Module 14**

*This module forms the theoretical and practical foundation for the remaining performance engineering sequence. Mastery of Roofline, TMA, and working-set analysis is prerequisite for Modules 15-16.*
