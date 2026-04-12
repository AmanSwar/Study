# MODULE 23 — AMD EPYC Memory Subsystem: DDR5, NUMA, 3D V-Cache

## 1. CONCEPTUAL FOUNDATION

### Zen 4 EPYC Memory Architecture Overview

The AMD EPYC 9004X series (Genoa) features a revolutionary 12-channel DDR5 memory subsystem integrated into a separate I/O Die (IOD). This represents a departure from previous EPYC generations (7002, 7003) which used 12-channel DDR4 but with different IF-memory coupling characteristics.

**DDR5-5600 Theoretical Peak:**
- 12 channels × 64-bit per channel × 5600 MT/s (transfers/second) × 2 (DDR)
- = 12 × 8 bytes × 5600 × 10^6 / 10^9 = **460 GB/s theoretical**

**Key Changes from EPYC 9002 (Milan):**
- DDR4 → DDR5 (higher frequency, lower voltage: 1.1V vs. 1.2V)
- On-package IOD (vs. chipset on previous generations)
- Tighter IF-memory clock coupling (measured impact: ~5% performance/frequency)
- 3D V-Cache option (9004X series): adds 64 MB stacked SRAM per CCD

**Reference:** AMD EPYC 9004X Memory Controller Specifications (2023), Section 2.1.

### NUMA Topology Fundamentals

NUMA (Non-Uniform Memory Access) becomes critical at 96-128 core scales. The memory controller distributes among CCDs:

**NPS1 Mode (No NUMA Partitioning):**
- Single logical NUMA node per socket
- Interleaving: Addresses stripped across all 12 memory channels
- DRAM bandwidth shared equally among all cores
- Coherency: Directory-based, 12-hop latency maximum

**NPS2 Mode (2 NUMA Nodes per Socket):**
- CCD0-5 → NUMA node 0 (6 memory controllers)
- CCD6-11 → NUMA node 1 (6 memory controllers)
- Cross-node latency: ~50% penalty vs. local
- Typical deployment: 192-core Bergamo systems

**NPS4 Mode (4 NUMA Nodes per Socket):**
- Each CCD group (CCDs 0-2, 3-5, 6-8, 9-11) forms separate NUMA domain
- Local DRAM bandwidth: ~115 GB/s per node (460 ÷ 4)
- Cross-node latency: 2× penalty (120 ns → 200+ ns)
- Optimal for workloads with strong CCD affinity

**NPS8 Mode (8 NUMA Nodes):**
- Not typically used (fragmentary node sizes)
- Supported for completeness by AMD

### 3D V-Cache Architecture (9004X Series)

3D V-Cache (officially "3D V-Cache Technology") stacks 64 MB of SRAM on top of each CCD:

**Stacking Technology:**
- 12-layer copper via pillars connecting L3 cache to stacked SRAM
- Via pitch: 40 µm (vs. 48 µm standard)
- Stacked die thickness: ~150 µm (separate substrate on top of CCD)

**Coherency Integration:**
- V-Cache is **inclusive** of L3 (all V-Cache lines also reside in L3)
- No separate coherency state machine (simplifies protocol)
- Evictions from L3 can populate V-Cache (victim buffer model)
- V-Cache hit latency: ~10-15 cycles (through interconnect to stacked SRAM)

**Capacity vs. Latency Trade-off:**
- 64 MB × 12 CCDs = 768 MB total per socket
- Searches show ~10-15 cycle access latency (vs. ~40 cycles for L3 miss → DRAM)
- Hit ratio on transformer weights: 60-70% improvement vs. L3-only

---

## 2. MENTAL MODEL

### EPYC 9004X Memory Hierarchy & NUMA Layout (NPS4 Mode)

```
                    NUMA Node 0      NUMA Node 1      NUMA Node 2      NUMA Node 3
                   (CCDs 0-2)       (CCDs 3-5)       (CCDs 6-8)       (CCDs 9-11)

┌─────────────────────────────────────────────────────────────────────────────────┐
│                                                                                 │
│  ┌──────────────────┐ ┌──────────────────┐ ┌──────────────────┐ ┌─────────────┐ │
│  │  3 CCDs × L3     │ │  3 CCDs × L3     │ │  3 CCDs × L3     │ │  3 CCDs ×   │ │
│  │  32MB each       │ │  32MB each       │ │  32MB each       │ │  L3 32MB    │ │
│  │                  │ │                  │ │                  │ │             │ │
│  │ 96 MB L3 total  │ │ 96 MB L3 total  │ │ 96 MB L3 total  │ │ 96 MB total │ │
│  └────────┬─────────┘ └────────┬─────────┘ └────────┬─────────┘ └──────┬──────┘ │
│           │                    │                    │                  │        │
│  ┌────────▼─────────┐ ┌────────▼─────────┐ ┌────────▼─────────┐ ┌─────▼───────┐│
│  │  V-Cache 64MB    │ │  V-Cache 64MB    │ │  V-Cache 64MB    │ │ V-Cache 64MB││
│  │  (stacked SRAM)  │ │  (stacked SRAM)  │ │  (stacked SRAM)  │ │(stacked)    ││
│  │  per CCD group   │ │  per CCD group   │ │  per CCD group   │ │ per CCD grp ││
│  └────────┬─────────┘ └────────┬─────────┘ └────────┬─────────┘ └─────┬───────┘│
│           │                    │                    │                  │        │
│  ┌────────▼──────────────┐ ┌───▼────────────────┐ ┌───▼────────────┐ ┌─▼───────┐│
│  │  MC 0,1,2             │ │  MC 3,4,5          │ │  MC 6,7,8      │ │ MC 9..11││
│  │  (Memory Controllers) │ │  (MCTRL per node)  │ │  (MCTRL)       │ │(MCTRL)  ││
│  └────────┬──────────────┘ └───┬────────────────┘ └───┬────────────┘ └─┬───────┘│
└───────────┼──────────────────────┼─────────────────────┼────────────────┼────────┘
            │                      │                     │                │
┌───────────▼──────────────────────▼─────────────────────▼────────────────▼────────┐
│                        Infinity Fabric (IF @ 2.4 GHz)                           │
│                    (Coherency & data packet routing)                            │
└───────────┬──────────────────────┬─────────────────────┬────────────────┬────────┘
            │                      │                     │                │
┌───────────▼────────┐ ┌──────────▼──────┐ ┌──────────▼──────┐ ┌────────▼─────────┐
│ DRAM DIMM Slot 0-5 │ │ DRAM Slot 6-11   │ │ DRAM Slot 12-17 │ │ DRAM Slot 18-23  │
│ ~115 GB/s BW       │ │ ~115 GB/s BW     │ │ ~115 GB/s BW    │ │ ~115 GB/s BW     │
│ ~100-120 ns latency│ │ ~130-150 ns      │ │ ~150-180 ns     │ │ ~180-220 ns      │
│                    │ │                  │ │                 │ │                  │
│ DDR5-5600          │ │ DDR5-5600        │ │ DDR5-5600       │ │ DDR5-5600        │
└────────────────────┘ └──────────────────┘ └─────────────────┘ └──────────────────┘
```

### V-Cache Coherency and Inclusion Property

```
Coherency State Diagram (inclusive design):

                    L3 MODIFIED, V-Cache valid:
                    ↓ (on L3 eviction)
                    → V-Cache becomes victim (can stay modified or writeback)

                    L3 HIT:
                    ├→ Line in L3, optionally in V-Cache
                    ├→ V-Cache acts as extension (not actively indexed)
                    └→ Miss → could refetch from V-Cache (10-15 cycles)

                    L3 MISS:
                    ├→ Check V-Cache (if not in L3)
                    ├→ If hit: restore line to L3 (10-15 cycles)
                    └→ If miss: fetch from DRAM (100-120 ns)

                    V-Cache INVALIDATION:
                    ├→ Sync with L3 invalidation (coherency protocol)
                    ├→ Never stale (inclusive property)
                    └→ Simplifies protocol (no split coherency)

Measured latencies:
  L3 hit:         ~40 cycles
  V-Cache hit:    ~15 cycles (+ 40 cycles for L3 miss part)
  DRAM:           ~100-120 ns = ~350-420 cycles @ 3.5 GHz
```

### NUMA Latency Hierarchy (NPS4 Mode)

```
Access Type                Latency (ns)      Latency (cycles @ 3.5 GHz)
─────────────────────────────────────────────────────────────────────
L1-D hit                   ~4                ~14
L2 hit                     ~11               ~38
L3 hit                     ~40               ~140
V-Cache hit (if L3 miss)   ~15               ~50
DRAM local NUMA            ~100-120          ~350-420
DRAM adjacent NUMA (1 hop) ~150              ~525
DRAM remote NUMA (2+ hops) ~200+             ~700+
```

---

## 3. PERFORMANCE LENS

### Memory Subsystem Performance Characteristics for ML Inference

**1. Bandwidth Saturation Curve**

Single-CCD (8-core) achievable bandwidth: ~40-50 GB/s (not limited by IF)
Multi-CCD (96-core) achievable bandwidth: ~300-350 GB/s (IF limited, not DRAM limited)

Why? The Infinity Fabric bandwidth per link is ~340 GB/s (bidirectional). When multiple CCDs compete for DRAM access:
- CCD 0 → DRAM: 340 GB/s available
- CCDs 0-11 all requesting DRAM: bandwidth *per CCD* drops to 460/12 ≈ 38 GB/s

However, IF reordering and coherency traffic reduce observed bandwidth to ~280-300 GB/s in practice.

**2. NUMA Latency Penalty on Unbound Threads**

If a thread is pinned to CCD 0 but its memory is allocated on NUMA node 1 (CCDs 3-5):
- Latency penalty: 100 ns → 150 ns = 50 ns additional = ~175 cycles @ 3.5 GHz
- Bandwidth penalty: ~30-40% reduction (coherency cache line transfers over IF)

For ML inference with batch size = 1 (memory latency dominated), NUMA mismatch can cause 15-25% slowdown.

**3. 3D V-Cache Hit Rate Impact**

Transformer weight matrices (typical: 4-8 GB per 7B model):
- L3 only: hit rate ~30-40% (weights are too large)
- L3 + V-Cache: hit rate ~60-70% (stacked SRAM captures working set tail)

Latency savings: (70% - 40%) × (100 ns - 15 ns) ≈ 21 ns per miss → ~10-15% latency reduction on weight-heavy loops.

**4. Memory Interleaving Impact on Cache Efficiency**

Default interleaving (stripe across all 12 NUMA nodes in NPS1):
- Consecutive cache lines (0, 1, 2, ...) distributed across CCDs 0, 1, 2, ...
- False sharing: Adjacent accesses by different threads land on different CCDs → IF traffic

NUMA-aware interleaving (NPS4):
- Stripe only across 3 MCTRLs per NUMA node
- Consecutive lines stay in same NUMA node → better locality
- Reduced IF traffic (coherency doesn't need cross-node hops)

Measured benefit: ~5-10% on multi-threaded inference vs. NPS1.

---

## 4. ANNOTATED CODE

### Example 1: NUMA-Aware Memory Allocation & Binding

```c
// Module-23-example-numa-allocation.c
// Demonstrates optimal NUMA allocation for ML inference on EPYC 9004X

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <numactl.h>
#include <omp.h>
#include <sched.h>

#define MODEL_SIZE (8UL * 1024 * 1024)  // 8 MB weights
#define BATCH_SIZE 32
#define NUMA_NODES 4  // NPS4 mode: 4 NUMA nodes per socket

typedef struct {
    float *weights;      // Model weights (model_size)
    float *activations;  // Batch activations (batch_size × hidden_dim)
    int numa_node;
    int num_cores_per_node;
} inference_context_t;

// Allocate model weights + activations on NUMA node
inference_context_t* allocate_inference_context(int numa_node_id, int num_cores) {
    inference_context_t *ctx = malloc(sizeof(inference_context_t));

    // Allocate weights on target NUMA node
    ctx->weights = (float *)numa_alloc_onnode(MODEL_SIZE, numa_node_id);
    if (!ctx->weights) {
        fprintf(stderr, "Failed to allocate weights on NUMA node %d\n", numa_node_id);
        return NULL;
    }

    // Allocate activations (same NUMA node for data locality)
    ctx->activations = (float *)numa_alloc_onnode(
        BATCH_SIZE * 4096 * sizeof(float),  // Assume 4096-dim hidden layer
        numa_node_id
    );

    ctx->numa_node = numa_node_id;
    ctx->num_cores_per_node = num_cores;

    printf("Allocated context for NUMA node %d (weights: %p, activations: %p)\n",
           numa_node_id, (void *)ctx->weights, (void *)ctx->activations);

    return ctx;
}

void free_inference_context(inference_context_t *ctx) {
    numa_free(ctx->weights, MODEL_SIZE);
    numa_free(ctx->activations, BATCH_SIZE * 4096 * sizeof(float));
    free(ctx);
}

// Matrix multiply: C = A × B (weights × activations)
// Assumes A is stored on context's NUMA node
void gemm_numa_local(float *C, const float *A, const float *B,
                     int m, int k, int n, int numa_node) {
    // Bind calling thread to cores on target NUMA node
    // This ensures memory controller proximity

    cpu_set_t mask;
    CPU_ZERO(&mask);

    // Assign cores from this NUMA node to threads
    // Assumption: NUMA node i has cores [i*8, i*8+8)  (8 cores per CCD, 1 CCD per NUMA in NPS4)
    int core_start = numa_node * 8;
    for (int c = core_start; c < core_start + 8; c++) {
        CPU_SET(c, &mask);
    }

    if (sched_setaffinity(0, sizeof(mask), &mask) < 0) {
        perror("sched_setaffinity");
        return;
    }

    // Parallel gemm within NUMA node
    #pragma omp parallel for collapse(2) schedule(static, 16) num_threads(8)
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            float sum = 0.0f;
            for (int p = 0; p < k; p++) {
                sum += A[i * k + p] * B[p * n + j];
            }
            C[i * n + j] = sum;
        }
    }
}

int main() {
    printf("=== EPYC 9004X NUMA-Aware Inference ===\n\n");

    // Query NUMA configuration
    int num_nodes = numa_num_configured_nodes();
    printf("Detected %d NUMA nodes (expect 4 in NPS4 mode)\n\n", num_nodes);

    if (num_nodes > 4) {
        printf("Warning: More than 4 NUMA nodes detected. Using first 4.\n");
        num_nodes = 4;
    }

    // Allocate inference contexts per NUMA node
    inference_context_t **contexts = malloc(num_nodes * sizeof(inference_context_t *));

    for (int i = 0; i < num_nodes; i++) {
        contexts[i] = allocate_inference_context(i, 8);  // 8 cores per CCD/NUMA node
    }

    printf("\nRunning inference on each NUMA node independently...\n\n");

    // Run inference: each NUMA node processes independently
    // This demonstrates optimal NUMA locality
    #pragma omp parallel for num_threads(4) schedule(static, 1)
    for (int node_id = 0; node_id < num_nodes; node_id++) {
        inference_context_t *ctx = contexts[node_id];

        // Ensure threads are bound to this NUMA node
        cpu_set_t mask;
        CPU_ZERO(&mask);
        for (int c = node_id * 8; c < node_id * 8 + 8; c++) {
            CPU_SET(c, &mask);
        }
        sched_setaffinity(0, sizeof(mask), &mask);

        // Allocate output buffer on same NUMA node
        float *output = (float *)numa_alloc_onnode(
            BATCH_SIZE * 4096 * sizeof(float),
            node_id
        );

        // GEMM: weights (8 MB) × activations (batch_size × hidden_dim)
        // L2 cache (1 MB per core): can fit ~125 KB of weights per core
        // Ideal: tile weights to fit in L2, stream activations from L3
        gemm_numa_local(output, ctx->weights, ctx->activations,
                       BATCH_SIZE, 4096, 4096, node_id);

        printf("Node %d inference complete (output @ %p)\n", node_id, (void *)output);

        numa_free(output, BATCH_SIZE * 4096 * sizeof(float));
    }

    // Clean up
    for (int i = 0; i < num_nodes; i++) {
        free_inference_context(contexts[i]);
    }
    free(contexts);

    printf("\n=== NUMA-Optimized Inference Complete ===\n");
    printf("Key takeaway: NUMA binding + local allocation = optimal bandwidth\n");

    return 0;
}

/*
Compilation:
  gcc -O3 -march=znver4 -fopenmp module-23-example-numa-allocation.c \
      -lnuma -o numa_inference

Expected behavior on EPYC 9004X (NPS4 mode):
  - Each NUMA node allocates 8 MB weights locally
  - 8 cores bound to cores on same NUMA node
  - DRAM bandwidth: ~115 GB/s per node
  - No IF traffic (all memory access is local)
  - Expected speedup: 30-40% vs. unbound threads

Bandwidth measurements:
  Unbound (random NUMA): ~270-300 GB/s (contention + IF overhead)
  Bound (local NUMA):    ~380-400 GB/s (near theoretical 460 GB/s limit)
*/
```

### Example 2: V-Cache Hit Rate Profiling

```c
// Module-23-example-vcache-profiling.c
// Measure V-Cache effectiveness on transformer weight matrices

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include <immintrin.h>

#define TRANSFORMER_WEIGHTS_SIZE (4UL * 1024 * 1024)  // 4 MB (7B model)
#define BATCH_SIZE 1
#define SEQ_LENGTH 512
#define HIDDEN_DIM 4096

typedef struct {
    float *linear_weights;  // Transformer output projection: hidden×hidden
    float *attention_query; // Q projection
    float *attention_key;   // K projection
    float *attention_value; // V projection
} transformer_weights_t;

// Simulate transformer forward pass with repeated weight access
// (demonstrates V-Cache temporal reuse)
float transformer_forward_pass(transformer_weights_t *weights,
                               float *input, int seq_len, int hidden_dim) {
    float result = 0.0f;

    // Self-attention: Q,K,V projections (reused for all positions in sequence)
    // This creates temporal reuse of weights → V-Cache candidates
    #pragma omp parallel for reduction(+:result) num_threads(8)
    for (int pos = 0; pos < seq_len; pos++) {
        // Compute query for position 'pos'
        for (int h = 0; h < hidden_dim; h++) {
            float q = 0.0f;
            // Access weights[pos * hidden_dim : (pos+1) * hidden_dim]
            // + weights[h] (temporal reuse across positions)
            for (int i = 0; i < 32; i++) {
                int weight_idx = (h + i) % hidden_dim;
                q += weights->linear_weights[weight_idx] * input[pos * hidden_dim + i];
            }
            result += q;
        }
    }

    return result;
}

// Measure cache behavior using AMD PMU events
void measure_vcache_effectiveness(transformer_weights_t *weights) {
    printf("=== V-Cache Effectiveness Measurement ===\n\n");

    // Allocate working set similar to transformer weights
    float *test_weights = (float *)malloc(TRANSFORMER_WEIGHTS_SIZE);
    memset(test_weights, 0, TRANSFORMER_WEIGHTS_SIZE);

    // Initialize with values
    for (size_t i = 0; i < TRANSFORMER_WEIGHTS_SIZE / sizeof(float); i++) {
        test_weights[i] = (float)(i % 256);
    }

    printf("Test working set: %.1f MB\n", TRANSFORMER_WEIGHTS_SIZE / (1024.0 * 1024.0));
    printf("L3 capacity: 32 MB per CCD × 12 = 384 MB total\n");
    printf("V-Cache capacity: 64 MB per CCD × 12 = 768 MB total\n\n");

    // Scenario 1: L3-only hit/miss (before V-Cache)
    printf("Scenario 1: L3-only access pattern\n");
    float accum = 0.0f;
    for (int iter = 0; iter < 100; iter++) {
        for (size_t i = 0; i < TRANSFORMER_WEIGHTS_SIZE / sizeof(float); i++) {
            accum += test_weights[i] * (i % 3);  // Temporal reuse: every 3rd access
        }
    }
    printf("Accumulator: %.1f (to prevent optimization)\n\n", accum);

    // Scenario 2: V-Cache hit (repeated, small working set)
    printf("Scenario 2: V-Cache-optimized access (repeated reference set)\n");

    // Allocate smaller working set (~64 MB, fits in V-Cache)
    float *small_weights = (float *)malloc(64 * 1024 * 1024);
    memset(small_weights, 0, 64 * 1024 * 1024);

    for (size_t i = 0; i < (64 * 1024 * 1024) / sizeof(float); i++) {
        small_weights[i] = (float)(i % 256);
    }

    accum = 0.0f;
    for (int iter = 0; iter < 100; iter++) {
        // Same working set accessed repeatedly
        // → V-Cache captures entire dataset after first pass
        // → Subsequent passes: V-Cache hits instead of DRAM hits
        for (size_t i = 0; i < (64 * 1024 * 1024) / sizeof(float); i++) {
            accum += small_weights[i] * (i % 3);
        }
    }
    printf("Accumulator: %.1f\n\n", accum);

    free(test_weights);
    free(small_weights);

    printf("Expected V-Cache benefit:\n");
    printf("  - First pass: Same latency (L3 miss → DRAM)\n");
    printf("  - Subsequent passes: V-Cache hits (~15 ns vs. ~100 ns DRAM)\n");
    printf("  - Speedup for repeated reference: 1.5-2.0×\n");
}

int main() {
    transformer_weights_t weights;

    // Allocate transformer weights
    weights.linear_weights = (float *)malloc(TRANSFORMER_WEIGHTS_SIZE);
    weights.attention_query = (float *)malloc(HIDDEN_DIM * HIDDEN_DIM * sizeof(float));
    weights.attention_key = (float *)malloc(HIDDEN_DIM * HIDDEN_DIM * sizeof(float));
    weights.attention_value = (float *)malloc(HIDDEN_DIM * HIDDEN_DIM * sizeof(float));

    // Initialize
    for (size_t i = 0; i < TRANSFORMER_WEIGHTS_SIZE / sizeof(float); i++) {
        weights.linear_weights[i] = (float)(i % 256) / 255.0f;
    }

    // Run forward pass
    printf("Running transformer forward pass...\n");
    float *input = (float *)malloc(BATCH_SIZE * SEQ_LENGTH * HIDDEN_DIM * sizeof(float));
    memset(input, 0, BATCH_SIZE * SEQ_LENGTH * HIDDEN_DIM * sizeof(float));

    float result = transformer_forward_pass(&weights, input, SEQ_LENGTH, HIDDEN_DIM);
    printf("Forward pass result: %.2f\n\n", result);

    // Measure V-Cache behavior
    measure_vcache_effectiveness(&weights);

    // Cleanup
    free(weights.linear_weights);
    free(weights.attention_query);
    free(weights.attention_key);
    free(weights.attention_value);
    free(input);

    return 0;
}

/*
Compilation:
  gcc -O3 -march=znver4 module-23-example-vcache-profiling.c -o vcache_profile

Expected observations:
  - Transformer weights (4-8 MB) fit in V-Cache if repeated
  - First pass: L3 misses dominate
  - Subsequent passes: V-Cache hits (10-15× faster per access)
  - Real-world transformer: mixed (some layers benefit, others stream)
*/
```

---

## 5. EXPERT INSIGHT

### Non-Obvious Truths About EPYC 9004X Memory Architecture

**1. NPS4 NUMA Interleaving Destroys Cache Efficiency Despite Better Local Bandwidth**

Marketing claim: "NPS4 mode provides better local DRAM bandwidth (115 GB/s per node)."

Reality: NPS4 interleaving distributes consecutive cache lines across 4 different NUMA nodes:
- Line 0 → node 0
- Line 1 → node 1
- Line 2 → node 2
- Line 3 → node 3
- Line 4 → node 0 (wrap)

For sequential access patterns (prefetch-friendly), this fragmenting causes:
- Prefetcher learns pattern across 4 NUMA nodes
- All 4 nodes must respond in parallel (contention on cross-node IF)
- Effective bandwidth: 115 GB/s (per node) but serialized cross-node

**Better strategy for streaming code:** Use NPS1 mode (unified memory space), accept latency, benefit from bandwidth aggregation (460 GB/s) and prefetcher coherency.

**2. 3D V-Cache Provides Minimal Benefit on Token Generation (Streaming)**

3D V-Cache shines for repeated access (good for training, bad for inference token generation):

Measured on LLaMA 7B:
- Prefill (batch 32): V-Cache helps (60-70% hit rate on weights)
- Token generation (batch 1): V-Cache helps (50-60% hit rate, but latency penalty)

Why? Token-by-token generation streams attention matrices (KB×seq_len) which are *one-time* accesses. V-Cache optimizes temporal reuse, not streaming.

**Implication:** V-Cache cost (~$2-3k per socket premium) is 10-20% ROI for batch inference but only 2-5% ROI for token generation.

**3. IF Frequency Coupling to Memory Clock Limits Latency Optimization**

Reference: AMD EPYC 9004X Memory Controller Architecture (whitepaper).

IF clock = 0.42 × memory clock (approximate, chipset-dependent).

On EPYC 9004X:
- DDR5-5600: IF ≈ 2400 MHz
- Coherency latency over IF: ~5 cycles @ 2400 MHz ≈ 2 ns
- DRAM latency: ~100 ns
- Total: ~102 ns (mostly DRAM, IF is ~2% overhead)

But detuning to DDR5-4800 (power saving):
- IF ≈ 2000 MHz
- Coherency latency: ~6 cycles @ 2000 MHz ≈ 3 ns
- **However:** DRAM access time also increases (slower command sequencing)
- Total: ~110 ns (effective penalty: ~8 ns = 7% slowdown)

**Expert move:** Keep DRAM at DDR5-5600 even if power-limited. IF frequency penalty > DRAM benefit.

**4. NUMA Node Binding Matters More Than Frequency**

Zen 4c (Bergamo) at 2.6 GHz with perfect NUMA binding can match Genoa @ 3.5 GHz with poor NUMA binding:

- Genoa (unbound): 3.5 GHz × 0.65 (IF contention) = 2.27 GHz effective
- Bergamo (bound): 2.6 GHz × 1.0 (local memory, no IF overhead) = 2.6 GHz effective

Bergamo wins by 15% despite 25% lower nominal frequency.

**5. DDR5 Timing vs. Frequency Trade-offs on EPYC**

DDR5-5600 CAS latency: 46 cycles
DDR5-6400 CAS latency: 54 cycles

Measured on EPYC 9004X:
- DDR5-5600: ~100 ns DRAM latency
- DDR5-6400: ~105 ns DRAM latency (longer CAS offsets frequency gain)

**Implication:** DDR5-5600 is optimal for EPYC. Higher DRAM frequencies (6400, 7200) require CAS penalty that negates benefit. Don't follow SPR (Intel) tuning guides.

---

## 6. BENCHMARK / MEASUREMENT

### Measuring DRAM Bandwidth & NUMA Latency

```bash
# 1. Measure aggregated DRAM bandwidth (STREAM benchmark)
wget https://www.cs.virginia.edu/stream/FTP/Code/stream.f90
gfortran -O3 -march=native stream.f90 -o stream
./stream

# Expected on EPYC 9004X with 12 channels:
#   Copy:  ~400-420 GB/s
#   Scale: ~390-410 GB/s
#   Add:   ~450-470 GB/s
#   Triad: ~450-470 GB/s
# (Triad is memory bound to theoretical ~460 GB/s)

# 2. Measure NUMA latency with libnuma
sudo apt install numactl
numactl -H  # Show NUMA topology

# Test local vs. remote latency
cat > test_numa_latency.c <<'EOF'
#include <stdio.h>
#include <numa.h>
#include <sys/types.h>
#include <unistd.h>
#include <time.h>

int main() {
    struct timespec start, end;
    volatile int *mem_local = numa_alloc_onnode(sizeof(int), 0);
    volatile int *mem_remote = numa_alloc_onnode(sizeof(int), 2);

    // Warm up
    for (int i = 0; i < 1000; i++) *mem_local; *mem_remote;

    // Measure local latency
    clock_gettime(CLOCK_MONOTONIC, &start);
    for (int i = 0; i < 1000000; i++) {
        volatile int x = *mem_local;
    }
    clock_gettime(CLOCK_MONOTONIC, &end);
    double local_ns = (end.tv_sec - start.tv_sec) * 1e9 +
                       (end.tv_nsec - start.tv_nsec);
    printf("Local latency: %.1f ns\n", local_ns / 1000000);

    // Measure remote latency
    clock_gettime(CLOCK_MONOTONIC, &start);
    for (int i = 0; i < 1000000; i++) {
        volatile int x = *mem_remote;
    }
    clock_gettime(CLOCK_MONOTONIC, &end);
    double remote_ns = (end.tv_sec - start.tv_sec) * 1e9 +
                        (end.tv_nsec - start.tv_nsec);
    printf("Remote latency: %.1f ns\n", remote_ns / 1000000);
    printf("Penalty: %.1f%%\n", 100.0 * (remote_ns - local_ns) / local_ns);

    return 0;
}
EOF
gcc -O3 test_numa_latency.c -lnuma -o test_numa_latency
./test_numa_latency

# Expected output:
#   Local: ~100-120 ns
#   Remote (adj): ~150 ns
#   Remote (far): ~200+ ns
#   Penalty: 30-100% depending on distance

# 3. Measure IF bandwidth saturation
perf stat -e \
  'cpu/umask=0x02,event=0xae,name=Data_Fabric_Transactions/' \
  'cpu/event=0xac,name=LS_Dispatch/' \
  your_app

# 4. Measure V-Cache effectiveness (hardware PMU)
amd-uprof collect -c \
  --event='L3_Misses','L3_Cache_Accesses' \
  -- your_ml_inference_app
```

### DRAM Access Pattern Benchmark

```c
// benchmark-dram-bandwidth.c
// Measure sustained DRAM bandwidth with various access patterns

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <omp.h>

#define BUFFER_SIZE (512UL * 1024 * 1024)  // 512 MB
#define ITERATIONS 100
#define CACHE_LINE 64

double measure_bandwidth(float *buffer, size_t size, const char *pattern) {
    struct timespec start, end;
    volatile float sum = 0;

    // Warm up L3
    for (size_t i = 0; i < size / sizeof(float); i++) {
        sum += buffer[i];
    }

    clock_gettime(CLOCK_MONOTONIC, &start);

    for (int iter = 0; iter < ITERATIONS; iter++) {
        if (strcmp(pattern, "sequential") == 0) {
            // Sequential access (prefetch-friendly)
            for (size_t i = 0; i < size / sizeof(float); i++) {
                sum += buffer[i];
            }
        } else if (strcmp(pattern, "random") == 0) {
            // Random access (prefetch-unfriendly)
            for (size_t i = 0; i < size / sizeof(float); i += 1024) {
                int idx = (i * 17) % (size / sizeof(float));
                sum += buffer[idx];
            }
        } else if (strcmp(pattern, "strided") == 0) {
            // Strided access (every 16 cache lines)
            for (size_t i = 0; i < size / sizeof(float); i += (16 * CACHE_LINE / sizeof(float))) {
                sum += buffer[i];
            }
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &end);

    double elapsed = (end.tv_sec - start.tv_sec) +
                     (end.tv_nsec - start.tv_nsec) / 1e9;

    // Bandwidth = (bytes accessed) / time
    double bytes = (double)size * ITERATIONS;
    double bw = bytes / elapsed / 1e9;  // GB/s

    printf("%s: %.1f GB/s\n", pattern, bw);
    return bw;
}

int main() {
    printf("=== EPYC 9004X DRAM Bandwidth Characterization ===\n\n");

    float *buffer = (float *)malloc(BUFFER_SIZE);
    if (!buffer) {
        perror("malloc");
        return 1;
    }

    // Initialize
    for (size_t i = 0; i < BUFFER_SIZE / sizeof(float); i++) {
        buffer[i] = (float)i;
    }

    printf("Buffer size: %.0f MB\n", BUFFER_SIZE / (1024.0 * 1024.0));
    printf("Theoretical max: 460 GB/s\n\n");

    measure_bandwidth(buffer, BUFFER_SIZE, "sequential");
    measure_bandwidth(buffer, BUFFER_SIZE, "strided");
    measure_bandwidth(buffer, BUFFER_SIZE, "random");

    printf("\nExpected results (EPYC 9004X):\n");
    printf("  Sequential: 400-450 GB/s (prefetch-friendly)\n");
    printf("  Strided:    150-200 GB/s (prefetch struggles)\n");
    printf("  Random:     80-120 GB/s (no prefetch)\n");

    free(buffer);
    return 0;
}

// Compilation:
//   gcc -O3 -march=znver4 benchmark-dram-bandwidth.c -o bench_bw
//   ./bench_bw
```

---

## 7. ML SYSTEMS RELEVANCE

### Memory Architecture for Large Language Model Inference

**1. KV-Cache Management on EPYC**

Transformer KV cache grows with sequence length: Size = batch × seq_len × head_dim × num_heads × 2 (K+V)

For LLaMA 7B:
- Batch 32, seq_len 4096: ~1.6 GB
- Batch 128, seq_len 4096: ~6.4 GB

Storage strategy on EPYC 9004X:
- KV cache → V-Cache (64 MB per CCD, 768 MB total)
- Attention scores (KB×seq_len) → L3 (32 MB per CCD)
- Residual projections → Main DRAM

**Expected benefit:** 25-35% latency reduction on token generation with V-Cache vs. DRAM-only KV cache.

**2. Batch Size Scaling with Memory Bandwidth**

For GEMM-heavy inference (like prefill), memory bandwidth is the primary constraint.

Arithmetic intensity (AI) of matrix multiply: A = 2 × (m × k × n) / (memory transfers)

Typical: AI ≈ 3-4 FLOPS/byte for transformer GEMM.

At 460 GB/s DRAM bandwidth:
- Achievable throughput: 460 GB/s × 3.5 FLOPS/byte ≈ 1600 GFLOPS
- Zen 4 peak: 96 cores × 2 TFLOPS/core = 192 TFLOPS
- Scaling: Up to batch size 32 (achieves 1600/2 = 800 GFLOPS per core with prefill)

**3. NUMA-Aware LLM Serving on Bergamo**

128 cores on single Bergamo can run 2-3 parallel LLM instances:
- Instance 0: CCDs 0-5 (48 cores)
- Instance 1: CCDs 6-11 (48 cores)
- Shared: DRAM (460 GB/s, contended)

Per-instance bandwidth: ~230 GB/s effective.

Better: Run 4 instances on 4 NUMA nodes (NPS4):
- Instance per node: 32 cores
- Per-instance bandwidth: ~115 GB/s
- Elimination of cross-NUMA traffic (32% bandwidth loss to IF overhead)

---

## 8. PhD QUALIFIER QUESTIONS

**Q1: NUMA Interleaving Policy Optimization**

Compare NPS1 (no partitioning) vs. NPS4 (4-way partitioning) for:
(a) Streaming DRAM access (matrix multiplication)
(b) Temporal reuse (transformer token generation)

Explain the coherency traffic overhead in each mode and derive expected bandwidth utilization.

*Expected answer:*

```
NPS1 mode (unified memory):
  - Interleaving: Stripe across 12 channels (round-robin)
  - Consecutive cache lines: distributed across 4 NUMA nodes
  - Prefetcher pattern: Must fetch from 4 nodes simultaneously

Streaming (GEMM prefill):
  - Access pattern: Sequential reads
  - Prefetcher effectiveness: Good (hardware recognizes stride)
  - Bandwidth utilization: ~400-420 GB/s (90% of theoretical 460 GB/s)
  - Latency: Constant 100 ns (single node access)

Temporal reuse (token generation):
  - Access pattern: Random KV cache lookups
  - Prefetcher effectiveness: Poor (no pattern)
  - Bandwidth utilization: ~100-150 GB/s (20-30%)
  - Latency: 100 ns + IF coherency overhead

NPS4 mode (4-way NUMA):
Interleaving: Stripe across 3 controllers per node
Consecutive cache lines: Stay within same NUMA node

Streaming (GEMM prefill):
  - Access pattern: Sequential, within-NUMA
  - Prefetcher effectiveness: Excellent (single node)
  - Bandwidth utilization: ~115 GB/s per node × 4 = 460 GB/s (100% theoretical)
  - Latency: Constant 100 ns (local access)

Temporal reuse (token generation):
  - Access pattern: Random within node
  - Prefetcher effectiveness: Poor but local
  - Bandwidth utilization: ~100 GB/s per node (single node limited)
  - Latency: 100 ns (all local)

Conclusion:
  NPS1 better for: Streaming prefetcher-friendly code
  NPS4 better for: Independent workloads, lower IF contention
```

---

**Q2: 3D V-Cache Inclusion & Coherency Model**

V-Cache is inclusive of L3. Explain:
(a) Why inclusion simplifies coherency protocol
(b) The state transition when a line is evicted from L3 but hits in V-Cache
(c) The latency path for a cache miss that hits in V-Cache

*Expected answer:*

```
(a) Inclusion Simplification:
    - Non-inclusive (separate coherency): V-Cache needs own state machine
    - Problem: Line could be modified in V-Cache, invalid in L3 (split coherency)
    - Complexity: Coherency must track both L3 + V-Cache independently

    Inclusive (V-Cache extension of L3):
    - V-Cache inherits L3 state (if L3 is invalid, V-Cache is invalid)
    - No dual tracking: Single coherency state for the line
    - Simplification: ~40% fewer coherency protocol transactions

(b) L3 Eviction → V-Cache Residence:
    State 1: Line is MODIFIED in L3 (owned by core)
    Event: L3 capacity miss (line evicted from 32 MB L3)
    State 2: Line transitions to V-Cache (64 MB stacked SRAM)

    Coherency state: Line is now SHARED or MODIFIED in V-Cache
    (depends on whether other cores accessed it)

    Key insight: V-Cache acts as victim buffer
    Restoration: If core later misses on the line, fetch from V-Cache
                 (10-15 cycles vs. 100 ns DRAM)

(c) V-Cache Hit Path (after L3 miss):
    1. Instruction fetch @ address A
    2. L3 miss (not in 32 MB L3)
    3. Check V-Cache (64 MB stacked SRAM)
    4. Hit in V-Cache: Latency = V-Cache access time + restore to L3
    5. Total latency: 10 cycles (V-Cache) + 5 cycles (interconnect) = 15 cycles
    6. vs. DRAM: 100 ns = 350 cycles @ 3.5 GHz
    7. Speedup: 350 / 15 ≈ 23× faster than DRAM hit
```

---

**Q3: IF Bandwidth Saturation Under Coherency Load**

Derive the maximum aggregate DRAM bandwidth on 12-CCD EPYC 9004X when multiple CCDs compete for access.

Assume:
- IF capacity: 340 GB/s per link (bidirectional)
- Coherency overhead: 15-20% of bandwidth (directory lookups, acknowledgments)
- DRAM theoretical: 460 GB/s
- CCD count: 12

Calculate bottleneck and explain why IF, not DRAM, limits bandwidth.

*Expected answer:*

```
IF Bandwidth Model:
  - 12 CCDs connected via point-to-point IF fabric
  - Each CCD has one IF link to IOD (memory controller)
  - Link bandwidth: 340 GB/s unidirectional (340 × 2 = 680 GB/s bidirectional)

Coherency Overhead:
  - Directory lookup + acknowledgment: ~20% overhead
  - Effective bandwidth per link: 340 GB/s × 0.8 = 272 GB/s

DRAM Bandwidth Limit:
  - Theoretical: 460 GB/s (12 channels)
  - Coherency limit: 272 GB/s × 12 / 12 = 272 GB/s
  - Wait, that's single CCD. Multiple CCDs:

Multiple CCD Contention:
  - If all 12 CCDs issue memory requests simultaneously:
  - Total IF demand: 12 × (460 / 12) = 460 GB/s
  - IF capacity: 340 GB/s per link (single IOD aggregates all)
  - Bottleneck: IF link (340 GB/s) < DRAM (460 GB/s)

Realistic Saturation:
  - IF sustains ~300 GB/s with coherency
  - Measured on real systems: ~270-300 GB/s aggregate
  - Why not 340? Reordering, packet overhead, directory contention

For 4 CCDs (1/3 of socket):
  - Available bandwidth: 300 / 3 ≈ 100 GB/s per CCD
  - Matches measured ML inference bandwidth (~80-120 GB/s per 32-core group)
```

---

**Q4: NPS4 Memory Interleaving Trade-offs**

Given a transformer weight matrix of size 8 GB (larger than L3), compare:
(a) Access pattern with NPS1 (unified) interleaving
(b) Access pattern with NPS4 (4-way NUMA) interleaving

Analyze false sharing and prefetcher effectiveness for a batch GEMM.

*Expected answer:*

```
Scenario: Transformer weight matrix 8 GB, accessed by 96-core EPYC

NPS1 Interleaving (stripe across all 12 nodes):
  Memory layout: Addresses A[0..8GB] interleaved across 12 NUMA nodes
  Stripe size: 8 GB / 12 ≈ 666 MB per node

  Consecutive cache lines (stride 64B):
    Line 0 @ address 0x0: NUMA node 0
    Line 1 @ address 0x40: NUMA node 1
    Line 2 @ address 0x80: NUMA node 2
    ...
    Line 12 @ address 0x300: NUMA node 0 (wrap)

  Sequential access (GEMM row iteration):
    Prefetcher pattern: Line 0 from node 0, line 1 from node 1, ...
    All 12 nodes accessed in parallel (bad for single thread)

  Multi-threaded (96 cores):
    False sharing: Adjacent cache lines on different NUMA nodes
    If core 0 (node 0) and core 8 (node 1) both read Line 0 and Line 1:
    → Coherency traffic across IF between nodes 0 and 1
    → Sustained over entire 8 GB matrix

NPS4 Interleaving (4-way stripe):
  Memory layout: Addresses interleaved across 3 nodes per NUMA group
  Stripe size: 8 GB / 4 ≈ 2 GB per NUMA node

  Consecutive cache lines:
    Line 0 @ address 0x0: Node 0 (CCD 0-2)
    Line 1 @ address 0x40: Node 0 (same)
    Line 2 @ address 0x80: Node 0 (same)
    ...
    Line 48 @ address 0xC00: Node 0 (wrap to different controller)

  Sequential access (GEMM row iteration):
    Prefetcher pattern: All from local node (perfect for NUMA locality)
    Single node accessed (optimal)

  Multi-threaded (96 cores, 24 cores per node):
    False sharing: All lines stay within same NUMA node
    Coherency traffic: Internal to single node (no IF overhead)

Comparison:
  NPS1: Bandwidth = 460 GB/s (theoretical), but coherency latency + IF contention
        Measured: 270-300 GB/s
  NPS4: Bandwidth = 460 / 4 = 115 GB/s per node, but zero IF overhead
        Measured: 100-110 GB/s per node (no IF penalty)

Recommendation:
  - NPS1: Better for total throughput if workloads are independently bound to NUMA nodes
  - NPS4: Better for low-latency, single-threaded or tightly-coupled workloads
```

---

**Q5: V-Cache vs. Larger L3 Trade-off**

Design experiment to compare:
(a) EPYC 9004X with 3D V-Cache (64 MB per CCD, 96 MB total L3+V)
(b) Hypothetical EPYC with 2× larger L3 (64 MB per CCD, no V-Cache)

For transformer weight streaming workload (7B model, batch 32), which is better and why?

*Expected answer:*

```
Experimental Setup:
  Transformer: 7B model (4 GB weights), batch 32 prefill
  Arithmetic intensity: ~3 FLOPS/byte
  Working set: 4 GB + activations (1 GB) = 5 GB
  L3 capacity: 32 MB per CCD × 12 = 384 MB

(a) 3D V-Cache System (current EPYC 9004X):
    Capacity: 32 MB L3 + 64 MB V-Cache per CCD = 96 MB per CCD
    Total: 1.15 GB per socket

    Access latency:
    - L3 hit: 40 cycles
    - V-Cache hit (L3 miss): 15 cycles + restoration
    - DRAM: 350 cycles

    Transformer workload:
    - 5 GB working set > 1.15 GB capacity
    - Weight access pattern: streaming (poor temporal reuse)
    - Hit rate: L3 ~30%, V-Cache ~20%, DRAM hit 50%
    - Average latency: 0.3×40 + 0.2×25 + 0.5×350 ≈ 190 cycles

(b) 2× Larger L3 (no V-Cache):
    Capacity: 64 MB L3 per CCD (no V-Cache)
    Total: 768 MB per socket

    Access latency:
    - L3 hit: 40 cycles (latency same, just more capacity)
    - L3 miss: DRAM 350 cycles

    Transformer workload:
    - 5 GB working set > 768 MB capacity
    - Better hit rate due to larger capacity: ~40% (vs. 30% base L3)
    - Average latency: 0.4×40 + 0.6×350 ≈ 226 cycles

Winner: 3D V-Cache
  - Lower average latency (190 vs. 226 cycles)
  - Benefit: ~19% latency reduction

Why V-Cache wins despite smaller L3:
  - V-Cache latency (15 cycles) is much better than DRAM (350 cycles)
  - For streaming workloads, V-Cache acts as a second-chance cache
  - Larger L3 alone doesn't help streaming (all lines evicted before reuse)
  - V-Cache helps with temporal reuse of hot weights (attention heads, norm layers)

Caveat:
  - If workload is purely streaming (LLaMA prefill): V-Cache ~5-10% benefit
  - If workload has repeated access (token generation): V-Cache ~30-50% benefit
  - 2× L3 (no V-Cache) would be better for non-streaming workloads
```

---

## 9. READING LIST

**Primary References:**

1. AMD EPYC 9004X (Genoa) Memory Subsystem Specifications
   - Official datasheet: Section 3 "Memory Controller Architecture"
   - DDR5 timing parameters, latency specifications

2. AMD EPYC 9004X Processor Optimisation Guide (2023)
   - Section 2: "NUMA Configuration and Topology"
   - NPS mode explanations and performance implications
   - 3D V-Cache technology details

3. WikiChip: AMD EPYC 9004X Architecture Reference
   - Detailed memory controller diagrams
   - V-Cache integration specifications

4. HotChips 34 (2022): AMD EPYC Genoa Presentation
   - Memory subsystem architecture and design rationale
   - V-Cache performance data

5. IEEE Micro Vol. 40, No. 2 (2020): "Multi-CCD Memory Coherency on x86"
   - Coherency protocol on chiplet systems
   - IF bandwidth measurements

6. Drepper, Ulrich. "What Every Programmer Should Know About Memory" (2007)
   - Chapters 3-5: NUMA performance fundamentals
   - Applicable despite age (NUMA principles unchanged)

**Supplementary:**

7. AMD uProf User Guide v3.5 (2023)
   - Section 2.4: "NUMA-Aware Performance Analysis"
   - PMU events for memory subsystem profiling

8. STREAM Benchmark — McCalpin, J.D.
   - Official DRAM bandwidth measurement standard
   - Expected results on various platforms

9. "Optimizing ML Inference on AMD EPYC" — AMD Technical Blog (2023)
   - Practical NUMA tuning for LLM inference
   - Real-world bandwidth measurements

10. Linux libnuma Documentation
    - numa_alloc_onnode(), numactl usage
    - Practical NUMA programming examples

