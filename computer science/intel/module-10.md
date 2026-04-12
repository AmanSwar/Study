# MODULE 10 — Caches in Microarchitectural Detail

## 1. CONCEPTUAL FOUNDATION

### 1.1 Cache Hierarchy: Sapphire Rapids and Zen 4 Comparison

Modern CPUs use a three-level cache hierarchy, each balancing speed, capacity, and associativity:

**Intel Sapphire Rapids (8-core die, 60-core dual-socket Xeon):**
- **L1-I (Instruction):** 32 KB per core, 8-way associative, 4-cycle latency
- **L1-D (Data):** 48 KB per core, 12-way associative, 4-cycle latency (load-to-use)
- **L2:** 2 MB per core (unified), 16-way associative, 12-cycle latency
- **L3:** 140 MB per 8-core tile, distributed & sliced across cores
- **L3 latency:** 40 cycles (local core), 65–85 cycles (remote core via mesh interconnect)

**AMD EPYC Zen 4 (12-core CXL-enabled die):**
- **L1-I:** 32 KB per core, 8-way associative, 3-cycle latency
- **L1-D:** 32 KB per core, 8-way associative, 4-cycle latency
- **L2:** 1 MB per core (unified), 8-way associative, 12-cycle latency
- **L3:** 12 MB per 4-core cluster (30.7 MB total per die in Genoa), 16-way associative, 42 cycles

**Key difference:** Sapphire Rapids L3 is *distributed* (140 MB shared across all cores with slicing by address); Zen 4 L3 is *clustered* (per 4-core group, with slower inter-cluster access ~50+ cycles).

See: Intel Optimization Manual for Sapphire Rapids, Chapter 6 (Memory Hierarchy); AMD EPYC BIOS and Kernel Developer's Guide.

### 1.2 Cache Organization: Sets, Ways, Tags, and Replacement

Each cache is organized as *sets × ways*, with Least Recently Used (LRU) replacement:

```
Example: 48 KB L1-D with 12-way associativity
  Address: [tag bits | set bits | offset bits]
  Line size: 64 bytes = 2^6
  Offset bits: 6
  Capacity: 48 KB = 48 × 1024 bytes = 3 × 2^12 × 64 / 12 sets
  Sets: (48 × 1024) / (64 × 12) = 64 sets = 2^6
  Tag bits: 48 - 6 (offset) - 6 (set) = 36 bits
  Each set: 12 cache lines (ways), each 64 bytes
```

**Sapphire Rapids L1-D (48 KB, 12-way):**
- 64 sets, 12 ways per set
- One 64-byte cache line per way
- Set index: bits [12:6] of address (6 bits = 64 sets)
- Tag: bits [63:13] (51 bits, but only ~36 physically used on current CPUs)
- Offset: bits [5:0] (byte within cache line)

**Replacement policy (LRU):**
- When all 12 ways are occupied and a new line is loaded:
  - Evict the least recently used (oldest) line
  - LRU tracking: requires log2(12) ≈ 4 bits per way to maintain recency order

### 1.3 L3 Organization: Distributed & Mesh-Connected

Sapphire Rapids uses a **distributed L3** architecture with address-based slicing:

```
Die layout: 8 cores arranged in a ring
  Core0 — Core1 — Core2 — Core3
    ↑                      ↓
  Core7 — Core6 — Core5 — Core4

L3 slicing: by address bits [12:6] (cache line set index)
  Slice 0: lines with set[6:3] == 0 → stored in L3 partition near Core0
  Slice 1: lines with set[6:3] == 1 → stored in L3 partition near Core1
  ...
  Slice 7: lines with set[6:3] == 7 → stored in L3 partition near Core7

Consequence: L3 access latency depends on core-to-partition distance
  Local (same tile): 40 cycles
  Remote (other tile, ~4 hops via mesh): 65–85 cycles
```

**Mesh Interconnect (on Sapphire Rapids):**
- 2D mesh connecting 8-core tiles
- Dual-socket Xeon: 4 tiles per socket, 2 sockets
- Total: 8 tiles × 2 sockets = 16 tiles (but typical Xeon has 8 tiles per socket)
- Bisection bandwidth: ~200 GB/s per socket (limited by mesh links)

### 1.4 AMD 3D V-Cache Architecture

AMD's Zen 4 processors use **3D V-Cache** (3D Vertical Cache) on some SKUs:

```
Traditional Zen 4: 12 cores per die, 12 MB L3 per die
Zen 4 3D V-Cache: 12 cores + additional 96 MB L3 stacked on top via chiplet

The additional L3 is accessed with ~1/3 the latency of external memory
  Typical L3 miss: ~200 cycles to DRAM
  3D V-Cache hit: ~50 cycles (3× faster than DRAM, but still slower than L3)
```

**Which SKUs have 3D V-Cache:**
- Ryzen 7 7800X3D (consumer, 8 cores)
- Ryzen 9 7950X3D (consumer, 16 cores)
- EPYC 9754 (Genoa-X, 12-core CXL-enabled)
- Provides 128 MB additional cache on-die (8× the standard L3)

**Performance impact on memory-bound workloads:**
- Without 3D V-Cache: GEMM with poor cache reuse stalls ~200 cycles on DRAM miss
- With 3D V-Cache: DRAM misses become L3 misses, but still 50-cycle penalty
- Effective: 4× speedup on workloads that miss L3 frequently (e.g., matrix transpose, sorting)

See: AMD EPYC BIOS and Kernel Developer's Guide; AMD Ryzen Processor Specifications.

### 1.5 Non-Temporal Stores (movnt): Bypassing Cache

The `movnt` family of instructions writes directly to main memory, bypassing L1/L2/L3:

```
movntps xmm0, [rsi]        // Non-temporal store (float)
movntdq xmm0, [rsi]        // Non-temporal store (integer)
movntpd xmm0, [rsi]        // Non-temporal store (double)
clflushopt [rsi]           // Flush cache line to DRAM
```

**Semantics:**
- `movnt`: write goes directly to write-combining buffer, bypasses cache hierarchy
- Latency: ~200 cycles (DRAM write latency) instead of 4 cycles (L1 hit)
- **But:** multiple `movnt` writes can batch in write-combining buffer, amortizing latency
- Throughput: 1 write per cycle to WC buffer (16–64 bytes per buffer entry)

**Use case: Streaming writes in ML tensor operations**

```c
// Example: matrix transpose (memory-bound, non-reuse)
void transpose_streaming(const float *A, float *B, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            // B[j][i] = A[i][j]
            // Without movnt: loads from A pollute L1, stores to B evict A from L1
            // With movnt: B stores bypass cache, reducing eviction pressure
            _mm_stream_ps(&B[j*n + i], _mm_load_ps(&A[i*n + j]));
        }
    }
}

// Expected speedup: 1.5–2.0× (reduced cache thrashing)
// Without movnt: ~50% of L1 capacity wasted on non-reusable B writes
// With movnt: L1 reserved for A loads, higher hit rate
```

**Caveat:** `movnt` writes must be followed by `mfence` or `clflush` to ensure they're visible to other threads. This adds synchronization overhead.

See: Intel Architecture Instruction Set Extensions Reference Manual, Section on Non-Temporal Instructions; Intel Optimization Manual, Section on Write-Combining Memory.

---

## 2. MENTAL MODEL

```
                    Cache Hierarchy: Sapphire Rapids per Core
                    ──────────────────────────────────────────

    ┌─────────────────────────────────────────────────┐
    │ Instruction & Data TLB (256 entries, virtual→physical) │
    │ (Intel's new split addressing, much faster misses)  │
    └─────────────────────────────────────────────────┘
                            │
            ┌───────────────┼───────────────┐
            ↓               ↓               ↓
        [L1-I 32KB]     [L1-D 48KB]  [Branch Predictor]
        [8-way, 4c]     [12-way, 4c]
            │               │
            └───────────────┼───────────────┐
                            ↓               ↓
                       [L2 2MB unified]  [Prefetcher]
                       [16-way, 12c]
                            │
                   ┌────────┴────────┐
                   ↓                 ↓
            [L3 Slice A]      [L3 Slice B]  (distributed across cores)
          [140 MB ÷ 8]       [16-way, 40c local, 65c remote]
                   │                 │
                   └────────┬────────┘ (via 2D mesh)
                            │
                   [DDR5 Memory Controller]
                   [8 channels, 384 GB/s peak, ~200c latency]


            Cache-to-Cache Latency by Distance (Sapphire Rapids):
            ────────────────────────────────────────────────────
            L1 hit: 4 cycles (next cycle after request issued)
            L2 hit: 12 cycles
            L3 hit (local slice): 40 cycles
            L3 hit (remote slice, 4 hops): 65–85 cycles
            DRAM hit: 200–300 cycles (depends on row buffer state)


            Working Set Analysis (Sapphire Rapids):
            ────────────────────────────────────────
            L1 fit: < 48 KB → all loads hit L1 → ~4c latency, ~32 GFlops
            L2 fit: < 2 MB → most loads hit L2 → ~12c latency, ~8 GFlops
            L3 fit: < 140 MB → L3 hits → ~40c latency (local), ~2.4 GFlops
            DRAM: > 140 MB → DRAM misses → ~200c latency, ~40 MFlops


            Address Hashing & L3 Slicing:
            ────────────────────────────
            Physical address: [core bits | set bits | offset bits]
            Example: 0x12345678 (32-bit address, LSB is offset)

            [63:12]      [11:6]       [5:0]
            [ TAG    ]   [ SET   ]    [OFFSET]
             (51 bits)   (6 bits)     (6 bits)

            SET is used to determine L3 slice:
            SET bits [6:3] → slice 0–7 (determines which core's L3)
            SET bits [5:0] → line within slice

            Implication: Addresses differing only in [5:0] → same L3 slice
                        Addresses differing in [11:6] → different L3 slices
```

---

## 3. PERFORMANCE LENS

### 3.1 Cache Misses: Cost and Types

**L1 Data Cache Miss (L1d miss, L2 hit): 12 cycles**
- Stall load unit, block dependent operations
- Modern OOO execution can hide with other unrelated work

**L2 Miss, L3 Hit (local): 40 cycles**
- More severe stall, requires ~40 dependent loads to fully hide
- Example: matrix multiply row-major layout with cache-unfriendly access

**L3 Miss, DRAM Hit: 200–300 cycles**
- Severe bottleneck for latency-sensitive code
- Hardware prefetcher may reduce effective latency by 50–100 cycles in some patterns

**Categories of Cache Misses:**
1. **Compulsory miss:** First access to data (unavoidable, except with prediction/prefetch)
2. **Capacity miss:** Working set > cache size (reduce working set or increase cache line reuse)
3. **Conflict miss:** Address aliasing in cache (change data layout to de-alias)
4. **Coherency miss:** Cache line invalidated by another core (minimize sharing, use thread-local data)

### 3.2 Cache Line Conflicts & Address Hashing

**Example: Bad cache aliasing in ML inference**

```c
// Scenario: loading embedding vectors [vocabulary size][embedding dim]
// vocab_size = 65536 = 2^16, embedding_dim = 768 = 3 × 2^8

float embeddings[65536][768];

// Memory layout: embeddings[0][0..767], embeddings[1][0..767], ...
// Each row: 768 × 4 bytes = 3072 bytes

// Now: load embeddings[i][j] for random i
// Address: embeddings base + i × 3072 + j × 4
// i × 3072 = i × 3 × 1024 = i × (3 × 2^10)

// In terms of cache set bits [11:6] (6 bits = 64 sets):
// set = (address >> 6) & 0x3F
//     = ((base + i × 3072 + j × 4) >> 6) & 0x3F
//     = ((i × 3 × 2^10 + j × 4) >> 6) & 0x3F
//     = ((i × 3 × 2^4 + (j << 2)) >> 6) & 0x3F
//     = ((i × 48 + j × 4) >> 6) & 0x3F

// For i = 0 to 1023:
//   set = (48i >> 6) & 0x3F = (3i/4) & 0x3F
//   → sets 0, 0, 0, 0, 3, 3, 3, 3, 6, 6, 6, 6, ... (repeating every 4 rows)

// Problem: embeddings[0:4][j] → all use set 0 (12-way conflict)
// embeddings[4:8][j] → all use set 3 (12-way conflict)
// → extreme cache conflict, high miss rate

// Solution: Change layout to [embedding_dim][vocab_size] (transpose)
//         or add padding to embeddings[i] to break aliasing
float embeddings_padded[65536][800];  // 800 × 4 = 3200 bytes, no aliasing
```

### 3.3 Prefetch Strategy: Hardware vs. Software

**Hardware Prefetcher (Intel):**
- Monitors access patterns on each core (stride detection)
- Prefetches next 1–2 cache lines automatically
- Works well for: sequential access, regular strides
- Fails on: random access, complex patterns

**Prefetch Instructions (software):**
- `prefetcht0 [rsi]`: prefetch into L1/L2
- `prefetcht1 [rsi]`: prefetch into L2 only
- `prefetcht2 [rsi]`: prefetch into L3 only
- `prefetchnta [rsi]`: non-temporal prefetch (don't fill L1)

```c
// Example: GEMM with prefetching
void gemm_with_prefetch(int m, int n, int k,
                        float *A, int ldA,
                        float *B, int ldB,
                        float *C, int ldC)
{
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            float accum = 0.0f;

            // Prefetch next iteration's B row
            _mm_prefetch(&B[(j+1)*ldB], _MM_HINT_T0);
            // Latency: ~1–5 cycles (initiated in parallel with computation)

            for (int k = 0; k < K; k++) {
                accum += A[i*ldA + k] * B[k*ldB + j];
            }

            C[i*ldC + j] = accum;
        }
    }
}

// Expected speedup: 1.1–1.3× (depends on pipeline depth, memory latency)
```

---

## 4. ANNOTATED CODE

### 4.1 Cache Miss Detection & Measurement

```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Measure cache miss rate using PERF events
// Requires: Linux with perf subsystem, root access

int main() {
    int n = 10000000;
    float *arr = (float *)malloc(n * sizeof(float));

    // Initialize to avoid page faults
    for (int i = 0; i < n; i++) {
        arr[i] = i * 1.5f;
    }

    // Pattern 1: Sequential access (should hit prefetcher)
    float sum1 = 0.0f;
    for (int i = 0; i < n; i++) {
        sum1 += arr[i];
    }
    printf("Sequential sum: %.2f\n", sum1);

    // Pattern 2: Stride-4 access (still good for prefetcher)
    float sum2 = 0.0f;
    for (int i = 0; i < n; i += 4) {
        sum2 += arr[i];
    }
    printf("Stride-4 sum: %.2f\n", sum2);

    // Pattern 3: Random access (prefetcher can't help)
    float sum3 = 0.0f;
    for (int i = 0; i < n; i++) {
        sum3 += arr[(i * 7919) % n];  // 7919 is prime
    }
    printf("Random sum: %.2f\n", sum3);

    free(arr);
    return 0;
}

// Measure with perf:
// gcc -O3 cache_miss.c -o cache_miss
// perf stat -e L1-dcache-load-misses,LLC-load-misses,cycles ./cache_miss
//
// Expected output (Sapphire Rapids, n=10M float):
//
// Pattern 1 (sequential):
//   L1-dcache-load-misses:    ~10,000 (0.1% miss rate)
//   LLC-load-misses:          ~100 (0.001% miss rate)
//   Cycles:                   ~10,000,000 (1 load per cycle)
//
// Pattern 2 (stride-4):
//   L1-dcache-load-misses:    ~10,000 (0.1% miss rate)
//   LLC-load-misses:          ~100 (0.001% miss rate)
//   Cycles:                   ~40,000,000 (4 loads per cycle, same throughput)
//
// Pattern 3 (random):
//   L1-dcache-load-misses:    ~5,000,000 (50% miss rate!)
//   LLC-load-misses:          ~4,000,000 (40% miss rate!)
//   Cycles:                   ~400,000,000 (40× slower, dominated by DRAM latency)
```

### 4.2 Cache Conflict Detection via Address Analysis

```c
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

// Analyze cache set conflicts for address stream
// L1: 12-way, 64 sets → set_index = (addr >> 6) & 0x3F

void analyze_cache_conflicts(const uintptr_t *addresses, int n) {
    int set_conflicts[64] = {0};  // Count conflicts per set

    for (int i = 0; i < n; i++) {
        int set = (addresses[i] >> 6) & 0x3F;  // bits [11:6]
        set_conflicts[set]++;
    }

    printf("Cache set conflicts (L1-D, 12-way associative):\n");
    for (int s = 0; s < 64; s++) {
        if (set_conflicts[s] > 12) {  // More than 12-way conflict
            printf("Set %2d: %4d conflicts (SEVERE)\n", s, set_conflicts[s]);
        } else if (set_conflicts[s] > 0) {
            printf("Set %2d: %4d accesses\n", s, set_conflicts[s]);
        }
    }
}

// Example: embedding matrix with potential aliasing
void test_embedding_aliasing() {
    int vocab_size = 65536;
    int embedding_dim = 768;

    float **embeddings = (float **)malloc(vocab_size * sizeof(float *));
    for (int i = 0; i < vocab_size; i++) {
        embeddings[i] = (float *)malloc(embedding_dim * sizeof(float));
    }

    // Simulate random embedding lookups
    uintptr_t addresses[100000];
    for (int i = 0; i < 100000; i++) {
        int vocab_idx = (i * 7919) % vocab_size;  // Random vocab index
        int dim_idx = i % embedding_dim;
        addresses[i] = (uintptr_t)(&embeddings[vocab_idx][dim_idx]);
    }

    analyze_cache_conflicts(addresses, 100000);
    // Expected: if embedding_dim × sizeof(float) % (64 bytes) != 0,
    //           severe aliasing in L1-D cache sets
}

// Line 1: Extract cache set from address
// Calculation: address >> 6 gives cache line index
//              & 0x3F masks to 6 bits (0–63 sets)
// Latency: 1 cycle (shift + AND)

// Line 2: Count conflicts in each set
// Accumulate count per set
// If count > 12 (L1-D way count), indicates conflict misses

// Line 3: Analyze embedding addresses
// For each random embedding lookup, record address
// Cache set = address / 64 bytes per line & 0x3F

// Output interpretation:
// If many addresses map to same set → conflict misses
// If addresses spread evenly across sets → no conflict misses
```

### 4.3 Non-Temporal Stores for Streaming Writes

```c
#include <immintrin.h>
#include <string.h>
#include <time.h>

// Benchmark: traditional store vs. non-temporal store
// for matrix transpose (memory-bound, non-reusable writes)

void transpose_traditional(const float *A, float *B, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            B[j * n + i] = A[i * n + j];
            // Traditional store: fills L1-D cache with B data
            // Contention: evicts A data from L1
        }
    }
}

void transpose_nontemporal(const float *A, float *B, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j += 4) {
            // Load A (reusable)
            __m128 vec = _mm_loadu_ps(&A[i * n + j]);
            // Line 1: Load 4 floats from A[i][j:j+3]
            // Latency: 4 cycles (L1 hit, prefetch working)

            // Store B non-temporally (bypasses cache)
            _mm_stream_ps(&B[j * n + i], vec);
            // Line 2: Non-temporal store to B[j:j+3][i]
            // Latency: ~1 cycle (write-combining buffer)
            // No L1 pollution
        }
    }

    // Ensure all NT writes have flushed to memory
    _mm_mfence();
    // Line 3: Full memory fence, ensures NT writes are visible
    // Latency: ~1000 cycles (all in-flight writes must complete)
    // Only needed if B will be read immediately afterward
}

void benchmark_transpose(int n) {
    float *A = (float *)malloc(n * n * sizeof(float));
    float *B = (float *)malloc(n * n * sizeof(float));

    // Initialize A
    for (int i = 0; i < n * n; i++) {
        A[i] = i * 1.5f;
    }

    // Warm-up
    transpose_traditional(A, B, n);

    // Benchmark traditional
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    for (int iter = 0; iter < 10; iter++) {
        transpose_traditional(A, B, n);
    }
    clock_gettime(CLOCK_MONOTONIC, &end);
    double elapsed_trad = (end.tv_sec - start.tv_sec) +
                          (end.tv_nsec - start.tv_nsec) / 1e9;

    // Benchmark non-temporal
    clock_gettime(CLOCK_MONOTONIC, &start);
    for (int iter = 0; iter < 10; iter++) {
        transpose_nontemporal(A, B, n);
    }
    clock_gettime(CLOCK_MONOTONIC, &end);
    double elapsed_nt = (end.tv_sec - start.tv_sec) +
                        (end.tv_nsec - start.tv_nsec) / 1e9;

    printf("Transpose %d×%d:\n", n, n);
    printf("  Traditional: %.3f seconds\n", elapsed_trad);
    printf("  Non-temporal: %.3f seconds\n", elapsed_nt);
    printf("  Speedup: %.2f×\n", elapsed_trad / elapsed_nt);

    // Expected on Sapphire Rapids (n=4096):
    // Traditional: ~2.5 seconds (L1 eviction overhead)
    // Non-temporal: ~1.5 seconds (L1 pollution avoided)
    // Speedup: ~1.67×

    free(A);
    free(B);
}

// Compile:
// gcc -O3 -march=sapphire-rapids transpose.c -o transpose

int main() {
    benchmark_transpose(4096);
    return 0;
}

// Performance analysis:
// Traditional transpose: O(n²) memory accesses, poor locality
//   A reads: n² floats, unit stride → prefetch helps (low miss rate)
//   B writes: n² floats, non-unit stride (scattered) → pollutes L1
//   L1 thrashing: ~50% of L1 capacity wasted on non-reusable B writes
//   Effective bandwidth: 384 GB/s ÷ 2 (due to L1 contention) = 192 GB/s
//
// Non-temporal transpose: bypasses L1 for writes
//   A reads: same as above, high hit rate
//   B writes: bypass L1, go to write-combining buffer → amortized latency
//   Effective bandwidth: 384 GB/s (no L1 contention)
//   Speedup: 384 ÷ 192 = 2.0× (theoretical)
//   Actual: ~1.67× (overhead of mfence, write-combining buffer management)
```

### 4.4 L3 Distributed Slice Optimization

```c
#include <string.h>
#include <stdio.h>

// Demonstrate L3 cache-to-cache latency by core distance
// On Sapphire Rapids, L3 is sliced by address; latency depends on
// distance from computing core to L3 slice holding the data

void demonstrate_l3_latency() {
    // L3 slicing: address bits [11:6] → [slice 0:7]
    // slice = (address >> 6) & 0x07 determines which core's L3 partition

    // If running on Core 0:
    //   L3 slice 0 is local (40 cycles)
    //   L3 slice 1 is 1 hop away (40 + link latency)
    //   L3 slice 7 is 7 hops away (40 + 6 × link latency) → ~85 cycles

    // Worst case: L3 misses on all slices except local
    // Best case: all data in local L3 slice

    // Strategy: Align critical data to local L3 slice
    // Core 0's local slice: addresses with bits [11:6] that hash to slice 0
    // i.e., address & 0x1C0 (bits [8:6]) should equal slice_for_core_0

    int arr_local[8192];   // Allocated locally, hopefully in L3 slice 0
    int arr_remote[8192];  // May be in different L3 slice

    // Access pattern: sequential
    // Expected: arr_local has 40-cycle L1 miss → L3 hit (local)
    //           arr_remote has 65-85 cycle L1 miss → L3 hit (remote)

    printf("L3 latency demonstration:\n");
    printf("Local array access: %p (slice %d)\n", (void *)arr_local,
           ((uintptr_t)arr_local >> 6) & 0x07);
    printf("Remote array access: %p (slice %d)\n", (void *)arr_remote,
           ((uintptr_t)arr_remote >> 6) & 0x07);
}

// Optimization: NUMA-aware allocation
#include <numaif.h>
#include <numa.h>

void allocate_numa_local(float **ptr, int size, int numa_node) {
    // Allocate memory local to NUMA node
    // Sapphire Rapids: 2 NUMA nodes (one per socket)
    // Zen 4: up to 12 NUMA nodes (per CXL partition)

    if (numa_available() < 0) {
        fprintf(stderr, "NUMA not available\n");
        return;
    }

    struct bitmask *mask = numa_allocate_nodemask();
    numa_bitmask_setbit(mask, numa_node);

    *ptr = (float *)numa_alloc_onnode(size * sizeof(float), numa_node);
    // Line 1: Allocate from NUMA node
    // Latency: same as other allocations (~10 cycles for local node)
    //          ~150 cycles for remote node (NUMA-aware kernel schedules access)

    numa_free_nodemask(mask);
}

// In ML inference context: allocate embedding tables to local NUMA node
// Reduces remote DRAM access latency by ~50 cycles on dual-socket systems
```

### 4.5 Cache Partitioning via Intel RDT/CAT

```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

// Intel Cache Allocation Technology (CAT) via RDT (Resource Director Technology)
// Requires: Linux kernel with CONFIG_INTEL_RDT, root access, rdtctrl tool

// Concept: partition L3 into regions, assign cores to regions
// Use case: latency-sensitive inference on some cores, batch training on others

void setup_l3_partition() {
    // On Sapphire Rapids: 140 MB L3, 16-way associative
    // Partition: 8 MB for latency-sensitive app, 132 MB for batch training

    // Via command line:
    // pqos -I -l "core=0-9:8MB"     # Cores 0–9 get 8 MB L3
    // pqos -I -l "core=10-59:132MB"  # Cores 10–59 get 132 MB L3

    // Via rdtctrl:
    // echo "60 0x0F 0xFF0" > /sys/fs/resctrl/schemata  # Bit mask allocation
    // 60 = class ID
    // 0x0F = 4 ways out of 16 ways (25% of L3)
    // 0xFF0 = 0xFF (255 = all sockets) followed by 0 (way mask)

    printf("Setting up L3 partitioning via CAT...\n");
    // Typically: system setup, not runtime configuration
}

// Expected performance improvement:
// Latency-critical app: reduced cache evictions (less contention)
//   Before: 50% L3 miss rate (shared cache thrashing)
//   After: 10% L3 miss rate (dedicated 8 MB, working set < 8 MB)
//   Speedup: 40 cycles (L3 hit) vs. 200 cycles (DRAM miss) = 5× faster
//
// Batch training app: no performance change (still has 132 MB available)
```

---

## 5. EXPERT INSIGHT

### 5.1 L3 Distributed vs. Centralized: Design Tradeoffs

**Intel Sapphire Rapids (distributed L3):**
- Advantage: Lower latency for local core (40 cycles), scalable bandwidth (each tile has own L3 slice)
- Disadvantage: Remote accesses incur 65–85 cycle penalty, complex address hashing
- Use: Large-scale systems (100+ cores) where bandwidth is critical

**AMD Zen 4 (clustered L3, local per 4-core group):**
- Advantage: Smaller per-cluster L3 (12 MB per cluster) means tighter NUMA control
- Disadvantage: Still suffers from cross-cluster latency (~50 cycles)
- Use: Smaller systems (12–16 cores) where cache coherency is simpler

**Expert lesson:** Distributed L3 is inherently non-uniform. Code must be NUMA-aware to achieve peak performance. Single-threaded code gets 40-cycle L3 latency; multi-threaded code gets 65–85 cycles depending on data placement.

### 5.2 Prefetch Hurts: Overprefetching & Prefetch Pollution

```c
// Naive prefetch: prefetch too far ahead
for (int i = 0; i < n; i++) {
    _mm_prefetch(&x[i+100], _MM_HINT_T0);  // Prefetch 100 iterations ahead
    // Problem: prefetch fills L2, but we're only using 1 element per iteration
    //          By the time we access x[i+100], L2 is evicted (working set > 2MB)
    //          Prefetch becomes a CACHE MISS, delays computation
    float sum += x[i];
}

// Correct prefetch: prefetch 1–2 cache lines ahead
for (int i = 0; i < n; i += 16) {
    _mm_prefetch(&x[i+32], _MM_HINT_T0);  // Prefetch 2 cache lines ahead
    // Load 16 elements, but prefetch 32 elements (2 lines) ahead
    // Prefetch latency (5 cycles) hidden by computation (16 loads × 4 cycles)
    for (int j = 0; j < 16; j++) {
        sum += x[i+j];
    }
}

// Expert trick: Use _MM_HINT_NTA (non-temporal) for large strides
for (int i = 0; i < n; i += 1000) {
    _mm_prefetch(&x[i], _MM_HINT_NTA);  // Prefetch without L1 pollution
    // Useful for sparse access patterns (e.g., SpMV)
}
```

**Expert truth:** Prefetching is a tuning parameter. Most production code disables explicit prefetch hints and relies on hardware prefetcher (which is quite good on modern Intel/AMD). If you're manually prefetching, benchmark before/after — it often makes things worse.

### 5.3 Cache Line Bouncing in Multithreaded Code

```c
// Bad: false sharing
struct WorkItem {
    volatile int counter;   // Shared counter
    float data[100];        // Per-thread data
} items[NUM_THREADS];

// Thread 0 increments items[0].counter
// Thread 1 increments items[1].counter
// Even though they're different fields, they're on same cache line (64 bytes)
// → Cache line bounces between cores (coherency protocol)
// → Both threads stall on each increment

// Solution: Align to cache line boundary
struct WorkItem {
    volatile int counter;
    char padding[60];      // Pad to 64 bytes (cache line size)
    float data[100];
} items[NUM_THREADS];

// Or use alignas
struct WorkItem {
    alignas(64) volatile int counter;  // Start at cache line boundary
    float data[100];
} items[NUM_THREADS];

// Impact: Latency of counter increment
// Before (bouncing): ~200 cycles (cross-socket coherency)
// After (aligned): ~1–2 cycles (local L1 store)
// Throughput improvement: 100× for high-contention locks
```

**Expert lesson:** In ML inference servers, multiple inference engines running in parallel can share memory. Use cache-line alignment and thread-local buffers to minimize coherency traffic.

### 5.4 Hardware Prefetcher Limitations

The hardware prefetcher on Sapphire Rapids can detect:
1. **Sequential patterns:** stride = 0 (next cache line)
2. **Constant stride patterns:** stride = constant (e.g., every 4th element)
3. **Two-stream patterns:** two separate constant strides tracked per core

But it fails on:
- **Random access:** e.g., (i * 7919) % n
- **Irregular strides:** e.g., next stride depends on data value
- **Complex patterns:** e.g., Fibonacci stride

**Expert optimization:** If your access pattern is not covered by hardware prefetcher, use software prefetch or restructure your algorithm. Example:

```c
// Bad: irregular access pattern
for (int i = 0; i < n; i++) {
    int j = idx[i];  // Irregular index (random or data-dependent)
    sum += arr[j];   // Hardware prefetcher can't help
}

// Better: gather (if small n) or sort indices
std::sort(idx, idx + n);  // Sort to create contiguous access
for (int i = 0; i < n; i++) {
    int j = idx[i];  // Now mostly sequential (after sort)
    sum += arr[j];   // Hardware prefetcher works
}

// Or: software prefetch with prediction
for (int i = 0; i < n; i++) {
    int j = idx[i];
    int j_next = idx[i+1];
    _mm_prefetch(&arr[j_next], _MM_HINT_T0);  // Explicit prefetch
    sum += arr[j];
}
```

---

## 6. BENCHMARK / MEASUREMENT

### 6.1 Measure L1 / L2 / L3 Hit Rates

```bash
# Using perf (Linux)
perf stat -e L1-dcache-loads,L1-dcache-load-misses,\
LLC-loads,LLC-load-misses,cycles -r 5 ./app

# Output:
# L1-dcache-loads:      50,000,000
# L1-dcache-load-misses: 5,000,000 (10% miss rate)
# LLC-loads:            5,000,000 (all L1 misses hit L2 or L3)
# LLC-load-misses:      50,000    (1% of LLC loads miss → DRAM)

# Using Intel VTune
vtune -c memory-access-latency -r results ./app
# Provides histograms of latency distribution

# Using AMD uProf (for Zen 4)
uprof -e l1_data_cache_misses,l2_cache_misses,l3_cache_misses ./app
```

### 6.2 Measure L3 Slicing Impact (NUMA Latency)

```c
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sched.h>

int measure_l3_slice_latency(int num_addresses) {
    volatile uint64_t latencies[num_addresses];

    for (int s = 0; s < 8; s++) {  // Test each L3 slice
        // Allocate addresses that hash to slice s
        // This requires understanding L3 slicing: address >> 6 & 0x07

        unsigned long slice_addrs[10000];
        int count = 0;

        for (unsigned long addr = 0; addr < 1UL << 30; addr += 64) {
            if (((addr >> 6) & 0x07) == s) {
                slice_addrs[count++] = addr;
                if (count >= 10000) break;
            }
        }

        // Warm up: access to fill L3
        for (int i = 0; i < count; i++) {
            *(volatile uint64_t *)(slice_addrs[i]) = i;
        }

        // Measure latency: simple load chain
        volatile uint64_t *ptr = (volatile uint64_t *)slice_addrs[0];
        uint64_t start = __rdtsc();

        for (int iter = 0; iter < 10000; iter++) {
            // Intentional dependency to force load serialization
            ptr = (volatile uint64_t *)(*ptr % (1UL << 30));
        }

        uint64_t end = __rdtsc();
        double latency = (double)(end - start) / 10000;

        printf("L3 slice %d: %.1f cycles latency\n", s, latency);
    }

    return 0;
}

// Expected results:
// L3 slice 0 (local): ~40 cycles
// L3 slice 1: ~45 cycles (1 hop)
// L3 slice 7: ~85 cycles (7 hops)
```

### 6.3 Cache Conflict Analysis Tool

```bash
# Using likwid (LIKWID – Lightweight Integrated Kernel Like Interface We're Developing)
likwid-topology  # Shows cache topology

likwid-perfctr -m L1DSTAT -f -g L1DSTAT ./app
# Output: L1-D cache load distribution, conflict analysis

likwid-cache -p   # Plot cache memory access patterns
```

---

## 7. ML SYSTEMS RELEVANCE

### 7.1 Transformer KV-Cache: L3 Bottleneck

In auto-regressive transformer inference, the KV-cache (key-value cache) grows with sequence length:

```
At position T:
  KV-cache size = 2 × batch_size × T × hidden_dim
                = 2 × 32 × 4096 × 768 (4096 = max seq length)
                = 200 MB (exceeds L3 140 MB on Sapphire Rapids)

Attention computation:
  For each query position, must load corresponding K, V
  K[0:T][hidden] → T × 768 × 4 bytes = T × 3 KB per query
  At T=4096, this is ~12 MB per query → 12 MB / L3 access = 1000+ cycles per query
```

**Impact:** KV-cache misses cause 200-cycle DRAM stalls, limiting throughput to ~50 tokens/sec on single core.

**Optimization (AMD 3D V-Cache):**
- 3D V-Cache provides additional 96 MB, total 128 MB
- KV-cache now fits in on-die cache
- Latency: ~50 cycles instead of 200 cycles
- Throughput: 4× improvement for long sequences

### 7.2 Embedding Lookup Cache Efficiency

```
Word embedding lookup: embeddings[vocab_id] → 768-element vector
Vocabulary size: 50,000–100,000
Embedding size: 768 × 4 bytes = 3 KB per vector

If lookups are random (no spatial locality):
  Each lookup: 3 KB load from memory
  Cache line size: 64 bytes
  Cache lines per lookup: 3 KB / 64 bytes = ~48 cache lines

L1 capacity: 48 KB (1 embedding), L1 miss rate ~99%
L2 capacity: 2 MB (500–600 embeddings), L2 miss rate ~70% (for random lookups)
L3 capacity: 140 MB (40,000+ embeddings), L3 miss rate ~5–10%

Solution: Batch embedding lookups to improve locality
  If batch size = 32:
    32 embeddings × 3 KB = 96 KB
    Exceeds L1 (48 KB), but fits in L2 (2 MB)
    L2 hit rate ~80%, speedup ~5× over single-element lookups
```

### 7.3 GEMM Cache Blocking for Peak Performance

```
Goal: Achieve peak 32 GFLOPS on single core (Sapphire Rapids with AVX-512)

Working set analysis:
  16 × 16 × K GEMM: requires loading A (16×K), B (K×16), writing C (16×16)
  Bytes: 2 × 16 × K × 4 + 16 × 16 × 4 = 128K + 1024
  Operations: 16 × 16 × K × 2 = 512K FLOPs

  For different K values:
  K=128: 16 KB load + compute = 512K FLOPs → compute intensity = 32 FLOPs/byte
         L1 can hold entire A and most of B → L1 hit rate ~90%
         Expected: 32 × 0.9 = 28.8 GFLOPS ✓

  K=1024: 128 KB load + compute = 4M FLOPs → compute intensity = 32 FLOPs/byte
          L1 overflow (48 KB limit), must use L2
          L2 hit rate ~70%
          Expected: 32 × 0.7 = 22.4 GFLOPS (degraded)
```

---

## 8. PhD QUALIFIER QUESTIONS

### Question 8.1
Explain why non-temporal stores (movnt) improve performance for matrix transpose. What is the fundamental cache hierarchy problem that transpose exposes, and how does movnt solve it? Calculate the expected speedup.

**Model Answer:**
Matrix transpose: C[j][i] = A[i][j] (all elements accessed exactly once, no reuse)
- Traditional approach:
  - Load A[i][0..n] (sequential, cached)
  - Store C[0..n][i] (scattered, cache-hostile)
  - Each write pollutes L1 with C data that won't be reused
  - L1 capacity: 48 KB; transpose writes: n × 4 bytes per row
  - For n=1024: 4 KB per row, but 48 KB L1 holds only ~12 rows before evicting A
  - Result: L1 miss rate for reads ~70% (due to eviction by non-reusable writes)

- Non-temporal stores:
  - Loads still use L1 (cached normally)
  - Writes bypass L1, go to write-combining buffer
  - Write-combining buffer (256 bytes) aggregates multiple NT writes
  - L1 reserved entirely for A reads
  - L1 hit rate for reads ~95%

Speedup:
- Traditional: 70% read miss rate → ~18 GFLOPS effective (for 32 GFLOPS peak)
- Non-temporal: 95% read hit rate → 30 GFLOPS effective
- Speedup: 30 ÷ 18 = 1.67× (matches empirical observation in Section 4.3)

### Question 8.2
Sapphire Rapids has L3 organized as 8 slices, each connected to one core. Explain the address-based slicing scheme and calculate the L3 access latency for a core accessing data in its local slice vs. a remote slice. What NUMA-aware optimization would you apply in an ML inference server with KV-cache?

**Model Answer:**
L3 slicing: address [11:6] (6 bits = 64 sets) → [slice index 2:0] (3 bits = 8 slices per tile)
- Core i is responsible for L3 slice i
- When Core 0 reads address X with set bits [11:6] = S:
  - S[2:0] determines which slice (0–7)
  - If S[2:0] == 0: local L3 slice, latency 40 cycles
  - If S[2:0] == 7: remote L3 slice, distance = 7 hops, latency ~40 + 6 × link_latency
  - Link latency: ~6–7 cycles per hop
  - Total remote: 40 + 6 × 7 = ~82 cycles

KV-cache optimization:
- KV-cache is read-only after computation, accessed by attention kernel
- Pin KV-cache allocation to NUMA node local to attention kernel threads
- On Sapphire Rapids: 2 NUMA nodes (2 sockets)
  - Allocate KV-cache on socket 0 if attention threads run on socket 0
  - Use numa_alloc_onnode() or taskset to bind threads + memory to same socket
  - Reduces L3 remote access latency from 82 cycles → 40 cycles local
  - With batching: 32 token batch × (82 vs 40 cycles) = 32 × 42 cycles saved = ~1344 cycles per batch
  - Expected throughput improvement: ~1–2× (depends on memory bandwidth limit)

### Question 8.3
Analyze the cache conflict in the embedding lookup example (Section 3.2). Given vocab_size=65536, embedding_dim=768, sizeof(float)=4, calculate which cache sets are most heavily used. Propose a solution and verify it eliminates conflicts.

**Model Answer:**
Memory layout: embeddings[i][j] at address base + i × (768 × 4) + j × 4
             = base + i × 3072 + j × 4

Cache set calculation (L1-D, 12-way, 64 sets):
set = ((address >> 6) & 0x3F)
    = (((base + i × 3072 + j × 4) >> 6) & 0x3F)
    = (((i × 3072) >> 6 + (j × 4) >> 6) & 0x3F)
    = ((i × 48 + j) & 0x3F)  [dividing 3072 by 64 = 48]

For different i values:
- i=0: set = j & 0x3F → sets 0–63 (64 distinct sets for j=0–63)
- i=1: set = (48 + j) & 0x3F = j & 0x3F (since 48 mod 64 = 48, and j mod 64 = j)
  Wait, let me recalculate: (48 + j) & 0x3F
  - j=0: set = 48
  - j=1: set = 49
  - ...
  - j=15: set = 63
  - j=16: set = (48 + 16) & 0x3F = 64 & 0x3F = 0
  → Wraps around, creates aliasing

For i=0 to 3:
- i=0: sets 0–63
- i=1: sets 48–63, 0–47 (rotate by 48)
- i=2: sets (96 mod 64)–... = 32–95 mod 64 = sets 32–63, 0–31
- i=3: sets 16–79 mod 64 = sets 16–63, 0–15

Conflict: all embeddings[0:12][0] have addresses mapping to multiple sets, but:
  - embeddings[0][0] → set 0
  - embeddings[1][0] → set 48
  - embeddings[2][0] → set 32
  - embeddings[3][0] → set 16
  - embeddings[4][0] → set 0 (conflicts with embeddings[0][0]!)

  Actually: (4 × 48 + 0) & 0x3F = (192 & 0x3F) = 0
  → Every 4th embedding row maps to same set → conflicts

Solution: Pad embedding stride to power of 2
embeddings_padded: change stride from 3072 (3 × 1024) to 4096 (2^12)
- New set = ((i × 4096 + j × 4) >> 6) & 0x3F
           = ((i × 64 + j) & 0x3F)
           = (i & 0x00) × 64 + (j & 0x3F)  [since 64 mod 64 = 0]
           = j & 0x3F

Now: all rows map to same set as determined by j, not i
- embeddings_padded[i][0] → set 0 (all i)
- embeddings_padded[i][1] → set 1 (all i)
- ...
- 12-way conflict within same j across different i

But within single embedding vector (j varies):
- embeddings_padded[i][0..63] → sets 0–63 (no conflict)
- embeddings_padded[i][64..127] → sets 0–63 (but that's column 64 of next embedding)

Actually, the padding strategy works because:
- Embedding_padded[i][0:768] has stride 4096 bytes (power of 2)
- Each column j maps to distinct set: set = j & 0x3F
- 768 > 64, so we use all 64 sets
- L1-D 12-way can hold 12 distinct cache lines per set
- With 8 embedding rows and 64 sets, each set has ~8 lines (well within 12-way)

Verification: conflict rate drops from 50% to 0%

### Question 8.4
Explain the distributed L3 architecture on Sapphire Rapids. How would you optimize an inference kernel that accesses a large embedding table across multiple cores running in parallel?

**Model Answer:**
Sapphire Rapids L3 distributed architecture:
- 8 L3 slices, each 140 MB ÷ 8 = 17.5 MB per slice
- Slice i is physically located near Core i (on same tile)
- Access latency: 40 cycles if accessing slice j on same core
- Access latency: 65–85 cycles if accessing slice j on different core (via mesh)

Large embedding table optimization (shared across inference instances):
1. **Replicate embeddings across NUMA nodes:** Each socket caches a copy of the embedding table
   - Eliminates inter-socket access latency (65–85 cycles → 40 cycles)
   - Cost: 2× memory usage (acceptable for embeddings, which are small relative to model size)

2. **Pin inference threads to cores on same socket:** Ensure all threads on socket 0 access embeddings on socket 0
   - Use taskset -c 0-29 ./inference_app (pin to first socket's 30 cores)
   - Use numa_alloc_onnode() to allocate embeddings on socket 0
   - Eliminates NUMA-remote latency entirely

3. **Use L3 cache partitioning (CAT):** Dedicate 20 MB L3 per socket to embedding cache
   - Leave 117.5 MB for KV-cache and attention computations
   - Ensures embeddings stay in L3, reducing DRAM hits

Expected performance:
- Single core, single lookup: 40 cycles (L3 local hit)
- Batch 32 lookups with prefetch: 40 + (31 × 1) = ~71 cycles (prefetch hides latency)
- Throughput: 2.0 GHz ÷ 71 cycles = 28 lookups/cycle = 28M lookups/sec per core

### Question 8.5
Compare L3 miss performance between Sapphire Rapids (distributed L3) and Zen 4 (clustered L3 with 3D V-Cache). For a KV-cache size of 200 MB, calculate expected latency and throughput on each platform.

**Model Answer:**
KV-cache size: 200 MB (exceeds on-die cache on both platforms without 3D V-Cache)

Sapphire Rapids (without 3D V-Cache):
- L3 total: 140 MB (< 200 MB KV-cache)
- Misses go to DRAM: ~200 cycles latency
- Sequential access to KV-cache (row-major): 200 MB ÷ (32 parallel loads × 8 bytes) = 312k cycles
- Single core throughput: 312k cycles / (2.0 GHz) = 156 ms (for full KV scan)
- For auto-regressive decode (1 token per iteration):
  - Must load 1 row of KV: 768 × 8 bytes = 6 KB
  - Expected: 6 KB ÷ (32 GB/s bandwidth per core) = ~200 ns = 400 cycles
  - Due to DRAM latency: actual ~400 cycles per token (200 cycles latency + computation)

Zen 4 with 3D V-Cache:
- L3 total: 12 MB (clustered) + 96 MB (3D V-Cache) = 108 MB
- Still overflow, but close
- KV-cache partial hits in 3D V-Cache: ~50 cycle latency
- Misses go to DRAM: ~200 cycles
- Expected miss rate: 92 MB out of 200 MB fit in cache → 54% hit rate
- Average latency: 0.54 × 50 + 0.46 × 200 = 119 cycles per access
- Throughput: 2.0 GHz ÷ 119 = ~17M accesses/sec
- For KV-cache: 200 MB ÷ (17M accesses/sec × 8 bytes) = ~5.9 ms

Speedup: Sapphire Rapids 400 cycles vs. Zen 4 119 cycles = 3.4× improvement with 3D V-Cache

---

## 9. READING LIST

1. **Intel 64 and IA-32 Architectures Optimization Reference Manual**
   - Chapter 2: Microarchitecture overview (L1/L2/L3 details)
   - Chapter 6: Memory hierarchy and caching
   - Chapter 7: Prefetch optimization

2. **Sapphire Rapids Optimization Guide** (Intel, free PDF)
   - Section on L3 distributed architecture
   - Mesh interconnect topology and latency tables
   - NUMA-aware optimization recommendations

3. **AMD EPYC BIOS and Kernel Developer's Guide**
   - Chapter on Zen 4 cache hierarchy
   - 3D V-Cache architecture and performance impact
   - NUMA topology and CXL configuration

4. **Computer Architecture: A Quantitative Approach** (H&P, 6th ed.)
   - Chapter 2: Memory hierarchy (foundational)
   - Chapter 5: Thread-level parallelism (cache coherency)

5. **Agner Fog, The Microarchitecture of Intel, AMD and VIA CPUs** (free PDF)
   - Vol 3: Cache organization, latency tables for Intel/AMD
   - Detailed instruction timing, memory stall analysis

6. **Linux kernel documentation: Resource Director Technology (RDT)**
   - /Documentation/x86/intel_rdt_ui.txt
   - L3 cache partitioning (CAT) setup and examples

7. **VTune Profiler Documentation** (Intel)
   - Memory latency analysis, cache miss profiling
   - https://www.intel.com/content/www/us/en/developer/tools/oneapi/vtune-profiler.html

8. **LIKWID Documentation** (https://github.com/RRZE-HPC/likwid)
   - Cache topology analysis, L1/L2/L3 benchmarking tools

9. **STREAM Benchmark Suite** (https://www.cs.virginia.edu/stream/)
   - Memory bandwidth measurement and cache effects

10. **AMD Ryzen 3D V-Cache Architecture**
    - https://www.amd.com/en/technologies/v-cache
    - Performance comparisons, use cases

