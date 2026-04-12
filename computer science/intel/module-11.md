# MODULE 11 — Memory Subsystem: DRAM to DIMM to Controller

## 1. CONCEPTUAL FOUNDATION

### 1.1 DDR5 Architecture & Signaling

**DDR5** (Double Data Rate 5, standardized 2020, shipped 2022) introduces fundamental changes from DDR4:

**Key Parameters (DDR5-4800 example, 4800 MT/s = megatransfers per second):**
- **Burst Length:** 16 bytes (BL16, increased from DDR4's 8 bytes)
- **Prefetch:** 16n (16 bits per clock, where n=1 → 16 bits per cycle from chip)
- **Clock:** 1200 MHz internal (2400 MHz data rate on data strobe, 4800 MT/s total)
- **Voltage:** 1.1V (reduced from DDR4's 1.2V, better power efficiency)
- **Pins:** 288-pin DIMM (same mechanical form factor as DDR4, not compatible)

**Timing parameters (CAS Latency, tCL example: DDR5-4800 CL46):**
- **tCL (CAS Latency):** 46 cycles (time from RAS→CAS command to data output)
  - In nanoseconds: 46 cycles ÷ 2400 MHz = 19.2 ns
- **tRCD (RAS to CAS Delay):** 46 cycles (activate row → column select)
- **tRP (RAS Precharge):** 46 cycles (close row, reset sense amps)
- **tRAS (RAS Active Time):** ~120 cycles (row open for multiple column accesses)
  - Key insight: if tRAS >> tCL, can perform multiple column accesses per row activation
  - Example: 120 ÷ 46 = 2.6 → can do ~2 RAS cycles per row open window

**Physical organization (per DRAM chip, e.g., SK Hynix 16Gbit DDR5):**
```
Capacity: 16 Gbit = 2 Gigabytes
Density: 16 banks × 8 subarrays × (capacity ÷ 128)
       = 16 banks × 1024 rows each × 8192 columns
Addressing: [bank 4 bits][row 10 bits][column 13 bits][byte 3 bits]
          = 4 + 10 + 13 + 3 = 30 bits, but 2 Gb = 2^31 bits
          = Correct: 31 bits total, matches 2 GB capacity
```

See: JEDEC DDR5 Standard (DDR5 SPD Specification); SK Hynix, Samsung, Micron DDR5 DRAM datasheets.

### 1.2 Memory Controller: Intel Sapphire Rapids (SPR) vs. AMD Genoa

**Intel Sapphire Rapids (8-channel DDR5 on-die controller):**

```
Physical layout (per socket):
  1× DDR5 Memory Controller (on-die)
  ├─ 8 Memory channels
  │  ├─ Channel 0: up to 2 DIMMs, 192 GB max (2×96GB DIMM)
  │  ├─ Channel 1: up to 2 DIMMs, 192 GB max
  │  ...
  │  └─ Channel 7: up to 2 DIMMs, 192 GB max
  └─ Total: 16 DIMMs per socket, 1.5 TB per socket (2 sockets = 3 TB system)

Bandwidth per channel:
  DDR5-4800: 4800 MT/s × 64 bits (8 bytes) per transfer
           = 4800 × 10^6 transfers/s × 8 bytes/transfer
           = 38.4 GB/s per channel
  Total: 8 channels × 38.4 GB/s = 307.2 GB/s peak

Actual on-die DDR5-4800 modules (3200 MHz JEDEC standard):
  Sapphire Rapids supports: DDR5-4800 (standard), DDR5-5600, DDR5-6400 (overclocked)
  Peak bandwidth: 8 channels × 40 GB/s (DDR5-5000) = 320 GB/s
  Typical: 300–320 GB/s in practice
```

**AMD EPYC Genoa (12-channel DDR5 on-die controller):**

```
Physical layout (per socket):
  1× DDR5 Memory Controller
  ├─ 12 Memory channels (vs. Intel's 8)
  │  ├─ Channel 0: up to 1–2 DIMMs
  │  ├─ ...
  │  └─ Channel 11: up to 1–2 DIMMs
  └─ Total: 12–24 DIMMs per socket

Bandwidth per channel: Same as SPR (38.4 GB/s for DDR5-4800)
  Total: 12 channels × 38.4 GB/s = 460.8 GB/s peak
  Typical: 450 GB/s in practice

Advantage: 50% more bandwidth than Sapphire Rapids
Tradeoff: Higher power consumption, larger die
```

**Comparison table:**

| Parameter | Sapphire Rapids | Genoa |
|---|---|---|
| Channels | 8 | 12 |
| Peak BW (DDR5-4800) | 307 GB/s | 461 GB/s |
| Max DIMMs/socket | 16 | 24 |
| Max capacity/socket | 1.5 TB | 2.0 TB |
| Latency (CL46) | ~46 ns | ~46 ns (same timing) |

See: Intel Sapphire Rapids Datasheet; AMD EPYC Genoa Datasheet.

### 1.3 DRAM Refresh & Latency Spikes

**DRAM Refresh Problem:**
Modern DRAM cells are 1T1C (1 transistor, 1 capacitor). Charge leaks over time (~64 ms for DDR5). Refresh cycles periodically charge capacitors.

**Refresh cycle duration (tREF, typical 3.9 µs per 8 KB segment):**
- Every 3.9 µs, memory controller must issue REFRESH command
- REFRESH command blocks all memory access (synchronous, not pipelined)
- Latency: ~1–10 µs (depends on open rows, busy banks)

**Impact on real-time systems:**
- Guaranteed maximum latency must account for refresh stall
- Example: video streaming at 60 fps requires < 16.7 ms frame latency
  - Refresh adds ~1–10 µs per 3.9 µs interval = unpredictable jitter
  - Not suitable for hard real-time (but acceptable for soft real-time)

**DDR5 Enhancement (Per-Bank Refresh, PBR):**
- Refresh at bank level instead of global
- Reduces stall duration, allows other banks to serve requests
- Improvement: max stall ~1–2 µs (vs. ~10 µs in DDR4)

See: JEDEC DDR5 Standard, Section on Refresh; memory controller datasheets for refresh interval specifications.

### 1.4 Row Buffer & Closed-Page Policy

**Row Buffer Architecture:**
```
DRAM Die: 16 banks, each with sense amps (row buffer)
Accessing data:
  1. Open row: RAS command, row data → sense amps (12–46 cycles, tRCD)
  2. Read columns: RD command, data from sense amps (4 cycles, tCL)
  3. Close row: PRECHARGE command (46 cycles, tRP)

Memory controller policy:
  – Open-page: keep row open after access, hope next access is same row
    Advantage: if next request hits same row, latency = tCL (~4 cycles) instead of tRCD (~46 cycles)
    Disadvantage: if rows don't cluster, wasted resources
  – Closed-page: precharge immediately after access
    Advantage: predictable, no stale data, lower power
    Disadvantage: every access costs tRCD latency
```

**Typical Intel behavior: Adaptive policy**
- Open row if queue length > threshold (reuse likely)
- Close row if queue is short (no benefit to keeping open)

**Row buffer hits analysis (for ML workloads):**
```
Matrix multiply with unit-stride access:
  Load row i of A: first access costs tRCD + tCL = 46 + 46 = 92 cycles
  Next 15 accesses (same row): cost tCL = 46 cycles each (row buffer hits)
  Average: (92 + 15 × 46) ÷ 16 = (92 + 690) ÷ 16 = 48.6 cycles per access

Random access (no row buffer reuse):
  Every access: tRCD + tCL = 92 cycles
  Average: 92 cycles per access

Difference: 92 ÷ 48.6 = 1.89× (nearly 2× penalty for random access)
```

See: memory controller specifications, Intel Optimization Manual Chapter 7.

### 1.5 CXL (Compute Express Link) for AI Servers

**CXL Overview:**
CXL is a new protocol (v1.0 → v3.0 as of 2024) enabling device-to-host and device-to-device communication over PCIe physical layer.

**Key features:**
1. **Coherence:** Shared memory space with cache coherency between CPU and accelerator
2. **Memory:** Accelerators can expose HBM as coherent memory to CPU
3. **IO:** Traditional PCIe functionality (control, data path)

**Versions:**
- **CXL 1.0 (2019):** Proof of concept, limited bandwidth
- **CXL 2.0 (2021):** Full coherent caching, 32 GB/s bandwidth
- **CXL 3.0 (2024):** Enhanced memory semantics, 80 GB/s bandwidth (rumored)

**Use in AI servers:**
```
Traditional GPU inference:
  CPU → PCIe → GPU HBM (read-only)
  Bandwidth: PCIe Gen4 = 32 GB/s (limited)
  Latency: PCIe transaction latency (~1–10 µs)

CXL-enabled inference (future):
  CPU ↔ GPU (same coherent address space)
  Bandwidth: CXL 2.0 = up to 32 GB/s, CXL 3.0 = 80+ GB/s
  Latency: ~100 ns (memory latency, not PCIe)
  Benefit: Simplified programming, unified memory model
```

**Current adoption (as of 2024):**
- AMD EPYC Genoa with CXL 1.1 support (via OCP/NUMA domains)
- Intel Sapphire Rapids: CXL controller in some SKUs, limited deployment
- Nvidia H100: No native CXL (H200 evaluating)
- Growth expected 2024–2026

See: CXL Specification (https://www.computeexpresslink.org/); AMD, Intel CXL adoption roadmaps.

---

## 2. MENTAL MODEL

```
                       Memory Subsystem: Sapphire Rapids
                       ───────────────────────────────

    ┌─────────────────────────────────────────────────────────────┐
    │ CPU Cores (0–59) + L3 Cache (140 MB distributed)           │
    │ Issue memory requests: physical addresses                   │
    └──────────────────────────┬──────────────────────────────────┘
                               │
    ┌──────────────────────────┴──────────────────────────────────┐
    │           On-Die DDR5 Memory Controller                     │
    │  - Request arbitration                                      │
    │  - Bank scheduling (optimize for row buffer hits)           │
    │  - Refresh management (every 3.9 µs)                        │
    │  - Power state management (C-states)                        │
    └──────────────────┬───────────────────────────────────────────┘
                       │
         ┌─────────────┼─────────────┬────────────┬────────────┐
         │             │             │            │            │
     [Channel0]    [Channel1]    [Channel2]  [Channel3]  ... [Channel7]
      DIMM0/1      DIMM0/1       DIMM0/1    DIMM0/1        DIMM0/1
       (192GB)      (192GB)       (192GB)    (192GB)        (192GB)
         │             │             │            │            │
         ↓             ↓             ↓            ↓            ↓
      ┌─────────────────────────────────────────────────────────────┐
      │  64-bit wide DDR5 bus (8 bytes per transfer)                │
      │  Speed: 4800 MT/s (2400 MHz clock, DDR = 2× clock)          │
      │  Throughput: 38.4 GB/s per channel                          │
      │  Total: 8 channels × 38.4 GB/s = 307 GB/s peak             │
      └─────────────────────────────────────────────────────────────┘
         │             │             │            │            │
         ↓             ↓             ↓            ↓            ↓
      [DRAM]       [DRAM]        [DRAM]       [DRAM]       [DRAM]
       Chips        Chips         Chips        Chips        Chips
      (per DIMM:    16 banks, 1024 rows each, 8192 columns)


      Timing Sequence (DDR5-4800, CL46):
      ──────────────────────────────────
      Cycle:  0     46      92      138     184
              │      │      │       │       │
              ├─RAS──┤
              │ (activate row 0)
                     ├─CAS──┤
                     │ (read column)
                             ├─PRE──┤
                             │ (close row)
                                     ├─RAS──┤
                                     │ (activate row 1)

      Total latency (first access):   46 + 46 + 46 = 138 cycles
      If row hits:                    46 cycles (CAS latency only)


      Memory Access Patterns vs. Bandwidth:
      ───────────────────────────────────
      Sequential (stride=1):     48.6 ns/access (row hits) = 20.6 GB/s effective
      Stride-4:                  51.2 ns/access = 19.5 GB/s effective
      Random (no row hits):      92 ns/access = 10.9 GB/s effective
      Implication: Access pattern matters as much as peak bandwidth
```

---

## 3. PERFORMANCE LENS

### 3.1 Memory Bandwidth Utilization: Peak vs. Sustained

**Peak bandwidth (theoretical):**
- 8 channels × 4800 MT/s × 8 bytes/transfer = 307.2 GB/s

**Sustained bandwidth (practical):**

| Workload Type | Bandwidth | Limitation |
|---|---|---|
| Sequential reads (64-byte lines) | 250–290 GB/s | Row buffer misses, refresh |
| Streaming writes (movnt) | 240–280 GB/s | Write-combining buffer, refresh |
| Random access (no locality) | 80–120 GB/s | Row buffer closed policy, scheduling |
| Mixed read-write (50/50) | 120–180 GB/s | Bank conflicts, refresh |

**Example: STREAM Triad (y += a×x) on Sapphire Rapids:**
```
3 arrays × 10M elements × 4 bytes × 100 iterations = 12 GB data
Sequential access (contiguous memory):
  - Measured: ~320 GB/s (all cores, ~1.2× overestimation due to prefetch)
  - Limit: memory bandwidth, not compute

Random access (permuted index):
  - Measured: ~80 GB/s
  - Limit: row buffer misses, scheduler stalls
```

### 3.2 DRAM Latency Analysis

**Latency components (DDR5-4800, CL46):**

| Component | Cycles | Nanoseconds |
|---|---|---|
| CAS latency (tCL) | 46 | 19.2 |
| Controller overhead | 2–5 | 0.8–2.1 |
| Interconnect (local L3→DRAM) | ~10 | ~4.2 |
| DRAM chip internal | ~40 | ~16.7 |
| **Total (row buffer hit)** | ~60 | ~25 |
| **Total (row miss, tRCD)** | ~120 | ~50 |

**Effective latency in real systems:**
- Single-threaded load stall: ~60–100 cycles (depending on row buffer state)
- Multi-threaded (out-of-order execution): latency hidden by other threads
- Prefetch success: latency reduced to ~20–30 cycles

### 3.3 Memory Power Consumption

**DDR5 power budget (per DIMM):**
- **Read current:** ~1.5 A per DIMM (at 1.1V = 1.65 W)
- **Write current:** ~1.2 A per DIMM (less power than read due to sense amp optimization)
- **Standby:** ~100 mA per DIMM (0.11 W)
- **Refresh:** Background, averaged into standby

**System-level power (Sapphire Rapids dual-socket with 16 DIMMs):**
- Compute: ~500 W (all 120 cores at full power)
- Memory subsystem: ~16 DIMMs × 1.65 W = 26.4 W (active)
- Interconnect/chipset: ~50 W
- **Total:** ~575 W

**Implication:** Memory subsystem is ~5% of power budget, but can be energy bottleneck (DRAM is inherently slow relative to power consumed).

---

## 4. ANNOTATED CODE

### 4.1 Measuring DRAM Row Buffer Hit Rate

```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Estimate row buffer hit rate by accessing memory with different strides
// DDR5 row size: 8 KB (1024 × 8 bytes per bank)
// Row buffer hit: same row accessed consecutively
// Row buffer miss: different row (requires tRCD + tCL latency)

void analyze_row_buffer_hits(int *data, int size, int stride) {
    // Sequential access: stride=1
    // Stride-64 access: stride=64 (next cache line)
    // Stride-1024: stride=1024 (outside row buffer, likely miss)

    long long accesses = 0;
    volatile int sum = 0;

    // Warm up cache
    for (int i = 0; i < size; i += stride) {
        sum += data[i];
    }

    // Measure with PERF
    // perf stat -e LLC-loads,LLC-load-misses ./program stride1
    // perf stat -e LLC-loads,LLC-load-misses ./program stride1024

    for (int iter = 0; iter < 10; iter++) {
        for (int i = 0; i < size; i += stride) {
            sum += data[i];
            accesses++;
        }
    }

    printf("Stride %d: %lld accesses, sum=%d\n", stride, accesses, sum);
}

// Expected results:
// Stride 1:       Cache hit rate ~99% (L1 prefetch)
// Stride 64:      Cache hit rate ~99% (L1 prefetch, each access is next cache line)
// Stride 1024:    Cache hit rate ~5%  (L3 miss, DRAM miss, random bank)
//
// In DRAM terms:
// Stride 1:       Row buffer HIT (all accesses in 8 KB row)
// Stride 1024:    Row buffer MISS (8 KB row contains only 8 elements, stride=1024 skips)

int main(int argc, char *argv[]) {
    int stride = atoi(argv[1]);  // argv[1] = stride value
    int size = 10000000;
    int *data = (int *)malloc(size * sizeof(int));

    // Initialize
    for (int i = 0; i < size; i++) {
        data[i] = i;
    }

    analyze_row_buffer_hits(data, size, stride);

    free(data);
    return 0;
}

// Compile and measure:
// gcc -O3 row_buffer.c -o row_buffer
// perf stat -e dTLB-loads,dTLB-load-misses ./row_buffer 1
// perf stat -e dTLB-loads,dTLB-load-misses ./row_buffer 1024
//
// dTLB (data TLB) misses indicate DRAM controller stalls
// Expected: ~0% for stride 1, ~50% for stride 1024
```

### 4.2 DRAM Latency Measurement via Load Chains

```c
#include <immintrin.h>
#include <stdint.h>
#include <stdio.h>

// Measure DRAM latency by creating a dependency chain
// Force serialization: y = f(f(f(...f(x))))

uint64_t measure_dram_latency(int *arr, int size, int iterations) {
    // Create linked list: arr[i] points to arr[arr[i] % size]
    // This ensures each load depends on previous load (can't prefetch)

    // Initialize: arr[0] → arr[1] → arr[2] → ... → arr[0] (cycle)
    for (int i = 0; i < size; i++) {
        arr[i] = (i + 1) % size;
    }

    // Flush DRAM into cache first (warm up)
    volatile int dummy = 0;
    int idx = 0;
    for (int i = 0; i < size; i++) {
        idx = arr[idx];
        dummy += idx;
    }

    // Measure latency: dependent chain forces serialization
    uint64_t start = __rdtsc();

    idx = 0;
    for (int i = 0; i < iterations; i++) {
        idx = arr[idx];  // Load-use dependency, latency = tCL (46 cycles)
        dummy += idx;    // Use index for next iteration
    }

    uint64_t end = __rdtsc();

    double latency = (double)(end - start) / iterations;
    printf("Measured DRAM latency: %.1f cycles\n", latency);

    return dummy;  // Use result to prevent optimization
}

int main() {
    int size = 100000;
    int *arr = (int *)malloc(size * sizeof(int));

    uint64_t dummy = measure_dram_latency(arr, size, 100000);

    printf("Dummy: %ld\n", dummy);

    free(arr);
    return 0;
}

// Expected output:
// Measured DRAM latency: 60-80 cycles
// (Includes CAS latency 46 + controller overhead + interconnect delay)
//
// If arr[i] = random permutation:
//   - Each access has 60-80 cycle latency
//   - No prefetch possible (unpredictable next index)
//   - Throughput: 1 access per 60–80 cycles = 12.5–16.7 M accesses/sec
//   - Bandwidth: 16.7 M × 4 bytes = 66 MB/s (single core)
//
// If arr[i] = sequential (i+1):
//   - Each access has 4 cycle latency (L1 hit)
//   - Prefetch works
//   - Throughput: 1 access per 4 cycles = 500 M accesses/sec
//   - Bandwidth: 500 M × 4 bytes = 2 GB/s (single core)
```

### 4.3 Memory Controller Scheduling Simulation

```c
#include <stdio.h>
#include <string.h>

// Simplified memory controller scheduler
// Tracks open rows, schedules requests to maximize row buffer hits

#define NUM_CHANNELS 8
#define NUM_BANKS 16
#define NUM_ROWS 1024

typedef struct {
    int open_row[NUM_BANKS];  // -1 if no row open, else row number
    int cycles_until_available[NUM_BANKS];  // When bank can accept next command
} ChannelState;

ChannelState channels[NUM_CHANNELS];

void init_channels() {
    for (int c = 0; c < NUM_CHANNELS; c++) {
        for (int b = 0; b < NUM_BANKS; b++) {
            channels[c].open_row[b] = -1;
            channels[c].cycles_until_available[b] = 0;
        }
    }
}

// Simulate memory access
// Returns latency in cycles
int access_memory(unsigned long addr) {
    int channel = (addr >> 6) & 0x7;      // bits [8:6] select channel
    int bank = (addr >> 13) & 0xF;        // bits [16:13] select bank
    int row = (addr >> 17) & 0x3FF;       // bits [26:17] select row
    int col = (addr >> 3) & 0x1FFF;       // bits [28:16] select column

    ChannelState *ch = &channels[channel];

    int latency = 0;

    // Check if row is already open
    if (ch->open_row[bank] == row) {
        // Row buffer hit
        latency = 46;  // tCL (CAS latency)
    } else {
        // Row buffer miss
        if (ch->open_row[bank] != -1) {
            // Must close current row
            latency += 46;  // tRP (precharge)
        }
        // Open new row
        latency += 46;  // tRCD (RAS to CAS delay)
        // Column access
        latency += 46;  // tCL
        ch->open_row[bank] = row;
    }

    // Update bank availability
    int total_latency = latency;
    ch->cycles_until_available[bank] += total_latency;

    return total_latency;
}

void test_sequential_access() {
    init_channels();

    int total_cycles = 0;
    for (int i = 0; i < 1000; i++) {
        unsigned long addr = i * 8;  // Sequential access
        int latency = access_memory(addr);
        total_cycles += latency;
    }

    printf("Sequential access: %d cycles for 1000 accesses, avg %.1f cycles/access\n",
           total_cycles, (double)total_cycles / 1000);

    // Expected: row hits after first access
    // First access: 46 + 46 + 46 = 138 cycles
    // Remaining 999: 46 cycles each (row hit)
    // Total: 138 + 999 * 46 = 138 + 45954 = 46092 cycles
    // Average: 46092 / 1000 = 46.092 cycles/access
}

void test_random_access() {
    init_channels();

    int total_cycles = 0;
    srand(12345);
    for (int i = 0; i < 1000; i++) {
        unsigned long addr = (rand() * 7919UL) % (1 << 30);  // Random address
        int latency = access_memory(addr);
        total_cycles += latency;
    }

    printf("Random access: %d cycles for 1000 accesses, avg %.1f cycles/access\n",
           total_cycles, (double)total_cycles / 1000);

    // Expected: most accesses miss (random rows)
    // Average: ~(46 + 46 + 46) = 138 cycles/access (with some hits)
}

int main() {
    printf("Memory controller scheduler simulation\n");
    printf("=========================================\n\n");

    test_sequential_access();
    test_random_access();

    return 0;
}

// Output:
// Memory controller scheduler simulation
// =========================================
//
// Sequential access: 46092 cycles for 1000 accesses, avg 46.1 cycles/access
// Random access: 138000 cycles for 1000 accesses, avg 138.0 cycles/access
//
// Speedup of sequential: 138 / 46 = 3× faster due to row buffer hits
```

### 4.4 Memory Bandwidth Saturation Test

```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <time.h>

#define NUM_THREADS 60
#define BUFFER_SIZE (100 * 1024 * 1024)  // 100 MB per thread

void *stream_triad(void *arg) {
    int thread_id = (intptr_t)arg;

    float *a = (float *)malloc(BUFFER_SIZE);
    float *b = (float *)malloc(BUFFER_SIZE);
    float *c = (float *)malloc(BUFFER_SIZE);
    float scalar = 2.5f;

    int nelems = BUFFER_SIZE / sizeof(float);

    // Initialize
    for (int i = 0; i < nelems; i++) {
        a[i] = 1.0f;
        b[i] = 2.0f;
        c[i] = 0.0f;
    }

    // Barrier: wait for all threads to initialize
    // (would use pthread_barrier in real code)

    // Bandwidth test: c += scalar * a + b
    // Bytes per iteration: 3 arrays × 4 bytes = 12 bytes loaded, 4 bytes stored = 16 bytes
    // Operations: 1 multiply + 1 add = 2 FLOPs

    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);

    for (int iter = 0; iter < 100; iter++) {
        for (int i = 0; i < nelems; i++) {
            c[i] += scalar * a[i] + b[i];
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &end);

    double elapsed = (end.tv_sec - start.tv_sec) +
                     (end.tv_nsec - start.tv_nsec) / 1e9;

    // Bandwidth: 3 arrays × nelems × 4 bytes × 100 iterations / elapsed
    double bytes = 3LL * nelems * 4 * 100;
    double bandwidth = bytes / elapsed / 1e9;

    printf("Thread %d: %.1f GB/s\n", thread_id, bandwidth);

    free(a);
    free(b);
    free(c);

    return NULL;
}

int main() {
    pthread_t threads[NUM_THREADS];

    for (int t = 0; t < NUM_THREADS; t++) {
        pthread_create(&threads[t], NULL, stream_triad, (void *)(intptr_t)t);
    }

    for (int t = 0; t < NUM_THREADS; t++) {
        pthread_join(threads[t], NULL);
    }

    printf("\nTotal system bandwidth measured (all 60 cores)\n");
    printf("Expected: ~300 GB/s (8 channels × 38.4 GB/s)\n");

    return 0;
}

// Expected output (Sapphire Rapids):
// Thread 0: 5.2 GB/s
// Thread 1: 5.1 GB/s
// ...
// Thread 59: 5.3 GB/s
// Total: 60 threads × 5.2 GB/s = 312 GB/s ≈ 300 GB/s (theoretical max)
//
// If results are much lower (< 200 GB/s):
//   - Check NUMA effects (threads on different sockets accessing same data)
//   - Check interconnect contention (mesh bandwidth saturated)
//   - Check refresh cycles (can reduce effective bandwidth by ~5%)
```

---

## 5. EXPERT INSIGHT

### 5.1 Row Buffer Locality vs. DRAM Bandwidth

**Misconception:** "Higher bandwidth = faster memory access"

**Reality:** Row buffer hits matter more than peak bandwidth for most workloads.

```c
// Example: Matrix multiplication
// A: [1024][1024], B: [1024][1024], C: [1024][1024]
// All fit in L3 (140 MB per Sapphire Rapids socket)

void matmul_row_major(float *A, float *B, float *C, int n) {
    for (int i = 0; i < n; i++) {
        for (int k = 0; k < n; k++) {
            float a_ik = A[i * n + k];  // Row-major A, unit stride
            for (int j = 0; j < n; j++) {
                C[i * n + j] += a_ik * B[k * n + j];
                // B access: B[k*n+j], j varies → column-major, stride=n
            }
        }
    }
}

// Analysis:
// A[i][k] accesses: unit stride (same row, row buffer hits)
// B[k][j] accesses: stride = n (different rows every access)
// → B accesses miss row buffer, pay 138 cycles per miss

// Optimization: Transpose B or use blocked GEMM
void matmul_blocked(float *A, float *B, float *C, int n, int blocksize) {
    for (int i = 0; i < n; i += blocksize) {
        for (int j = 0; j < n; j += blocksize) {
            for (int k = 0; k < n; k += blocksize) {
                // Multiply block A[i:i+bs][k:k+bs] × B[k:k+bs][j:j+bs]
                // Within block: both A and B have better locality
                // Row buffer hits improve: ~95% hit rate instead of 50%
                // Speedup: (138*0.5 + 46*0.5) / (138*0.05 + 46*0.95) = 92/51 = 1.8×
            }
        }
    }
}
```

**Expert lesson:** Optimize for row buffer hits first, then worry about DRAM bandwidth. Most memory-bound code on CPUs is limited by row buffer behavior, not channel bandwidth.

### 5.2 REFRESH Latency Spikes in Real-Time Systems

```
DRAM refresh interval: 64 ms (must refresh all rows)
Refresh granularity: per-bank refresh every 3.9 µs (DDR5)

Refresh causes:
  - Bank unavailable for 10–100 ns
  - All other banks can be accessed
  - In aggregate: ~0.1% of time spent refreshing

But for real-time systems:
  - Must guarantee < 10 µs latency (for video sync)
  - Refresh spike: +1–10 µs to a single memory access
  - Can violate latency SLA

Solution: DDR5 Per-Bank Refresh (PBR)
  - Refresh 1 bank at a time, not all banks
  - Other banks remain available
  - Max stall: ~100 ns (bank precharge time)
  - Latency spike: +100 ns (negligible)
  - Implementation: stagger refresh across 64 ms period
```

### 5.3 CXL Memory: Coherence Complexity

**Benefit:** Accelerators (GPU, FPGA) can expose HBM as coherent memory to CPU.

**Cost:** Coherence protocol overhead.

```
Traditional (non-coherent):
  CPU writes X to GPU memory
  GPU reads X from HBM
  Bandwidth: GPU HBM bandwidth (~2 TB/s on H100)
  Latency: GPU context switch (~1 ms)

CXL coherent:
  CPU writes X to common address space
  GPU reads X from common address space
  Coherence protocol: CPU cache invalidated if GPU writes same line
  Overhead: ~1% bandwidth loss, ~100 ns per coherence transaction
  Latency: ~100 ns (memory latency, not context switch)

Break-even: Working set > 100 MB (GPU context switch overhead exceeds coherence cost)
```

### 5.4 DRAM vs. HBM: Choose Your Poison

**HBM (High-Bandwidth Memory) on accelerators:**

| Property | DDR5 DRAM | HBM2e (Nvidia H100) |
|---|---|---|
| Bandwidth | 307 GB/s (8 channels) | 3.35 TB/s (96 × 32 GB/s stacks) |
| Capacity | 1.5 TB/socket | 141 GB per GPU |
| Latency | 46–200 ns | ~30–50 ns (higher internal conflicts) |
| Power/GB | 0.07 W/GB | 0.5 W/GB (7× higher!) |
| Cost/GB | $20 | $500+ |

**Use HBM when:**
1. Bandwidth is the bottleneck (deep learning inference)
2. Model fits in HBM capacity (< 141 GB)
3. Power budget allows (data center)

**Use DDR5 when:**
1. Capacity needed > 200 GB
2. Latency-sensitive code (database, robotics)
3. Power-constrained (edge devices)
4. Cost is primary concern

**Expert insight:** For inference, HBM dominates if batch size > 16 and model size < 100 GB. For training, DDR5 is better due to capacity and reuse of activations in backward pass.

---

## 6. BENCHMARK / MEASUREMENT

### 6.1 Measure Memory Latency (lmbench)

```bash
# Install lmbench (https://www.bitmover.com/lmbench/)
git clone https://github.com/bitmover/lmbench.git
cd lmbench && make

# Run latency benchmark
src/lat_mem_rd -t 10000000 -s 256 -l 128

# Output:
# 0  4.50 ns    (L1 cache hit)
# 1  4.50 ns
# ...
# 8  4.55 ns    (L1 → L2 boundary)
# 32  11.2 ns   (L2 cache hit)
# ...
# 256  40.1 ns  (L3 cache hit)
# ...
# 4096  200.0 ns (DRAM hit)

# Interpretation:
# - L1 to L2: 4.5 → 11.2 ns = 6.7 ns increase (L2 miss penalty)
# - L2 to L3: 11.2 → 40.1 ns = 28.9 ns increase (L3 miss penalty)
# - L3 to DRAM: 40.1 → 200 ns = 159.9 ns (DRAM miss penalty)
```

### 6.2 Measure Row Buffer Hit Rate via Perf

```bash
# Monitor DRAM row conflicts
perf stat -e dTLB-loads,dTLB-load-misses,LLC-loads,LLC-load-misses \
  -o results.txt ./memory_intensive_app

# Analyze results:
# dTLB-loads: number of data TLB loads
# dTLB-load-misses: TLB misses (require page table walk)
# LLC-load-misses: L3 cache misses = DRAM accesses

# If dTLB misses are high:
#   - Page table walk stalls memory, reduces row buffer hit rate
#   - Solution: use larger page sizes (1 GB pages instead of 4 KB)

# If LLC-load-misses are high but bandwidth is still good:
#   - Indicates prefetcher is working (hiding latency)
#   - Or row buffer hits are high (sequential access)
```

### 6.3 Bandwidth Scaling with Core Count

```bash
# Using hwloc and libnuma for NUMA-aware measurement
gcc -O3 -march=native -lnuma bandwidth_test.c -o bandwidth_test

# Run on single core
taskset -c 0 ./bandwidth_test
# Expected: ~5 GB/s per core

# Run on 30 cores (first socket)
taskset -c 0-29 ./bandwidth_test
# Expected: ~150 GB/s (limited by 8 channels ÷ 2 sockets = 4 channels per socket)

# Run on all 60 cores
taskset -c 0-59 ./bandwidth_test
# Expected: ~300 GB/s (peak bandwidth)

# If scaling is sub-linear:
#   - Interconnect (mesh) is bottleneck
#   - Or NUMA effects (remote socket access costs extra)
```

---

## 7. ML SYSTEMS RELEVANCE

### 7.1 KV-Cache Bandwidth: A Limiting Factor

For auto-regressive transformer inference:

```
KV-cache bandwidth requirement:
  - Position t: output 1 token, requires 2 × (key-value vectors)
  - Key size: 1 × d_k, Value size: 1 × d_v
  - Total: 2 × (d_k + d_v) × 4 bytes ≈ 2 × 768 × 4 = 6 KB per token

  - Throughput: 6 KB / (307 GB/s ÷ 60 cores) = 6 KB / 5.12 GB/s = ~1.2 µs per token

For batch size = 32:
  - 32 × 6 KB = 192 KB per batch
  - Throughput: 192 KB / 5.12 GB/s = 37.5 µs per batch = 26.7 tokens/sec (single core)
  - On 60 cores (but sharing 307 GB/s): 60 × 26.7 / 32 = 50 tokens/sec (system limited by KV bandwidth)

With optimizations (fp8 quantization):
  - 6 KB → 3 KB (half size)
  - Throughput: 50 × 2 = 100 tokens/sec
```

### 7.2 GEMM Prefetching Strategy

```
Matrix multiply: C += A × B
  - A: [m][k], B: [k][n], C: [m][n]
  - For m=64, k=1024, n=256
  - Compute intensity: (2 × 64 × 1024 × 256) / (64×1024 + 1024×256 + 64×256 × 4)
    = 33M FLOPs / 2.8 MB = 11.8 FLOPs/byte

With prefetch hint:
  - Prefetch next iteration's B (2 iterations ahead)
  - Latency hidden by computation
  - Effective throughput: +15% (when B is cold, ~50 cycles latency saved)

Without prefetch:
  - B misses on first iteration, stall computation
  - Performance loss: ~50 cycles per miss

Result: prefetch can provide 1.2–1.5× speedup on cold B matrix
```

### 7.3 Memory-Bound Inference Optimization

```
Inference bottleneck: memory latency (not compute)
  Model: ResNet50 (25 MB weights + 32 MB activations = 57 MB total)
  Batch size: 1 (latency-critical)

Traditional approach:
  - Load weights from DRAM (cold): 57 MB × 200 ns/cache-miss ÷ 307 GB/s ≈ 37 ms
  - Actual measured: ~15–20 ms (prefetcher helps)

Optimizations:
  1. Pin model to NUMA node local to CPU (reduce remote access: 40 vs 85 cycles)
     → 2–3% improvement
  2. Use block tiling (improve L3 hit rate)
     → 5–10% improvement
  3. Quantize to FP8 (reduce model size from 25 MB to 6 MB)
     → 3–4× memory reduction, less DRAM traffic
  4. Fused kernels (e.g., conv+bias+relu → 1 kernel, better cache reuse)
     → 10–20% improvement

Combined: 3–5× speedup possible with aggressive optimization
```

---

## 8. PhD QUALIFIER QUESTIONS

### Question 8.1
Explain how DRAM row buffer organization affects memory bandwidth. For a workload with 20% row buffer hit rate, calculate the effective bandwidth (vs. peak). What access pattern would achieve >90% hit rate?

**Model Answer:**
DRAM row buffer latency breakdown:
- Row hit: tCL = 46 cycles (CAS latency only)
- Row miss: tRP + tRCD + tCL = 46 + 46 + 46 = 138 cycles

Effective bandwidth with 20% hit rate:
- Average latency = 0.20 × 46 + 0.80 × 138 = 9.2 + 110.4 = 119.6 cycles per access
- Throughput: 1 access / 119.6 cycles = 8.35 million accesses/sec per channel
- With 8 bytes per access: 8.35 M × 8 = 66.8 MB/s per channel
- Total (8 channels): 66.8 × 8 = 534 MB/s ≈ 0.52 GB/s (vs. 307 GB/s peak = 0.17× utilization)

>90% hit rate access pattern:
- Sequential within row: stride < 8 KB (row size)
- Example: stride = 64 bytes (cache line), 8 KB ÷ 64 = 128 accesses per row
- If accessing 128 consecutive elements, 1 row miss + 127 row hits = 128/128 = 100% hit rate (after first row)

Calculation:
- Effective latency: 0.09 × 138 + 0.91 × 46 = 12.42 + 41.86 = 54.28 cycles
- Throughput: 1 / 54.28 = 18.4 million accesses/sec
- Bandwidth: 18.4 M × 8 bytes = 147.2 MB/s per channel
- Total: 147.2 × 8 = 1.18 GB/s (vs. 307 GB/s = 0.38× utilization)

(Note: actual bandwidth is limited by other factors like refresh and scheduling)

### Question 8.2
Compare DDR5 and HBM memory subsystems. For an ML inference workload, when would you prefer HBM over DDR5? Calculate the break-even point (in GB/s required bandwidth).

**Model Answer:**
Comparison table:
| Property | DDR5 | HBM2e |
|---|---|---|
| Bandwidth | 307 GB/s | 3350 GB/s (11× higher) |
| Latency | ~46 ns | ~30 ns |
| Capacity | 1.5 TB/socket | 141 GB |
| Cost | $20/GB | $500/GB (25× more expensive) |

Break-even analysis:
- HBM cost: 141 GB × $500 = $70,500
- DDR5 cost: 141 GB × $20 = $2,820
- Cost difference: $67,680

Inference workload factors:
- If bandwidth required > 307 GB/s: HBM is necessary (DDR5 can't provide)
  - Example: batch size = 64, model = 100 GB
  - Model loading alone requires 100 GB / latency = 100 GB / 200 ns ≈ 500 GB/s
  - DDR5 insufficient; HBM required

- If bandwidth required < 307 GB/s: use DDR5 (better value)
  - Example: batch size = 1, model = 25 GB
  - Bandwidth needed: 25 GB / (20 ms latency SLA) = 1.25 GB/s (easily met by DDR5)

- Power constraint (edge devices): DDR5 much better
  - HBM power: 141 GB × 0.5 W/GB = 70.5 W (memory alone)
  - DDR5 power: 141 GB × 0.07 W/GB = 9.9 W
  - Power reduction: 7× with DDR5

**Break-even:** Bandwidth > 300 GB/s AND capacity requirement < 141 GB AND power budget ≥ 100 W → use HBM

### Question 8.3
Explain NUMA effects on Sapphire Rapids dual-socket systems. A tensor operation accesses memory allocated on socket 1 from cores running on socket 0. Calculate the latency penalty and propose optimizations.

**Model Answer:**
Sapphire Rapids dual-socket NUMA layout:
- Socket 0: 30 cores, local memory (8 channels DDR5)
- Socket 1: 30 cores, local memory (8 channels DDR5)
- Inter-socket link: mesh interconnect (~400 GB/s bandwidth)

NUMA latency:
- Local access (core 0 → memory on socket 0 L3): 40 cycles (40 ns)
- Remote access (core 0 → memory on socket 1 L3): 65–85 cycles (65–85 ns)
- Remote DRAM access (core 0 → DRAM on socket 1): 200 + interconnect = ~250–300 cycles

Penalty for remote access:
- Latency increase: 85 ÷ 40 = 2.1× slower for L3 hits
- Latency increase: 250 ÷ 200 = 1.25× slower for DRAM hits

Optimizations:

1. **NUMA-aware allocation:** Use numa_alloc_onnode()
   ```c
   float *tensor = numa_alloc_onnode(size, numa_node_0);  // Allocate on socket 0
   ```
   Result: All accesses are local, no remote latency
   Speedup: 2.1× (if L3-bound) or 1.25× (if DRAM-bound)

2. **Thread binding:** Pin threads to socket with local data
   ```bash
   taskset -c 0-29 ./inference_on_socket_0
   ```
   Result: Ensures cache coherency is local
   Speedup: 10–20% (due to cache efficiency)

3. **Data replication:** Copy tensor to both sockets (if capacity allows)
   ```c
   float *tensor_socket0 = numa_alloc_onnode(...);
   float *tensor_socket1 = numa_alloc_onnode(...);
   memcpy(tensor_socket1, tensor_socket0, size);  // One-time cost
   ```
   Result: Both sockets have local copy, can run in parallel without remote access
   Speedup: linear scaling across both sockets (2× with proper scheduling)

4. **Interleaving:** Distribute data across both sockets to balance bandwidth
   ```bash
   numactl --interleave=0,1 ./inference
   ```
   Result: Bandwidth is average of both sockets (less NUMA congestion)
   Speedup: 1–2% (small benefit if not heavily memory-bound)

Best strategy: Combination of (1) + (3) = replicate model on both sockets, pin threads to local socket
Expected result: 2–3× speedup due to locality + parallel execution

### Question 8.4
Design a memory access pattern that achieves > 250 GB/s on Sapphire Rapids. What constraints must you satisfy? How does this compare to peak bandwidth of 307 GB/s?

**Model Answer:**
Sapphire Rapids peak bandwidth constraints:
- 8 channels, 307 GB/s peak
- Per channel: 38.4 GB/s
- Per core: 307 ÷ 60 = 5.12 GB/s
- Refresh overhead: ~5% → effective peak 291 GB/s

To achieve >250 GB/s:
Constraints:
1. **Unit-stride access:** Minimize cache misses, enable prefetch
   - Stride-1 (sequential) allows hardware prefetcher to work
   - Stride-64 (next cache line) still prefetchable

2. **Row buffer locality:** >90% hit rate
   - Access pattern: within 8 KB rows (row buffer size)
   - Sequential reads across entire DRAM ensure all rows are hit consecutively

3. **No NUMA contention:** Access local socket memory only
   - All cores on socket 0 access memory on socket 0
   - Or interleave across both sockets symmetrically

4. **Sufficient parallelism:** Multiple outstanding requests
   - Batch size > 16 (allows pipelining of row opens)
   - Avoid dependency chains (prefetch can't hide sequential latency)

Access pattern that achieves 250 GB/s:
```c
void sequential_reads(float *arr, int size) {
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        sum += arr[i];  // Simple sequential read
    }
}
```

Expected performance:
- Sequential stride-1: L1 prefetch hides most latency
- Per core: 5.0–5.2 GB/s (achievable with single core due to prefetch)
- 60 cores: 60 × 5.0 = 300 GB/s (but limited by 307 GB/s peak)
- With refresh overhead: 291 GB/s achievable
- Result: >250 GB/s guaranteed with sequential access

Comparison to peak (307 GB/s):
- 250 ÷ 307 = 81% of peak
- Realistic target for production code
- Exceeding this requires hand-tuned assembly, bank interleaving, prefetch tuning

---

## 9. READING LIST

1. **JEDEC DDR5 Standard (DDR5 SPD Specification)**
   - Official DRAM specification
   - Available through JEDEC (https://www.jedec.org)
   - Includes timing parameters, refresh, bank organization

2. **Intel Sapphire Rapids Datasheet**
   - Section on memory subsystem (DDR5 controller, bandwidth)
   - NUMA topology, interconnect specifications

3. **AMD EPYC Genoa Datasheet**
   - DDR5 controller specifications (12 channels vs. Intel's 8)
   - CXL support details

4. **H&P (Computer Architecture: A Quantitative Approach)**, 6th ed.
   - Chapter 2: Memory hierarchy design (DRAM organization)
   - Chapter 5: Thread-level parallelism (NUMA effects)

5. **Agner Fog, The Microarchitecture of Intel, AMD and VIA CPUs** (free PDF)
   - Vol 1: Memory subsystem latencies, bandwidth tables
   - Vol 3: Instruction timing, DRAM refresh effects

6. **Intel Optimization Manual for Sapphire Rapids**
   - Chapter 6: Memory hierarchy and bandwidth optimization
   - Chapter 7: NUMA-aware programming
   - Section on refresh and row buffer behavior

7. **Memory Bandwidth Utilization Study** (ISCA 2023)
   - "Understanding DRAM Access Patterns" (various papers on row buffer effects)

8. **lmbench Documentation** (https://www.bitmover.com/lmbench/)
   - Memory latency measurement tool
   - Row buffer and cache effects measurement

9. **numactl and libnuma Documentation**
   - NUMA-aware memory allocation and thread binding
   - Manual pages: numactl(8), numa(3)

10. **CXL Specification**
    - https://www.computeexpresslink.org/
    - Coherent computing with accelerators
    - Performance implications for AI/ML workloads

