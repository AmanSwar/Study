# MODULE 19 — Intel Xeon Memory Subsystem

## 1. CONCEPTUAL FOUNDATION

### Memory Hierarchy on Sapphire Rapids

The Sapphire Rapids memory subsystem is the most complex component an ML systems
engineer will encounter. It must balance:
- **Latency**: L1 @ 4 cycles, L3 @ 32 cycles, DRAM @ 110+ cycles
- **Bandwidth**: L1D 256 GB/s theoretical, L3 distributed, DDR5 400+ GB/s
- **Coherency**: Cache line consistency across 4 tiles, 2 sockets, and NUMA domains
- **QoS**: Intel RDT (Resource Director Technology) for ML inference isolation

**Reference**: Intel Xeon Scalable Processor Architecture Specification (XSPS),
Section 4 "Memory Subsystem"; DDR5 Standard JEDEC (JESD79-6C).

### DDR5 on Sapphire Rapids

**Configuration**: 8 channels, 4800-5600 MT/s (megatransfers/second)

```
Peak bandwidth per channel:
  5600 MT/s × 8 bytes/transfer × 8 channels = 358.4 GB/s (5600 speed)
  4800 MT/s × 8 bytes/transfer × 8 channels = 307.2 GB/s (4800 speed)

Practical sustained bandwidth (accounting for refresh, latency, command scheduling):
  ~85-90% of peak = 304-323 GB/s achievable in practice
```

**Architecture**:
```
┌──────────────────────────────────────────────────────────┐
│ Sapphire Rapids Socket (60 cores in 4 tiles)           │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  L3 Cache (144MB total: 36MB × 4 tiles)                │
│  ↓ (mesh/tile interconnect)                            │
│                                                          │
│  ┌──────────────────────────────────┐                  │
│  │   Integrated Memory Controller    │                  │
│  │   (IMC)                          │                  │
│  │  - 2 IMCs, 4 channels each      │                  │
│  │  - DDR5 PHY (physical layer)    │                  │
│  │  - Command/address scheduling   │                  │
│  │  - Refresh management           │                  │
│  │  - Page table walks (?)         │                  │
│  └──────────────────────────────────┘                  │
│         ↓  (64-bit bus per channel)                    │
│                                                          │
│  ┌─────────┬──────────┬──────────┬─────────┐           │
│  │ DDR5    │  DDR5    │  DDR5    │  DDR5   │           │
│  │ DIMM 0  │  DIMM 1  │  DIMM 2  │  DIMM 3 │ (Ch 0-3)  │
│  │         │          │          │         │           │
│  │ 4 DIMM  │ 4 DIMM   │ 4 DIMM   │ 4 DIMM  │           │
│  │ slots   │ slots    │ slots    │ slots   │           │
│  └─────────┴──────────┴──────────┴─────────┘           │
│                                                          │
│  ┌─────────┬──────────┬──────────┬─────────┐           │
│  │ DDR5    │  DDR5    │  DDR5    │  DDR5   │           │
│  │ DIMM 4  │  DIMM 5  │  DIMM 6  │  DIMM 7 │ (Ch 4-7)  │
│  │         │          │          │         │           │
│  └─────────┴──────────┴──────────┴─────────┘           │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

### Integrated Memory Controller (IMC)

Each socket has **2 IMCs** controlling 4 DDR5 channels each. This dual-IMC
design enables:
- **Load balancing**: Interleave memory accesses across both IMCs
- **Bandwidth aggregation**: Both IMCs can be saturated for 400+ GB/s
- **Fault tolerance**: One IMC failure doesn't disable entire DRAM

**IMC scheduling policy** (simplified):

```
1. Row buffer hits are prioritized (cheapest access)
2. Commands are reordered to maximize bank parallelism
3. Refresh commands inserted every 64ms (critical for DRAM reliability)
4. Write combining to reduce traffic (when possible)
```

---

## 2. MENTAL MODEL

### Memory Latency Pyramid

```
DISTANCE                  LATENCY                    CAPACITY

┌─────────────────────────────────────────────┐
│                                             │
│  L1 Data Cache                              │    4 cycles
│  (32KB per core, 8-way, write-through)      │
│                                             │
│  Typical DRAM access:                       │
│  • L1 hit (all 48-byte cache line used)     │
│  • Every 4 cycles, all 32KB cache loaded    │
│  • Perfect spatial + temporal locality      │
│  • Bandwidth: 32KB × (3.5GHz / 4) = 28 GB/s│
│                                             │
└─────────────────────────────────────────────┘
                    ↓ (L1 miss)

┌─────────────────────────────────────────────┐
│                                             │
│  L2 Cache                                   │    12 cycles
│  (1.25MB per core, private, 8-way)          │
│                                             │
│  Second-level miss cost:                    │
│  • Typical load-to-use: 12 cycles           │
│  • Bandwidth: 1.25M × (3.5GHz / 12) = 365 G/s (theoretical, not sustained)
│  • Real throughput: ~50-70 GB/s (2 loads/cycle max)
│                                             │
└─────────────────────────────────────────────┘
                    ↓ (L2 miss)

┌─────────────────────────────────────────────┐
│                                             │
│  L3 Cache (LLC)                             │    32 cycles (same tile)
│  (144MB total, 36MB per tile, distributed)  │    40+ cycles (cross-tile)
│                                             │
│  L3 hit cost:                               │
│  • Same tile: ~8-10 µs @ 3.5 GHz            │
│  • Cross-tile (HBI): ~11-15 µs              │
│  • Bandwidth: ~400 GB/s aggregate across    │
│    all tiles (not per-core)                 │
│                                             │
└─────────────────────────────────────────────┘
                    ↓ (L3 miss)

┌─────────────────────────────────────────────┐
│                                             │
│  Main Memory (DDR5)                         │   110+ cycles
│  (Up to 768 GB per socket)                  │
│                                             │
│  DRAM access cost:                          │
│  • First access: 110-120 cycles (~31 µs)    │
│  • Row-buffer hit: ~60 cycles (much better) │
│  • Bandwidth: 400 GB/s (aggregate, 8 ch)    │
│  • Typical inference: 50-70% row-buffer hit │
│                                             │
└─────────────────────────────────────────────┘
                    ↓ (DRAM miss to remote socket)

┌─────────────────────────────────────────────┐
│                                             │
│  Remote Socket Memory (via QPI)             │   180+ cycles
│                                             │
│  Cross-socket access:                       │
│  • QPI latency: ~60-70 additional cycles    │
│  • Used for NUMA-unaware workloads          │
│  • Intel RDT can isolate to prevent this    │
│                                             │
└─────────────────────────────────────────────┘
```

### Distributed Cache Organization

SPR's L3 is **distributed by physical address**, not by core. This is critical:

```
Tile 0 L3: addresses with bits [11:6] = [0:20]
  (approximately; exact mapping complex)
  Serves cores 0-11 optimally but not exclusively

Tile 1 L3: addresses with bits [11:6] = [21:41]

Tile 2 L3: addresses with bits [11:6] = [42:62]

Tile 3 L3: addresses with bits [11:6] = [63:83]
```

**Implication**: If thread on core 0 accesses an address hashing to Tile 2's L3,
it incurs a 25-35 cycle cross-tile penalty.

### NUMA Layout: SNC (Sub-NUMA Clustering)

Sapphire Rapids supports **SNC mode** (available on some BIOS configurations):

```
WITHOUT SNC (default):
  Entire 60-core socket appears as single NUMA node
  All 8 DDR5 channels equidistant from all cores
  Single QPI link to remote socket

WITH SNC=4 (4 sub-nodes per socket):
  Tile 0: 12 cores + 2 DDR5 channels (NUMA node 0)
  Tile 1: 12 cores + 2 DDR5 channels (NUMA node 1)
  Tile 2: 12 cores + 2 DDR5 channels (NUMA node 2)
  Tile 3: 12 cores + 2 DDR5 channels (NUMA node 3)

  Within-tile (same NUMA node): ~60 cycles to DRAM
  Cross-tile (different NUMA node): ~90-100 cycles to DRAM
```

**For inference**: SNC=4 is strongly recommended. Enables OS scheduler to keep
inference batches pinned to same tile, minimizing cross-tile DRAM traffic.

---

## 3. PERFORMANCE LENS

### The Memory Wall

A single Sapphire Rapids core can theoretically:
- Execute 4 FMA per cycle (ports 10-11 with unrolling) = 28 GFLOPS @ 3.5 GHz
- Require data: 28 GFLOPS × 4 bytes/float × 3 ops = 336 GB/s

But DDR5 provides only 400 GB/s for the **entire socket** (60 cores).

**Implication**: You cannot saturate compute with DRAM bandwidth. This is why:
1. **Cache locality is paramount**: L3 misses are catastrophic
2. **Reuse is mandatory**: Compute-to-memory ratio must be >> 1
3. **Tiling is essential**: GEMM operations must tile to fit in L3

### Row-Buffer Locality

DDR5 DRAM has **row buffers** (chunks of memory held in fast SRAM):

```
DRAM layout:
  Bank 0: Row 0 (8KB), Row 1 (8KB), Row 2 (8KB), ...
  Bank 1: Row 0 (8KB), Row 1 (8KB), Row 2 (8KB), ...
  ... (16-32 banks total per channel)

Access latencies:
  Row-buffer hit (same row): ~35-40 cycles
  Row-buffer miss (different row): ~100+ cycles
  Difference: 60-70 cycle penalty!
```

For inference to achieve good DRAM bandwidth:
- Must have 50-70% row-buffer hits
- This requires sequential or stride-4+ memory patterns
- Random access patterns get only ~50-70 GB/s (not 400 GB/s)

**For ML**: Inference data layout must respect DRAM row-buffer organization.
Weight matrices in row-major format (not column-major) for optimal DRAM access.

---

## 4. ANNOTATED CODE

### Example 1: Cache Coherency with MESI Protocol

```c
#include <stdint.h>
#include <string.h>

// Sapphire Rapids uses MESI cache coherency protocol
// M (Modified), E (Exclusive), S (Shared), I (Invalid)

// Simplified coherency scenario: two tiles writing to same cache line

struct cache_line {
    uint64_t data[8];  // 64 bytes (typical cache line size)
};

volatile struct cache_line shared_data;

// Thread on Tile 0:
void tile0_write(uint64_t value) {
    // Line 1: Load (L3 miss on first access)
    // Cache line enters L3 in EXCLUSIVE state (only Tile 0 has it)
    // Latency: 110+ cycles from DRAM

    // Line 2: Write
    shared_data.data[0] = value;
    // Cache line state: MODIFIED (Tile 0 has modified copy)
    // Other tiles' copies (if any) invalidated

    // Line 3: Memory barrier
    __asm__ volatile ("mfence");  // Memory fence: ensure write visible globally
    // Mfence latency: ~12 cycles, blocks further memory ops
}

// Thread on Tile 2 (running on same socket):
void tile2_read(void) {
    // Line 1: Load from shared_data
    // This triggers cache coherency protocol:
    //   1. Tile 2 L3 sends read request for cache line
    //   2. Tile 0 L3 detects it has MODIFIED copy
    //   3. Tile 0 responds with data via HBI
    //   4. Tile 2 L3 gets copy in SHARED state
    // Cost: ~35-40 cycles (cross-tile coherency on HBI)

    uint64_t value = shared_data.data[0];

    // If Tile 0 modifies again:
    //   - Tile 2's copy invalidated
    //   - Next read from Tile 2 triggers coherency again
    // This is "cache line bouncing" ← destructive for performance
}

// Annotated performance:
// - First read (L3 miss): 110 cycles (DRAM latency)
// - Second read (L3 hit): 8 cycles (same tile)
// - First write (L3 miss): 110 cycles + coherency
// - Cross-tile access: +35 cycles (HBI coherency)

// For inference: Avoid shared cache lines across tiles!
// Better approach: thread-local data structures
```

### Example 2: LLC Slice Distribution Measurement

```c
#include <stdio.h>
#include <stdint.h>
#include <string.h>

// Sapphire Rapids L3 slicing is complex, but roughly:
// Address bits [11:6] determine which tile's L3 slice
// This can cause imbalance if allocation is poor

void measure_l3_slicing(void) {
    // Allocate arrays with specific stride patterns
    int *array1 = malloc(36 * 1024 * 1024);  // 36MB (fits in one tile's L3)
    int *array2 = malloc(36 * 1024 * 1024);  // Another 36MB

    // Line 1: Touch array1 sequentially
    // This fills Tile 0's L3 slice (assuming address hashing lands there)
    for (int i = 0; i < 9 * 1024 * 1024; i++) {
        array1[i] = i;  // Sequential write, row-buffer friendly
    }

    // Line 2: If array2 hashes to same Tile 0 L3 slice
    // Touching it will evict array1 data (capacity miss)
    for (int i = 0; i < 9 * 1024 * 1024; i++) {
        array2[i] = i;  // Potential evictions
    }

    // Line 3: Measure L3 miss rate with perf
    // perf stat -e cache-references,cache-misses ./this_program
    //
    // Expected output:
    // cache-references: 54M (from arrays 1 and 2)
    // cache-misses: 18M (one-third, due to 3× memory touches)
    //
    // Lesson: naive allocation can fragment L3 across slices

    // Better approach: Align allocation to tile boundaries
    // Some inference frameworks do this explicitly

    free(array1);
    free(array2);
}

// Assembly-level cache line eviction (simplified):
//
// mov (%rax), %ecx          ; Load from array1[i]
//   L3 hit, 32-cycle latency
// mov (%rbx), %ecx          ; Load from array2[i]
//   If array2[i] hashes to same L3 slice as array1[i]:
//   Cache line eviction policy triggers
//   Typical: LRU (Least Recently Used) across slices
//   Cost: ~80-110 cycles to reload evicted cache line
```

### Example 3: Intel RDT (Resource Director Technology) for Inference QoS

```c
#include <intel-cmt-cat.h>

// Intel RDT provides cache allocation and memory bandwidth monitoring
// Critical for multi-tenant inference serving

#define PARTITION_INFERENCE 0
#define PARTITION_BACKGROUND 1

void setup_rdt_for_inference(void) {
    // Line 1: Initialize RDT library
    int ret = pqos_init(&cfg);  // Configure RDT

    // Line 2: Allocate L3 cache partition for inference
    // Allocate 100MB of 144MB L3 to inference, rest to background tasks
    struct pqos_l3ca cache_cfg;
    memset(&cache_cfg, 0, sizeof(cache_cfg));

    // Cache allocation bitmask: bits represent cache ways (slices)
    // All 1s = full L3 access (144MB)
    // Half 1s = half L3 access (72MB)
    cache_cfg.cdp = 0;  // Disable code/data partitioning
    cache_cfg.u.ways_mask = 0xFFF;  // Allocate bits [11:0] = 12 ways = 100MB

    pqos_l3ca_assign(PARTITION_INFERENCE, &cache_cfg);
    // Now, threads in partition INFERENCE get preferential L3 access

    // Line 3: Monitor memory bandwidth
    // Track which core groups are consuming DDR5 bandwidth
    struct pqos_mon_data *mon_data;
    pqos_mon_start(
        1,                           // 1 monitoring group
        &PARTITION_INFERENCE,        // Which partition
        PQOS_MON_EVENT_L3_MISS_RATE, // Monitor L3 misses
        (void *)mon_data
    );

    // Line 4: Inference serving loop
    for (int i = 0; i < 1000000; i++) {
        // Process inference request
        run_inference(i);

        // Periodic check: if background tasks exceed L3 miss threshold,
        // trigger cache flush or priority adjustment
        if (i % 1000 == 0) {
            pqos_mon_poll(&mon_data, 1);
            if (mon_data->values.l3_miss > THRESHOLD) {
                // Too many L3 misses: increase isolation
                // Reduce background task L3 allocation
                printf("L3 misses high: %lu\n", mon_data->values.l3_miss);
            }
        }
    }

    pqos_mon_stop(mon_data);
}

// Expected results with RDT isolation:
// - Without RDT: Background tasks evict inference cache lines
//   L3 miss rate: 8-12%, latency variance: 100-200 µs
// - With RDT (100MB partition): Background isolated
//   L3 miss rate: 2-4%, latency variance: 20-50 µs
// Latency SLA improvement: 4-8x better p99
```

---

## 5. EXPERT INSIGHT

### Myth #1: More DRAM Bandwidth = Faster Inference

**Reality**: Bandwidth is not the bottleneck for well-optimized inference.
**Latency** is.

For INT8 quantized BERT inference:
- Typical DRAM bandwidth needed: ~20-30 GB/s (not 400 GB/s peak)
- Bottleneck: L3 miss latency (~32-40 cycles per miss)
- Optimization: Reduce L3 misses (increase reuse), not increase bandwidth

**Example**: A naïve implementation might achieve 50 GB/s DRAM bandwidth but 40% L3
miss rate. An optimized version might use only 15 GB/s but 2% L3 miss rate and run
6x faster due to latency reduction.

### Myth #2: NUMA is Only for Servers, Not Inference

**Reality**: NUMA becomes critical when inference batch size approaches core count.

60-core Sapphire Rapids with 16-token batches:
- Without NUMA awareness: threads bouncing across tiles → L3 misses
- With NUMA awareness (SNC=4): each tile handles 4 tokens independently

Example timeline:
```
Token 0: Core 0 (Tile 0) starts → must load KV cache from Tile 0 DRAM
Token 1: Core 12 (Tile 1) starts → must load KV cache from Tile 1 DRAM
(Good: no cross-tile coherency)

vs.

Token 0: Core 0 (Tile 0) starts → loads from Tile 2 DRAM (NUMA miss!)
Token 1: Core 12 (Tile 1) starts → loads from Tile 0 DRAM (NUMA miss!)
(Bad: excessive cross-tile traffic)
```

### Myth #3: Row-Buffer Hits are Unpredictable

**Reality**: Row-buffer hits are 90% predictable if you understand DRAM layout.

```
A typical DDR5 DRAM channel:
  16 banks × 32K rows × 8KB per row = 4GB per channel
  Row address = physical_address >> 15  (8KB rows)

Sequential access pattern:
  addr1 = 0x1000 (row 0x0)
  addr2 = 0x1008 (row 0x0) ← Row-buffer hit! (same 8KB row)
  addr3 = 0x2000 (row 0x0) ← Still row 0x0!
  addr4 = 0x3000 (row 0x0) ← Still row 0x0!
  ...
  addr9 = 0x9000 (row 0x0) ← Still row 0x0!

"Row buffer hit rate" for sequential pattern: 99%+

Random access pattern:
  addr_rand[0] = 0x12345678 (row 0x91A, bank 3)
  addr_rand[1] = 0x9ABCDEF0 (row 0x4D5E, bank 7) ← Different row!
  addr_rand[2] = 0x3C5A0B18 (row 0x1E2D, bank 5) ← Different row!

  Row buffer hit rate for random pattern: 5-15%
```

**For inference data layout**: Column-major matrices are disastrous for row-buffer
hits. Row-major or strided layouts (stride = cache line width) are essential.

### Myth #4: Persistent Memory / CXL is Equivalent to DRAM

**Reality**: Persistent memory and CXL have different characteristics:

```
DRAM latency:          110 cycles (DRAM hit)
Persistent memory lat: 400+ cycles (1-2 µs, on-device controller)
CXL memory latency:    ~300-500 cycles (depends on CXL device)

For inference:
  - DRAM: 50-100 µs per forward pass
  - Persistent memory: 200-400 µs per forward pass
  - CXL: 100-300 µs per forward pass (highly variable)

Use cases:
  - DRAM: hot data, frequently accessed models
  - Persistent memory: archived models, checkpoints (not inference)
  - CXL: extended capacity for large batch sizes (1000s of tokens)
```

SPR supports **CXL 2.0** (some configurations), but latency overhead makes it
unsuitable for tight SLA inference. It's valuable for batch offline inference
or training data staging.

---

## 6. BENCHMARK / MEASUREMENT

### Measuring L3 Miss Rate per Tile

```bash
# Use perf with NUMA-aware event counting
perf stat -e cache-references,cache-misses \
          -C 0-11 ./inference_server

# Count for cores 0-11 (Tile 0) separately
# Output:
# cache-references:  50M  (references to L3)
# cache-misses:      4M   (misses requiring DRAM)
# Miss rate: 8%

# Repeat for cores 12-23 (Tile 1), etc.
# Expected: 2-6% miss rate with good inference code
```

### Measuring DRAM Bandwidth

```bash
# Intel MLC (Memory Latency Checker) is gold standard
/opt/intel/mlc/mlc --bandwidth_matrix -w5

# Output matrix shows bandwidth between sockets/NUMA nodes:
#
#            Reads   Writes  RW Mixed
# Socket0:   350 GB/s 300GB/s 200 GB/s
# Socket1:   350 GB/s 300GB/s 200 GB/s
# 0←→1:      80 GB/s  80 GB/s 60 GB/s

# Interpretation:
# - Within-socket: 350 GB/s is good (close to 400 GB/s peak)
# - Cross-socket: 80 GB/s is expensive (use NUMA affinity to avoid)
```

### Measuring Row-Buffer Hit Rate

```bash
# Use Intel VTune with DRAM RAS events
vtune -c uarch-memory-bandwidth -- ./inference_server

# Look for "DRAM Read Latency" distribution:
# - Row-buffer hit: < 70 cycles
# - Row-buffer miss: > 100 cycles

# If > 50% of DRAM reads > 100 cycles:
#   → Data layout is poor (random access pattern)
#   → Action: reorganize data into sequential/strided pattern
```

### Measuring LLC Slice Balance

```bash
# Check if L3 data is evenly distributed across 4 tiles
# Use Intel PCM with CHA (Caching Agent) events

sudo /opt/pcm/pcm -e "UNC_CHA_TOR_OCCUPANCY_CHAx:u" -- sleep 10

# Output per CHA (each tile has one CHA):
# CHA 0: 45M events (Tile 0)
# CHA 1: 42M events (Tile 1)
# CHA 2: 48M events (Tile 2)
# CHA 3: 44M events (Tile 3)

# Good: balanced (±10% spread)
# Bad: one CHA >> others (data hashing imbalance)
```

---

## 7. ML SYSTEMS RELEVANCE

### KV-Cache Management on SPR

For LLM inference, the **Key-Value cache** is the largest consumer of memory:

```
LLaMA-70B with sequence length 2048 tokens, batch=16:

KV cache memory:
  Per-token: 70B × 2 (K,V) × hidden_dim × layers / batch_bits
  = 70B × 2 × 4096 × 80 / 8 (INT8 per token)
  = 70B × 2 × 4096 × 80 bytes
  ≈ 45 GB for single sequence

Distributed across 120 cores (2 sockets):
  NUMA node 0 (Tile 0): 11 GB
  NUMA node 1 (Tile 1): 11 GB
  NUMA node 2 (Tile 2): 11 GB
  NUMA node 3 (Tile 3): 12 GB
  (Remote socket): same split, but with QPI latency

With SNC=4: OS scheduler keeps attention reads to local NUMA node
  → ~60 cycles DRAM latency
  → ~95% of DRAM accesses are local

Without SNC: random tile scheduling
  → ~90 cycles DRAM latency (cross-NUMA)
  → Only 60% local, 40% remote
  → 30% latency degradation
```

### Persistent Memory for Model Weights

For **extremely large models** (> 200GB), Sapphire Rapids can address persistent
memory via PMEM:

```c
// Load model weights from persistent memory
// Requires kernel support for DAX (Direct Access) mode

#include <libpmem.h>

int model_size_bytes = 250e9;  // 250 GB model
void *pmem_addr = pmem_map_file("/mnt/pmem0/model.bin",
                                 model_size_bytes,
                                 PMEM_FILE_CREATE,
                                 0644, NULL, NULL);

float *weights = (float *)pmem_addr;

// Access weights (via DRAM cache if available)
float w = weights[i];  // First access: 400+ cycles
                       // Subsequent accesses: cached in DRAM

// Trade-off: Persistent memory latency is 3-4x DRAM,
// but allows unlimited capacity beyond DRAM slots
```

For **batch inference** where throughput > latency matters, persistent memory
is viable. For **real-time serving**, avoid it.

---

## 8. PhD QUALIFIER QUESTIONS

**Question 1**: Sapphire Rapids has 8 DDR5 channels, each supporting 5600 MT/s.
Calculate the peak bandwidth. Then explain why actual sustained bandwidth is only
85-90% of peak. What is the role of row-buffer hits vs. row-buffer misses in this
efficiency gap?

**Question 2**: Explain the MESI cache coherency protocol on SPR. What is a "cache
line bounce" and why does it occur when multiple tiles access the same cache line?
Give a concrete example from inference code (e.g., atomic counter) and propose a
refactoring to eliminate it.

**Question 3**: SPR supports SNC=4 (Sub-NUMA Clustering with 4 nodes per socket).
Explain how this helps inference serving compared to SNC=1 (monolithic NUMA node).
What OS scheduler behavior changes enable the performance improvement? Would you
recommend SNC=4 for all workloads?

**Question 4**: The L3 cache is distributed by physical address (bits [11:6] determine
slice). This can cause imbalance if allocations hash poorly to the same slice.
Propose a memory allocation strategy that ensures even distribution of inference
working sets across all 4 tiles' L3 slices. What alignment constraints are necessary?

**Question 5**: For an LLM inference serving 512 concurrent requests with KV-cache
in persistent memory, estimate the latency per token generation. Compare this to
DDR5-only deployment. What is the capacity trade-off, and when is persistent memory
justified?

---

## 9. READING LIST

1. **Intel Xeon Scalable Processor Architecture Specification (XSPS)**, Section 4
   "Memory Subsystem," covers IMC architecture, DDR5 support, cache organization.

2. **JEDEC DDR5 Standard (JESD79-6C)**, essential for understanding DDR5 timing
   parameters (tRCD, tRP, tRAS) and row-buffer behavior.

3. **Intel Resource Director Technology (RDT) Whitepaper**, available at Intel.com.
   Documents CAT (Cache Allocation Technology), MBM (Memory Bandwidth Monitoring),
   and MBA (Memory Bandwidth Allocation).

4. **Sapphire Rapids Memory Subsystem Deep Dive**, Anandtech (2023), empirical
   measurements of L3 miss latency, DRAM bandwidth, NUMA penalties.

5. **"The MESI Cache Coherence Protocol"** (Culler, Singh, Gupta), from multiprocessor
   textbooks. SPR implements MESI variant; understanding correctness is critical for
   correctness proofs in concurrent inference serving.

6. **Intel 64 and IA-32 Architectures Software Developer Manual Volume 3A**,
   Chapter 8 "Memory Protection" and 9 "APIC and Interrupts." Covers page tables,
   virtual memory, TLB behavior (impacts inference serving with large models).

7. **Intel Memory Latency Checker (MLC) User Guide**, available from Intel. Tool
   for measuring bandwidth matrices, latency vs. stride, row-buffer effects.

8. **"Optimizing Memory Bandwidth in Multicore Processors"** (McKee, Reilly),
   ISPASS 2011. Foundational for understanding DRAM row-buffer interactions with
   cache hierarchies.

---

**Module 19 Complete**: 1089 lines. Establishes memory subsystem mastery required
for final optimization module.
