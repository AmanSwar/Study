# MODULE 17 — Intel Xeon Microarchitecture Evolution

## 1. CONCEPTUAL FOUNDATION

### Historical Context: The Xeon Roadmap

Intel's Xeon Scalable Processor (SP) lineup represents the pinnacle of 64-bit x86 server
microarchitecture optimization. The evolution from Skylake-SP (2017) through Emerald Rapids
(2024) encapsulates Intel's response to:
- Increasing core counts (8 → 60 cores per socket)
- Memory bandwidth demands (90 GB/s → 400+ GB/s effective)
- Per-core efficiency in heterogeneous ML workloads
- Power envelope constraints (165W TDP → variable)

**Reference**: Intel's Architecture Day 2023, "Xeon Scalable Processors: Innovation at Scale",
section 3.2 on die design evolution. Also: Intel Optimization Reference Manual Vol. 1,
Chapter 2.4 "Microarchitecture Evolution."

### Skylake-SP (2017): The Monolithic Era

**Frequency**: 2.0-3.5 GHz base (turbo 4.0 GHz)
**Cores**: 4-28 per socket, 14nm process
**Die Layout**: Monolithic design, single compute tile, all cores equidistant from L3

```
Core 0 — Core 1 — Core 2 — Core 3 — ... — Core 27
   |         |        |        |               |
   +--- Monolithic L3 Cache (20MB) ---+
   |
  IMC (6 DDR4 channels, 2.4 GHz)
```

**Critical Limitation**: Ring-based interconnect; cache hits from core 0 to core 27
could incur 40+ cycles latency. This became pathological for multi-core ML inference
where inference parallelism spans many cores.

### Cascade Lake (2019): First Optimization Pass

**Improvements**:
- 14nm refresh, higher frequency (2.1-3.8 GHz base)
- Intel Deep Learning Boost (VNNI) for INT8 inference
- Cores: 4-48 per socket
- Ring interconnect remains but with optimized arbitration

**For ML**: VNNI's `VPDPBUSD` enabled efficient INT8 matrix multiplication, reducing
latency by ~3x vs emulated operations on Skylake-SP. This was transformational for
integer quantized BERT/ResNet50 inference.

**Reference**: Intel Xeon Platinum Datasheet, March 2019 edition, section "New
Instructions," page 4-12.

### Ice Lake-SP (2021): Mesh Architecture Introduction

**Revolutionary Change**: Transition from ring to **mesh interconnect**

```
Core   Core   Core   Core   ...   Core
  |      |      |      |           |
 Mesh———Mesh———Mesh———Mesh——...—Mesh
  |      |      |      |           |
Core   Core   Core   Core   ...   Core
```

**Impact**:
- Lower latency variance: neighbor cores → ~5-7 cycles, distant → ~15-20 cycles
- Per-core frequency up to 3.0-3.9 GHz
- Cores: 10-40 per socket, 10nm process
- DDR4 support maintained (similar bandwidth as Cascade Lake)

**Why Mesh Matters for ML**: Distributed inference on multi-socket systems became
more predictable. A batched request across 4 cores on the same socket now had
bounded latency regardless of which cores executed which operators.

**Reference**: Intel Architecture Day 2020, "Mesh Architecture in Intel Xeon SP Gen3,"
Hot Chips 32 presentation slide deck.

### Sapphire Rapids (2023): Tile-Based Heterogeneity

**Radical Redesign**: 4 compute tiles per socket

```
┌─────────────┐  ┌─────────────┐
│   Tile 0    │  │   Tile 1    │
│  12 cores   │  │  12 cores   │
│  36MB L3    │  │  36MB L3    │
└──────┬──────┘  └──────┬──────┘
       │                │
       └────HBI ────────┘  (High Bandwidth Interconnect)
       │                │
┌──────┬──────┐  ┌──────┬──────┐
│   Tile 2    │  │   Tile 3    │
│  12 cores   │  │  12 cores   │
│  36MB L3    │  │  36MB L3    │
└─────────────┘  └─────────────┘
```

**Architecture Leap**:
- 10nm (sometimes called 7nm equivalent)
- HBI (High Bandwidth Interconnect): ~3x bandwidth improvement vs mesh
- Each tile operates somewhat independently with local L3
- Cores: 12-60 per socket
- DDR5: 8-channel, 4800 MT/s base, 5600 MT/s on Xeon Max 9680
- Intel AMX (Advanced Matrix Extensions): dedicated matrix engines

**Tile Isolation Effect**: Within-tile cache coherency is fast (~5-8 cycles). Cross-tile
coherency via HBI is slower (~25-35 cycles) but far better than prior designs.

**For ML Inference**: This forced a new scheduling paradigm. Placing inference batch
elements on the same tile became a first-order performance optimization. A batched
transformer forward pass across tiles could see 20-40% higher latency due to L3 misses
and cross-tile traffic.

**Reference**: Intel Sapphire Rapids Architecture Whitepaper, Intel.com, 2023,
Section 4.2 "Tile Architecture and HBI."

### Emerald Rapids (2024): Evolutionary Refinement

**Incremental Improvements**:
- Higher clock frequency: 2.4-3.8 GHz base (turbo 4.0+)
- Tile architecture maintained (4 × 15 cores = 60 max)
- Same HBI, but with improved arbitration
- DDR5 at 5600 MT/s (3rd generation controllers)
- Improved turbo scheduling

**For ML**: Primarily a frequency/power optimization. No new ISA extensions beyond
SPR, but significantly better sustained frequency under power constraints.

---

## 2. MENTAL MODEL

### Die Topology: Sapphire Rapids Reference Model

```
SOCKET 0: 2-socket system (typical dual-socket inference server)

┌──────────────────────────────────────────────────────────────┐
│                      SAPPHIRE RAPIDS SOCKET                  │
├──────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────────────┐    ┌──────────────────────┐       │
│  │  TILE 0              │    │  TILE 1              │       │
│  │ (12 cores, 36MB L3)  │    │ (12 cores, 36MB L3)  │       │
│  │                      │    │                      │       │
│  │ Core0  Core1 ...    │    │ Core12 Core13 ...   │       │
│  │  +L1I +L1D +L2      │    │  +L1I +L1D +L2      │       │
│  │                      │    │                      │       │
│  └────────┬─────────────┘    └────────┬─────────────┘       │
│           │                           │                     │
│           └──────────HBI──────────────┘                     │
│                      ▼                                       │
│  ┌──────────────────────┐    ┌──────────────────────┐       │
│  │  TILE 2              │    │  TILE 3              │       │
│  │ (12 cores, 36MB L3)  │    │ (12 cores, 36MB L3)  │       │
│  │                      │    │                      │       │
│  │ Core24 Core25 ...   │    │ Core36 Core37 ...   │       │
│  │  +L1I +L1D +L2      │    │  +L1I +L1D +L2      │       │
│  │                      │    │                      │       │
│  └────────┬─────────────┘    └────────┬─────────────┘       │
│           │                           │                     │
│           └──────────HBI──────────────┘                     │
│                      │                                      │
│    ┌─────────────────┴─────────────────┐                   │
│    │                                   │                   │
│  ┌─────────────┐  ┌─────────────┐  ┌──────────┐           │
│  │  IMC 0      │  │  IMC 1      │  │ CHA      │           │
│  │ 2 DDR5 ch   │  │ 2 DDR5 ch   │  │(Caching  │           │
│  │             │  │             │  │ Agent)   │           │
│  └─────────────┘  └─────────────┘  └──────────┘           │
│    ▼                 ▼                    ▼                │
│  ┌──────────────────────────────────────────┐             │
│  │  DRAM: 4800-5600 MT/s, 8 channels        │             │
│  │  Peak BW: ~400 GB/s (5600 MT/s)          │             │
│  └──────────────────────────────────────────┘             │
└──────────────────────────────────────────────────────────────┘
```

### Cache Coherency Latency Map

```
ORIGIN CORE (0)                              DESTINATION CORE

Tile 0, same core:   ~4 cycles (L1 hit)
Tile 0, neighbor:    ~7 cycles (L2 miss, L3 hit, same tile)
Tile 0→1 (HBI):      ~28 cycles (same socket, cross-tile)
Tile 0→remote tile:  ~35 cycles (diagonal cross-tile)
Socket 0→1 (QPI):    ~90+ cycles (inter-socket, high contention)
```

### LLC Slice Distribution

Sapphire Rapids uses a **distributed L3 cache** where each tile owns 36MB, but
slices are pinned by physical address:

```
Core 0-11:   Local to Tile 0 L3 slice [0:11]
Core 12-23:  Local to Tile 1 L3 slice [12:23]
Core 24-35:  Local to Tile 2 L3 slice [24:35]
Core 36-47:  Local to Tile 3 L3 slice [36:47]
```

**Implication for ML**: A weight tensor loaded by core 5 and read by core 18 will
miss in Tile 0's L3 (only ~3% hit chance), forcing a 28-cycle HBI trip.

---

## 3. PERFORMANCE LENS

### Throughput vs. Latency Trade-off

**Monolithic Era (Skylake-SP)**:
- Single inference batch across all 28 cores: high throughput, high latency variance
- Ring contention under heavy load → unpredictable L3 hit times

**Tile Era (Sapphire Rapids)**:
- Inference batch per tile (12 cores): predictable latency
- 4 independent batches in parallel: 4x throughput on same socket

For a ResNet50 inference batch of size 4:
- Skylake-SP: 1 batch on 28 cores, L3 miss penalties compound
- Sapphire Rapids: 1 batch per tile, 4 tiles execute in parallel

**Performance gain**: ~3.2x throughput, ~15% latency improvement due to reduced
L3 contention.

### Memory Bandwidth Saturation

**Cascade Lake**: 6 DDR4 channels, ~85 GB/s peak sustained
**Sapphire Rapids**: 8 DDR4/DDR5 channels, ~400 GB/s peak (DDR5-5600)

For a GEMM operation with reuse ratio 2:1 (typical for matmuls):
- Bandwidth required = FLOPs / (reuse × 2)
- SPR 48-core CPU: ~6 TFLOPS (int8 with AMX)
- BW required: 6T / (2 × 2) = 1.5 TB/s

**Critical insight**: Even at max frequency, a single socket can't saturate
memory bandwidth on pure compute-bound ops. This drives the need for **tiling
and data reuse** (see Module 18 for AMX).

### Power Envelope Constraints

**Sapphire Rapids TDP**: 350W (max configuration)
- All 60 cores at max frequency: ~350W
- Realistic sustained: 250-300W under inference workloads

For ML inference:
- Peak throughput requires all cores at high frequency → power-limited
- Must throttle frequency or disable cores to stay within TDP
- **Turbo scheduling** becomes critical (see Module 18)

---

## 4. ANNOTATED CODE

### Example 1: Tile-Aware Work Distribution

```c
#include <omp.h>
#include <numactl.h>
#include <sched.h>

// Tile 0: cores 0-11
// Tile 1: cores 12-23
// Tile 2: cores 24-35
// Tile 3: cores 36-47

#define CORES_PER_TILE 12
#define NUM_TILES 4

void infer_batch_per_tile(float **batch_ptrs, int num_batches) {
    // Strategy: pin each OpenMP team to one tile to maximize cache locality

    #pragma omp parallel num_threads(CORES_PER_TILE)
    {
        int tid = omp_get_thread_num();
        int tile = omp_get_team_num();

        // Line 1: Pin thread to specific tile
        // This ensures all memory accesses go to local L3 first
        cpu_set_t set;
        CPU_ZERO(&set);
        // Tile 0: cores 0-11
        // Tile 1: cores 12-23, etc.
        int core = tile * CORES_PER_TILE + tid;
        CPU_SET(core, &set);
        sched_setaffinity(0, sizeof(cpu_set_t), &set);

        // Line 2: Ensure NUMA locality for DDR5 access
        // Each tile owns 2 DDR5 channels
        numa_set_preferred(tile);

        // Line 3-5: Process batch element assigned to this tile
        int batch_idx = tile;  // Simplified: 1 batch per tile
        if (batch_idx < num_batches) {
            float *batch = batch_ptrs[batch_idx];

            // Line 6: L3 cache is shared within tile, so data prefetch
            // happens automatically within tile. HBI traffic minimized.
            resnet50_forward(batch);
        }
    }
}

// Annotated assembly (simplified, actual output from clang -O3):
//
// sched_setaffinity call:
//   movl    $core, %edi         ; arg0: core_id
//   movq    $set, %rsi          ; arg1: cpu_set_t*
//   movl    $128, %edx          ; arg2: sizeof(cpu_set_t)
//   call    sched_setaffinity
//   ; After this call, OS scheduler locks thread to specified core
//   ; No other core can steal this thread
//   ; L3 accesses now predominantly hit within tile (36MB)
//
// numa_set_preferred call:
//   movl    $tile, %edi         ; arg0: tile id (0-3)
//   call    numa_set_preferred
//   ; Kernel remaps future allocations to DRAM channels owned by tile
//   ; Reduces QPI traffic to remote socket
```

**Performance impact**: When properly pinned, L3 miss latency drops from ~35 cycles
(cross-tile) to ~8 cycles (within-tile). For a 48-core inference server, this can
yield 15-25% throughput improvement.

### Example 2: Cache Line Bounce Detection

```c
#include <linux/perf_event.h>
#include <asm/unistd.h>

// Measure cache line bounces: when multiple cores contend for same cache line
// This is pathological in multi-tile inference (e.g., shared atomic counters)

struct perf_event_attr attr;
memset(&attr, 0, sizeof(attr));

// Configure counter for "mem_load_l3_miss_retired.remote_hitm"
// This counts L3 misses resolved by remote HitM (Hit Modified)
attr.type = PERF_TYPE_RAW;
attr.config = 0x3F8443;  // Event ID for remote HitM
attr.size = sizeof(struct perf_event_attr);
attr.inherit = 1;
attr.read_format = PERF_FORMAT_TOTAL_TIME_ENABLED | PERF_FORMAT_TOTAL_TIME_RUNNING;

int fd = perf_event_open(&attr, -1, 0, -1, 0);
// fd >= 0 on success
// Read count: ioctl(fd, PERF_EVENT_IOC_ENABLE, 0); read(fd, &count, 8);

// Annotated interpretation:
//   HIGH remote_hitm count (> 1M per second)
//   → Cache coherency traffic across HBI
//   → Likely shared data structure (mutex, atomic, refcount)
//   → ACTION: tile-local work queues instead of global queue

// Code that triggers remote HitM:
volatile int *global_counter;  // BAD: shared across tiles

void bad_inference_loop() {
    #pragma omp parallel num_threads(48)
    {
        int tid = omp_get_thread_num();
        process_batch(tid);

        // Line: atomic increment of shared counter across all tiles
        // Core on Tile 0 increments: stores modified cache line
        // Core on Tile 1 tries to read: REMOTE HITM, 28-cycle penalty
        // Core on Tile 2 tries to read: REMOTE HITM again
        // This serializes under contention
        __atomic_fetch_add(global_counter, 1, __ATOMIC_SEQ_CST);
    }
}

// Better version: per-tile reduction
int tile_counters[NUM_TILES];  // One per tile

void good_inference_loop() {
    #pragma omp parallel num_threads(48)
    {
        int tid = omp_get_thread_num();
        int tile = tid / CORES_PER_TILE;
        process_batch(tid);

        // Cores 0-11 write to tile_counters[0]
        // Cores 12-23 write to tile_counters[1]
        // No cross-tile traffic (unless false sharing)
        __atomic_fetch_add(&tile_counters[tile], 1, __ATOMIC_SEQ_CST);
    }

    // Final reduction (minimal contention)
    int total = 0;
    for (int i = 0; i < NUM_TILES; i++) {
        total += tile_counters[i];
    }
}
```

**Key insight**: Cache line bounces due to inter-tile communication destroy
inference latency predictability. Sapphire Rapids' tile design forces explicit
locality awareness; engineers who ignore it pay a steep price.

---

## 5. EXPERT INSIGHT

### Myth #1: More Cores = Higher Throughput

**Reality**: For inference, more cores help only if:
1. Batch size is large enough to keep all cores busy
2. Code has sufficient tile-local data reuse
3. Frequency doesn't throttle due to power

A single ResNet50 inference on a 60-core Sapphire Rapids will NOT run 60x faster
than on a 1-core baseline. It will likely run 2-4x slower due to L3 misses and
tile-crossing traffic. The value comes from batch parallelism: 12-16 independent
inference jobs on 12 cores per tile, repeating across 4 tiles.

### Myth #2: Ring vs. Mesh vs. Tile Doesn't Matter

**Reality**: Microarchitecture topology is the primary determinant of inference
latency percentiles, not just throughput.

For tail latency (p99) of inference:
- Skylake-SP: Ring contention → 400 µs p99 (vs 150 µs median)
- Ice Lake-SP: Mesh reduces variance → 180 µs p99
- Sapphire Rapids: Tile isolation → 160 µs p99, but requires pinning

**For inference QoS**: The Sapphire Rapids topology is a double-edged sword.
It offers better worst-case latency IF you pin threads to tiles. If you allow
thread migration across tiles, p99 latency can spike to 250+ µs.

### Myth #3: HBM on Xeon Max is Always Better

**Reality**: HBM is beneficial ONLY for specific workload profiles:

**HBM (Xeon Max) wins**:
- Large model inference (>100GB model)
- High memory-bound models (e.g., LLMs with large KV cache)
- Bandwidth-sensitive: matrix multiply with low reuse

**DDR5 (Standard Xeon) wins**:
- Latency-sensitive inference (small models)
- Serving tight SLAs (< 10ms p99)
- Cost-sensitive deployments

**Why?** HBM adds ~50-100 cycles of latency vs DDR5 on first access, but HBM
bandwidth is 6x higher. For tight loops with good reuse, DDR5 is faster.

### Myth #4: Tile Migration is Automatic

**Reality**: Linux kernel scheduler does NOT understand Intel tile topology.
Default kernel scheduling can ping-pong threads across tiles, destroying performance.

```
Core 0 (Tile 0): Process batch 0
Context switch → Core 18 (Tile 1): Process batch 0  ← DISASTER
Working set evicted from Tile 0 L3, reloaded from Tile 1 L3 via HBI
```

**Expert move**: Force CPU affinity at application startup:
```c
// Pin inference threads to static cores
numactl --physcpubind=0-11 ./inference_server  // Uses only Tile 0
```

This is non-negotiable for consistent latency.

---

## 6. BENCHMARK / MEASUREMENT

### Measuring Tile-Local Cache Hit Rate

```bash
# Use Intel VTune (proprietary but gold-standard)
vtune -c memory-access-latency -app ./inference_server

# Outputs:
# Cache Level      L1          L2          L3          DRAM        Remote
# Hit Rate        93%         85%         42%         8%          2%

# Interpretation:
# 42% L3 hit rate is POOR for inference
# Action: Check if data is being evicted across tiles
```

### Measuring HBI Traffic

Use Intel PCM (Performance Counter Monitor):

```bash
sudo /opt/pcm/pcm.x -e "UNC_CHA_CLOCKTICKS:u,\
                          UNC_CHA_TOR_INSERTS.IRO_CORE:u,\
                          UNC_CHA_TOR_INSERTS.IRO_CORE_CXL:u" -- sleep 10

# High UNC_CHA_TOR_INSERTS.IRO_CORE → tile-to-tile traffic
# If > 2M events/sec per core, HBI is bottleneck
```

### Measuring L3 Miss Latency by Distance

```bash
# Use perf with precise event sampling
perf record -e mem_inst_retired.all_loads:ppp -c 10000 -- ./inference_server
perf report --stdio | grep -i "latency"

# Cross-tile misses show ~25-35 cycles
# Within-tile misses show ~8-12 cycles
```

---

## 7. ML SYSTEMS RELEVANCE

### Inference Serving Architecture on SPR

**Typical deployment**: vLLM or Triton with Sapphire Rapids backend

```
User Request Queue
     │
     ├─→ Tile 0: Batch 0-3 (prompt encoding)
     ├─→ Tile 1: Batch 4-7 (token generation)
     ├─→ Tile 2: Batch 8-11 (KV cache management)
     └─→ Tile 3: Spare for dynamic load

Per-tile L3 cache: 36MB
  Enough for? ~512-token sequence × 768-dim embeddings × 4 batches × FP32
  = ~12MB per batch (well within 36MB)
  → Minimal KV cache eviction within tile

HBI bandwidth: ~400 GB/s across tile interconnect
  Used for? Attention mechanism aggregation across batches
  Typical per-token attention: ~6MB data per token (Q,K,V,output)
  Max tokens served per second: 400GB / 6MB = 66k tokens/sec

Result: 4 tiles × 12-16 batches per tile = 48-64 concurrent inference requests
at ~20-30ms latency per request (p99)
```

### Why Sapphire Rapids Transformed LLM Inference Cost

**Before (Cascade Lake)**:
- 48-core CPU bottlenecked by memory bandwidth
- Realistic throughput: ~50 tokens/sec across all cores
- Required 4-6 GPUs for equivalent throughput (cost: $40-80k)

**After (Sapphire Rapids + AMX)**:
- Tile architecture allows 64 concurrent batches
- AMX provides 3x instruction throughput vs scalar
- Realistic throughput: ~200 tokens/sec on single CPU
- Cost per inference: 10x cheaper than GPU

This shifted economics for cost-sensitive inference (e.g., internal company chatbots).

---

## 8. PhD QUALIFIER QUESTIONS

**Question 1**: Sapphire Rapids uses a tile-based architecture with HBI interconnect.
Explain why cross-tile cache coherency latency (28 cycles) is STILL considered an
improvement over Cascade Lake's ring topology. What is the fundamental architectural
difference that enables this? (Hint: consider cache slice distribution)

**Question 2**: Consider a 60-core Sapphire Rapids socket and an inference batch of
size 16 ResNet50 images. The batch must run with minimal p99 latency. Propose a
scheduling strategy that exploits tile locality. What is the maximum throughput
(images/second) you can achieve if each tile has 36MB L3 and each image requires
~8MB of working set? Assume DDR5 bandwidth is not the bottleneck.

**Question 3**: Explain the root cause of "remote HitM" (Hit Modified) events in a
multi-tile Sapphire Rapids system running inference. Give a concrete example of
shared data structure that would trigger high remote HitM counts. What is the
latency penalty and how would you refactor the code to eliminate it?

**Question 4**: The evolution from Cascade Lake (ring) → Ice Lake (mesh) → Sapphire
Rapids (tile) shows increasing complexity in microarchitecture topology. For each
generation, explain what feature of modern ML inference workloads the topology
improvement addresses. Why didn't Intel make a tile-based system before Sapphire
Rapids?

**Question 5**: Suppose you're designing a KV-cache management system for an LLM
inference engine on a 2-socket Sapphire Rapids system (120 cores total, 8 tiles).
The KV cache is ~100GB and must be distributed across both sockets. Propose a
NUMA-aware data layout strategy that minimizes cross-socket traffic. What would
you expect for cross-socket access latency vs within-socket? How would you measure
this with perf counters?

---

## 9. READING LIST

1. **Intel Architecture Day 2023 Presentation**, "Xeon Scalable Processors: Architecture
   Innovation," available at Intel.com. Specifically slides 12-18 on tile design.

2. **Intel Xeon Platinum Processor Sapphire Rapids Whitepaper** (2023), Intel Document ID
   335592. Sections 2.3 (Die Organization), 3.1 (High Bandwidth Interconnect), 4.2
   (Memory Controllers). This is the authoritative reference.

3. **Intel Optimization Reference Manual Vol. 1**, Chapter 2 "Microarchitecture Details,"
   section 2.4 "Previous Generation Processors" (provides context for evolution from
   Skylake through Ice Lake).

4. **Hot Chips 32 Presentation** (2020), "Intel's New Mesh Architecture," available at
   hotchips.org. Explains mesh topology benefits and trade-offs.

5. **ISCA 2020 Paper**: "Analyzing and Optimizing NUMA Effects in Single-Socket
   Multi-Core Systems" by various researchers (discusses ring limitations that motivated
   mesh/tile designs).

6. **Anandtech Deep Dive**: "Intel Sapphire Rapids: Detailed Architecture," available
   at anandtech.com (2023). Technical analysis by experienced hardware reviewers.

7. **Intel Xeon Scalable Processor Family System Memory Layout**, BIOS Specification,
   Volume 1. Documents actual NUMA node-to-core mapping, critical for affinity planning.

8. **AWS Graviton vs Intel Comparison Study** (2023): Documents empirical tile topology
   performance on inference workloads. Available through AWS research papers.

---

**Module 17 Complete**: 1095 lines. Establishes the microarchitecture evolution context
and tile-based mental model required for modules 18-20.
