# MODULE 15 — Profiling Tools Mastery

## 1. CONCEPTUAL FOUNDATION

The modern performance engineer operates with a sophisticated set of observability tools. Unlike theoretical frameworks (Module 14), these tools provide **ground truth** about actual hardware behavior. The challenge is not access to data—it is interpretation, artifact avoidance, and extracting signal from noise.

### Linux perf: Architecture and Event Categories

**perf** is the canonical performance counter subsystem on Linux, built atop the Linux Performance Events (LPE) kernel interface. It supports:

1. **Hardware Performance Counter Events**: Direct CPU events (cycles, instructions, cache misses)
2. **Software Events**: Kernel-maintained counters (page faults, context switches, task migrations)
3. **Trace Events**: Kernel instrumentation points (sched events, syscalls)
4. **PEBS (Precise Event Based Sampling)**: Instruction-accurate sampling for skid mitigation
5. **BTS (Branch Trace Store)**: Intel-specific, logs every taken branch to ring buffer

The fundamental challenge: **event multiplexing**. Modern CPUs have only 4-6 hardware counters per core (Sapphire Rapids: 4 fixed + 8 programmable). To measure > 8 events simultaneously, perf must:
- Run multiple passes, remultiplexing counter banks
- Calculate scaled values: `count × (total_events / measured_events)`
- Introduce systematic error (~10% on large event sets)

**Implication for experts**: Measure in small batches (< 6 events per run) to avoid multiplexing error. For comprehensive analysis, run 3-5 passes with disjoint event sets, then combine results offline.

Reference: Aczel et al., "Performance Characterization of the Core i7 9700K", MEMSYS 2020.

### Hardware Performance Counters: Semantic Precision

Not all event counts are created equal. Consider three cache miss metrics:

| Event | Semantics | Latency Impact | Pitfall |
|-------|-----------|----------------|---------|
| `cache-misses` | L1D cache misses | 12 cycles (L2 hit) to 40 cycles (L3 hit) | Doesn't distinguish L2/L3 |
| `LLC-load-misses` | L3 cache misses + coherency stalls | 300 cycles (DRAM local) to 600 (NUMA) | Undercount if load hits L3 |
| `mem_load_l3_miss_retired.local_dram` | Retired loads that missed L3, fetched from local DRAM | 300 ns | **Most precise for DRAM pressure** |

**Junior interpretation**: "I see 1000 cache-misses, so I'm memory-bound."
**Expert interpretation**: "I see 1000 LLC-load-misses, but 800 are L2→L3 hits (12 cycle penalty), not DRAM (300 cycle)—actually, memory isn't the bottleneck here."

The Intel *Performance Analysis and Tuning on Modern CPUs* (Dendibakh, Ch. 5) provides the precise event semantics for each microarchitecture. Use it as a reference, not memory.

### Intel VTune: Higher-Level Abstraction

VTune abstracts away raw counter interpretation and provides:
1. **Hotspot Analysis**: Identifies functions consuming most cycles (top-down call tree)
2. **Microarchitecture Exploration**: Automatic TMA calculation, bottleneck highlighting
3. **Memory Access Analysis**: Shows memory bandwidth, DRAM vs L3 pressure, NUMA traffic
4. **Threading Analysis**: Lock contention, synchronization overhead, thread scaling
5. **I/O Analysis**: Memory-mapped I/O, NIC interrupts, disk stalls

VTune's advantage: it *correlates* events with source code. When you see "42% DRAM_Bound", VTune shows you *which loop* causes it.

**Cost**: VTune is commercial, requires Intel CPU (older AMD support exists but less robust). Linux perf is free but requires manual interpretation.

Reference: Intel VTune Profiler User Guide, https://www.intel.com/content/www/en/us/en/docs/vtune/user-guide/current/overview.html

### Flamegraphs: Off-CPU Insight

**perf record → perf script → stackcollapse → flamegraph.pl** is the standard workflow for generating call-stack visualizations. Flamegraphs show:
- **On-CPU flamegraph**: which functions consume CPU time (samples when CPU is executing)
- **Off-CPU flamegraph**: which functions are blocked (samples on sleep, futex wait, I/O)
- **Memory flamegraph**: which functions allocate memory

The **Brendan Gregg** versions are definitive (http://www.brendangregg.com/flamegraphs.html).

**Critical insight**: An off-CPU flamegraph of a multithreaded inference server reveals where threads spend time *not running*:
- Busy-waiting on network receive? → Lock contention, need lock-free queues
- Blocked in malloc? → Jemalloc memory fragmentation
- In futex syscall? → Thread synchronization bottleneck

### perf c2c: False Sharing and Cache Coherency

**perf c2c record** (Cache-to-Cache) captures which CPU cores are fighting over cache lines (false sharing). It works by:
1. Setting HW breakpoints on memory lines
2. Recording which CPUs load/store each line
3. Calculating "hitm" (core-to-core cache transfer latency)

Example: two threads on cores 0 and 1 both access `int x` in the same 64-byte cache line:

```
Core 0: increment x[0]   (write)
Core 1: increment x[1]   (write)
```

Cache line is 64 bytes (8 int32s). Both x[0] and x[1] live in same line. Core 0 writes it → line marked "Modified" on core 0. Core 1 tries to access x[1] → must fetch from core 0's L2 (cache coherency). This is **false sharing**: no true data dependency, but hardware cache coherency cost.

**Diagnosis**:
```bash
perf c2c record -g -- ./multithreaded_app
perf c2c report
# Output: which memory lines are causing hitm, which cores are fighting
```

**Optimization**: Pad data to avoid shared cache lines, or use thread-local copies.

Reference: Semeraro et al., "Performance Analysis of Cache Coherency Protocol on Multicore Processors", ISPASS 2019.

### LIKWID: Hardware Topology and Counter Framework

**LIKWID** (Like I Knew What I'm Doing) is a performance counter abstraction layer:
- **likwid-topology**: prints cache hierarchy, NUMA layout, CPU affinities
- **likwid-perfctr**: groups counter events intelligently (avoids multiplexing)
- **likwid-bench**: microbenchmark suite for memory bandwidth, FLOP rates

Example (Sapphire Rapids topology):
```bash
likwid-topology
# Output:
#   6 cores per socket, 12 threads per socket
#   L1D: 48 KB, 12-cycle latency, 4-way associativity
#   L2: 1.25 MB, 12-cycle latency, 20-way associativity
#   L3: 51.2 MB, 40-cycle latency, 19-way associativity
#   DDR5: 12.8 GB/s sustained bandwidth per core
```

LIKWID groups automatically handle event multiplexing, giving accurate results without scaling. **Expert move**: use LIKWID for complex multi-socket, multi-NUMA characterization; use perf for quick spot checks.

Reference: Treibig et al., "LIKWID: A Lightweight Performance-Oriented Tool Family", ISPASS 2010.

### valgrind cachegrind and callgrind: Simulation vs Hardware

**cachegrind** simulates a cache hierarchy and reports:
- L1I, L1D, LL (L2+L3 combined) hit/miss rates
- Instruction count breakdown

**Advantages**:
- Works on older CPUs without performance counter support
- Deterministic (no sampling error)
- Shows *why* a miss happens (capacity, associativity, coherency)

**Disadvantages**:
- 50-100× slower than native execution
- Single-threaded (ignores NUMA, cache coherency)
- Simplistic cache model (Sapphire Rapids L3 is far more complex)

**When to use**: Quick cache analysis on a slow/old system. For modern CPUs, prefer hardware counters.

Reference: Nethercote & Seward, "Valgrind: A Framework for Heavyweight Dynamic Binary Instrumentation", PLDI 2007.

---

## 2. MENTAL MODEL

```
┌──────────────────────────────────────────────────────────────────┐
│              PROFILING TOOL HIERARCHY & TRADEOFFS                 │
├──────────────────────────────────────────────────────────────────┤
│                                                                    │
│  Abstraction Level          Tool              Accuracy  Overhead  │
│  ───────────────────────────────────────────────────────────────  │
│  ┌─────────────────────┐                                          │
│  │ Source Code (perf)  │ ← attach symbols to CPU samples          │
│  │ Flamegraph          │   fine-grained, < 5% overhead           │
│  └─────────────────────┘                                          │
│           ▲                                                        │
│           │ (provides call stacks)                               │
│  ┌─────────────────────────────────────────┐                     │
│  │ perf stat              (statistical)     │ ← multiplexing     │
│  │ perf record + report   (sampling-based)  │   error ~10%      │
│  │ Intel VTune            (HW counters)     │   < 5% overhead   │
│  └─────────────────────────────────────────┘                     │
│           ▲                                                        │
│           │ (raw event counts)                                   │
│  ┌────────────────────────────────────────────┐                  │
│  │ Hardware Performance Counters              │ ← physical truth  │
│  │ • cycles, instructions, cache-misses      │   only 4-8       │
│  │ • LLC-loads, LLC-stores, branch-misses    │   counters       │
│  │ • PEBS events, BTS (branch trace)         │                  │
│  └────────────────────────────────────────────┘                  │
│           ▲                                                        │
│           │                                                        │
│  ┌────────────────────────────────────────────┐                  │
│  │ CPU Execution (microarchitecture)          │                  │
│  │ • port contention, speculative execution  │                  │
│  │ • L1/L2/L3 coherency, prefetch patterns   │                  │
│  │ • branch prediction, instruction window  │                  │
│  └────────────────────────────────────────────┘                  │
│                                                                    │
│  Measurement Strategy:                                            │
│  1. Quick check: perf stat (< 1 minute)                         │
│  2. Identify hotspot: perf record (5 min + flamegraph)          │
│  3. Deep dive: Intel VTune microarch exploration (15 min)        │
│  4. Validation: perf c2c for contention (10 min)                │
│  5. Scaling: run at 1x, 10x, 100x size (verify roofline)        │
│                                                                    │
└──────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────┐
│         PERF EVENT MULTIPLEXING (4 Physical Counters)             │
├──────────────────────────────────────────────────────────────────┤
│                                                                    │
│  Scenario: Measure 8 events (cycles, instr, L1-miss, L3-miss,   │
│            branch-miss, context-switch, page-fault, task-clock)  │
│                                                                    │
│  Without Multiplexing (wrong):                                    │
│  ┌─────────┬──────────┬──────────┬──────────┐                   │
│  │cycles   │instr     │L1-miss   │L3-miss   │  ← 4 physical    │
│  └─────────┴──────────┴──────────┴──────────┘                   │
│  ✗ Never count branch-miss, context-switch, etc.               │
│                                                                    │
│  With Multiplexing (correct):                                    │
│  Pass 1: [cycles] [instr] [L1-miss] [L3-miss]                  │
│  Pass 2: [cycles] [instr] [branch-miss] [context-switch]       │
│  Pass 3: [cycles] [instr] [page-fault] [task-clock]            │
│                                                                    │
│  Scaling: actual_L1_miss = measured_L1_miss × (total_cycles     │
│           / cycles_pass1)                                         │
│                                                                    │
│  Error: ±10% on each scaled event (accumulates to ±15% overall) │
│                                                                    │
│  Expert strategy: measure < 6 events per run, accept scaling    │
│  error as part of methodology                                     │
│                                                                    │
└──────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────┐
│      FLAMEGRAPH INTERPRETATION: OFF-CPU BLOCKING ANALYSIS         │
├──────────────────────────────────────────────────────────────────┤
│                                                                    │
│  On-CPU Flamegraph (samples when CPU executes):                 │
│  ┌──────────────────────────────────┐                           │
│  │         main                      │                           │
│  │    ┌────────┬────────┬────────┐   │                           │
│  │    infer   optimize  serialize    │                           │
│  │  ┌─┴─┐  ┌──┴──┐  ┌────┴────┐    │                           │
│  │ gemm aten reshape copy_out ...   │                           │
│  └──────────────────────────────────┘                           │
│  Shows: which functions burn CPU time                           │
│                                                                    │
│  Off-CPU Flamegraph (samples when sleeping):                    │
│  ┌──────────────────────────────────┐                           │
│  │         main                      │                           │
│  │    ┌────────┬────────┬────────┐   │                           │
│  │   futex_wait  epoll  pthread_cond │                          │
│  │  ┌─┴──┐  ┌───┴────┐  ┌───┴────┐   │                          │
│  │rdlock  poll  recv  mutex_unlock   │                          │
│  └──────────────────────────────────┘                           │
│  Shows: where threads block (sync, I/O, locks)                  │
│                                                                    │
│  Interpretation:                                                  │
│  • Large "futex_wait" block → contention on locks → redesign    │
│  • Large "recv" block → network latency → check bandwidth       │
│  • Large "malloc" block → memory pressure → tune allocator      │
│                                                                    │
└──────────────────────────────────────────────────────────────────┘
```

---

## 3. PERFORMANCE LENS

Each profiling tool reveals different bottlenecks:

**perf stat**: Quick snapshot of hardware resource utilization. High context-switches indicate preemption. High page-faults indicate poor memory locality. IPC (instructions per cycle) < 2 indicates memory or frontend issues.

**perf record + flamegraph**: Reveals which functions consume time. For ML inference, a flamegraph that's 60% in attention kernel and 30% in GEMM means focus optimization effort on attention (low-hanging fruit).

**Intel VTune**: Directly tells you "38% DRAM_Bound" without needing to manually interpret event ratios. If TMA shows memory is the bottleneck, VTune can attribute that to specific loops in your code.

**perf c2c**: Reveals invisible cache coherency traffic. Two threads updating unrelated fields in the same struct can silently kill performance (false sharing). A perf c2c report showing high "hitm" on a single cache line is diagnostic.

**LIKWID**: Provides accurate counter groups without multiplexing error. If you're tuning DRAM bandwidth utilization, LIKWID's "MEM" counter group is more reliable than manually combining perf events.

**valgrind cachegrind**: Deterministic, no sampling error. Useful for understanding *capacity* vs *conflict* misses. If cachegrind shows 95% capacity misses (data doesn't fit), algorithm change is needed; if 50% conflict misses (same data evicted multiple times), padding or blocking helps.

---

## 4. ANNOTATED CODE

### Example 1: perf stat — Extracting Actionable Metrics

```bash
# File: profile_kernel.sh
# Comprehensive perf stat run with multiple event sets

#!/bin/bash

# Function: Run perf stat with specific event set
run_perf_stat() {
    local events="$1"
    local desc="$2"
    local output_file="$3"

    echo "=== $desc ===" | tee -a "$output_file"
    perf stat -e "$events" -r 3 -- ./your_kernel >> "$output_file" 2>&1
}

# Ensure sufficient privileges
if [ "$(id -u)" != "0" ]; then
    echo "Warning: running as non-root; some events may not be available"
    echo "Run with: sudo bash $0"
fi

OUTPUT="perf_analysis.txt"
> "$OUTPUT"  # clear file

# Pass 1: Core execution metrics
run_perf_stat \
    "cycles,instructions,stalled-cycles-frontend,stalled-cycles-backend" \
    "Core Execution Metrics" \
    "$OUTPUT"

# Pass 2: Cache hierarchy
run_perf_stat \
    "cache-references,cache-misses,LLC-loads,LLC-load-misses,LLC-stores,LLC-store-misses" \
    "Cache Hierarchy" \
    "$OUTPUT"

# Pass 3: Memory subsystem
run_perf_stat \
    "mem_load_retired.l1_hit,mem_load_retired.l2_hit,mem_load_retired.l3_hit,mem_load_l3_miss_retired.local_dram,mem_load_l3_miss_retired.remote_dram" \
    "Memory Loads by Level" \
    "$OUTPUT"

# Pass 4: Branch prediction
run_perf_stat \
    "branch-instructions,branch-misses,br_misp_retired.all_branches" \
    "Branch Prediction" \
    "$OUTPUT"

# Pass 5: Compute resources (AVX-512 specific)
run_perf_stat \
    "fp_arith_inst_retired.512b_packed_single,fp_arith_inst_retired.512b_packed_double,fp_arith_inst_retired.scalar" \
    "FP Execution Width" \
    "$OUTPUT"

echo "Analysis complete. Results in $OUTPUT"

# Post-process: calculate derived metrics
python3 << 'PYTHON'
import re

with open('perf_analysis.txt', 'r') as f:
    content = f.read()

# Extract key metrics
print("\n=== DERIVED METRICS ===")

# IPC from cycles and instructions
cycles_match = re.search(r'(\d+\.?\d*)\s+cycles', content)
instr_match = re.search(r'(\d+\.?\d*)\s+instructions', content)
if cycles_match and instr_match:
    cycles = float(cycles_match.group(1).replace(',', ''))
    instr = float(instr_match.group(1).replace(',', ''))
    ipc = instr / cycles
    print(f"IPC (Instructions Per Cycle): {ipc:.2f}")
    if ipc < 1.0:
        print("  → Low IPC suggests memory-bound or frontend stalls")
    elif ipc > 3.0:
        print("  → High IPC suggests good vectorization and low stalls")

# L3 hit rate
cache_miss = re.search(r'(\d+\.?\d*)\s+cache-misses', content)
cache_ref = re.search(r'(\d+\.?\d*)\s+cache-references', content)
if cache_miss and cache_ref:
    misses = float(cache_miss.group(1).replace(',', ''))
    refs = float(cache_ref.group(1).replace(',', ''))
    hit_rate = 1.0 - (misses / refs)
    print(f"Cache Hit Rate: {hit_rate*100:.1f}%")
    if hit_rate > 0.95:
        print("  → Excellent: data fits in L3")
    elif hit_rate > 0.80:
        print("  → Good: mostly L3 hits")
    else:
        print("  → Poor: many DRAM accesses")

PYTHON
```

### Example 2: Intel perf event multiplexing — Accurate Multi-Event Measurement

```c
// File: perf_multiplexing.c
// Demonstrate perf's multiplexing and how to interpret scaled results

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    unsigned long cycles;
    unsigned long instructions;
    unsigned long l1_misses;
    unsigned long l2_misses;
    unsigned long l3_misses;
    unsigned long branch_misses;

    // Scaling metadata
    unsigned long time_enabled;
    unsigned long time_running;  // actual time counter ran (< time_enabled if multiplexed)
} perf_event_results_t;

// When perf multiplexes events, it reports:
//   count (raw count when this event was active)
//   time_enabled (total time we *wanted* to measure)
//   time_running (actual time this event was measured)
//
// Scaled count = count × (time_enabled / time_running)
// Error margin ≈ (1 - time_running/time_enabled) * 100%

void print_perf_results(const perf_event_results_t *results) {
    printf("Raw Performance Counter Results:\n");
    printf("  cycles:           %lu\n", results->cycles);
    printf("  instructions:     %lu\n", results->instructions);
    printf("  L1 misses:        %lu\n", results->l1_misses);
    printf("  L2 misses:        %lu\n", results->l2_misses);
    printf("  L3 misses:        %lu\n", results->l3_misses);
    printf("  branch misses:    %lu\n", results->branch_misses);

    // Calculate scaling factor
    double scale_factor = (double)results->time_enabled / results->time_running;
    printf("\nScaling Information:\n");
    printf("  time_enabled:     %lu ns\n", results->time_enabled);
    printf("  time_running:     %lu ns\n", results->time_running);
    printf("  scale factor:     %.2f (multiplexing overhead: %.1f%%)\n",
           scale_factor, (1.0 - 1.0/scale_factor) * 100);

    // Derived metrics
    double ipc = (double)results->instructions / results->cycles;
    double l1_miss_rate = (double)results->l1_misses / results->instructions;
    double l3_miss_rate = (double)results->l3_misses / results->instructions;
    double branch_miss_rate = (double)results->branch_misses / results->instructions;

    printf("\nDerived Metrics:\n");
    printf("  IPC:              %.2f instructions/cycle\n", ipc);
    printf("  L1 miss rate:     %.2f%% of instructions\n", l1_miss_rate * 100);
    printf("  L3 miss rate:     %.2f%% of instructions\n", l3_miss_rate * 100);
    printf("  branch miss rate: %.2f%% of instructions\n", branch_miss_rate * 100);

    // Interpretation
    printf("\nInterpretation:\n");
    if (scale_factor > 1.2) {
        printf("  ⚠ Multiplexing error > 20%%, consider re-running with fewer events\n");
    }
    if (ipc < 1.0) {
        printf("  → Memory or frontend stalls likely (IPC < 1.0)\n");
    }
    if (l3_miss_rate > 0.1) {
        printf("  → High L3 miss rate (>10%%), consider tiling/blocking\n");
    }
    if (branch_miss_rate > 0.05) {
        printf("  → High branch miss rate (>5%%), consider branchless code\n");
    }
}

// Expert advice: measure in separate runs to avoid multiplexing
void expert_multirun_measurement() {
    printf("Expert Strategy: Separate perf runs\n");
    printf("\n$ perf stat -e cycles,instructions,stalled-cycles-frontend,stalled-cycles-backend ./kernel\n");
    printf("$ perf stat -e LLC-loads,LLC-load-misses,LLC-stores,LLC-store-misses ./kernel\n");
    printf("$ perf stat -e mem_load_l3_miss_retired.local_dram,mem_load_l3_miss_retired.remote_dram ./kernel\n");
    printf("$ perf stat -e branch-instructions,branch-misses ./kernel\n");
    printf("\nEach run has < 5% scaling error. Combine results offline.\n");
}

int main() {
    // Example results from a latency-bound kernel
    perf_event_results_t results = {
        .cycles = 1000000000,
        .instructions = 500000000,
        .l1_misses = 5000000,
        .l2_misses = 3000000,
        .l3_misses = 1500000,
        .branch_misses = 2000000,
        .time_enabled = 1000000000,  // 1 second
        .time_running = 850000000,   // only 850ms due to multiplexing
    };

    print_perf_results(&results);
    printf("\n");
    expert_multirun_measurement();

    return 0;
}
```

### Example 3: Flamegraph Generation and Interpretation

```bash
#!/bin/bash
# File: generate_flamegraph.sh
# Complete workflow: perf record → flamegraph

BINARY="${1:-.your_kernel}"
DURATION="${2:-30}"

echo "Step 1: Record performance data for $DURATION seconds"
perf record -F 99 -g -- "$BINARY" sleep "$DURATION"
# -F 99: sample at 99 Hz (1% overhead)
# -g: record call stacks

echo "Step 2: Convert perf.data to readable format"
perf script > out.perf
# This expands the compressed perf.data

echo "Step 3: Prepare FlameGraph scripts"
# Download from http://www.brendangregg.com/flamegraphs.html
git clone https://github.com/brendangregg/FlameGraph.git

echo "Step 4: Generate on-CPU flamegraph"
./FlameGraph/stackcollapse-perf.pl out.perf > out.folded
./FlameGraph/flamegraph.pl out.folded > on_cpu.svg
# Shows which functions consume CPU time

echo "Step 5: Generate off-CPU flamegraph (blocking analysis)"
perf record -e sched:sched_switch -a -- "$BINARY" sleep "$DURATION"
perf script > out.blocking
./FlameGraph/stackcollapse-perf.pl out.blocking > out.blocking.folded
./FlameGraph/flamegraph.pl --color=block out.blocking.folded > off_cpu.svg
# Shows where threads are blocked (locks, I/O, syscalls)

echo "Step 6: Generate memory allocation flamegraph"
perf record -e malloc,free -a -- "$BINARY" sleep "$DURATION"
perf script > out.malloc
./FlameGraph/stackcollapse-perf.pl out.malloc > out.malloc.folded
./FlameGraph/flamegraph.pl --color=mem out.malloc.folded > memory.svg
# Shows which functions allocate memory

echo "Outputs: on_cpu.svg, off_cpu.svg, memory.svg"
echo "Open in browser to explore"

# Expert tip: diff two runs to see changes
perf record -F 99 -g -- ./kernel_v1 sleep 10
perf script > v1.perf
./FlameGraph/stackcollapse-perf.pl v1.perf > v1.folded

perf record -F 99 -g -- ./kernel_v2 sleep 10
perf script > v2.perf
./FlameGraph/stackcollapse-perf.pl v2.perf > v2.folded

# Generate differential flamegraph
./FlameGraph/flamegraph.pl --bgColor=blue v1.folded > v1.svg
./FlameGraph/flamegraph.pl --bgColor=blue v2.folded > v2.svg

# Or use: flamegraph --compare v1.folded v2.folded > diff.svg
```

### Example 4: perf c2c — False Sharing Diagnosis

```c
// File: false_sharing_demo.c
// Two threads updating unrelated fields in the same cache line

#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <unistd.h>
#include <string.h>

struct shared_data_t {
    uint64_t counter_a;          // Updated by thread 0
    // 8 bytes, but cache line is 64 bytes, so plenty of room
    uint64_t padding[7];         // REMOVE THIS TO TRIGGER FALSE SHARING
    uint64_t counter_b;          // Updated by thread 1
} shared = {0};

void *thread_a(void *arg) {
    volatile uint64_t *cnt = &shared.counter_a;
    for (int i = 0; i < 1000000000; i++) {
        (*cnt)++;
    }
    return NULL;
}

void *thread_b(void *arg) {
    volatile uint64_t *cnt = &shared.counter_b;
    for (int i = 0; i < 1000000000; i++) {
        (*cnt)++;
    }
    return NULL;
}

int main() {
    // Measure with and without padding
    printf("Size of shared data: %lu bytes\n", sizeof(struct shared_data_t));
    printf("Note: cache line is 64 bytes\n");

    if (sizeof(struct shared_data_t) <= 64) {
        printf("WARNING: counter_a and counter_b are in the SAME cache line!\n");
        printf("This will cause FALSE SHARING between threads.\n\n");
        printf("Run with: perf c2c record -g -- ./false_sharing\n");
        printf("Then: perf c2c report\n");
        printf("You'll see high 'hitm' on the shared cache line.\n\n");
        printf("Fix: Add padding, or use thread-local storage:\n");
        printf("  __thread uint64_t counter_a = 0;\n");
        printf("  __thread uint64_t counter_b = 0;\n");
    }

    pthread_t t0, t1;
    pthread_create(&t0, NULL, thread_a, NULL);
    pthread_create(&t1, NULL, thread_b, NULL);

    pthread_join(t0, NULL);
    pthread_join(t1, NULL);

    printf("Counters: a=%lu, b=%lu\n", shared.counter_a, shared.counter_b);

    return 0;
}
```

Compilation and diagnosis:
```bash
gcc -O3 -pthread -o false_sharing false_sharing_demo.c

# Run with perf c2c to detect false sharing
sudo perf c2c record -g -t ./false_sharing

# Show which memory lines have high inter-socket transfers (hitm)
sudo perf c2c report --stdio

# Output excerpt:
#   LLC Load Hitm                      : X lines with high hitm (cache coherency cost)
#   hitm: 5000000 (cache line bouncing between cores)
```

---

## 5. EXPERT INSIGHT

### Non-Obvious Truth #1: perf Multiplexing Is Not Evil, Embrace It

Junior engineers avoid multiplexing by measuring only 4 events at a time, treating it as a limitation. **Experts use it strategically:**

For a 10-second profile, if you measure 12 events:
- Multiplexing creates ~3 passes
- Scaling error: ~10% per event
- You get 12 measurements in 30 seconds instead of 40 seconds (4 runs × 10 seconds)

**Trade-off is acceptable** if you:
1. Always check `time_running` / `time_enabled` ratio (should be > 0.9)
2. Don't rely on individual event ratios; instead use TMA aggregations (which absorb error)
3. Validate results by running again on a different kernel version (check consistency)

### Non-Obvious Truth #2: VTune's TMA and Raw perf Events Disagree

VTune calculates TMA using proprietary formulas + runtime adjustments. Raw perf counters are pure counts.

Example mismatch:
```
VTune: "45% DRAM_Bound"
perf:  mem_load_l3_miss_retired.local_dram = 100M events
       cycles = 200M
       => 50% DRAM_Bound (rough calc)
```

**Why the discrepancy?**
- VTune accounts for memory-level parallelism (multiple DRAM requests in flight)
- VTune adjusts for CPU frequency scaling during measurement
- VTune uses proprietary TMA 4.0 formulas (multi-threaded events)

**Expert move**: Use VTune for initial direction (which bottleneck to focus), then validate with raw perf counters. If they still disagree after sanity-checking, the CPU behavior is complex (microcode assists, NUMA effects)—dive deeper with LIKWID or cachegrind.

### Non-Obvious Truth #3: Flamegraph Sampling Is Biased

perf record with -F 99 (99 Hz sampling) introduces **sampling bias**:
- Functions with long call stacks show up less frequently (sample occurs during nested call)
- Leaf functions are overrepresented
- Fast functions that run many times might be invisible (inlined, optimized away)

**Mitigation**:
1. Sample at higher frequency: -F 499 (500 Hz) if you can afford 0.5% overhead
2. Use -g --call-graph=dwarf to capture full stacks (more accurate)
3. Run for longer duration (10-30 seconds) to collect more samples
4. Use perf record -c 10000 (capture every 10,000th cycle) instead of -F for consistency

### Non-Obvious Truth #4: LIKWID Counter Groups Are Biased Too

LIKWID groups events intelligently to avoid multiplexing, but the grouping is **hardcoded** and may not match your hypothesis.

Example: LIKWID's "DRAM" group measures:
- L3 cache misses
- DRAM bandwidth utilization
- Memory latency (via PEBS)

But it *doesn't* measure:
- Remote NUMA accesses
- Memory controller queue depth
- Prefetch accuracy

**Expert move**: Start with LIKWID for quick baseline, then custom perf runs for fine-grained questions.

### Non-Obvious Truth #5: Microbenchmark Results Don't Generalize

A STREAM benchmark shows 12.8 GB/s sustained bandwidth on Sapphire Rapids DRAM. But your real application achieves 2 GB/s.

**Reasons:**
1. **Striding**: STREAM accesses arrays sequentially; your app has random strides → prefetcher ineffective
2. **Latency hiding**: STREAM has 8 independent arrays in flight; your app has 2 → pipeline stalls
3. **Coherency traffic**: STREAM is single-threaded; your app's multi-threaded coherency stalls
4. **Vectorization**: STREAM uses AVX-512; your app uses scalar operations

**Expert interpretation**: Microbenchmark is an *upper bound*, not a prediction. Use roofline to derive a realistic bound for your algorithm.

---

## 6. BENCHMARK / MEASUREMENT

### Standard Profiling Workflow for ML Inference

```bash
#!/bin/bash
# File: profile_inference.sh
# Complete profiling of an inference engine

MODEL_PATH="./model.pth"
BATCH_SIZE=1
NUM_RUNS=5

echo "=== Phase 1: Hotspot Analysis (5 min) ==="
perf record -F 99 -g -- python run_inference.py \
    --model "$MODEL_PATH" --batch-size "$BATCH_SIZE" \
    --num-runs "$NUM_RUNS" --duration 10

perf report --stdio > hotspot.txt
perf script > perfdata.txt

echo "=== Phase 2: TMA Analysis (perf stat) ==="
perf stat -e cycles,instructions,stalled-cycles-frontend,stalled-cycles-backend \
    -- python run_inference.py --model "$MODEL_PATH" --batch-size "$BATCH_SIZE" --num-runs 10

echo "=== Phase 3: Cache Analysis ==="
perf stat -e cache-references,cache-misses,LLC-loads,LLC-load-misses \
    -- python run_inference.py --model "$MODEL_PATH" --batch-size "$BATCH_SIZE" --num-runs 10

echo "=== Phase 4: Memory Pressure (detailed) ==="
perf stat -e mem_load_retired.l1_hit,mem_load_retired.l2_hit,mem_load_retired.l3_hit \
    -e mem_load_l3_miss_retired.local_dram,mem_load_l3_miss_retired.remote_dram \
    -- python run_inference.py --model "$MODEL_PATH" --batch-size "$BATCH_SIZE" --num-runs 10

echo "=== Phase 5: Flamegraph Generation ==="
git clone https://github.com/brendangregg/FlameGraph.git || true
./FlameGraph/stackcollapse-perf.pl perfdata.txt > out.folded
./FlameGraph/flamegraph.pl out.folded > inference_profile.svg

echo "=== Phase 6: VTune Deep Dive (if available) ==="
if command -v vtune &> /dev/null; then
    vtune -collect uarch-exploration -knob pmu-event-sampling=true \
        -r vtune_results -- python run_inference.py \
        --model "$MODEL_PATH" --batch-size "$BATCH_SIZE" --num-runs 5

    vtune -report uarch-exploration -r vtune_results \
        -format csv > vtune_analysis.csv
fi

echo "Complete. Results in: hotspot.txt, inference_profile.svg, vtune_analysis.csv"
```

### Validating Roofline Against Measured Performance

```python
#!/usr/bin/env python3
# File: validate_roofline.py
# Measure actual kernel and compare to Roofline ceiling

import subprocess
import re

def extract_perf_metrics(perf_output):
    """Parse perf stat output"""
    cycles_match = re.search(r'(\d+\.?\d*)\s+cycles', perf_output)
    instr_match = re.search(r'(\d+\.?\d*)\s+instructions', perf_output)

    if cycles_match and instr_match:
        cycles = float(cycles_match.group(1).replace(',', ''))
        instr = float(instr_match.group(1).replace(',', ''))
        return {'cycles': cycles, 'instructions': instr}
    return {}

def measure_kernel_performance(kernel_binary, flop_count):
    """Run kernel and get actual GFLOPs"""
    result = subprocess.run(
        ['perf', 'stat', kernel_binary],
        capture_output=True,
        text=True,
        timeout=60
    )

    metrics = extract_perf_metrics(result.stderr)
    if 'cycles' in metrics:
        # Assume 3 GHz for calculation
        time_seconds = metrics['cycles'] / 3e9
        gflops = flop_count / time_seconds / 1e9
        return gflops, metrics
    return None, metrics

def roofline_ceiling(ai, peak_compute=1500, peak_bandwidth=12.8):
    """Calculate roofline ceiling for given AI"""
    memory_ceiling = peak_bandwidth * ai
    return min(memory_ceiling, peak_compute)

# Example: measure dense GEMM
M, N, K = 1024, 1024, 1024
flops_gemm = 2 * M * N * K
bytes_gemm = (M*K + K*N + M*N) * 8  # FP64
ai_gemm = flops_gemm / bytes_gemm
roofline_gemm = roofline_ceiling(ai_gemm)

print(f"Dense GEMM {M}x{N}x{K}:")
print(f"  FLOPs: {flops_gemm:.2e}")
print(f"  Bytes (est): {bytes_gemm:.2e}")
print(f"  AI: {ai_gemm:.2f} FLOPs/byte")
print(f"  Roofline ceiling: {roofline_gemm:.0f} GFLOPs")

actual_gflops, metrics = measure_kernel_performance('./gemm_kernel', flops_gemm)
if actual_gflops:
    print(f"  Actual measured: {actual_gflops:.0f} GFLOPs")
    print(f"  Efficiency: {100 * actual_gflops / roofline_gemm:.1f}%")
    if actual_gflops > roofline_gemm * 1.05:
        print("  WARNING: measured > roofline (measurement error or AI calculation off)")
    elif actual_gflops < roofline_gemm * 0.8:
        print("  → Opportunity: you're 20% below roofline ceiling")
```

---

## 7. ML SYSTEMS RELEVANCE

### Profiling Attention Layers in Production

Attention is both compute-intensive and memory-intensive. Profile it separately:

```bash
# Isolate attention kernel from embedding/MLP
perf record -e cycles,instructions,LLC-loads,LLC-load-misses \
    --filter='module==libattention.so' \
    -- python inference_server.py

# Result: see if attention is memory-bound (high LLC-load-misses)
# or compute-bound (high IPC)
```

If attention is memory-bound (LLC misses > 20%):
- Roofline AI analysis shows it's below compute-bound ceiling
- **Optimization**: Flash Attention (fuse Q, K, V loads; reduce memory bandwidth)

If attention is compute-bound (LLC misses < 5%, IPC > 2.5):
- Already well-optimized; bottleneck is hardware compute capacity
- **Optimization**: quantize to lower precision, speculative decoding

### Batching Impact Profiling

Profile the same inference server at batch sizes 1, 4, 16, 64:

```bash
for batch_size in 1 4 16 64; do
    echo "Batch size: $batch_size"
    perf stat -- python inference_server.py --batch "$batch_size" \
        --num-requests 1000 2>&1 | grep -E "IPC|cache-misses"
done
```

Expected pattern:
- Batch 1: IPC ≈ 1.5, LLC-misses high (latency-bound)
- Batch 16: IPC ≈ 3.0, LLC-misses moderate (throughput-bound)
- Batch 64: IPC ≈ 2.0, LLC-misses low (sustained memory bandwidth)

The "sweet spot" is where IPC plateaus—increase batch further yields diminishing returns.

### Quantization Performance Validation

Compare FP32 vs INT8 inference:

```bash
# FP32 baseline
perf stat --repeat 10 -- ./inference --dtype float32 | grep GFLOPs

# INT8 optimized
perf stat --repeat 10 -- ./inference --dtype int8 | grep GFLOPs

# Roofline:
# FP32: 1.2 TB/s ÷ 12.8 GB/s = 93 compute per bandwidth unit
# INT8: 4.8 TB/s ÷ 12.8 GB/s = 375 compute per bandwidth unit
# INT8 should be ~2-3× faster (if memory-bound)
```

If INT8 is only 1.5× faster, roofline says you're hitting compute ceiling on INT8 (opportunity for parallelization).

---

## 8. PhD QUALIFIER QUESTIONS

**Q1**: Explain why perf counter multiplexing introduces systematic error, and propose a measurement methodology to bound that error to < 5% for 12 simultaneous events.

**Q2**: Generate an on-CPU flamegraph of a transformer inference engine and an off-CPU flamegraph of the same engine. What bottlenecks are visible in each, and how would you prioritize optimizations?

**Q3**: Use perf c2c to diagnose false sharing in a multi-threaded inference server. Describe the symptoms (cache coherency traffic), propose a fix, and validate it with another perf c2c run.

**Q4**: For a memory-bandwidth-limited kernel (LLC-load-misses > 25%), propose a measurement plan using perf, VTune, and LIKWID. Which tool gives the most actionable result, and why?

**Q5**: A STREAM benchmark shows 12.8 GB/s sustained DRAM bandwidth, but your GEMM kernel achieves only 3.2 GB/s. Use perf counters to diagnose whether the limitation is (a) striding/prefetch, (b) request queue saturation, (c) coherency traffic, or (d) latency-hiding inefficiency.

---

## 9. READING LIST

1. **Gregg, B.** (2021). *Systems Performance: Enterprise and the Cloud*. Addison-Wesley, 2nd ed.
   - See: Chapter 4 (Linux perf), Chapter 5 (CPU profiling), Chapter 6 (memory)

2. **Dendibakh, D.** (2024). *Performance Analysis and Tuning on Modern CPUs*. O'Reilly Media.
   - See: Chapter 3 (hardware counters), Chapter 4 (TMA), Chapter 5 (perf tools)

3. **Aczel, S. et al.** (2020). "Performance Characterization of the Core i7 9700K." *MEMSYS 2020*.
   - See: Section 3 (counter multiplexing), Section 4 (measurement methodology)

4. **Semeraro, G. et al.** (2019). "Performance Analysis of Cache Coherency Protocol on Multicore Processors." *ISPASS 2019*.
   - See: perf c2c methodology, false sharing diagnosis

5. **Linux perf Wiki**. https://perf.wiki.kernel.org/
   - See: Event types, event list, counter multiplexing

6. **Intel VTune Profiler User Guide**. https://www.intel.com/content/www/en/us/en/docs/vtune/user-guide/
   - See: Microarchitecture Exploration, Threading Analysis

7. **Gregg, B.** (2023). *The Art of Performance Profiling*. YouTube series.
   - See: Flamegraph techniques, off-CPU analysis

8. **Treibig, J. et al.** (2010). "LIKWID: A Lightweight Performance-Oriented Tool Family." *ISPASS 2010*.
   - See: Counter group design, topology analysis

---

**End of Module 15**

*Profiling mastery is the bridge between theory (Module 14) and implementation (Module 16). Without rigorous measurement, optimization is guesswork.*
