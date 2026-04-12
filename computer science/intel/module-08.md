# MODULE 8 — Instruction-Level Parallelism & Superscalar Execution

## 1. CONCEPTUAL FOUNDATION

Instruction-level parallelism (ILP) is the foundation of superscalar processors. While out-of-order execution (Module 6) and branch prediction (Module 7) are necessary infrastructure, ILP determines the **maximum** instruction throughput a processor can achieve. This module bridges theory and practice, quantifying ILP, describing execution port architecture, and explaining modern optimization techniques (loop unrolling, software pipelining, uop cache exploitation).

### ILP vs IPC: Theoretical Capacity vs Achieved Throughput

**Instruction-Level Parallelism (ILP)** is the maximum number of instructions that could execute in parallel if dependencies permitted:

$$\text{ILP} = \frac{\text{average number of independent instructions in window}}{\text{longest dependency latency}}$$

**Instruction Per Cycle (IPC)** is the achieved instruction throughput, always ≤ ILP:

$$\text{IPC} = \frac{\text{instructions completed}}{\text{cycles elapsed}} \leq \text{min}(\text{ILP}, \text{fetch width}, \text{execution ports})$$

**Example**:

```c
// Compute ILP
int a = x + y;      // Instr 1, latency 1
int b = x * y;      // Instr 2, latency 3 (independent of 1)
int c = a + b;      // Instr 3, latency 1 (depends on 1 and 2)
```

**ILP calculation**:
- Window: 3 instructions
- Dependencies: 1 and 2 are independent, 3 depends on both
- Critical path latency: max(1, 3) + 1 = 4 cycles (longest chain)
- ILP = 2 (instructions 1 and 2 can execute in parallel)

**IPC on 6-wide processor**:
- Fetch width: 6 instructions/cycle
- Execution ports: 6 available
- Bottleneck: ILP = 2 (can only parallelize 1 and 2)
- Achieved IPC = 2 (not 6)

### Superscalar Execution: Multiple Ports and Instruction Mix

Modern superscalar processors (6-8 wide) replicate execution units to exploit ILP:

**Intel Skylake (6-wide superscalar)**:
```
Ports and functional units:
Port 0: ALU, branch, shift      (1 unit, can execute 1 instr/cycle)
Port 1: ALU, multiply           (1 unit, can execute 1 instr/cycle)
Port 2: Load (AGU)              (1 unit, can execute 1 load/cycle)
Port 3: Load (AGU)              (1 unit, can execute 1 load/cycle)
Port 4: Store address (AGU)     (1 unit, can execute 1 store addr/cycle)
Port 5: ALU, shift              (1 unit, can execute 1 instr/cycle)
Port 6: Branch, ALU             (1 unit, can execute 1 instr/cycle)
Port 7: Store data write        (1 unit, coordinates store)
Port 8: (Reserved)

Aggregated capacity:
- ALU capacity: 4 ports (0, 1, 5, 6) = 4 instructions/cycle
- Load capacity: 2 ports (2, 3) = 2 instructions/cycle
- Store capacity: 1 port (4) for address = 1 store address/cycle
- Branch capacity: 2 ports (0, 6) = 2 branches/cycle
```

**Instruction mix limits IPC**:
- Pure ALU code: IPC capped at 4 (limited by ALU port count)
- Load-heavy code: IPC capped at 2 (limited by load port count)
- Mixed code: Dynamic scheduling selects ports based on instruction types

**Port pressure**: When all instances of a port type are busy, subsequent instructions of that type must stall.

### Latency vs Throughput

Each instruction has two performance metrics:

**Latency**: Cycles from start to result availability
- Integer add: 1 cycle latency
- Integer multiply: 3 cycles latency (Skylake)
- Load from L1: 4 cycles latency
- Load from L3: 12-40 cycles latency
- Divider: 12-40 cycles latency

**Throughput**: Maximum instructions per cycle of that type
- Integer add: 1 instruction per cycle (can issue one per cycle)
- Integer multiply: 1 instruction per cycle (latency 3, but can issue new mults every cycle due to pipeline)
- Load: 2 per cycle (2 load ports)
- Divide: 1 per 10-40 cycles (low throughput due to sequential execution in divider)

**Key insight**: Latency and throughput are **independent**. A multiply with latency 3 can sustain throughput 1 per cycle because Skylake has a deeply pipelined multiplier.

### Loop Unrolling: Trading Code Size for ILP

Loop unrolling manually increases the number of independent instructions in the loop body, enabling parallelization:

**Example: Scalar loop (no unrolling)**

```c
int sum = 0;
for (int i = 0; i < 100; i += 1) {
    sum += arr[i];  // Load, add, dependency chain
}
```

**Execution analysis**:
- Iteration 0: Load arr[0], add to sum (latency 4 for load + 1 for add = 5 cycles minimum)
- Iteration 1: Load arr[1], add to sum (must wait for iteration 0 to complete)
- Dependency chain serializes iterations, CPI ≈ 5

**Example: 4× unrolled loop**

```c
int sum = 0;
for (int i = 0; i < 100; i += 4) {
    sum += arr[i];     // Load 1, add 1
    sum += arr[i+1];   // Load 2, add 2 (independent)
    sum += arr[i+2];   // Load 3, add 3 (independent)
    sum += arr[i+3];   // Load 4, add 4 (independent)
}
```

**Execution analysis**:
- Load 1, 2, 3, 4 can execute in parallel (cycle 0-2, two per cycle on Skylake's 2 load ports)
- Adds 1, 2, 3, 4 can execute in parallel (cycle 4-6, on ALU ports)
- Loop branch (cycle 1)
- **Total per iteration**: 7 cycles for 4 iterations = 1.75 cycles per original iteration
- Speedup: 5 / 1.75 = **2.8×** (compared to unrolled baseline)

**Why unrolling helps**:
1. **Reduces branch frequency** (1 branch per 4 iterations instead of per 1)
2. **Increases parallelism** (4 independent load/add chains instead of 1)
3. **Improves cache efficiency** (fewer iterations, better prefetch behavior)

### Software Pipelining: Interleaving Loop Iterations

Software pipelining is a compiler technique that **overlaps iterations of a loop**, hiding latency:

**Concept**:

```
Unrolled loop (sequential):
Iter 0: Load, [4-cycle latency], Add
Iter 1: Load, [4-cycle latency], Add
Iter 2: Load, [4-cycle latency], Add

Software-pipelined loop (overlapped):
Iter 0: Load
Iter 1: Load, Add (data from Iter 0's load)
Iter 2: Load, Add (data from Iter 1's load)
Iter 3: Load, Add (data from Iter 2's load)
Iter 4: Add (data from Iter 3's load)
```

**Effect**: Hides the 4-cycle load latency by interleaving 4 independent iterations.

### uop Cache (DSB): A Critical but Invisible Bottleneck

Modern processors use a **micro-operation cache (uop cache)** that stores decoded instructions. Fetching from uop cache (DSB - Decoded Stream Buffer) is faster than decoding from instruction cache (MITE - Legacy Decode).

**uop cache characteristics** (Skylake):
- Size: 64 lines × 6 uops = 384 uops (1.5 KB)
- Hit rate: 90%+ on typical code (loops stay in cache)
- Miss rate: 10% (cache misses, large code, indirect jumps)
- **Performance impact**: DSB hit = decode free; DSB miss = 1-6 cycle decode latency

**DSB vs MITE trade-off**:

```
DSB (uop cache) path:
  PC → DSB lookup → 6 uops/cycle → Low decode latency

MITE (legacy decode) path:
  PC → I-cache lookup → Decode (1 instr/cycle) → Legacy decode latency
```

**Performance implication**: Code that fits in DSB (typical loops, function bodies) has lower latency. Code with frequent DSB misses (very large functions, self-modifying code) suffers 1-2 cycle latency penalty per miss.

### Micro-fusion and Macro-fusion: Combining Operations

Modern processors combine multiple µops into single operations to reduce execution pressure:

**Micro-fusion (µfusion)**: Hardware combines 2 µops during decoding
- Example: `add rax, [rip + 8]` (ALU + memory load) → 1 fused µop
- Advantage: Reduces µop count, frees execution units
- Limitation: Only specific patterns (ALU operation + memory access)

**Macro-fusion**: Hardware fuses instructions at decode time
- Example: `cmp eax, ebx; je target` (compare + conditional jump) → 1 fused µop
- Advantage: Single µop for common pattern, saves execution unit cycles
- Limitation: Specific patterns only (cmp/test + conditional branch)

**Impact**: Fused operations reduce total µop count, allowing higher IPC from same execution units.

### LSD (Loop Stream Detector): Exploiting Loop Locality

The LSD is a small cache that detects and optimizes **tight loops** (loops with small iteration count and small body size):

**LSD characteristics** (Skylake):
- Detects loops with: iteration count < 64, loop size < 128 bytes
- When detected: Caches loop in LSD, prevents frequent fetch/decode
- Cycles loop locally without accessing instruction cache
- **Latency reduction**: Zero additional fetch latency once loop detected

**Activation conditions**:
1. Loop must execute backward branch (detected as backward)
2. Iteration count must be small (< 64)
3. Loop body must fit in LSD (< 128 bytes, typically 20-30 instructions)

**Example: LSD-optimized loop**

```c
// Small loop: Perfect for LSD exploitation
for (int i = 0; i < 32; i++) {
    result += arr[i];       // 5 instructions (load, add, loop branch)
}
```

**Execution**:
- Iteration 1-3: Normal fetch, decode
- Iteration 4+: Loop detected by LSD
- Iterations 5+: LSD supplies instructions directly, zero fetch latency
- **Speedup**: Zero fetch latency after initial iterations, flat IPC

**Comparison: Non-LSD-optimized loop**

```c
// Large loop: LSD cannot optimize
for (int i = 0; i < 100; i++) {
    // Large loop body (200+ bytes)
    // ... 50 instructions ...
}
```

**Execution**:
- Every iteration: Fetch (potentially DSB miss if loop doesn't fit in uop cache)
- Fetch latency: 1-2 cycles
- Effective CPI: 1.5-2.5 (due to fetch latency from I-cache misses)

**Citation**: Mutlu lectures on ILP, Agner Fog optimization manual; Intel optimization reference manual Chapters 3-4.

## 2. MENTAL MODEL

```
EXECUTION PORT ARCHITECTURE (SKYLAKE)
════════════════════════════════════════════════════════════

Decode Stage (6 µops/cycle max)
         │
         ├→ Execution port 0 (ALU, Branch, Shift)     ┐
         ├→ Execution port 1 (ALU, Multiply)          │
         ├→ Execution port 2 (Load AGU)               │
         ├→ Execution port 3 (Load AGU)               ├─ 6 ports
         ├→ Execution port 4 (Store address)          │ (can execute
         ├→ Execution port 5 (ALU, Shift)             │  6 instrs/cycle)
         └→ Execution port 6 (Branch, ALU)            ┘

         ↓
    Execution Units (functional units)

    P0: ALU, Branch logic    P1: ALU, Multiply (3-cycle latency)
    P2,P3: Load units (L1 interface, 4-cycle latency)
    P4: Store AGU
    P5,P6: ALU

Constraints:
- Only 1 multiply per cycle (Port 1)
- Only 2 loads per cycle (Ports 2, 3)
- Only 1 store address per cycle (Port 4)
- But 4 ALU operations per cycle (Ports 0, 1, 5, 6)

INSTRUCTION MIX IMPACT ON MAXIMUM IPC
════════════════════════════════════════════════════════════

Code Type | Bottleneck | Max IPC | Notes
----------|-----------|---------|----------
Pure ALU  | ALU ports | 4.0     | 4 ALU ports (0,1,5,6)
Pure load | Load ports| 2.0     | 2 load ports (2,3)
Pure store| Store port| 1.0     | 1 store address per cycle
Mult-only | Mult port | 1.0     | 1 multiply per cycle (Port 1)
Mixed ALU+Load | Load ports | 2.0 | Load latency dominates (4 cycles)
  (50% ALU, 50% Load)

LOOP UNROLLING EFFECT ON ILP
════════════════════════════════════════════════════════════

Original loop (scalar):
┌─────────────────────────┐
│ i = 0:                  │
│   load arr[0]           │ ── Iteration 0 (dependency chain)
│   add sum               │
│   → next iteration      │
│ i = 1: (depends on 0)   │
│   load arr[1]           │ ── Iteration 1 (must wait for iteration 0)
│   add sum               │
│   → next iteration      │
└─────────────────────────┘

Dependency chain: Each iteration waits for previous
CPI ≈ 5 (latency of load + add)

4× Unrolled loop (parallel):
┌──────────────────────────────┐
│ i = 0:                       │
│   load arr[0]                │
│   load arr[1] (parallel)     │ ← Load 1, 2, 3, 4 execute
│   load arr[2] (parallel)     │   in parallel (2 per cycle)
│   load arr[3] (parallel)     │
│   add sum (from arr[0])      │
│   add sum (from arr[1])      │ ← Adds execute after loads ready
│   add sum (from arr[2])      │
│   add sum (from arr[3])      │
│   → next iteration           │
└──────────────────────────────┘

ILP: 4 independent loads and 4 independent adds
CPI ≈ 1.75 (2 loads per cycle for 2 cycles, 1 add per cycle)
Speedup: 5 / 1.75 ≈ 2.8×

UOP CACHE (DSB) vs INSTRUCTION CACHE (MITE)
════════════════════════════════════════════════════════════

DSB (Decoded Stream Buffer) - uop cache:
PC → ┌─────────────────┐    ┌────────────────┐
     │ DSB Lookup      │───→│ 6 µops/cycle   │ (decoded, ready to execute)
     │ (64 lines,      │    │ Direct to exec │ (Zero decode latency)
     │  6 µops each)   │    │                │
     └─────────────────┘    └────────────────┘
     Hit rate: 90% (typical)
     Hit latency: 0 cycles (cache, decoded)

MITE (Legacy Decode) - instruction cache:
PC → ┌─────────────────┐    ┌───────────┐   ┌────────────────┐
     │ I-cache Lookup  │───→│ Decode    │──→│ 1 µop/cycle    │ (legacy decode)
     │ (32-64 KB)      │    │ Logic     │   │ 1-2 cycles     │ (1-2 cycle latency)
     └─────────────────┘    └───────────┘   └────────────────┘
     Miss rate: 10% (typical)
     Miss latency: 1-2 cycles (decode latency)

Performance impact:
- DSB hit: 0 cycles decode latency
- DSB miss: 1-2 cycles decode latency (restart from MITE)

Total throughput:
- With DSB hits (90%): 90% × 6 µops + 10% × 1 µop = 5.5 µops/cycle average
- Without DSB (forced MITE): 1 µop/cycle decode latency, 4-5 µops/cycle actual

LSD (LOOP STREAM DETECTOR)
════════════════════════════════════════════════════════════

Detects small loops (< 64 iterations, < 128 bytes loop body)

Normal loop execution:
Cycle: 1  2  3  4  5  6  7
Fetch:  ├─ Iter 1
Decode: ├─ Iter 1
Exec:   ├─ Iter 1
Fetch:       ├─ Iter 2
Decode:      ├─ Iter 2
...

With LSD (after detection, ~4 iterations):
Cycle: 1  2  3  4  5  6  7
Fetch:  ├─ (loop detected, LSD takes over)
Decode: ├─ (LSD supplies to execution)
Exec:   ├─ Iter 1
Exec:      ├─ Iter 2
Exec:         ├─ Iter 3
Exec:            ├─ Iter 4

LSD supplies: Zero fetch/decode latency, flat execution
Benefit: Loop not re-fetched/re-decoded on each iteration

MICRO-FUSION: Combining Operations
════════════════════════════════════════════════════════════

Without fusion (2 µops):
add rax, [rip+8]    → Load µop + ALU µop = 2 µops, uses 2 ports

With fusion (1 µop):
add rax, [rip+8]    → Fused µop (AGU + ALU) = 1 µop, uses 1 port

Throughput: 2 µops/cycle (without) vs 4+ µops/cycle (with fusion)

Patterns that fuse:
- ALU op + memory: add rax, [rip+8] → 1 µop
- Compare + branch: cmp eax, ebx; jne label → 1 µop (macro-fusion)

DEPENDENCY CHAINS vs PARALLEL PATHS
════════════════════════════════════════════════════════════

Serial dependency chain (low ILP):
Instr 1: a = x + y      (latency 1, result in reg a)
Instr 2: b = a * z      (latency 3, depends on a)
Instr 3: c = b + 1      (latency 1, depends on b)
Total: 1 + 3 + 1 = 5 cycles minimum

Critical path: 5 cycles
ILP: 1 (only 1 instruction can execute at any time due to dependencies)

Parallel paths (high ILP):
Instr 1: a = x + y      (latency 1)
Instr 2: b = m + n      (latency 1, independent)
Instr 3: c = p * q      (latency 3, independent)
Instr 4: d = a + b      (latency 1, depends on 1, 2)
Instr 5: e = c + d      (latency 1, depends on 3, 4)
Total: max(latencies along critical paths) = 3 + 1 = 4 cycles

But 3, 2, 1 can execute in parallel (cycle 1)
4 depends on 1, 2 (can start cycle 2)
5 depends on 3, 4 (can start cycle 3-4)

ILP: 3 (instructions 1, 2, 3 can execute in parallel)
Actual CPI on 6-wide: 1.4 (5 instrs / 3.6 cycles)
```

## 3. PERFORMANCE LENS

### ILP-to-IPC Translation

Maximum theoretical IPC is limited by port count, but achieved IPC is typically 40-60% of maximum due to:
1. **Dependency stalls** (cannot execute independent instructions)
2. **Memory latency** (loads cause ripple stalls)
3. **Branch mispredictions** (pipeline flushes)
4. **Port contention** (instruction mix creates bottleneck)

**Formula for achievable IPC**:

$$\text{IPC}_{\text{achieved}} = \frac{\text{ILP} \times \text{port\_count}}{(\text{port\_count} + \text{average\_port\_conflict\_stalls})}$$

**Example**:
- ILP = 3 (average 3 independent instructions)
- Port count = 6
- Port conflict stalls = 2 (due to instruction mix bottlenecking on load/store ports)
- IPC = (3 × 6) / (6 + 2) = 18 / 8 = 2.25

### Loop Unrolling Trade-Offs

**Benefits**:
1. **Increased ILP** (4×: 4 independent iterations)
2. **Reduced branch frequency** (4×: 1 branch per 4 iterations)
3. **Better instruction scheduling** (more room for compiler to reorder)

**Costs**:
1. **Code size increase** (4×: code 4× larger)
2. **Register pressure** (more live values)
3. **uop cache misses** (larger loop may not fit in 384-uop DSB)
4. **Instruction cache pressure** (larger code, fewer cache lines available)

**Optimal unroll factor**: Usually 2-4 for modern processors
- 2×: Small overhead, good ILP improvement
- 4×: Maximum practical ILP, but register pressure becomes significant
- 8×+: Code size penalties outweigh ILP benefits

### Software Pipelining Benefit

Software pipelining achieves best IPC by overlapping iterations:

$$\text{IPC}_{\text{pipelined}} = \frac{\text{instructions\_per\_iteration}}{\text{loop\_latency}} \approx \frac{\text{instructions}}{\text{longest\_dependency\_latency}}$$

**Example**:
- Loop body: 5 instructions
- Longest dependency latency: 4 cycles (load latency)
- Software-pipelined CPI: 5 / 4 = 1.25

**Comparison to unrolled loop**:
- Unrolled 4×: 1.75 cycles (20 instructions, 12 cycles for 4 iterations)
- Software-pipelined: 1.25 cycles (achieves better locality)

### uop Cache Impact

DSB miss rate directly impacts throughput:

$$\text{Effective throughput} = \text{DSB\_hit\_rate} \times 6 + \text{DSB\_miss\_rate} \times 1$$

**Example**:
- 90% DSB hit rate: 0.9 × 6 + 0.1 × 1 = 5.5 µops/cycle
- 80% DSB hit rate: 0.8 × 6 + 0.2 × 1 = 5.0 µops/cycle
- 50% DSB hit rate: 0.5 × 6 + 0.5 × 1 = 3.5 µops/cycle

A 50% DSB hit rate (one miss every 2 accesses) reduces throughput by 36%.

## 4. ANNOTATED CODE EXAMPLES

### Example 4a: Loop Unrolling and ILP Demonstration

```c
#include <stdio.h>
#include <time.h>

// Scenario: Array reduction (sum all elements)

// Version 1: No unrolling (scalar loop)
int sum_scalar(int* arr, int n) {
    int sum = 0;
    for (int i = 0; i < n; i++) {
        sum += arr[i];  // Load (latency 4) + Add (latency 1) = min 5 cycles
    }
    return sum;
}

// Version 2: 2× unrolled
int sum_unroll_2(int* arr, int n) {
    int sum = 0;
    for (int i = 0; i < n; i += 2) {
        sum += arr[i];      // Load + add (independent)
        sum += arr[i + 1];  // Loads can execute in parallel (2 load ports)
    }
    return sum;
}

// Version 3: 4× unrolled
int sum_unroll_4(int* arr, int n) {
    int sum = 0;
    for (int i = 0; i < n; i += 4) {
        sum += arr[i];      // 4 independent loads and adds
        sum += arr[i + 1];  // Execute: 2 loads per cycle (2 load ports)
        sum += arr[i + 2];  // Cycle 0: Load [0], [1]
        sum += arr[i + 3];  // Cycle 1: Load [2], [3]
                            // Cycle 2-3: Adds (accumulate)
    }
    return sum;
}

// Version 4: 8× unrolled
int sum_unroll_8(int* arr, int n) {
    int sum = 0;
    for (int i = 0; i < n; i += 8) {
        // 8 independent iterations, demonstrates maximum parallelism
        sum += arr[i];
        sum += arr[i + 1];
        sum += arr[i + 2];
        sum += arr[i + 3];
        sum += arr[i + 4];
        sum += arr[i + 5];
        sum += arr[i + 6];
        sum += arr[i + 7];
    }
    return sum;
}

// Benchmark
int main() {
    int arr[10000];
    for (int i = 0; i < 10000; i++) arr[i] = i;

    int iterations = 1000;

    // Benchmark scalar
    clock_t start = clock();
    int result = 0;
    for (int iter = 0; iter < iterations; iter++) {
        result = sum_scalar(arr, 10000);
    }
    clock_t time_scalar = clock() - start;

    // Benchmark 2× unroll
    start = clock();
    for (int iter = 0; iter < iterations; iter++) {
        result = sum_unroll_2(arr, 10000);
    }
    clock_t time_unroll_2 = clock() - start;

    // Benchmark 4× unroll
    start = clock();
    for (int iter = 0; iter < iterations; iter++) {
        result = sum_unroll_4(arr, 10000);
    }
    clock_t time_unroll_4 = clock() - start;

    // Benchmark 8× unroll
    start = clock();
    for (int iter = 0; iter < iterations; iter++) {
        result = sum_unroll_8(arr, 10000);
    }
    clock_t time_unroll_8 = clock() - start;

    printf("Scalar:    %ld cycles\n", time_scalar);
    printf("Unroll 2×: %ld cycles (%.2fx speedup)\n", time_unroll_2,
           (double)time_scalar / time_unroll_2);
    printf("Unroll 4×: %ld cycles (%.2fx speedup)\n", time_unroll_4,
           (double)time_scalar / time_unroll_4);
    printf("Unroll 8×: %ld cycles (%.2fx speedup)\n", time_unroll_8,
           (double)time_scalar / time_unroll_8);

    return 0;
}

// Expected results (Intel Skylake, 2.4 GHz):
// Scalar:     ~1000 ms (~2.4 billion cycles for 1000 iterations of 10000-element sum)
//             CPI ≈ 5.0 (load latency dominates)
// Unroll 2×:  ~550 ms (2.2× speedup, CPI ≈ 2.5)
// Unroll 4×:  ~400 ms (2.5× speedup, CPI ≈ 2.0)
// Unroll 8×:  ~380 ms (2.6× speedup, CPI ≈ 1.9)

// Line-by-line explanation:
// Scalar: Each iteration depends on previous (sum = sum + arr[i])
//         Load latency 4 cycles minimum per iteration
//
// Unroll 2×: Two independent loads (2 load ports) in parallel
//           Loads can overlap with adds from previous iteration
//           Effective latency: ~2.5 cycles per original iteration
//
// Unroll 4×: Four independent loads, 2 per cycle (Skylake's 2 load ports)
//           After initial 4-cycle load latency, adds overlap
//           Steady state: 2 loads/cycle + 1 add/cycle = 1 cycle per original iteration
//           Actual: ~2 cycles per original iteration (accounting for pipeline fill)
//
// Unroll 8×: Marginal improvement over 4× (register pressure, DSB pressure)
//           Most benefit already captured by 4× unroll
```

### Example 4b: Software Pipelining Illustration

```c
// Software pipelining: Compiler rearranges iterations to hide latency

// Original loop (programmer writes):
void original_loop(float* arr, float* result, int n) {
    for (int i = 0; i < n; i++) {
        float x = arr[i];       // Load: 4-cycle latency
        result[i] = x * 2.0;    // Multiply: 5-cycle latency (FP multiply on Skylake)
    }
}

// Compiler transforms to software-pipelined version:
void software_pipelined_loop(float* arr, float* result, int n) {
    // Prologue: Load first iteration
    float x0 = arr[0];          // Cycle 0: Load

    // Main loop: Load next while computing previous
    for (int i = 1; i < n; i++) {
        float x_next = arr[i];  // Cycle i: Load iteration i
        result[i - 1] = x0 * 2.0; // Cycle i: Multiply from iteration i-1
        x0 = x_next;            // Next iteration uses x_next
    }

    // Epilogue: Finish last iteration
    result[n - 1] = x0 * 2.0;   // Cycle n: Final multiply
}

// Pipeline execution timeline:
// Original (serialized):
// Cycle:  0  1  2  3  4  5  6  7  8  9  10
// Iter 0: LD [4 cycles] MUL [5 cycles]
// Iter 1:                       LD [4 cycles] MUL [5 cycles]
// Iter 2:                                          LD [4 cycles] MUL

// Software-pipelined (overlapped):
// Cycle:  0  1  2  3  4  5  6  7  8  9  10
// Iter 0: LD
// Iter 1:    LD          MUL
// Iter 2:       LD          MUL
// Iter 3:          LD          MUL
// Iter 4:             LD          MUL

// Result: Loads and multiplies interleaved
// CPI improvement: From 5 cycles per iteration to 1 cycle per iteration

// Key insight:
// - Original: Load (4 cycles) + Multiply (5 cycles) = 9 cycles minimum
//   Actually: Must wait for load result (4 cycles), then multiply (5 cycles)
//   Total: 4 + 5 = 9 cycles
//
// - Pipelined: Overlap load latency with multiply from previous iteration
//   Load iteration N: cycle N
//   Multiply iteration N-1: cycle N+4 (when load data available)
//   CPI: 1 cycle per iteration (at steady state)
//
// - Speedup: 9× (9 cycles vs 1 cycle per iteration)

#include <stdio.h>

void measure_pipelining(float* arr, float* result, int n, int iterations) {
    clock_t start, end;

    // Original loop
    start = clock();
    for (int iter = 0; iter < iterations; iter++) {
        for (int i = 0; i < n; i++) {
            float x = arr[i];
            result[i] = x * 2.0;
        }
    }
    end = clock();
    double time_original = (double)(end - start) / CLOCKS_PER_SEC;

    // Software-pipelined loop
    start = clock();
    for (int iter = 0; iter < iterations; iter++) {
        float x0 = arr[0];
        for (int i = 1; i < n; i++) {
            float x_next = arr[i];
            result[i - 1] = x0 * 2.0;
            x0 = x_next;
        }
        result[n - 1] = x0 * 2.0;
    }
    end = clock();
    double time_pipelined = (double)(end - start) / CLOCKS_PER_SEC;

    printf("Original:       %.6f seconds\n", time_original);
    printf("Software-piped: %.6f seconds\n", time_pipelined);
    printf("Speedup: %.2fx\n", time_original / time_pipelined);
}
```

### Example 4c: uop Cache (DSB) Impact

```c
// DSB impact: Small loop (fits in DSB) vs large loop (DSB miss)

#include <stdio.h>
#include <time.h>

// Small loop: Fits in uop cache (DSB)
void small_loop_dsb_hit(int* arr, int n) {
    int sum = 0;
    for (int i = 0; i < n; i++) {
        sum += arr[i];  // ~5 instructions total (loop overhead)
                        // Loop body ~20 bytes, fits in DSB
    }
}

// Large loop: Doesn't fit in DSB (DSB miss, fall back to MITE decode)
void large_loop_dsb_miss(int* arr, int n) {
    int sum = 0;
    for (int i = 0; i < n; i++) {
        // Simulate large loop body (not just the sum)
        int x = arr[i];
        int y = x * 2;
        int z = y + 3;
        int w = z >> 1;
        // Loop body: 40+ bytes, doesn't fit in DSB
        // Falls back to MITE decode (1 µop/cycle, 1-2 cycle latency)
        // (Obviously, real loop would have different pattern, but principle same)
    }
}

int main() {
    int arr[10000];
    for (int i = 0; i < 10000; i++) arr[i] = i;

    // Benchmark small loop (DSB hit)
    clock_t start = clock();
    for (int iter = 0; iter < 100000; iter++) {
        small_loop_dsb_hit(arr, 100);
    }
    clock_t time_small = clock() - start;

    // Benchmark large loop (DSB miss)
    start = clock();
    for (int iter = 0; iter < 100000; iter++) {
        large_loop_dsb_miss(arr, 100);
    }
    clock_t time_large = clock() - start;

    printf("Small loop (DSB hit):  %ld cycles\n", time_small);
    printf("Large loop (DSB miss): %ld cycles\n", time_large);
    printf("Slowdown (DSB miss):   %.2fx\n", (double)time_large / time_small);

    // Expected: Large loop 20-30% slower due to decode latency penalty
    // DSB miss forces restart from MITE (legacy decode), adding 1-2 cycle latency
}

// DSB characteristics (Skylake):
// Entries: 64 lines
// µops per entry: 6
// Total capacity: 384 µops
// Hit rate: 90% (typical code)
// Miss rate: 10% (cache misses, large functions, indirect jumps)
//
// Performance impact:
// Hit: 6 µops/cycle (decode free, µops cached)
// Miss: 1 µop/cycle (legacy decode latency)
//
// Average throughput:
// 90% × 6 + 10% × 1 = 5.5 µops/cycle (with 90% DSB hit)
// 50% × 6 + 50% × 1 = 3.5 µops/cycle (with 50% DSB hit)
//
// Code that stays in DSB: Fast
// Code that misses DSB: 36% throughput penalty (5.5 vs 3.5)
```

### Example 4d: Execution Port Pressure and Bottlenecks

```c
// Demonstrate execution port bottlenecks

#include <stdio.h>
#include <time.h>

// Code 1: ALU-heavy (uses ports 0, 1, 5, 6)
void alu_heavy(int* arr, int n) {
    int a = 0, b = 0;
    for (int i = 0; i < n; i++) {
        a = a + 1;       // Port 0
        b = b * 2;       // Port 1 (multiply)
        a = a ^ 0xff;    // Port 5
        b = b & 0xf;     // Port 6
        // 4 ALU instructions per cycle (max 4 ALU ports)
        // No load/store, ports 2, 3, 4 unused
    }
}

// Code 2: Load-heavy (uses ports 2, 3)
void load_heavy(int* arr, int n) {
    int sum = 0;
    for (int i = 0; i < n; i += 2) {
        int x = arr[i];      // Port 2 (load)
        int y = arr[i + 1];  // Port 3 (load, parallel)
        sum += x + y;        // Port 0 (add)
        // 2 loads per cycle (max 2 load ports)
        // Limited by load ports, ALU ports underutilized
    }
}

// Code 3: Mixed (balanced use of ALU and load ports)
void balanced(int* arr, int n) {
    int a = 0, sum = 0;
    for (int i = 0; i < n; i += 2) {
        int x = arr[i];      // Port 2 (load)
        int y = arr[i + 1];  // Port 3 (load)
        a = a + 1;           // Port 0 (ALU)
        sum += x + y;        // Port 1 (ALU)
        a = a * y;           // Port 5 (ALU)
        // 2 loads + 3 ALU per iteration
        // Better utilization (4 out of 6 ports used)
    }
}

int main() {
    int arr[10000];
    for (int i = 0; i < 10000; i++) arr[i] = i;

    clock_t start, end;

    // Benchmark ALU-heavy
    start = clock();
    for (int iter = 0; iter < 100000; iter++) {
        alu_heavy(arr, 100);
    }
    end = clock();
    double time_alu = (double)(end - start) / CLOCKS_PER_SEC;

    // Benchmark load-heavy
    start = clock();
    for (int iter = 0; iter < 100000; iter++) {
        load_heavy(arr, 100);
    }
    end = clock();
    double time_load = (double)(end - start) / CLOCKS_PER_SEC;

    // Benchmark balanced
    start = clock();
    for (int iter = 0; iter < 100000; iter++) {
        balanced(arr, 100);
    }
    end = clock();
    double time_balanced = (double)(end - start) / CLOCKS_PER_SEC;

    printf("ALU-heavy:  %.6f seconds\n", time_alu);
    printf("Load-heavy: %.6f seconds\n", time_load);
    printf("Balanced:   %.6f seconds\n", time_balanced);

    // Expected results:
    // ALU-heavy: Fast (IPC 4, utilizing 4 ALU ports)
    // Load-heavy: Slow (IPC 2, limited by 2 load ports, loads have 4-cycle latency)
    // Balanced: Medium (IPC 3-4, better utilization than pure load)
    return 0;
}

// Port utilization analysis:
// ALU-heavy: Ports 0, 1, 5, 6 utilized; ports 2, 3, 4 idle
//   - IPC = 4 (limited by ALU port count)
//   - No memory latency stalls
//
// Load-heavy: Ports 2, 3 (loads) utilized; ports 0, 1, 5, 6 underutilized
//   - IPC = 2 (limited by load port count and load latency)
//   - Each load causes 4-cycle latency, ripples to subsequent adds
//
// Balanced: Mix of ports, better utilization
//   - IPC = 3-4 (load latency partially hidden by ALU work)
//   - Ports 0, 1, 2, 3, 5 utilized; port 6 occasional
```

## 5. EXPERT INSIGHT

### Insight 1: ILP is Often Overestimated in Compiler Metrics

Compiler passes (e.g., `llvm-mca`) report theoretical ILP of 3-4 for typical code, but achieved IPC is often 1.5-2.0. The gap comes from:

1. **Register pressure**: Compiler's register allocation limits available values
2. **Memory latency**: Loads introduce 4-60 cycle latencies that serialize computation
3. **Branch misprediction**: Flushes pipelines, resets per-instruction progress
4. **Port contention**: Instruction mix creates bottlenecks despite high theoretical ILP

**Implication**: Simply unrolling loops (often recommended as "first optimization") may not improve performance if memory latency or register pressure is limiting. Measure before optimizing.

### Insight 2: Multiply is Free (in Throughput, Not Latency)

Integer multiply on Skylake: latency 3 cycles, throughput 1 per cycle. This seems contradictory but makes sense with deep pipelining:

```
Port 1 (multiplier):
  Cycle 0: Instr 0 (mult) enters, exits stage 1
  Cycle 1: Instr 0 stage 2, Instr 1 (new mult) enters stage 1
  Cycle 2: Instr 0 stage 3, Instr 1 stage 2, Instr 2 (new mult) enters stage 1
  Cycle 3: Instr 0 exits (result ready), Instr 1 stage 3, Instr 2 stage 2
           Instr 3 (new mult) can enter stage 1

Result: Every cycle can issue a new multiply (throughput 1), but result takes 3 cycles (latency 3).
```

**Implication**: Code with multiply-heavy arithmetic can achieve high IPC (sustaining 1 multiply per cycle), even though each individual multiply has 3-cycle latency. This is only true if multiplies are independent.

### Insight 3: Dependency Chain Length Determines Maximum Sustainable IPC

**Recurrence latency** = latency of the longest dependency chain in a loop:

```c
for (i = 0; i < N; i++) {
    sum = sum + arr[i];  // Latency: load (4) + add (1) = 5
}

Recurrence latency = 5 cycles
Maximum sustainable IPC = 1 / 5 = 0.2 (not 1 instruction per cycle!)
```

Modern compilers and hardware cannot parallelize beyond recurrence latency unless multiple independent accumulators are introduced:

```c
for (i = 0; i < N; i += 4) {
    sum1 += arr[i];      // Accum 1
    sum2 += arr[i+1];    // Accum 2 (independent)
    sum3 += arr[i+2];    // Accum 3 (independent)
    sum4 += arr[i+3];    // Accum 4 (independent)
}

Now recurrence latency = 5 / 4 = 1.25 cycles (each accum has latency 5, but 4 in parallel)
Maximum sustainable IPC = 4 (if sufficient ports) or 2 (limited by load ports)
```

**Implication**: Recurrence latency is a fundamental limit, not bypass-able with better prediction or wider pipelines.

### Insight 4: DSB Pressure is Invisible But Real

Most engineers ignore uop cache, assuming decode is "free." In reality:

- DSB miss: 1 µop/cycle (legacy decode)
- DSB hit: 6 µops/cycle (cached)

A 50% DSB miss rate (every other fetch from MITE) cuts throughput from 6 to 3.5 µops/cycle (42% reduction).

**How to trigger DSB misses**:
1. Large functions (> 384 µops)
2. Nested loops with many iterations (uop cache overwritten by inner loop)
3. Indirect jumps (BTB miss, restart fetch from different address)

**Detection**: Use VTune or perf to measure DSB hit rate. If < 80%, investigate code structure.

### Insight 5: Loop Stream Detector (LSD) is Underutilized

LSD is powerful but fragile. Activation requires:
1. Backward branch (loop detected)
2. Loop iteration count < 64
3. Loop body < 128 bytes

Many developers don't know LSD exists, missing 5-10% free speedup on small loops.

**To exploit LSD**:
- Keep loop bodies small (< 128 bytes)
- Avoid large data structures in loop (pressure on uop cache)
- Unroll loops to increase iteration count only if necessary

**Verification**: Use `perf stat -e lsd_overflow` to check if LSD is saturating (missing optimizations).

## 6. BENCHMARK / MEASUREMENT

### Measurement 1: ILP Extraction

```bash
# Use LLVM-MCA to estimate theoretical ILP
llvm-mca -march=skylake -iterations=100 < your_code.ll

# Output: IPC, bottleneck analysis
# IPC: 3.2 (theoretical max, if no memory latency, perfect prediction)
#
# Compare to perf on real hardware:
# perf stat -e cycles,instructions ./program
# IPC: 1.8 (achieved, bottlenecks: memory latency, port contention)
```

### Measurement 2: uop Cache (DSB) Hit Rate

```bash
# Linux perf: measure DSB hits/misses
perf stat -e dsb_2_cycles,mite_uops,dsb_uops ./program

# Calculate DSB hit rate:
# DSB_hit_rate = dsb_uops / (dsb_uops + mite_uops)
#
# If DSB_hit_rate < 80%, investigate loop/function sizes
```

### Measurement 3: Port Utilization

```bash
# Intel VTune (proprietary, most accurate):
vtune -c general-exploration ./program
# Provides port utilization breakdown (Port 0, 1, 2, 3, 4, 5, 6, 7)

# Linux perf (limited):
perf stat -e port_utilization.0c,port_utilization.1c,port_utilization.2c, \
            port_utilization.3c,port_utilization.4c,port_utilization.5c, \
            port_utilization.6c ./program
```

### Measurement 4: Recurrence Latency

```c
#include <stdio.h>
#include <time.h>

// Measure recurrence latency (dependency chain limit)
void measure_recurrence_latency() {
    const int iterations = 100000;
    const int n = 10000;
    int arr[n];
    for (int i = 0; i < n; i++) arr[i] = i;

    // Single accumulator (high recurrence latency)
    clock_t start = clock();
    int sum = 0;
    for (int iter = 0; iter < iterations; iter++) {
        for (int i = 0; i < n; i++) {
            sum += arr[i];  // Latency: load (4) + add (1) = 5
        }
    }
    clock_t time_single = clock() - start;

    // Multiple accumulators (low recurrence latency)
    start = clock();
    int sum1 = 0, sum2 = 0, sum3 = 0, sum4 = 0;
    for (int iter = 0; iter < iterations; iter++) {
        for (int i = 0; i < n; i += 4) {
            sum1 += arr[i];      // Accum 1
            sum2 += arr[i + 1];  // Accum 2 (parallel)
            sum3 += arr[i + 2];  // Accum 3 (parallel)
            sum4 += arr[i + 3];  // Accum 4 (parallel)
        }
    }
    clock_t time_multi = clock() - start;

    printf("Single accum: %ld cycles (CPI ≈ 5)\n", time_single);
    printf("Multi accum:  %ld cycles (CPI ≈ 1.5)\n", time_multi);
    printf("Speedup: %.2fx\n", (double)time_single / time_multi);
}
```

## 7. ML SYSTEMS RELEVANCE

### Relevance 1: Tensor Contraction (Matrix Multiplication) Optimization

GEMM (General Matrix Multiply) is the foundation of deep learning. Optimization requires:

1. **Loop tiling**: Break into cache-friendly blocks
2. **Unrolling**: Extract parallelism within each block
3. **Port utilization**: Balance ALU and load operations

```c
// Optimized GEMM for Skylake
void gemm_optimized(float* A, float* B, float* C, int N, int M, int K) {
    for (int ii = 0; ii < N; ii += 64) {           // Tile 1
        for (int jj = 0; jj < M; jj += 64) {       // Tile 2
            for (int kk = 0; kk < K; kk += 8) {    // Tile 3
                // Inner loop: 4×4 unroll to exploit 4 ALU ports
                for (int i = ii; i < ii + 64; i += 4) {
                    for (int j = jj; j < jj + 64; j += 4) {
                        // 16 independent multiply-accumulates
                        // Store in 16 different registers (physical regs available)
                        // Execute on 4 ALU ports (1 per cycle, overlapped over 4 cycles)
                        for (int k = kk; k < kk + 8; k++) {
                            C[i + 0][j + 0] += A[i + 0][k] * B[k][j + 0];
                            C[i + 0][j + 1] += A[i + 0][k] * B[k][j + 1];
                            C[i + 0][j + 2] += A[i + 0][k] * B[k][j + 2];
                            C[i + 0][j + 3] += A[i + 0][k] * B[k][j + 3];
                            // ... continue for i+1, i+2, i+3
                        }
                    }
                }
            }
        }
    }
}

// ILP analysis:
// - 16 independent accumulators: high ILP (16 parallel multiply-adds)
// - 4 ALU ports on Skylake: limits to 4 multiplies per cycle
// - Multiply latency 5 cycles: each accumulator is ready every 5 cycles
// - Unroll factor 4: 4 multiply-accums per cycle × 4 = 16 instructions/cycle potential
// - Achieved: ~3-4 IPC (memory latency, port contention)
```

### Relevance 2: Transformer Attention Computation

Attention computes Q · K^T (matrix multiply-reduce):

```python
def attention(Q, K, V, seq_len):
    scores = []
    for q_idx in range(seq_len):
        score = 0.0
        for k_idx in range(seq_len):
            score += Q[q_idx] · K[k_idx]  # Dot product (dependency chain)
        scores.append(score)
    return softmax(scores)
```

**ILP analysis**:
- Dot product: sum of element-wise products (scalar = sum of K products)
- Dependency chain: each product depends on previous sum
- Recurrence latency: (load + multiply + add) × K = ~8 × K cycles
- For K = 64: 512 cycles minimum per dot product (without parallelism)

**Optimization: Unroll and parallelize dot product**:

```python
def attention_optimized(Q, K, V, seq_len):
    for q_idx in range(seq_len):
        # Unroll dot product (4 parallel accumulators)
        acc0 = acc1 = acc2 = acc3 = 0.0
        for k_idx in range(0, seq_len, 4):
            acc0 += Q[q_idx] · K[k_idx]      # Load pair 0
            acc1 += Q[q_idx] · K[k_idx + 1]  # Load pair 1 (parallel)
            acc2 += Q[q_idx] · K[k_idx + 2]  # Load pair 2 (parallel)
            acc3 += Q[q_idx] · K[k_idx + 3]  # Load pair 3 (parallel)
        score = acc0 + acc1 + acc2 + acc3    # Final reduction
```

This parallelism is critical for inference latency: without it, dot product is sequential; with it, 4× speedup (or 2× limited by load ports).

### Relevance 3: Softmax and Exp Computation

Softmax requires exp(x), which has high latency (10+ cycles on Skylake):

```c
float softmax_scalar(float* logits, int n) {
    // Compute exp for each logit
    for (int i = 0; i < n; i++) {
        logits[i] = expf(logits[i]);  // 10+ cycles per exp (serial)
    }
}
```

**Optimization: Unroll and parallelize exp calls**:

```c
float softmax_vectorized(float* logits, int n) {
    // Unroll 4×: 4 independent exp calls in parallel
    for (int i = 0; i < n; i += 4) {
        logits[i]     = expf(logits[i]);
        logits[i + 1] = expf(logits[i + 1]);  // Can execute in parallel
        logits[i + 2] = expf(logits[i + 2]);
        logits[i + 3] = expf(logits[i + 3]);
    }
}

// Performance:
// Scalar: 10 cycles per exp, n iterations = 10n cycles
// Vectorized: 4 exp in parallel, 10 cycles for first, pipelined every cycle after
//            = 10 + n cycles (nearly 10× speedup)
```

### Relevance 4: Attention Softmax and Reduction (Multi-Accumulator Pattern)

Softmax reduction (normalization) is a critical bottleneck:

```python
# Sum all exp values (normalization denominator)
sum_exp = 0.0
for i in range(seq_len):
    sum_exp += exp_logits[i]
```

This is a classic reduction with recurrence latency. Optimization:

```python
# 4-accumulator reduction (unroll 4×)
sum0 = sum1 = sum2 = sum3 = 0.0
for i in range(0, seq_len, 4):
    sum0 += exp_logits[i]
    sum1 += exp_logits[i + 1]
    sum2 += exp_logits[i + 2]
    sum3 += exp_logits[i + 3]
final_sum = sum0 + sum1 + sum2 + sum3
```

This 4× unroll reduces recurrence latency from 5 cycles to ~1.25 cycles (per original element), enabling 4 parallel loads and adds.

### Relevance 5: Quantization Kernels and Branchless Computation

Quantization to int8 requires clamping:

```c
int8_t quantize(float x) {
    if (x < 0) return 0;
    if (x > 127) return 127;
    return (int8_t)x;
}
```

This has 2 branches, both potentially mispredicted. Branchless alternative:

```c
int8_t quantize_branchless(float x) {
    // Clamp without branches: max(0, min(127, x))
    int8_t result = (int8_t)x;
    if (result < 0) result = 0;     // CMOV (conditional move, not branch)
    if (result > 127) result = 127; // CMOV
    return result;
}

// Or using bitwise ops:
int8_t quantize_bitwise(float x) {
    int clamped = (int)x;
    // Clamp: if negative, set to 0; if > 127, set to 127
    int mask_neg = (clamped < 0) ? -1 : 0;
    int mask_pos = (clamped > 127) ? -1 : 0;
    clamped = (clamped & ~mask_neg) | (0 & mask_neg);        // Clamp to 0
    clamped = (clamped & ~mask_pos) | (127 & mask_pos);      // Clamp to 127
    return (int8_t)clamped;
}
```

Branchless version: No branch mispredictions, latency = CMOV latency (1 cycle), better for batched quantization kernels.

## 8. PhD QUALIFIER QUESTIONS

**Question 8.1**: Define instruction-level parallelism (ILP) and explain how it differs from instruction-per-cycle (IPC). Why can ILP be high (3-4) while IPC is low (1-2)? Provide a concrete code example.

**Expected answer structure**:
- ILP: Average number of independent instructions that could execute in parallel if unlimited ports/latency
- IPC: Actual instructions completed per cycle, limited by ports, latency, dependencies
- ILP ≠ IPC because:
  1. Port contention (limited execution units)
  2. Memory latency (long dependency chains)
  3. Branch mispredictions (pipeline flushes)
  4. Register pressure (limited architectural registers)
- Example: Loop with ILP 4 (4 independent iterations), but load port limited to 2 → IPC capped at 2

---

**Question 8.2**: Explain loop unrolling and its impact on performance. Derive the speedup formula in terms of unroll factor, load latency, and port count. Why is there a limit to beneficial unroll factors?

**Expected answer structure**:
- Unrolling increases parallelism: 4× unroll = 4 independent iterations
- Speedup = (original CPI) / (unrolled CPI)
- Original CPI = load_latency / 1 (serialized iterations)
- Unrolled CPI = load_latency / min(unroll_factor, port_count)
- Speedup = min(unroll_factor, port_count)
- Limits: Register pressure, uop cache pressure, code size
- Optimal: 2-4× (diminishing returns beyond)

---

**Question 8.3**: Explain the uop cache (DSB) and its impact on throughput. What is the difference between DSB hits and misses? Calculate the average throughput if DSB hit rate is 80%.

**Expected answer structure**:
- DSB: Cache of decoded instructions (384 µops capacity)
- DSB hit: 6 µops/cycle (cached, no decode latency)
- DSB miss: 1 µop/cycle (legacy decode from MITE)
- Average throughput: 0.8 × 6 + 0.2 × 1 = 5.0 µops/cycle
- Conditions for DSB miss: Function size > 384 µops, indirect jumps, self-modifying code
- Optimization: Keep loops small, avoid large function bodies, minimize DSB evictions

---

**Question 8.4**: Explain the relationship between latency and throughput for instruction classes (ALU, multiply, load, divide). Provide examples from Skylake. Why can a multiply with 3-cycle latency sustain 1 per cycle throughput?

**Expected answer structure**:
- Latency: Cycles from start to result availability
- Throughput: Maximum instructions per cycle
- Examples (Skylake):
  - Add: 1-cycle latency, 1 per cycle throughput (1 ALU port)
  - Multiply: 3-cycle latency, 1 per cycle throughput (deeply pipelined multiplier)
  - Load: 4-cycle latency (L1), 2 per cycle throughput (2 load ports)
  - Divide: 12-40 cycle latency, 1 per ~40 cycles throughput (sequential divider)
- Multiply can sustain 1/cycle because:
  - Pipelined into 3 internal stages
  - New multiply can start each cycle, previous 2 still in progress
  - Result available after 3 stages (3 latency), but new multiply already issuing

---

**Question 8.5**: Explain software pipelining and how it hides memory latency. Provide a code example showing the transformation from original loop to software-pipelined version. Calculate the CPI improvement.

**Expected answer structure**:
- Software pipelining: Overlap iterations so earlier iterations' results are ready when later iterations need them
- Example: Load (4-cycle latency) + Use in next iteration
  - Original: Load, [4 cycles], Use, next Load (5 cycles per iteration)
  - Pipelined: Load, Load, [4 cycles], Use (1 cycle per iteration once warmed up)
- Transformation:
  1. Prologue: Load first iteration
  2. Main loop: Load next while using previous
  3. Epilogue: Use last iteration
- CPI improvement: 5 cycles → 1 cycle per iteration (5× speedup)
- Requires: Compiler support (automatic or pragma-directed)

## 9. READING LIST

1. **Hennessy, J. L., & Patterson, D. A.** (2017). *Computer Architecture: A Quantitative Approach* (6th ed.). Chapter 3.1-3.3: "Fundamentals of Computer Design" and "Instruction-Level Parallelism."
   - **Exact sections**: 3.1 (performance metrics), 3.2 (ILP), 3.3 (superscalar techniques)
   - Provides formulas for IPC, latency, throughput

2. **Mutlu, O.** (2017-2023). Carnegie Mellon 18-447 Lecture Series: "Instruction-Level Parallelism" (Lectures 15-18).
   - **Topics**: Loop unrolling, software pipelining, superscalar execution, port utilization
   - **Material**: Detailed examples, performance analysis, trade-offs

3. **Agner Fog.** *Instruction Tables: Lists of Instruction Latencies, Throughputs and Micro-operation Breakdowns for Intel, AMD and VIA CPUs* (2023).
   - **Sections**: Complete latency/throughput tables for all instruction types per microarchitecture
   - Skylake, Ice Lake, SPR, Zen 4 specifications
   - **URL**: https://www.agner.org/optimize/instruction_tables.pdf

4. **Agner Fog.** *Microarchitecture of Intel, AMD, and VIA CPUs: An Optimization Guide* (2023).
   - **Sections**: Chapter 2 (execution ports), Chapter 3 (instruction scheduling), Chapter 4 (loop unrolling)
   - **URL**: https://www.agner.org/optimize/microarchitecture.pdf

5. **Intel 64 and IA-32 Architectures Optimization Reference Manual** (2023).
   - **Sections**: Chapter 2 (microarchitecture), Chapter 3 (optimization techniques)
   - Specifies port counts, execution unit capabilities, throughput limits
   - **URL**: https://www.intel.com/content/dam/www/public/us/en/documents/manuals/64-ia-32-architectures-optimization-manual.pdf

6. **Massalin, H.** (1992). "Superoptimizer: A Look at the Smallest Program." *Proceedings of the 2nd International Conference on Architectural Support for Programming Languages and Operating Systems*, 122-126.
   - Early work on instruction scheduling and optimization
   - Historical context for modern compiler techniques

7. **Blume, R., & Gao, G. R.** (1993). "Software Pipelining, Modulo Scheduling, and Machines." Technical Report SECS-92-21, University of Delaware.
   - Definitive paper on software pipelining techniques
   - Algorithms for automatic modulo scheduling

---

**Module 8 Total Lines**: 1342 (comprehensive ILP and superscalar execution coverage)

