# Module 1: Hexagon SoC Architecture Internals

**PhD-Level Reference Document**
**Target Audience:** Advanced engineers with C/C++ and computer architecture background
**Prerequisite Knowledge:** VLIW processors, cache hierarchies, memory systems, DSP concepts

---

## Table of Contents

1. [Full Hexagon Processor Anatomy](#1-full-hexagon-processor-anatomy)
2. [Hardware Threading Model](#2-hardware-threading-model)
3. [Memory Hierarchy](#3-memory-hierarchy)
4. [The Hexagon ISA](#4-the-hexagon-isa)
5. [Hexagon NPU within Snapdragon SoC](#5-hexagon-npu-within-snapdragon-soc)
6. [Generational Differences (v65-v75)](#6-generational-differences-v65-v75)
7. [Self-Assessment Questions](#7-self-assessment-questions)

---

## 1. Full Hexagon Processor Anatomy

### 1.1 Processor Core Overview

The Qualcomm Hexagon processor is a 32-bit very long instruction word (VLIW) digital signal processor with specialized vector and tensor acceleration units. The microarchitecture consists of multiple distinct execution pipelines and specialized compute engines operating in parallel under a unified instruction dispatch system.

#### Core Components:

**Scalar Execution Engine (32-bit integer/floating-point)**
- 6-stage integer pipeline
- Separate floating-point pipeline (IEEE 754 compliant)
- ALU, multiplier, shifter, logic units

**HVX (Hexagon Vector eXtensions)**
- 128-bit or 256-bit SIMD vector unit (generation dependent)
- Dual-port access to unified L2 cache
- Dedicated vector registers (32/64 128-bit registers)

**HTA (Hexagon Tensor Accelerator) / HMX (Matrix eXtensions)**
- Introduced in v68 and later
- Hardware acceleration for matrix operations
- INT8/FP16/TF32 support

### 1.2 Scalar Core Pipeline Architecture

The scalar execution core implements a 6-stage pipeline with support for out-of-order execution within packet boundaries.

```
Stage 1: FETCH     - Fetch 128-bit packet from I-cache
Stage 2: DECODE    - Decode up to 4 instructions, check dependencies
Stage 3: DISPATCH  - Assign instructions to execution units
Stage 4: EXECUTE   - Integer/logic operations, address generation
Stage 5: MEMORY    - L1 cache access
Stage 6: WRITEBACK - Register file update
```

**Detailed Pipeline Stages:**

| Stage | Operation | Duration | Notes |
|-------|-----------|----------|-------|
| FETCH | I-cache access, I-TLB translation | 1 cycle | 128-bit aligned packets only |
| DECODE | Instruction parsing, immediate extraction | 1 cycle | Parallel decoding of 4 instructions |
| DISPATCH | Slot assignment, register renaming prep | 1 cycle | Resolves instruction class conflicts |
| EXECUTE | Computation, branch resolution | 1-3 cycles | Variable latency (MUL takes 2-3) |
| MEMORY | L1 D-cache or LSU operations | 2-3 cycles | Load latency = 2 cycles hit |
| WRITEBACK | Register file commit | 1 cycle | Synchronized across threads |

#### ASCII Block Diagram of Scalar Pipeline:

```
┌─────────────┐
│   I-Cache   │ (L1 Instruction Cache, 32 KB, 4-way)
│   (128b)    │
└──────┬──────┘
       │
       ▼
   ┌───────────┐
   │  FETCH    │ Stage 1
   │  128-bit  │
   └─────┬─────┘
         │
         ▼
   ┌──────────────────┐
   │  DECODE (4-way)  │ Stage 2
   │ Parallel decode  │
   └─────┬────────────┘
         │
         ▼
   ┌──────────────────────────────┐
   │  DISPATCH & RESOURCE ALLOC    │ Stage 3
   │  • Check dependencies         │
   │  • Assign to execution units  │
   │  • Register renaming          │
   └─────┬────────────────────────┘
         │
    ┌────┴────┬──────┬──────┬──────┐
    │          │      │      │      │
    ▼          ▼      ▼      ▼      ▼
  ┌───┐    ┌──────┐ ┌───┐ ┌────┐ ┌────┐
  │ALU│    │MULT  │ │LSU│ │HVX │ │BR  │ Execution Units
  │   │    │(2-3) │ │   │ │    │ │PRED│
  └─┬─┘    └──┬───┘ └─┬─┘ └─┬──┘ └─┬──┘
    │         │      │     │      │
    ▼         ▼      ▼     ▼      ▼
   ┌─────────────────────────────────┐
   │  MEMORY (Stage 5)               │
   │  D-Cache Access, LSU Operations │
   └─────────────────────────────────┘
         │
         ▼
   ┌──────────────────┐
   │  WRITEBACK       │ Stage 6
   │  Register Update │
   └──────────────────┘
```

### 1.3 HVX (Hexagon Vector eXtensions) Architecture

HVX provides SIMD acceleration for digital signal processing tasks. The unit operates at the same clock as the scalar core but with independent execution pipelines.

**HVX Specifications by Generation:**

| Generation | Width | Registers | Peak Throughput | Ops/Cycle |
|------------|-------|-----------|-----------------|-----------|
| v65/v66    | 128b  | 32×128b   | 256 bits/cycle  | 8 FP32 ops |
| v68+       | 256b  | 32×256b   | 512 bits/cycle  | 16 FP32 ops |

**HVX Register File Organization:**

```
Vector Register File (v68+, 256-bit width):
┌──────────────────────────────────────────────────┐
│                                                  │
│  V0:V1 (512b pair for double-width ops)        │
│  ├─ v0[255:0]  ┌─────────────────────────┐     │
│  └─ v1[255:0]  │                         │     │
│                │  256-bit Vector Data    │     │
│  V2:V3         │                         │     │
│  ...           │  Can be interpreted as: │     │
│  V30:V31       │  • 32×8b (i8 SIMD)      │     │
│                │  • 16×16b (i16 SIMD)    │     │
│                │  • 8×32b (i32 SIMD)     │     │
│                │  • 4×64b (i64 SIMD)     │     │
│                └─────────────────────────┘     │
│                                                  │
│  Qn (Predicate registers, 256b mask)           │
│  V0-V31: General-purpose vector registers      │
│                                                  │
└──────────────────────────────────────────────────┘
```

**HVX Execution Pipelines:**

The HVX unit contains multiple independent execution paths:

1. **Vector ALU Path**: Arithmetic, logic, shift operations (1 cycle latency)
2. **Vector Multiply Path**: 256-bit multiply-accumulate (3 cycle latency)
3. **Vector Memory Path**: Cache access, streaming loads (2-3 cycle latency)
4. **Vector Permute Path**: Shuffles, rotations, interleaves (1 cycle latency)

**HVX Instruction Encoding (Dual-Issue):**

```
HVX allows dual-issue of certain instruction types:
┌─ Vector ALU + Vector Memory
├─ Vector ALU + Vector Multiply
├─ Vector Memory + Vector Memory (dual-load)
└─ Vector Multiply + Vector Multiply (not always)

Example: Load + Arithmetic in parallel
  v0.uh = vmem(r0+#0):128t    // Load from memory
  v1 = vadd(v2, v3)           // Can execute simultaneously
```

### 1.4 HTA (Hexagon Tensor Accelerator) / HMX Architecture

The Hexagon Tensor Accelerator (HTA), also known as HMX in later generations, provides dedicated hardware for matrix multiplication and tensor operations critical to deep learning inference.

**HMX Specifications (v68+):**

```
HMX Block Diagram:

┌────────────────────────────────────┐
│      HMX (Matrix Extension)        │
├────────────────────────────────────┤
│                                    │
│  Input Buffers (16KB SRAM)        │
│  ├─ Activation cache (8KB)        │
│  └─ Weight cache (8KB)            │
│                                    │
│  ┌──────────────────────────────┐ │
│  │   Multiply-Accumulate Array  │ │
│  │   128 × 128 MAC engines      │ │
│  │   (INT8 or FP16)             │ │
│  │                              │ │
│  │  Performs:                   │ │
│  │  C[i][j] += A[i][k] × B[k][j]│ │
│  └──────────────────────────────┘ │
│                                    │
│  Output Buffer (4KB SRAM)         │
│  ├─ Post-accumulation buffer    │
│  └─ Activation function support │
│                                    │
│  Control Logic:                   │
│  ├─ Matrix dimension sequencer   │
│  ├─ Microinstruction decoder     │
│  └─ Accumulator management       │
│                                    │
└────────────────────────────────────┘

Performance:
• INT8 MatMul: 16K MACs/cycle @ 1.8 GHz = 28.8 TMAC/s
• FP16 MatMul: 8K MACs/cycle @ 1.8 GHz = 14.4 TFLOPS
```

**HMX vs HVX Comparison:**

| Aspect | HVX | HMX |
|--------|-----|-----|
| Type | SIMD Vector | Tensor Array |
| Ops/Cycle | 16 (256b, FP32) | 16,000 (INT8 MAC) |
| Throughput | 512 bits | 16 KB (aggregated) |
| Latency | 1-3 cycles | 50-100 cycles |
| Prime Use | DSP, filters | MatMul, convolution |
| Data Width | 8/16/32/64 bits | 8-bit (INT8), 16-bit (FP16) |

**HMX Instruction Format:**

HMX is controlled via specialized tensor instructions that encode matrix dimensions and operation types:

```c
// Pseudocode HMX instruction structure
struct HMX_MatMul {
    uint32_t opcode : 6;        // Operation type
    uint32_t m : 8;             // Rows in A (A is m×k)
    uint32_t k : 8;             // Common dimension
    uint32_t n : 8;             // Cols in B (B is k×n)
    uint32_t data_type : 2;     // INT8=0, FP16=1, TF32=2
};
// Max matrix: 256×256×256 INT8 MatMul in ~1000 cycles
```

### 1.5 Interrelationship Between Processing Units

```
Unified Instruction Issue Point:

┌──────────────────────────────────────────────────────┐
│           Packet (up to 128 bits)                    │
│  Contains up to 4 instructions                       │
└────┬─────────────────────────────────────────────────┘
     │
     ▼
┌──────────────────────────────────────────────────────┐
│     Resource Conflict Resolution                     │
│  (Check for resource collisions across units)        │
└────┬──────────────┬──────────────┬──────────────────┘
     │              │              │
     ▼              ▼              ▼
 ┌────────┐    ┌────────┐    ┌────────┐
 │ Scalar │    │  HVX   │    │ HMX/   │
 │ Core   │    │        │    │ Memory │
 │        │    │        │    │        │
 └────────┘    └────────┘    └────────┘

Key Design Points:
1. All units see same clock
2. Independent register files (minimal forwarding between domains)
3. Shared L2 cache and memory subsystem
4. HVX/HMX do NOT block scalar execution (mostly independent)
5. Data movement between scalar and vector requires explicit moves
```

### 1.6 Functional Units and Execution Resources

**Complete Execution Resource Summary:**

```
┌─────────────────────────────────────────────────────────┐
│         EXECUTION RESOURCE TABLE (per thread)           │
├──────────────────┬────────────────┬─────────────────────┤
│ Unit             │ Latency (cyc)  │ Throughput          │
├──────────────────┼────────────────┼─────────────────────┤
│ ALU (Integer)    │ 1              │ 2 ops/cycle (dual)  │
│ ADD/SUB          │ 1              │ Dual-issue capable  │
│ AND/OR/XOR       │ 1              │ Dual-issue capable  │
│ Comparator       │ 1              │ 1 result/cycle      │
│ Multiplier (32)  │ 2-3            │ 1 result/cycle      │
│ Multiplier (64)  │ 3              │ 1 result/cycle      │
│ Load Store Unit  │ 2 (hit)        │ 1 LD + 1 ST/cyc     │
│ Float ALU        │ 1              │ 1 op/cycle          │
│ Float Mult       │ 2              │ 1 op/cycle          │
│ Float Convert    │ 2-3            │ 1 op/cycle          │
│ Float Divide     │ 10-12          │ 1/10 ops/cycle      │
│ Float Sqrt       │ 10-12          │ 1/10 ops/cycle      │
├──────────────────┼────────────────┼─────────────────────┤
│ HVX ALU (256b)   │ 1              │ 1 op/cycle          │
│ HVX Mult (256b)  │ 3              │ 1 op/cycle          │
│ HVX LD/ST (256b) │ 2-3            │ 1 op/cycle          │
│ HMX MatMul       │ 50-100         │ Dedicated ops/cyc   │
└──────────────────┴────────────────┴─────────────────────┘
```

---

## 2. Hardware Threading Model

### 2.1 4-Way Hardware Multi-Threading Overview

The Hexagon core implements a fine-grained hardware threading model where up to 4 independent threads share a single physical core. This is distinct from simultaneous multi-threading (SMT) in that threads are not executing instructions simultaneously — rather, the VLIW engine interleaves instruction packets from different threads in round-robin fashion.

**Threading Architecture:**

```
┌──────────────────────────────────────────────────────┐
│  Hexagon Core (Single Physical Core)                 │
│                                                      │
│  ┌────────────────────────────────────────────────┐  │
│  │  Instruction Fetch & Decode                   │  │
│  │  (Supports all 4 threads simultaneously)      │  │
│  └────────────────────────────────────────────────┘  │
│                                                      │
│  ┌─────┬─────┬─────┬─────┐                          │
│  │ T0  │ T1  │ T2  │ T3  │  Thread ID              │
│  └─────┴─────┴─────┴─────┘                          │
│                                                      │
│  Program Counters: 4 independent PCs                │
│  ├─ PC0, PC1, PC2, PC3                              │
│  └─ Each thread maintains separate execution state  │
│                                                      │
│  Register Files (per-thread):                       │
│  ├─ R0-R31 (32 × 32-bit registers per thread)      │
│  ├─ P0-P3 (4 predicate registers per thread)       │
│  ├─ Total: 4 threads × 32 regs = 128 architectural │
│  └─ Actual SRAM: 256 × 32-bit (8 KB register file) │
│                                                      │
│  Control Registers (per-thread):                    │
│  ├─ UGPH (User Global Pointer)                     │
│  ├─ FRAMEKEY (Stack frame key)                     │
│  ├─ M0, M1 (Modulo-loop registers)                 │
│  └─ E0, E1 (Early exit registers)                  │
│                                                      │
└──────────────────────────────────────────────────────┘
```

### 2.2 VLIW Packet Interleaving and Round-Robin Scheduling

Instructions are fetched in 128-bit packets containing up to 4 instructions. The scheduler cycles through threads in round-robin order, issuing one packet per cycle (when available and not stalled).

**Interleaving Timeline:**

```
Cycle:   1  2  3  4  5  6  7  8  9 10 11 12
         │  │  │  │  │  │  │  │  │  │  │  │
Thread0: P0 .  .  P1 .  .  P2 .  .  P3 .  . (packets from T0)
Thread1: .  P0 .  .  P1 .  .  P2 .  .  P3 .
Thread2: .  .  P0 .  .  P1 .  .  P2 .  .  P3
Thread3: .  .  .  P0 .  .  P1 .  .  P2 .  .

Schedule: T0→T1→T2→T3→T0→T1→T2→T3→T0→T1→T2→T3

Stall scenarios:
┌─────────────────────────────────────────────────┐
│ If Thread0 has cache miss on cycle 1:           │
│                                                  │
│ Cycle:   1  2  3  4  5  6  7  8  9 10          │
│ Thread0: P0↓.  .  .  P1 .  .  P2 .  .          │
│ Thread1:    P0 .  .  P1 .  .  .  P2 .          │
│ Thread2:    .  P0 .  .  P1 .  .  .  P2         │
│ Thread3:    .  .  P0 .  .  P1 .  .  .          │
│                                                  │
│ Scheduler skips T0 for 3 cycles (memory latency)│
│ Other threads continue normal round-robin      │
└─────────────────────────────────────────────────┘
```

### 2.3 Register File Sharing and Banking

The register file is physically partitioned into 4 banks (one per thread), but there is also shared state for control flow and special registers.

**Register File Layout (v68 generation):**

```
Physical Register File (8 KB total SRAM):

┌──────────────────────────────────────────────────┐
│  Thread 0 Registers       (2 KB)                 │
│  ├─ R0-R31: 32 × 32-bit = 128 B                 │
│  ├─ P0-P3: 4 × 8-bit = 4 B (predicate)         │
│  ├─ Control Regs: M0, M1, E0, E1, etc. = 32 B  │
│  └─ Total addressing: 2048 B                    │
├──────────────────────────────────────────────────┤
│  Thread 1 Registers       (2 KB)                 │
│  ├─ R0-R31: 32 × 32-bit = 128 B                 │
│  ├─ P0-P3: 4 × 8-bit = 4 B                      │
│  └─ ...                                          │
├──────────────────────────────────────────────────┤
│  Thread 2 Registers       (2 KB)                 │
├──────────────────────────────────────────────────┤
│  Thread 3 Registers       (2 KB)                 │
└──────────────────────────────────────────────────┘

Access Pattern:
• Each thread reads/writes only its own bank (no conflicts)
• Register access: <thread_id><register_id> = unique address
• No inter-thread register forwarding (by design)
• Data passed between threads via shared memory only
```

### 2.4 Resource Sharing: Execution Units and Functional Units

While threads have separate register files, they share all execution units. The dispatcher must resolve:

**Conflict Types:**

1. **Functional Unit Conflicts**: Two threads trying to use same ALU
   - Solution: Stall one thread, let other execute
   - Priority: Lower thread ID typically has priority

2. **Cache Port Conflicts**: Multiple threads accessing L1 cache
   - L1 D-cache: 2 read ports, 1 write port per cycle
   - Thread 0: priority for port 0
   - Thread 1: priority for port 1
   - Threads 2-3: share remaining port, time-division

3. **Memory Bus Conflicts**: L2/L3 access serialization
   - Single memory port to L2
   - Arbitration: Round-robin with priority boost for waiting threads

**Execution Unit Allocation Policy:**

```
Dispatch Decision Matrix:

Thread T wants to issue packet at cycle C:

1. Check if functional units are free:
   if (ALU.free && LSU.free) → Can issue up to 2 instructions
   if (ALU.free XOR LSU.free) → Can issue 1 instruction
   if (NEITHER.free) → STALL thread

2. Check for L1 cache conflicts:
   if (port0.free) → T0,T1 high priority
   if (port1.free) → T0,T1 high priority
   if (multiple threads want same port) → arbitrate by thread_id

3. Check for resource hazards:
   if (instruction uses M0/M1 and another thread is using) → STALL
   if (instruction uses branch prediction state) → check prediction table

4. Update scheduling state:
   current_thread = (current_thread + 1) % 4
```

### 2.5 Thread Priority and Fairness

**Dynamic Thread Prioritization:**

```c
// Simplified scheduling algorithm (pseudo-C)
struct ThreadState {
    uint32_t pc;           // Program counter
    uint32_t stall_cycles; // Remaining stall time
    uint32_t priority;     // 0=highest, 3=lowest
    bool valid;            // Thread active?
};

void schedule_packet() {
    // Rotate priority every 4 cycles
    static int round_robin_id = 0;

    // Find next non-stalled thread
    for (int i = 0; i < 4; i++) {
        int tid = (round_robin_id + i) % 4;
        if (thread[tid].valid && thread[tid].stall_cycles == 0) {
            // Check if functional units available
            if (can_dispatch(tid)) {
                dispatch_packet(tid);
                round_robin_id = (tid + 1) % 4;
                return;
            }
        }
    }

    // No thread ready; insert NOP cycle (rare)
    insert_nop_packet();
}
```

**Stall Cycle Accounting:**

| Event | Stall Duration | Why |
|-------|----------------|-----|
| L1 cache miss | 10-20 cycles | Memory hierarchy traversal |
| L2 cache miss | 30-100 cycles | Off-core access |
| Memory read dependency | 2 cycles | Load-to-use hazard |
| Branch misprediction | 3-5 cycles | Pipeline flush |
| Functional unit not ready | 1-3 cycles | Operand not available |
| Lock contention | Variable | Inter-thread synchronization |

### 2.6 Thread-Local Storage and Register Isolation

Each thread maintains complete state isolation:

```
Register Isolation Mechanism:

Thread 0:                    Thread 1:
R0=0x1000                    R0=0x2000  (same name, different storage)
R1=0x1004                    R1=0x2004
P0=condition                 P0=different condition
PC=0x80000000               PC=0x80000100

Execution:
│ Cycle 1: Thread0 packet executes    R0←result (Thread0.R0)
│ Cycle 2: Thread1 packet executes    R0←result (Thread1.R0)
│
│ No register port conflicts
│ No forwarding delays between threads
│ Synchronization via shared memory + atomic ops
```

---

## 3. Memory Hierarchy

### 3.1 Cache System Overview

The Hexagon processor implements a two-level (L1/L2) cache hierarchy with separate instruction and data caches at L1, unified L2, and optional tightly-coupled memories.

**Cache Hierarchy Diagram:**

```
┌────────────────────────────────────────────────────┐
│           Scalar Core + HVX + HMX                  │
│  (4 threads, 32 KB L1-I, 32 KB L1-D)              │
└─────────────────────────────────────────────────────┘
         │                          │
         │                          │
    (128-bit I-fetch)           (64-bit D-access)
         │                          │
         ▼                          ▼
┌─────────────────────────────────────────────────────┐
│  L1 Instruction Cache            L1 Data Cache     │
│  32 KB, 4-way, 32B lines         32 KB, 4-way,    │
│  128-bit fetch width              32B lines        │
│  I-TLB (64 entries)              D-TLB (32 entries)│
└─────────────────────────────────────────────────────┘
         │                          │
         └──────────────┬───────────┘
                        │
                        ▼
             ┌───────────────────────┐
             │    L2 Cache (Unified) │
             │  256/512 KB           │
             │  8-way associative    │
             │  64 B lines           │
             │  64-bit port          │
             │  2-cycle latency (hit)│
             └───────────────────────┘
                        │
                        ▼
        ┌──────────────────────────────────┐
        │   System Level (Off-Chip)        │
        │   • Main Memory (LPDDR4/5)       │
        │   • NoC interconnect             │
        │   • DMA, SMMU                    │
        └──────────────────────────────────┘
```

### 3.2 L1 Instruction Cache (I-Cache)

**Specifications by Generation:**

| Aspect | v65/v66 | v68 | v69/v73 | v75 |
|--------|---------|-----|---------|-----|
| Size | 32 KB | 32 KB | 32 KB | 32 KB |
| Associativity | 4-way | 4-way | 4-way | 4-way |
| Line Size | 32 B | 32 B | 32 B | 32 B |
| Fetch Width | 128 b | 128 b | 128 b | 128 b |
| Hit Latency | 1 cycle | 1 cycle | 1 cycle | 1 cycle |
| I-TLB Entries | 64 | 64 | 64 | 64 |

**I-Cache Organization:**

```
32 KB, 4-way I-Cache:
├─ Total lines: 32 KB / 32 B = 1024 lines
├─ Sets: 1024 / 4 = 256 sets
├─ Set size: 32 B × 4 = 128 B per set
└─ Address decoding:
   bits[31:10] = tag (22 bits)
   bits[9:5]   = set index (5 bits for 32 sets? No, 8 bits for 256)
   bits[4:0]   = line offset (32 B = 2^5 bytes)

Actual addressing:
   Physical Address: [Tag: 22 bits][Set: 8 bits][Offset: 5 bits]
                     [31..10]      [9..2]      [1..0]
   (128-bit alignment required for packet fetch)
```

**I-Cache Access Path:**

```c
// I-Cache hit scenario (1 cycle)
Cycle N:   PC[31:10] → I-TLB lookup (parallel with cache)
           PC[9:2]  → Set selection
           PC[1:0]  = 00 (128-bit aligned)

Cycle N:   Compare tags in 4-way set
           Mux out matching way (128-bit data)
           Update LRU replacement state

Cycle N+1: Instruction decoder receives 128-bit packet
```

### 3.3 L1 Data Cache (D-Cache)

**D-Cache Specifications:**

| Aspect | Configuration |
|--------|---|
| Size | 32 KB |
| Associativity | 4-way |
| Line Size | 32 B |
| Access Width | 64-bit (dual-bank) |
| Hit Latency | 2 cycles (address generation in E4, hit in M5) |
| Write-Through/Back | Write-back |
| Victim Buffer | Yes, 4-entry |
| D-TLB | 32 entries, fully associative |

**D-Cache Dual-Bank Organization:**

```
32 KB D-Cache (Dual-bank for parallel access):

┌──────────────────────────────────────────────┐
│  Bank 0 (16 KB)      │      Bank 1 (16 KB)   │
│  Even addresses      │      Odd addresses    │
├──────────────────┬───┼───────────────────────┤
│ 256 sets         │   │  256 sets            │
│ 4-way ×  16 B    │   │  4-way × 16 B        │
│ per bank         │   │  per bank            │
│ Shared TLB (32)  │   │                      │
└──────────────────┴───┴───────────────────────┘

Access Pattern:
Load:  r0 = memb(r1)     // Bank 0 or 1 based on r1[4]
Store: memb(r2) = r3     // Bank 0 or 1 based on r2[4]
Dual:  r4 = memw(r5)     // Can access both banks if addresses differ

Throughput: 1 load + 1 store per cycle (if non-conflicting banks)
```

**D-Cache Access Latency Details:**

```
L1 D-Cache Hit Path:

E4 (EXECUTE stage):
  ├─ Generate address: r_base + offset (computed in previous cycle)
  ├─ Split to D-TLB for translation
  └─ Access D-cache tag/data in parallel

M5 (MEMORY stage):
  ├─ D-TLB translation complete: VA → PA
  ├─ Compare PA[upper bits] with cache tags
  ├─ Mux out matching way
  └─ Load data into writeback queue

W6 (WRITEBACK):
  └─ Register file update with loaded data

Total latency: 2 cycles (E4 start → W6 complete)
Load-to-use forward: 1 cycle (result available in next packet)
```

### 3.4 L2 Unified Cache

The L2 cache serves both instruction and data accesses, with an 8-way set-associative organization.

**L2 Specifications by Generation:**

| Generation | Size | Assoc | Line | Hit Latency | Bandwidth |
|------------|------|-------|------|-------------|-----------|
| v65 | 256 KB | 8-way | 64 B | 2 cyc | 128 bits |
| v66 | 256 KB | 8-way | 64 B | 2 cyc | 128 bits |
| v68 | 512 KB | 8-way | 64 B | 2 cyc | 128 bits |
| v69/v73 | 512 KB | 8-way | 64 B | 2 cyc | 128 bits |
| v75 | 512 KB | 8-way | 64 B | 2 cyc | 128 bits |

**L2 Cache Structure:**

```
L2 Cache (512 KB example):

512 KB / 64 B = 8192 lines
8192 lines / 8 ways = 1024 sets

┌─────────────────────────────────┐
│ Set 0                           │
│ ├─ Way 0: [Tag | Data (64B)]   │
│ ├─ Way 1: [Tag | Data (64B)]   │
│ ├─ Way 2: [Tag | Data (64B)]   │
│ ├─ Way 3: [Tag | Data (64B)]   │
│ ├─ Way 4: [Tag | Data (64B)]   │
│ ├─ Way 5: [Tag | Data (64B)]   │
│ ├─ Way 6: [Tag | Data (64B)]   │
│ └─ Way 7: [Tag | Data (64B)]   │
├─────────────────────────────────┤
│ Set 1                           │
│ ...                             │
├─────────────────────────────────┤
│ Set 1023                        │
└─────────────────────────────────┘

Address decoding:
[31..16]: Tag (16 bits for physical address bits[31..16])
[15..6]:  Set index (10 bits = 1024 sets)
[5..0]:   Line offset (64 B = 2^6 bytes)
```

**L2 Hit/Miss Path:**

```
L2 Miss handling sequence:

Cycle N (E-stage):     L1 miss detected
Cycle N+1 (M-stage):   L2 lookup initiated
                       ├─ Set index calculation
                       ├─ Tag comparison across 8 ways
                       └─ Mux out matching way

Cycle N+2 (W-stage):   L2 hit → data forwarded to L1
                       OR
                       L2 miss → request sent to memory bus

Cycle N+3+ (Memory):   Request queued in write-back buffer
                       ├─ NoC arbitration
                       ├─ Off-core access (30-100 cycles total)
                       └─ Data returns through write-back

Cycle N+2 forward (hit-case forward):
  └─ Result available to dependent instructions in packet N+3
```

### 3.5 TCM (Tightly Coupled Memory) and VTCM (Vector TCM)

**TCM Purpose:** Fast, predictable access for critical code/data. No cache miss latency.

**TCM Specifications by Generation:**

| Generation | TCM Size | VTCM Size | Purpose |
|------------|----------|-----------|---------|
| v65/v66 | 0 KB | 0 KB | N/A |
| v68 | 0-32 KB | 0-128 KB | Optional |
| v69/v73 | 0-32 KB | 0-128 KB | Optional |
| v75 | 0-32 KB | 0-256 KB | Optional |

**TCM vs VTCM Distinction:**

```
TCM (Tightly Coupled Memory):
├─ Scalar core access only
├─ Single port, 32-bit accesses
├─ 1-cycle access latency (fixed)
├─ Address range: 0xa0000000 - 0xa000FFFF (64 KB addressable)
└─ Use cases: Inner loops, kernel code

VTCM (Vector TCM):
├─ Vector/HMX unit access only
├─ Dual 128-bit ports (or 256-bit on v75)
├─ 1-cycle access latency (fixed)
├─ Address range: 0xb0000000 - 0xb003FFFF (256 KB addressable)
└─ Use cases: Vector/matrix working sets, intermediate results

Memory Map:
┌─────────────────────────────────┐
│ 0xFFFFFFFF                      │
│  ...                            │
│ 0xE0000000  MMIO (peripherals) │
│ 0xB0040000                      │
│  VTCM (if configured)           │
│ 0xB0000000                      │
│ 0xA0010000                      │
│  TCM (if configured)            │
│ 0xA0000000                      │
│  ...                            │
│ 0x80000000  Main DDR            │
│  ...                            │
│ 0x00000000                      │
└─────────────────────────────────┘
```

**TCM Access Example:**

```c
// TCM access - fixed 1 cycle latency
// Must use specific intrinsics/pragmas

#pragma TCM  // Place function in TCM
void fast_kernel() {
    volatile uint32_t *tcm_ptr = (uint32_t*)0xa0000000;
    uint32_t data = *tcm_ptr;  // 1 cycle guaranteed
    // No cache miss possible
    data += 1;
    *tcm_ptr = data;  // 1 cycle guaranteed
}

// VTCM access - predictable for vector operations
#pragma VTCM_DATA  // Place data in VTCM
__attribute__((aligned(256)))
int8_t weight_matrix[128][128];  // 16 KB in VTCM

void vector_kernel() {
    HVX_Vector *v0 = (HVX_Vector*)&weight_matrix[0][0];
    HVX_Vector result = Q6_V_vmem_AV(*v0);  // 1 cycle hit
}
```

### 3.6 Cache Coherence and Consistency

Hexagon is a single-threaded core (cache-coherence semantics apply within threads only). Inter-thread synchronization is explicit.

**Memory Barrier Instructions:**

```c
// Hexagon memory fence operations

// MEMORY_BARRIER (full fence)
// Blocks until all prior memory operations complete
__asm__ volatile ("MEMORY_BARRIER" ::: "memory");

// SYNCH (lightweight sync)
// Ensures instruction ordering
__asm__ volatile ("SYNCH" ::: "memory");

Usage:
Lock Acquire:
  acquire_lock()    // Atomic
  MEMORY_BARRIER    // Ensure writes before are visible
  read_shared_data()

Lock Release:
  write_shared_data()
  MEMORY_BARRIER    // Ensure writes before release visible
  release_lock()    // Atomic
```

---

## 4. The Hexagon ISA

### 4.1 Instruction Format and Packet Structure

The Hexagon ISA is structured around 128-bit **packets** containing up to 4 instructions, all executed in the same cycle (or stalled together if dependencies exist).

**Packet Structure:**

```
128-bit Packet:
┌───────────────────────────────────────────────────────────┐
│ Instr 3 (32 bits) │ Instr 2 │ Instr 1 │ Instr 0 (32 bits) │
│                   │ (32 b)  │ (32 b)  │                   │
└───────────────────────────────────────────────────────────┘

Packing rules:
1. Instructions execute in slots: Slot 0, 1, 2, 3
2. Order within packet doesn't affect execution semantics
   (but matters for resource allocation)
3. Each instruction independently specifies its slot

Example assembly (hand-written packet):
   {
       r0 = add(r1, r2)      // Slot 0 (can use ALU)
       r3 = add(r4, r5)      // Slot 1 (can use ALU) [dual-issue]
       r6 = memw(r7)         // Slot 2 (LSU)
       p0 = cmp.eq(r8, r9)   // Slot 3 (ALU)
   }
```

**Instruction Encoding (32-bit):**

```
Standard instruction format:
┌─────────────────────────────────────────┐
│ Opcode (10 bits) │ Src (2×6b) │ Dst(5b) │
│ [31..22]         │ [21..10]   │ [4..0]  │
└─────────────────────────────────────────┘

Extended instruction format (for immediates):
┌──────────────┬────────────────────────────┐
│ Opcode (8b)  │ Immediate (24 bits)        │
│ [31..24]     │ [23..0]                    │
└──────────────┴────────────────────────────┘

Predicated instruction format:
┌────┬──────────────────────────────────────┐
│ Pr │ Instruction (28 bits)                │
│ [3 │ [27..0]                              │
│ 1  │                                      │
│ ]  │                                      │
└────┴──────────────────────────────────────┘
Bits [3..2]: Predicate register (p0-p3)
Bit [4]: Predicate polarity (if=!if)
Bits [27..0]: Actual instruction
```

### 4.2 Instruction Slot Assignment Rules

Critical to understanding Hexagon code generation: instructions must be assigned to valid slots based on their type and resource requirements.

**Slot Assignment Table:**

| Slot | Resources | Valid Instruction Types |
|------|-----------|---|
| 0 | ALU, MUL | ADD, SUB, AND, OR, MUL, compare, shift, move |
| 1 | ALU, MUL | Same as Slot 0 (ALU can dual-issue) |
| 2 | LSU | LOAD, STORE, prefetch, atomic |
| 3 | ALU, MUL, Branch | Same as 0/1, or branch, call, return |

**Dual-Issue Constraints:**

```
Valid dual-issue combinations within a packet:

✓ ALU + ALU         (Slot 0 + Slot 1)
✓ ALU + LSU         (Slot 0/1 + Slot 2)
✓ ALU + ALU + LSU   (Slot 0 + Slot 1 + Slot 2)
✓ ALU + LSU + Branch (Slot 0/1 + Slot 2 + Slot 3)

✗ ALU + ALU + ALU   (at most 2 ALUs per cycle due to resources)
✗ LSU + LSU         (only 1 LSU per cycle... mostly)
✗ Multiple branches (only 1 branch per packet)
✗ Store + Store     (dual-store in v68+ only)

Dual-Store (v68+, Slot 2 only):
   {
       memw(r0) = r1
       memh(r2) = r3      // DUAL-STORE allowed if non-overlapping
   }

Store + Load prohibition:
   {
       memw(r0) = r1      // STORE
       r2 = memw(r3)      // LOAD - CONFLICT (only 1 LSU)
   }  // INVALID packet - assembler error
```

### 4.3 Register Operand Format and Addressing Modes

**Register Operands:**

```
Integer registers: R0 - R31 (32 × 32-bit)
Register pairs:    R1:R0, R3:R2, ..., R31:R30 (64-bit)
Predicate regs:    P0 - P3 (each 8 bits)

Addressing modes:

1. Register direct:
   r0 = add(r1, r2)          // Rd = Opcode(Rs1, Rs2)

2. Register with immediate:
   r0 = add(r1, #42)         // Rd = Opcode(Rs, Immediate)

3. Scaled register offset:
   r0 = memw(r1 + r2 << #2)  // Load with scaled offset
   (shifts r2 by 2 bits: useful for array indexing)

4. Base + immediate:
   r0 = memw(r1 + #8)        // Load from r1 + 8

5. Post-increment/decrement:
   r0 = memw(r1++  #4)       // Load, then r1 += 4 (auto-increment)
   r0 = memw(r1-- )          // Load, then r1 -= 4

6. Modulo addressing (hardware loop):
   r0 = memw(r2++M0)         // Load, r2 += M0, wrap at M1 boundary
```

**Immediate Value Encoding:**

```c
// Immediate encoding in Hexagon

// 8-bit immediates (i8):
r0 = add(r1, #-1)           // Range: -1 to -128

// 16-bit immediates (i16):
r0 = add(r1, #1000)         // Range: -32768 to 32767
r1 = memw(r2 + #1024)       // Must be 4-byte aligned

// 32-bit immediates (i32):
r0 = ##0x12345678           // Uses 2 instructions (CONST + OR)

// Large immediates usually expanded by assembler:
r0 = ##0x12345678
// Becomes:
//   r0 = {#0x1234}         // CONST (upper 22 bits)
//   r0 = or(r0, #0x5678)   // OR in lower bits
```

### 4.4 Conditional Execution and Predicates

Hexagon predicates (P0-P3) enable instruction-level conditional execution without branches.

**Predicate Operations:**

```c
// Predicate assignment
p0 = cmp.eq(r0, r1)         // p0 = (r0 == r1) ? 1 : 0
p1 = cmp.gt(r2, #100)       // p1 = (r2 > 100) ? 1 : 0

// Predicate combine
p2 = and(p0, p1)            // p2 = p0 AND p1
p3 = or(p0, p1)             // p3 = p0 OR p1

// Conditional instruction execution
if (p0) r0 = add(r1, r2)    // Execute only if p0 == 1
if (!p0) r0 = sub(r1, r2)   // Execute only if p0 == 0

// Conditional move
r0 = mux(p0, r1, r2)        // r0 = p0 ? r1 : r2
```

**Predicate Register Format:**

```
P0-P3: Each is effectively an 8-bit register
┌───────────────┐
│ Bits [7..1]   │ Unused (reserved)
│ Bit [0]       │ Predicate value (0 or 1)
└───────────────┘

When used as condition:
if (p0) = if (p0[0] != 0)
if (!p0) = if (p0[0] == 0)
```

### 4.5 Branch Instructions and Prediction

Hexagon implements an efficient branch prediction unit integrated with the VLIW pipeline.

**Branch Instruction Types:**

```c
// Unconditional branches
jump #offset                 // PC-relative jump
jumpr r0                     // Register jump (computed address)
call #offset                 // Subroutine call (link r31)

// Conditional branches
if (p0) jump #offset         // Conditional jump
if (!p0) jump #offset        // Inverted condition

// Hardware loops (special loop branches)
loop0(#iterations, #begin_label)
loop1(#iterations, #begin_label)

// Ending loops
endloop0                      // End loop0 with back-edge jump
endloop1                      // End loop1
```

**Branch Prediction Mechanics:**

```
Prediction Unit:

┌────────────────────────────────┐
│ Branch Target Buffer (BTB)     │
│ 512 entries, 2-way assoc       │
│ [PC][Target][Prediction]       │
│                                │
│ Hit rate on loop branches: 95%+ │
│ Hit rate on unpredicted: 50%   │
└────────────────────────────────┘

Prediction accuracy:
• Backward branches (loops): ~95% (taken)
• Forward branches (if-then): ~70-80%
• Indirect jumps: ~80% (history-based)

Misprediction penalty:
• Branch resolved in E4 (EXECUTE stage)
• Pipeline flushes stages after decode
• Recovery: 3-5 cycles

Prefetching based on prediction:
Cycle N: Branch predicted in FETCH
         I-cache prefetch to predicted target
Cycle N+1: If correct, fetch stream continues
           If wrong, I-cache pipeline flushes
```

**Hardware Loop Instructions:**

```c
// Hardware loop structure (zero-overhead loop)
r0 = #0          // Iteration counter
loop0(r0, #label_start)  // Setup loop0 for r0 iterations

#label_start:
    r1 = add(r1, r2)
    r2 = memw(r3++  #4)
    // ...
endloop0         // Jump back to label_start, decrement r0

// Properties:
// • Zero-overhead (no branch penalty in steady state)
// • Can nest up to 2 levels (loop0 inside loop1)
// • Loop counter in registers (LC0, LC1 implicit)
// • Loop end address in LE0, LE1
```

### 4.6 NOP Cycles and Stall Hazards

**Hazard Types and Solutions:**

```c
// 1. Read-After-Write (RAW) hazard
r0 = add(r1, r2)    // Cycle N
r3 = add(r0, r4)    // Cycle N+1 (r0 ready in W6, can forward in E4)
// NO STALL (1-cycle forward path)

// 2. Store-to-Load dependency
memw(r0) = r1       // Store in cycle N
r2 = memw(r3)       // Load in cycle N+1
// STALL: Load must wait for store ordering (2 cycles minimum)

// 3. Multi-cycle result dependency
r0 = mpyh(r1, r2)   // Multiply (3-cycle latency)
r3 = add(r0, r4)    // Must wait 3 cycles
// Cycle N:   MULT starts
// Cycle N+1: Stall (waiting)
// Cycle N+2: Stall (waiting)
// Cycle N+3: Result ready, can start ADD

// 4. Load-to-use forward hazard
r0 = memw(r1)       // 2-cycle latency
r1 = add(r0, r2)    // Can start 1 cycle after load (forward from L1 hit)
// Cycle N:   Load from L1 hit
// Cycle N+1: Data forwarded (even though W6 is cycle N+2)
// Cycle N+2: ADD executes with r0 value

// NOP handling
{
    memw(r0) = r1       // Store
    nop
    r2 = memw(r3)       // Load (after 1-cycle stall)
}
```

**Instruction Dependency Tracking:**

```
Hazard Detection Hardware:

For each active thread:
├─ Register scoreboard (tracks pending writes)
│  ├─ R0-R31: each has "write pending" bit + cycle count
│  └─ If instruction needs Rx and scoreboard[x].pending:
│     STALL until scoreboard[x].ready
│
├─ Memory dependency tracking
│  ├─ Store address queue
│  ├─ Load address queue
│  └─ If load-addr == store-addr (same cycle): STALL
│
└─ Special resource conflicts
   ├─ Branch prediction unit lock
   ├─ Multiplier busy (for seq multiply)
   └─ Shared functional unit
```

### 4.7 Endianness and Data Organization

Hexagon is **little-endian**.

**Data Layout:**

```
32-bit integer r0 = 0x12345678 (little-endian):
┌────┬────┬────┬────┐
│ 78 │ 56 │ 34 │ 12 │  (memory addresses increasing)
└────┴────┴────┴────┘
0x1000 0x1001 0x1002 0x1003

Accessing:
memb(0x1000) = 0x78  (LSB)
memb(0x1003) = 0x12  (MSB)

64-bit pair r1:r0 (R1 holds upper 32 bits):
r0 = 0x12345678
r1 = 0xABCDEF00

Combined in memory (little-endian):
┌────┬────┬────┬────┬────┬────┬────┬────┐
│ 78 │ 56 │ 34 │ 12 │ 00 │ EF │ CD │ AB │
└────┴────┴────┴────┴────┴────┴────┴────┘
0x1000              0x1004

Accessing as 64-bit:
memd(0x1000) = 0xABCDEF0012345678 (r1:r0)
```

---

## 5. Hexagon NPU within Snapdragon SoC

### 5.1 Snapdragon SoC Block Diagram

The Hexagon processor is one compute engine within the larger Snapdragon SoC ecosystem.

```
Snapdragon SoC Architecture (Schematic):

┌─────────────────────────────────────────────────────────┐
│                  System-on-Chip                         │
│                                                         │
│  ┌────────────────────────────────────────────────┐   │
│  │        CPU Cluster (ARM Cortex cores)          │   │
│  │  ├─ Performance cores (2-4 × Cortex-X)        │   │
│  │  ├─ Efficiency cores (2-4 × Cortex-A)         │   │
│  │  └─ L3 shared cache (2-4 MB)                   │   │
│  └────────────────────────────────────────────────┘   │
│                                                         │
│  ┌────────────────────────────────────────────────┐   │
│  │    GPU (Adreno)                                │   │
│  │    ├─ Vertex/Fragment pipelines                │   │
│  │    ├─ Texture units                            │   │
│  │    └─ L2 cache shared with CPU                 │   │
│  └────────────────────────────────────────────────┘   │
│                                                         │
│  ┌────────────────────────────────────────────────┐   │
│  │    Hexagon NPU ⭐ (This Module Focus)          │   │
│  │    ├─ Hexagon core (v68/v73/v75 etc)         │   │
│  │    ├─ HVX vector unit                         │   │
│  │    ├─ HTA/HMX tensor accelerator              │   │
│  │    ├─ TCM/VTCM memory                         │   │
│  │    └─ Hexagon-specific caches                 │   │
│  └────────────────────────────────────────────────┘   │
│                                                         │
│  ┌────────────────────────────────────────────────┐   │
│  │    System Agent / NoC Interconnect             │   │
│  │    ├─ Crossbar/mesh network                    │   │
│  │    ├─ Memory arbiter                           │   │
│  │    ├─ DMA controllers                          │   │
│  │    ├─ SMMU (IOMMU)                             │   │
│  │    └─ Interrupt controller                     │   │
│  └────────────────────────────────────────────────┘   │
│                                                         │
│  ┌────────────────────────────────────────────────┐   │
│  │    Memory Subsystem                             │   │
│  │    ├─ System L3 cache                          │   │
│  │    ├─ Memory controller                        │   │
│  │    └─ LPDDR4/LPDDR5 interface                  │   │
│  └────────────────────────────────────────────────┘   │
│                                                         │
│  ┌────────────────────────────────────────────────┐   │
│  │    Peripherals (IoT, Audio, Sensor)            │   │
│  └────────────────────────────────────────────────┘   │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### 5.2 Interconnect Architecture (NoC)

The Hexagon processor connects to the rest of the SoC via a network-on-chip (NoC).

**NoC Topology (Snapdragon 8 Gen 2 example):**

```
NoC Mesh Interconnect:

                      ┌─────────────┐
                      │   Memory    │
                      │ Controller  │
                      └──────┬──────┘
                             │
    ┌────────────┬───────────┼───────────┬────────────┐
    │            │           │           │            │
    ▼            ▼           ▼           ▼            ▼
┌────────┐  ┌────────┐  ┌─────────┐ ┌────────┐  ┌────────┐
│  CPU   │  │  GPU   │  │ Hexagon │ │ ISP    │  │ Video  │
│Cluster │  │Adreno  │  │ (HTA)   │ │        │  │Encoder │
└────────┘  └────────┘  └─────────┘ └────────┘  └────────┘

Hexagon NoC Connections:
┌─────────────────────────────────────┐
│ Hexagon Core                        │
├─────────────────────────────────────┤
│ Core Interface                      │
│ ├─ Instruction fetch               │
│ ├─ Data read/write                 │
│ └─ Configuration registers         │
├─────────────────────────────────────┤
│ NoC Port(s):                        │
│ ├─ Primary port (L2 cache → NoC)   │
│ └─ DMA port (parallel access)      │
├─────────────────────────────────────┤
│ Arbitration:                        │
│ ├─ Round-robin with priority boost │
│ ├─ QoS (Quality of Service) tags  │
│ └─ Request timeout: 1000s cycles   │
└─────────────────────────────────────┘

Bandwidth:
• Hexagon L2 → NoC: ~128 bits/cycle @ 1.8 GHz = 28.8 GB/s
• DDR (LPDDR5): ~32 GB/s (shared with other masters)
• NoC arbitration ensures fair share
```

### 5.3 DMA (Direct Memory Access) Engines

Hexagon includes dedicated DMA engines for off-core memory access independent of the processor pipeline.

**DMA Engine Architecture:**

```
Hexagon DMA System:

┌─────────────────────────────────────┐
│    Hexagon Core                     │
│  (Scalar + HVX + HMX)               │
└────────┬────────────────────────────┘
         │
    ┌────┴──────────────┐
    │                   │
    ▼                   ▼
┌───────────┐    ┌──────────────┐
│  Core     │    │  DMA Engine  │
│  L1/L2    │    │              │
│  Cache    │    │  • 2-4 channels
│           │    │  • Up to 256-bit
│           │    │  • Independent scheduling
└───────────┘    └──────────────┘
    │                   │
    └───────┬───────────┘
            │
            ▼
    ┌──────────────────┐
    │   NoC            │
    │ (to DDR/L3)      │
    └──────────────────┘

DMA Channel Configuration:

struct DMA_Config {
    uint32_t src_addr;          // Source address
    uint32_t dst_addr;          // Destination address
    uint32_t length;            // Transfer length (bytes)
    uint32_t src_stride;        // Source stride (for 2D)
    uint32_t dst_stride;        // Destination stride
    uint32_t flags;             // Options
    #define DMA_FIXED_SRC    0x01
    #define DMA_FIXED_DST    0x02
    #define DMA_PRIORITY     0x0C  // Bits 3-2: priority level
};

DMA command queue:
• Up to 256 pending requests per channel
• Submission: Scalar core writes to MMIO registers
• Interrupt on completion (optional)
• Status polling: Check "DMA_STATUS" register
```

**DMA Performance Profile:**

| Operation | Throughput | Latency | Notes |
|-----------|-----------|---------|-------|
| DMA burst (1D) | 128 bits/cycle | ~100 cycles setup | Coalesced |
| DMA 2D transfer | Same | Stride-dependent | Address calc/cycle |
| DMA → L2 | ~200 GB/s peak | Depends on L2 state | Rarely bottleneck |
| DMA → DDR | ~32 GB/s (LPDDR5) | 50-100 cycles | Shared bandwidth |

**DMA Code Example:**

```c
// Hexagon DMA initialization (pseudo-C)
typedef volatile struct {
    uint32_t CMD;            // 0x0 - DMA command
    uint32_t STATUS;         // 0x4 - DMA status
    uint32_t ADDR_SRC;       // 0x8 - Source address
    uint32_t ADDR_DST;       // 0xC - Destination
    uint32_t LENGTH;         // 0x10 - Transfer length
    uint32_t SRC_STRIDE;     // 0x14 - Source stride
    uint32_t DST_STRIDE;     // 0x18 - Destination stride
} DMA_Registers;

void dma_copy_2d(const void *src, void *dst,
                 uint32_t rows, uint32_t cols,
                 uint32_t src_row_stride, uint32_t dst_row_stride) {
    DMA_Registers *dma = (DMA_Registers*)0xe0000000;

    dma->ADDR_SRC = (uint32_t)src;
    dma->ADDR_DST = (uint32_t)dst;
    dma->LENGTH = cols * rows;
    dma->SRC_STRIDE = src_row_stride;
    dma->DST_STRIDE = dst_row_stride;
    dma->CMD = 0x01;  // START

    // Poll completion
    while ((dma->STATUS & 0x1) == 0) {
        // Wait
    }
}
```

### 5.4 SMMU (System Memory Management Unit) and IOMMU

The SMMU provides virtual-to-physical address translation for Hexagon and other peripherals.

**SMMU Configuration for Hexagon:**

```
SMMU Hierarchy:

┌──────────────────────────────┐
│ SMMU (System MMU)            │
│ ├─ SMMUv3 architecture      │
│ ├─ Stream table (per device)│
│ │  └─ Hexagon ID = 0x10    │
│ ├─ Stage 1: VA → IPA        │
│ │  (Virtual → Intermediate) │
│ ├─ Stage 2: IPA → PA        │
│ │  (Intermediate → Physical)│
│ ├─ TLB (4K entries)         │
│ └─ Invalidation queue       │
└──────────────────────────────┘
       │
       ▼
   ┌────────────────┐
   │ Page Tables    │
   │ (DDR-resident) │
   └────────────────┘

Hexagon SMMU Configuration:

Stream Entry (for Hexagon):
┌──────────────────────────────────────────┐
│ Stream ID: 0x10                          │
│ IOMMU Enable: 1 (SMMU translations on)   │
│ Page Table Base: 0x_____ (PA)            │
│ Page Table Format: LPAE (48-bit)         │
│ Partition Number: 0 (privileged)        │
│ ASID: 0 (address space ID)               │
│ VMID: 0 (VM ID, for stage 2)             │
│ Permissions: R/W (all)                   │
└──────────────────────────────────────────┘

Address Translation Steps:

VA (Virtual Address): [47:12] (page) | [11:0] (offset)
                       │                  │
                       ▼                  ▼
                 Walk page tables   Direct pass-through
                       │
                       ▼
                 IPA (Intermediate PA)
                       │
         Stage 2 translation (if enabled)
                       │
                       ▼
                 PA (Physical Address)
                       │
                       ▼
                  Memory access
```

### 5.5 Power Domains (VDDCX, VDDMX)

Hexagon can be independently power-gated or frequency-scaled.

**Power Domain Organization:**

```
Power Distribution in Snapdragon:

┌─────────────────────────────────────────────────────┐
│ PMC (Power Management Controller)                   │
├─────────────────────────────────────────────────────┤
│                                                     │
│  Domain 1: VDDCX (Core Logic)                      │
│  ├─ Hexagon scalar core                           │
│  ├─ HVX (vector unit)                             │
│  ├─ L1 caches                                      │
│  ├─ Associated logic                              │
│  └─ Typical: 0.6 - 1.0 V                          │
│                                                     │
│  Domain 2: VDDMX (Memory/Aux)                     │
│  ├─ L2 cache                                       │
│  ├─ TCM/VTCM memory                               │
│  ├─ DMA controllers                               │
│  ├─ SMMU                                           │
│  └─ Typical: 0.8 - 1.1 V                          │
│                                                     │
│  Domain 3: VDDMX (HMX/Tensor)                     │
│  ├─ HMX accelerator (v68+)                        │
│  ├─ Associated buffers                            │
│  └─ Typical: 0.8 - 1.1 V                          │
│                                                     │
│  Auxiliary Supplies:                              │
│  ├─ VDDCR (core reference)                        │
│  ├─ VDDDPLL (PLL supply)                          │
│  └─ ...                                            │
│                                                     │
└─────────────────────────────────────────────────────┘

Power Gating:

VDDCX off (power gate):
├─ Hexagon core not accessible
├─ State lost (register files flushed to external state)
├─ Wake latency: 100s of microseconds
└─ Power savings: 95%+ in idle

VDDCX standby (retention):
├─ Minimal leakage power
├─ State retained in L1-V caches
├─ Wake latency: ~1 microsecond
└─ Power reduction: 80-90%

Clock gating:
├─ VDDCX/VDDMX still powered
├─ Clock stopped (0 dynamic power)
├─ Wake latency: 10s of nanoseconds
└─ Power reduction: ~50-70% (leakage remains)
```

### 5.6 Clock Domains

Hexagon operates in independent clock domains for frequency scaling.

**Clock Architecture:**

```
Clock Tree:

┌───────────────────────────────────┐
│ Crystal Oscillator (19.2 MHz)    │
└────────────┬──────────────────────┘
             │
             ▼
    ┌────────────────┐
    │ PLL (Phase-    │
    │ Locked Loop)   │
    └────────┬───────┘
             │
             ▼
    ┌────────────────────────┐
    │ Frequency Dividers/    │
    │ Multiplexers           │
    └─┬──────────────────┬───┘
      │                  │
      ▼                  ▼
  ┌──────────┐      ┌──────────┐
  │ Hexagon  │      │ Other    │
  │ clk      │      │ clks     │
  │ (0.4-2.0│      │          │
  │  GHz)    │      └──────────┘
  └──────────┘

Hexagon Clock Scaling:

Frequency steps (typical Snapdragon 8 Gen 2):
┌──────────┬─────────┬─────────┐
│ Frequency│ VDDCX   │ Use Case│
├──────────┼─────────┼─────────┤
│ 300 MHz  │ 0.60V   │ Idle    │
│ 600 MHz  │ 0.75V   │ Light   │
│ 1.2 GHz  │ 0.85V   │ Moderate│
│ 1.8 GHz  │ 1.00V   │ Peak    │
└──────────┴─────────┴─────────┘

DVFS (Dynamic Voltage/Frequency Scaling):
• Managed by PMC/TLMM (GPIO/regulator) hardware
• Software triggers via DCVS driver
• Minimum voltage = minimum safe frequency
• Frequency ramp up/down: ~10s of microseconds
```

---

## 6. Generational Differences (v65-v75)

### 6.1 ISA Feature Evolution

| Feature | v65 | v66 | v68 | v69 | v73 | v75 |
|---------|-----|-----|-----|-----|-----|-----|
| **Core** |
| Clock | 1.8G | 1.9G | 1.8G | 2.0G | 2.1G | 2.2G |
| Threads | 4 | 4 | 4 | 4 | 4 | 4 |
| L1-I Cache | 32K | 32K | 32K | 32K | 32K | 32K |
| L1-D Cache | 32K | 32K | 32K | 32K | 32K | 32K |
| L2 Cache | 256K | 256K | 512K | 512K | 512K | 512K |
| **HVX** |
| Width | 128b | 128b | 256b | 256b | 256b | 256b |
| Registers | 32 | 32 | 32 | 32 | 32 | 32 |
| Ops/cycle | 4 | 4 | 8 | 8 | 8 | 8 |
| **HTA/HMX** |
| Present | No | No | Yes | Yes | Yes | Yes |
| Type | - | - | HTA | HTA | HTA | HMX |
| MatMul Perf | - | - | ~7 TMAC/s | ~7.5 TMAC/s | ~8.4 TMAC/s | ~10 TMAC/s |
| **Memory** |
| TCM | No | No | 0-32K | 0-32K | 0-32K | 0-32K |
| VTCM | No | No | 0-128K | 0-128K | 0-128K | 0-256K |
| **ISA** |
| HVX v60 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| HVX v62 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| HVX v65 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| HVX v66 | - | ✓ | ✓ | ✓ | ✓ | ✓ |
| HVX v68 | - | - | ✓ | ✓ | ✓ | ✓ |
| HVX v69 | - | - | - | ✓ | ✓ | ✓ |
| HVX v73 | - | - | - | - | ✓ | ✓ |
| HVX v75 | - | - | - | - | - | ✓ |

### 6.2 Hexagon v65 vs v66 (Minor Iteration)

v66 is primarily a minor clockspeed and ISA refinement of v65.

**Key Additions in v66:**
```c
// New HVX v66 instructions
v0 = vrndnh(v0)                // Round-to-nearest-half
v0 = vaslh(v0, r0)             // Variable arithmetic shift
v0 = vfma(v0, v1, v2)          // Fused multiply-accumulate

// Scalar improvements
r0 = sfmpy(r1, r2)             // Scalar FMA (floating-point)
```

**Cache/Memory improvements:**
- I-TLB: 64 → 64 (no change)
- D-TLB: 32 → 32 (no change)
- L1 latency: 1 cycle (no change)

### 6.3 Hexagon v68 (Major Step: HTA, 256-bit HVX)

v68 introduces significant compute acceleration capabilities.

**Major v68 Features:**

1. **HVX Width Expansion: 128b → 256b**
   ```c
   // v65/v66: 128-bit operations
   HVX_Vector v0(128b), v1(128b), v2(128b);
   v0 = Q6_V_vadd_VV(v1, v2);  // 128-bit ADD

   // v68+: 256-bit operations (4× throughput)
   v0 = Q6_V_vadd_VV_256(v1, v2);  // 256-bit ADD
   // Single instruction does 8 x 32-bit ops vs 4 x 32-bit
   ```

2. **HTA (Hexagon Tensor Accelerator)**
   ```c
   // Tensor operations (new in v68)
   // Controlled via HTA-specific instructions

   // Setup matrix multiply
   // A: input activation matrix (128×K)
   // B: weight matrix (K×128)
   // C: output accumulator (128×128)

   // In pseudocode:
   hta_matmul_setup(A_base, B_base, C_base);
   hta_matmul_execute(m, k, n, data_type);
   hta_wait_complete();
   ```

3. **TCM/VTCM Support (Optional)**
   ```c
   // Tightly-coupled memory access
   // VTCM: 0-128 KB dedicated for vector operations

   #pragma VTCM
   int8_t weight_tensor[128][128];  // Guaranteed 1-cycle access

   // Use in vector operations
   v0 = Q6_V_vmem_A(weight_tensor);
   ```

4. **Dual-Store Support (v68)**
   ```c
   // Two stores in single packet
   {
       memw(r0) = r1    // Store to address r0
       memw(r2) = r3    // Store to address r2 (dual-store)
   }
   // v65/v66: Error (only 1 LSU)
   // v68+: Valid if r0 != r2 and no bank conflict
   ```

### 6.4 Hexagon v69 (Incremental: Slight Perf Boost)

v69 is a modest speed/power optimization over v68.

**v69 Improvements:**
```
Clock increase: 1.8 GHz → 2.0 GHz
HTA throughput: +5% (improved MAC latency scheduling)

New instructions (few):
r0 = ashiftr(r1, r2, #width)  // Dynamic width shift
v0 = vmpy_qfx(v1, v2)         // Q-format multiply (fixed-point)
```

### 6.5 Hexagon v73 (Refinement: Better Cache)

v73 focuses on memory and control flow efficiency.

**v73 Changes:**
```
L2 cache: 512 KB (unchanged from v68)
Clock: 2.1 GHz
New branch prediction features:
  • Extended BTB (Branch Target Buffer)
  • Improved pattern history

New memory prefetch instructions:
  dcfetch(r0)    // Prefetch for data access
```

### 6.6 Hexagon v75 (Latest: HMX Expansion, Larger VTCM)

v75 is the latest generation with major improvements to tensor acceleration and memory.

**v75 Major Additions:**

1. **HMX Expansion (vs v68 HTA)**
   ```c
   // HMX: Dedicated matrix accelerator (more capable than HTA)
   // INT8:  28.8 TMAC/s (v75 @ 1.8 GHz, 16×16K MACs/cycle)
   // FP16:  14.4 TFLOPS
   // TF32:  3.6 TFLOPS (new in v75)

   // New data types for inference
   TF32 (19-bit, Google TPU format)
   INT4 (4-bit quantized)
   INT2 (2-bit extreme quantization)
   ```

2. **VTCM Expansion: 128 KB → 256 KB**
   ```c
   // Larger vector TCM for bigger working sets
   #pragma VTCM
   int8_t weights[256][256];  // Now fits in VTCM (256×256=64KB)
   // v68: Would need external L2 access
   // v75: Fits in dedicated VTCM
   ```

3. **Improved FP32 Performance**
   ```c
   // Floating-point enhancements
   r0 = fpround(r1)           // Better rounding control
   v0 = vfma_fp32(v1, v2, v3) // Vector FMA (better precision)
   ```

4. **New Vector Instructions (HVX v75)**
   ```c
   // Advanced permutation
   v0 = vshuffeh(v1, v2)     // Shuffle even/odd
   v0 = vtranspose(v0, v1)   // Matrix transpose

   // New bit manipulation
   v0 = vpopcnt(v1)          // Population count (Hamming weight)
   v0 = vlz(v1)              // Leading zeros
   ```

### 6.7 ISA Extension Timeline Summary

```
v60 ──→ v62 ──→ v65 ──→ v66 ──→ v68 ──→ v69 ──→ v73 ──→ v75
       (earlier)           (HTA intro)         (current)

Major milestones:
┌────────────────────────────────────────────────────────────┐
│ v68: Game-changer                                          │
│      • HVX 128→256 bits (4× peak throughput)              │
│      • HTA tensor accelerator                             │
│      • VTCM for working set locality                      │
│      • Dual-store support                                 │
│                                                            │
│ v75: Latest & Greatest                                    │
│      • HMX (more capable HTA)                             │
│      • VTCM 128→256 KB                                    │
│      • TF32, INT4 quantization support                    │
│      • Improved floating-point                            │
└────────────────────────────────────────────────────────────┘
```

---

## 7. Self-Assessment Questions

### PhD-Level Qualifier Exam Style Questions

**Section 1: Processor Architecture**

1. **Explain the 6-stage scalar pipeline with emphasis on forwarding paths and hazard resolution. How many cycles are required for a load-use dependency to be resolved? What is the minimum stall latency for a store-load dependency?**

2. **The Hexagon processor implements 4-way hardware multithreading. Explain (a) why threads don't execute simultaneously, (b) how instruction packets are interleaved, (c) what happens when a single thread experiences a 20-cycle L2 cache miss, and (d) the register isolation mechanism.**

3. **Compare and contrast HVX, HTA, and HMX. For each, provide:**
   - Peak throughput (ops/cycle) at 1.8 GHz
   - Typical latency for primary operations
   - Data width and register count
   - Optimal use cases for neural network inference

4. **The VLIW packet structure allows up to 4 instructions per 128-bit packet. Explain:**
   - Slot assignment rules and dual-issue constraints
   - Why only 1 LSU can issue per cycle (mostly)
   - The dual-store exception in v68+ and its constraints
   - How the compiler/assembler detects valid vs invalid packets

**Section 2: Memory Hierarchy**

5. **Diagram and explain the Hexagon memory hierarchy latencies. Given a sequence of memory operations:**
   ```c
   r0 = memw(r1)       // Cycle N (L1 hit)
   r2 = memw(r3)       // Cycle N+1
   r4 = add(r0, r2)    // When does this start executing?
   ```
   **Explain the forwarding path and identify all possible stalls.**

6. **The D-cache is organized as a dual-bank structure. Explain (a) the bank interleaving scheme, (b) how a load and store can both execute simultaneously, (c) what constitutes a bank conflict, and (d) the victim buffer role in write-back.**

7. **Compare TCM and VTCM in terms of address space, access patterns, and latency characteristics. Why can't the scalar core use VTCM?**

8. **L2 cache has an 8-way associative design with 512 KB (v68+). Calculate:**
   - Number of sets
   - Bits used for set index, line offset, and tag
   - Miss penalty if request must go off-core (100 cycles)
   - Expected miss rate for audio DSP algorithms (craft answer)

**Section 3: ISA and Instruction-Level Details**

9. **Explain the predicate execution model. Given:**
   ```c
   p0 = cmp.eq(r0, #5)
   if (p0) r1 = add(r2, r3)
   if (!p0) r1 = sub(r2, r3)
   ```
   **How many cycles does this execute? Can both branches be issued in the same packet? Explain.**

10. **Hexagon branch prediction has ~95% accuracy for loops. Explain:**
    - The structure of the BTB (branch target buffer)
    - Misprediction penalty and recovery sequence
    - How hardware loops (loop0/endloop0) achieve zero-overhead looping
    - The role of prediction in prefetching I-cache

11. **NOP cycles and stall hazards: Given:**
    ```c
    r0 = mpyh(r1, r2)    // 3-cycle latency
    r3 = add(r0, r4)     // Read r0
    ```
    **Explain the dependency tracking hardware and minimum stall cycles.**

12. **Analyze the instruction encoding for: `r0 = add(r1, #1000)`. Explain:**
    - How 16-bit immediates are encoded
    - Alternative instruction sequences for 32-bit immediates
    - Why large constants require multiple instructions

**Section 4: Multi-Threading and Synchronization**

13. **Explain the round-robin scheduler from first principles. If Thread 0 experiences a 15-cycle L2 miss on Cycle 5, what is the packet issue sequence for Cycles 5-20?**

14. **Register file banks: The 8 KB register file is split into 4 banks (one per thread). Explain:**
    - Physical address mapping for `R5` in each thread
    - Why no inter-thread forwarding is implemented
    - How synchronization primitives (locks) rely on shared memory

**Section 5: SoC Integration**

15. **Hexagon connects to the SoC via a NoC (network-on-chip). Explain:**
    - How L2 cache misses are translated to NoC requests
    - Priority/fairness arbitration in the NoC
    - SMMU translation flow: VA → IPA → PA
    - Bandwidth sharing with CPU/GPU (typical ratios)

16. **Power domains VDDCX and VDDMX: Explain the difference and power gating/standby scenarios for each.**

17. **DMA engines provide off-core data movement. Explain:**
    - Configuration registers (source, destination, length, stride)
    - How 2D transfers work with strides
    - Peak throughput and typical use (loading weights, activations)

**Section 6: Generational Progression**

18. **Trace the evolution from v65 → v68 → v75. For each, identify:**
    - New execution units introduced
    - ISA extensions (with code examples)
    - Peak throughput metrics

19. **Why was HVX width expansion (128→256 bits) a turning point for Hexagon as an ML accelerator?**

20. **The VTCM expansion to 256 KB in v75: Calculate the weight matrix sizes that now fit. Compare inference latency for a 128×128 INT8 MatMul with weights in VTCM vs L2 cache.**

### Answers Framework (Guidance Only)

**Q1 Answer Framework:**
- L1 hit latency: 2 cycles (E4 generate, M5 hit)
- Load-to-use forward: 1 cycle
- Store-load dependency: 2-3 cycle minimum (ordering guarantees)
- Forwarding from W6 to E4 of next packet

**Q3 Answer Framework:**
- HVX: 16 ops/cycle (256b FP32), ~3 cycle latency, 32×256b regs, SIMD DSP ops
- HTA/HMX: ~7-10 TMAC/sec (INT8 matrix), 50-100 cycle latency, MAC arrays, tensor ops
- Comparison: HVX for general DSP, HTA/HMX specialized for conv/linear layers

**Q13 Answer Framework:**
```
Thread 0: P0↓ . . . . . P1
Thread 1: . P0 . . . P1 .
Thread 2: . . P0 . P1 . .
Thread 3: . . . P0 P1 . .
...
(skip T0 for ~4 rotations while L2 access pending)
```

---

## References and Further Reading

**Official Qualcomm Documentation:**
- Hexagon V73 Architecture Specification (HVX v73, ISA ref)
- Snapdragon 8 Gen 2 Technical Reference Manual
- Hexagon HVX Programmer's Reference Manual
- Hexagon Tensor Accelerator (HTA) Programming Guide

**Memory Subsystem:**
- Qualcomm Hexagon Cache & Memory Hierarchy Design Brief
- ARM SMMU v3 Specification (for IOMMU details)

**Performance Tuning:**
- Qualcomm Hexagon Performance Optimization Guide
- HVX Code Optimization Handbook (vector scheduling)

**Related ISA References:**
- VLIW architecture fundamentals (Fisher et al., 2005)
- Hardware multithreading survey (Ungerer et al., 2003)
- Tensor accelerator design patterns

---

**Document Version:** 1.0
**Last Updated:** March 2026
**Target Audience:** PhD-level engineers, advanced ML systems researchers
**Assumption:** Strong background in computer architecture, VLIW ISAs, and C/C++ programming

