# MODULE 6 — Out-of-Order Execution: The Heart of Modern CPUs

## 1. CONCEPTUAL FOUNDATION

Out-of-order (OOO) execution is the dominant execution model in modern high-performance processors. Unlike in-order pipelines (Module 5), OOO processors execute instructions based on **dataflow** (when operands are ready) rather than program order. This enables exploitation of **instruction-level parallelism (ILP)** beyond what in-order designs achieve.

### The Fundamental Problem: Serialization Under Dependencies

In an in-order pipeline, a true data dependency forces serialization:

```
Program order:
I1: r1 = r2 + r3       (latency 1 cycle)
I2: r4 = r1 * r5       (depends on r1, cannot start until I1 result available)

Timeline:
Cycle:  1  2  3  4  5  6  7
I1:     IF ID EX MEM WB
I2:           IF (stall) ID EX MEM WB

I2 must wait in ID stage for 2 cycles until r1 is available (forwarding reduces to 1 cycle).
Critical path latency: 3 cycles (EX of I1 + 1 cycle stall + EX of I2)
```

In-order execution cannot overlap I1 and I2; they form a **dependency chain**. Execution is **serialized**.

### Out-of-Order Execution: Breaking Program Order

An OOO processor decouples **instruction fetch/decode order** from **execution order**:

1. **Fetch and decode** instructions in program order into a **reorder buffer (ROB)**
2. **Execute** instructions as soon as operands are ready (dataflow order)
3. **Commit** results back to architectural state in program order

```
Program order:
I1: r1 = r2 + r3       (latency 1)
I2: r4 = r1 * r5       (depends on r1)
I3: r6 = r7 + r8       (independent of I1, I2)

In-order execution:
Cycle:  1  2  3  4  5  6  7
I1:     IF ID EX MEM WB
I2:           (stall) ID EX MEM WB
I3:                   IF ID EX MEM WB

Out-of-order execution:
Cycle:  1  2  3  4  5  6  7
I1:     IF ID EX MEM WB
I2:        IF (wait) (wait)  EX MEM WB
I3:        IF ID    EX MEM WB

I3 executes immediately after fetch (independent); I2 waits for I1 result.
Total: 7 cycles vs 8 cycles (12.5% speedup just from reordering I3)
```

### Register Renaming: Eliminating False Dependencies

**True dependencies** (RAW: read-after-write) are unavoidable. But **false dependencies** (WAR, WAW) arise from limited architectural registers.

**Scenario: WAR (Write-After-Read) false dependency**

```c
int x = a + b;      // I1: write r1
int y = c + d;      // I2: write r1 (reuse r1)
int z = x + 1;      // I3: read r1 (needs I1 value, not I2)
```

In-order execution:
```
I1: r1 = r2 + r3       (reads r2, r3; writes r1)
I2: r1 = r4 + r5       (reads r4, r5; writes r1)
I3: r6 = r1 + 1        (reads r1)

If I3 reads r1 before I2 writes, reads wrong value (WAR hazard).
Force serial: I1 → I2 → I3
```

**Register renaming solution**:

Allocate **physical registers** (P0, P1, P2, ...) from a pool of 160+ registers (vs 16 architectural registers). Each write gets a new physical register.

```
Architectural: r1, r2, r3, r4, r5, r6
Physical pool: P0 - P159

Rename:
I1: r1 = r2 + r3       →   P16 = P0 + P1       (r1 → P16)
I2: r1 = r4 + r5       →   P17 = P2 + P3       (r1 → P17, new physical register)
I3: r6 = r1 + 1        →   P18 = P17 + 1       (reads latest r1, which is P17)
```

Now I1 and I2 write to **different physical registers** (P16 vs P17). I3 reads P17 (the correct value). I1 and I2 can execute in parallel (no WAR conflict).

**Citation**: Mutlu's OOO execution lectures (CMU 18-447) detail register renaming with formal algorithms. Sohi & Smith's seminal paper "The Microarchitecture of Superscalar Processors" (1995) provides rigorous treatment. H&P Chapter 3 covers Tomasulo's algorithm.

### Tomasulo's Algorithm: The Heart of OOO Execution

Tomasulo's algorithm (1967, refined by modern processors) manages OOO execution:

1. **Reservation Stations (RS)**: Buffers that hold instructions waiting for operands
   - Each execution unit (ALU, multiplier, load/store) has multiple RS (Skylake: 4-8 per unit)
   - Instruction enters RS when decoded
   - When both operands ready, RS broadcasts instruction to execution unit

2. **Common Data Bus (CDB)**: Broadcasts all execution results
   - Any instruction in any RS can read from CDB (eliminates WAR hazards via renaming)
   - Results propagate to dependent instructions in parallel

3. **Reorder Buffer (ROB)**: Tracks all in-flight instructions in program order
   - Commits results to architectural state in order
   - Ensures **precise exception semantics** (if I5 faults, I1-I4 complete, I5+ don't)

4. **Register Renaming Tags**: Physical register mappings
   - When instruction writes, gets a new physical register (tag)
   - Dependent instructions use tags instead of architectural registers

**Data flow in Tomasulo's algorithm**:

```
1. Decode & Dispatch
   I1: add r1, r2, r3    → RS#4 (ALU), tag=P16
   I2: mul r4, r1, r5    → RS#3 (MUL), tag=P17 (waits for P16 from I1)

2. Execute when operands ready
   I1: add EX → CDB broadcasts P16 = (r2 + r3)
   RS#3 sees P16 on CDB, marks I2's operand ready

3. I2 executes
   I2: mul EX → CDB broadcasts P17 = (P16 * r5)

4. Commit in order
   ROB retires I1 (r1 = P16), then I2 (r4 = P17)
```

**Critical insight**: Execution happens out-of-order (I2 starts after I1 finishes), but **all results are made visible in program order** (I1's result before I2's).

### Execution Ports: The Functional Unit Bottleneck

Modern processors replicate execution units to support superscalar execution:

**Intel Skylake (6-wide superscalar)**:
- Port 0: ALU, shift, branch
- Port 1: ALU, multiply
- Port 2: Load (AGU)
- Port 3: Load (AGU)
- Port 4: Store address (AGU)
- Port 5: ALU, shift
- Port 6: Branch, ALU
- (Ports 7-8 reserved for stores, load/store coordination)

**AMD Zen 4**:
- 4 ALU ports (Ports 0-3)
- 3 AGU ports (Ports 4-6)
- 2 Load/Store ports (Ports 7-8)

This is **structural parallelism**: 6-8 instructions can execute per cycle across multiple ports. OOO execution chooses which instructions to send to which ports.

**Implication for ILP**: Maximum theoretical IPC is limited by port count (6 on Skylake), not by dependency chains alone.

## 2. MENTAL MODEL

```
OOO EXECUTION PIPELINE OVERVIEW
════════════════════════════════════════════════════════════

Fetch Queue        Reorder Buffer (ROB)    Execution Units
┌─────────────┐    ┌────────────────────┐  ┌──────────────┐
│ I1, I2, I3  │    │ I1 P16 ← P0 + P1   │  │ Port 0: ALU  │
│ I4, I5, I6  │    │ I2 P17 ← P16 * P2  │  │ Port 1: ALU  │
│             │    │ I3 P18 ← P3 + P4   │  │ Port 2: LSU  │
│ max 64 uops │    │ ...                │  │ Port 3: LSU  │
└──────────────┘   │ max 512 entries    │  │ Port 4: AGU  │
      ↓            │ (Intel Raptor Lake)│  │ Port 5: ALU  │
 Decode Stage      │                    │  │ Port 6: ALU  │
      │            │ Commit when result │  └──────────────┘
      ↓            │ available & in order   Reservation
Dispatch Queue     │                    │  Stations
      │            └────────────────────┘  ┌──────────────┐
      ↓                                    │  RS (x8-12)  │
Register Renaming  REGISTER RENAMING      │  Track ready  │
      │            ┌────────────────────┐  │  operands    │
      ↓            │ Arch r1 → Phys P16 │  │              │
Reservation        │ Arch r2 → Phys P0  │  │ When both    │
Stations           │ Arch r3 → Phys P1  │  │ operands OK  │
      │            │ ...                │  │ → dispatch   │
      ↓            │ 160+ phys regs     │  │   to port    │
Execute            └────────────────────┘  └──────────────┘
      │
      ↓
Common Data Bus (CDB)
      │
      ↓ [results broadcast to all waiting instructions]
      │
      ↓
Write-Back + ROB Retire (in order)
      │
      ↓
Architectural State (r1, r2, ..., r16)

TOMASULO'S ALGORITHM: INSTRUCTION FLOW
════════════════════════════════════════════════════════════

Decode:
    add r1, r2, r3
    → RS (ALU): P16 = P0 + P1 (P0, P1 are physical mappings of r2, r3)

Wait:
    If P0 and P1 not ready: Instruction waits in RS until values arrive on CDB

Execute:
    When P0 and P1 on CDB: RS broadcasts instruction to Port 0 (ALU)
    ALU computes P0 + P1 → P16 (stores result in physical register P16)
    CDB broadcasts: P16 = <result>

Register Update:
    All waiting instructions that depend on r1 see P16 on CDB
    Their input gets updated from "waiting" to <result value>

Commit:
    When instruction reaches head of ROB and result available:
    r1 ← P16 (rename r1 to map to P16)

DEPENDENCY CHAIN vs OUT-OF-ORDER EXECUTION
════════════════════════════════════════════════════════════

In-order (no parallelism):
    I1: r1 ← r2 + r3     (latency 1)
    I2: r4 ← r1 * r5     (depends on r1, latency 3)
    I3: r6 ← r7 + r8     (independent)

    Timeline:
    Cycle: 1 2 3 4 5 6 7
    I1:    . EX. . . .
    I2:    . . . EX. . .
    I3:    . . . . EX. .

    Total: 7 cycles, 3 instructions, CPI = 2.33

Out-of-order (parallelism):
    I1: r1 ← r2 + r3
    I2: r4 ← r1 * r5
    I3: r6 ← r7 + r8

    Timeline:
    Cycle: 1 2 3 4 5 6
    I1:    . EX. . . .
    I2:    . . . EX. .
    I3:    . EX. . . .    ← executes in parallel with I1

    Total: 6 cycles, 3 instructions, CPI = 2.0 (17% faster)

    Out-of-order reorders I3 to execute early (independent path)

MEMORY DISAMBIGUATION
════════════════════════════════════════════════════════════

Challenge: Load L1 and store S1 have unknown addresses at decode time
L1: r1 = mem[r2 + offset]        (address unknown at fetch)
S1: mem[r3 + offset] = r4        (address unknown at fetch)

Can we reorder? Depends on whether r2 ≠ r3 at runtime.

OOO solution:
1. Issue L1 and S1 out of order (don't block on address)
2. Track load/store queue: detect addresses at execution time
3. If addresses overlap: Forward (if S1 executes first, L1 reads S1's data)
                   OR Flush (if L1 executed speculatively before S1, results invalid)
4. Modern CPUs: Speculative execution of loads; recover if store detected later

Load Queue:
    ┌─────────────────────────┐
    │ L1: addr=?, data=wait   │
    │ L2: addr=?, data=wait   │
    │ L3: addr=?, data=ready  │
    └─────────────────────────┘
                 ↑ When L3 gets address, checks if any prior store to same address

INSTRUCTION WINDOW AND ROB SIZE
════════════════════════════════════════════════════════════

Instruction window = number of instructions in flight simultaneously

Small window (32): Fewer instructions in flight → lower ILP potential
    - In-order CPU: window = pipeline depth (5-8 instructions)
    - Limited parallelism

Large window (512): More instructions in flight → higher ILP potential
    - Skylake: 512-entry ROB
    - Zen 4: 320-entry ROB
    - Can find parallelism across many instructions (e.g., independent array accesses)

ROB size is key bottleneck:
    - Too small: ROB fills, blocks fetch → pipeline stalls
    - Too large: More power, area, complexity
    - Skylake (512) found to be near-optimal for 6-wide fetch
```

## 3. PERFORMANCE LENS

### Out-of-Order Execution's Performance Impact

The fundamental performance metric is **effective IPC** (instructions per cycle). OOO execution achieves higher IPC than in-order by exploiting parallelism:

**Metric 1: Instruction-Level Parallelism (ILP)**

$$\text{ILP} = \frac{\text{average number of independent instructions}}{\text{execution latency}}$$

For example, consider a sequence with 3 independent additions (latency 1 each):

$$\text{ILP} = \frac{3}{1} = 3 \text{ instructions per cycle available}$$

Modern 6-wide processors with 6 execution ports can achieve IPC up to 6 if ILP ≥ 6.

**Metric 2: CPI under OOO Execution**

$$\text{CPI}_{\text{OOO}} = \text{CPI}_{\text{in-order}} \times \text{(1 - stall\_reduction)}$$

Typical improvement: 30-50% CPI reduction for memory-bound code, 50-70% for compute-bound code with good ILP.

**Metric 3: ROB Saturation**

If ROB fills (all 512 entries occupied), fetch must stall. This creates a **bottleneck**:

$$\text{Fetch stall} = \begin{cases}
0 & \text{if } \#\text{in-flight instructions} < \text{ROB size} \\
\text{cycles stalled} & \text{if } \#\text{in-flight} \geq \text{ROB size}
\end{cases}$$

In memory-bound code (loads take 60+ cycles), the ROB fills quickly because in-flight instructions accumulate. Stalls reduce effective throughput.

### Performance Implications of Register Renaming

Renaming enables **parallel execution** of WAW and WAR hazards:

**Without renaming** (4 architectural registers):
```c
r1 = a + b;    (write r1)
r2 = c + d;    (write r2)
r1 = r2 * e;   (WAR: must wait for first read of r1 in later instruction, can't start)
```

**With renaming** (160+ physical registers):
```c
P1 = a + b;    (write P1, map r1 → P1)
P2 = c + d;    (write P2, map r2 → P2)
P3 = P2 * e;   (write P3, map r1 → P3; no conflict with P1)
```

Now the second write to r1 (P3) is independent and can execute in parallel.

### Performance Implications of Memory Disambiguation

Loads and stores are not truly independent if they overlap in address space. Speculation is required:

**Scenario: Speculative Load Execution**

```c
L1: r1 = mem[r2];          (address unknown at fetch)
S1: mem[r3] = r4;          (address unknown at fetch)
```

OOO processor **speculates** L1 executes before S1 (loads go before stores). If r2 ≠ r3, correct. If r2 = r3:
- L1 fetches stale data (before S1's store)
- Results are invalid
- Pipeline flushes, re-executes from S1 onward (15-20 cycle penalty)

**Performance impact**:
- If r2 ≠ r3 (usual case): L1 and S1 execute in parallel, good throughput
- If r2 = r3 (rare case, ~1-5% in typical code): 15-cycle flush penalty

## 4. ANNOTATED CODE EXAMPLES

### Example 4a: Register Renaming Effect

```c
#include <stdint.h>

// Scenario: Register reuse without renaming (simulated in-order)
// Assume 4 architectural registers (r1, r2, r3, r4)
int32_t compute_no_rename(int32_t a, int32_t b,
                           int32_t c, int32_t d) {
    // Forced to reuse registers (in-order dependency)
    int32_t r1 = a + b;      // Write r1
    int32_t r2 = c + d;      // Write r2 (independent, could parallel)
    int32_t r3 = r1 * r2;    // Read r1 (depends on first write)
    int32_t r1 = r3 + 10;    // Write r1 AGAIN (WAW: must wait for previous r1 write)
    return r1;
}

// In-order execution (no renaming):
// Cycle:  1  2  3  4  5  6  7  8  9
// r1←a+b: IF ID EX MEM WB
// r2←c+d: IF ID EX MEM WB       (independent, but can't start until r1 done)
// r3←r1*r2: IF(stall) ID EX MEM WB
// r1←r3+10:     IF(stall) ID EX MEM WB
//
// Total: 9 cycles, 4 instructions, CPI = 2.25

// Scenario: Register renaming (OOO execution)
int32_t compute_with_rename(int32_t a, int32_t b,
                             int32_t c, int32_t d) {
    int32_t r1_v1 = a + b;        // P1 = a + b (maps r1 → P1)
    int32_t r2_v1 = c + d;        // P2 = c + d (maps r2 → P2)
    int32_t r3 = r1_v1 * r2_v1;   // P3 = P1 * P2 (maps r3 → P3)
    int32_t r1_v2 = r3 + 10;      // P4 = P3 + 10 (maps r1 → P4, no conflict with P1)
    return r1_v2;
}

// Out-of-order execution (with renaming):
// Cycle:  1  2  3  4  5  6  7
// P1←a+b: IF ID EX MEM WB
// P2←c+d: IF ID EX MEM WB       (parallel, no dependency on r1)
// P3←P1*P2: IF ID EX MEM WB     (depends on P1, P2; can start cycle 3)
// P4←P3+10: IF ID EX MEM WB     (depends on P3; can start cycle 4)
//
// Total: 7 cycles, 4 instructions, CPI = 1.75
// 22% speedup vs in-order (7 vs 9 cycles)

// Assembly (x86-64, -O3 with OOO):
// add eax, ebx          ; P1 = a + b
// add ecx, edx          ; P2 = c + d  (parallel)
// imul eax, ecx         ; P3 = P1 * P2
// add eax, 10           ; P4 = P3 + 10

// The first add and second add execute in parallel on different ALU ports
// No register pressure (2 outputs, 4 inputs, plenty of physical registers)
```

**Key insight**: Register renaming breaks the WAW dependency between the two writes to r1. Without renaming, the second write must wait for the first to complete. With renaming, they write to different physical registers and can execute independently.

### Example 4b: Tomasulo's Algorithm in Action

```c
// Simulate Tomasulo's algorithm for this sequence:
// I1: add r1, r2, r3       (latency 1, Port 0)
// I2: mul r4, r1, r5       (latency 3, Port 1, depends on r1 from I1)
// I3: add r6, r7, r8       (latency 1, Port 2, independent)

#include <stdio.h>

struct ReservationStation {
    int instruction_id;
    int operand1_ready;
    int operand2_ready;
    int latency;
    int result_physical_reg;
};

// Simulated execution timeline:
// Cycle 1: Decode
//   I1 dispatched to RS#0 (ALU): P16 = P0 + P1 (both P0, P1 ready, assume from prior loads)
//   I2 dispatched to RS#1 (MUL): P17 = P16 + P2 (P16 NOT ready, waits for I1)
//   I3 dispatched to RS#2 (ALU): P18 = P3 + P4 (both ready)

// Cycle 1, end: I3 starts execution (independent)
// Cycle 2, end: I1 finishes, broadcasts P16 on CDB
//   RS#1 sees P16 on CDB: marks P16 ready
//   I2 can now start execution (both operands ready)
// Cycle 2, also: I3 finishes (started at cycle 1, 1-cycle latency)

// Cycle 3, end: I2 finishes (started at cycle 2, 3-cycle latency)

// Timeline:
// Cycle: 1 2 3 4 5
// I1:    D E . . .     (D=decode, E=execute)
// I2:    D . . E .     (waits for I1 on CDB)
// I3:    D E . . .     (executes in parallel with I1)
//
// CDB broadcasts:
//   Cycle 2: P16 (from I1)
//   Cycle 3: P18 (from I3)
//   Cycle 5: P17 (from I2)

// Commit (in order):
// Cycle 6: I1 commits: r1 ← P16
// Cycle 7: I2 commits: r4 ← P17
// Cycle 8: I3 commits: r6 ← P18

void simulate_tomasulo_algorithm() {
    printf("Tomasulo Simulation\n");
    printf("Cycle | I1 (add) | I2 (mul) | I3 (add) | CDB\n");
    printf("------|----------|----------|----------|--------\n");
    printf("1     | Dispatch | Dispatch | Dispatch | ------\n");
    printf("2     | Execute* | Wait     | Execute* | P16\n");
    printf("3     | Commit   | Execute  | Commit   | P18\n");
    printf("4     |          | Execute  |          | ------\n");
    printf("5     |          | Commit   |          | P17\n");

    // Key insight: I3 executes at the same time as I1 and I2
    // In-order would take 5+ cycles; OOO takes 5 cycles total
    // Speedup depends on parallelism available
}

// Example: Tracing CDB broadcasts
// When I1 finishes (cycle 2):
//   CDB broadcasts: { destination=P16, value=<result of I1> }
//
// All RS entries check CDB:
//   RS#1 (I2): "My source1 is P16" → Updates operand1 = <result>
//   RS#2 (I3): "My sources are P3, P4, not P16" → No update
//
// This broadcast mechanism is the core of Tomasulo's algorithm

int main() {
    simulate_tomasulo_algorithm();
    return 0;
}
```

### Example 4c: Memory Disambiguation and Store-to-Load Forwarding

```c
#include <stdint.h>
#include <stdio.h>

// Scenario: Load and store, potentially overlapping addresses
void memory_disambiguation_example(int32_t* arr, int idx1, int idx2) {
    // S1: store to arr[idx1]
    arr[idx1] = 42;

    // L1: load from arr[idx2]
    int32_t value = arr[idx2];

    // Are idx1 and idx2 the same? Unknown at compile/decode time.
    // OOO CPU must speculate whether L1 can bypass S1 or must wait.
}

// OOO execution (memory disambiguation):
//
// Scenario A: idx1 != idx2 (addresses don't overlap)
// Cycle: 1  2  3  4  5  6  7
// S1:    ID EX MEM WB
// L1:    ID EX MEM WB          (can execute in parallel, different addresses)
//        └─→ independent, both execute simultaneously
//
// Scenario B: idx1 == idx2 (addresses overlap)
// Ideal (with forwarding):
// Cycle: 1  2  3  4  5  6
// S1:    ID EX MEM WB
//           └─→ Data from S1 available at end of EX/MEM
// L1:    ID EX [forward] ────→ L1 reads value from S1 (0 stall)
//
// Speculation failure: L1 executed before S1 resolved, must flush
// Cycle: 1  2  3  4  5  6  7  8  9
// S1:    ID EX MEM WB
// L1:    ID EX MEM WB          (wrongly read from cache, result invalid)
//              ↑ flush triggered, restart at cycle 6

void store_forwarding_demo() {
    int32_t arr[64] __attribute__((aligned(64)));

    // Scenario 1: Clear forward (same index, stores to loads)
    arr[0] = 100;
    int32_t x = arr[0];  // Forwards from store above, 0 latency

    // Scenario 2: Index-based forwarding (address computed at runtime)
    for (int i = 0; i < 64; i++) {
        arr[i] = i * 2;
        int32_t val = arr[i];  // Can forward if loop iteration is predictable

        // OOO CPU: if same i, forwards
        //          if different i, separate addresses, parallel execution
    }

    // Scenario 3: Uncertain (address computed late)
    int32_t idx = compute_index();  // Address unknown until here
    arr[idx] = 42;
    int32_t z = arr[idx];  // Must wait for idx calculation + store address gen + load

    // Load queue tracking:
    // Load Queue Entry:
    // │ LQE#0: arr[idx] → Waiting for address
    // │ LQE#1: arr[idx2] → Address = 0x7fff0000 (known)
    //
    // When LQE#0 gets address (from idx calculation):
    // → Check store queue: "Is there a store to 0x7fff0000 pending?"
    // → Yes: Forward from store, or wait if store not ready
    // → No:  Load from cache (L1, L3, etc.)
}

// Intel Skylake load-store queue size: 72-entry load queue, 42-entry store queue
// AMD Zen 4: 60-entry load queue, 44-entry store queue
//
// These queues track in-flight loads and stores; when full, block new loads/stores
```

### Example 4d: Instruction Window and ROB Saturation

```c
// Scenario: Large loop with many in-flight loads (ROB saturation)

void matmul_roob_saturation(float* A, float* B, float* C, int N) {
    // Matrix multiplication: C += A * B
    // Inner loop: Load A[i][k], B[k][j], multiply, accumulate to C[i][j]

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            float sum = C[i*N + j];
            for (int k = 0; k < N; k++) {
                // Sequence of operations:
                float a_val = A[i*N + k];      // Load 1: 60 cycles (L3 hit)
                float b_val = B[k*N + j];      // Load 2: 60 cycles (L3 hit)
                sum += a_val * b_val;           // Dependent on both loads
            }
            C[i*N + j] = sum;
        }
    }

    // Analysis:
    // Per iteration of k-loop (3 instructions):
    // - Load A: 60 cycles latency
    // - Load B: 60 cycles latency
    // - Multiply-add: 1 cycle latency (but depends on loads)
    //
    // In-flight instructions at any time:
    // If k-loop has 256 iterations, and each iteration takes 60+ cycles:
    // - Iteration 0: load_A, load_B (cycle 0-1)
    // - Iteration 1: load_A, load_B (cycle 1-2)
    // - ... continue fetching ...
    // - Iteration 60: load_A, load_B (cycle 60-61) → In-flight instructions: ~180
    //
    // Skylake ROB size: 512
    // This example would have ~180 in-flight instructions (ROB fill at 35%)
    //
    // But if N is large and k-loop is deep enough:
    // In-flight instructions could exceed 512 → ROB fills
    // → Fetch stalls until earlier instructions retire
    // → Pipeline efficiency drops due to fetch stall

    // ROB saturation occurs when:
    // (load latency) * (loop depth) > ROB size
    // Example: 60 cycles * 10 iterations = 600 instructions
    // Exceeds 512-entry ROB → fetch stalls
    //
    // This is MEMORY LATENCY AMPLIFICATION:
    // A 60-cycle memory latency becomes a fetch stall if loop is deep enough
}

// Impact: ROB saturation reduces effective IPC
// Before saturation: IPC = ~4-5 (4 loads in parallel, good port utilization)
// After saturation: IPC = 1-2 (fetch blocked, pipeline efficiency collapses)
```

## 5. EXPERT INSIGHT

### Insight 1: Register Renaming is NOT "Free" (False Dependency Breaking Costs)

Register renaming requires:
- **Rename table** (16 entries → 160+ physical registers): high-frequency lookup (20-30 GPU cycles on modern processors)
- **Free list** of available physical registers: must be updated on every register write
- **ROB** to track which physical register maps to which architectural register: must be updated on commit

Each of these operations has **latency** and **area cost**. Modern CPUs spend 5-10% of die area on rename hardware.

**Implication**: WAW and WAR hazards ARE eliminated by renaming, but this doesn't come for free. The processor must maintain consistency between architectural and physical registers during speculative execution and recovery.

**In ML systems**: When code has many short-lived registers (e.g., temporary variables in tight loops), renaming pressure increases. Modern compilers optimize to minimize register pressure by reusing registers. This is sometimes at odds with parallelism optimization.

### Insight 2: Memory Disambiguation Speculates Aggressively (And Recovers Expensively)

Modern processors assume loads execute **before** all prior stores (conservative but useful assumption for parallelism). If a load is discovered to depend on a later store, the processor must:

1. Flush the load result and all dependent instructions
2. Restart execution from the store
3. Cost: 10-20 cycles of pipeline flush + re-execution

**Quantitative**: A 5% store-to-load forwarding miss rate in a deeply nested loop:
- Base execution: 1000 instructions, 10 cycles
- Forwarding misses: 50 instructions × 1 (per miss), but 5 complete flushes × 15 cycles = 75 cycles overhead
- Total penalty: 75 / 10010 ≈ 0.75% execution time loss

But this is CONSERVATIVE. Studies show real-world forwarding miss rates are 0.1-1%, making speculation highly profitable.

**Implication**: Compact storage layouts (e.g., struct-of-arrays vs array-of-structs) improve memory disambiguation by reducing address aliasing likelihood.

### Insight 3: The ROB is a Power-Hungry Bottleneck (Not a Pure Speedup Source)

Modern ROB sizes (512 entries on Skylake, 320 on Zen 4) seem large, but they **saturate quickly** in memory-bound code:

**ROB saturation scenario**:
```
Instructions in flight = (memory latency) × (fetch width) / (commit rate)
                       = 60 cycles × 6 instructions/cycle / 4 commit/cycle
                       = 90 instructions average
```

But **peaks** are higher. A single L3 miss creates a 60-cycle bubble. If 10 loads miss:
- Total in-flight: 600+ instructions
- Exceeds 512-entry ROB → fetch stalls

**Power implication**: A 512-entry ROB consumes **5-8 watts** on modern processors (Skylake). This is ~5% of total processor power for a single structure.

**Insight**: Larger ROBs have diminishing returns. AMD Zen 4's 320-entry ROB vs Skylake's 512 has only 3-5% performance loss for typical code, but 20-30% less power in the ROB hardware.

### Insight 4: Execution Port Utilization is the Bottleneck, Not ILP

Modern processors have 6-8 execution ports but **cannot fully utilize** all ports simultaneously due to instruction mix and dependencies.

**Reality check**: In practice:
- Integer-heavy code: Ports 0, 1, 5, 6 utilized; ports 2-4 (LSU/AGU) unused → IPC capped at 4
- Load-heavy code: Ports 2-4 (LSU) utilized; ports 0, 1, 5, 6 (ALU) unused → IPC capped at 3-4
- Mixed code: All ports potentially used, but difficult to sustain high IPC

**Measurement**: `perf stat -e port_utilization` reveals bottleneck:
```
port_utilization.0c_1c_2c: 0.3   (0-2 ports utilized per cycle: LOW utilization)
port_utilization.3c_4c: 0.2     (3-4 ports utilized: MEDIUM)
port_utilization.5c_6c: 0.1     (5-6 ports utilized: HIGH, rare)
```

**Implication for ML**: Vectorization (SIMD) is crucial because it packs more work into the same ports. A single AVX-512 instruction (8 elements) uses 1 port but does 8× the work of a scalar instruction on the same port.

### Insight 5: Commit Rate is a Often-Overlooked Bottleneck

**Commit bandwidth** (instructions per cycle that exit the ROB and become architectural) is limited by:
- **ROB port count** (typically 1-4 instructions/cycle can commit)
- **Exception handling** (if instruction faults, must flush all subsequent instructions; limits throughput)
- **Misprediction recovery** (recovery clears ROB; introduces stall)

In a 512-entry ROB with 4-instruction/cycle commit:
```
Max cycles before ROB fills = 512 / 6 (fetch) = 85 cycles
Max cycles before ROB empties = 512 / 4 (commit) = 128 cycles
```

If average latency is >128 cycles (e.g., DRAM hit), ROB can fill, causing fetch stalls.

**Implication**: Code with low commit rate (many exceptions, many mispredictions) artificially constrains ROB fill rate, causing fetch stalls even if ILP is high.

## 6. BENCHMARK / MEASUREMENT

### Measurement 1: ILP Extraction (Instructions Per Cycle)

```bash
# Measure IPC via perf on real hardware
perf stat -e cycles,instructions,uops_issued.any ./your_program

# Example output:
# 1,000,000 cycles
# 4,000,000 instructions
# IPC = 4,000,000 / 1,000,000 = 4.0

# Compare to in-order processor (IPC typically 0.8-1.2)
# OOO achieves 4.0x higher IPC on parallelizable code

# Further breakdown (Skylake PMCs):
perf stat -e core_power.cores,stalls_total ./program

# Identify bottleneck:
# - High IPC (>4): Good parallelism, check port utilization
# - Low IPC (<2): Bottleneck is data dependency or memory latency
```

### Measurement 2: ROB Occupancy (Instruction Window Fullness)

```bash
# Intel VTune Profiler (proprietary, but most accurate)
vtune -c general-exploration -knob enable-stack-collection=true \
      -knob sampling-interval=10 ./your_program

# Provides: ROB occupancy, RS fill rate, port utilization

# Linux perf (limited, but available):
perf stat -e resource_stalls.rob ./program
# Shows cycles ROB was full (fetch stalled)

# Interpretation:
# resource_stalls.rob > 10% of cycles → ROB is bottleneck
# Optimize by: reducing latency chains, increasing parallelism, or using faster memory
```

### Measurement 3: Memory Disambiguation Misses

```bash
# Linux perf (Intel Skylake):
perf stat -e ls_mispredicts,ld_blocks_overlap_resolution ./program

# Skylake PMCs:
# ls_mispredicts: Store-to-load forwarding prediction failures
# ld_blocks_overlap_resolution: Load stalled waiting for store address resolution

# Example:
# 10000 instructions
# 50 ls_mispredicts (0.5% rate)
# Expected penalty: 50 × 15 = 750 cycles
# Total impact: 750 / (10000 cycles / 3 IPC) ≈ 2.25% slowdown
```

### Measurement 4: Register Renaming Pressure

```c
#include <stdio.h>
#include <time.h>

// High register pressure: Many live values
void high_register_pressure(int* arr, int n) {
    int sum0 = 0, sum1 = 0, sum2 = 0, sum3 = 0,
        sum4 = 0, sum5 = 0, sum6 = 0, sum7 = 0;

    for (int i = 0; i < n; i++) {
        sum0 += arr[i * 8 + 0];
        sum1 += arr[i * 8 + 1];
        sum2 += arr[i * 8 + 2];
        sum3 += arr[i * 8 + 3];
        sum4 += arr[i * 8 + 4];
        sum5 += arr[i * 8 + 5];
        sum6 += arr[i * 8 + 6];
        sum7 += arr[i * 8 + 7];
    }
    // 8 live accumulators require 8 registers
    // Modern CPU: 160+ physical registers → easily accommodated
    // Old CPU with 32 physical registers: Would need spill/reload (slow)
}

// Low register pressure: Reuse registers
void low_register_pressure(int* arr, int n) {
    int sum = 0;

    for (int i = 0; i < n; i++) {
        sum += arr[i];
        // 1 live value → 1 register used
    }
}

int main() {
    int arr[10000];
    for (int i = 0; i < 10000; i++) arr[i] = i;

    // Measure high vs low pressure
    clock_t start = clock();
    high_register_pressure(arr, 1000);
    clock_t high_time = clock() - start;

    start = clock();
    low_register_pressure(arr, 1000);
    clock_t low_time = clock() - start;

    printf("High pressure: %ld cycles\n", (long)high_time);
    printf("Low pressure:  %ld cycles\n", (long)low_time);
    printf("Ratio: %.2fx\n", (double)high_time / low_time);

    // Expected: High pressure ~1.2x faster (better parallelism, 8 accums in parallel)
    //           Low pressure slower (serial accumulation)
    return 0;
}

// Compile and profile:
// gcc -O3 -march=native -o test test.c
// perf stat -e port_utilization ./test
```

## 7. ML SYSTEMS RELEVANCE

### Relevance 1: Batch Processing and Instruction Window Utilization

Transformer batch inference processes multiple sequences simultaneously:
```python
# Batched inference
batch_size = 32
seq_len = 128
for b in range(batch_size):
    for t in range(seq_len):
        output[b, t] = attention(queries[b, t], keys[b], values[b])
```

**OOO perspective**: The outer batch loop creates 32 independent attention computations. Each can proceed in parallel (different registers, independent memory accesses).

**ROB utilization**:
- Without batching: Single sequence, limited ILP, ROB occupancy ~50-100 instructions
- With batching: 32 sequences, each with ~10 in-flight operations → 320 in-flight instructions, ROB occupancy ~350 (near saturation on Skylake)

Batching **increases instruction window utilization**, allowing the processor to exploit more parallelism.

### Relevance 2: Embedding Lookup Interleaving and Memory Disambiguation

Embedding lookups in transformer decoders often serialize due to address dependencies:

```python
token_ids = [...]  # 32-token batch
embeddings = []
for token_id in token_ids:
    embed = embedding_table[token_id]  # Load token_id, use as address
    embeddings.append(embed)
```

**Memory disambiguation analysis**:
- Each lookup: Load token_id (4 cycles), compute address (1 cycle), load embedding (60 cycles) = ~65 cycles serial
- With 32 tokens: Total time if serialized = 32 × 65 = 2080 cycles

**OOO optimization**:
- Interleave lookups: Fetch token_ids in a batch first, then embeddings
```python
token_ids_batch = [...]  # All 32 token IDs
embeddings = [embedding_table[tid] for tid in token_ids_batch]  # Parallel loads
```

Now:
- 32 token ID loads: cycle 0-4 (parallel on 2 load ports, 16 instructions)
- 32 embedding loads: cycle 4-64 (parallel, 32 in-flight load instructions)
- Total: ~64 cycles (not 2080) → 32× speedup

**OOO benefit**: Address dependencies are broken by reordering; ROB provides the 32-instruction window for parallelism.

### Relevance 3: Attention Computation and Tomasulo-style Forwarding

Attention computes Q · K^T:

```c
float32_t result = 0.0f;
for (int i = 0; i < 64; i++) {
    float32_t q = Q[i];       // Load Q element
    float32_t k = K[i];       // Load K element
    result += q * k;          // Accumulate
}
```

**Tomasulo's algorithm in action**:
- Load Q[i]: RS entry (LSU #1), latency 4 cycles
- Load K[i]: RS entry (LSU #2), latency 4 cycles (parallel, different port)
- Multiply q*k: RS entry (Port 1), depends on both loads
- Accumulate result: RS entry (Port 0), depends on previous accumulation (latency 3)

**Timeline**:
```
Cycle: 1 2 3 4 5 6 7 8 9
Load Q: D E E . . . .  (MEM stage)
Load K: D E . . . E .
Mul:    D . (wait for both loads) EX . . .
Acc:    D . . . . EX .
```

**OOO benefit**: Q and K loads execute in parallel (Tomasulo's CDB broadcasts both simultaneously). Multiply starts as soon as both are ready (cycle 4-5). Without OOO, serialized dependency chain would take 15+ cycles.

### Relevance 4: LSTMs and Recurrent Dependencies

LSTMs have **recurrent state** that must be computed sequentially:

```python
h_t = output(h_{t-1}, x_t)  # Depends on previous hidden state
```

**OOO challenge**: Each timestep depends on the previous timestep's hidden state. This is an unavoidable **RAW dependency** (true dependency).

**Latency analysis**:
- LSTM cell: 4 matrix-vector products (each ~10 cycles), activations (2 cycles) = ~42 cycles
- Per timestep: h_{t-1} → h_t, then h_t → h_{t+1}
- Recurrence latency: 42 cycles per timestep

**OOO cannot parallelize recurrence** (it's a true dependency), but can:
1. **Parallelize internal operations**: 4 matrix products in LSTM cell can execute in parallel (different ports)
2. **Interleave multiple sequences**: Process 32 batch elements simultaneously, interleaving their recurrence chains

Example:
```python
h = [h_0, h_1, ..., h_31]  # 32 batch elements, different hidden states
for t in range(seq_len):
    h = [LSTM(h[b], x[b, t]) for b in range(32)]  # 32 in parallel
```

Now ROB contains 32 independent LSTM computations. Even though each single LSTM is serial (42 cycles), the 32 independent LSTMs execute in parallel.

### Relevance 5: Vectorization and Port Utilization in GEMM Kernels

Matrix multiplication (GEMM) is the workhorse of ML inference. Optimized GEMM kernels tile and unroll to maximize ILP:

```c
// Tiled GEMM with unrolling (Skylake-optimized)
for (int ii = 0; ii < N; ii += 64) {
    for (int jj = 0; jj < N; jj += 64) {
        for (int kk = 0; kk < N; kk += 64) {
            // Innermost loop: 4x4 unroll to exploit 4 ALU ports
            for (int i = ii; i < ii + 64; i += 4) {
                for (int j = jj; j < jj + 64; j += 4) {
                    for (int k = kk; k < kk + 64; k += 64) {
                        // 4×4 = 16 independent multiply-accumulates
                        C[i+0][j+0] += A[i+0][k] * B[k][j+0];
                        C[i+0][j+1] += A[i+0][k] * B[k][j+1];
                        C[i+0][j+2] += A[i+0][k] * B[k][j+2];
                        C[i+0][j+3] += A[i+0][k] * B[k][j+3];
                        // ... continue for i+1, i+2, i+3
                    }
                }
            }
        }
    }
}
```

**OOO execution**:
- 16 accumulators (C[i][j] for 4×4 tile): Can be stored in 16 different physical registers
- 16 independent multiply-accumulates: Can be issued to 4 ALU ports (4 instructions/cycle)
- Per 64-element block of k: 4 cycles to issue 16 multiply-accum, ~1 more cycle to commit results
- Total: Sustained ~3-4 IPC on 4 ALU ports

**Register renaming benefit**: 16 live accumulators require 16 physical registers (small fraction of 160+ available). No register spilling, no stalls.

**Port utilization**: 4 multiply-accum instructions per cycle → utilizing all 4 ALU ports. This is near-optimal for Skylake.

## 8. PhD QUALIFIER QUESTIONS

**Question 6.1**: Explain Tomasulo's algorithm in detail. Describe how it solves:
1. Register renaming for WAW and WAR hazards
2. True (RAW) dependencies through reservation stations and the CDB
3. Precise exception handling through the ROB

Draw a diagram showing a 3-instruction sequence:
```
I1: add r1, r2, r3       (ALU latency 1)
I2: mul r4, r1, r5       (MUL latency 3, depends on r1)
I3: sub r6, r2, r7       (ALU latency 1, independent)
```

Show reservation station state, CDB broadcasts, and ROB entries at each cycle. Assume 2 ALU ports (I1 and I3 can execute in parallel).

**Expected answer structure**:
- RS entries: Track instruction ID, operand ready bits, physical register sources
- When I1 finishes EX, broadcasts P16 on CDB; all RS entries listening for P16 update state
- I3 executes in parallel with I1 (independent operands)
- I2 executes after I1 (waits on P16 from CDB)
- ROB commits in order: I1 → I2 → I3
- Time: 5-6 cycles total (vs 7+ in-order)

---

**Question 6.2**: Register renaming eliminates WAW (write-after-write) and WAR (write-after-read) hazards. Explain with examples why these are "false" dependencies and how renaming makes them "disappear." Include a concrete code example showing speedup.

**Expected answer structure**:
- WAW: Two writes to same register
  ```
  r1 = a + b;    (write r1)
  r1 = c + d;    (write r1 again)
  ```
  Without renaming: Must serialize (1st write finishes before 2nd starts)
  With renaming: r1→P1 (first write), r1→P2 (second write), different registers, parallel execution

- WAR: Read then write to same register
  ```
  x = r1 + r2;   (read r1)
  r1 = c + d;    (write r1)
  ```
  Without renaming: Write must wait for read to complete
  With renaming: Read uses architectural r1, write creates new physical register, parallel execution

- Quantitative: Show 2-3 instruction sequence with CPI before/after renaming

---

**Question 6.3**: Modern out-of-order CPUs use memory disambiguation to execute loads speculatively before stores. Explain:
1. Why speculation is necessary (address dependencies unknown at decode)
2. How stores-to-loads forwarding works in hardware
3. What happens when speculation fails (misprediction recovery)
4. The performance cost of misprediction

Provide a code example where speculation succeeds and fails.

**Expected answer structure**:
- Speculation: Load issues before store address is known; assumes no conflict
- Forwarding: If store address matches load address, data bypassed from store queue
- Misprediction: Load executed too early, data came from cache (before store), results invalid
- Recovery: Flush load and all dependent instructions, re-execute (15-20 cycle cost)
- Quantitative: Store-to-load misprediction rate 0.5-2%, penalty 15+ cycles

---

**Question 6.4**: The reorder buffer (ROB) is a critical structure in OOO processors. Explain:
1. Why the ROB is necessary (not just reservation stations)
2. What happens when the ROB fills (how does fetch stall?)
3. How ROB size affects maximum in-flight instructions
4. The trade-off between larger ROB (more parallelism) and smaller ROB (lower power, area)

Calculate: For a processor with 512-entry ROB, 6-wide fetch, 4-wide commit, and average instruction latency 40 cycles (memory-bound workload), approximately how many instructions are in-flight on average? When does the ROB fill?

**Expected answer structure**:
- ROB tracks all in-flight instructions; necessary for precise exceptions (commit in order)
- ROB fills when incoming instructions > outgoing (retire rate) over sustained period
- Fetch stalls when ROB occupancy approaches max (typically 90%+ of 512)
- In-flight = (latency × fetch_width) / commit_rate = (40 × 6) / 4 = 60 instructions average
- ROB saturation time: 512 entries / (6 fetch/cycle - 4 commit/cycle) = 256 cycles after latency spikes
- Larger ROB: More potential ILP (if dependency chains are deep); diminishing returns beyond 512 on 6-wide fetch
- Skylake: 512-entry ROB; Zen 4: 320-entry (similar performance, lower power)

---

**Question 6.5**: Execution ports in superscalar processors are a finite resource. Explain:
1. How ports are allocated to different instruction types (ALU vs LSU vs FPU)
2. How the scheduler/dispatcher decides which instruction to send to which port
3. What "port pressure" means and how it limits IPC
4. Provide concrete examples from Skylake (6 ports) and Zen 4 (8 ports)

Calculate the maximum sustainable IPC for:
- Pure integer ALU code (all adds/subtracts)
- Pure load/store code (mixed loads and stores)
- Mixed code (50% ALU, 30% load, 20% store)

Assume instruction latencies and throughput from Agner Fog tables.

**Expected answer structure**:
- Ports: Fixed allocation to functional units (Port 0, 1 for ALU; Port 2, 3 for load; Port 4 for store, etc.)
- Scheduler: Logic that assigns decoded instructions to available ports in order of readiness
- Port pressure: When all instances of a needed port type are busy; instruction must stall waiting for port
- Skylake: 4 ALU-capable ports, 2 load ports, 1 store port
- Maximum IPC:
  - Pure ALU: min(4 ports, 1.0 latency) = 4 IPC
  - Pure load (60-cycle latency, 2 ports): Depends on dependencies; if independent, 2 IPC
  - Mixed: Weighted average of port requirements; typically 3-4 IPC
- Measurement: Use `perf stat -e port_utilization` to verify bottleneck

## 9. READING LIST

1. **Hennessy, J. L., & Patterson, D. A.** (2017). *Computer Architecture: A Quantitative Approach* (6th ed.). Chapter 3: "Instruction-Level Parallelism and Its Exploitation."
   - **Exact sections**: 3.1-3.5 on pipelining, 3.6-3.9 on out-of-order execution, Tomasulo's algorithm
   - Provides formal algorithms for register renaming, CDB operation, ROB management

2. **Mutlu, O.** (2017-2023). Carnegie Mellon 18-447 Lecture Series: "Out-of-Order Execution Fundamentals" (Lectures 7-11).
   - **Topics covered**: Tomasulo's algorithm, register renaming, memory disambiguation, ROB, execution ports
   - Includes microarchitecture-specific details (Skylake, Zen 4)

3. **Sohi, G. S., & Smith, J. E.** (2005). "The Microarchitecture of Superscalar Processors." *Proceedings of the IEEE*, 83(12), 1609-1624.
   - Seminal paper defining OOO execution, register renaming, memory disambiguation
   - **Key sections**: Section 2 (instruction window), Section 3 (register renaming), Section 4 (memory hierarchy)

4. **Agner Fog.** *Microarchitecture of Intel, AMD, and VIA CPUs: An Optimization Guide* (2023).
   - **Sections**: Chapter 2 (execution ports per architecture), Chapter 3 (instruction latency/throughput tables)
   - Provides exact numbers for Skylake, Ice Lake, SPR (Intel); Zen 4 (AMD)
   - **URL**: https://www.agner.org/optimize/microarchitecture.pdf

5. **Intel 64 and IA-32 Architectures Optimization Reference Manual** (2023).
   - **Sections**: Chapter 2.1-2.5 (microarchitecture details), Table 2-15 (execution ports)
   - Specifies ROB size, RS count, execution latencies per port

6. **AMD Zen 4 Processor Microarchitecture Reference Manual** (Public Specifications, 2022).
   - Specifies: ROB size (320 entries), execution port count (8), scheduler architecture
   - Compares to Zen 3 and Skylake for context

7. **Palacharla, S., Jouppi, N. P., & Smith, J. E.** (1997). "Complexity-Effective Superscalar Processors." *ACM SIGARCH Computer Architecture News*, 25(2), 206-218.
   - Analyzes complexity vs performance trade-offs in OOO design
   - Discusses why certain ROB sizes and port counts are optimal

---

**Module 6 Total Lines**: 1289 (comprehensive OOO execution coverage)

