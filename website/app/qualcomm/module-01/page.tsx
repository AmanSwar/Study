import Link from 'next/link'
import { ArrowRight } from 'lucide-react'
import { Breadcrumbs } from '@/components/layout/Breadcrumbs'
import { ModuleHeader } from '@/components/content/ModuleHeader'
import { TheorySection, SubSection } from '@/components/content/TheorySection'
import { CodeBlock } from '@/components/content/CodeBlock'
import { DataTable } from '@/components/content/DataTable'
import { CalloutBox } from '@/components/content/CalloutBox'
import { SelfAssessment } from '@/components/content/SelfAssessment'
import { KeyTakeaways } from '@/components/content/KeyTakeaways'
import { ReferenceList } from '@/components/content/ReferenceList'
import { TableOfContents } from '@/components/content/TableOfContents'

export const metadata = { title: 'Module 1: Hexagon SoC Architecture Internals' }

const tocItems = [
  { id: 'processor-anatomy', title: '1. Full Hexagon Processor Anatomy', level: 2 },
  { id: 'threading-model', title: '2. Hardware Threading Model', level: 2 },
  { id: 'memory-hierarchy', title: '3. Memory Hierarchy', level: 2 },
  { id: 'hexagon-isa', title: '4. The Hexagon ISA', level: 2 },
  { id: 'snapdragon-soc', title: '5. Hexagon NPU within Snapdragon SoC', level: 2 },
  { id: 'generational-differences', title: '6. Generational Differences (v65-v75)', level: 2 },
  { id: 'self-assessment', title: '7. Self-Assessment Questions', level: 2 },
  { id: 'references', title: 'References and Further Reading', level: 2 },
]

export default function Module01Page() {
  return (
    <>
      <Breadcrumbs
        items={[
          { label: 'Qualcomm', href: '/qualcomm' },
          { label: 'Module 1' },
        ]}
      />

      <div className="flex gap-0">
        <article className="flex-1 min-w-0">
          <ModuleHeader
            number={1}
            title="Hexagon SoC Architecture Internals"
            track="qualcomm"
            part={0}
            readingTime="60 min"
            prerequisites={['VLIW processors', 'Cache hierarchies', 'Memory systems', 'DSP concepts', 'C/C++ and computer architecture']}
            description="PhD-level reference on the Qualcomm Hexagon processor: scalar VLIW core, HVX vector unit, HTA/HMX tensor accelerators, hardware threading, memory hierarchy, ISA, SoC integration, and generational evolution from v65 to v75."
          />

          {/* ============ Section 1: Processor Anatomy ============ */}
          <TheorySection id="processor-anatomy" title="1. Full Hexagon Processor Anatomy">
            <SubSection id="processor-overview" title="1.1 Processor Core Overview">
              <p>
                The Qualcomm Hexagon processor is a 32-bit very long instruction word (VLIW) digital signal processor with
                specialized vector and tensor acceleration units. The microarchitecture consists of multiple distinct execution
                pipelines and specialized compute engines operating in parallel under a unified instruction dispatch system.
              </p>

              <h4 className="text-lg font-semibold text-text-primary mt-6 mb-3">Core Components:</h4>

              <p><strong>Scalar Execution Engine (32-bit integer/floating-point)</strong></p>
              <ul className="list-disc pl-6 my-4 space-y-1">
                <li>6-stage integer pipeline</li>
                <li>Separate floating-point pipeline (IEEE 754 compliant)</li>
                <li>ALU, multiplier, shifter, logic units</li>
              </ul>

              <p><strong>HVX (Hexagon Vector eXtensions)</strong></p>
              <ul className="list-disc pl-6 my-4 space-y-1">
                <li>128-bit or 256-bit SIMD vector unit (generation dependent)</li>
                <li>Dual-port access to unified L2 cache</li>
                <li>Dedicated vector registers (32/64 128-bit registers)</li>
              </ul>

              <p><strong>HTA (Hexagon Tensor Accelerator) / HMX (Matrix eXtensions)</strong></p>
              <ul className="list-disc pl-6 my-4 space-y-1">
                <li>Introduced in v68 and later</li>
                <li>Hardware acceleration for matrix operations</li>
                <li>INT8/FP16/TF32 support</li>
              </ul>
            </SubSection>

            <SubSection id="scalar-pipeline" title="1.2 Scalar Core Pipeline Architecture">
              <p>
                The scalar execution core implements a 6-stage pipeline with support for out-of-order execution within packet boundaries.
              </p>

              <CodeBlock language="text" title="Scalar Pipeline Stages">
{`Stage 1: FETCH     - Fetch 128-bit packet from I-cache
Stage 2: DECODE    - Decode up to 4 instructions, check dependencies
Stage 3: DISPATCH  - Assign instructions to execution units
Stage 4: EXECUTE   - Integer/logic operations, address generation
Stage 5: MEMORY    - L1 cache access
Stage 6: WRITEBACK - Register file update`}
              </CodeBlock>

              <p><strong>Detailed Pipeline Stages:</strong></p>
              <DataTable
                headers={['Stage', 'Operation', 'Duration', 'Notes']}
                rows={[
                  ['FETCH', 'I-cache access, I-TLB translation', '1 cycle', '128-bit aligned packets only'],
                  ['DECODE', 'Instruction parsing, immediate extraction', '1 cycle', 'Parallel decoding of 4 instructions'],
                  ['DISPATCH', 'Slot assignment, register renaming prep', '1 cycle', 'Resolves instruction class conflicts'],
                  ['EXECUTE', 'Computation, branch resolution', '1-3 cycles', 'Variable latency (MUL takes 2-3)'],
                  ['MEMORY', 'L1 D-cache or LSU operations', '2-3 cycles', 'Load latency = 2 cycles hit'],
                  ['WRITEBACK', 'Register file commit', '1 cycle', 'Synchronized across threads'],
                ]}
              />

              <h4 className="text-lg font-semibold text-text-primary mt-6 mb-3">ASCII Block Diagram of Scalar Pipeline:</h4>
              <CodeBlock language="text" title="Scalar Pipeline Block Diagram">
{`┌─────────────┐
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
   └──────────────────┘`}
              </CodeBlock>
            </SubSection>

            <SubSection id="hvx-architecture" title="1.3 HVX (Hexagon Vector eXtensions) Architecture">
              <p>
                HVX provides SIMD acceleration for digital signal processing tasks. The unit operates at the same clock as
                the scalar core but with independent execution pipelines.
              </p>

              <p><strong>HVX Specifications by Generation:</strong></p>
              <DataTable
                headers={['Generation', 'Width', 'Registers', 'Peak Throughput', 'Ops/Cycle']}
                rows={[
                  ['v65/v66', '128b', '32×128b', '256 bits/cycle', '8 FP32 ops'],
                  ['v68+', '256b', '32×256b', '512 bits/cycle', '16 FP32 ops'],
                ]}
              />

              <p><strong>HVX Register File Organization:</strong></p>
              <CodeBlock language="text" title="HVX Register File Layout">
{`Vector Register File (v68+, 256-bit width):
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
└──────────────────────────────────────────────────┘`}
              </CodeBlock>

              <p><strong>HVX Execution Pipelines:</strong></p>
              <p>The HVX unit contains multiple independent execution paths:</p>
              <ol className="list-decimal pl-6 my-4 space-y-1">
                <li><strong>Vector ALU Path</strong>: Arithmetic, logic, shift operations (1 cycle latency)</li>
                <li><strong>Vector Multiply Path</strong>: 256-bit multiply-accumulate (3 cycle latency)</li>
                <li><strong>Vector Memory Path</strong>: Cache access, streaming loads (2-3 cycle latency)</li>
                <li><strong>Vector Permute Path</strong>: Shuffles, rotations, interleaves (1 cycle latency)</li>
              </ol>

              <p><strong>HVX Instruction Encoding (Dual-Issue):</strong></p>
              <CodeBlock language="text" title="HVX Dual-Issue Combinations">
{`HVX allows dual-issue of certain instruction types:
┌─ Vector ALU + Vector Memory
├─ Vector ALU + Vector Multiply
├─ Vector Memory + Vector Memory (dual-load)
└─ Vector Multiply + Vector Multiply (not always)

Example: Load + Arithmetic in parallel
  v0.uh = vmem(r0+#0):128t    // Load from memory
  v1 = vadd(v2, v3)           // Can execute simultaneously`}
              </CodeBlock>
            </SubSection>

            <SubSection id="hta-hmx-architecture" title="1.4 HTA (Hexagon Tensor Accelerator) / HMX Architecture">
              <p>
                The Hexagon Tensor Accelerator (HTA), also known as HMX in later generations, provides dedicated hardware
                for matrix multiplication and tensor operations critical to deep learning inference.
              </p>

              <p><strong>HMX Specifications (v68+):</strong></p>
              <CodeBlock language="text" title="HMX Block Diagram">
{`HMX Block Diagram:

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
• FP16 MatMul: 8K MACs/cycle @ 1.8 GHz = 14.4 TFLOPS`}
              </CodeBlock>

              <p><strong>HMX vs HVX Comparison:</strong></p>
              <DataTable
                headers={['Aspect', 'HVX', 'HMX']}
                rows={[
                  ['Type', 'SIMD Vector', 'Tensor Array'],
                  ['Ops/Cycle', '16 (256b, FP32)', '16,000 (INT8 MAC)'],
                  ['Throughput', '512 bits', '16 KB (aggregated)'],
                  ['Latency', '1-3 cycles', '50-100 cycles'],
                  ['Prime Use', 'DSP, filters', 'MatMul, convolution'],
                  ['Data Width', '8/16/32/64 bits', '8-bit (INT8), 16-bit (FP16)'],
                ]}
              />

              <p><strong>HMX Instruction Format:</strong></p>
              <p>HMX is controlled via specialized tensor instructions that encode matrix dimensions and operation types:</p>
              <CodeBlock language="c" title="HMX Instruction Pseudocode">
{`// Pseudocode HMX instruction structure
struct HMX_MatMul {
    uint32_t opcode : 6;        // Operation type
    uint32_t m : 8;             // Rows in A (A is m×k)
    uint32_t k : 8;             // Common dimension
    uint32_t n : 8;             // Cols in B (B is k×n)
    uint32_t data_type : 2;     // INT8=0, FP16=1, TF32=2
};
// Max matrix: 256×256×256 INT8 MatMul in ~1000 cycles`}
              </CodeBlock>
            </SubSection>

            <SubSection id="unit-interrelationship" title="1.5 Interrelationship Between Processing Units">
              <CodeBlock language="text" title="Unified Instruction Issue Point">
{`Unified Instruction Issue Point:

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
5. Data movement between scalar and vector requires explicit moves`}
              </CodeBlock>
            </SubSection>

            <SubSection id="execution-resources" title="1.6 Functional Units and Execution Resources">
              <p><strong>Complete Execution Resource Summary:</strong></p>
              <DataTable
                headers={['Unit', 'Latency (cyc)', 'Throughput']}
                rows={[
                  ['ALU (Integer)', '1', '2 ops/cycle (dual)'],
                  ['ADD/SUB', '1', 'Dual-issue capable'],
                  ['AND/OR/XOR', '1', 'Dual-issue capable'],
                  ['Comparator', '1', '1 result/cycle'],
                  ['Multiplier (32)', '2-3', '1 result/cycle'],
                  ['Multiplier (64)', '3', '1 result/cycle'],
                  ['Load Store Unit', '2 (hit)', '1 LD + 1 ST/cyc'],
                  ['Float ALU', '1', '1 op/cycle'],
                  ['Float Mult', '2', '1 op/cycle'],
                  ['Float Convert', '2-3', '1 op/cycle'],
                  ['Float Divide', '10-12', '1/10 ops/cycle'],
                  ['Float Sqrt', '10-12', '1/10 ops/cycle'],
                  ['HVX ALU (256b)', '1', '1 op/cycle'],
                  ['HVX Mult (256b)', '3', '1 op/cycle'],
                  ['HVX LD/ST (256b)', '2-3', '1 op/cycle'],
                  ['HMX MatMul', '50-100', 'Dedicated ops/cyc'],
                ]}
                caption="Per-thread execution resource latency and throughput summary"
              />
            </SubSection>
          </TheorySection>

          {/* ============ Section 2: Hardware Threading Model ============ */}
          <TheorySection id="threading-model" title="2. Hardware Threading Model">
            <SubSection id="multithreading-overview" title="2.1 4-Way Hardware Multi-Threading Overview">
              <p>
                The Hexagon core implements a fine-grained hardware threading model where up to 4 independent threads share
                a single physical core. This is distinct from simultaneous multi-threading (SMT) in that threads are not
                executing instructions simultaneously — rather, the VLIW engine interleaves instruction packets from
                different threads in round-robin fashion.
              </p>

              <p><strong>Threading Architecture:</strong></p>
              <CodeBlock language="text" title="Hexagon Threading Architecture">
{`┌──────────────────────────────────────────────────────┐
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
└──────────────────────────────────────────────────────┘`}
              </CodeBlock>
            </SubSection>

            <SubSection id="vliw-interleaving" title="2.2 VLIW Packet Interleaving and Round-Robin Scheduling">
              <p>
                Instructions are fetched in 128-bit packets containing up to 4 instructions. The scheduler cycles through
                threads in round-robin order, issuing one packet per cycle (when available and not stalled).
              </p>

              <p><strong>Interleaving Timeline:</strong></p>
              <CodeBlock language="text" title="VLIW Packet Interleaving Timeline">
{`Cycle:   1  2  3  4  5  6  7  8  9 10 11 12
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
└─────────────────────────────────────────────────┘`}
              </CodeBlock>
            </SubSection>

            <SubSection id="register-banking" title="2.3 Register File Sharing and Banking">
              <p>
                The register file is physically partitioned into 4 banks (one per thread), but there is also shared state
                for control flow and special registers.
              </p>

              <p><strong>Register File Layout (v68 generation):</strong></p>
              <CodeBlock language="text" title="Physical Register File Layout">
{`Physical Register File (8 KB total SRAM):

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
• Data passed between threads via shared memory only`}
              </CodeBlock>
            </SubSection>

            <SubSection id="resource-sharing" title="2.4 Resource Sharing: Execution Units and Functional Units">
              <p>
                While threads have separate register files, they share all execution units. The dispatcher must resolve:
              </p>

              <p><strong>Conflict Types:</strong></p>
              <ol className="list-decimal pl-6 my-4 space-y-2">
                <li>
                  <strong>Functional Unit Conflicts</strong>: Two threads trying to use same ALU
                  <ul className="list-disc pl-6 my-2 space-y-1">
                    <li>Solution: Stall one thread, let other execute</li>
                    <li>Priority: Lower thread ID typically has priority</li>
                  </ul>
                </li>
                <li>
                  <strong>Cache Port Conflicts</strong>: Multiple threads accessing L1 cache
                  <ul className="list-disc pl-6 my-2 space-y-1">
                    <li>L1 D-cache: 2 read ports, 1 write port per cycle</li>
                    <li>Thread 0: priority for port 0</li>
                    <li>Thread 1: priority for port 1</li>
                    <li>Threads 2-3: share remaining port, time-division</li>
                  </ul>
                </li>
                <li>
                  <strong>Memory Bus Conflicts</strong>: L2/L3 access serialization
                  <ul className="list-disc pl-6 my-2 space-y-1">
                    <li>Single memory port to L2</li>
                    <li>Arbitration: Round-robin with priority boost for waiting threads</li>
                  </ul>
                </li>
              </ol>

              <p><strong>Execution Unit Allocation Policy:</strong></p>
              <CodeBlock language="text" title="Dispatch Decision Matrix">
{`Dispatch Decision Matrix:

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
   current_thread = (current_thread + 1) % 4`}
              </CodeBlock>
            </SubSection>

            <SubSection id="thread-priority" title="2.5 Thread Priority and Fairness">
              <p><strong>Dynamic Thread Prioritization:</strong></p>
              <CodeBlock language="c" title="Scheduling Algorithm Pseudo-C">
{`// Simplified scheduling algorithm (pseudo-C)
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
}`}
              </CodeBlock>

              <p><strong>Stall Cycle Accounting:</strong></p>
              <DataTable
                headers={['Event', 'Stall Duration', 'Why']}
                rows={[
                  ['L1 cache miss', '10-20 cycles', 'Memory hierarchy traversal'],
                  ['L2 cache miss', '30-100 cycles', 'Off-core access'],
                  ['Memory read dependency', '2 cycles', 'Load-to-use hazard'],
                  ['Branch misprediction', '3-5 cycles', 'Pipeline flush'],
                  ['Functional unit not ready', '1-3 cycles', 'Operand not available'],
                  ['Lock contention', 'Variable', 'Inter-thread synchronization'],
                ]}
              />
            </SubSection>

            <SubSection id="thread-isolation" title="2.6 Thread-Local Storage and Register Isolation">
              <p>Each thread maintains complete state isolation:</p>
              <CodeBlock language="text" title="Register Isolation Mechanism">
{`Register Isolation Mechanism:

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
│ Synchronization via shared memory + atomic ops`}
              </CodeBlock>
            </SubSection>
          </TheorySection>

          {/* ============ Section 3: Memory Hierarchy ============ */}
          <TheorySection id="memory-hierarchy" title="3. Memory Hierarchy">
            <SubSection id="cache-overview" title="3.1 Cache System Overview">
              <p>
                The Hexagon processor implements a two-level (L1/L2) cache hierarchy with separate instruction and data
                caches at L1, unified L2, and optional tightly-coupled memories.
              </p>

              <p><strong>Cache Hierarchy Diagram:</strong></p>
              <CodeBlock language="text" title="Hexagon Cache Hierarchy">
{`┌────────────────────────────────────────────────────┐
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
        └──────────────────────────────────┘`}
              </CodeBlock>
            </SubSection>

            <SubSection id="l1-instruction-cache" title="3.2 L1 Instruction Cache (I-Cache)">
              <p><strong>Specifications by Generation:</strong></p>
              <DataTable
                headers={['Aspect', 'v65/v66', 'v68', 'v69/v73', 'v75']}
                rows={[
                  ['Size', '32 KB', '32 KB', '32 KB', '32 KB'],
                  ['Associativity', '4-way', '4-way', '4-way', '4-way'],
                  ['Line Size', '32 B', '32 B', '32 B', '32 B'],
                  ['Fetch Width', '128 b', '128 b', '128 b', '128 b'],
                  ['Hit Latency', '1 cycle', '1 cycle', '1 cycle', '1 cycle'],
                  ['I-TLB Entries', '64', '64', '64', '64'],
                ]}
              />

              <p><strong>I-Cache Organization:</strong></p>
              <CodeBlock language="text" title="I-Cache Organization">
{`32 KB, 4-way I-Cache:
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
   (128-bit alignment required for packet fetch)`}
              </CodeBlock>

              <p><strong>I-Cache Access Path:</strong></p>
              <CodeBlock language="c" title="I-Cache Hit Scenario">
{`// I-Cache hit scenario (1 cycle)
Cycle N:   PC[31:10] → I-TLB lookup (parallel with cache)
           PC[9:2]  → Set selection
           PC[1:0]  = 00 (128-bit aligned)

Cycle N:   Compare tags in 4-way set
           Mux out matching way (128-bit data)
           Update LRU replacement state

Cycle N+1: Instruction decoder receives 128-bit packet`}
              </CodeBlock>
            </SubSection>

            <SubSection id="l1-data-cache" title="3.3 L1 Data Cache (D-Cache)">
              <p><strong>D-Cache Specifications:</strong></p>
              <DataTable
                headers={['Aspect', 'Configuration']}
                rows={[
                  ['Size', '32 KB'],
                  ['Associativity', '4-way'],
                  ['Line Size', '32 B'],
                  ['Access Width', '64-bit (dual-bank)'],
                  ['Hit Latency', '2 cycles (address generation in E4, hit in M5)'],
                  ['Write-Through/Back', 'Write-back'],
                  ['Victim Buffer', 'Yes, 4-entry'],
                  ['D-TLB', '32 entries, fully associative'],
                ]}
              />

              <p><strong>D-Cache Dual-Bank Organization:</strong></p>
              <CodeBlock language="text" title="D-Cache Dual-Bank Layout">
{`32 KB D-Cache (Dual-bank for parallel access):

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

Throughput: 1 load + 1 store per cycle (if non-conflicting banks)`}
              </CodeBlock>

              <p><strong>D-Cache Access Latency Details:</strong></p>
              <CodeBlock language="text" title="L1 D-Cache Hit Path">
{`L1 D-Cache Hit Path:

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
Load-to-use forward: 1 cycle (result available in next packet)`}
              </CodeBlock>
            </SubSection>

            <SubSection id="l2-unified-cache" title="3.4 L2 Unified Cache">
              <p>
                The L2 cache serves both instruction and data accesses, with an 8-way set-associative organization.
              </p>

              <p><strong>L2 Specifications by Generation:</strong></p>
              <DataTable
                headers={['Generation', 'Size', 'Assoc', 'Line', 'Hit Latency', 'Bandwidth']}
                rows={[
                  ['v65', '256 KB', '8-way', '64 B', '2 cyc', '128 bits'],
                  ['v66', '256 KB', '8-way', '64 B', '2 cyc', '128 bits'],
                  ['v68', '512 KB', '8-way', '64 B', '2 cyc', '128 bits'],
                  ['v69/v73', '512 KB', '8-way', '64 B', '2 cyc', '128 bits'],
                  ['v75', '512 KB', '8-way', '64 B', '2 cyc', '128 bits'],
                ]}
              />

              <p><strong>L2 Cache Structure:</strong></p>
              <CodeBlock language="text" title="L2 Cache Structure">
{`L2 Cache (512 KB example):

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
[5..0]:   Line offset (64 B = 2^6 bytes)`}
              </CodeBlock>

              <p><strong>L2 Hit/Miss Path:</strong></p>
              <CodeBlock language="text" title="L2 Miss Handling">
{`L2 Miss handling sequence:

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
  └─ Result available to dependent instructions in packet N+3`}
              </CodeBlock>
            </SubSection>

            <SubSection id="tcm-vtcm" title="3.5 TCM (Tightly Coupled Memory) and VTCM (Vector TCM)">
              <p><strong>TCM Purpose:</strong> Fast, predictable access for critical code/data. No cache miss latency.</p>

              <p><strong>TCM Specifications by Generation:</strong></p>
              <DataTable
                headers={['Generation', 'TCM Size', 'VTCM Size', 'Purpose']}
                rows={[
                  ['v65/v66', '0 KB', '0 KB', 'N/A'],
                  ['v68', '0-32 KB', '0-128 KB', 'Optional'],
                  ['v69/v73', '0-32 KB', '0-128 KB', 'Optional'],
                  ['v75', '0-32 KB', '0-256 KB', 'Optional'],
                ]}
              />

              <p><strong>TCM vs VTCM Distinction:</strong></p>
              <CodeBlock language="text" title="TCM vs VTCM Memory Map">
{`TCM (Tightly Coupled Memory):
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
└─────────────────────────────────┘`}
              </CodeBlock>

              <p><strong>TCM Access Example:</strong></p>
              <CodeBlock language="c" title="TCM/VTCM Access Example">
{`// TCM access - fixed 1 cycle latency
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
}`}
              </CodeBlock>
            </SubSection>

            <SubSection id="cache-coherence" title="3.6 Cache Coherence and Consistency">
              <p>
                Hexagon is a single-threaded core (cache-coherence semantics apply within threads only). Inter-thread
                synchronization is explicit.
              </p>

              <p><strong>Memory Barrier Instructions:</strong></p>
              <CodeBlock language="c" title="Hexagon Memory Fence Operations">
{`// Hexagon memory fence operations

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
  release_lock()    // Atomic`}
              </CodeBlock>
            </SubSection>
          </TheorySection>

          {/* ============ Section 4: The Hexagon ISA ============ */}
          <TheorySection id="hexagon-isa" title="4. The Hexagon ISA">
            <SubSection id="instruction-format" title="4.1 Instruction Format and Packet Structure">
              <p>
                The Hexagon ISA is structured around 128-bit <strong>packets</strong> containing up to 4 instructions, all
                executed in the same cycle (or stalled together if dependencies exist).
              </p>

              <p><strong>Packet Structure:</strong></p>
              <CodeBlock language="text" title="128-bit Packet Structure">
{`128-bit Packet:
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
   }`}
              </CodeBlock>

              <p><strong>Instruction Encoding (32-bit):</strong></p>
              <CodeBlock language="text" title="Instruction Encoding Format">
{`Standard instruction format:
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
Bits [27..0]: Actual instruction`}
              </CodeBlock>
            </SubSection>

            <SubSection id="slot-assignment" title="4.2 Instruction Slot Assignment Rules">
              <p>
                Critical to understanding Hexagon code generation: instructions must be assigned to valid slots based on their
                type and resource requirements.
              </p>

              <p><strong>Slot Assignment Table:</strong></p>
              <DataTable
                headers={['Slot', 'Resources', 'Valid Instruction Types']}
                rows={[
                  ['0', 'ALU, MUL', 'ADD, SUB, AND, OR, MUL, compare, shift, move'],
                  ['1', 'ALU, MUL', 'Same as Slot 0 (ALU can dual-issue)'],
                  ['2', 'LSU', 'LOAD, STORE, prefetch, atomic'],
                  ['3', 'ALU, MUL, Branch', 'Same as 0/1, or branch, call, return'],
                ]}
              />

              <p><strong>Dual-Issue Constraints:</strong></p>
              <CodeBlock language="text" title="Dual-Issue Combinations">
{`Valid dual-issue combinations within a packet:

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
   }  // INVALID packet - assembler error`}
              </CodeBlock>
            </SubSection>

            <SubSection id="addressing-modes" title="4.3 Register Operand Format and Addressing Modes">
              <p><strong>Register Operands:</strong></p>
              <CodeBlock language="text" title="Register Operands and Addressing Modes">
{`Integer registers: R0 - R31 (32 × 32-bit)
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
   r0 = memw(r2++M0)         // Load, r2 += M0, wrap at M1 boundary`}
              </CodeBlock>

              <p><strong>Immediate Value Encoding:</strong></p>
              <CodeBlock language="c" title="Immediate Encoding in Hexagon">
{`// Immediate encoding in Hexagon

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
//   r0 = or(r0, #0x5678)   // OR in lower bits`}
              </CodeBlock>
            </SubSection>

            <SubSection id="conditional-execution" title="4.4 Conditional Execution and Predicates">
              <p>
                Hexagon predicates (P0-P3) enable instruction-level conditional execution without branches.
              </p>

              <p><strong>Predicate Operations:</strong></p>
              <CodeBlock language="c" title="Predicate Operations">
{`// Predicate assignment
p0 = cmp.eq(r0, r1)         // p0 = (r0 == r1) ? 1 : 0
p1 = cmp.gt(r2, #100)       // p1 = (r2 > 100) ? 1 : 0

// Predicate combine
p2 = and(p0, p1)            // p2 = p0 AND p1
p3 = or(p0, p1)             // p3 = p0 OR p1

// Conditional instruction execution
if (p0) r0 = add(r1, r2)    // Execute only if p0 == 1
if (!p0) r0 = sub(r1, r2)   // Execute only if p0 == 0

// Conditional move
r0 = mux(p0, r1, r2)        // r0 = p0 ? r1 : r2`}
              </CodeBlock>

              <p><strong>Predicate Register Format:</strong></p>
              <CodeBlock language="text" title="Predicate Register Format">
{`P0-P3: Each is effectively an 8-bit register
┌───────────────┐
│ Bits [7..1]   │ Unused (reserved)
│ Bit [0]       │ Predicate value (0 or 1)
└───────────────┘

When used as condition:
if (p0) = if (p0[0] != 0)
if (!p0) = if (p0[0] == 0)`}
              </CodeBlock>
            </SubSection>

            <SubSection id="branch-prediction" title="4.5 Branch Instructions and Prediction">
              <p>
                Hexagon implements an efficient branch prediction unit integrated with the VLIW pipeline.
              </p>

              <p><strong>Branch Instruction Types:</strong></p>
              <CodeBlock language="c" title="Branch Instruction Types">
{`// Unconditional branches
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
endloop1                      // End loop1`}
              </CodeBlock>

              <p><strong>Branch Prediction Mechanics:</strong></p>
              <CodeBlock language="text" title="Branch Prediction Unit">
{`Prediction Unit:

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
           If wrong, I-cache pipeline flushes`}
              </CodeBlock>

              <p><strong>Hardware Loop Instructions:</strong></p>
              <CodeBlock language="c" title="Hardware Loop Structure">
{`// Hardware loop structure (zero-overhead loop)
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
// • Loop end address in LE0, LE1`}
              </CodeBlock>
            </SubSection>

            <SubSection id="hazards" title="4.6 NOP Cycles and Stall Hazards">
              <p><strong>Hazard Types and Solutions:</strong></p>
              <CodeBlock language="c" title="Hazard Types and Resolution">
{`// 1. Read-After-Write (RAW) hazard
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
}`}
              </CodeBlock>

              <p><strong>Instruction Dependency Tracking:</strong></p>
              <CodeBlock language="text" title="Hazard Detection Hardware">
{`Hazard Detection Hardware:

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
   └─ Shared functional unit`}
              </CodeBlock>
            </SubSection>

            <SubSection id="endianness" title="4.7 Endianness and Data Organization">
              <p>Hexagon is <strong>little-endian</strong>.</p>

              <p><strong>Data Layout:</strong></p>
              <CodeBlock language="text" title="Little-Endian Data Layout">
{`32-bit integer r0 = 0x12345678 (little-endian):
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
memd(0x1000) = 0xABCDEF0012345678 (r1:r0)`}
              </CodeBlock>
            </SubSection>
          </TheorySection>

          {/* ============ Section 5: Snapdragon SoC ============ */}
          <TheorySection id="snapdragon-soc" title="5. Hexagon NPU within Snapdragon SoC">
            <SubSection id="snapdragon-block" title="5.1 Snapdragon SoC Block Diagram">
              <p>
                The Hexagon processor is one compute engine within the larger Snapdragon SoC ecosystem.
              </p>

              <CodeBlock language="text" title="Snapdragon SoC Architecture">
{`Snapdragon SoC Architecture (Schematic):

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
│  │    ��─ Crossbar/mesh network                    │   │
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
└─────────────────────────────────────────────────────────┘`}
              </CodeBlock>
            </SubSection>

            <SubSection id="noc-architecture" title="5.2 Interconnect Architecture (NoC)">
              <p>
                The Hexagon processor connects to the rest of the SoC via a network-on-chip (NoC).
              </p>

              <p><strong>NoC Topology (Snapdragon 8 Gen 2 example):</strong></p>
              <CodeBlock language="text" title="NoC Mesh Interconnect">
{`NoC Mesh Interconnect:

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
• NoC arbitration ensures fair share`}
              </CodeBlock>
            </SubSection>

            <SubSection id="dma-engines" title="5.3 DMA (Direct Memory Access) Engines">
              <p>
                Hexagon includes dedicated DMA engines for off-core memory access independent of the processor pipeline.
              </p>

              <p><strong>DMA Engine Architecture:</strong></p>
              <CodeBlock language="text" title="Hexagon DMA System">
{`Hexagon DMA System:

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
• Status polling: Check "DMA_STATUS" register`}
              </CodeBlock>

              <p><strong>DMA Performance Profile:</strong></p>
              <DataTable
                headers={['Operation', 'Throughput', 'Latency', 'Notes']}
                rows={[
                  ['DMA burst (1D)', '128 bits/cycle', '~100 cycles setup', 'Coalesced'],
                  ['DMA 2D transfer', 'Same', 'Stride-dependent', 'Address calc/cycle'],
                  ['DMA → L2', '~200 GB/s peak', 'Depends on L2 state', 'Rarely bottleneck'],
                  ['DMA → DDR', '~32 GB/s (LPDDR5)', '50-100 cycles', 'Shared bandwidth'],
                ]}
              />

              <p><strong>DMA Code Example:</strong></p>
              <CodeBlock language="c" title="Hexagon DMA 2D Copy">
{`// Hexagon DMA initialization (pseudo-C)
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
}`}
              </CodeBlock>
            </SubSection>

            <SubSection id="smmu-iommu" title="5.4 SMMU (System Memory Management Unit) and IOMMU">
              <p>
                The SMMU provides virtual-to-physical address translation for Hexagon and other peripherals.
              </p>

              <p><strong>SMMU Configuration for Hexagon:</strong></p>
              <CodeBlock language="text" title="SMMU Hierarchy and Translation Flow">
{`SMMU Hierarchy:

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
                  Memory access`}
              </CodeBlock>
            </SubSection>

            <SubSection id="power-domains" title="5.5 Power Domains (VDDCX, VDDMX)">
              <p>Hexagon can be independently power-gated or frequency-scaled.</p>

              <p><strong>Power Domain Organization:</strong></p>
              <CodeBlock language="text" title="Power Distribution in Snapdragon">
{`Power Distribution in Snapdragon:

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
└─ Power reduction: ~50-70% (leakage remains)`}
              </CodeBlock>
            </SubSection>

            <SubSection id="clock-domains" title="5.6 Clock Domains">
              <p>Hexagon operates in independent clock domains for frequency scaling.</p>

              <p><strong>Clock Architecture:</strong></p>
              <CodeBlock language="text" title="Hexagon Clock Tree and DVFS">
{`Clock Tree:

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
• Frequency ramp up/down: ~10s of microseconds`}
              </CodeBlock>
            </SubSection>
          </TheorySection>

          {/* ============ Section 6: Generational Differences ============ */}
          <TheorySection id="generational-differences" title="6. Generational Differences (v65-v75)">
            <SubSection id="isa-feature-evolution" title="6.1 ISA Feature Evolution">
              <DataTable
                headers={['Feature', 'v65', 'v66', 'v68', 'v69', 'v73', 'v75']}
                rows={[
                  ['Clock', '1.8G', '1.9G', '1.8G', '2.0G', '2.1G', '2.2G'],
                  ['Threads', '4', '4', '4', '4', '4', '4'],
                  ['L1-I Cache', '32K', '32K', '32K', '32K', '32K', '32K'],
                  ['L1-D Cache', '32K', '32K', '32K', '32K', '32K', '32K'],
                  ['L2 Cache', '256K', '256K', '512K', '512K', '512K', '512K'],
                  ['HVX Width', '128b', '128b', '256b', '256b', '256b', '256b'],
                  ['HVX Registers', '32', '32', '32', '32', '32', '32'],
                  ['HVX Ops/cycle', '4', '4', '8', '8', '8', '8'],
                  ['HTA/HMX Present', 'No', 'No', 'Yes', 'Yes', 'Yes', 'Yes'],
                  ['Type', '-', '-', 'HTA', 'HTA', 'HTA', 'HMX'],
                  ['MatMul Perf', '-', '-', '~7 TMAC/s', '~7.5 TMAC/s', '~8.4 TMAC/s', '~10 TMAC/s'],
                  ['TCM', 'No', 'No', '0-32K', '0-32K', '0-32K', '0-32K'],
                  ['VTCM', 'No', 'No', '0-128K', '0-128K', '0-128K', '0-256K'],
                  ['HVX v60', '✓', '✓', '✓', '✓', '✓', '✓'],
                  ['HVX v62', '✓', '✓', '✓', '✓', '✓', '✓'],
                  ['HVX v65', '✓', '✓', '✓', '✓', '✓', '✓'],
                  ['HVX v66', '-', '✓', '✓', '✓', '✓', '✓'],
                  ['HVX v68', '-', '-', '✓', '✓', '✓', '✓'],
                  ['HVX v69', '-', '-', '-', '✓', '✓', '✓'],
                  ['HVX v73', '-', '-', '-', '-', '✓', '✓'],
                  ['HVX v75', '-', '-', '-', '-', '-', '✓'],
                ]}
              />
            </SubSection>

            <SubSection id="v65-vs-v66" title="6.2 Hexagon v65 vs v66 (Minor Iteration)">
              <p>v66 is primarily a minor clockspeed and ISA refinement of v65.</p>

              <p><strong>Key Additions in v66:</strong></p>
              <CodeBlock language="c" title="v66 New Instructions">
{`// New HVX v66 instructions
v0 = vrndnh(v0)                // Round-to-nearest-half
v0 = vaslh(v0, r0)             // Variable arithmetic shift
v0 = vfma(v0, v1, v2)          // Fused multiply-accumulate

// Scalar improvements
r0 = sfmpy(r1, r2)             // Scalar FMA (floating-point)`}
              </CodeBlock>

              <p><strong>Cache/Memory improvements:</strong></p>
              <ul className="list-disc pl-6 my-4 space-y-1">
                <li>I-TLB: 64 → 64 (no change)</li>
                <li>D-TLB: 32 → 32 (no change)</li>
                <li>L1 latency: 1 cycle (no change)</li>
              </ul>
            </SubSection>

            <SubSection id="v68" title="6.3 Hexagon v68 (Major Step: HTA, 256-bit HVX)">
              <p>v68 introduces significant compute acceleration capabilities.</p>

              <p><strong>Major v68 Features:</strong></p>

              <p><strong>1. HVX Width Expansion: 128b → 256b</strong></p>
              <CodeBlock language="c" title="v68 HVX Width Expansion">
{`// v65/v66: 128-bit operations
HVX_Vector v0(128b), v1(128b), v2(128b);
v0 = Q6_V_vadd_VV(v1, v2);  // 128-bit ADD

// v68+: 256-bit operations (4× throughput)
v0 = Q6_V_vadd_VV_256(v1, v2);  // 256-bit ADD
// Single instruction does 8 x 32-bit ops vs 4 x 32-bit`}
              </CodeBlock>

              <p><strong>2. HTA (Hexagon Tensor Accelerator)</strong></p>
              <CodeBlock language="c" title="v68 HTA Tensor Operations">
{`// Tensor operations (new in v68)
// Controlled via HTA-specific instructions

// Setup matrix multiply
// A: input activation matrix (128×K)
// B: weight matrix (K×128)
// C: output accumulator (128×128)

// In pseudocode:
hta_matmul_setup(A_base, B_base, C_base);
hta_matmul_execute(m, k, n, data_type);
hta_wait_complete();`}
              </CodeBlock>

              <p><strong>3. TCM/VTCM Support (Optional)</strong></p>
              <CodeBlock language="c" title="v68 VTCM Usage">
{`// Tightly-coupled memory access
// VTCM: 0-128 KB dedicated for vector operations

#pragma VTCM
int8_t weight_tensor[128][128];  // Guaranteed 1-cycle access

// Use in vector operations
v0 = Q6_V_vmem_A(weight_tensor);`}
              </CodeBlock>

              <p><strong>4. Dual-Store Support (v68)</strong></p>
              <CodeBlock language="c" title="v68 Dual-Store">
{`// Two stores in single packet
{
    memw(r0) = r1    // Store to address r0
    memw(r2) = r3    // Store to address r2 (dual-store)
}
// v65/v66: Error (only 1 LSU)
// v68+: Valid if r0 != r2 and no bank conflict`}
              </CodeBlock>
            </SubSection>

            <SubSection id="v69" title="6.4 Hexagon v69 (Incremental: Slight Perf Boost)">
              <p>v69 is a modest speed/power optimization over v68.</p>

              <p><strong>v69 Improvements:</strong></p>
              <CodeBlock language="text" title="v69 Improvements">
{`Clock increase: 1.8 GHz → 2.0 GHz
HTA throughput: +5% (improved MAC latency scheduling)

New instructions (few):
r0 = ashiftr(r1, r2, #width)  // Dynamic width shift
v0 = vmpy_qfx(v1, v2)         // Q-format multiply (fixed-point)`}
              </CodeBlock>
            </SubSection>

            <SubSection id="v73" title="6.5 Hexagon v73 (Refinement: Better Cache)">
              <p>v73 focuses on memory and control flow efficiency.</p>

              <p><strong>v73 Changes:</strong></p>
              <CodeBlock language="text" title="v73 Changes">
{`L2 cache: 512 KB (unchanged from v68)
Clock: 2.1 GHz
New branch prediction features:
  • Extended BTB (Branch Target Buffer)
  • Improved pattern history

New memory prefetch instructions:
  dcfetch(r0)    // Prefetch for data access`}
              </CodeBlock>
            </SubSection>

            <SubSection id="v75" title="6.6 Hexagon v75 (Latest: HMX Expansion, Larger VTCM)">
              <p>v75 is the latest generation with major improvements to tensor acceleration and memory.</p>

              <p><strong>v75 Major Additions:</strong></p>

              <p><strong>1. HMX Expansion (vs v68 HTA)</strong></p>
              <CodeBlock language="c" title="v75 HMX Capabilities">
{`// HMX: Dedicated matrix accelerator (more capable than HTA)
// INT8:  28.8 TMAC/s (v75 @ 1.8 GHz, 16×16K MACs/cycle)
// FP16:  14.4 TFLOPS
// TF32:  3.6 TFLOPS (new in v75)

// New data types for inference
TF32 (19-bit, Google TPU format)
INT4 (4-bit quantized)
INT2 (2-bit extreme quantization)`}
              </CodeBlock>

              <p><strong>2. VTCM Expansion: 128 KB → 256 KB</strong></p>
              <CodeBlock language="c" title="v75 Larger VTCM">
{`// Larger vector TCM for bigger working sets
#pragma VTCM
int8_t weights[256][256];  // Now fits in VTCM (256×256=64KB)
// v68: Would need external L2 access
// v75: Fits in dedicated VTCM`}
              </CodeBlock>

              <p><strong>3. Improved FP32 Performance</strong></p>
              <CodeBlock language="c" title="v75 FP32 Enhancements">
{`// Floating-point enhancements
r0 = fpround(r1)           // Better rounding control
v0 = vfma_fp32(v1, v2, v3) // Vector FMA (better precision)`}
              </CodeBlock>

              <p><strong>4. New Vector Instructions (HVX v75)</strong></p>
              <CodeBlock language="c" title="v75 New HVX Instructions">
{`// Advanced permutation
v0 = vshuffeh(v1, v2)     // Shuffle even/odd
v0 = vtranspose(v0, v1)   // Matrix transpose

// New bit manipulation
v0 = vpopcnt(v1)          // Population count (Hamming weight)
v0 = vlz(v1)              // Leading zeros`}
              </CodeBlock>
            </SubSection>

            <SubSection id="isa-timeline" title="6.7 ISA Extension Timeline Summary">
              <CodeBlock language="text" title="ISA Extension Timeline">
{`v60 ──→ v62 ──→ v65 ──→ v66 ──→ v68 ──→ v69 ──→ v73 ──→ v75
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
└────────────────────────────────────────────────────────────┘`}
              </CodeBlock>
            </SubSection>
          </TheorySection>

          <KeyTakeaways items={[
            'Hexagon is a 32-bit VLIW DSP with three primary compute domains: scalar (6-stage pipeline), HVX (SIMD vector unit, 128b → 256b in v68+), and HTA/HMX (tensor accelerator for matmul/conv).',
            '4-way fine-grained hardware multithreading interleaves packets in round-robin order — threads do not execute simultaneously, but stalls in one thread don\'t block others.',
            'Cache hierarchy: L1-I (32 KB, 4-way, 1-cycle), L1-D (32 KB, 4-way, 2-cycle, dual-bank), L2 (256-512 KB, 8-way, 2-cycle), plus optional TCM (scalar) and VTCM (vector) for guaranteed 1-cycle access.',
            'VLIW packets contain up to 4 instructions executed in parallel, with strict slot assignment rules (slots 0-3) and dual-issue constraints (e.g., only 1 LSU per cycle, except dual-store in v68+).',
            'HMX matrix accelerator delivers 28.8 TMAC/s INT8 in v75 — orders of magnitude beyond HVX SIMD throughput — making Hexagon competitive for production NN inference on mobile.',
            'v68 was the major architectural inflection point: 256-bit HVX, HTA introduction, VTCM, and dual-store. v75 expanded to HMX, 256 KB VTCM, and TF32/INT4 support.',
          ]} />

          {/* ============ Self-Assessment ============ */}
          <div id="self-assessment">
            <SelfAssessment questions={[
              {
                question: 'Section 1 — Q1: Explain the 6-stage scalar pipeline with emphasis on forwarding paths and hazard resolution. How many cycles are required for a load-use dependency to be resolved? What is the minimum stall latency for a store-load dependency?',
                answer: 'The scalar pipeline has stages FETCH→DECODE→DISPATCH→EXECUTE(E4)→MEMORY(M5)→WRITEBACK(W6). L1 hit latency is 2 cycles (E4 generates address, M5 returns hit data). Load-to-use forwarding allows the result to be available 1 cycle later (forwarded from M5/W6 to E4 of the dependent instruction). For store-load dependencies, the minimum stall is 2-3 cycles because the store must be ordered before the dependent load can issue — the store address queue must commit before the load can read the same address. The scoreboard tracks pending writes per register, stalling dependent instructions until the result is ready.',
              },
              {
                question: 'Section 1 — Q2: The Hexagon processor implements 4-way hardware multithreading. Explain (a) why threads don\'t execute simultaneously, (b) how instruction packets are interleaved, (c) what happens when a single thread experiences a 20-cycle L2 cache miss, and (d) the register isolation mechanism.',
                answer: '(a) Threads don\'t execute simultaneously because there is only one instruction issue port — the VLIW dispatcher issues one packet per cycle, picking from the next ready thread in round-robin order. This is fine-grained multithreading, not SMT. (b) The scheduler cycles T0→T1→T2→T3→T0... issuing one packet per thread per cycle, so a 4-thread workload sees each thread execute every 4th cycle. (c) When T0 has a 20-cycle L2 miss, the scheduler skips T0 for those 20 cycles while T1, T2, T3 continue normal round-robin. The other threads see no slowdown — this is the core benefit of hardware multithreading: latency hiding. (d) Register isolation: each thread has its own bank in the 8 KB physical register file (2 KB per thread). The same logical name R5 maps to different physical storage per thread. There is no inter-thread register forwarding by design; data passes between threads only via shared memory and atomic operations.',
              },
              {
                question: 'Section 1 — Q3: Compare and contrast HVX, HTA, and HMX. For each, provide: peak throughput (ops/cycle) at 1.8 GHz, typical latency for primary operations, data width and register count, and optimal use cases for neural network inference.',
                answer: 'HVX (Vector SIMD): 16 ops/cycle (256b FP32 in v68+), 1-3 cycle latency, 32 × 256b registers, optimal for general DSP, pre/post-processing, activation functions, layernorm, softmax. HTA (Hexagon Tensor Accelerator, v68): ~7-8 TMAC/s INT8 at 1.8 GHz, 50-100 cycle latency for primary matmul ops, dedicated MAC array (no GPRs — programmed via specialized instructions), optimal for convolution and dense linear layers. HMX (Matrix eXtension, v75): ~10 TMAC/s INT8 (16×16K MACs/cycle), 50-100 cycle latency, 16 KB activation + 8 KB weight buffers, optimized for production NN inference (Conv2D, GEMM, attention QKV projections). HVX is for general DSP work; HTA/HMX are specialized for the matmul-dominated portions of inference, providing orders of magnitude higher throughput than HVX SIMD for those workloads.',
              },
              {
                question: 'Section 1 — Q4: The VLIW packet structure allows up to 4 instructions per 128-bit packet. Explain: slot assignment rules and dual-issue constraints; why only 1 LSU can issue per cycle (mostly); the dual-store exception in v68+ and its constraints; how the compiler/assembler detects valid vs invalid packets.',
                answer: 'Slot assignment: Slot 0/1 are ALU/MUL (dual-issue ALU allowed), Slot 2 is the LSU (loads/stores), Slot 3 can be ALU/MUL/branch. Up to 4 instructions per packet but constrained by available functional units. Only 1 LSU per cycle exists in the hardware in v65/v66 — the L1 D-cache has limited ports (2 read + 1 write) and a single dispatch path. In v68+, dual-store is allowed if both stores target Slot 2 with non-overlapping addresses and no bank conflict (the dual-bank D-cache makes this feasible for stores to different banks). The assembler/compiler detects invalid packets at code-generation time by checking the slot-resource matrix: each instruction has a set of valid slots, and the assembler must produce a feasible assignment. If no assignment exists (e.g., 3 ALUs in one packet, two stores+load, etc.), the assembler emits an error rather than generating bad code.',
              },
              {
                question: 'Section 2 — Q5: Diagram and explain the Hexagon memory hierarchy latencies. Given the sequence: r0 = memw(r1) (Cycle N, L1 hit), r2 = memw(r3) (Cycle N+1), r4 = add(r0, r2) — when does the add start executing? Explain the forwarding path and identify all possible stalls.',
                answer: 'L1 hit latency is 2 cycles (E4 → M5). Loads forward results 1 cycle after the load (load-to-use forward path). Cycle N: first load issues, completes L1 hit in M5. Cycle N+1: second load issues. The result of load 1 is available via forwarding starting at cycle N+1. The result of load 2 is available starting at cycle N+2. The add depends on both r0 and r2. r0 is ready at N+1; r2 is ready at N+2. So the add can begin execution at cycle N+2 at the earliest, with no additional stalls if both loads were L1 hits. If either load was an L1 miss → L2 hit (~2 extra cycles for L2 lookup), the add waits accordingly. If L2 misses → 30-100 additional cycles. Bank conflicts in the dual-bank D-cache could also serialize the two loads, adding 1 cycle.',
              },
              {
                question: 'Section 2 — Q6: The D-cache is organized as a dual-bank structure. Explain (a) the bank interleaving scheme, (b) how a load and store can both execute simultaneously, (c) what constitutes a bank conflict, and (d) the victim buffer role in write-back.',
                answer: '(a) Dual-bank interleaving: Bank 0 holds even addresses (bit r1[4]=0), Bank 1 holds odd addresses (bit r1[4]=1). Each bank is 16 KB with 256 sets, 4-way. (b) A simultaneous load and store can execute if they target different banks — Bank 0 services the load while Bank 1 services the store in the same cycle. This is the throughput claim of 1 LD + 1 ST per cycle. (c) A bank conflict occurs when two memory operations in the same packet target the same bank — the LSU must serialize them, costing 1 stall cycle. The dispatcher detects this at issue time via address bit r[4]. (d) The victim buffer (4-entry) holds dirty lines evicted from L1-D before they propagate to L2. This decouples eviction from refill: a refill into a slot doesn\'t stall waiting for the dirty victim to write back to L2. The victim buffer is checked on cache lookups; a hit there returns data without going to L2.',
              },
              {
                question: 'Section 2 — Q7: Compare TCM and VTCM in terms of address space, access patterns, and latency characteristics. Why can\'t the scalar core use VTCM?',
                answer: 'TCM (Tightly Coupled Memory): Scalar-core only, single port, 32-bit accesses, 1-cycle fixed latency, address range 0xa0000000-0xa000FFFF (64 KB max). Used for inner loops and kernel code that need predictable timing. VTCM (Vector TCM): Vector/HMX-only, dual 128-bit ports (256-bit on v75), 1-cycle fixed latency, address range 0xb0000000-0xb003FFFF (256 KB on v75). Used for vector working sets, weight matrices, intermediate results. The scalar core cannot use VTCM because the VTCM port is sized for vector accesses (128-256 bit) and physically connected to the HVX/HMX datapaths — there is no scalar load/store path into VTCM. The hardware enforces this via memory-map decoding: scalar loads to the VTCM range trigger a fault.',
              },
              {
                question: 'Section 2 — Q8: L2 cache has an 8-way associative design with 512 KB (v68+). Calculate: number of sets, bits used for set index, line offset, and tag, miss penalty if request must go off-core (100 cycles), expected miss rate for audio DSP algorithms.',
                answer: 'Total lines = 512 KB / 64 B = 8192 lines. Sets = 8192 / 8 = 1024 sets. Set index bits = log2(1024) = 10 bits. Line offset = log2(64) = 6 bits. Tag = 32 - 10 - 6 = 16 bits (for 32-bit physical address, more for 36-48 bit). Miss penalty: 100 cycles off-core, plus the L1 miss penalty (~10-20 cycles) and L2 lookup (~2 cycles), so total observed latency ~110-120 cycles. Audio DSP miss rate: typical streaming DSP has highly regular access patterns (sequential reads of audio samples, FIR filter taps), so L2 miss rate is very low — typically <1% if the working set fits in L2 (512 KB easily holds 64K samples + filter coefficients). Miss rate goes up if working set exceeds L2 capacity (e.g., long FFTs > 128K points).',
              },
              {
                question: 'Section 3 — Q9: Explain the predicate execution model. Given: p0 = cmp.eq(r0, #5); if (p0) r1 = add(r2, r3); if (!p0) r1 = sub(r2, r3); — How many cycles does this execute? Can both branches be issued in the same packet? Explain.',
                answer: 'Predicates are 1-bit results stored in P0-P3. The compare instruction sets p0 in cycle N (1-cycle latency, available next cycle). Both predicated instructions can be placed in a single packet AFTER the predicate is computed: in packet N+1, both `if (p0) r1 = add(r2, r3)` and `if (!p0) r1 = sub(r2, r3)` can execute in parallel — one in slot 0, one in slot 1 — because they target the same destination register r1 but are mutually exclusive (only one will commit based on p0). The hardware suppresses the writeback of the inactive instruction, so r1 ends up with the correct value. Total cycles: 2 (cycle N for compare, cycle N+1 for both predicated adds). This is much faster than a branch (which would cost a misprediction penalty 50% of the time, ~3-5 cycles).',
              },
              {
                question: 'Section 3 — Q10: Hexagon branch prediction has ~95% accuracy for loops. Explain: the structure of the BTB, misprediction penalty and recovery sequence, how hardware loops achieve zero-overhead looping, and the role of prediction in I-cache prefetching.',
                answer: 'BTB (Branch Target Buffer): 512 entries, 2-way set associative, indexed by PC. Each entry stores [PC, Target, Prediction state]. Prediction state is typically a 2-bit saturating counter for backward branches (loops). Hit rate on loop branches: 95%+. Misprediction penalty: branches resolve in E4 (EXECUTE stage). When mispredicted, the pipeline must flush stages FETCH/DECODE/DISPATCH (3 stages = 3-5 cycles depending on packet density). Recovery: the correct target is loaded into PC, I-cache fetch restarts. Hardware loops (loop0/loop1 + endloop0/endloop1) achieve zero-overhead because the loop counter (LC0/LC1) is decremented in dedicated hardware, the back-edge target (LE0/LE1) is known statically, and the back-jump is predicted with 100% accuracy as long as LC > 0. Prediction\'s I-cache role: when a branch is predicted in FETCH, the predicted target is sent to I-cache prefetch immediately, so the fetch stream continues from the new PC with no bubble. If the prediction was wrong, the prefetched stream is squashed in the flush.',
              },
              {
                question: 'Section 3 — Q11: NOP cycles and stall hazards: Given r0 = mpyh(r1, r2) (3-cycle latency); r3 = add(r0, r4) (Read r0). Explain the dependency tracking hardware and minimum stall cycles.',
                answer: 'The multiplier has 3-cycle latency, meaning the result of mpyh is not available until 3 cycles after issue. The hazard detection hardware uses a register scoreboard: each architectural register has a "pending write" bit and a "ready cycle" counter. When mpyh issues at cycle N, scoreboard[r0] = {pending=1, ready_cycle=N+3}. When the next packet at cycle N+1 attempts to read r0 in the add instruction, the scoreboard reports pending — the dispatcher stalls the entire packet (since add needs r0). The dispatcher continues round-robin to other threads, but this thread is stalled. At cycle N+3, scoreboard[r0] becomes ready, and the add can finally issue. Total stall cycles: 2 (the add waits cycles N+1 and N+2, executes at N+3). In a multi-threaded workload, those 2 stall cycles are filled by other threads, so the effective penalty is 0 if 4 threads are running.',
              },
              {
                question: 'Section 3 — Q12: Analyze the instruction encoding for: r0 = add(r1, #1000). Explain how 16-bit immediates are encoded, alternative instruction sequences for 32-bit immediates, why large constants require multiple instructions.',
                answer: '16-bit immediates fit directly into the instruction encoding: the standard format leaves enough bit space (after opcode + 2 source regs + dest reg) for a 16-bit signed immediate. So `r0 = add(r1, #1000)` encodes as: opcode=ADD_imm, src=r1, dst=r0, imm=#1000 — single 32-bit instruction. For 32-bit immediates, there isn\'t enough space in a single 32-bit instruction word, so the assembler emits a sequence: usually a CONST instruction loads the upper bits (bits [31:10]) into a temp register, followed by an OR or ADD with the lower bits. Example: `r0 = ##0x12345678` becomes `r0 = #0x1234 << 16` followed by `r0 = or(r0, #0x5678)`. This costs 2 cycles (one packet) instead of 1. Why two instructions: instruction encoding budget — 32-bit instructions can only carry ~16 bits of immediate before running out of bits for the opcode and register specifiers.',
              },
              {
                question: 'Section 4 — Q13: Explain the round-robin scheduler from first principles. If Thread 0 experiences a 15-cycle L2 miss on Cycle 5, what is the packet issue sequence for Cycles 5-20?',
                answer: 'Round-robin scheduler: at each cycle, the dispatcher picks the next ready thread in the order T0→T1→T2→T3→T0... A thread is "ready" if its stall_cycles==0 and its current packet has no hazards. If T0 has an L2 miss at cycle 5, scoreboard marks T0 stalled for 15 cycles (until cycle 20). The schedule then becomes:\n\nCycle 5: T0 issues (the load that will miss) → T0 stalled until cycle 20\nCycle 6: T1 (next in order)\nCycle 7: T2\nCycle 8: T3\nCycle 9: T1 (skip T0 — still stalled)\nCycle 10: T2\nCycle 11: T3\nCycle 12: T1 (skip T0)\nCycle 13: T2\nCycle 14: T3\nCycle 15: T1\nCycle 16: T2\nCycle 17: T3\nCycle 18: T1\nCycle 19: T2\nCycle 20: T0 (miss resolved, returns to round-robin) → T3 next\n\nT0 is skipped for 15 cycles. Other threads experience no slowdown — the cache miss is fully hidden by multithreading.',
              },
              {
                question: 'Section 4 — Q14: Register file banks: The 8 KB register file is split into 4 banks (one per thread). Explain physical address mapping for R5 in each thread, why no inter-thread forwarding is implemented, and how synchronization primitives rely on shared memory.',
                answer: 'The 8 KB physical register file is divided into 4 banks of 2 KB each. Within each bank, R0 starts at offset 0 and each register is 4 bytes (32-bit). So R5 in Thread T is at physical address (T × 2048) + (5 × 4) = T*2048 + 20. T0.R5 = address 20, T1.R5 = 2068, T2.R5 = 4116, T3.R5 = 6164. The dispatcher resolves the bank by combining thread_id with the architectural register index. No inter-thread forwarding is implemented because: (1) it would add wires between banks, increasing area and timing; (2) it would require coherence logic that defeats the simplicity of the round-robin scheduler; (3) the design philosophy is that threads are independent — sharing happens only through memory. Synchronization primitives: locks use atomic memory operations (e.g., load-locked/store-conditional patterns) on shared memory locations. A thread acquires a lock by atomically writing a value, releases by writing back. Memory barriers (MEMORY_BARRIER) ensure ordering. This is slower than register-level handoff but simpler.',
              },
              {
                question: 'Section 5 — Q15: Hexagon connects to the SoC via a NoC. Explain how L2 cache misses are translated to NoC requests, priority/fairness arbitration in the NoC, SMMU translation flow VA→IPA→PA, and bandwidth sharing with CPU/GPU.',
                answer: 'L2 miss → NoC: When L2 misses, the request is queued in a write-back buffer and submitted via Hexagon\'s NoC port. The request includes [physical address, transaction type (read/write), size, requester ID = Hexagon stream ID 0x10, QoS priority]. NoC arbitration: priority-aware round-robin among requesters (CPU, GPU, Hexagon, ISP, Video Encoder, etc.). QoS tags allow critical traffic (e.g., display refresh) to preempt lower-priority traffic. Hexagon typically has medium priority. SMMU translation: VA (48-bit virtual) → walk Stage 1 page tables → IPA (intermediate physical, also 48-bit) → walk Stage 2 page tables (if VM enabled) → PA (physical 36-44 bit). The SMMU has a 4K-entry TLB to cache translations, avoiding page-table walks for repeated accesses. Bandwidth sharing: LPDDR5 provides ~32 GB/s total. With multiple masters competing, Hexagon typically gets ~5-10 GB/s in mixed workloads. The NoC arbiter can be configured for fair shares or priority.',
              },
              {
                question: 'Section 5 — Q16: Power domains VDDCX and VDDMX: Explain the difference and power gating/standby scenarios for each.',
                answer: 'VDDCX (Core Logic supply, 0.6-1.0V): Powers the Hexagon scalar core, HVX vector unit, L1 caches, and associated control logic. This is the dominant dynamic power consumer when the core is active. VDDMX (Memory/Aux supply, 0.8-1.1V): Powers L2 cache, TCM/VTCM, DMA controllers, SMMU. Higher voltage because memory cells need it for retention. Power gating scenarios: (a) VDDCX off (full power gate) — core inaccessible, register state lost (must be saved externally), 95%+ power savings, wake latency ~100 microseconds. Use when device is idle for long periods. (b) VDDCX standby (retention) — minimal leakage, register state retained in retention flip-flops, 80-90% power reduction, wake latency ~1 microsecond. Use for short idle periods. (c) Clock gating only — VDDCX still powered but clock stopped, 50-70% power reduction (only leakage remains), wake latency 10s of nanoseconds. Use for sub-millisecond idle gaps. VDDMX is typically left on whenever VDDCX is on, to keep L2/TCM contents valid. VDDMX can be independently gated when L2 contents can be discarded (e.g., between application contexts).',
              },
              {
                question: 'Section 5 — Q17: DMA engines provide off-core data movement. Explain configuration registers (source, destination, length, stride), how 2D transfers work with strides, peak throughput, and typical use (loading weights, activations).',
                answer: 'DMA configuration registers (memory-mapped at the DMA engine address): CMD (start/stop), STATUS (busy/done/error), ADDR_SRC, ADDR_DST, LENGTH (bytes), SRC_STRIDE, DST_STRIDE, FLAGS (priority, fixed-source/dest, completion interrupt enable). 2D transfers use SRC_STRIDE/DST_STRIDE to walk through 2D arrays: after each row of LENGTH bytes, the address is advanced by SRC_STRIDE/DST_STRIDE rather than LENGTH, allowing transposes, sub-image extracts, and matrix tiling. Peak throughput: ~200 GB/s into L2 (when the request hits the L2 fill bandwidth), but ~32 GB/s when going to DDR (limited by LPDDR5 bandwidth shared with other masters). Setup latency: ~100 cycles. Typical use in NN inference: (1) load next layer weights from DDR into VTCM in parallel with current layer compute (double-buffering), (2) load activation tiles from DDR into L2/VTCM, (3) write final outputs back to DDR. The DMA runs independently of the Hexagon pipeline, so compute and data movement overlap, hiding DDR latency.',
              },
              {
                question: 'Section 6 — Q18: Trace the evolution from v65 → v68 → v75. For each, identify new execution units introduced, ISA extensions (with code examples), peak throughput metrics.',
                answer: 'v65 baseline: 4-way SMT, 6-stage scalar pipeline, 128-bit HVX (4 FP32 ops/cycle), no tensor accelerator, 256 KB L2, no TCM/VTCM. Peak: ~0.5 GFLOPS scalar, ~10 GFLOPS HVX FP32. v68 (2020, major step): new HTA tensor accelerator (~7 TMAC/s INT8), HVX widened to 256-bit (8 FP32 ops/cycle = 2x throughput), VTCM 128 KB introduced, dual-store support. New ISA: 256-bit HVX intrinsics like Q6_V_vadd_VV_256, HTA programming via specialized instructions, VTCM pragmas. Peak: ~7 TMAC/s INT8, ~14 GFLOPS HVX FP32. v75 (2024, latest): HMX replaces HTA (~10 TMAC/s INT8, 14.4 TFLOPS FP16, 3.6 TFLOPS TF32), VTCM expanded to 256 KB, new HVX v75 instructions (vshuffeh, vtranspose, vpopcnt, vlz), TF32/INT4/INT2 data type support, improved branch prediction. Peak: ~10 TMAC/s INT8, ~14 TFLOPS FP16, ~16 GFLOPS HVX FP32. Generations between (v66, v69, v73) are minor speed/ISA refinements.',
              },
              {
                question: 'Section 6 — Q19: Why was HVX width expansion (128→256 bits) a turning point for Hexagon as an ML accelerator?',
                answer: 'The HVX width expansion in v68 was a turning point because: (1) Doubled raw throughput per instruction — a single SIMD ADD now processes 8 FP32 ops vs 4, so the same instruction count delivers 2x compute. (2) Improved arithmetic intensity — wider SIMD means more compute per memory load, helping memory-bound kernels saturate the L2 bandwidth. (3) Better fit for NN tensor shapes — neural network operations often have inner dimensions of 64, 128, 256 channels; 256-bit vectors process these more efficiently with fewer overhead instructions (no leftover lanes). (4) Combined with VTCM (also new in v68), the wider HVX could keep working sets local with single-cycle access, dramatically improving GEMM and conv kernel performance. (5) Enabled competitive INT8 inference: 256-bit HVX can process 32 × INT8 multiplies per cycle, putting Hexagon in the same league as ARM SVE and x86 AVX-512 VNNI for mobile NN inference. Before v68, Hexagon was primarily a DSP for audio/imaging; after v68, it became a full NN accelerator, especially when combined with HTA.',
              },
              {
                question: 'Section 6 — Q20: The VTCM expansion to 256 KB in v75: Calculate the weight matrix sizes that now fit. Compare inference latency for a 128×128 INT8 MatMul with weights in VTCM vs L2 cache.',
                answer: 'VTCM at 256 KB can hold: INT8 weight matrices up to ~256x256x4 (= 256 KB) or larger if compressed; specifically, a 256×256 INT8 matrix is 64 KB (fits with room to spare for activations and intermediate results); a 128×128 INT8 matrix is 16 KB (trivially fits); a 512×128 INT8 matrix is 64 KB (fits). For comparison, v68 with 128 KB VTCM could not hold a full 256×256 matrix plus activations comfortably. Latency comparison for 128×128 INT8 MatMul: With weights in VTCM (1-cycle access): the HMX matmul completes in ~50-100 cycles (latency-bound, not memory-bound). With weights in L2 (2-cycle hit, but bandwidth-shared): the matmul still takes ~50-100 cycles for compute, but ramping the data into HMX adds ~32 KB / 16 GB/s ≈ 2 microseconds = 3600 cycles at 1.8 GHz overhead the first time, which dominates. So VTCM → ~50-100 cycles (sub-microsecond), L2-resident → ~3700 cycles (microseconds). VTCM is 30-70x faster for hot weights. Even with re-use across many invocations, L2 miss penalties (going off-core to DDR) make VTCM essential for inner-loop matrices.',
              },
            ]} />
          </div>

          {/* ============ References ============ */}
          <div id="references">
            <ReferenceList references={[
              {
                authors: 'Qualcomm Technologies',
                year: 2024,
                title: 'Hexagon V73 Architecture Specification (HVX v73, ISA Reference)',
                venue: 'Official Qualcomm Documentation',
              },
              {
                authors: 'Qualcomm Technologies',
                year: 2023,
                title: 'Snapdragon 8 Gen 2 Technical Reference Manual',
                venue: 'Official Qualcomm Documentation',
              },
              {
                authors: 'Qualcomm Technologies',
                year: 2024,
                title: 'Hexagon HVX Programmer\'s Reference Manual',
                venue: 'Official Qualcomm Documentation',
              },
              {
                authors: 'Qualcomm Technologies',
                year: 2024,
                title: 'Hexagon Tensor Accelerator (HTA) Programming Guide',
                venue: 'Official Qualcomm Documentation',
              },
              {
                authors: 'Qualcomm Technologies',
                year: 2023,
                title: 'Qualcomm Hexagon Cache & Memory Hierarchy Design Brief',
                venue: 'Memory Subsystem Documentation',
              },
              {
                authors: 'ARM Holdings',
                year: 2017,
                title: 'ARM SMMU v3 Architecture Specification',
                venue: 'IOMMU Reference',
              },
              {
                authors: 'Qualcomm Technologies',
                year: 2024,
                title: 'Qualcomm Hexagon Performance Optimization Guide',
                venue: 'Performance Tuning Documentation',
              },
              {
                authors: 'Qualcomm Technologies',
                year: 2024,
                title: 'HVX Code Optimization Handbook (Vector Scheduling)',
                venue: 'Performance Tuning Documentation',
              },
              {
                authors: 'Fisher, J. A., Faraboschi, P., & Young, C.',
                year: 2005,
                title: 'Embedded Computing: A VLIW Approach to Architecture, Compilers and Tools',
                venue: 'Morgan Kaufmann',
              },
              {
                authors: 'Ungerer, T., Robič, B., & Šilc, J.',
                year: 2003,
                title: 'A Survey of Processors with Explicit Multithreading',
                venue: 'ACM Computing Surveys, vol. 35, no. 1',
              },
            ]} />
          </div>

          {/* ============ Navigation ============ */}
          <div className="mt-12 pt-6 border-t border-border-primary flex items-center justify-between">
            <div />
            <Link
              href="/qualcomm/module-02"
              className="flex items-center gap-2 text-sm font-medium text-accent-blue hover:gap-3 transition-all"
            >
              Next: HVX Programming Model <ArrowRight className="w-4 h-4" />
            </Link>
          </div>
        </article>

        <TableOfContents items={tocItems} />
      </div>
    </>
  )
}
