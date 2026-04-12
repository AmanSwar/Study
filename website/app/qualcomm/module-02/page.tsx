import Link from 'next/link'
import { ArrowLeft, ArrowRight } from 'lucide-react'
import { Breadcrumbs } from '@/components/layout/Breadcrumbs'
import { ModuleHeader } from '@/components/content/ModuleHeader'
import { TheorySection, SubSection } from '@/components/content/TheorySection'
import { CodeBlock } from '@/components/content/CodeBlock'
import { CalloutBox } from '@/components/content/CalloutBox'
import { SelfAssessment } from '@/components/content/SelfAssessment'
import { KeyTakeaways } from '@/components/content/KeyTakeaways'
import { ReferenceList } from '@/components/content/ReferenceList'
import { TableOfContents } from '@/components/content/TableOfContents'

export const metadata = { title: 'Module 2: HVX (Hexagon Vector eXtensions) Programming Model' }

const tocItems = [
  { id: 'introduction', title: 'Introduction', level: 2 },
  { id: 'register-file', title: 'HVX Register File Architecture', level: 2 },
  { id: 'data-types-lanes', title: 'HVX Data Types and Lane Semantics', level: 2 },
  { id: 'intrinsic-families', title: 'Core HVX Intrinsic Families', level: 2 },
  { id: 'convolution-kernels', title: 'Writing Efficient Convolution Kernels', level: 2 },
  { id: 'depthwise-convolution', title: 'Vectorizing Depthwise Convolution', level: 2 },
  { id: 'pointwise-gemm', title: 'Pointwise Convolution and GEMM for INT8', level: 2 },
  { id: 'predicates-masks', title: 'Predicate Registers and Masked Operations', level: 2 },
  { id: 'pipelining', title: 'Loop Alignment and Software Pipelining', level: 2 },
  { id: 'intrinsics-vs-asm', title: 'HVX Intrinsics vs Inline Assembly', level: 2 },
  { id: 'self-assessment', title: 'Self-Assessment Questions', level: 2 },
  { id: 'references', title: 'References', level: 2 },
  { id: 'appendix', title: 'Appendix: Quick Reference', level: 2 },
]

export default function Module02Page() {
  return (
    <>
      <Breadcrumbs
        items={[
          { label: 'Qualcomm', href: '/qualcomm' },
          { label: 'Module 2' },
        ]}
      />

      <div className="flex gap-0">
        <article className="flex-1 min-w-0">
          <ModuleHeader
            number={2}
            title="HVX (Hexagon Vector eXtensions) Programming Model"
            track="qualcomm"
            part={0}
            readingTime="4-6 hours"
            prerequisites={['Module 1: Hexagon ISA Fundamentals', 'SIMD concepts', 'Integer arithmetic']}
            description="PhD-level deep technical coverage of HVX programming: register architecture, intrinsic families, data types, convolution and GEMM kernels, predicates, software pipelining, and inline assembly. Primary mechanism for high-throughput mobile ML inference on Hexagon DSP cores."
          />

          {/* ============ Introduction ============ */}
          <TheorySection id="introduction" title="Introduction">
            <p>
              The Hexagon Vector eXtensions (HVX) represent Qualcomm&apos;s primary mechanism for achieving high throughput
              on the Hexagon DSP core. Unlike the scalar instruction set covered in Module 1, HVX operates on wide
              vectors—either 128 bytes (HVX-128 mode, standard for Snapdragon) or 256 bytes (HVX-256 mode, when
              enabled)—in a single instruction.
            </p>

            <p>HVX&apos;s architecture was specifically designed for mobile ML inference workloads:</p>
            <ul className="list-disc pl-6 my-4 space-y-1">
              <li><strong>Lane-based parallelism</strong>: Vectors are conceptually divided into &quot;lanes,&quot; each operating on a data element</li>
              <li><strong>Data-parallel execution</strong>: All lanes execute the same operation simultaneously</li>
              <li><strong>Flexible data widths</strong>: Operations adapt to 8-bit, 16-bit, 32-bit, and 64-bit element sizes</li>
              <li><strong>VTCM integration</strong>: Direct access to Vector TCM (Tightly Coupled Memory) enables high-bandwidth data movement</li>
            </ul>

            <p>
              This module provides deep technical coverage of HVX programming, from low-level register architecture to
              high-level optimization patterns for neural network inference.
            </p>

            <CalloutBox type="info" title="Scope Note">
              <p>
                This module focuses on HVX-128 mode (128 bytes = 1024 bits per register) as the primary target,
                with notes on HVX-256 scaling where relevant.
              </p>
            </CalloutBox>
          </TheorySection>

          {/* ============ HVX Register File Architecture ============ */}
          <TheorySection id="register-file" title="HVX Register File Architecture">
            <SubSection id="register-file-overview" title="2.1 Overview of the HVX Register File">
              <p>
                The HVX vector register file is fundamentally different from scalar registers. Rather than holding a
                single 32-bit or 64-bit value, each vector register holds a <strong>full vector</strong> of data elements.
              </p>

              <h4 className="text-lg font-semibold text-text-primary mt-6 mb-3">Register Configuration in HVX-128 Mode</h4>
              <CodeBlock language="text" title="V0 Register Layout (128 bytes)">
{`┌─────────────────────────────────────────────────────────────┐
│                      V0 (128 bytes)                         │
├─────────────────────────────────────────────────────────────┤
│ Byte   │ Byte   │ Byte   │ ...      │ Byte  │ Byte  │ Byte  │
│  [0]   │  [1]   │  [2]   │          │ [125] │ [126] │ [127] │
│        ← 16 × 8-bit elements OR →                            │
│        ← 8 × 16-bit elements OR →                            │
│        ← 4 × 32-bit elements OR →                            │
│        ← 2 × 64-bit elements →                               │
└─────────────────────────────────────────────────────────────┘`}
              </CodeBlock>

              <p><strong>Key architectural facts:</strong></p>
              <ul className="list-disc pl-6 my-4 space-y-1">
                <li><strong>32 vector registers</strong>: V0–V31 in HVX-128 mode</li>
                <li><strong>Register width</strong>: 128 bytes (1024 bits) per register</li>
                <li><strong>Addressable as pairs</strong>: (V0:1), (V2:3), ..., (V30:31) for 256-byte operations</li>
                <li><strong>No partial register access</strong>: You cannot directly access a sub-register (e.g., the lower 64 bytes of V0); instructions consume/produce full registers</li>
              </ul>

              <CalloutBox type="tip" title="Expert Insight">
                <p>
                  The HVX register file is fundamentally <strong>wide but not deep</strong>—32 registers of 128 bytes
                  each total only 4 KB. This is why careful register allocation and judicious use of VTCM are critical
                  for large workloads.
                </p>
              </CalloutBox>
            </SubSection>

            <SubSection id="register-pairs" title="2.2 Register Pairs and Accumulation">
              <p>When working with wider accumulation widths, HVX uses <strong>register pairs</strong>:</p>
              <CodeBlock language="text" title="Register Pair Layout">
{`┌──────────────────────┐
│   V1:0 (256 bytes)   │
├──────────────────────┤
│   High half: V1      │ (128 bytes)
│   Low half: V0       │ (128 bytes)
├──────────────────────┤`}
              </CodeBlock>

              <p>
                Register pairs are always <strong>even:odd</strong> (V0:1, V2:3, V4:5, ..., V30:31). This restriction is
                a hardware constraint to maintain symmetry in the vector pipeline.
              </p>

              <h4 className="text-lg font-semibold text-text-primary mt-6 mb-3">When to Use Register Pairs</h4>
              <ol className="list-decimal pl-6 my-4 space-y-1">
                <li><strong>32-bit accumulation from 16-bit multiplies</strong>: <code className="px-1.5 py-0.5 rounded bg-bg-code text-accent-cyan text-[0.9em] font-mono">vrmpyacc(Vdd32, Vss32, Rtt)</code> requires a register pair to hold the accumulated 32-bit results</li>
                <li><strong>64-bit intermediate values</strong>: Some reduction operations require wider intermediate results</li>
                <li><strong>Double-width output from multiply</strong>: 16×16→32-bit multiply produces a register pair</li>
              </ol>
            </SubSection>

            <SubSection id="predicate-q-registers" title="2.3 Predicate Register File (Q Registers)">
              <p>
                In addition to vector registers, HVX provides a small <strong>predicate register file</strong> for
                conditional operations:
              </p>
              <CodeBlock language="text" title="Q0 Predicate Register">
{`┌─────────────────────────────────────┐
│  Q0 (Predicate Register, 128 bits)  │
├─────────────────────────────────────┤
│ Bit 0 │ Bit 1 │ ... │ Bit 127       │
│ (Lane 0) (Lane 1) ... (Lane 127)    │
│  for 8-bit mode ───────────────────│
│ Bit 0 │ Bit 1 │ ... │ Bit 63        │
│ (Lane 0) (Lane 1) ... (Lane 63)     │
│  for 16-bit mode ──────────────────│
└─────────────────────────────────────┘`}
              </CodeBlock>

              <p><strong>Q Register Set:</strong></p>
              <ul className="list-disc pl-6 my-4 space-y-1">
                <li><strong>4 predicate registers</strong>: Q0, Q1, Q2, Q3</li>
                <li><strong>Width</strong>: 128 bits, subdivided by lane width (16 lanes for 8-bit, 8 lanes for 16-bit, etc.)</li>
                <li><strong>Operations</strong>: Each bit corresponds to a lane-level condition</li>
              </ul>

              <p><strong>Typical usage pattern:</strong></p>
              <CodeBlock language="c" title="Predicate Usage Pattern">
{`// Generate a predicate from a comparison
Q0 = vmpyh(V0, V1)  // Compare operation (result: predicate)
// Use predicate in masked operation
V2 = Q0 ? V3 : V4   // Conditional select: if Q0[i], V2[i] = V3[i], else V4[i]`}
              </CodeBlock>

              <CalloutBox type="tip" title="Expert Insight">
                <p>
                  Predicates are <strong>not</strong> regular conditional masks like AVX-512 in x86. They are
                  specifically tied to vector-level comparisons and logical operations. This design choice reflects
                  the Hexagon&apos;s focus on data-parallel neural network workloads where branch divergence is rare.
                </p>
              </CalloutBox>
            </SubSection>

            <SubSection id="register-allocation-strategy" title="2.4 Register Allocation Strategy">
              <p>The 32-register limit is tight for complex kernels. Typical strategies:</p>

              <p><strong>Strategy 1: Blocking with VTCM</strong></p>
              <CodeBlock language="text" title="Register Allocation Strategy 1">
{`Registers used:
  - V0–V3:    Input tile (128 bytes × 4 = 512 bytes useful data)
  - V4–V7:    Weights (or filter coefficients)
  - V8–V15:   Accumulation registers (8 for partial sums)
  - V16–V31:  Scratch/temporal storage

Reserve V30:31 for callee-save requirements (ABI)`}
              </CodeBlock>

              <p><strong>Strategy 2: Temporal Reuse</strong></p>
              <CodeBlock language="text" title="Register Allocation Strategy 2">
{`In a loop over batches:
  Load input chunk    → V0:V3
  Load weights        → V4:V7 (reused across loop iterations)
  Accumulate results  → V8:V15 (updated in-place)
  Store output        → (from V8:V15)
  (Shuffle/permute ops use V16–V29 as temporaries)`}
              </CodeBlock>
            </SubSection>
          </TheorySection>

          {/* ============ Data Types and Lane Semantics ============ */}
          <TheorySection id="data-types-lanes" title="HVX Data Types and Lane Semantics">
            <SubSection id="lane-width-mapping" title="2.5 Understanding Lanes and Data Width Interpretation">
              <p>
                The critical insight in HVX programming is that <strong>the same register holds different numbers of
                elements depending on the operation data width</strong>:
              </p>

              <h4 className="text-lg font-semibold text-text-primary mt-6 mb-3">Lane Width Mapping</h4>
              <CodeBlock language="text" title="Lane Width Interpretations">
{`Register V0 (128 bytes = 1024 bits):

┌──────────────────────────────────────────┐
│  8-bit mode: 128 lanes (unsigned or signed)
├──────────────────────────────────────────┤
│  [0] [1] [2] ... [127]
│  ub0 ub1 ub2     ub127  (unsigned byte)
│  -or-
│  b0  b1  b2  ... b127   (signed byte)
├──────────────────────────────────────────┤
│  16-bit mode: 64 lanes
├──────────────────────────────────────────┤
│  [0]      [1]      ... [63]
│  uh0      uh1          uh63   (unsigned half)
│  -or-
│  h0       h1  ...  h63        (signed half)
├──────────────────────────────────────────┤
│  32-bit mode: 32 lanes
├──────────────────────────────────────────┤
│  [0]          [1]          ... [31]
│  uw0          uw1              uw31   (unsigned word)
│  -or-
│  w0           w1   ...         w31    (signed word)
├──────────────────────────────────────────┤
│  64-bit mode: 16 lanes
├──────────────────────────────────────────┤
│  [0]                  [1]                ... [15]
│  ud0                  ud1                    ud15   (unsigned dword)
│  -or-
│  d0                   d1    ...              d15    (signed dword)
└──────────────────────────────────────────┘`}
              </CodeBlock>

              <p>
                This <strong>dynamic lane interpretation</strong> is why HVX is so efficient for quantized ML: the same
                vector hardware executes 8-bit, 16-bit, or 32-bit operations without format conversion.
              </p>
            </SubSection>

            <SubSection id="signed-unsigned-ops" title="2.6 Signed vs. Unsigned Operations">
              <p>
                HVX distinguishes between signed and unsigned operations through <strong>intrinsic naming conventions</strong>:
              </p>
              <CodeBlock language="c" title="Signed vs Unsigned Multiply-Accumulate">
{`// 8-bit multiply-accumulate: signed × signed → 16-bit
// Intrinsic: Q6_Vh_vmpyacc_VhVb (signed)
Vdd16_result = Q6_Vh_vmpyacc_VhVb(Vdd16_accum, Vs_coeff, Vu_input);

// vs.

// 8-bit multiply-accumulate: unsigned × unsigned → 16-bit
// Intrinsic: Q6_Vuh_vmpyacc_VuhVub (unsigned)
Vdd16_result = Q6_Vuh_vmpyacc_VuhVub(Vdd16_accum, Vus_coeff, Vuu_input);`}
              </CodeBlock>

              <p><strong>Naming convention:</strong></p>
              <ul className="list-disc pl-6 my-4 space-y-1">
                <li>Lower-case &apos;h&apos; → signed half (16-bit result from 8-bit input)</li>
                <li>Uppercase &apos;UH&apos; → unsigned half</li>
              </ul>
            </SubSection>

            <SubSection id="widening-narrowing" title="2.7 Widening and Narrowing Operations">
              <p>
                Real-world quantized inference requires <strong>widening</strong> (expanding to prevent overflow) and{' '}
                <strong>narrowing</strong> (quantizing back down).
              </p>

              <h4 className="text-lg font-semibold text-text-primary mt-6 mb-3">Widening: 8-bit → 16-bit</h4>
              <CodeBlock language="c" title="Widening 8-bit to 16-bit">
{`// Load 8-bit signed bytes into a vector register
V0 = Q6_V_vmem_R(R0);  // 128 × int8

// Widen to 16-bit using vunpacke/vunpacko
// vunpacke: Extract even-indexed lanes
V1 = Q6_Vh_vunpacke_Vb(V0);  // V1[0] = sign_extend(V0[0]), V1[1] = sign_extend(V0[2]), ...

// vunpacko: Extract odd-indexed lanes
V2 = Q6_Vh_vunpacko_Vb(V0);  // V2[0] = sign_extend(V0[1]), V2[1] = sign_extend(V0[3]), ...`}
              </CodeBlock>

              <p><strong>Data movement diagram:</strong></p>
              <CodeBlock language="text" title="Widening Data Movement">
{`V0 (128 bytes, 128 × 8-bit):
[b0][b1][b2][b3]...[b127]

After vunpacke (even lanes):
V1 (64 × 16-bit):
[b0 ext][b2 ext][b4 ext]...[b126 ext]  (sign-extended to 16-bit)

After vunpacko (odd lanes):
V2 (64 × 16-bit):
[b1 ext][b3 ext][b5 ext]...[b127 ext]  (sign-extended to 16-bit)`}
              </CodeBlock>

              <h4 className="text-lg font-semibold text-text-primary mt-6 mb-3">Narrowing: 32-bit → 8-bit (with Saturation)</h4>
              <CodeBlock language="c" title="Narrowing 32-bit to 8-bit with Saturation">
{`// After accumulation, we have 32-bit results in register pair Vdd32
// Narrow and saturate to 8-bit using vpacke/vpacko + clip

// Extract even lanes (32-bit → 16-bit), then saturate
V0_even = Q6_Vh_vpacke_Vw(Vdd32);  // Pack even lanes of 32-bit to 16-bit

// Extract odd lanes (32-bit → 16-bit)
V0_odd = Q6_Vh_vpacko_Vw(Vdd32);   // Pack odd lanes of 32-bit to 16-bit

// Combine and saturate to 8-bit range
V0_result = Q6_Vb_vpacke_Vh(V0_even);  // Pack even lanes of 16-bit to 8-bit with saturation`}
              </CodeBlock>

              <CalloutBox type="tip" title="Expert Insight">
                <p>
                  Quantization in HVX requires careful orchestration of widening and narrowing operations. The key is
                  to <strong>widen early</strong> (to prevent overflow during accumulation) and <strong>narrow late</strong>{' '}
                  (to minimize intermediate register pressure).
                </p>
              </CalloutBox>
            </SubSection>

            <SubSection id="quantization-reasoning" title="2.8 Data Type Quantization Reasoning">
              <p>For INT8 quantized neural networks:</p>
              <CodeBlock language="text" title="Quantization Formula">
{`Formula: quantized_value = round(fp32_value / scale) + zero_point

Inverse: fp32_value = (quantized_value - zero_point) × scale`}
              </CodeBlock>

              <p>When implementing in HVX:</p>
              <ol className="list-decimal pl-6 my-4 space-y-1">
                <li><strong>Input multiplication</strong>: <code className="px-1.5 py-0.5 rounded bg-bg-code text-accent-cyan text-[0.9em] font-mono">int8_input × int8_weight</code> → requires 16-bit intermediate (signed)</li>
                <li><strong>Accumulation</strong>: Sum of products → requires 32-bit intermediate</li>
                <li><strong>Requantization</strong>: <code className="px-1.5 py-0.5 rounded bg-bg-code text-accent-cyan text-[0.9em] font-mono">(accum &gt;&gt; shift) + zero_point</code> → 32-bit → 8-bit (with saturation)</li>
              </ol>
            </SubSection>
          </TheorySection>

          {/* ============ Core HVX Intrinsic Families ============ */}
          <TheorySection id="intrinsic-families" title="Core HVX Intrinsic Families">
            <SubSection id="load-store-ops" title="2.9 Load and Store Operations">
              <h4 className="text-lg font-semibold text-text-primary mt-6 mb-3">2.9.1 Aligned Vector Load: Q6_V_vmem_R</h4>
              <p><strong>Signature:</strong></p>
              <CodeBlock language="c" title="Q6_V_vmem_R Signature">
{`HVX_Vector Q6_V_vmem_R(const void *Rs)`}
              </CodeBlock>
              <p><strong>Semantics</strong>: Load a full vector (128 bytes in HVX-128 mode) from a 128-byte-aligned address.</p>
              <p><strong>Constraints:</strong></p>
              <ul className="list-disc pl-6 my-4 space-y-1">
                <li>Address must be 128-byte aligned (bits [6:0] = 0)</li>
                <li>If alignment is not met, behavior is undefined</li>
              </ul>
              <p><strong>Example:</strong></p>
              <CodeBlock language="c" title="Aligned Load Example">
{`// Load 128 bytes of int8 data from memory
int8_t input_buffer[VECTOR_SIZE];  // Size: 128
// Ensure 128-byte alignment (use compiler attributes or manual allocation)
__attribute__((aligned(128))) int8_t aligned_input[128];

HVX_Vector V0 = Q6_V_vmem_R((const void *)aligned_input);`}
              </CodeBlock>
              <p><strong>Performance:</strong></p>
              <ul className="list-disc pl-6 my-4 space-y-1">
                <li><strong>Latency</strong>: 1 cycle (assuming cache hit)</li>
                <li><strong>Throughput</strong>: Can issue multiple loads per cycle (dual-issue on modern cores)</li>
                <li><strong>Bandwidth</strong>: 128 bytes per cycle (HVX-128), 256 bytes per cycle (HVX-256)</li>
              </ul>

              <h4 className="text-lg font-semibold text-text-primary mt-6 mb-3">2.9.2 Unaligned Vector Load: Q6_V_vmemu_R</h4>
              <p><strong>Signature:</strong></p>
              <CodeBlock language="c" title="Q6_V_vmemu_R Signature">
{`HVX_Vector Q6_V_vmemu_R(const void *Rs)`}
              </CodeBlock>
              <p><strong>Semantics</strong>: Load a full vector from an <strong>arbitrary</strong> address (not necessarily aligned).</p>
              <p><strong>Mechanism</strong>: The hardware performs 2-3 sub-vector loads and combines them:</p>
              <CodeBlock language="text" title="Unaligned Load Mechanism">
{`If address has lower 7 bits = offset (not aligned):
  Load from (address & ~127)       → temp_low
  Load from (address + 128)        → temp_high
  Shift/align and combine          → result_vector`}
              </CodeBlock>
              <p><strong>Performance:</strong></p>
              <ul className="list-disc pl-6 my-4 space-y-1">
                <li><strong>Latency</strong>: 2–3 cycles (due to alignment resolution)</li>
                <li><strong>Throughput</strong>: Half that of aligned loads</li>
                <li><strong>Penalty</strong>: ~3× cost vs. aligned load</li>
              </ul>

              <CodeBlock language="c" title="Unaligned Load Usage">
{`// Load from potentially unaligned address
const int8_t *unaligned_ptr = input + some_offset;
HVX_Vector V_data = Q6_V_vmemu_R((const void *)unaligned_ptr);`}
              </CodeBlock>

              <CalloutBox type="tip" title="Expert Insight">
                <p>
                  For input buffers where alignment is difficult to guarantee, vmemu is necessary, but the performance
                  cost is significant. In tight inner loops, align input data when possible to use vmem.
                </p>
              </CalloutBox>

              <h4 className="text-lg font-semibold text-text-primary mt-6 mb-3">2.9.3 Vector Gather: Q6_V_vgather_AsR</h4>
              <CodeBlock language="c" title="Q6_V_vgather_AsR Signature">
{`HVX_Vector Q6_V_vgather_AsR(int As, const void *Rs)`}
              </CodeBlock>
              <p>Where As is a 32-bit immediate offset (scaled).</p>
              <p><strong>Semantics</strong>: Load scattered elements from an array indexed by offsets.</p>
              <p><strong>Example: Gather every 4th element:</strong></p>
              <CodeBlock language="c" title="Vector Gather Example">
{`// Input: array of int32
int32_t sparse_array[256];

// Gather with stride 4 (every 4th element)
// Offset immediate: 4 (scale = 4 bytes per element)
HVX_Vector V0 = Q6_V_vgather_AsR(4, (const void *)sparse_array);
// V0[0] = sparse_array[0]
// V0[1] = sparse_array[4]  (stride of 4)
// V0[2] = sparse_array[8]
// ...`}
              </CodeBlock>
              <p><strong>Constraints:</strong></p>
              <ul className="list-disc pl-6 my-4 space-y-1">
                <li>Offset must be a compile-time constant</li>
                <li>Can only use fixed strides (1, 2, 4, 8 bytes)</li>
              </ul>
              <p><strong>Performance</strong>: ~5 cycles per gather (much slower than sequential loads)</p>

              <h4 className="text-lg font-semibold text-text-primary mt-6 mb-3">2.9.4 Vector Scatter: Q6_V_vscatter_AsRV</h4>
              <CodeBlock language="c" title="Q6_V_vscatter_AsRV Signature">
{`void Q6_V_vscatter_AsRV(int As, void *Rt, HVX_Vector Vs)`}
              </CodeBlock>
              <p><strong>Semantics</strong>: Write scattered elements to an array with a fixed stride.</p>
              <CodeBlock language="c" title="Vector Scatter Example">
{`int32_t output_array[256];
HVX_Vector V_results = ...;  // Results from computation

// Scatter with stride 4
Q6_V_vscatter_AsRV(4, (void *)output_array, V_results);
// output_array[0] = V_results[0]
// output_array[4] = V_results[1]
// output_array[8] = V_results[2]
// ...`}
              </CodeBlock>
              <p><strong>Performance</strong>: ~5 cycles per scatter</p>

              <CalloutBox type="info" title="Design Note">
                <p>
                  Gather and scatter are expensive. In well-optimized neural network kernels, avoid their use in hot
                  loops. Instead, <strong>rearrange data in VTCM</strong> to enable sequential access patterns.
                </p>
              </CalloutBox>
            </SubSection>

            <SubSection id="arithmetic-ops" title="2.10 Arithmetic Operations">
              <h4 className="text-lg font-semibold text-text-primary mt-6 mb-3">2.10.1 Vector Multiply: Q6_V_vmpy_VhVb</h4>
              <CodeBlock language="c" title="Q6_V_vmpy_VhVb Signature">
{`HVX_Vector Q6_V_vmpy_VhVb(HVX_Vector Vt, HVX_Vector Rs)`}
              </CodeBlock>
              <p>Multiply signed bytes by signed half-words → signed 16-bit results</p>
              <p><strong>Naming breakdown:</strong></p>
              <ul className="list-disc pl-6 my-4 space-y-1">
                <li>V (output): HVX_Vector register</li>
                <li>vmpy: multiply operation</li>
                <li>VhVb: Vh (signed half-word operand) × Vb (signed byte operand)</li>
              </ul>

              <p><strong>Semantics for Q6_V_vmpy_VhVb:</strong></p>
              <CodeBlock language="text" title="vmpy Semantics">
{`Vt: 64 × 16-bit signed values [h0, h1, ..., h63]
Rs: 128 × 8-bit signed values [b0, b1, ..., b127]

Multiply operation (element-wise pairing):
  h0 × b0 = 16-bit result
  h1 × b1 = 16-bit result
  ...
  h63 × b63 = 16-bit result

Output: 64 × 16-bit signed results`}
              </CodeBlock>

              <p><strong>Common usage in convolution:</strong></p>
              <CodeBlock language="c" title="vmpy Convolution Usage">
{`// Input: bytes (8-bit), Weights: half-words (16-bit)
HVX_Vector V_input_bytes = Q6_V_vmem_R(input_ptr);     // 128 × int8
HVX_Vector V_weights_hw = Q6_V_vmem_R(weights_ptr);   // 64 × int16

// Multiply
HVX_Vector V_products = Q6_V_vmpy_VhVb(V_weights_hw, V_input_bytes);`}
              </CodeBlock>

              <h4 className="text-lg font-semibold text-text-primary mt-6 mb-3">2.10.2 Multiply-Accumulate: Q6_Vh_vmpyacc_VhVb</h4>
              <CodeBlock language="c" title="Q6_Vh_vmpyacc_VhVb Signature">
{`HVX_Vector Q6_Vh_vmpyacc_VhVb(HVX_Vector Vd, HVX_Vector Vt, HVX_Vector Rs)`}
              </CodeBlock>
              <p><strong>Semantics</strong>: Vd += Vt × Rs (multiply-accumulate in a single instruction)</p>

              <p><strong>Data flow:</strong></p>
              <CodeBlock language="text" title="vmpyacc Data Flow">
{`Vd (input accumulator):  [a0    ][a1    ]...[a63    ]
Vt (weights):            [w0    ][w1    ]...[w63    ]
Rs (inputs):             [x0][x1][x2][x3]...[x127]

After vmpyacc:
Vd[0] += w0 × x0
Vd[1] += w1 × x1
...
Vd[63] += w63 × x63

New Vd:
                         [a0+w0×x0][a1+w1×x1]...[a63+w63×x63]`}
              </CodeBlock>

              <p><strong>Accumulation pattern in a loop:</strong></p>
              <CodeBlock language="c" title="vmpyacc Loop Pattern">
{`HVX_Vector V_accum = Q6_V_vsplat_R(0);  // Initialize to 0

for (int i = 0; i < num_iterations; i++) {
    HVX_Vector V_weights = Q6_V_vmem_R(weights_ptr + i * VECTOR_SIZE);
    HVX_Vector V_input = Q6_V_vmem_R(input_ptr + i * VECTOR_SIZE);

    // Accumulate products
    V_accum = Q6_Vh_vmpyacc_VhVb(V_accum, V_weights, V_input);
}

// Result in V_accum after loop`}
              </CodeBlock>

              <h4 className="text-lg font-semibold text-text-primary mt-6 mb-3">2.10.3 Dual Multiply: Q6_V_vdmpy_VhVb</h4>
              <CodeBlock language="c" title="Q6_V_vdmpy_VhVb Signature">
{`HVX_Vector Q6_V_vdmpy_VhVb(HVX_Vector Vt, HVX_Vector Rs)`}
              </CodeBlock>
              <p>
                <strong>Semantics</strong>: Execute two multiplies in parallel (useful for complex numbers or dual-channel operations).
              </p>
              <p><strong>Example: Complex number multiplication:</strong></p>
              <CodeBlock language="text" title="Complex Multiply Example">
{`Input: [Re0][Im0][Re1][Im1]...[Re31][Im31]  (32 × complex, 16-bit each)
Weight: [Wr][Wi][Wr][Wi]...[Wr][Wi]         (32 × complex weight)

Output:
  [Re0×Wr - Im0×Wi][Re0×Wi + Im0×Wr]...    (Complex results)`}
              </CodeBlock>

              <CalloutBox type="tip" title="Expert Insight">
                <p>
                  vdmpy is specialized for FFT, OFDM, and complex-valued signal processing. For neural networks,
                  vmpy and vmpyacc are the primary workhorses.
                </p>
              </CalloutBox>

              <h4 className="text-lg font-semibold text-text-primary mt-6 mb-3">2.10.4 Reduction Multiply-Accumulate: Q6_Ww_vrmpyacc_WwVhVb</h4>
              <CodeBlock language="c" title="Q6_Ww_vrmpyacc_WwVhVb Signature">
{`HVX_VectorPair Q6_Ww_vrmpyacc_WwVhVb(HVX_VectorPair Vdd, HVX_Vector Vt, HVX_Vector Rs)`}
              </CodeBlock>
              <p>Where Ww denotes a register pair holding 32-bit results.</p>
              <p><strong>Semantics</strong>: Reduce multiply-accumulate—multiply 64 pairs, sum pairs together → 32 accumulation results.</p>
              <p><strong>Data flow:</strong></p>
              <CodeBlock language="text" title="vrmpyacc Data Flow">
{`Vt (weights): 64 × 16-bit [w0, w1, w2, ..., w63]
Rs (input):   128 × 8-bit  [x0, x1, x2, ..., x127]

Pairing:
  Pair 0: w0×x0 + w1×x1
  Pair 1: w2×x2 + w3×x3
  ...
  Pair 31: w62×x62 + w63×x63

Output (Vdd as register pair): 32 × 32-bit results`}
              </CodeBlock>

              <p><strong>Critical for dot-product-heavy operations:</strong></p>
              <CodeBlock language="c" title="vrmpyacc in Dot Product">
{`HVX_VectorPair Vdd_accum = { Q6_V_vsplat_R(0), Q6_V_vsplat_R(0) };

for (int i = 0; i < num_iter; i++) {
    HVX_Vector V_weights = Q6_V_vmem_R(weights + i * 64);      // 64 × int16
    HVX_Vector V_inputs = Q6_V_vmem_R(inputs + i * 128);       // 128 × int8

    // Reduction MAC: produces 32 × 32-bit sums
    Vdd_accum = Q6_Ww_vrmpyacc_WwVhVb(Vdd_accum, V_weights, V_inputs);
}

// Extract results from register pair
HVX_Vector V_low = Vdd_accum.lo;   // First 16 results
HVX_Vector V_high = Vdd_accum.hi;  // Next 16 results`}
              </CodeBlock>

              <h4 className="text-lg font-semibold text-text-primary mt-6 mb-3">2.10.5 Multiply-Accumulate Across: Q6_V_vmpa_VVV</h4>
              <CodeBlock language="c" title="Q6_V_vmpa_VVV Signature">
{`HVX_Vector Q6_V_vmpa_VVV(HVX_Vector Vd, HVX_Vector Vt, HVX_Vector Rs)`}
              </CodeBlock>
              <p>
                <strong>Semantics</strong>: A variant that interprets the accumulation differently—useful for depthwise operations.
              </p>
            </SubSection>

            <SubSection id="shuffle-permute" title="2.11 Shuffle and Permute Operations">
              <p>
                Shuffle and permute operations rearrange elements within a vector. They are essential for:
              </p>
              <ul className="list-disc pl-6 my-4 space-y-1">
                <li>Input reordering to match weight layout</li>
                <li>Transposition for matrix operations</li>
                <li>Lane-level communication in reduction operations</li>
              </ul>

              <h4 className="text-lg font-semibold text-text-primary mt-6 mb-3">2.11.1 Vector Shuffle: Q6_V_vshuff_VVR</h4>
              <CodeBlock language="c" title="Q6_V_vshuff_VVR Signature">
{`HVX_Vector Q6_V_vshuff_VVR(HVX_Vector Vt, HVX_Vector Rs, int Rtt)`}
              </CodeBlock>
              <p><strong>Semantics</strong>: Selectively interleave or shuffle elements from two vectors based on a shift amount.</p>
              <p><strong>Example: Interleave odd/even:</strong></p>
              <CodeBlock language="c" title="vshuff Interleave Example">
{`HVX_Vector V0 = {0, 2, 4, 6, 8, ...};   // Even elements
HVX_Vector V1 = {1, 3, 5, 7, 9, ...};   // Odd elements

// Shuffle with shift=1 interleaves: [0,1,2,3,4,5,6,7,...]
HVX_Vector V_interleaved = Q6_V_vshuff_VVR(V1, V0, 1);`}
              </CodeBlock>

              <h4 className="text-lg font-semibold text-text-primary mt-6 mb-3">2.11.2 Vector Deal: Q6_V_vdeal_VVR</h4>
              <CodeBlock language="c" title="Q6_V_vdeal_VVR Signature">
{`HVX_Vector Q6_V_vdeal_VVR(HVX_Vector Vt, HVX_Vector Rs, int Rtt)`}
              </CodeBlock>
              <p><strong>Semantics</strong>: Inverse of shuffle—&quot;deals&quot; elements from two vectors into separate positions.</p>
              <p><strong>Example: De-interleave:</strong></p>
              <CodeBlock language="text" title="vdeal Example">
{`Input: [0, 1, 2, 3, 4, 5, 6, 7, ...]

After vdeal:
  V0 (even): [0, 2, 4, 6, ...]
  V1 (odd):  [1, 3, 5, 7, ...]`}
              </CodeBlock>

              <h4 className="text-lg font-semibold text-text-primary mt-6 mb-3">2.11.3 Vector Rotate: Q6_V_vror_VRI</h4>
              <CodeBlock language="c" title="Q6_V_vror_VRI Signature">
{`HVX_Vector Q6_V_vror_VRI(HVX_Vector Vt, int Rs)`}
              </CodeBlock>
              <p><strong>Semantics</strong>: Rotate elements circularly by Rs positions.</p>
              <CodeBlock language="c" title="Vector Rotate Example">
{`HVX_Vector V = {0, 1, 2, 3, 4, 5, ..., 127};

// Rotate right by 1 lane (at current lane width)
HVX_Vector V_rot = Q6_V_vror_VRI(V, 1);
// In 8-bit mode: {127, 0, 1, 2, ..., 126}
// In 16-bit mode: {63, 0, 1, 2, ..., 62}`}
              </CodeBlock>
              <p><strong>Performance</strong>: 1 cycle (very fast for lane-level communication)</p>

              <h4 className="text-lg font-semibold text-text-primary mt-6 mb-3">2.11.4 Vector Align: Q6_V_vlalign_VVI</h4>
              <CodeBlock language="c" title="Q6_V_vlalign_VVI Signature">
{`HVX_Vector Q6_V_vlalign_VVI(HVX_Vector Vt, HVX_Vector Rs, int Rii)`}
              </CodeBlock>
              <p><strong>Semantics</strong>: Align (shift/rotate) two vectors to extract a contiguous sub-vector.</p>
              <p><strong>Use case: Sliding window in convolution:</strong></p>
              <CodeBlock language="text" title="vlalign Sliding Window">
{`V_prev: [... last 16 bytes of previous tile]
V_curr: [first 112 bytes of current tile ...]

vlalign with appropriate shift:
Result: [last few of V_prev | first few of V_curr]
       = [16 bytes of valid sliding window]`}
              </CodeBlock>

              <h4 className="text-lg font-semibold text-text-primary mt-6 mb-3">2.11.5 Pack Even/Odd</h4>
              <CodeBlock language="c" title="Pack Even/Odd Signatures">
{`HVX_Vector Q6_Vb_vpacke_Vh(HVX_Vector Vt);  // Pack even lanes of 16-bit to 8-bit
HVX_Vector Q6_Vb_vpacko_Vh(HVX_Vector Vt);  // Pack odd lanes of 16-bit to 8-bit`}
              </CodeBlock>
              <p><strong>Semantics</strong>: Narrow from 16-bit to 8-bit by extracting even or odd lanes with saturation.</p>
              <CodeBlock language="text" title="Pack Even/Odd Example">
{`Vt (64 × 16-bit): [100, 200, 300, 400, ..., 8000, 8100]

vpacke (even lanes):  [100, 300, ..., 8000]  → saturate to int8
vpacko (odd lanes):   [200, 400, ..., 8100]  → saturate to int8`}
              </CodeBlock>
              <p><strong>Saturation semantics:</strong></p>
              <ul className="list-disc pl-6 my-4 space-y-1">
                <li>Values &gt; 127 saturate to 127</li>
                <li>Values &lt; -128 saturate to -128</li>
              </ul>
            </SubSection>

            <SubSection id="reduction-ops" title="2.12 Reduction Operations">
              <h4 className="text-lg font-semibold text-text-primary mt-6 mb-3">2.12.1 Horizontal Add</h4>
              <p>Horizontal addition sums elements across a vector.</p>
              <p><strong>Pattern: Tree reduction:</strong></p>
              <CodeBlock language="text" title="Tree Reduction Pattern">
{`Input V0 (8 × 32-bit): [a0, a1, a2, a3, a4, a5, a6, a7]

Step 1: Add adjacent pairs
  [a0+a1, a2+a3, a4+a5, a6+a7]

Step 2: Add adjacent pairs again
  [a0+a1+a2+a3, a4+a5+a6+a7]

Step 3: Final sum
  [a0+a1+a2+a3+a4+a5+a6+a7]`}
              </CodeBlock>

              <CalloutBox type="info" title="Note">
                <p>HVX doesn&apos;t have a direct &quot;sum all lanes&quot; intrinsic. Horizontal reduction typically requires:</p>
                <ol className="list-decimal pl-6 mt-2 space-y-1">
                  <li>Multiple vror + vadd to combine lanes hierarchically, or</li>
                  <li>Scalar extraction via pointer cast (slow)</li>
                </ol>
              </CalloutBox>

              <h4 className="text-lg font-semibold text-text-primary mt-6 mb-3">2.12.2 Dot Product via Reduction MAC</h4>
              <p>The most efficient dot product uses vrmpyacc:</p>
              <CodeBlock language="c" title="Dot Product via vrmpyacc" collapsible defaultCollapsed>
{`int32_t dot_product_hvx(const int8_t *a, const int16_t *w, int length) {
    HVX_VectorPair Vdd_accum = {
        Q6_V_vsplat_R(0),
        Q6_V_vsplat_R(0)
    };

    for (int i = 0; i < length; i += 64) {
        HVX_Vector V_a = Q6_V_vmem_R((const void *)(a + i));      // 128 × int8
        HVX_Vector V_w = Q6_V_vmem_R((const void *)(w + i));      // 64 × int16

        // Reduction MAC: 64 multiplies → 32 partial sums
        Vdd_accum = Q6_Ww_vrmpyacc_WwVhVb(Vdd_accum, V_w, V_a);
    }

    // Extract and sum the 32 partial results from register pair
    int32_t *results_low = (int32_t *)&Vdd_accum.lo;
    int32_t *results_high = (int32_t *)&Vdd_accum.hi;

    int32_t final_sum = 0;
    for (int i = 0; i < 16; i++) {
        final_sum += results_low[i] + results_high[i];
    }

    return final_sum;
}`}
              </CodeBlock>
            </SubSection>
          </TheorySection>

          {/* ============ Convolution Kernels ============ */}
          <TheorySection id="convolution-kernels" title="Writing Efficient Convolution Kernels">
            <SubSection id="conv-fundamentals" title="2.13 Convolution Fundamentals in HVX">
              <p>A 2D convolution operation:</p>
              <CodeBlock language="text" title="2D Convolution Formula">
{`output[y][x] = sum over (ky, kx) in kernel:
                    input[y + ky][x + kx] × kernel[ky][kx] + bias`}
              </CodeBlock>
              <p>For INT8 quantized inference:</p>
              <ul className="list-disc pl-6 my-4 space-y-1">
                <li>Input: INT8 (range ±128)</li>
                <li>Weights: INT8 (range ±128)</li>
                <li>Accumulation: INT32 (range ±2^31)</li>
                <li>Output: INT8 (after requantization)</li>
              </ul>
            </SubSection>

            <SubSection id="conv-3x3" title="2.14 Simple 3×3 Conv2D in HVX: INT8">
              <h4 className="text-lg font-semibold text-text-primary mt-6 mb-3">2.14.1 Data Layout for 3×3 Convolution</h4>
              <p><strong>Input layout</strong> (CHW format, tile-wise):</p>
              <CodeBlock language="text" title="Input Buffer Layout">
{`Input buffer in VTCM (tile: 18×18 input for 16×16 output):
┌─────────────────────────────────────────┐
│ Row 0:  [16 input pixels] [2 padding]   │ (18 bytes)
│ Row 1:  [16 input pixels] [2 padding]   │
│ ...
│ Row 17: [16 input pixels] [2 padding]   │
└─────────────────────────────────────────┘

Reason for 18×18:
  16 output pixels require 16+2 = 18 input pixels (1-pixel border padding)
  16 output rows require 16+2 = 18 input rows`}
              </CodeBlock>

              <h4 className="text-lg font-semibold text-text-primary mt-6 mb-3">2.14.2 Annotated 3×3 Conv Implementation</h4>
              <CodeBlock language="c" title="Naive 3×3 Conv2D INT8 HVX Implementation" collapsible defaultCollapsed>
{`/**
 * Convolution 3x3 for INT8 quantized inference
 *
 * Parameters:
 *   input_tile:     18x18 INT8 (padded input)
 *   kernel:         9 INT8 coefficients (flattened 3x3)
 *   bias:           INT32 per-channel bias
 *   output_tile:    16x16 INT8 (output)
 *   input_depth:    Number of input channels
 *   shift:          Right-shift amount for requantization
 *   zero_point:     Zero point for output quantization
 */

void conv3x3_int8_hvx(
    const int8_t *input_tile,      // 18×18×C (row-major, C input channels)
    const int8_t *kernel,          // 9×C (3×3×C filters)
    const int32_t *bias,           // Per-channel bias
    int8_t *output_tile,           // 16×16 output
    int input_depth,
    int shift,
    int zero_point)
{
    // Step 1: Initialize accumulators
    HVX_Vector V_accum[16];  // One accumulator per output row
    for (int i = 0; i < 16; i++) {
        V_accum[i] = Q6_V_vsplat_R(bias[0]);  // Splat bias across lanes
    }

    // Step 2: Convolution loop
    for (int ky = 0; ky < 3; ky++) {
        for (int kx = 0; kx < 3; kx++) {
            for (int out_y = 0; out_y < 16; out_y++) {
                int in_y = out_y + ky;

                // Load input row
                const int8_t *row_ptr = input_tile + in_y * 18;
                HVX_Vector V_input = Q6_V_vmem_R((const void *)row_ptr);

                // Load kernel coefficient
                int8_t kernel_coeff = kernel[ky * 3 + kx];
                HVX_Vector V_kernel = Q6_V_vsplat_R(kernel_coeff);

                // Multiply-accumulate
                V_accum[out_y] = Q6_Vh_vmpyacc_VhVb(
                    V_accum[out_y],
                    V_kernel,
                    V_input
                );
            }
        }
    }

    // Step 3: Requantize from INT32 → INT8
    for (int out_y = 0; out_y < 16; out_y++) {
        HVX_Vector V_shifted = Q6_V_vasrw_VVI(V_accum[out_y], shift);
        HVX_Vector V_zero = Q6_V_vsplat_R(zero_point);
        HVX_Vector V_with_zp = Q6_V_vaddw_VVV(V_shifted, V_zero);

        // Pack down to 8-bit with saturation
        HVX_Vector V_even = Q6_Vh_vpacke_Vw(V_accum[out_y]);
        HVX_Vector V_odd = Q6_Vh_vpacko_Vw(V_accum[out_y]);
        HVX_Vector V_output = Q6_V_vpacke_VhVh(V_even, V_odd);

        // Store output row
        Q6_V_vmem_ARI(output_tile + out_y * 16, V_output);
    }
}`}
              </CodeBlock>

              <CalloutBox type="critical" title="Critical Detail">
                <p>The above is a <strong>naive</strong> implementation. A production kernel would:</p>
                <ol className="list-decimal pl-6 mt-2 space-y-1">
                  <li>Process multiple channels (depth) in parallel</li>
                  <li>Batch process output rows (software pipeline)</li>
                  <li>Use VTCM for weight caching</li>
                  <li>Apply loop unrolling and dual-issue scheduling</li>
                </ol>
              </CalloutBox>
            </SubSection>
          </TheorySection>

          {/* ============ Depthwise Convolution ============ */}
          <TheorySection id="depthwise-convolution" title="Vectorizing Depthwise Convolution">
            <SubSection id="depthwise-strategy" title="2.15 Depthwise Convolution Strategy">
              <p>Depthwise convolution processes each input channel independently:</p>
              <CodeBlock language="text" title="Depthwise Convolution Formula">
{`For each input channel c:
  For each spatial position (y, x):
    output[c][y][x] = sum over (ky, kx):
                          input[c][y+ky][x+kx] × kernel[c][ky][kx]`}
              </CodeBlock>
              <p>Key difference from standard conv:</p>
              <ul className="list-disc pl-6 my-4 space-y-1">
                <li>Each channel has its own 3×3 kernel</li>
                <li>No cross-channel communication</li>
                <li>More parallelizable across channels</li>
              </ul>
            </SubSection>

            <SubSection id="depthwise-reordering" title="2.16-2.17 Efficient Depthwise: Data Reordering">
              <p><strong>Problem</strong>: Depthwise conv has poor data locality—each channel&apos;s kernel is separate.</p>
              <p><strong>Solution</strong>: Pre-reorder data into <strong>&quot;channel-packed&quot;</strong> format:</p>
              <CodeBlock language="text" title="Channel-Packed Layout">
{`Original layout (interleaved channels):
[C0, C1, C2, ..., C7, C0, C1, C2, ..., C7, ...]

Channel-packed layout (grouped by channel):
[C0, C0, C0, ..., C0, C1, C1, C1, ..., C1, ..., C7, C7, ..., C7]
│─ 64 values ─│ │─ 64 values ─│         │─ 64 values ─│`}
              </CodeBlock>

              <CodeBlock language="c" title="Optimized Depthwise with Channel Packing" collapsible defaultCollapsed>
{`void depthwise_conv3x3_hvx_optimized(
    const int8_t *input_packed,    // Pre-reordered into channel-packed format
    const int8_t *kernel_packed,
    const int32_t *bias,
    int8_t *output_packed,
    int num_tiles,
    int shift,
    int zero_point)
{
    for (int tile = 0; tile < num_tiles; tile++) {
        HVX_Vector V_accum[8];  // One accumulator per channel

        for (int c = 0; c < 8; c++) {
            V_accum[c] = Q6_V_vsplat_R(bias[c]);
        }

        // 3×3 kernel with channel-parallel processing
        for (int ky = 0; ky < 3; ky++) {
            for (int kx = 0; kx < 3; kx++) {
                int in_offset = tile * (16*16*8) + (ky * 16 + kx) * 64;
                HVX_Vector V_input[8];

                for (int c = 0; c < 8; c++) {
                    V_input[c] = Q6_V_vmem_R((const void *)(input_packed + in_offset + c * 64));
                }

                HVX_Vector V_kernel[8];
                for (int c = 0; c < 8; c++) {
                    int8_t k = kernel_packed[c * 9 + ky * 3 + kx];
                    V_kernel[c] = Q6_V_vsplat_R(k);
                }

                // Multiply-accumulate for all channels in parallel
                for (int c = 0; c < 8; c++) {
                    V_accum[c] = Q6_Vh_vmpyacc_VhVb(V_accum[c], V_kernel[c], V_input[c]);
                }
            }
        }

        // Requantize and store for all channels
        for (int c = 0; c < 8; c++) {
            HVX_Vector V_shifted = Q6_V_vasrw_VVI(V_accum[c], shift);
            HVX_Vector V_output = Q6_Vb_vsaturate_Vw(V_shifted);
            Q6_V_vmem_ARI(output_packed + tile * (16*16*8) + c * 128, V_output);
        }
    }
}`}
              </CodeBlock>
            </SubSection>
          </TheorySection>

          {/* ============ Pointwise GEMM ============ */}
          <TheorySection id="pointwise-gemm" title="Pointwise Convolution and GEMM for INT8">
            <SubSection id="pointwise-conv" title="2.18 Pointwise Convolution (1×1 Conv)">
              <p>Pointwise convolution is essentially a <strong>matrix multiply</strong>:</p>
              <CodeBlock language="text" title="Pointwise Conv Formula">
{`output[h][w][oc] = sum over ic:
                       input[h][w][ic] × weight[oc][ic]`}
              </CodeBlock>
              <p>Where:</p>
              <ul className="list-disc pl-6 my-4 space-y-1">
                <li>Input: H × W × IC</li>
                <li>Weight: OC × IC</li>
                <li>Output: H × W × OC</li>
              </ul>
              <p>This becomes a <strong>batch of matrix multiplies</strong> (one per spatial location).</p>
            </SubSection>

            <SubSection id="gemm-vrmpyacc" title="2.19 GEMM Implementation Using vrmpyacc">
              <CodeBlock language="c" title="INT8 GEMM Implementation" collapsible defaultCollapsed>
{`/**
 * General Matrix Multiply for INT8:
 *   C[m×n] = A[m×k] × B[k×n] + bias[n]
 */

void gemm_int8_hvx(
    const int8_t *A,       // m × k
    const int8_t *B,       // k × n (pre-transposed)
    int8_t *C,
    const int32_t *bias,   // n elements
    int m, int k, int n,
    int shift, int zero_point)
{
    const int TILE_M = 16;
    const int TILE_N = 16;

    for (int i_tile = 0; i_tile < m; i_tile += TILE_M) {
        for (int j_tile = 0; j_tile < n; j_tile += TILE_N) {
            HVX_VectorPair Vdd_accum[TILE_M];
            for (int i = 0; i < TILE_M; i++) {
                Vdd_accum[i].lo = Q6_V_vsplat_R(bias[j_tile]);
                Vdd_accum[i].hi = Q6_V_vsplat_R(bias[j_tile]);
            }

            // Compute A[i_tile:i_tile+TILE_M, :] × B[:, j_tile:j_tile+TILE_N]
            for (int k_idx = 0; k_idx < k; k_idx++) {
                const int8_t *B_row = B + k_idx * n + j_tile;
                HVX_Vector V_B_row = Q6_V_vmem_R((const void *)B_row);

                for (int i = 0; i < TILE_M; i++) {
                    int A_idx = (i_tile + i) * k + k_idx;
                    int8_t A_val = A[A_idx];

                    HVX_Vector V_A_splat = Q6_V_vsplat_R(A_val);

                    HVX_Vector V_product = Q6_V_vmpy_VhVb(V_A_splat, V_B_row);

                    Vdd_accum[i] = Q6_Ww_vadd_WwWw(
                        Vdd_accum[i],
                        {V_product, Q6_V_vsplat_R(0)}
                    );
                }
            }

            // Requantize and store this tile of output
            for (int i = 0; i < TILE_M; i++) {
                HVX_Vector V_result_low = Vdd_accum[i].lo;
                V_result_low = Q6_V_vasrw_VVI(V_result_low, shift);

                HVX_Vector V_zp = Q6_V_vsplat_R(zero_point);
                V_result_low = Q6_V_vaddw_VVV(V_result_low, V_zp);

                HVX_Vector V_even = Q6_Vh_vpacke_Vw(V_result_low);
                HVX_Vector V_output_low = Q6_Vb_vpacke_Vh(V_even);

                int C_idx = (i_tile + i) * n + j_tile;
                Q6_V_vmem_ARI((void *)(C + C_idx), V_output_low);
            }
        }
    }
}`}
              </CodeBlock>
            </SubSection>

            <SubSection id="vtcm-tiling" title="2.20 VTCM Tiling for Large GEMMs">
              <p>When matrices are too large for the register file, use <strong>VTCM blocking</strong>:</p>
              <CodeBlock language="text" title="VTCM Layout for Tiled GEMM">
{`VTCM Layout (assuming 256 KB VTCM):
┌─────────────────────────────────────┐
│ A_tile (16×128 = 2 KB)              │ (Reloaded each iteration)
├─────────────────────────────────────┤
│ B_tile (128×16 = 2 KB)              │ (Loaded once per block)
├─────────────────────────────────────┤
│ C_tile (16×16 = 256 bytes)          │ (Output accumulator)
├─────────────────────────────────────┤
│ Free space for intermediate buffers  │ (~250 KB)
└─────────────────────────────────────┘`}
              </CodeBlock>

              <CodeBlock language="c" title="VTCM Tiled GEMM" collapsible defaultCollapsed>
{`void gemm_int8_vtcm_tiled(
    const int8_t *A_ddr,
    const int8_t *B_ddr,
    int8_t *C_ddr,
    int m, int k, int n)
{
    int8_t *A_tile_vtcm = vtcm_buffer;
    int8_t *B_tile_vtcm = A_tile_vtcm + (16 * 128);
    int32_t *C_tile_vtcm = (int32_t *)(B_tile_vtcm + (128 * 16));

    const int TILE_M = 16;
    const int TILE_K = 128;
    const int TILE_N = 16;

    for (int i = 0; i < m; i += TILE_M) {
        for (int j = 0; j < n; j += TILE_N) {
            // Initialize C_tile
            for (int ii = 0; ii < TILE_M; ii++) {
                for (int jj = 0; jj < TILE_N; jj++) {
                    C_tile_vtcm[ii * TILE_N + jj] = 0;
                }
            }

            // K-loop: process A and B in chunks
            for (int kk = 0; kk < k; kk += TILE_K) {
                int chunk_k = min(TILE_K, k - kk);

                // DMA: Load A and B to VTCM
                dma_memcpy_ddr_to_vtcm(A_tile_vtcm, A_ddr + i * k + kk, TILE_M * chunk_k);
                dma_memcpy_ddr_to_vtcm(B_tile_vtcm, B_ddr + kk * n + j, chunk_k * TILE_N);

                // Compute C += A_tile × B_tile (using HVX)
                gemm_compute_tile_hvx(A_tile_vtcm, B_tile_vtcm, C_tile_vtcm, TILE_M, chunk_k, TILE_N);
            }

            // DMA: Store C back to DDR
            dma_memcpy_vtcm_to_ddr(C_ddr + i * n + j, C_tile_vtcm, TILE_M * TILE_N);
        }
    }
}`}
              </CodeBlock>
            </SubSection>

            <SubSection id="weight-packing" title="2.21 Weight Packing Strategies">
              <p><strong>Efficient weight formats for GEMM:</strong></p>
              <ol className="list-decimal pl-6 my-4 space-y-2">
                <li>
                  <strong>Transposed format</strong>: B stored as (K × N) instead of (N × K)
                  <ul className="list-disc pl-6 mt-1 space-y-1">
                    <li>Enables contiguous row loads during MAC loop</li>
                  </ul>
                </li>
                <li>
                  <strong>Channel-interleaved</strong>: For multi-channel weights
                  <CodeBlock language="text" title="Channel Interleaving">
{`Standard: [OC0_IC0, OC0_IC1, ..., OC0_ICk, OC1_IC0, ...]
Packed:   [OC0_IC0, OC1_IC0, OC2_IC0, ..., OCm_IC0, OC0_IC1, ...]`}
                  </CodeBlock>
                </li>
                <li>
                  <strong>Preconditioned scales</strong>: Pre-multiply weights by quantization scale
                </li>
              </ol>
            </SubSection>
          </TheorySection>

          {/* ============ Predicates and Masks ============ */}
          <TheorySection id="predicates-masks" title="Predicate Registers and Masked Operations">
            <SubSection id="q-register-semantics" title="2.22 Q Register Semantics">
              <p>Predicate registers (Q0–Q3) enable <strong>data-dependent execution</strong> within vectors.</p>
              <p><strong>Generated by comparison operations:</strong></p>
              <CodeBlock language="c" title="Predicate Generation">
{`// Compare result produces a predicate
HVX_VectorPred Q0 = Q6_Q_vcmp_equ_VbVb(V0, V1);  // Q0[i] = 1 if V0[i] == V1[i]`}
              </CodeBlock>

              <p><strong>Bit interpretation by lane width:</strong></p>
              <CodeBlock language="text" title="Predicate Bit Interpretation">
{`For 8-bit lane width (128 lanes):
  Q0 = 128 bits, each bit corresponds to one lane

For 16-bit lane width (64 lanes):
  Q0 = 128 bits, but only lower 64 bits are valid
  (upper 64 bits should be zero or undefined)`}
              </CodeBlock>
            </SubSection>

            <SubSection id="masked-operations" title="2.23 Masked Operations">
              <p><strong>Conditional select</strong>: result = Q0 ? V_true : V_false</p>
              <CodeBlock language="c" title="Conditional Select">
{`HVX_Vector V_result = Q6_V_vmux_QVV(Q0, V_true_value, V_false_value);
// For each lane i:
//   result[i] = Q0[i] ? V_true_value[i] : V_false_value[i]`}
              </CodeBlock>

              <p><strong>Vectorized boundary handling with predicates:</strong></p>
              <CodeBlock language="c" title="Vectorized Boundary Handling" collapsible defaultCollapsed>
{`void conv_boundary_vectorized_hvx(
    const int8_t *input_tile,   // 18×18
    const int8_t *kernel,
    int8_t *output_tile,        // 16×16
    int shift)
{
    HVX_Vector V_zero = Q6_V_vsplat_R(0);

    // Create a mask for valid positions (interior 16×16 of 18×18 input)
    HVX_VectorPred Q_valid = Q6_Q_vsetq_R(16 * 128);

    HVX_Vector V_accum = Q6_V_vsplat_R(0);

    for (int ky = 0; ky < 3; ky++) {
        for (int kx = 0; kx < 3; kx++) {
            int in_y = ky;
            HVX_Vector V_input = Q6_V_vmem_R((const void *)(input_tile + in_y * 18));

            // Apply padding mask
            HVX_Vector V_padded = Q6_V_vmux_QVV(Q_valid, V_input, V_zero);

            int8_t kernel_coeff = kernel[ky * 3 + kx];
            HVX_Vector V_kernel = Q6_V_vsplat_R(kernel_coeff);
            V_accum = Q6_Vh_vmpyacc_VhVb(V_accum, V_kernel, V_padded);
        }
    }

    HVX_Vector V_output = Q6_Vb_vsaturate_Vw(Q6_V_vasrw_VVI(V_accum, shift));
    Q6_V_vmem_ARI(output_tile, V_output);
}`}
              </CodeBlock>
            </SubSection>
          </TheorySection>

          {/* ============ Loop Alignment and Software Pipelining ============ */}
          <TheorySection id="pipelining" title="Loop Alignment and Software Pipelining">
            <SubSection id="falign" title="2.24 The .falign Directive">
              <p>
                The Hexagon processor has a <strong>packet-based instruction encoding</strong>. Multiple instructions can
                be grouped into a <strong>packet</strong> (up to 4 instructions in most Snapdragon generations) and
                execute in parallel.
              </p>
              <p><strong>Problem</strong>: Loop entry and exit don&apos;t naturally align to packet boundaries.</p>
              <p><strong>Solution</strong>: Use .falign (function align) to align loop entry to a 8-byte boundary:</p>
              <CodeBlock language="text" title=".falign Directive Example">
{`    .p2align 3                    // Align to 8 bytes (2^3)

.loop_start:
    {
        V0 = vmem(R0++#1)       ; Instruction 1
        V1 = vmem(R1++#1)       ; Instruction 2
        R2 = add(R2, #1)        ; Instruction 3
        if (P0) jump .loop_start ; Instruction 4
    }`}
              </CodeBlock>
            </SubSection>

            <SubSection id="prologue-epilogue" title="2.25 Loop Prologue and Epilogue">
              <p>
                <strong>Problem</strong>: A pipelined loop requires multiple iterations to &quot;fill&quot; the pipeline before
                producing useful results.
              </p>
              <p><strong>Solution</strong>: Separate into prologue, kernel, and epilogue:</p>
              <CodeBlock language="text" title="Prologue/Kernel/Epilogue Structure">
{`Prologue:
  Issue iteration 1, 2, ..., n (pipeline fill)

Kernel:
  Issue iteration n+1, n+2, ... (steady state)
  Data from iteration 1 is now ready

Epilogue:
  Drain remaining iterations from pipeline
  (Issue new iterations without completing old ones)`}
              </CodeBlock>

              <CodeBlock language="c" title="Software-Pipelined Dot Product" collapsible defaultCollapsed>
{`int32_t dot_product_pipelined(const int8_t *a, const int16_t *w, int length) {
    HVX_Vector V_accum = Q6_V_vsplat_R(0);

    // Prologue: Load first iteration
    HVX_Vector V_data_prev = Q6_V_vmem_R((const void *)a);

    // Main loop with pipeline
    int i = 0;
    for (i = 1; i < (length / 64) - 1; i++) {
        HVX_Vector V_data_curr = Q6_V_vmem_R((const void *)(a + i * 64));
        HVX_Vector V_weights = Q6_V_vmem_R((const void *)(w + i * 64));
        V_accum = Q6_Ww_vrmpyacc_WwVhVb(
            {V_accum, Q6_V_vsplat_R(0)},
            V_weights,
            V_data_prev
        ).lo;
        V_data_prev = V_data_curr;
    }

    // Epilogue: Process final iteration
    HVX_Vector V_weights_final = Q6_V_vmem_R((const void *)(w + i * 64));
    V_accum = Q6_Ww_vrmpyacc_WwVhVb(
        {V_accum, Q6_V_vsplat_R(0)},
        V_weights_final,
        V_data_prev
    ).lo;

    int32_t *results = (int32_t *)&V_accum;
    int32_t sum = 0;
    for (int j = 0; j < 32; j++) sum += results[j];

    return sum;
}`}
              </CodeBlock>
            </SubSection>

            <SubSection id="auto-pipelining" title="2.26 Compiler Auto-Pipelining">
              <p>
                The Hexagon compiler (HexagonTools) can <strong>automatically detect and pipeline loops</strong> under
                certain conditions:
              </p>
              <p><strong>Conditions for auto-pipelining:</strong></p>
              <ol className="list-decimal pl-6 my-4 space-y-1">
                <li>Loop has a predictable trip count</li>
                <li>Instruction dependencies allow for overlapping</li>
                <li>No function calls inside the loop</li>
                <li>Minimal register pressure</li>
              </ol>
            </SubSection>

            <SubSection id="initiation-interval" title="2.27 Initiation Interval (II) and Throughput">
              <p><strong>Initiation Interval (II)</strong>: Cycles between issuing successive iterations.</p>
              <CodeBlock language="text" title="Initiation Interval Example">
{`Unpipelined loop:
  Iteration 1: Load (1 cycle) → MAC (4 cycles) → Store (1 cycle) = 6 cycles
  Iteration 2 starts after iteration 1 completes
  Throughput: 6 cycles per iteration

Pipelined loop (II = 2):
  Cycle 0: Load iter1, MAC iter2, Store iter3
  Cycle 1: Load iter2, MAC iter3, Store iter4
  Cycle 2: Load iter3, MAC iter4, Store iter5
  ...
  Throughput: 2 cycles per iteration (3× faster!)`}
              </CodeBlock>
              <p><strong>Achieving II=1</strong>: Requires careful scheduling and sufficient registers.</p>
            </SubSection>
          </TheorySection>

          {/* ============ HVX Intrinsics vs Inline Assembly ============ */}
          <TheorySection id="intrinsics-vs-asm" title="HVX Intrinsics vs Inline Assembly">
            <SubSection id="when-to-use-each" title="2.28 When to Use Each">
              <p><strong>HVX Intrinsics (Recommended):</strong></p>
              <ul className="list-disc pl-6 my-4 space-y-1">
                <li>Correctness guaranteed by compiler</li>
                <li>Auto-scheduling for optimal instruction parallelism</li>
                <li>Easier to maintain and debug</li>
                <li>Portable across Hexagon versions (mostly)</li>
              </ul>
              <p><strong>Inline Assembly:</strong></p>
              <ul className="list-disc pl-6 my-4 space-y-1">
                <li>When intrinsics don&apos;t exist for a specific operation</li>
                <li>Tight inner loops where micro-optimizations matter</li>
                <li>Fine-grained control over instruction scheduling</li>
                <li>Risk of introducing bugs (compiler won&apos;t help)</li>
              </ul>
            </SubSection>

            <SubSection id="inline-asm-syntax" title="2.29 Inline Asm Syntax for Hexagon">
              <CodeBlock language="c" title="Inline Asm Custom MAC">
{`// Example: Custom multiply-accumulate via inline asm
HVX_Vector custom_mac_asm(
    HVX_Vector V_accum,
    HVX_Vector V_weights,
    HVX_Vector V_input)
{
    HVX_Vector result;

    asm("{ %0 = vmpyacc(%1, %2, %3) }"
        : "=r" (result)
        : "r" (V_accum), "r" (V_weights), "r" (V_input)
        : );

    return result;
}`}
              </CodeBlock>
            </SubSection>

            <SubSection id="packet-construction-asm" title="2.30 Packet Construction in Inline Asm">
              <p>Hexagon&apos;s parallel packet syntax in inline asm:</p>
              <p><strong>Packet rules:</strong></p>
              <ol className="list-decimal pl-6 my-4 space-y-1">
                <li>Each curly-brace pair encloses one packet (up to 4 instructions)</li>
                <li>Instructions in the same packet execute in parallel</li>
                <li>No data dependencies within a packet (compiler enforces)</li>
                <li>Comments within asm strings must use // or ;</li>
              </ol>
            </SubSection>

            <SubSection id="pmu" title="2.31 Performance Monitoring">
              <p>Use the <strong>Hexagon PMU (Performance Monitoring Unit)</strong> to measure cycles, instructions, and CPI of kernel execution. PMU access is via mrc instructions reading specific coprocessor registers (p10, c9). Compare measured cycle count against theoretical roofline to identify opportunities for optimization.</p>
            </SubSection>
          </TheorySection>

          <KeyTakeaways items={[
            'HVX provides 32 vector registers (V0-V31) at 128 bytes each in HVX-128 mode (256 bytes in HVX-256), with 4 predicate registers (Q0-Q3). The total register file is only ~4 KB — wide but not deep.',
            'The same vector register can be interpreted at 8/16/32/64-bit lane widths (128/64/32/16 lanes respectively). This dynamic lane interpretation enables efficient quantized inference without format conversion.',
            'Quantized inference requires widening early (8→16 bit before multiply) and narrowing late (32→16→8 bit with saturation). Use vunpacke/vunpacko for widening and vpacke/vpacko for narrowing.',
            'The vrmpyacc reduction multiply-accumulate is the workhorse for dot products and GEMM: it multiplies 64 pairs of values and produces 32 × 32-bit accumulated results in a single instruction.',
            'Aligned loads (Q6_V_vmem_R) cost 1 cycle; unaligned loads (vmemu) cost 2-3 cycles — 3× penalty. Always align hot data to 128-byte boundaries.',
            'For large GEMMs and convolutions, tile into VTCM-sized blocks (256 KB on v75) with DMA double-buffering. The register file is too small for even modest matrices.',
            'Software pipelining with prologue/kernel/epilogue achieves II=2 (or better) throughput, often 3-5× faster than naive loops. The Hexagon compiler can auto-pipeline if the loop has predictable trip count and minimal register pressure.',
          ]} />

          {/* ============ Self Assessment ============ */}
          <div id="self-assessment">
            <SelfAssessment questions={[
              {
                question: 'Q1 (Register File): You have a 128-byte vector register (HVX-128 mode) containing 16-bit signed integers. How many lanes does this register have?',
                answer: 'Correct answer: (c) 64. Explanation: 128 bytes = 1024 bits. With 16-bit lanes, 1024 / 16 = 64 lanes. The same register can be interpreted as 128 × 8-bit, 64 × 16-bit, 32 × 32-bit, or 16 × 64-bit, depending on the operation\'s data width.',
              },
              {
                question: 'Q2 (Register Pairs): When using register pairs (e.g., V0:1) for a 32-bit multiply-accumulate, which registers are valid pairs?',
                answer: 'Correct answer: (b) V0:1, V2:3, V4:5, ..., V30:31. Register pairs in HVX must always be even:odd combinations. This is a hardware constraint to maintain symmetry in the vector pipeline. You cannot pair V1:2 or V3:4.',
              },
              {
                question: 'Q3 (Predicate Registers): What is the purpose of predicate registers (Q0–Q3)?',
                answer: 'Correct answer: (b) To enable conditional/masked operations. Q registers store per-lane condition bits generated from comparisons (e.g., Q6_Q_vcmp_equ_VbVb). They are used in masked operations like vmux to conditionally select between two vectors lane-by-lane. Unlike AVX-512 mask registers, HVX predicates are tied specifically to vector-level comparisons and are designed for data-parallel workloads where branch divergence is rare.',
              },
              {
                question: 'Q4 (Data Types): In INT8 quantized inference, what is the typical accumulation data width needed to prevent overflow from 8-bit multiply?',
                answer: 'Correct answer: (c) 32-bit (for sums of products). Explanation: An int8 × int8 multiply produces a result that fits in 16 bits. But when accumulating many products in a dot product, the sum can grow large — for a kernel with 64+ taps, you can easily exceed 16-bit range. 32-bit accumulation provides headroom for sums of thousands of int8×int8 products. After accumulation, the 32-bit result is requantized back to int8.',
              },
              {
                question: 'Q5 (Widening): What do the intrinsics vunpacke and vunpacko do?',
                answer: 'Correct answer: (b) Extract even and odd lanes (widening from 8-bit to 16-bit). Explanation: vunpacke takes a 128 × int8 vector and extracts the 64 even-indexed elements, sign-extending each to 16-bit. vunpacko does the same for odd-indexed elements. Together they let you process all 128 input bytes as 128 × 16-bit values across two registers, providing the headroom needed for 8-bit×8-bit multiplies that can\'t safely stay in 8-bit.',
              },
              {
                question: 'Q6 (Loads): Which load intrinsic is fastest for aligned data access?',
                answer: 'Correct answer: (a) Q6_V_vmem_R (aligned). 1-cycle latency for 128-byte load when address is 128-byte aligned. vmemu (unaligned) costs 2-3 cycles because the hardware must perform 2-3 sub-vector loads and combine them. vgather is even slower (~5 cycles). Always align hot inner-loop buffers to 128 bytes using __attribute__((aligned(128))).',
              },
              {
                question: 'Q7 (Reduction): What does Q6_Ww_vrmpyacc_WwVhVb do differently from Q6_Vh_vmpyacc_VhVb?',
                answer: 'Correct answer: (b) It produces 32-bit outputs instead of 16-bit, with reduction across lanes. Q6_Vh_vmpyacc_VhVb does element-wise multiply-accumulate producing 64 × 16-bit results. Q6_Ww_vrmpyacc_WwVhVb is a reduction MAC: it multiplies 64 byte-pairs and adds them in pairs, producing 32 × 32-bit sums in a register pair. This is critical for dot products because it both does the multiplies AND a partial reduction in one instruction, halving the number of operations needed.',
              },
              {
                question: 'Q8 (Narrowing): In a convolution kernel, you need to narrow 32-bit accumulated results back to 8-bit INT8. Which sequence is correct?',
                answer: 'Correct answer: (c) Use vpacke (32→16) and vpacko (32→16), then combine and pack to 8-bit. There\'s no single 32→8 instruction in HVX. You first extract the even and odd 32-bit lanes into 16-bit values (vpacke/vpacko_Vw), then combine the 16-bit results and pack them into 8-bit with saturation (vpacke_Vh). Saturation ensures values outside [-128, 127] clamp to the boundaries instead of wrapping.',
              },
              {
                question: 'Q9 (Conv Kernel): For a 3×3 convolution kernel with INT8 inputs and weights, what is the minimum number of vector registers needed to store the 9 kernel coefficients?',
                answer: 'Correct answer: (c) Depends on the implementation; could be 1 if using splat. The 9 coefficients are tiny (9 bytes), so you could fit them all in a single vector register. But for an efficient kernel, you typically splat each coefficient across all lanes (Q6_V_vsplat_R) before the multiply-accumulate, since the kernel coefficient is broadcast against many input pixels. This uses 1 register at a time per coefficient. Some implementations preload all 9 splatted coefficients into V0-V8 to minimize splat operations in the inner loop.',
              },
              {
                question: 'Q10 (Depthwise): In a depthwise convolution (vs. standard convolution), what is the key advantage?',
                answer: 'Correct answer: (b) Each channel is processed independently—better data locality. In standard convolution, each output element requires summing across all input channels (cross-channel reduction). In depthwise convolution, each input channel maps to one output channel with no cross-channel summation. This means: (1) better data locality per channel, (2) more parallelism opportunity (channels are independent), (3) much lower compute (fewer total multiplies). Total ops: standard conv = K_h × K_w × C_in × C_out × H × W; depthwise = K_h × K_w × C × H × W. For C=64, this is 64× fewer ops.',
              },
              {
                question: 'Q11 (Large GEMM): For a 1000×1000 GEMM in DDR memory (too large for VTCM), what is the primary HVX optimization technique?',
                answer: 'Correct answer: (b) Tile into VTCM-sized blocks and pipeline loads/computes. The full 1000×1000 INT8 matrix is 1 MB — far larger than the 256 KB VTCM. The technique: divide into 16×128 or 16×16 tiles, DMA each tile from DDR into VTCM, compute the partial GEMM in HVX with all data hot in VTCM, then DMA results back. Use double-buffering: while computing on tile N, DMA-load tile N+1 in parallel, hiding DDR latency. This is the fundamental technique for any HVX kernel with working sets larger than VTCM.',
              },
              {
                question: 'Q12 (.falign): What does the .falign directive accomplish?',
                answer: 'Correct answer: (b) Aligns loop entry to packet boundaries for better performance. The Hexagon ISA fetches 128-bit (16-byte) packets, and packet decode happens at packet-aligned addresses. If a loop entry point is unaligned, the first packet may be split across two fetches, costing an extra cycle. .falign (or .p2align 3 for 8-byte alignment) ensures the loop entry sits at a clean packet boundary, saving startup cycles.',
              },
              {
                question: 'Q13 (Initiation Interval): In a software-pipelined loop with initiation interval (II) of 2, what does this mean?',
                answer: 'Correct answer: (b) A new iteration starts every 2 cycles. II measures the steady-state throughput of a pipelined loop. II=2 means the loop body has been scheduled so that a fresh iteration begins every 2 cycles, with multiple iterations in flight simultaneously. For a loop body that takes 6 cycles end-to-end with II=2, you have ~3 iterations overlapped. This gives 3× the throughput of the unpipelined version. Achieving II=1 (one iteration per cycle) is the holy grail and requires sufficient registers and zero data dependencies that span the iteration boundary.',
              },
              {
                question: 'Q14 (Inline Asm): When should you prefer inline assembly over intrinsics?',
                answer: 'Correct answer: (b) Only when necessary (missing intrinsics, micro-optimizations). HVX intrinsics are the recommended path: they\'re safer (the compiler verifies correctness), they get auto-scheduled for optimal packet packing, they\'re portable across Hexagon versions, and they\'re easier to debug. Inline assembly is useful only when (1) an intrinsic doesn\'t exist for an operation you need, (2) you need precise control over packet construction that the compiler isn\'t producing, or (3) you\'re writing the absolute hottest inner loop where every cycle matters.',
              },
              {
                question: 'Q15 (Essay): Explain the data flow for vshuff and vdeal operations. Why are these important for convolution kernels?',
                answer: 'vshuff interleaves elements from two source vectors into a single destination, producing patterns like [A0, B0, A1, B1, ...] from inputs A and B. vdeal does the inverse — it de-interleaves a single vector into two outputs, separating elements by stride. Importance for convolution: (1) Input data often arrives in column-major or channel-interleaved format, but HVX kernels prefer row-major, channel-grouped layouts for efficient sequential MACs. Shuffles reorder data without going through memory. (2) Transposition: vshuff/vdeal can construct matrix transposes via multiple invocations. (3) Cross-lane communication: when reducing partial sums across lanes (e.g., the final stage of a dot product), shuffles enable lane-to-lane communication that pure arithmetic can\'t. (4) Cache locality: pre-shuffling data into the format the inner kernel expects avoids repeated unaligned loads or scalar fallback.',
              },
              {
                question: 'Q16 (Essay): A convolution kernel processes 16×16 tiles with 16-bit intermediate accumulation. You observe that throughput is only 50% of theoretical peak. What are three likely bottlenecks?',
                answer: 'Three likely bottlenecks: (1) Register pressure: with 16×16 = 256 output elements at 16 bits each, you need many vector registers just for accumulators. With only 32 HVX registers total (V0-V31), you may have zero registers left for inputs/weights, forcing spills to VTCM or memory. Solution: reduce tile size or accumulate to wider width and pack later. (2) Data dependencies (MAC chains): if each MAC depends on the previous one, the pipeline can\'t overlap iterations. The result is II > 1 because of read-after-write hazards. Solution: use multiple independent accumulators (V0-V3 instead of just V0) to hide the latency. (3) Memory bandwidth: if input/weight loads aren\'t prefetched and the working set exceeds L1, you\'re waiting on DDR. Solution: stage data into VTCM via DMA before the inner loop starts. (4) Packet utilization: not filling all 4 instruction slots in each 128-bit packet wastes issue bandwidth.',
              },
              {
                question: 'Q17 (Essay): Contrast the GEMM tiling strategy with the convolution tiling strategy. Why are they different?',
                answer: 'GEMM tiling: Block A (M×K), B (K×N), C (M×N) into VTCM-sized chunks. The K dimension (inner reduction dimension) is typically the largest, so the inner loop iterates over K. A_tile and B_tile are loaded into VTCM, the partial product accumulates into C_tile in registers/VTCM, and after the K-loop finishes, C_tile is committed back to memory. Convolution tiling: Tile the spatial domain (H × W) into chunks, with the small kernel (3×3, 5×5) and weights staying resident in VTCM or registers. A typical conv tile is 18×18 input → 16×16 output (with 1-pixel padding). The kernel is reused across all spatial positions in the tile, providing high arithmetic intensity through spatial locality. Why different: GEMM has no spatial reuse — each output element needs a fresh dot product over the K dimension, so reuse comes from blocking the K loop. Convolution has spatial reuse — overlapping receptive fields share many input pixels, so reuse comes from caching input and weights and walking the output grid.',
              },
            ]} />
          </div>

          {/* ============ References ============ */}
          <div id="references">
            <ReferenceList references={[
              {
                authors: 'Qualcomm Technologies',
                year: 2024,
                title: 'Qualcomm Hexagon V60 Programmer\'s Reference Manual',
                venue: 'Official Qualcomm Documentation',
              },
              {
                authors: 'Qualcomm Technologies',
                year: 2024,
                title: 'Hexagon Vector Extensions (HVX) Programmer\'s Guide',
                venue: 'Official Qualcomm Documentation',
              },
              {
                authors: 'Qualcomm Technologies',
                year: 2024,
                title: 'Qualcomm Snapdragon Neural Processing Engine (NPE) SDK',
                venue: 'Official Qualcomm SDK',
              },
              {
                authors: 'Sze, V., Chen, Y. H., Yang, T. J., & Emer, J. S.',
                year: 2017,
                title: 'Efficient Processing of Deep Neural Networks: A Tutorial and Survey',
                venue: 'Proceedings of the IEEE, vol. 105, no. 12',
              },
              {
                authors: 'Howard, A. G., et al.',
                year: 2017,
                title: 'MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications',
                venue: 'arXiv preprint arXiv:1704.04861',
              },
            ]} />
          </div>

          {/* ============ Appendix ============ */}
          <TheorySection id="appendix" title="Appendix: Quick Reference">
            <SubSection id="register-cheat-sheet" title="Register Configuration Cheat Sheet">
              <CodeBlock language="text" title="HVX Register Configurations">
{`HVX-128 Mode:
  Registers: V0–V31 (32 total)
  Width: 128 bytes per register
  Predicates: Q0–Q3 (128 bits each)

HVX-256 Mode (if enabled):
  Registers: V0–V31 (same register file)
  Width: 256 bytes per register
  (Register pairs form 512-byte operations)`}
              </CodeBlock>
            </SubSection>

            <SubSection id="lane-naming" title="Lane Width and Intrinsic Naming">
              <CodeBlock language="text" title="Intrinsic Naming Convention">
{`Lane Width    Intrinsics            Example
─────────────────────────────────────────────
8-bit         Q6_V_op_VbVb         Q6_V_vmpy_VhVb (h=16-bit, b=8-bit)
16-bit        Q6_V_op_VhVh         Q6_Vh_vmpyacc_VhVb
32-bit        Q6_V_op_VwVw         Q6_V_vmpy_VhVb (result is 32-bit)
64-bit (pair) Q6_Ww_op_WwWw        Q6_Ww_vrmpyacc_WwVhVb`}
              </CodeBlock>
            </SubSection>

            <SubSection id="optimization-patterns" title="Common Optimization Patterns">
              <CodeBlock language="text" title="Common HVX Patterns">
{`Pattern: Reduce to scalar sum
────────────────────────────
for (int lane = 0; lane < 32; lane++) {
    sum += vec[lane];
}
// Cannot be done in HVX alone; need scalar fallback

Pattern: Dot product (MAC chain)
─────────────────────────────────
for (i = 0; i < 64; i++) {
    accum = vmpyacc(accum, coeff[i], data[i]);
}

Pattern: Convolution (2D sliding window)
────────────────────────────────────────
for (y = 0; y < height; y++) {
    for (x = 0; x < width; x++) {
        for (ky = 0; ky < 3; ky++) {
            row = input[y+ky];
            for (kx = 0; kx < 3; kx++) {
                accum += row[x+kx] * kernel[ky][kx];
            }
        }
        output[y][x] = requantize(accum);
    }
}`}
              </CodeBlock>
            </SubSection>
          </TheorySection>

          {/* ============ Navigation ============ */}
          <div className="mt-12 pt-6 border-t border-border-primary flex items-center justify-between">
            <Link
              href="/qualcomm/module-01"
              className="flex items-center gap-2 text-sm font-medium text-accent-blue hover:gap-3 transition-all"
            >
              <ArrowLeft className="w-4 h-4" /> Prev: Hexagon SoC Architecture
            </Link>
            <Link
              href="/qualcomm/module-03"
              className="flex items-center gap-2 text-sm font-medium text-accent-blue hover:gap-3 transition-all"
            >
              Next: HTA/HMX Tensor Accelerators <ArrowRight className="w-4 h-4" />
            </Link>
          </div>
        </article>

        <TableOfContents items={tocItems} />
      </div>
    </>
  )
}
