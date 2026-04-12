# Module 2: HVX (Hexagon Vector eXtensions) Programming Model

**Level**: PhD / Expert
**Target Audience**: Advanced embedded systems engineers, ML accelerator specialists
**Estimated Reading Time**: 4-6 hours
**Prerequisites**: Module 1 (Hexagon ISA Fundamentals), understanding of SIMD concepts, integer arithmetic

---

## Table of Contents

1. [Introduction](#introduction)
2. [HVX Register File Architecture](#hvx-register-file-architecture)
3. [HVX Data Types and Lane Semantics](#hvx-data-types-and-lane-semantics)
4. [Core HVX Intrinsic Families](#core-hvx-intrinsic-families)
5. [Writing Efficient Convolution Kernels](#writing-efficient-convolution-kernels)
6. [Vectorizing Depthwise Convolution](#vectorizing-depthwise-convolution)
7. [Pointwise Convolution and GEMM for INT8](#pointwise-convolution-and-gemm-for-int8)
8. [Predicate Registers and Masked Operations](#predicate-registers-and-masked-operations)
9. [Loop Alignment and Software Pipelining](#loop-alignment-and-software-pipelining)
10. [HVX Intrinsics vs Inline Assembly](#hvx-intrinsics-vs-inline-assembly)
11. [Self-Assessment Questions](#self-assessment-questions)
12. [References](#references)

---

## Introduction

The Hexagon Vector eXtensions (HVX) represent Qualcomm's primary mechanism for achieving high throughput on the Hexagon DSP core. Unlike the scalar instruction set covered in Module 1, HVX operates on wide vectors—either 128 bytes (HVX-128 mode, standard for Snapdragon) or 256 bytes (HVX-256 mode, when enabled)—in a single instruction.

HVX's architecture was specifically designed for mobile ML inference workloads:

- **Lane-based parallelism**: Vectors are conceptually divided into "lanes," each operating on a data element
- **Data-parallel execution**: All lanes execute the same operation simultaneously
- **Flexible data widths**: Operations adapt to 8-bit, 16-bit, 32-bit, and 64-bit element sizes
- **VTCM integration**: Direct access to Vector TCM (Tightly Coupled Memory) enables high-bandwidth data movement

This module provides deep technical coverage of HVX programming, from low-level register architecture to high-level optimization patterns for neural network inference.

> **Scope Note**: This module focuses on HVX-128 mode (128 bytes = 1024 bits per register) as the primary target, with notes on HVX-256 scaling where relevant.

---

## HVX Register File Architecture

### 2.1 Overview of the HVX Register File

The HVX vector register file is fundamentally different from scalar registers. Rather than holding a single 32-bit or 64-bit value, each vector register holds a **full vector** of data elements.

#### Register Configuration in HVX-128 Mode

```
┌─────────────────────────────────────────────────────────────┐
│                      V0 (128 bytes)                         │
├─────────────────────────────────────────────────────────────┤
│ Byte   │ Byte   │ Byte   │ ...      │ Byte  │ Byte  │ Byte  │
│  [0]   │  [1]   │  [2]   │          │ [125] │ [126] │ [127] │
│        ← 16 × 8-bit elements OR →                            │
│        ← 8 × 16-bit elements OR →                            │
│        ← 4 × 32-bit elements OR →                            │
│        ← 2 × 64-bit elements →                               │
└─────────────────────────────────────────────────────────────┘
```

**Key architectural facts:**

- **32 vector registers**: V0–V31 in HVX-128 mode
- **Register width**: 128 bytes (1024 bits) per register
- **Addressable as pairs**: (V0:1), (V2:3), ..., (V30:31) for 256-byte operations
- **No partial register access**: You cannot directly access a sub-register (e.g., the lower 64 bytes of V0); instructions consume/produce full registers

> **Expert Insight**: The HVX register file is fundamentally **wide but not deep**—32 registers of 128 bytes each total only 4 KB. This is why careful register allocation and judicious use of VTCM are critical for large workloads.

### 2.2 Register Pairs and Accumulation

When working with wider accumulation widths, HVX uses **register pairs**:

```
┌──────────────────────┐
│   V1:0 (256 bytes)   │
├──────────────────────┤
│   High half: V1      │ (128 bytes)
│   Low half: V0       │ (128 bytes)
├──────────────────────┤
```

Register pairs are always **even:odd** (V0:1, V2:3, V4:5, ..., V30:31). This restriction is a hardware constraint to maintain symmetry in the vector pipeline.

#### When to Use Register Pairs

1. **32-bit accumulation from 16-bit multiplies**: `vrmpyacc(Vdd32, Vss32, Rtt)` requires a register pair to hold the accumulated 32-bit results
2. **64-bit intermediate values**: Some reduction operations require wider intermediate results
3. **Double-width output from multiply**: 16×16→32-bit multiply produces a register pair

### 2.3 Predicate Register File (Q Registers)

In addition to vector registers, HVX provides a small **predicate register file** for conditional operations:

```
┌─────────────────────────────────────┐
│  Q0 (Predicate Register, 128 bits)  │
├─────────────────────────────────────┤
│ Bit 0 │ Bit 1 │ ... │ Bit 127       │
│ (Lane 0) (Lane 1) ... (Lane 127)    │
│  for 8-bit mode ───────────────────│
│ Bit 0 │ Bit 1 │ ... │ Bit 63        │
│ (Lane 0) (Lane 1) ... (Lane 63)     │
│  for 16-bit mode ──────────────────│
└─────────────────────────────────────┘
```

**Q Register Set**:

- **4 predicate registers**: Q0, Q1, Q2, Q3
- **Width**: 128 bits, subdivided by lane width (16 lanes for 8-bit, 8 lanes for 16-bit, etc.)
- **Operations**: Each bit corresponds to a lane-level condition

**Typical usage pattern**:

```c
// Generate a predicate from a comparison
Q0 = vmpyh(V0, V1)  // Compare operation (result: predicate)
// Use predicate in masked operation
V2 = Q0 ? V3 : V4   // Conditional select: if Q0[i], V2[i] = V3[i], else V4[i]
```

> **Expert Insight**: Predicates are **not** regular conditional masks like AVX-512 in x86. They are specifically tied to vector-level comparisons and logical operations. This design choice reflects the Hexagon's focus on data-parallel neural network workloads where branch divergence is rare.

### 2.4 Register Allocation Strategy

The 32-register limit is tight for complex kernels. Typical strategies:

**Strategy 1: Blocking with VTCM**
```
Registers used:
  - V0–V3:    Input tile (128 bytes × 4 = 512 bytes useful data)
  - V4–V7:    Weights (or filter coefficients)
  - V8–V15:   Accumulation registers (8 for partial sums)
  - V16–V31:  Scratch/temporal storage

Reserve V30:31 for callee-save requirements (ABI)
```

**Strategy 2: Temporal Reuse**
```
In a loop over batches:
  Load input chunk    → V0:V3
  Load weights        → V4:V7 (reused across loop iterations)
  Accumulate results  → V8:V15 (updated in-place)
  Store output        → (from V8:V15)
  (Shuffle/permute ops use V16–V29 as temporaries)
```

---

## HVX Data Types and Lane Semantics

### 2.5 Understanding Lanes and Data Width Interpretation

The critical insight in HVX programming is that **the same register holds different numbers of elements depending on the operation data width**:

#### Lane Width Mapping

```
Register V0 (128 bytes = 1024 bits):

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
└──────────────────────────────────────────┘
```

This **dynamic lane interpretation** is why HVX is so efficient for quantized ML: the same vector hardware executes 8-bit, 16-bit, or 32-bit operations without format conversion.

### 2.6 Signed vs. Unsigned Operations

HVX distinguishes between signed and unsigned operations through **intrinsic naming conventions**:

```c
// 8-bit multiply-accumulate: signed × signed → 16-bit
// Intrinsic: Q6_Vh_vmpyacc_VhVb (signed)
Vdd16_result = Q6_Vh_vmpyacc_VhVb(Vdd16_accum, Vs_coeff, Vu_input);

// vs.

// 8-bit multiply-accumulate: unsigned × unsigned → 16-bit
// Intrinsic: Q6_Vuh_vmpyacc_VuhVub (unsigned)
Vdd16_result = Q6_Vuh_vmpyacc_VuhVub(Vdd16_accum, Vus_coeff, Vuu_input);
```

**Naming convention**:
- Lower-case 'h' → signed half (16-bit result from 8-bit input)
- Uppercase 'UH' → unsigned half

### 2.7 Widening and Narrowing Operations

Real-world quantized inference requires **widening** (expanding to prevent overflow) and **narrowing** (quantizing back down).

#### Widening: 8-bit → 16-bit

```c
// Load 8-bit signed bytes into a vector register
V0 = Q6_V_vmem_R(R0);  // 128 × int8

// Widen to 16-bit using vunpacke/vunpacko
// vunpacke: Extract even-indexed lanes
V1 = Q6_Vh_vunpacke_Vb(V0);  // V1[0] = sign_extend(V0[0]), V1[1] = sign_extend(V0[2]), ...

// vunpacko: Extract odd-indexed lanes
V2 = Q6_Vh_vunpacko_Vb(V0);  // V2[0] = sign_extend(V0[1]), V2[1] = sign_extend(V0[3]), ...
```

**Data movement diagram**:
```
V0 (128 bytes, 128 × 8-bit):
[b0][b1][b2][b3]...[b127]

After vunpacke (even lanes):
V1 (64 × 16-bit):
[b0 ext][b2 ext][b4 ext]...[b126 ext]  (sign-extended to 16-bit)

After vunpacko (odd lanes):
V2 (64 × 16-bit):
[b1 ext][b3 ext][b5 ext]...[b127 ext]  (sign-extended to 16-bit)
```

#### Narrowing: 32-bit → 8-bit (with Saturation)

```c
// After accumulation, we have 32-bit results in register pair Vdd32
// Narrow and saturate to 8-bit using vpacke/vpacko + clip

// Extract even lanes (32-bit → 16-bit), then saturate
V0_even = Q6_Vh_vpacke_Vw(Vdd32);  // Pack even lanes of 32-bit to 16-bit

// Extract odd lanes (32-bit → 16-bit)
V0_odd = Q6_Vh_vpacko_Vw(Vdd32);   // Pack odd lanes of 32-bit to 16-bit

// Combine and saturate to 8-bit range
V0_result = Q6_Vb_vpacke_Vh(V0_even);  // Pack even lanes of 16-bit to 8-bit with saturation
```

> **Expert Insight**: Quantization in HVX requires careful orchestration of widening and narrowing operations. The key is to **widen early** (to prevent overflow during accumulation) and **narrow late** (to minimize intermediate register pressure).

### 2.8 Data Type Quantization Reasoning

For INT8 quantized neural networks:

```
Formula: quantized_value = round(fp32_value / scale) + zero_point

Inverse: fp32_value = (quantized_value - zero_point) × scale
```

When implementing in HVX:

1. **Input multiplication**: `int8_input × int8_weight` → requires 16-bit intermediate (signed)
2. **Accumulation**: Sum of products → requires 32-bit intermediate
3. **Requantization**: `(accum >> shift) + zero_point` → 32-bit → 8-bit (with saturation)

---

## Core HVX Intrinsic Families

### 2.9 Load and Store Operations

#### 2.9.1 Aligned Vector Load: Q6_V_vmem_R

**Signature**:
```c
HVX_Vector Q6_V_vmem_R(const void *Rs)
```

**Semantics**: Load a full vector (128 bytes in HVX-128 mode) from a 128-byte-aligned address.

**Constraints**:
- Address must be 128-byte aligned (bits [6:0] = 0)
- If alignment is not met, behavior is undefined

**Example**:
```c
// Load 128 bytes of int8 data from memory
int8_t input_buffer[VECTOR_SIZE];  // Size: 128
// Ensure 128-byte alignment (use compiler attributes or manual allocation)
__attribute__((aligned(128))) int8_t aligned_input[128];

HVX_Vector V0 = Q6_V_vmem_R((const void *)aligned_input);
```

**Performance**:
- **Latency**: 1 cycle (assuming cache hit)
- **Throughput**: Can issue multiple loads per cycle (dual-issue on modern cores)
- **Bandwidth**: 128 bytes per cycle (HVX-128), 256 bytes per cycle (HVX-256)

#### 2.9.2 Unaligned Vector Load: Q6_V_vmemu_R

**Signature**:
```c
HVX_Vector Q6_V_vmemu_R(const void *Rs)
```

**Semantics**: Load a full vector from an **arbitrary** address (not necessarily aligned).

**Mechanism**: The hardware performs 2-3 sub-vector loads and combines them:
```
If address has lower 7 bits = offset (not aligned):
  Load from (address & ~127)       → temp_low
  Load from (address + 128)        → temp_high
  Shift/align and combine          → result_vector
```

**Performance**:
- **Latency**: 2–3 cycles (due to alignment resolution)
- **Throughput**: Half that of aligned loads
- **Penalty**: ~3× cost vs. aligned load

**Usage**:
```c
// Load from potentially unaligned address
const int8_t *unaligned_ptr = input + some_offset;
HVX_Vector V_data = Q6_V_vmemu_R((const void *)unaligned_ptr);
```

> **Expert Insight**: For input buffers where alignment is difficult to guarantee, `vmemu` is necessary, but the performance cost is significant. In tight inner loops, align input data when possible to use `vmem`.

#### 2.9.3 Vector Gather: Q6_V_vgather_AsR

**Signature**:
```c
HVX_Vector Q6_V_vgather_AsR(int As, const void *Rs)
```

Where `As` is a 32-bit immediate offset (scaled).

**Semantics**: Load scattered elements from an array indexed by offsets.

**Example: Gather every 4th element**:
```c
// Input: array of int32
int32_t sparse_array[256];

// Gather with stride 4 (every 4th element)
// Offset immediate: 4 (scale = 4 bytes per element)
HVX_Vector V0 = Q6_V_vgather_AsR(4, (const void *)sparse_array);
// V0[0] = sparse_array[0]
// V0[1] = sparse_array[4]  (stride of 4)
// V0[2] = sparse_array[8]
// ...
```

**Constraints**:
- Offset must be a compile-time constant
- Can only use fixed strides (1, 2, 4, 8 bytes)

**Performance**: ~5 cycles per gather (much slower than sequential loads)

#### 2.9.4 Vector Scatter: Q6_V_vscatter_AsRV

**Signature**:
```c
void Q6_V_vscatter_AsRV(int As, void *Rt, HVX_Vector Vs)
```

**Semantics**: Write scattered elements to an array with a fixed stride.

**Example**:
```c
int32_t output_array[256];
HVX_Vector V_results = ...;  // Results from computation

// Scatter with stride 4
Q6_V_vscatter_AsRV(4, (void *)output_array, V_results);
// output_array[0] = V_results[0]
// output_array[4] = V_results[1]
// output_array[8] = V_results[2]
// ...
```

**Performance**: ~5 cycles per scatter

> **Design Note**: Gather and scatter are expensive. In well-optimized neural network kernels, avoid their use in hot loops. Instead, **rearrange data in VTCM** to enable sequential access patterns.

### 2.10 Arithmetic Operations

#### 2.10.1 Vector Multiply: Q6_V_vmpy_VhVb

**Signature**:
```c
HVX_Vector Q6_V_vmpy_VhVb(HVX_Vector Vt, HVX_Vector Rs)
```

Multiply signed bytes by signed half-words → signed 16-bit results

**Naming breakdown**:
- `V` (output): HVX_Vector register
- `vmpy`: multiply operation
- `VhVb`: Vh (signed half-word operand) × Vb (signed byte operand)

**Data flow diagram**:
```
Vt (16-bit signed): [h0 ][h1 ][h2 ]...[h63 ]
Rs (8-bit signed):  [b0][b1][b2]...[b127]

After vmpy (lane-by-lane):
Output (16-bit signed):
[h0×b0][h1×b1][h2×b2]...[h63×b127]  (128 lanes → 64 lanes due to width doubling)

Wait, this isn't quite right. Let me correct:
Actually, vmpy duplicates/broadcasts across lanes.
```

Actually, let me reconsider the semantics. For `Q6_V_vmpy_VhVb`:

```
Vt: 64 × 16-bit signed values [h0, h1, ..., h63]
Rs: 128 × 8-bit signed values [b0, b1, ..., b127]

Multiply operation (element-wise pairing):
  h0 × b0 = 16-bit result
  h1 × b1 = 16-bit result
  ...
  h63 × b63 = 16-bit result

Output: 64 × 16-bit signed results
```

**Common usage in convolution**:
```c
// Input: bytes (8-bit), Weights: half-words (16-bit)
HVX_Vector V_input_bytes = Q6_V_vmem_R(input_ptr);     // 128 × int8
HVX_Vector V_weights_hw = Q6_V_vmem_R(weights_ptr);   // 64 × int16

// Multiply
HVX_Vector V_products = Q6_V_vmpy_VhVb(V_weights_hw, V_input_bytes);
```

#### 2.10.2 Multiply-Accumulate: Q6_Vh_vmpyacc_VhVb

**Signature**:
```c
HVX_Vector Q6_Vh_vmpyacc_VhVb(HVX_Vector Vd, HVX_Vector Vt, HVX_Vector Rs)
```

**Semantics**: `Vd += Vt × Rs` (multiply-accumulate in a single instruction)

**Data flow**:
```
Vd (input accumulator):  [a0    ][a1    ]...[a63    ]
Vt (weights):            [w0    ][w1    ]...[w63    ]
Rs (inputs):             [x0][x1][x2][x3]...[x127]

After vmpyacc:
Vd[0] += w0 × x0
Vd[1] += w1 × x1
...
Vd[63] += w63 × x63

New Vd:
                         [a0+w0×x0][a1+w1×x1]...[a63+w63×x63]
```

**Accumulation pattern in a loop**:
```c
HVX_Vector V_accum = Q6_V_vsplat_R(0);  // Initialize to 0

for (int i = 0; i < num_iterations; i++) {
    HVX_Vector V_weights = Q6_V_vmem_R(weights_ptr + i * VECTOR_SIZE);
    HVX_Vector V_input = Q6_V_vmem_R(input_ptr + i * VECTOR_SIZE);

    // Accumulate products
    V_accum = Q6_Vh_vmpyacc_VhVb(V_accum, V_weights, V_input);
}

// Result in V_accum after loop
```

#### 2.10.3 Dual Multiply: Q6_V_vdmpy_VhVb

**Signature**:
```c
HVX_Vector Q6_V_vdmpy_VhVb(HVX_Vector Vt, HVX_Vector Rs)
```

**Semantics**: Execute two multiplies in parallel (useful for complex numbers or dual-channel operations).

**Example: Complex number multiplication**:
```
Input: [Re0][Im0][Re1][Im1]...[Re31][Im31]  (32 × complex, 16-bit each)
Weight: [Wr][Wi][Wr][Wi]...[Wr][Wi]         (32 × complex weight)

Output:
  [Re0×Wr - Im0×Wi][Re0×Wi + Im0×Wr]...    (Complex results)
```

> **Expert Insight**: `vdmpy` is specialized for FFT, OFDM, and complex-valued signal processing. For neural networks, `vmpy` and `vmpyacc` are the primary workhorses.

#### 2.10.4 Reduction Multiply-Accumulate: Q6_Ww_vrmpyacc_WwVhVb

**Signature**:
```c
HVX_VectorPair Q6_Ww_vrmpyacc_WwVhVb(HVX_VectorPair Vdd, HVX_Vector Vt, HVX_Vector Rs)
```

Where `Ww` denotes a register pair holding 32-bit results.

**Semantics**: Reduce multiply-accumulate—multiply 64 pairs, sum pairs together → 32 accumulation results.

**Data flow**:
```
Vt (weights): 64 × 16-bit [w0, w1, w2, ..., w63]
Rs (input):   128 × 8-bit  [x0, x1, x2, ..., x127]

Pairing:
  Pair 0: w0×x0 + w1×x1
  Pair 1: w2×x2 + w3×x3
  ...
  Pair 31: w62×x62 + w63×x63

Output (Vdd as register pair): 32 × 32-bit results
```

**Critical for dot-product-heavy operations**:
```c
HVX_VectorPair Vdd_accum = { Q6_V_vsplat_R(0), Q6_V_vsplat_R(0) };

for (int i = 0; i < num_iter; i++) {
    HVX_Vector V_weights = Q6_V_vmem_R(weights + i * 64);      // 64 × int16
    HVX_Vector V_inputs = Q6_V_vmem_R(inputs + i * 128);       // 128 × int8

    // Reduction MAC: produces 32 × 32-bit sums
    Vdd_accum = Q6_Ww_vrmpyacc_WwVhVb(Vdd_accum, V_weights, V_inputs);
}

// Extract results from register pair
HVX_Vector V_low = Vdd_accum.lo;   // First 16 results
HVX_Vector V_high = Vdd_accum.hi;  // Next 16 results
```

#### 2.10.5 Multiply-Accumulate Across: Q6_V_vmpa_VVV

**Signature**:
```c
HVX_Vector Q6_V_vmpa_VVV(HVX_Vector Vd, HVX_Vector Vt, HVX_Vector Rs)
```

**Semantics**: A variant that interprets the accumulation differently—useful for depthwise operations.

---

### 2.11 Shuffle and Permute Operations

Shuffle and permute operations rearrange elements within a vector. They are essential for:
- Input reordering to match weight layout
- Transposition for matrix operations
- Lane-level communication in reduction operations

#### 2.11.1 Vector Shuffle: Q6_V_vshuff_VVR

**Signature**:
```c
HVX_Vector Q6_V_vshuff_VVR(HVX_Vector Vt, HVX_Vector Rs, int Rtt)
```

**Semantics**: Selectively interleave or shuffle elements from two vectors based on a shift amount.

**Example: Interleave odd/even**:
```c
HVX_Vector V0 = {0, 2, 4, 6, 8, ...};   // Even elements
HVX_Vector V1 = {1, 3, 5, 7, 9, ...};   // Odd elements

// Shuffle with shift=1 interleaves: [0,1,2,3,4,5,6,7,...]
HVX_Vector V_interleaved = Q6_V_vshuff_VVR(V1, V0, 1);
```

**Detailed operation**:
```
Vt: [t0, t1, t2, t3, ..., t127]
Rs: [r0, r1, r2, r3, ..., r127]
Rtt: Shift amount (typically 0-127, but semantics depend on data width)

After vshuff with Rtt:
Output: Interleaves elements from both inputs with offset
```

#### 2.11.2 Vector Deal: Q6_V_vdeal_VVR

**Signature**:
```c
HVX_Vector Q6_V_vdeal_VVR(HVX_Vector Vt, HVX_Vector Rs, int Rtt)
```

**Semantics**: Inverse of shuffle—"deals" elements from two vectors into separate positions.

**Example: De-interleave**:
```
Input: [0, 1, 2, 3, 4, 5, 6, 7, ...]

After vdeal:
  V0 (even): [0, 2, 4, 6, ...]
  V1 (odd):  [1, 3, 5, 7, ...]
```

#### 2.11.3 Vector Rotate: Q6_V_vror_VRI

**Signature**:
```c
HVX_Vector Q6_V_vror_VRI(HVX_Vector Vt, int Rs)
```

**Semantics**: Rotate elements circularly by Rs positions.

**Example**:
```c
HVX_Vector V = {0, 1, 2, 3, 4, 5, ..., 127};

// Rotate right by 1 lane (at current lane width)
HVX_Vector V_rot = Q6_V_vror_VRI(V, 1);
// In 8-bit mode: {127, 0, 1, 2, ..., 126}
// In 16-bit mode: {63, 0, 1, 2, ..., 62}
```

**Performance**: 1 cycle (very fast for lane-level communication)

#### 2.11.4 Vector Align: Q6_V_vlalign_VVI

**Signature**:
```c
HVX_Vector Q6_V_vlalign_VVI(HVX_Vector Vt, HVX_Vector Rs, int Rii)
```

**Semantics**: Align (shift/rotate) two vectors to extract a contiguous sub-vector.

**Use case: Sliding window in convolution**:
```
V_prev: [... last 16 bytes of previous tile]
V_curr: [first 112 bytes of current tile ...]

vlalign with appropriate shift:
Result: [last few of V_prev | first few of V_curr]
       = [16 bytes of valid sliding window]
```

**Data movement**:
```
V_prev: [v0  v1  v2  ... v127]
V_curr: [u0  u1  u2  ... u127]

vlalign(V_prev, V_curr, shift=112):
Result: [v112 v113 ... v127 u0 u1 ... u15]
        └─── 16 bytes from V_prev ───┘└─ 112 bytes from V_curr ─┘
```

#### 2.11.5 Pack Even/Odd: Q6_Vb_vpacke_Vh / Q6_Vb_vpacko_Vh

**Signature**:
```c
HVX_Vector Q6_Vb_vpacke_Vh(HVX_Vector Vt);  // Pack even lanes of 16-bit to 8-bit
HVX_Vector Q6_Vb_vpacko_Vh(HVX_Vector Vt);  // Pack odd lanes of 16-bit to 8-bit
```

**Semantics**: Narrow from 16-bit to 8-bit by extracting even or odd lanes with saturation.

**Example**:
```
Vt (64 × 16-bit): [100, 200, 300, 400, ..., 8000, 8100]

vpacke (even lanes):  [100, 300, ..., 8000]  → saturate to int8
vpacko (odd lanes):   [200, 400, ..., 8100]  → saturate to int8
```

**Saturation semantics**:
- Values > 127 saturate to 127
- Values < -128 saturate to -128

---

### 2.12 Reduction Operations

#### 2.12.1 Horizontal Add: Q6_Ww_vadd_WwWw

Horizontal addition sums elements across a vector.

**Pattern: Tree reduction**:
```
Input V0 (8 × 32-bit): [a0, a1, a2, a3, a4, a5, a6, a7]

Step 1: Add adjacent pairs
  [a0+a1, a2+a3, a4+a5, a6+a7]

Step 2: Add adjacent pairs again
  [a0+a1+a2+a3, a4+a5+a6+a7]

Step 3: Final sum
  [a0+a1+a2+a3+a4+a5+a6+a7]
```

**Intrinsic chain**:
```c
HVX_Vector V_result = Q6_V_vmem_R(data_ptr);  // Load 32 × 32-bit values

// Use reduction ops to sum all 32 values
// This typically requires scalar scalar fallback or multiple reductions
int32_t final_sum = 0;
for (int i = 0; i < 32; i++) {
    final_sum += ((int32_t *)&V_result)[i];  // Extract via pointer (not ideal)
}
```

> **Note**: HVX doesn't have a direct "sum all lanes" intrinsic. Horizontal reduction typically requires:
> 1. Multiple vror + vadd to combine lanes hierarchically, or
> 2. Scalar extraction via pointer cast (slow)

#### 2.12.2 Dot Product via Reduction MAC

The most efficient dot product uses `vrmpyacc`:

```c
int32_t dot_product_hvx(const int8_t *a, const int16_t *w, int length) {
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
}
```

---

## Writing Efficient Convolution Kernels

### 2.13 Convolution Fundamentals in HVX

A 2D convolution operation:
```
output[y][x] = sum over (ky, kx) in kernel:
                    input[y + ky][x + kx] × kernel[ky][kx] + bias
```

For INT8 quantized inference:
- Input: INT8 (range ±128)
- Weights: INT8 (range ±128)
- Accumulation: INT32 (range ±2^31)
- Output: INT8 (after requantization)

### 2.14 Simple 3×3 Conv2D in HVX: INT8

#### 2.14.1 Data Layout for 3×3 Convolution

**Input layout** (CHW format, tile-wise):
```
Input buffer in VTCM (tile: 18×18 input for 16×16 output):
┌─────────────────────────────────────────┐
│ Row 0:  [16 input pixels] [2 padding]   │ (18 bytes)
│ Row 1:  [16 input pixels] [2 padding]   │
│ ...
│ Row 17: [16 input pixels] [2 padding]   │
└─────────────────────────────────────────┘

Reason for 18×18:
  16 output pixels require 16+2 = 18 input pixels (1-pixel border padding)
  16 output rows require 16+2 = 18 input rows
```

**Kernel layout** (9 coefficients for 3×3):
```
Flattened as: [k[0,0], k[0,1], k[0,2], k[1,0], k[1,1], k[1,2], k[2,0], k[2,1], k[2,2]]
```

**Output layout** (16×16 output):
```
16 × 16 pixels, stored row-major
```

#### 2.14.2 Annotated 3×3 Conv Implementation

```c
/**
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
    // Kernel structure: kernel[output_channel][3*3*input_channel]
    // For single output channel, kernel is 9 × input_depth

    // Step 1: Initialize accumulators
    HVX_Vector V_accum[16];  // One accumulator per output row
    for (int i = 0; i < 16; i++) {
        V_accum[i] = Q6_V_vsplat_R(bias[0]);  // Splat bias across lanes
    }

    // Step 2: Convolution loop
    // For each position in the 3×3 kernel:
    for (int ky = 0; ky < 3; ky++) {  // Kernel row
        for (int kx = 0; kx < 3; kx++) {  // Kernel column

            // Step 3: Process 16×16 output region
            for (int out_y = 0; out_y < 16; out_y++) {
                int in_y = out_y + ky;  // Input row (with padding)

                // Load input row
                const int8_t *row_ptr = input_tile + in_y * 18;
                HVX_Vector V_input = Q6_V_vmem_R((const void *)row_ptr);
                // V_input: 128 bytes = 128 × int8 pixels
                // But we only have 18 pixels per row
                // (Reload mechanism: load at kx offset, extract valid lanes)

                // Load kernel coefficient (same for all positions in this iteration)
                int8_t kernel_coeff = kernel[ky * 3 + kx];
                HVX_Vector V_kernel = Q6_V_vsplat_R(kernel_coeff);
                // Splat the kernel coefficient across all lanes

                // Multiply-accumulate
                V_accum[out_y] = Q6_Vh_vmpyacc_VhVb(
                    V_accum[out_y],          // Accumulator (input and output)
                    V_kernel,                // Kernel coefficient (16-bit for mac)
                    V_input                  // Input data (8-bit)
                );
            }
        }
    }

    // Step 4: Requantize from INT32 → INT8
    for (int out_y = 0; out_y < 16; out_y++) {
        // V_accum[out_y] contains 32-bit accumulated results

        // Right-shift for scaling
        HVX_Vector V_shifted = Q6_V_vasrw_VVI(V_accum[out_y], shift);
        // V_shifted: Still 32-bit, but scaled down

        // Add zero point (broadcast across lanes)
        HVX_Vector V_zero = Q6_V_vsplat_R(zero_point);
        HVX_Vector V_with_zp = Q6_V_vaddw_VVV(V_shifted, V_zero);

        // Pack down to 8-bit with saturation
        // This requires unpacking even/odd lanes, then packing
        HVX_Vector V_even = Q6_Vh_vpacke_Vw(V_accum[out_y]);  // Extract even 32-bit → 16-bit
        HVX_Vector V_odd = Q6_Vh_vpacko_Vw(V_accum[out_y]);   // Extract odd 32-bit → 16-bit

        HVX_Vector V_output = Q6_V_vpacke_VhVh(V_even, V_odd);  // Pack 16-bit → 8-bit with saturation

        // Store output row
        Q6_V_vmem_ARI(output_tile + out_y * 16, V_output);
    }
}
```

> **Critical Detail**: The above is a **naive** implementation. A production kernel would:
> 1. Process multiple channels (depth) in parallel
> 2. Batch process output rows (software pipeline)
> 3. Use VTCM for weight caching
> 4. Apply loop unrolling and dual-issue scheduling

#### 2.14.3 Optimized 3×3 Conv with Batching

```c
void conv3x3_int8_optimized_hvx(
    const int8_t *input_tile,      // Pre-tiled in VTCM
    const int8_t *kernel,
    const int32_t *bias,
    int8_t *output_tile,
    int channel_count,
    int shift,
    int zero_point)
{
    // Assumption: We process 4 channels at once
    // and unroll the 3x3 kernel loop to use more registers

    const int CHANNELS_BATCH = 4;

    for (int ch = 0; ch < channel_count; ch += CHANNELS_BATCH) {
        // Initialize accumulators for 4 output channels
        HVX_Vector V_accum[16][CHANNELS_BATCH];
        for (int y = 0; y < 16; y++) {
            for (int c = 0; c < CHANNELS_BATCH; c++) {
                V_accum[y][c] = Q6_V_vsplat_R(bias[ch + c]);
            }
        }

        // Unrolled 3×3 kernel loop
#define CONV_COEFF(ky, kx, c) kernel[(ky)*3*channel_count + (kx)*channel_count + c]

        for (int y = 0; y < 16; y++) {
            // Load all 3 input rows for this output row
            HVX_Vector V_row[3];
            for (int ky = 0; ky < 3; ky++) {
                int in_y = y + ky;
                V_row[ky] = Q6_V_vmem_R((const void *)(input_tile + in_y * 18 * channel_count));
            }

            // 3×3 kernel positions
            for (int ky = 0; ky < 3; ky++) {
                for (int kx = 0; kx < 3; kx++) {
                    // For each channel in this batch
                    for (int c = 0; c < CHANNELS_BATCH; c++) {
                        int8_t coeff = CONV_COEFF(ky, kx, c);

                        // Load and multiply
                        const int8_t *in_ptr = input_tile + (y + ky) * 18 * channel_count + kx * channel_count + c;
                        HVX_Vector V_input = Q6_V_vmem_R((const void *)in_ptr);

                        HVX_Vector V_coeff_vec = Q6_V_vsplat_R(coeff);

                        V_accum[y][c] = Q6_Vh_vmpyacc_VhVb(
                            V_accum[y][c],
                            V_coeff_vec,
                            V_input
                        );
                    }
                }
            }
        }

        // Requantize and store
        for (int y = 0; y < 16; y++) {
            for (int c = 0; c < CHANNELS_BATCH; c++) {
                HVX_Vector V_shifted = Q6_V_vasrw_VVI(V_accum[y][c], shift);
                HVX_Vector V_zp = Q6_V_vsplat_R(zero_point);
                HVX_Vector V_with_zp = Q6_V_vaddw_VVV(V_shifted, V_zp);

                // Saturate and narrow to int8
                HVX_Vector V_output = Q6_Vb_vpacke_Vh(
                    Q6_Vh_vpacke_Vw(V_with_zp)
                );

                Q6_V_vmem_ARI(
                    output_tile + y * 16 + ch + c,
                    V_output
                );
            }
        }
    }
}
```

---

## Vectorizing Depthwise Convolution

### 2.15 Depthwise Convolution Strategy

Depthwise convolution processes each input channel independently:

```
For each input channel c:
  For each spatial position (y, x):
    output[c][y][x] = sum over (ky, kx):
                          input[c][y+ky][x+kx] × kernel[c][ky][kx]
```

Key difference from standard conv:
- Each channel has its own 3×3 kernel
- No cross-channel communication
- More parallelizable across channels

### 2.16 Depthwise 3×3 Implementation

```c
/**
 * Depthwise Convolution 3x3 INT8
 *
 * Key optimization: Process multiple channels in parallel using different vector registers
 */

void depthwise_conv3x3_int8_hvx(
    const int8_t *input,        // Input tensor (H × W × C)
    const int8_t *kernel,       // Kernel (C × 3 × 3)
    const int32_t *bias,        // Bias (C)
    int8_t *output,             // Output (H × W × C)
    int height,
    int width,
    int channels,
    int shift,
    int zero_point)
{
    // Process channels in groups (e.g., 8 at a time)
    const int CHANNELS_BATCH = 8;

    for (int y = 0; y < height - 2; y++) {  // y loop (output)
        for (int x = 0; x < width - 2; x++) {  // x loop (output)

            for (int ch = 0; ch < channels; ch += CHANNELS_BATCH) {
                // Initialize accumulators for this channel batch
                HVX_Vector V_accum[CHANNELS_BATCH];
                for (int c = 0; c < CHANNELS_BATCH; c++) {
                    V_accum[c] = Q6_V_vsplat_R(bias[ch + c]);
                }

                // 3×3 kernel convolution
                for (int ky = 0; ky < 3; ky++) {
                    for (int kx = 0; kx < 3; kx++) {
                        for (int c = 0; c < CHANNELS_BATCH; c++) {
                            int in_y = y + ky;
                            int in_x = x + kx;
                            int in_idx = (in_y * width + in_x) * channels + ch + c;
                            int kernel_idx = (ch + c) * 9 + ky * 3 + kx;

                            int8_t input_val = input[in_idx];
                            int8_t kernel_val = kernel[kernel_idx];

                            // Accumulate
                            // Note: This is scalar—not vectorized (simplified)
                            // For true HVX vectorization, would need to reorganize data
                            // into packed format
                            // (See next section for proper approach)
                        }
                    }
                }

                // Requantize
                for (int c = 0; c < CHANNELS_BATCH; c++) {
                    // ... similar to standard conv
                }
            }
        }
    }
}
```

### 2.17 Efficient Depthwise: Data Reordering

**Problem**: Depthwise conv has poor data locality—each channel's kernel is separate.

**Solution**: Pre-reorder data into **"channel-packed"** format:

```
Original layout (interleaved channels):
[C0, C1, C2, ..., C7, C0, C1, C2, ..., C7, ...]

Channel-packed layout (grouped by channel):
[C0, C0, C0, ..., C0, C1, C1, C1, ..., C1, ..., C7, C7, ..., C7]
│─ 64 values ─│ │─ 64 values ─│         │─ 64 values ─│
```

**Reordered depthwise (uses vshuff/vdeal)**:

```c
void depthwise_conv3x3_hvx_optimized(
    const int8_t *input_packed,    // Pre-reordered into channel-packed format
    const int8_t *kernel_packed,
    const int32_t *bias,
    int8_t *output_packed,
    int num_tiles,
    int shift,
    int zero_point)
{
    // With packed data, we can vectorize across channels
    // Example: Process 8 channels using 8 separate V-registers

    for (int tile = 0; tile < num_tiles; tile++) {
        HVX_Vector V_accum[8];  // One accumulator per channel

        for (int c = 0; c < 8; c++) {
            V_accum[c] = Q6_V_vsplat_R(bias[c]);
        }

        // 3×3 kernel with channel-parallel processing
        for (int ky = 0; ky < 3; ky++) {
            for (int kx = 0; kx < 3; kx++) {
                // Load all channels for this (ky, kx) position in parallel
                int in_offset = tile * (16*16*8) + (ky * 16 + kx) * 64;
                HVX_Vector V_input[8];

                for (int c = 0; c < 8; c++) {
                    V_input[c] = Q6_V_vmem_R((const void *)(input_packed + in_offset + c * 64));
                }

                // Load kernel coefficients
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
            HVX_Vector V_output = Q6_Vb_vsaturate_Vw(V_shifted);  // Saturate and narrow
            Q6_V_vmem_ARI(output_packed + tile * (16*16*8) + c * 128, V_output);
        }
    }
}
```

---

## Pointwise Convolution and GEMM for INT8

### 2.18 Pointwise Convolution (1×1 Conv)

Pointwise convolution is essentially a **matrix multiply**:

```
output[h][w][oc] = sum over ic:
                       input[h][w][ic] × weight[oc][ic]
```

Where:
- Input: H × W × IC
- Weight: OC × IC
- Output: H × W × OC

This becomes a **batch of matrix multiplies** (one per spatial location).

### 2.19 GEMM Implementation Using vrmpyacc

```c
/**
 * General Matrix Multiply for INT8:
 *   C[m×n] = A[m×k] × B[k×n] + bias[n]
 *
 * Parameters:
 *   A: m×k matrix, INT8 (inputs)
 *   B: k×n matrix, INT8 (weights, pre-transposed to k×n)
 *   C: m×n matrix, INT8 (output)
 *   m, k, n: Matrix dimensions
 *   shift, zero_point: Quantization parameters
 */

void gemm_int8_hvx(
    const int8_t *A,       // m × k
    const int8_t *B,       // k × n (pre-transposed)
    int8_t *C,
    const int32_t *bias,   // n elements
    int m, int k, int n,
    int shift, int zero_point)
{
    // Tile size: process 16 input rows at a time
    const int TILE_M = 16;
    const int TILE_N = 16;  // Output columns (align with HVX register width)

    for (int i_tile = 0; i_tile < m; i_tile += TILE_M) {
        for (int j_tile = 0; j_tile < n; j_tile += TILE_N) {
            int tile_m = min(TILE_M, m - i_tile);
            int tile_n = min(TILE_N, n - j_tile);

            // Initialize accumulators for this tile
            HVX_VectorPair Vdd_accum[TILE_M];
            for (int i = 0; i < tile_m; i++) {
                // Splat bias across both halves of the register pair
                Vdd_accum[i].lo = Q6_V_vsplat_R(bias[j_tile]);
                Vdd_accum[i].hi = Q6_V_vsplat_R(bias[j_tile]);
            }

            // Compute A[i_tile:i_tile+TILE_M, :] × B[:, j_tile:j_tile+TILE_N]
            for (int k_idx = 0; k_idx < k; k_idx++) {
                // Load B[k_idx, j_tile:j_tile+TILE_N] (one row of B, multiple columns)
                // B is stored transposed, so this is a contiguous load
                const int8_t *B_row = B + k_idx * n + j_tile;
                HVX_Vector V_B_row = Q6_V_vmem_R((const void *)B_row);  // TILE_N × int8

                // For each input row in this tile
                for (int i = 0; i < tile_m; i++) {
                    int A_idx = (i_tile + i) * k + k_idx;
                    int8_t A_val = A[A_idx];

                    // Splat A value across all output columns
                    HVX_Vector V_A_splat = Q6_V_vsplat_R(A_val);

                    // Multiply A[i, k_idx] × B[k_idx, j_tile:j_tile+TILE_N]
                    HVX_Vector V_product = Q6_V_vmpy_VhVb(V_A_splat, V_B_row);

                    // Accumulate
                    Vdd_accum[i] = Q6_Ww_vadd_WwWw(
                        Vdd_accum[i],
                        {V_product, Q6_V_vsplat_R(0)}  // Convert to register pair
                    );
                }
            }

            // Requantize and store this tile of output
            for (int i = 0; i < tile_m; i++) {
                // Combine both halves of register pair
                HVX_Vector V_result_low = Vdd_accum[i].lo;
                HVX_Vector V_result_high = Vdd_accum[i].hi;

                // Right-shift for requantization
                V_result_low = Q6_V_vasrw_VVI(V_result_low, shift);
                V_result_high = Q6_V_vasrw_VVI(V_result_high, shift);

                // Add zero point
                HVX_Vector V_zp = Q6_V_vsplat_R(zero_point);
                V_result_low = Q6_V_vaddw_VVV(V_result_low, V_zp);
                V_result_high = Q6_V_vaddw_VVV(V_result_high, V_zp);

                // Saturate and narrow to int8
                HVX_Vector V_even = Q6_Vh_vpacke_Vw(V_result_low);
                HVX_Vector V_odd = Q6_Vh_vpacko_Vw(V_result_low);
                HVX_Vector V_output_low = Q6_Vb_vpacke_Vh(V_even);

                V_even = Q6_Vh_vpacke_Vw(V_result_high);
                V_odd = Q6_Vh_vpacko_Vw(V_result_high);
                HVX_Vector V_output_high = Q6_Vb_vpacke_Vh(V_even);

                // Store
                int C_idx = (i_tile + i) * n + j_tile;
                Q6_V_vmem_ARI((void *)(C + C_idx), V_output_low);
                Q6_V_vmem_ARI((void *)(C + C_idx + 16), V_output_high);
            }
        }
    }
}
```

### 2.20 VTCM Tiling for Large GEMMs

When matrices are too large for the register file, use **VTCM blocking**:

```
VTCM Layout (assuming 256 KB VTCM):
┌─────────────────────────────────────┐
│ A_tile (16×128 = 2 KB)              │ (Reloaded each iteration)
├─────────────────────────────────────┤
│ B_tile (128×16 = 2 KB)              │ (Loaded once per block)
├─────────────────────────────────────┤
│ C_tile (16×16 = 256 bytes)          │ (Output accumulator)
├─────────────────────────────────────┤
│ Free space for intermediate buffers  │ (~250 KB)
└─────────────────────────────────────┘
```

**Tiling strategy**:

```c
void gemm_int8_vtcm_tiled(
    const int8_t *A_ddr,     // Full A matrix in DDR
    const int8_t *B_ddr,     // Full B matrix in DDR
    int8_t *C_ddr,
    int m, int k, int n)
{
    // VTCM pointers (allocated via HAP_mem_alloc)
    int8_t *A_tile_vtcm = vtcm_buffer;
    int8_t *B_tile_vtcm = A_tile_vtcm + (16 * 128);
    int32_t *C_tile_vtcm = (int32_t *)(B_tile_vtcm + (128 * 16));

    const int TILE_M = 16;
    const int TILE_K = 128;  // Large K to maximize cache reuse
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

                // DMA: Load A[i:i+TILE_M, kk:kk+TILE_K] to VTCM
                dma_memcpy_ddr_to_vtcm(
                    A_tile_vtcm,
                    A_ddr + i * k + kk,
                    TILE_M * chunk_k
                );

                // DMA: Load B[kk:kk+TILE_K, j:j+TILE_N] to VTCM
                dma_memcpy_ddr_to_vtcm(
                    B_tile_vtcm,
                    B_ddr + kk * n + j,
                    chunk_k * TILE_N
                );

                // Compute C += A_tile × B_tile (using HVX)
                gemm_compute_tile_hvx(
                    A_tile_vtcm, B_tile_vtcm, C_tile_vtcm,
                    TILE_M, chunk_k, TILE_N
                );
            }

            // DMA: Store C[i:i+TILE_M, j:j+TILE_N] back to DDR
            dma_memcpy_vtcm_to_ddr(
                C_ddr + i * n + j,
                C_tile_vtcm,
                TILE_M * TILE_N
            );
        }
    }
}
```

### 2.21 Weight Packing Strategies

**Efficient weight formats for GEMM**:

1. **Transposed format**: B stored as (K × N) instead of (N × K)
   - Enables contiguous row loads during MAC loop

2. **Channel-interleaved**: For multi-channel weights
   ```
   Standard: [OC0_IC0, OC0_IC1, ..., OC0_ICk, OC1_IC0, ...]
   Packed:   [OC0_IC0, OC1_IC0, OC2_IC0, ..., OCm_IC0, OC0_IC1, ...]
   ```

3. **Preconditioned scales**: Pre-multiply weights by quantization scale

---

## Predicate Registers and Masked Operations

### 2.22 Q Register Semantics

Predicate registers (Q0–Q3) enable **data-dependent execution** within vectors.

**Generated by comparison operations**:
```c
// Compare result produces a predicate
HVX_VectorPred Q0 = Q6_Q_vcmp_equ_VbVb(V0, V1);  // Q0[i] = 1 if V0[i] == V1[i]
```

**Bit interpretation by lane width**:
```
For 8-bit lane width (128 lanes):
  Q0 = 128 bits, each bit corresponds to one lane

For 16-bit lane width (64 lanes):
  Q0 = 128 bits, but only lower 64 bits are valid
  (upper 64 bits should be zero or undefined)
```

### 2.23 Masked Operations

**Conditional select**: `result = Q0 ? V_true : V_false`

```c
HVX_Vector V_result = Q6_V_vmux_QVV(Q0, V_true_value, V_false_value);
// For each lane i:
//   result[i] = Q0[i] ? V_true_value[i] : V_false_value[i]
```

**Practical example: Boundary handling in convolution**:

```c
void conv_with_padding_hvx(
    const int8_t *input,
    const int8_t *kernel,
    int8_t *output,
    int height, int width, int channels)
{
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            HVX_Vector V_accum = Q6_V_vsplat_R(0);

            // 3×3 kernel with padding
            for (int ky = -1; ky <= 1; ky++) {
                for (int kx = -1; kx <= 1; kx++) {
                    int in_y = y + ky;
                    int in_x = x + kx;

                    // Check boundary conditions
                    if (in_y >= 0 && in_y < height && in_x >= 0 && in_x < width) {
                        int8_t input_val = input[in_y * width + in_x];
                        int8_t kernel_val = kernel[(ky+1)*3 + (kx+1)];
                        V_accum += input_val * kernel_val;  // Simplified
                    } else {
                        // Padding: use zero (already in V_accum initialization)
                    }
                }
            }

            output[y * width + x] = saturate(V_accum >> SHIFT);
        }
    }
}
```

**Vectorized boundary handling with predicates**:

```c
void conv_boundary_vectorized_hvx(
    const int8_t *input_tile,   // 18×18
    const int8_t *kernel,
    int8_t *output_tile,        // 16×16
    int shift)
{
    HVX_Vector V_zero = Q6_V_vsplat_R(0);
    HVX_Vector V_padding_mask = Q6_V_vsplat_R(-1);  // All bits set

    // Create a mask for valid positions (interior 16×16 of 18×18 input)
    // Predicate Q0: 1 for valid positions, 0 for padding
    HVX_VectorPred Q_valid = Q6_Q_vsetq_R(16 * 128);  // Set first 16×128 bits

    // Compute convolution
    HVX_Vector V_accum = Q6_V_vsplat_R(0);

    for (int ky = 0; ky < 3; ky++) {
        for (int kx = 0; kx < 3; kx++) {
            // Load input row
            int in_y = ky;  // Input row offset
            HVX_Vector V_input = Q6_V_vmem_R((const void *)(input_tile + in_y * 18));

            // Apply padding mask
            HVX_Vector V_padded = Q6_V_vmux_QVV(Q_valid, V_input, V_zero);

            // Multiply-accumulate
            int8_t kernel_coeff = kernel[ky * 3 + kx];
            HVX_Vector V_kernel = Q6_V_vsplat_R(kernel_coeff);
            V_accum = Q6_Vh_vmpyacc_VhVb(V_accum, V_kernel, V_padded);
        }
    }

    // Store result
    HVX_Vector V_output = Q6_Vb_vsaturate_Vw(
        Q6_V_vasrw_VVI(V_accum, shift)
    );
    Q6_V_vmem_ARI(output_tile, V_output);
}
```

---

## Loop Alignment and Software Pipelining

### 2.24 The ".falign" Directive

The Hexagon processor has a **packet-based instruction encoding**. Multiple instructions can be grouped into a **packet** (up to 4 instructions in most Snapdragon generations) and execute in parallel.

**Problem**: Loop entry and exit don't naturally align to packet boundaries.

**Solution**: Use `.falign` (function align) to align loop entry to a 8-byte boundary:

```gas
    .p2align 3                    // Align to 8 bytes (2^3)

.loop_start:
    {
        V0 = vmem(R0++#1)       ; Instruction 1
        V1 = vmem(R1++#1)       ; Instruction 2
        R2 = add(R2, #1)        ; Instruction 3
        if (P0) jump .loop_start ; Instruction 4
    }
```

### 2.25 Loop Prologue and Epilogue

**Problem**: A pipelined loop requires multiple iterations to "fill" the pipeline before producing useful results.

**Solution**: Separate into prologue, kernel, and epilogue:

```
Prologue:
  Issue iteration 1, 2, ..., n (pipeline fill)

Kernel:
  Issue iteration n+1, n+2, ... (steady state)
  Data from iteration 1 is now ready

Epilogue:
  Drain remaining iterations from pipeline
  (Issue new iterations without completing old ones)
```

**Example: Software-pipelined load-mac loop**:

```c
/**
 * Software pipeline for dot product:
 *   sum = V0[0] * V1[0] + V0[1] * V1[1] + ...
 *
 * Pipeline stages:
 *   Stage 0: Load data
 *   Stage 1: Multiply
 *   Stage 2: Accumulate
 */

int32_t dot_product_pipelined(const int8_t *a, const int16_t *w, int length) {
    HVX_Vector V_accum = Q6_V_vsplat_R(0);

    // Prologue: Load first iteration
    HVX_Vector V_data_prev = Q6_V_vmem_R((const void *)a);

    // Main loop with pipeline
    int i = 0;
    for (i = 1; i < (length / 64) - 1; i++) {
        // Load next data (stage 0)
        HVX_Vector V_data_curr = Q6_V_vmem_R((const void *)(a + i * 64));

        // Load weights and process previous iteration (stages 1-2)
        HVX_Vector V_weights = Q6_V_vmem_R((const void *)(w + i * 64));
        V_accum = Q6_Ww_vrmpyacc_WwVhVb(
            {V_accum, Q6_V_vsplat_R(0)},
            V_weights,
            V_data_prev
        ).lo;

        // Move current to previous for next iteration
        V_data_prev = V_data_curr;
    }

    // Epilogue: Process final iteration
    HVX_Vector V_weights_final = Q6_V_vmem_R((const void *)(w + i * 64));
    V_accum = Q6_Ww_vrmpyacc_WwVhVb(
        {V_accum, Q6_V_vsplat_R(0)},
        V_weights_final,
        V_data_prev
    ).lo;

    // Extract final sum from V_accum
    int32_t *results = (int32_t *)&V_accum;
    int32_t sum = 0;
    for (int j = 0; j < 32; j++) sum += results[j];

    return sum;
}
```

### 2.26 Compiler Auto-Pipelining

The Hexagon compiler (HexagonTools) can **automatically detect and pipeline loops** under certain conditions:

**Conditions for auto-pipelining**:
1. Loop has a predictable trip count
2. Instruction dependencies allow for overlapping
3. No function calls inside the loop
4. Minimal register pressure

**Compiler hint**: Use `#pragma` to suggest pipelining:

```c
#pragma omp parallel for
#pragma omp critical
for (int i = 0; i < N; i += 64) {
    HVX_Vector V_in = Q6_V_vmem_R((const void *)(input + i));
    HVX_Vector V_out = some_operation(V_in);
    Q6_V_vmem_ARI((void *)(output + i), V_out);
}
```

### 2.27 Initiation Interval (II) and Throughput

**Initiation Interval (II)**: Cycles between issuing successive iterations.

**Example**:
```
Unpipelined loop:
  Iteration 1: Load (1 cycle) → MAC (4 cycles) → Store (1 cycle) = 6 cycles
  Iteration 2 starts after iteration 1 completes
  Throughput: 6 cycles per iteration

Pipelined loop (II = 2):
  Cycle 0: Load iter1, MAC iter2, Store iter3
  Cycle 1: Load iter2, MAC iter3, Store iter4
  Cycle 2: Load iter3, MAC iter4, Store iter5
  ...
  Throughput: 2 cycles per iteration (3× faster!)
```

**Achieving II=1**: Requires careful scheduling and sufficient registers.

---

## HVX Intrinsics vs Inline Assembly

### 2.28 When to Use Each

**HVX Intrinsics (Recommended)**:
- Correctness guaranteed by compiler
- Auto-scheduling for optimal instruction parallelism
- Easier to maintain and debug
- Portable across Hexagon versions (mostly)

**Inline Assembly**:
- When intrinsics don't exist for a specific operation
- Tight inner loops where micro-optimizations matter
- Fine-grained control over instruction scheduling
- Risk of introducing bugs (compiler won't help)

### 2.29 Inline Asm Syntax for Hexagon

```c
/**
 * Inline assembly for Hexagon
 *
 * Syntax:
 *   asm("asm_code"
 *       : output_operands
 *       : input_operands
 *       : clobbered_registers);
 */

// Example: Custom multiply-accumulate with different semantics
HVX_Vector custom_mac_asm(
    HVX_Vector V_accum,
    HVX_Vector V_weights,
    HVX_Vector V_input)
{
    HVX_Vector result;

    asm("{\n\t"
        "%0 = vmpyacc(%1, %2, %3)  // Accumulate multiply\n\t"
        "}"
        : "=r" (result)                  // Output
        : "r" (V_accum), "r" (V_weights), "r" (V_input)  // Inputs
        : );  // No clobbered registers

    return result;
}
```

### 2.30 Packet Construction in Inline Asm

Hexagon's parallel packet syntax in inline asm:

```c
void parallel_ops_asm(
    HVX_Vector *V_in,
    HVX_Vector *V_out,
    int *scalar_val)
{
    int tmp;
    HVX_Vector v_temp;

    asm("{\n\t"
        "%0 = vmem(%3)                 ; Load instruction 1\n\t"
        "%1 = vadd(%0, %4)             ; Add instruction 2 (parallel)\n\t"
        "%2 = add(%4, #1)              ; Scalar add instruction 3 (parallel)\n\t"
        "}\n\t"
        "{\n\t"
        "vmem(%3) = %1                 ; Store instruction 4\n\t"
        "}"
        : "=r" (v_temp), "=r" (*V_out), "=r" (tmp)
        : "r" (V_in), "r" (*scalar_val)
        : );
}
```

**Packet rules**:
1. Each `{}` encloses one packet (up to 4 instructions)
2. Instructions in the same packet execute in parallel
3. No data dependencies within a packet (compiler enforces)
4. Comments within asm strings must use `//` or `;`

### 2.31 Performance Monitoring

Use the **Hexagon PMU (Performance Monitoring Unit)** to measure:

```c
#include <hexagon_protos.h>

void measure_kernel_performance() {
    uint64_t cycles_start, cycles_end;
    uint64_t instrs_start, instrs_end;

    // Start counters
    __asm("mrc p10, 0, %0, c9, c13, 0" : "=r" (cycles_start));
    __asm("mrc p10, 0, %0, c9, c12, 1" : "=r" (instrs_start));

    // Kernel execution
    kernel_function();

    // Stop counters
    __asm("mrc p10, 0, %0, c9, c13, 0" : "=r" (cycles_end));
    __asm("mrc p10, 0, %0, c9, c12, 1" : "=r" (instrs_end));

    uint64_t total_cycles = cycles_end - cycles_start;
    uint64_t total_instrs = instrs_end - instrs_start;

    printf("Cycles: %llu, Instructions: %llu, CPI: %f\n",
           total_cycles, total_instrs, (double)total_cycles / total_instrs);
}
```

---

## Self-Assessment Questions

After studying this module, test your understanding:

### Section 2.1–2.4: Register File

**Q1**: You have an 128-byte vector register (HVX-128 mode) containing 16-bit signed integers. How many lanes does this register have?
- a) 16
- b) 32
- c) 64 ← **Correct**
- d) 128

**Q2**: When using register pairs (e.g., V0:1) for a 32-bit multiply-accumulate, which registers are valid pairs?
- a) V0:1, V1:2, V2:3, ...
- b) V0:1, V2:3, V4:5, ..., V30:31 ← **Correct**
- c) Any two adjacent registers
- d) V0:1, V2:3, V4:5, V6:7, V8:9, ... (all)

**Q3**: What is the purpose of predicate registers (Q0–Q3)?
- a) To hold vector data
- b) To enable conditional/masked operations ← **Correct**
- c) To store loop counters
- d) To cache kernel coefficients

### Section 2.5–2.8: Data Types

**Q4**: In INT8 quantized inference, what is the typical accumulation data width needed to prevent overflow from 8-bit multiply?
- a) 8-bit
- b) 16-bit (intermediate)
- c) 32-bit (for sums of products) ← **Correct**
- d) 64-bit

**Q5**: What do the intrinsics `vunpacke` and `vunpacko` do?
- a) Unpack a compressed vector
- b) Extract even and odd lanes (widening from 8-bit to 16-bit) ← **Correct**
- c) Unload and pack elements
- d) Unroll a packed register

### Section 2.9–2.12: Intrinsics

**Q6**: Which load intrinsic is fastest for aligned data access?
- a) Q6_V_vmem_R (aligned) ← **Correct**
- b) Q6_V_vmemu_R (unaligned)
- c) Q6_V_vgather_AsR
- d) They are all the same speed

**Q7**: What does `Q6_Ww_vrmpyacc_WwVhVb` do differently from `Q6_Vh_vmpyacc_VhVb`?
- a) It operates on unsigned values instead of signed
- b) It produces 32-bit outputs instead of 16-bit, with reduction across lanes ← **Correct**
- c) It performs double precision multiply
- d) It is interchangeable; no difference

**Q8**: In a convolution kernel, you need to narrow 32-bit accumulated results back to 8-bit INT8. Which sequence is correct?
- a) Use `vpacke` directly on 32-bit → 8-bit
- b) Use `vpacke` (32→16), then `vpacko` (16→8)
- c) Use `vpacke` (32→16) and `vpacko` (32→16), then combine and pack to 8-bit ← **Correct**
- d) Use `vpacks` (one instruction, 32→8)

### Section 2.13–2.20: Convolution & GEMM

**Q9**: For a 3×3 convolution kernel with INT8 inputs and weights, what is the minimum number of vector registers needed to store the 9 kernel coefficients?
- a) 1 (if replicated using vshuff)
- b) 9 (one per coefficient)
- c) Depends on the implementation; could be 1 if using splat ← **Correct**
- d) 18 (due to register pairing requirements)

**Q10**: In a depthwise convolution (vs. standard convolution), what is the key advantage?
- a) Fewer multiply operations total
- b) Each channel is processed independently—better data locality ← **Correct**
- c) No accumulation needed
- d) Always faster regardless of implementation

**Q11**: For a 1000×1000 GEMM in DDR memory (too large for VTCM), what is the primary HVX optimization technique?
- a) Use scatter/gather for all accesses
- b) Tile into VTCM-sized blocks and pipeline loads/computes ← **Correct**
- c) Process one element at a time with scalar operations
- d) Allocate more vector registers (not possible)

### Section 2.21–2.27: Pipelining & Assembly

**Q12**: What does the `.falign` directive accomplish?
- a) Aligns data buffers to cache line boundaries
- b) Aligns loop entry to packet boundaries for better performance ← **Correct**
- c) Aligns vector registers to 64-byte boundaries
- d) Aligns output data for efficient storage

**Q13**: In a software-pipelined loop with initiation interval (II) of 2, what does this mean?
- a) Two new iterations start every cycle
- b) A new iteration starts every 2 cycles ← **Correct**
- c) The loop has 2 instructions in parallel
- d) The loop completes in 2 cycles total

**Q14**: When should you prefer inline assembly over intrinsics?
- a) Always (assembly is faster)
- b) Only when necessary (missing intrinsics, micro-optimizations) ← **Correct**
- c) For clarity (assembly is more readable)
- d) Never (HVX compiler handles it optimally)

### Essay Questions

**Q15**: Explain the data flow for `vshuff` and `vdeal` operations. Why are these important for convolution kernels?

**Sketch Answer**:
- `vshuff`: Interleaves elements from two vectors (or shuffles within a vector)
- `vdeal`: Inverse operation—separates interleaved elements
- **Importance for convolution**:
  - Input data often arrives in column-major or channel-interleaved format
  - Shuffles reorder data to row-major or grouped-by-channel for efficient MAC
  - Reduces memory access latency and improves cache locality

**Q16**: A convolution kernel processes 16×16 tiles with 16-bit intermediate accumulation. You observe that throughput is only 50% of theoretical peak. What are three likely bottlenecks?

**Sketch Answer**:
1. **Register pressure**: Spilling accumulators to VTCM/DDR (slow)
2. **Data dependencies**: MAC chains that block pipeline (need II > 1)
3. **Bandwidth**: Waiting for data loads (DDR latency if not pre-fetching)
4. **Packet utilization**: Not filling all 4 instruction slots in each packet

**Q17**: Contrast the GEMM tiling strategy with the convolution tiling strategy. Why are they different?

**Sketch Answer**:
- **GEMM tiling**: Block A (M×K), B (K×N), C (M×N) into VTCM-sized chunks. Typically K is the large dimension (inner product dimension), so we iterate K-loop multiple times.
- **Convolution tiling**: Tile input spatially (H×W tile), weights stay in VTCM. Typical tile: 18×18 input for 16×16 output (includes padding).
- **Difference**: GEMM benefits from large K dimension (many MACs per load). Conv benefits from spatial locality (3×3 kernel reuse across output positions).

---

## References

### Official Hexagon Documentation

1. **Qualcomm Hexagon V60 Programmer's Reference Manual**
   - ISA specification for vector instructions
   - https://developer.qualcomm.com/hexagon

2. **Hexagon Vector Extensions (HVX) Programmers Guide**
   - Comprehensive intrinsic reference
   - Register allocation guidelines
   - Performance optimization tips

3. **Qualcomm Snapdragon Neural Processing Engine (NPE) SDK**
   - Framework for deploying optimized models
   - Example kernels for common ops (Conv2D, GEMM, etc.)

### Key Intrinsic References

- `Q6_V_vmem_R`: Aligned vector load (128 bytes)
- `Q6_V_vmemu_R`: Unaligned vector load
- `Q6_V_vmpy_VhVb`: Signed byte multiply to signed half-word
- `Q6_Vh_vmpyacc_VhVb`: Signed byte multiply-accumulate
- `Q6_Ww_vrmpyacc_WwVhVb`: Reduction multiply-accumulate (64→32 lanes)
- `Q6_V_vshuff_VVR`, `Q6_V_vdeal_VVR`: Shuffle and deal
- `Q6_Vb_vpacke_Vh`, `Q6_Vb_vpacko_Vh`: Pack with saturation
- `Q6_Vh_vunpacke_Vb`, `Q6_Vh_vunpacko_Vb`: Unpack (widen)

### Related Papers & Resources

1. **"Optimizing Deep Neural Networks for Mobile Devices" (2016)**
   - Techniques for quantized inference
   - Discusses lane-based parallelism

2. **Hexagon SDK Sample Code**
   - Github: https://github.com/qualcomm/qualcomm-hexagon-hlo
   - Reference implementations of Conv2D, depthwise, pointwise convolutions

3. **VTCM Memory Model**
   - 256 KB per core (subject to version)
   - Bandwidth: 64 bytes/cycle (HVX-128)
   - Usage: Cache weights, tile inputs, accumulate outputs

### Performance Tuning

- **Hexagon Profiler**: Built into QEMU and real devices
  - Measure cycle count, memory stalls, instruction utilization
  - Command: `hexagon-sim --profile` (in emulation)

- **Compiler Flags**:
  - `-O3 -mvectorize`: Enable HVX auto-vectorization
  - `-fno-inline`: Simplify profiling
  - `-g`: Debug symbols for gdb integration

---

## Appendix: Quick Reference

### Register Configuration Cheat Sheet

```
HVX-128 Mode:
  Registers: V0–V31 (32 total)
  Width: 128 bytes per register
  Predicates: Q0–Q3 (128 bits each)

HVX-256 Mode (if enabled):
  Registers: V0–V31 (same register file)
  Width: 256 bytes per register
  (Register pairs form 512-byte operations)
```

### Lane Width and Intrinsic Naming

```
Lane Width    Intrinsics            Example
─────────────────────────────────────────────
8-bit         Q6_V_op_VbVb         Q6_V_vmpy_VhVb (h=16-bit, b=8-bit)
16-bit        Q6_V_op_VhVh         Q6_Vh_vmpyacc_VhVb
32-bit        Q6_V_op_VwVw         Q6_V_vmpy_VhVb (result is 32-bit)
64-bit (pair) Q6_Ww_op_WwWw        Q6_Ww_vrmpyacc_WwVhVb
```

### Common Optimization Patterns

```
Pattern: Reduce to scalar sum
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
}
```

---

**End of Module 2**

---

## Acknowledgments

This module synthesizes content from:
- Qualcomm Hexagon ISA documentation
- Snapdragon neural processing research
- Mobile ML optimization literature
- Community contributions to open-source Hexagon tools

For questions, corrections, or additional topics, contact the curriculum maintainers.

