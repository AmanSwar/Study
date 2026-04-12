# MODULE 21: Edge AI Hardware Landscape

## Abstract

The proliferation of machine learning inference on edge devices—from datacenter-class mobile processors to microcontroller units with kilobytes of memory—requires deep understanding of the heterogeneous hardware landscape that enables deployment. This module surveys the complete taxonomy of edge processors, from high-performance application processors (smartphone SoCs) through specialized AI accelerators and down to ultra-constrained microcontroller units. We examine architectural design decisions, ISA extensions, memory hierarchies, and power characteristics that fundamentally constrain algorithm design and model optimization on edge. Special attention is given to ARM's dominance in mobile (Cortex-A/M lineages, NEON/SVE extensions, Mali GPU hierarchies), proprietary neural processing units (Qualcomm Hexagon, Google TPU Lite), and emerging alternatives (RISC-V). Understanding these hardware realities is prerequisite to all subsequent edge inference work.

---

## 1. Introduction: The Hardware Constraint Landscape

Edge AI's defining characteristic is **severe heterogeneity across compute, memory, and power dimensions**. Unlike datacenter deployment where a batch of identical GPUs processes thousands of samples, edge devices exhibit:

- **Compute disparity**: From 8 TOPS (Google Coral TPU) to 0.1 GOPS (ARM Cortex-M7)—a 80,000× range
- **Memory gap**: From 12 GB LPDDR5 (flagship phone) to 256 KB SRAM (MCU)—a 50,000× range
- **Power budget**: From 5W sustained (flagship phone) to 1 mW (batteryless wearable)—a 5,000× range

These constraints are **not simply scaled-down versions of datacenter systems**. They require fundamentally different algorithms:

- **Quantization necessity**: INT8/INT4 not for latency but for memory footprint
- **Sparse computation**: Networks structured for streaming inference, not batch processing
- **Memory-aware scheduling**: Operator fusion and buffer reuse become first-class concerns
- **Thermal coordination**: Peak performance achievable only when thermals permit

This module builds the hardware foundation required to reason about these constraints quantitatively.

### 1.1 Taxonomy Dimensions

Edge processors are categorized along three orthogonal axes:

| Dimension | Range | Impact |
|-----------|-------|--------|
| **Peak Compute** | 0.1 GOPS – 100 TOPS | Determines which models/quantizations feasible |
| **Peak Memory Bandwidth** | 1 GB/s – 500 GB/s | Dictates arithmetic intensity requirements |
| **Power Envelope** | 1 mW – 5 W | Limits sustained compute vs. burst capability |

Understanding these axes requires distinguishing hardware specifications (theoretical) from achievable throughput (actual), which depend heavily on memory access patterns and thermal state.

---

## 2. Hardware Taxonomy: From Application Processors to MCUs

### 2.1 Smartphone System-on-Chip (SoC)

Modern flagship smartphones integrate 10-20 billion transistors into single chips:

**Apple A17 Pro (2023) / M3 (2024)**:
- CPU: 6-core (2× P-cores at 3.78 GHz, 4× E-cores at 2.25 GHz)
- GPU: 6-core Apple GPU (6-core GPU variant: up to 5 TFLOPS FP32)
- Neural Engine: 16-core (proprietary architecture), 5.7 TFLOPS INT8
- Peak Memory: 128-bit LPDDR5X at 7.5 GB/s
- Power: 5W sustained, 10W burst (ML workload)

**Qualcomm Snapdragon 8 Gen 3 (2024)**:
- CPU: Octa-core (1× Cortex-X4 3.4 GHz, 3× Cortex-A720 3.1 GHz, 4× Cortex-A520 2.3 GHz)
- GPU: Adreno 850 (GFLOPS depends on frequency, up to ~3-4 TFLOPS for ML)
- Hexagon NPU: 8-core with HVX (8 lanes × 128-bit), HTA, HMX support
- Peak Memory: 256-bit LPDDR5X
- Power: 4-6W for mixed workload

The **heterogeneous compute** model is critical: CPU, GPU, and dedicated NPU execute in parallel with software coordination.

### 2.2 Tablet/Mid-Range SoCs

**MediaTek Dimensity 9200** (flagship Android tablet):
- CPU: Octa-core ARM
- GPU: Mali-G715 MP11 (more lanes than phones)
- APU (AI Processing Unit): Tri-core with HyperEngine, HyperMatrix support
- Peak Memory: 256-bit LPDDR5X at ~12-13 GB/s
- Power envelope: 3-8W

**Google Tensor (Pixel 8/8a)**:
- 8-core ARM CPU (custom Cortex variant)
- Mali-G715 MP11 GPU
- Tensor Processing Unit (TPU): Custom 2-core design (not disclosed) for Tensor-specific ops
- Peak compute for ML: 2-3 TFLOPS effective
- Key advantage: OS integration for real-time features (speech recognition, translation)

### 2.3 IoT Processors (WiFi/Cellular Edge)

**ARM Cortex-A9/A15 + Dedicated Accelerators**:
- Embedded in WiFi routers, surveillance cameras, industrial gateways
- 1-4 cores at 1-1.5 GHz
- Peak compute: 1-2 GOPS
- Memory: 512 MB – 2 GB DRAM
- Power: 0.5-2W sustained

**NVIDIA Jetson Orin Nano** (2.5W edge GPU compute):
- ARM Cortex-A78AE (8-core) @ 2.0 GHz
- NVIDIA GPU (128 CUDA cores) → 42 TFLOPS FP32 / 335 TFLOPS INT8
- 8 GB LPDDR5 @ 102 GB/s
- **Design philosophy**: Pin-compatible with x86 inference servers, enabling rapid prototyping

**Intel Movidius Myriad X**:
- Dedicated VPU for video processing
- 16 SHAVE cores (custom architecture)
- 1-4 TOPS INT8
- Power: 1W
- Common in edge cameras and AR headsets

### 2.4 Microcontroller Units (MCUs)

The extreme end of edge: **no operating system, no memory protection, bare-metal execution**.

**ARM Cortex-M7 (STM32H745)**:
- Single core @ 480 MHz
- 1.3 MB SRAM (unified)
- Hardware FPU (single-precision) but **no integer SIMD**
- Peak compute: ~960 MFLOPS (scalar FP32), ~200 MMACS for INT8 with CMSIS-NN
- Power: 20-50 mW in active mode, <1 µW sleep
- Cost: $2-5

**Arm Cortex-M55** (emerging, e.g., Renesas RA8 series):
- Single core @ 160 MHz
- NEON-like SIMD unit (MVE: M-Profile Vector Extension)
- 8 lanes × 32-bit for INT8 operations (theoretical 1.28 GOPS INT8)
- Memory: 64-512 KB SRAM
- Power: 3-10 mW active
- Cost: $1-3

**RISC-V MCU (SiFive HiFive1 Rev B)**:
- Single core @ 300 MHz
- RV32I base + M (multiply) + F (single FP)
- NO standard vector extension on most MCUs (RVV not yet mainstream)
- 16 KB SRAM
- Power: <2 mW average
- Emerging alternative to ARM monopoly

---

## 3. ARM Architecture Deep Dive: Cortex-A vs. Cortex-M

### 3.1 Cortex-A: Application-Class Design

**Design intent**: Run rich operating systems (Android, Linux) with memory protection and virtual memory.

**Key features**:
- **MMU (Memory Management Unit)**: Hardware page translation, TLB
- **Virtualization extensions**: For hypervisor/container isolation
- **Large caches**: L1 (32-64 KB) + L2 (256 KB – 4 MB) per cluster
- **Out-of-order execution**: Speculative execution, branch prediction
- **Heterogeneous Design**: Multiple frequency domains (P-cores at 3+ GHz, E-cores at 1-2 GHz)

**Cortex-A78 characteristics** (Snapdragon 8 Gen 1 flagship):
- Out-of-order, 8-wide issue
- 3-way SMT capability (in some variants)
- L1 cache: 64 KB I$, 64 KB D$ per core
- L2 cache: 256-512 KB per core
- 2 MB L3 cache (system-wide)
- **Power**: 3-4W for 4-core cluster at 3 GHz

### 3.2 Cortex-M: Embedded-Class Design

**Design intent**: Smallest code/data footprint, predictable latency, no OS overhead.

**Key differences from Cortex-A**:
- **NO MMU**: Only MPU (Memory Protection Unit) for coarse-grain protection
- **In-order execution**: Simpler pipeline (3-4 stages), no branch prediction (saves power)
- **Small caches**: Cortex-M7 has optional 64 KB I-cache, no D-cache (expensive)
- **Harvard architecture variants**: Some M-series have separate instruction/data buses
- **Interrupt latency**: Deterministic, configurable (M7: 12 cycles), critical for control

**Cortex-M7 timing model for ML**:
- Load (from SRAM): 1 cycle
- Load (from flash): 6-10 cycles (with prefetch buffer)
- Integer MAC: 1 cycle (if both operands ready)
- FP32 MAC: 4 cycles latency
- **No pipelining for scalar FP**: Each FP operation waits

### 3.3 ISA Differences: A64 vs. A32 vs. Thumb2

**ARMv8-A (64-bit)**:
- Default: A64 ISA (64-bit instructions)
- 31 integer registers (X0-X30, SP)
- 32 FP/SIMD registers (V0-V31, 128-bit)
- Fixed-width 32-bit instructions (simpler decoding)

**ARMv7-A (32-bit legacy)**:
- A32 ISA: 32-bit instructions (16 registers)
- Thumb2: 16/32-bit mixed encoding (reduces code size)
- Thumb2 is standard on mobile (smaller binaries)

**ARMv8-M (embedded 32-bit)**:
- Used in Cortex-M55, M85
- 32-bit base with optional vector extensions (MVE)
- Thumb-only execution (8-bit instructions decoded to micro-ops)

**Critical for ML**: Instruction encoding size and decode bandwidth determine achievable compute density.

---

## 4. ARM NEON/SVE/SME Vector Extension Evolution

### 4.1 NEON (ARMv7-A / ARMv8-A)

**Original SIMD extension** (2008+): 128-bit wide, 8-lane INT8 or 4-lane INT32.

**NEON characteristics**:
- Fixed-width 128-bit operations
- Separate register file (32× 128-bit Q-registers, V0-V31 in ARMv8)
- Relatively high latency: INT8 MAC latency ~3-4 cycles, throughput 1/cycle
- **Intrinsics example**:

```c
// INT8 multiply-accumulate: 8 lanes × INT8
int8x8_t a = vld1_s8(ptr_a);
int8x8_t b = vld1_s8(ptr_b);
int16x8_t product = vmull_s8(a, b);  // 8× INT8 → 8× INT16
int32x4_t acc = vaddw_s16(vacc, vget_low_s16(product));
```

- Used extensively in **TFLite Micro** (CMSIS-NN) for INT8 quantized ops
- Widely available on all ARMv8-A phones (universal support)

### 4.2 SVE (Scalable Vector Extension, ARMv8.2+)

**Game-changing for server/mobile**: Scalable Vector Length (from 128 bits to 2048 bits).

**SVE design principles**:
- Vector length discovered at **runtime** (VL)
- Typical mobile SVE: 128-256 bit (same as NEON footprint)
- Server SVE: 512-2048 bit (AWS Graviton2 has 512-bit)
- **Predication**: Partial vector operations without scalar loops

**SVE for mobile inference**:

```c
// Pseudo-code: SVE INT8 convolution kernel (simplified)
while (output_elements_remaining > 0) {
    uint64_t vl = svecount_elements(svbool_t pred);  // Dynamic VL
    svint8_t v_a = svld1_s8(pred, ptr_a);
    svint8_t v_b = svld1_s8(pred, ptr_b);
    svint16_t v_prod = svmul_s16_x(pred, svextw_s16_x(pred, v_a),
                                         svextw_s16_x(pred, v_b));
    // Accumulate with wider precision
    ptr_a += vl;
    ptr_b += vl;
}
```

- **Advantage**: Single binary runs on SVE 128-bit (Cortex-A55) and SVE 512-bit (Graviton)
- **Disadvantage**: Complex compiler support, predication overhead for short loops

### 4.3 SME (Scalable Matrix Extension, ARMv9.2+)

**Specialized for tensor operations**: 2D matrix tiles, systolic-array-like operations.

**SME characteristics**:
- Operates on large tiles (e.g., 256×256 bit for int8 on typical implementations)
- Hardware matrix multiplication in hardware
- **ZA tile register**: Accumulator tile, separate from vector registers

**SME example** (int8 matrix multiply):

```c
// Pseudo-code: SME matrix multiply (conceptual)
// Matrix A: (M × K), Matrix B: (K × N) → Accumulator (M × N)
svbool_t pred = svptrue_b8();
for (int m = 0; m < M; m += tile_m) {
    for (int n = 0; n < N; n += tile_n) {
        // Load tile from accumulator or zero
        for (int k = 0; k < K; k += tile_k) {
            smmla(za, a_tile, b_tile);  // Systolic-like multiply-add
        }
        // Write back result
    }
}
```

- **Current state**: SME is bleeding-edge; most edge inference still uses NEON
- **Future**: Phone SoCs (2025+) expected to include SME for flagship models

### 4.4 Practical Implications for Mobile Inference

| Extension | Typical Lane Width | Latency | Throughput | Adoption |
|-----------|-------------------|---------|-----------|----------|
| NEON | 128-bit (8× INT8) | 3-4 cycles | 1/cycle | 100% (all ARMv8-A) |
| SVE | 128-512 bit (variable) | 3-4 cycles | 1-2/cycle | ~30% (recent Cortex-A) |
| SME | Tile-based (256+ bit²) | 1-2 cycles | 2-4/cycle | <5% (emerging) |

---

## 5. ARM Mali GPU: Tile-Based Deferred Rendering Architecture

### 5.1 Mali Fundamentals

**Mali GPU evolution** (used in Qualcomm, MediaTek, Samsung SoCs):
- Mali-G72 (2016): 12 MP (multiprocessor) variant
- Mali-G710 (2020): Up to 10 MP
- Mali-G715 (2022): Same compute, better efficiency
- Mali-G715 MP11 (flagship Android tablets): 11 multiprocessors

**Key architectural difference from desktop GPUs**:

| Property | Desktop (NVIDIA GeForce) | Mobile (ARM Mali) |
|----------|--------------------------|-------------------|
| Rendering | Immediate mode | Tile-based deferred (TBDR) |
| Tile size | N/A | 16×16 pixels (typical) |
| On-chip storage | No | Yes (tile buffer for all attachments) |
| Memory bandwidth usage | High (multiple passes) | Low (one coherent pass) |

### 5.2 TBDR Pipeline for ML Acceleration

Mali's TBDR design, while optimized for graphics, offers advantages for ML:

**Stage 1: Parameter collection** (CPU feedback):
- Record all draw calls, parameter bindings
- Build command buffer (minimal GPU-side interpretation)

**Stage 2: Tiling pass**:
- Distribute primitives to tile bins (16×16 pixel regions)
- Per-tile work lists

**Stage 3: Rendering pass** (on-GPU):
- Per-tile rendering with minimal bandwidth to main memory
- Working set fits in on-chip SRAM (Mali-G715: ~32 KB per MP)

**For inference** (OpenCL/Vulkan compute):
- TBDR less relevant (compute doesn't rasterize)
- But low-bandwidth main memory access still valuable for conv ops

### 5.3 Compute Capabilities: OpenCL and Vulkan Compute

**Mali compute architecture**:
- Each MP: 2-4 quad-warp units (work groups)
- Per-MP: 64-256 KB local memory (SRAM)
- INT8 throughput: ~2-8 MACS/cycle per MP (aggregate: 16-32 TFLOPS for 8-core GPU on flagship)

**OpenCL kernels for conv2d** (simplified):

```c
__kernel void conv2d_i8(
    __global const char *input,     // NCHW layout
    __global const char *weights,   // OIHW layout
    __global int *output,            // Accumulator in INT32
    int IH, int IW, int OC, int IC, int KH, int KW
) {
    int oc = get_global_id(0);
    int oh = get_global_id(1);
    int ow = get_global_id(2);

    int acc = 0;
    for (int ic = 0; ic < IC; ic++) {
        for (int kh = 0; kh < KH; kh++) {
            for (int kw = 0; kw < KW; kw++) {
                char in_val = input[ic * IH * IW + (oh + kh) * IW + (ow + kw)];
                char w_val = weights[oc * IC * KH * KW + ic * KH * KW + kh * KW + kw];
                acc += (int)in_val * (int)w_val;
            }
        }
    }
    output[oc * (IH - KH + 1) * (IW - KW + 1) + oh * (IW - KW + 1) + ow] = acc;
}
```

**Performance bottleneck**: Memory bandwidth for activations (input is reused, weights less so).

### 5.4 Vulkan Compute for Inference

Modern approach (TFLite GPU delegate, XNNPACK on Mali):

```glsl
#version 450
layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(set = 0, binding = 0) buffer InputBuffer { int8_t data[]; } input_buf;
layout(set = 0, binding = 1) buffer WeightBuffer { int8_t data[]; } weight_buf;
layout(set = 0, binding = 2) buffer OutputBuffer { int32_t data[]; } output_buf;

void main() {
    uint oc = gl_GlobalInvocationID.x;
    uint oh = gl_GlobalInvocationID.y;
    uint ow = gl_GlobalInvocationID.z;

    int sum = 0;
    // Convolution logic
    output_buf.data[oc * output_numel + oh * output_w + ow] = sum;
}
```

**Vulkan advantages**:
- Direct GPU memory access (no copying through OpenGL)
- Multi-threaded command recording
- Synchronization more explicit and tuneable

---

## 6. Qualcomm Hexagon NPU Architecture

### 6.1 Hexagon DSP Evolution

Qualcomm's heterogeneous compute strategy places the **Hexagon DSP** (Digital Signal Processor) as the primary AI accelerator on Snapdragon:

**Timeline**:
- **Hexagon 680** (Snapdragon 821, 2016): 4 vector lanes, modest AI capability
- **Hexagon 685** (Snapdragon 855, 2019): Dedicated HVX (Hexagon Vector eXtension), peak 1.3 TOPS INT8
- **Hexagon 698** (Snapdragon 8 Gen 1, 2021): Dual-issue HVX + HTA (Hexagon Tensor Accelerator), ~5 TOPS INT8
- **Hexagon 786** (Snapdragon 8 Gen 3, 2024): HVX + HTA + HMX (Hexagon Matrix Extensions), up to 8-10 TOPS INT8

### 6.2 HVX (Hexagon Vector eXtension)

**Design**: 128-bit wide vector lanes, with multi-lane parallelism.

**HVX architecture**:
- 128 integer registers (128-bit each)
- 32 vector slots (128-bit each) for local operations
- **Execution**: Single-issue or dual-issue depending on instruction type

**HVX assembly example** (int8 convolution accumulation):

```arm
// Load input activations (128 bits = 16× int8)
V0 = vmem(R0 + #0)          // Load 16 int8 values

// Load weights (128 bits = 16× int8)
V1 = vmem(R1 + #0)          // Load 16 int8 values

// Multiply-accumulate (produces 8× int16 results)
V2 = vmpyubh(V0, V1)        // Unsigned multiply, both directions (lane 0,1 → lane 0 etc.)

// Add to accumulator
V3 += V2

// Increment pointers and loop
R0 += #16
R1 += #16
LOOP(R2, #loop_label)       // Repeat R2 times
```

**Key characteristics**:
- Data-parallel: all 16 lanes execute same instruction (SIMD)
- Memory bandwidth: 256 bits/cycle to local memory (dual 128-bit ports)
- Predication: per-lane masking supported

### 6.3 HTA (Hexagon Tensor Accelerator)

**Specialized hardware** for 2D tensor operations (matrix multiply):

**HTA design**:
- Dedicated multiply-accumulate array
- Operates on tiles (e.g., 8×8 or 16×16 int8 matrices)
- Throughput: 1 complete 8×8 int8 MAC per cycle
- Power efficiency: 10-50× better than HVX for matrix-heavy workloads

**Example use case**: Fully-connected layer in neural network:

```
Input: (B, 1024) int8  →  Weights: (1024, 4096) int8  →  Output: (B, 4096) int32

HTA processes as: 16 batches × 16 output channels per systolic wave
```

**Typical latency**: 8×8 matrix multiply: 16 cycles on HTA, vs. ~64 cycles on HVX

### 6.4 HMX (Hexagon Matrix eXtensions) - Gen 3

Latest generation (Snapdragon 8 Gen 3, 2024) adds **HMX**:
- Even larger systolic tiles (up to 16×16 or 32×32 for int8)
- Streaming matrix multiply support
- Integrated with HTA in hierarchical manner

**Qualcomm claims**: Up to 10 TOPS sustained INT8 with HVX+HTA+HMX combined.

---

## 7. Google Edge TPU and Coral Hardware

### 7.1 Edge TPU Design Philosophy

Google's **Tensor Processing Unit** family includes specialized **Edge TPU** designed for low-power on-device inference:

**Design constraints**:
- Power: <2W sustained
- Latency: <100ms for typical mobile models
- Cost: <$100 for USB/PCI-e variants
- Form factor: 40×40 mm for Coral Dev Board

### 7.2 Architecture: Systolic Arrays

**Systolic array principle**:
- 2D grid of MAC (multiply-accumulate) units
- Data flows in predictable patterns (Washington, 1987 seminal work)
- Minimal arbitration needed (data arrives when expected)

**Google Edge TPU specifications**:
- 128×128 MAC array for INT8 (16,384 MACs)
- Throughput: 16,384 MACs / 2 cycles = **8,192 MACS/cycle = 8 TOPS INT8**
- Memory: 8 MB on-chip SRAM (working set)
- External bandwidth: 64 GB/s via PCIe (Coral Dev Board) or USB 3 (Coral USB Accelerator)

**Data flow** (matrix multiply perspective):

```
Weight matrix A (K×N):  streams in from DRAM
Activation matrix B (B×K):  streams in from DRAM
Accumulator (B×N):  computed in SRAM, streamed back

Systolic computation:
  PE[i,j] = PE[i,j] + input_from_west[i] * input_from_north[j]
  Each PE holds partial sum until full K dimension accumulated
```

### 7.3 Quantization Strategy: 8-bit Asymmetric

Edge TPU primarily targets **INT8 inference** with asymmetric quantization:

**Quantization formula**:
```
x_quantized = clamp(round(x_float / scale) + zero_point, 0, 255)

Scale = (max - min) / 255
zero_point = round(-min / scale)
```

**Why asymmetric?** Activations are typically non-negative (ReLU) → asymmetric range.

**Quantization-aware training (QAT) for Edge TPU**:
```python
# TensorFlow Lite Quantizer (simulated)
quantized_model = tf.lite.TFLiteConverter.from_keras_model(model)
quantized_model.optimizations = [tf.lite.Optimize.DEFAULT]
quantized_model.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS_INT8
]

# Representative data for calibration
def representative_dataset_gen():
    for image in validation_set:
        yield [image.astype(np.float32)]

quantized_model.representative_dataset = representative_dataset_gen
tflite_quant_model = quantized_model.convert()
```

### 7.4 Coral USB Accelerator vs. Dev Board

| Property | USB Accelerator | Dev Board |
|----------|-----------------|-----------|
| Form Factor | USB-A dongle (38×31×15 mm) | SBC (90×60 mm) |
| Host Interface | USB 3.0 | Integrated SoC |
| Memory | 8 MB SRAM (Edge TPU) | 1 GB DRAM (SoC) + 8 MB TPU |
| Peak Bandwidth | ~400 MB/s USB | 64 GB/s local |
| Throughput | 4 TOPS (USB bottleneck) | 8 TOPS |
| Cost | $75 | $150 |
| Use Case | Add-on accelerator | Standalone edge inference |

---

## 8. Memory Architecture and Bandwidth Constraints

### 8.1 Memory Hierarchy Across Device Classes

**Smartphone (Android flagship)**:
```
CPU L1 I$: 64 KB
CPU L1 D$: 64 KB  ───┐
CPU L2 $: 512 KB  ───┼─→ CPU cluster
CPU L3 $: 2 MB  ──┘

GPU L1 $: per-MP, typically 16-32 KB
GPU L2 $: shared, 256 KB - 2 MB (partial unified with CPU)

Main DRAM: 8-12 GB LPDDR5
Bandwidth: 68-102 GB/s
Latency: 30-100 ns
```

**Microcontroller (STM32H745)**:
```
L1 I$: Instruction prefetch buffer (none for M7)
L1 TCM (Tightly Coupled Memory): 128 KB instruction, 128 KB data (fast)

Main SRAM: 1.3 MB (unified)
Bandwidth: minimal contention (single core)
Latency: 2-3 cycles (23 ns @ 480 MHz)

Flash (code+weights): 2 MB
Bandwidth: limited by prefetch buffer, typically 60-100 MB/s
Latency: 6-10 cycles from cold access
```

### 8.2 Bandwidth vs. Arithmetic Intensity

**Roofline model** (Williams et al., 2009) quantifies compute-vs-memory tradeoffs:

For a given kernel:
- Arithmetic Intensity (AI) = FLOPS / bytes_accessed
- Peak Compute = F FLOPS
- Peak Bandwidth = B GB/s = B × 10^9 bytes/s

**Achievable throughput** = min(F, B × AI)

**Example: GEMM on Cortex-M7**:

```
Peak compute: 200 MMACS INT8 (roughly)
Peak bandwidth: ~7.68 GB/s (reading from SRAM at 480 MHz with 128-bit bus)

INT8 GEMM: K dimension, M×N output, A is M×K, B is K×N
  - Reads: K bytes (repeated K times for same weight) + K bytes (repeated N times for same activation)
  - Minimal data reuse if K << M, N

AI = (2 * M * N * K) / (K * M + K * N)  [bytes for A and B reads only]
   = (2 * M * N * K) / K(M + N)
   = (2 * M * N) / (M + N)

For 512×512×512 GEMM: AI ≈ 256 bytes/ops, achieves full 200 MMACS
For 32×32×32 GEMM: AI ≈ 21 bytes/ops, limited by bandwidth (~150 MMACS)
```

### 8.3 Data Locality and Cache Behavior

**L1 cache miss penalty**:
- Cortex-A78: 12-15 cycles (to L2)
- Cortex-M7: 6-10 cycles (to SRAM)
- MCU flash access: 100-200 cycles (including prefetch wait)

**Implications for edge inference**:
1. **Keep model weights in cache**: Quantization (INT8) reduces footprint by 4×, fitting more in L1 (ARM A78 L1: 64 KB → 1024 int8 weights)
2. **Reuse activations**: Layer fusion (conv→relu→pool) avoids repeated DRAM traffic
3. **Streaming computation**: Process activations one tile at a time

---

## 9. Power Characteristics and Thermal Constraints

### 9.1 Power Model: Static + Dynamic

**Total power**:
```
P_total = P_static + P_dynamic
        = V² * I_leak + V² * C * f * activity
```

Where:
- V = supply voltage (typically 0.5-1.0 V for SoCs)
- I_leak = leakage current (grows with temperature)
- C = total capacitance
- f = frequency
- activity = fraction of transistors toggling

**Smartphone power envelope**:
- Idle (screen off): 10-50 mW (leakage + peripherals)
- Active screen: 500 mW - 1 W (display + SoC)
- Peak compute (max frequency): 3-5 W (CPU + GPU simultaneously)
- Sustained compute: 1-2 W (thermal limit kicks in)

### 9.2 Thermal Throttling Mechanism

Modern phones monitor **junction temperature** (T_j) and reduce frequency/voltage if T_j > threshold:

**Typical thermal management**:
- Target temperature: 80°C
- Throttling threshold: 85°C
- Critical shutdown: 90°C
- Hysteresis: 5°C (to avoid oscillation)

**Implication for inference**: Peak compute (8 TOPS) achievable for ~5-10 seconds before thermal throttling reduces to ~3 TOPS sustainable.

### 9.3 MCU Power Budget

**ARM Cortex-M7 at 480 MHz**:
- Active: 30-50 mW
- Sleep: <1 µW

**Thermal note**: MCU almost never throttles (dissipates as heat in die, not significant)

---

## 10. Conclusion and Systems Implications

### 10.1 Hardware-Algorithm Co-design

The extreme heterogeneity of edge hardware demands **algorithm design with hardware awareness**:

1. **For smartphone flagship** (Snapdragon 8 Gen 3, Apple A17):
   - Leverage heterogeneous execution: CPU for control, NPU for inference
   - Quantization to INT8 (memory footprint) and leverage HVX/HTA SIMD
   - Batch size 1 (single image)

2. **For tablet** (MediaTek Dimensity 9200):
   - More memory available (8-12 GB) → batch size 2-8 feasible
   - Mali GPU effective for large batch convolutions
   - Still bound by thermal envelopes (3-5W sustained)

3. **For IoT edge** (Jetson Orin Nano):
   - More power budget (2.5W) enables continuous compute
   - DRAM bandwidth abundant (102 GB/s)
   - FP16 or INT8 quantization depending on model precision needs

4. **For MCU** (STM32H745):
   - Memory hierarchy dominates: fit entire model in SRAM or use flash streaming
   - CMSIS-NN intrinsics essential for any acceleration
   - INT4 quantization often required for large models
   - Latency predictability critical for embedded control

### 10.2 Key Insights

- **Memory footprint, not compute, is often the constraint** on MCU (256 KB SRAM vs. 2-4 MB model)
- **Thermal limits are soft** (throttling) but real: sustained compute 50-70% of peak
- **Quantization is mandatory** (memory + compute) not optional
- **NPU efficiency**: Hexagon, TPU cores 5-10× more energy-efficient than GPU for dense matrix ops
- **Portability challenge**: No single binary runs optimal on MCU, mobile, and edge devices

### 10.3 Design Philosophy

Build edge inference systems **bottom-up from hardware constraints**:
1. Profile target device (compute, memory, power, thermal)
2. Design model architecture (MobileNet/EfficientNet families exist for a reason)
3. Quantize aggressively (INT8 baseline, INT4 for extreme cases)
4. Compile with layout-aware scheduling (tiling, fusion)
5. Measure on-device (simulation ≠ reality for edge)

---

## Further Reading

- Qualcomm: "Snapdragon 8 Gen 3 Summit" (2024) – Hexagon NPU architecture
- ARM: "SVE Specification" – ARMv8.2 scalable vector extension
- Google: "Coral TPU: Edge AI Accelerator" (2019)
- Lin et al.: "MCUNet: Tiny Deep Learning on IoT Devices" (NeurIPS 2020)
- Williams et al.: "Roofline: An Insightful Visual Performance Model for Floating-Point Programs and Multicore Architectures" (CACM 2009)
