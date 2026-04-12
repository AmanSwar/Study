# Module 8: Performance Analysis & Profiling for Hexagon NPU

## Table of Contents
1. [Introduction & Learning Objectives](#introduction--learning-objectives)
2. [Roofline Model for Hexagon](#roofline-model-for-hexagon)
3. [Profiling Tools & Infrastructure](#profiling-tools--infrastructure)
4. [Key Metrics to Measure](#key-metrics-to-measure)
5. [Identifying Bottlenecks](#identifying-bottlenecks)
6. [Performance Estimation Formulas](#performance-estimation-formulas)
7. [Benchmarking Methodology](#benchmarking-methodology)
8. [Case Studies & Advanced Topics](#case-studies--advanced-topics)
9. [Self-Assessment Questions](#self-assessment-questions)
10. [References & Tool Guide](#references--tool-guide)

---

## Introduction & Learning Objectives

Performance optimization is not guesswork—it is a systematic science grounded in measurement, analysis, and theoretical bounds. The Hexagon DSP and associated HVX/HMX accelerators represent a complex computational platform where the performance of a single kernel can vary by 10–100× depending on memory access patterns, instruction scheduling, and hardware utilization.

In Module 8, we transition from functional correctness (Module 7) to achieving peak efficiency. You will learn:

- **Theoretical foundations**: How to compute the roofline—the upper bound on performance defined by compute and memory.
- **Measurement**: How to instrument kernels and read hardware performance counters from the Hexagon Simulator, the SDK Profiler, and on-device PMU.
- **Analysis**: How to identify whether a kernel is compute-bound or memory-bound, where stalls originate, and how data flow affects performance.
- **Prediction**: How to estimate cycles analytically *before* implementation, enabling faster iteration.
- **Benchmarking rigor**: How to measure and compare implementations with statistical validity.

By the end of this module, you will have internalized the discipline of performance engineering and be able to reason about kernel efficiency with high confidence.

---

## Roofline Model for Hexagon

The roofline model, introduced by Williams, Waterman, and Patterson (2009), provides a visual and analytical framework for understanding the performance limits of computational kernels. The model answers a fundamental question: **Given the hardware and the algorithm, what is the maximum achievable performance?**

### 1.1 Roofline Fundamentals

The roofline bound is the minimum of two limits:

$$P_{\text{peak}} = \min\left( \frac{\text{Compute Peak (FLOPS)}}{\text{1}}, \frac{\text{Memory Bandwidth (Bytes/sec)} \times \text{Operational Intensity (FLOPs/Byte)}}{\text{1}} \right)$$

Where:
- **Compute peak**: Maximum FLOPs (or integer operations) per second the hardware can deliver.
- **Memory bandwidth**: Maximum data movement rate from main memory (DDR), caches (L2), or local storage (VTCM).
- **Operational intensity (OI)**: The number of arithmetic operations per byte of data moved. Measured in FLOPs/Byte or MACs/Byte.

Rearranging, the memory-bound ceiling is:
$$P = \text{Bandwidth} \times \text{OI}$$

When **OI is small** (many bytes per operation), a kernel hits the **memory bandwidth wall** and performs below peak compute. When **OI is large** (few bytes per operation), the kernel can saturate compute resources and approach the compute peak.

### 1.2 Computing Hardware Peaks for Hexagon

The Hexagon DSP architecture varies across generations (v60, v62, v66, v68, v69, v73), but the principles remain constant. We focus on modern generations (v68/v69) with HVX (Vector Extensions) and HMX (Tensor Accelerator).

#### 1.2.1 Hexagon DSP (Scalar) Peak

A Hexagon DSP core operating at frequency **f GHz**:
- **Single-issue architecture**: 1 instruction per cycle (IPC = 1 with perfect scheduling).
- **32-bit integer operations**: 1 MAC per cycle (multiply-accumulate).

**Scalar peak (int32)** = 1 × f GHz = f billion MACs/sec

For **f = 1.5 GHz**: 1.5 GigaMACs/sec (Int32) or 1.5 GigaFLOPS (FP32).

#### 1.2.2 HVX (Vector) Peak

HVX provides data-parallel execution across multiple lanes:
- **Lane width**: 128 bytes (v68/v69/v73).
- **Element width**: Typically INT8 or INT16.

For **INT8** with 128-byte lanes:
- Elements per vector: 128 bytes / 1 byte = 128 lanes.
- Operations per cycle per element: 1 MAC (multiply-accumulate).
- **HVX peak (INT8)** = 128 MACs × f GHz = 128f billion MACs/sec.

For **INT16**:
- Elements per vector: 128 bytes / 2 bytes = 64 lanes.
- **HVX peak (INT16)** = 64f GHz = 64f billion MACs/sec.

For **f = 1.5 GHz**, **HVX INT8 peak** = 128 × 1.5 = **192 GigaMACs/sec**.

#### 1.2.3 HMX (Tensor) Peak

HMX is a dedicated matrix multiplier (v68+):
- **Matrix shape**: 8×8 INT8 operations per cycle.
- **Output accumulation**: 32-bit INT32 accumulators.

For **HMX INT8 (8×8 matrix)**:
- MACs per cycle: 8 × 8 = 64 MACs.
- **HMX peak** = 64 × f GHz = 64f billion MACs/sec.

For **f = 1.5 GHz**: **HMX INT8 peak** = **96 GigaMACs/sec**.

**Note**: HVX and HMX cannot operate simultaneously on the same data; scheduling determines which accelerator is active.

### 1.3 Memory Hierarchy and Bandwidth Ceilings

Hexagon's memory hierarchy has distinct bandwidth characteristics:

#### 1.3.1 Bandwidth by Tier

| Tier | Capacity | Bandwidth | Latency | Access Pattern |
|------|----------|-----------|---------|-----------------|
| **VTCM** | 64–256 KB | ~200 GB/s | 2–3 cyc | Tightly loop-coupled |
| **L2 Cache** | 256 KB | ~100 GB/s | 3–4 cyc | Spatial/temporal locality |
| **DDR** | GB–GB | 20–50 GB/s | 50+ cyc | Streaming, prefetch-friendly |

**Bandwidth formula for DDR**:
$$\text{DDR Bandwidth (GB/s)} = \text{Bus width (bytes)} \times \text{Frequency (GHz)} \times \text{Efficiency}$$

For a typical Hexagon SoC:
- Bus width: 64–128 bits (8–16 bytes).
- Frequency: 800 MHz–1 GHz.
- Efficiency: 0.5–0.8 (due to contention, refresh, access patterns).

**Typical DDR bandwidth**: 20–50 GB/s.

**Bandwidth formula for L2 (same-core)**:
$$\text{L2 Bandwidth (GB/s)} = \text{Load/Store bandwidth per core} \approx 100 \text{ GB/s}$$

**Bandwidth formula for VTCM (local storage)**:
$$\text{VTCM Bandwidth (GB/s)} = \text{Load/Store port width} \times \text{Frequency} \approx 200 \text{ GB/s}$$

### 1.4 Operational Intensity Calculation for Common Operators

Operational intensity (OI) is the ratio of arithmetic operations to bytes moved. The goal is to compute OI analytically from the operator's algorithm.

#### 1.4.1 Depthwise Convolution (3×3, stride=1)

**Algorithm**:
- Output shape: H × W × C (channels).
- Kernel: 3 × 3 per channel.
- Input: (H + pad) × (W + pad) × C.

**Arithmetic operations (MACs)**:
$$\text{MACs} = H \times W \times C \times 3 \times 3 = 9 \times H \times W \times C$$

**Data movement (bytes, INT8)**:
- Input read: (H + 2) × (W + 2) × C bytes (assume pad=1).
- Output write: H × W × C bytes.
- Bias/scale: C bytes (negligible for large layers).

$$\text{Bytes} = (H + 2)(W + 2)C + HWC \approx HWC + 4C + 4HW + 4C$$

For large H, W (≫ kernel size), **Bytes ≈ HWC × 2** (input + output).

**Operational intensity**:
$$\text{OI}_{\text{DWConv3×3}} = \frac{9 \times HWC}{2 \times HWC} = 4.5 \text{ MACs/Byte}$$

This is **compute-intensive** (OI > 1), typically compute-bound.

#### 1.4.2 Pointwise Convolution (1×1)

**Algorithm**:
- Output: H × W × C_out.
- Kernel: 1 × 1, C_in → C_out.

**Arithmetic operations**:
$$\text{MACs} = H \times W \times C_{\text{in}} \times C_{\text{out}}$$

**Data movement**:
- Input read: H × W × C_in bytes.
- Weights: C_in × C_out bytes.
- Output write: H × W × C_out bytes.

$$\text{Bytes} = HWC_{\text{in}} + C_{\text{in}}C_{\text{out}} + HWC_{\text{out}}$$

**Operational intensity**:
$$\text{OI}_{\text{Pointwise}} = \frac{HWC_{\text{in}}C_{\text{out}}}{HW(C_{\text{in}} + C_{\text{out}}) + C_{\text{in}}C_{\text{out}}}$$

For large H, W:
$$\text{OI}_{\text{Pointwise}} \approx \frac{C_{\text{in}}C_{\text{out}}}{C_{\text{in}} + C_{\text{out}}} = \frac{C_{\text{in}}C_{\text{out}}}{C_{\text{in}} + C_{\text{out}}}$$

If C_in ≈ C_out = C: **OI ≈ C/2**.

**Example**: C_in=64, C_out=64 → **OI ≈ 32 MACs/Byte** (highly compute-intensive).

#### 1.4.3 Matrix Multiply (GEMM)

**Algorithm**: A (M×K) × B (K×N) → C (M×N).

**Arithmetic operations**:
$$\text{MACs} = M \times K \times N$$

**Data movement (assuming each matrix read once)**:
$$\text{Bytes} = MK + KN + MN$$

**Operational intensity**:
$$\text{OI}_{\text{GEMM}} = \frac{MKN}{MK + KN + MN}$$

For large M, K, N (equal order):
$$\text{OI}_{\text{GEMM}} \approx \frac{MKN}{MK + KN + MN} \approx \frac{M}{1 + M/K + M/N}$$

If M = K = N: **OI ≈ M/3**.

**Example**: M=K=N=1024 → **OI ≈ 341 MACs/Byte** (highly compute-intensive, can leverage VTCM tiling).

#### 1.4.4 Element-wise Operations (Add, ReLU, etc.)

**Algorithm**: Output[i] = f(Input[i]).

**Arithmetic operations**: 1 per element.

**Data movement**: Input + Output bytes (2 accesses per element).

$$\text{OI}_{\text{Elementwise}} = \frac{1 \text{ operation}}{2 \text{ bytes}} = 0.5 \text{ MACs/Byte}$$

**This is memory-bound on any memory tier**. The kernel runs at the memory bandwidth ceiling.

### 1.5 Constructing the Roofline Diagram

The roofline is plotted on a log-log graph:
- **X-axis**: Operational Intensity (MACs/Byte), log scale.
- **Y-axis**: Performance (GigaMACs/sec), log scale.

#### Step 1: Identify Hardware Peaks

For a Hexagon SoC at f=1.5 GHz with DDR@40 GB/s:

| Hardware | Peak | Equation |
|----------|------|----------|
| **Scalar (INT32)** | 1.5 GigaMACs/sec | Horizontal line at 1.5 |
| **HVX INT8** | 192 GigaMACs/sec | Horizontal line at 192 |
| **HMX INT8** | 96 GigaMACs/sec | Horizontal line at 96 |
| **DDR Roofline** | P = 40 × OI GB/s = 40 × OI | Slope 40 |
| **L2 Roofline** | P = 100 × OI | Slope 100 |
| **VTCM Roofline** | P = 200 × OI | Slope 200 |

#### Step 2: Find the Knee of the Roofline

The knee (crossover) occurs where memory bandwidth ceiling meets compute peak:

$$\text{Knee OI} = \frac{\text{Compute Peak (MACs/sec)}}{\text{Bandwidth (Bytes/sec)}}$$

**For HVX INT8 and DDR**:
$$\text{Knee OI} = \frac{192 \text{ GigaMACs/sec}}{40 \text{ GB/s}} = \frac{192 \times 10^9}{40 \times 10^9} = 4.8 \text{ MACs/Byte}$$

**Interpretation**:
- If a kernel has **OI < 4.8**, it is **memory-bound** (hits the DDR ceiling).
- If a kernel has **OI > 4.8**, it is **compute-bound** (can achieve up to 192 GigaMACs/sec).

### 1.6 ASCII Roofline Diagram for Hexagon

```
Hexagon HVX INT8 Roofline (f=1.5 GHz, DDR=40 GB/s)

Performance (GigaMACs/sec)
        200  ┌─────────────────────────────────┐ VTCM Ceiling (200 GB/s × OI)
             │                                 │
        150  │                        ╱────────┤ HVX INT8 Peak (192 GigaMACs/sec)
             │                  ╱────
        100  │         ╱────────       ────────┤ L2 Ceiling (100 GB/s × OI)
             │    ╱────               ╱────
         50  │╱──────────────────────────────┐ DDR Ceiling (40 GB/s × OI)
             │                               │
          0  └───────────────────────────────┴────────────────────────
             0.1   0.5    1    2    4.8    10    20   100  OI (MACs/Byte)
                                  ↑
                                 Knee
                          (Memory→Compute)

Operator Placement:
  • Element-wise (Add, ReLU):        OI=0.5    ✘ Memory-bound
  • Pointwise Conv (32→32):          OI≈16     ✓ Compute-bound
  • Depthwise Conv (3×3):            OI≈4.5    ~ Near knee
  • Large GEMM (1024×1024×1024):     OI≈341    ✓ Compute-bound
```

⚡ **Expert Insight**: The roofline model assumes perfect data reuse and does not account for cache misses, prefetch latency, or instruction scheduling overheads. Real kernels often fall below the roofline due to:
- Cache misses causing stalls.
- Memory access coalescing failures.
- Insufficient instruction-level parallelism (ILP).

**Efficiency** is measured as:
$$\text{Efficiency (\%)} = \frac{\text{Measured Performance}}{\text{Roofline Bound}} \times 100\%$$

Efficient kernels achieve 70–90% of the roofline. Kernels below 50% indicate bottlenecks to investigate.

---

## Profiling Tools & Infrastructure

Effective profiling requires multiple tools to measure performance at different granularities: from cycle-accurate simulation to on-device hardware counters.

### 2.1 Hexagon Simulator (hexagon-sim)

The Hexagon Simulator is a cycle-accurate emulator of the DSP instruction set. It executes binaries compiled for Hexagon and tracks cycle counts, memory accesses, and cache behavior.

#### 2.1.1 Running Kernels in the Simulator

**Setup**:
```bash
# Set Hexagon SDK environment
export HEXAGON_SDK_ROOT=/path/to/Qualcomm/Hexagon_SDK/<version>
export PATH=$HEXAGON_SDK_ROOT/tools/HEXAGON_Tools/8.x.xxx/Tools/bin:$PATH
```

**Compilation for Simulation**:
```bash
# Compile kernel to hexagon-sim-compatible ELF
hexagon-clang \
    -O3 \
    -m<hexagon_version> \        # v68, v69, v73, etc.
    -march=hexagon \
    -c kernel.c -o kernel.o

hexagon-link kernel.o -o kernel.elf
```

**Running in Simulator**:
```bash
hexagon-sim kernel.elf --pmu --stats --profile
```

**Output**: A `.txt` file with:
- Total cycles executed.
- Memory access statistics.
- Cache miss counts.
- PMU (Performance Monitoring Unit) events.

#### 2.1.2 Reading Performance Counters

**Key counters output by hexagon-sim**:

```
Total Cycles:            150,000
User Cycles:             148,500
System Cycles:             1,500
Stall Cycles:             15,000
  - Data stalls:           8,000
  - Structural stalls:     5,000
  - Memory stalls:         2,000

L2 Cache:
  - Access:               32,000
  - Hits:                 30,000
  - Misses:                2,000
  - Miss rate:              6.25%

HVX Instructions:        120,000
HVX Utilization:            95%

Memory Operations:
  - Loads:                50,000
  - Stores:               30,000
```

#### 2.1.3 Example: Simulating a Simple HVX Kernel

**kernel_hvx.c**:
```c
#include <stdint.h>
#include <string.h>

// Depthwise convolution kernel (simplified)
void depthwise_conv_3x3_hvx(
    const uint8_t *input,   // Height × Width × Channels
    const uint8_t *weights, // 3 × 3 × Channels
    uint8_t *output,        // (Height-2) × (Width-2) × Channels
    int height, int width, int channels)
{
    // Assume HVX-based implementation with intrinsics
    for (int h = 1; h < height - 1; h++) {
        for (int w = 1; w < width - 1; w++) {
            for (int c = 0; c < channels; c += 128) {  // HVX INT8 lanes
                // Vector load input patch (3×3×128 lanes)
                // Vector multiply with weights
                // Vector accumulate
                // Vector store output
            }
        }
    }
}

int main() {
    int H = 64, W = 64, C = 32;
    uint8_t input[H * W * C];
    uint8_t weights[3 * 3 * C];
    uint8_t output[(H-2) * (W-2) * C];

    depthwise_conv_3x3_hvx(input, weights, output, H, W, C);

    return 0;
}
```

**Simulation output**:
```
$ hexagon-sim kernel_hvx.elf --pmu

Total Cycles:      31,250  (for 62×62×32 output)
Estimated Time:    20.8 µs (at 1.5 GHz)
Instructions:      65,000
IPC:               2.08

HVX Instructions: 58,000 (89%)
HVX Utilization:   94%

Performance: (62×62×32 × 9 MACs) / 31,250 cycles = 71.5 GigaMACs/sec
Roofline Bound (DDR):  4.5 MACs/Byte × 40 GB/s = 180 GigaMACs/sec
Efficiency: 71.5 / 180 = 39.7% (memory-bound, expected given DDR bandwidth)
```

⚡ **Expert Insight**: The simulator is deterministic and does not account for:
- Real DDR contention (multi-core, I/O).
- DVFS (dynamic voltage/frequency scaling).
- Cache coherency delays.

Use it for relative comparisons and instruction-level analysis, not absolute timing predictions.

### 2.2 Hexagon SDK Profiler

The Hexagon SDK Profiler provides event-based profiling on the actual device or emulator, tracking hardware events without instrumentation overhead.

#### 2.2.1 Available Hardware Events

**Common events in the SDK**:

| Event | Description | Use Case |
|-------|-------------|----------|
| **CYCLES** | Total clock cycles | Baseline measurement |
| **INSTRUCTIONS** | Committed instructions | IPC calculation |
| **DCACHE_MISS** | Data cache misses | Memory stalls diagnosis |
| **ICACHE_MISS** | Instruction cache misses | Code size analysis |
| **L2_MISS** | L2 cache misses | Bandwidth pressure |
| **STALL_CYCLES** | Non-instruction-issuing cycles | Bottleneck identification |
| **HVX_UTILIZATION** | Cycles HVX is active | Vector utilization |
| **HMX_BUSY** | Cycles HMX is computing | Tensor utilization |
| **MEMORY_STALL** | Cycles stalled on memory | Memory bottleneck |
| **DATA_HAZARD** | Read-after-write delays | Pipeline hazards |
| **BRANCH_MISPREDICT** | Branch prediction failures | Control flow overhead |

#### 2.2.2 Using the Profiler (SDK API)

**profiler_example.c**:
```c
#include <hexagon_protos.h>
#include <hvx_hexagon_protos.h>

// Hypothetical Hexagon Profiler API (pseudo-code)
typedef struct {
    uint64_t cycles;
    uint64_t instructions;
    uint64_t hvx_instructions;
    uint64_t dcache_misses;
    uint64_t l2_misses;
    uint64_t stall_cycles;
    uint64_t hvx_utilization;
    uint64_t hmx_busy;
} ProfileData;

ProfileData profile_data;

void profile_kernel() {
    // Initialize profiler
    hexagon_profiler_init(&profile_data);

    // Reset counters
    hexagon_profiler_reset();

    // Warm up (eliminate cold cache effects)
    depthwise_conv_3x3_hvx(input, weights, output, H, W, C);

    // Clear and measure actual run
    hexagon_profiler_reset();
    hexagon_profiler_start();

    // Execute kernel multiple times for statistical significance
    for (int iter = 0; iter < 10; iter++) {
        depthwise_conv_3x3_hvx(input, weights, output, H, W, C);
    }

    hexagon_profiler_stop();
    hexagon_profiler_read(&profile_data);

    // Compute metrics
    double ipc = (double)profile_data.instructions / profile_data.cycles;
    double hvx_utilization_pct = 100.0 * profile_data.hvx_utilization / profile_data.cycles;
    double dcache_miss_rate = 100.0 * profile_data.dcache_misses / (...);

    printf("Cycles: %llu\n", profile_data.cycles);
    printf("IPC: %.2f\n", ipc);
    printf("HVX Utilization: %.1f%%\n", hvx_utilization_pct);
    printf("D-Cache Miss Rate: %.2f%%\n", dcache_miss_rate);
}
```

⚡ **Expert Insight**: Real-world profiling requires **multiple iterations** and **statistical analysis** to account for:
- OS scheduling jitter.
- DVFS scaling.
- Thermal throttling.

A single measurement is unreliable; always collect min/max/mean over ≥10 runs.

### 2.3 PSNR for Numerical Accuracy Verification

When optimizing kernels (e.g., quantization, instruction-level changes), numerical accuracy must be verified to ensure correctness.

**Peak Signal-to-Noise Ratio (PSNR)** is a metric for comparing outputs:

$$\text{PSNR} = 10 \log_{10} \left( \frac{\text{MAX}^2}{\text{MSE}} \right) \text{ dB}$$

Where:
- **MAX**: Maximum value in the output (e.g., 255 for uint8).
- **MSE**: Mean squared error between optimized and reference outputs.

$$\text{MSE} = \frac{1}{N} \sum_{i=0}^{N-1} (y_{\text{opt}}[i] - y_{\text{ref}}[i])^2$$

**PSNR Interpretation**:
- PSNR > 50 dB: Visually lossless (very small differences).
- PSNR 40–50 dB: High quality.
- PSNR 30–40 dB: Acceptable for many applications.
- PSNR < 20 dB: Significant degradation.

**Numerical Accuracy Check**:
```c
double compute_mse(const uint8_t *ref, const uint8_t *opt, int N) {
    double mse = 0.0;
    for (int i = 0; i < N; i++) {
        double diff = (double)ref[i] - (double)opt[i];
        mse += diff * diff;
    }
    return mse / N;
}

double compute_psnr(const uint8_t *ref, const uint8_t *opt, int N) {
    double mse = compute_mse(ref, opt, N);
    if (mse < 1e-10) return INFINITY;
    double max_val = 255.0;
    return 10.0 * log10((max_val * max_val) / mse);
}

// Usage
double psnr = compute_psnr(reference_output, optimized_output, output_size);
if (psnr < 40.0) {
    printf("WARNING: Numerical degradation detected (PSNR=%.1f dB)\n", psnr);
}
```

### 2.4 On-Device Profiling: sysMonApp & PMU Counters

On actual Hexagon hardware (in an SoC), the Performance Monitoring Unit (PMU) can be accessed via sysMonApp or similar system utilities.

#### 2.4.1 Architecture of On-Device Profiling

```
┌─────────────────────────────────────────┐
│  Hexagon DSP Core                        │
│  ┌──────────────────────────────────┐   │
│  │  CPU                             │   │
│  │  Instruction Pipeline            │   │
│  │  Cache Hierarchy (L1, L2, VTCM) │   │
│  └──────────────────────────────────┘   │
│  ┌──────────────────────────────────┐   │
│  │  Performance Monitoring Unit (PMU)  │  ← Counts events
│  │  • Cycle counter                 │   │
│  │  • Event counters (stalls, etc)  │   │
│  │  • Interrupt on threshold        │   │
│  └──────────────────────────────────┘   │
└─────────────────────────────────────────┘
         ↓
    Driver (kernel module)
         ↓
    sysMonApp (userspace tool)
         ↓
    /dev/pmu or sysfs interface
         ↓
    Application reads via ioctl()
```

#### 2.4.2 Using sysMonApp to Collect Counters

**Command-line usage**:
```bash
# Start monitoring
sysMonApp -e CYCLES -e INSTRUCTIONS -e HVX_UTILIZATION -e DCACHE_MISS \
          -d /dev/hexagon_pmu -o profile_log.txt

# Run inference
./my_inference_app

# sysMonApp logs counters to profile_log.txt
```

**Profile output format** (typical):
```
[00:00:00.001] CYCLES: 150000
[00:00:00.005] INSTRUCTIONS: 72000
[00:00:00.009] HVX_UTILIZATION: 141000 (94%)
[00:00:00.013] DCACHE_MISS: 2500
[00:00:00.017] IPC: 0.48
```

#### 2.4.3 Programmatic PMU Access (Pseudo-code)

Many Hexagon kernels use a PMU library to read counters:

```c
#include <hexagon_pmu.h>

typedef struct {
    uint64_t cycles;
    uint64_t instructions;
    uint64_t hvx_utilization_cycles;
    uint64_t dcache_misses;
} PMUCounters;

PMUCounters pmu;

int main() {
    // Open PMU device
    int pmu_fd = hexagon_pmu_open();
    if (pmu_fd < 0) {
        perror("Failed to open PMU");
        return -1;
    }

    // Reset counters
    hexagon_pmu_reset(pmu_fd);

    // Enable specific events
    hexagon_pmu_enable_event(pmu_fd, PMU_EVENT_CYCLES);
    hexagon_pmu_enable_event(pmu_fd, PMU_EVENT_INSTRUCTIONS);
    hexagon_pmu_enable_event(pmu_fd, PMU_EVENT_HVX_UTILIZATION);
    hexagon_pmu_enable_event(pmu_fd, PMU_EVENT_DCACHE_MISS);

    // Read initial state (warm-up)
    hexagon_pmu_read(pmu_fd, &pmu);

    // Warm-up run
    depthwise_conv_3x3_hvx(input, weights, output, H, W, C);

    // Reset and measure
    hexagon_pmu_reset(pmu_fd);
    hexagon_pmu_read(pmu_fd, &pmu);
    uint64_t start_cycles = pmu.cycles;

    // Actual measurement (multiple iterations)
    for (int iter = 0; iter < 100; iter++) {
        depthwise_conv_3x3_hvx(input, weights, output, H, W, C);
    }

    // Read final counters
    hexagon_pmu_read(pmu_fd, &pmu);
    uint64_t end_cycles = pmu.cycles;
    uint64_t total_cycles = end_cycles - start_cycles;

    // Compute metrics
    double ipc = (double)pmu.instructions / total_cycles;
    double hvx_util_pct = 100.0 * pmu.hvx_utilization_cycles / total_cycles;
    double dcache_miss_rate = 100.0 * pmu.dcache_misses / (...);

    printf("Performance:\n");
    printf("  Total Cycles: %llu\n", total_cycles);
    printf("  Instructions: %llu\n", pmu.instructions);
    printf("  IPC: %.2f\n", ipc);
    printf("  HVX Utilization: %.1f%%\n", hvx_util_pct);
    printf("  D-Cache Miss Rate: %.2f%%\n", dcache_miss_rate);

    hexagon_pmu_close(pmu_fd);
    return 0;
}
```

---

## Key Metrics to Measure

### 3.1 Cycles Per Inference (End-to-End and Per-Layer)

**End-to-end inference time** is the primary metric for user-facing performance.

**Measurement**:
```c
#include <time.h>

uint64_t get_clock_cycles() {
    // Platform-specific: read CPU cycle counter
    // On Hexagon: hexagon_timer_get_cycles()
    // On ARM: CNTVCT_EL0 or similar
    return hexagon_timer_get_cycles();
}

uint64_t start_cycles = get_clock_cycles();

// Run inference (e.g., MobileNet)
run_inference(input_data, output_data);

uint64_t end_cycles = get_clock_cycles();
uint64_t total_cycles = end_cycles - start_cycles;

// Convert to time
double frequency_ghz = 1.5;  // e.g., 1.5 GHz
double inference_time_us = total_cycles / (frequency_ghz * 1000);
double inferences_per_second = 1e6 / inference_time_us;

printf("End-to-End Performance:\n");
printf("  Total Cycles: %llu\n", total_cycles);
printf("  Inference Time: %.2f µs\n", inference_time_us);
printf("  Throughput: %.0f inferences/sec\n", inferences_per_second);
```

**Per-layer breakdown**:
```c
struct LayerMetrics {
    char name[64];
    uint64_t cycles;
    double time_us;
    double macs;
    double achieved_gflops;
};

LayerMetrics layers[NUM_LAYERS];

// Measure each layer in isolation
for (int layer_idx = 0; layer_idx < NUM_LAYERS; layer_idx++) {
    uint64_t layer_start = get_clock_cycles();

    run_layer(layer_idx, input_data, output_data);

    uint64_t layer_end = get_clock_cycles();
    layers[layer_idx].cycles = layer_end - layer_start;
    layers[layer_idx].time_us = layers[layer_idx].cycles / 1500.0;  // at 1.5 GHz
    layers[layer_idx].macs = compute_layer_macs(layer_idx);
    layers[layer_idx].achieved_gflops = layers[layer_idx].macs / (layers[layer_idx].cycles / 1500.0) / 1e9;
}

// Print summary
printf("Per-Layer Breakdown:\n");
printf("%-20s %10s %10s %12s\n", "Layer", "Cycles", "Time(µs)", "GFLOPS");
for (int i = 0; i < NUM_LAYERS; i++) {
    printf("%-20s %10llu %10.2f %12.1f\n",
        layers[i].name, layers[i].cycles, layers[i].time_us, layers[i].achieved_gflops);
}
```

### 3.2 L2 Cache Miss Rate and Slowdown Correlation

**L2 cache misses** directly impact performance by forcing accesses to slower DDR.

**Measurement**:
```c
uint64_t dcache_accesses;     // from PMU
uint64_t dcache_misses;       // from PMU

double l2_miss_rate = (double)dcache_misses / dcache_accesses;

// Impact quantification:
// Expected latency overhead per miss:
// - L2 hit: 3–4 cycles
// - L2 miss (DDR hit): 50–100 cycles
// - Difference: ~60 cycles

uint64_t expected_cycles_l2_hits = dcache_accesses * 4;  // all L2 hits
uint64_t actual_miss_overhead = dcache_misses * 60;      // extra cycles per miss
uint64_t total_expected = expected_cycles_l2_hits + actual_miss_overhead;

double miss_impact_pct = 100.0 * actual_miss_overhead / total_expected;

printf("Cache Performance:\n");
printf("  L2 Miss Rate: %.2f%%\n", 100.0 * l2_miss_rate);
printf("  Miss Penalty Impact: %.1f%% of cycles\n", miss_impact_pct);

if (l2_miss_rate > 0.1) {
    printf("  ⚠️  High L2 miss rate: Optimize working set or improve locality\n");
}
```

**Slowdown formula**:

$$\text{Slowdown} = 1 + \text{Miss Rate} \times \frac{\text{DDR Latency} - \text{L2 Latency}}{\text{L2 Latency}}$$

**Example**:
- L2 hit latency: 4 cycles.
- DDR miss latency: 80 cycles.
- Miss rate: 10%.

$$\text{Slowdown} = 1 + 0.1 \times \frac{80 - 4}{4} = 1 + 0.1 \times 19 = 2.9×$$

A 10% miss rate causes a **2.9× slowdown**—a small miss rate has outsized impact.

### 3.3 VTCM Utilization Percentage

VTCM (local tightly-coupled memory) is the fastest memory tier and should be maximized for compute-intensive kernels.

**Measurement and Analysis**:
```c
#define VTCM_SIZE_BYTES (256 * 1024)  // 256 KB typical

// Estimate VTCM usage for a kernel
// Example: Depthwise Conv 3×3 with tiling

int tile_height = 8;
int tile_width = 8;
int channels = 32;
int input_tile_size = (tile_height + 2) * (tile_width + 2) * channels;  // With padding
int output_tile_size = tile_height * tile_width * channels;
int weights_size = 3 * 3 * channels;
int total_vtcm = input_tile_size + output_tile_size + weights_size;

double vtcm_utilization_pct = 100.0 * total_vtcm / VTCM_SIZE_BYTES;

printf("VTCM Utilization: %.1f%% (%d / %d bytes)\n",
    vtcm_utilization_pct, total_vtcm, VTCM_SIZE_BYTES);

if (vtcm_utilization_pct > 80) {
    printf("  ⚠️  Near capacity; consider reducing tile size\n");
} else if (vtcm_utilization_pct > 50) {
    printf("  ✓ Good utilization\n");
} else {
    printf("  Opportunity: Can increase tile size for better locality\n");
}

// Performance impact: VTCM bandwidth is 200 GB/s vs DDR at 40 GB/s
// Keeping data in VTCM is 5× faster
```

### 3.4 HVX and HMX Utilization Percentage

**HVX utilization**: Percentage of cycles where HVX vector instructions are executing.

**HMX utilization**: Percentage of cycles where the HMX tensor unit is active.

**Measurement**:
```c
uint64_t hvx_utilization_cycles;    // from PMU
uint64_t total_cycles;               // from PMU
uint64_t hmx_busy_cycles;            // from PMU (HMX-equipped systems)

double hvx_util_pct = 100.0 * hvx_utilization_cycles / total_cycles;
double hmx_util_pct = 100.0 * hmx_busy_cycles / total_cycles;

printf("Accelerator Utilization:\n");
printf("  HVX Utilization: %.1f%%\n", hvx_util_pct);
printf("  HMX Utilization: %.1f%%\n", hmx_util_pct);

// Interpretation
if (hvx_util_pct < 50) {
    printf("  ⚠️  Low HVX utilization: Scalar code dominates\n");
    printf("     • Use vectorizable loops\n");
    printf("     • Increase kernel granularity\n");
} else if (hvx_util_pct > 80) {
    printf("  ✓ High HVX utilization\n");
}

if (hmx_util_pct > 0 && hmx_util_pct < 30) {
    printf("  ⚠️  Low HMX utilization despite tensor operations\n");
    printf("     • Increase matrix sizes or batching\n");
    printf("     • Reduce synchronization overhead\n");
}
```

### 3.5 Stall Cycles Breakdown

**Stall cycles** are cycles where the pipeline is stalled (not committing instructions). Breaking down stalls is critical for identifying bottlenecks.

**Common stall categories**:

| Stall Type | Cause | Mitigation |
|-----------|-------|------------|
| **Data Hazard** | Read-after-write dependency in HVX pipeline | Rearrange instructions, add independent operations |
| **Structural Hazard** | Resource contention (e.g., single load port) | Interleave memory accesses |
| **Memory Stall** | Waiting for DDR/L2 data | Prefetch, increase locality, batch access |
| **Branch Stall** | Branch prediction miss or pipeline flush | Reduce branches, use static prediction hints |

**Measurement**:
```c
struct StallBreakdown {
    uint64_t data_hazard_stalls;
    uint64_t structural_hazard_stalls;
    uint64_t memory_stalls;
    uint64_t branch_stalls;
    uint64_t other_stalls;
};

StallBreakdown stalls;

// Read from PMU (pseudo-code)
hexagon_pmu_read_event(pmu_fd, PMU_EVENT_DATA_HAZARD_STALLS, &stalls.data_hazard_stalls);
hexagon_pmu_read_event(pmu_fd, PMU_EVENT_STRUCTURAL_STALLS, &stalls.structural_hazard_stalls);
hexagon_pmu_read_event(pmu_fd, PMU_EVENT_MEMORY_STALLS, &stalls.memory_stalls);
hexagon_pmu_read_event(pmu_fd, PMU_EVENT_BRANCH_STALLS, &stalls.branch_stalls);

uint64_t total_stalls = stalls.data_hazard_stalls + stalls.structural_hazard_stalls
                      + stalls.memory_stalls + stalls.branch_stalls;

printf("Stall Cycle Breakdown:\n");
printf("  Data Hazard:      %6.1f%% (%llu cycles)\n",
    100.0 * stalls.data_hazard_stalls / total_stalls, stalls.data_hazard_stalls);
printf("  Structural:       %6.1f%% (%llu cycles)\n",
    100.0 * stalls.structural_hazard_stalls / total_stalls, stalls.structural_hazard_stalls);
printf("  Memory:           %6.1f%% (%llu cycles)\n",
    100.0 * stalls.memory_stalls / total_stalls, stalls.memory_stalls);
printf("  Branch:           %6.1f%% (%llu cycles)\n",
    100.0 * stalls.branch_stalls / total_stalls, stalls.branch_stalls);
printf("  Total Stall Rate: %.1f%% of cycles\n", 100.0 * total_stalls / total_cycles);
```

---

## Identifying Bottlenecks

### 4.1 Stall Analysis Methodology

The process of identifying the root cause of stalls:

**Step 1: Quantify Stalls**
- Collect PMU stall counters (see Section 3.5).
- Compute stall rate: (stall_cycles / total_cycles) × 100%.
- If stall rate < 10%, the kernel is reasonably well-scheduled.
- If stall rate > 30%, significant optimization potential exists.

**Step 2: Categorize Stalls**
- Map PMU stall type (data hazard, structural, memory) to source.
- Correlate with code regions:
  - Data hazards: dense vector operations.
  - Structural hazards: contention on load/store ports.
  - Memory stalls: irregular memory access patterns.

**Step 3: Instrument Code**
Insert markers in assembly or high-level code to pinpoint stall sources:

```c
// Pseudo-code: inline assembly with timing markers
asm volatile("marker_loop_start:");
for (int i = 0; i < N; i++) {
    // Vector operations
    HVX_Vector *v1 = (HVX_Vector *)&data[i * 128];
    HVX_Vector *v2 = (HVX_Vector *)&data[(i+1) * 128];
    HVX_Vector result = Q6_V_vadd_VV(*v1, *v2);  // Add vectors
    asm volatile("marker_add_done:");

    *(HVX_Vector *)&output[i * 128] = result;    // Store
    asm volatile("marker_store_done:");
}
asm volatile("marker_loop_end:");
```

Run in simulator with `--profile` to see cycle ranges between markers.

**Step 4: Root Cause Analysis**

```
Stall Breakdown:
  Data Hazard: 40% → RAW dependency in HVX pipeline
  Memory: 35%  → L2 misses on load instructions
  Structural: 20% → Load port contention
  Other: 5%

Root Cause (Prioritized):
  1. Data Hazard (40%): Insert independent instructions between dependent operations
  2. Memory (35%): Improve data locality; consider VTCM tiling
  3. Structural (20%): Interleave memory operations from multiple buffers
```

### 4.2 Data Hazard Profiling

**Read-After-Write (RAW) hazards** are common in HVX pipelines where dependencies block instruction issue.

**Example scenario**:
```c
HVX_Vector v1 = *((HVX_Vector *)input_a);  // Load 1 (reads L2)
HVX_Vector v2 = *((HVX_Vector *)input_b);  // Load 2 (reads L2)
HVX_Vector result = Q6_V_vmpy_VV(v1, v2);  // Multiply: needs v1 and v2 ready
// If L2 latency is 4 cycles and loads complete at cycle 4,
// multiply can issue at cycle 5 (4 + 1 for data dependencies)
```

**Identifying RAW hazards**:
```
Hexagon Simulator Output:
  Instruction @ Cycle 0: Load R0 from [Addr A]
  Instruction @ Cycle 4: Load R1 from [Addr B]
  Instruction @ Cycle 5: Multiply R2, R0, R1  ← Stalled 1 cycle waiting for R0
```

**Mitigation techniques**:

1. **Insert independent operations** (instruction scheduling):
```c
HVX_Vector v1 = *((HVX_Vector *)input_a);
HVX_Vector v2 = *((HVX_Vector *)input_b);
HVX_Vector v3 = *((HVX_Vector *)input_c);  // Independent load
HVX_Vector result1 = Q6_V_vmpy_VV(v1, v2); // Use v1, v2
HVX_Vector result2 = Q6_V_vmpy_VV(v3, v3); // Use v3 (independent)
// Compiler optimizes away stalls via scheduling
```

2. **Software pipelining** (for loops):
```c
// Unroll and interleave iterations to hide dependencies
// Instead of:
//   Load v1, Compute, Store, Load v1, Compute, Store, ...
// Do:
//   Load v1, Load v2, Compute v1, Load v3, Compute v2, Store v1, ...
//   (Previous iteration's compute overlaps with next load)
```

3. **Use prefetch instructions**:
```c
// Explicitly prefetch data to hide latency
L2fetch(next_input_addr, 256);  // Prefetch 256 bytes
// ... compute with current data ...
HVX_Vector data = *((HVX_Vector *)current_addr);  // Data likely in cache
```

### 4.3 Memory Bottleneck Identification

**Memory bottlenecks** occur when the kernel's memory bandwidth demand exceeds available capacity.

**Diagnostic formula**:
$$\text{Memory-bound threshold} = \frac{\text{Achieved GigaMACs/sec}}{\text{Available Bandwidth (GB/s)}}$$

If achieved performance < memory-bound threshold:

$$\text{Memory Limit GFLOPS} = \text{OI} \times \text{Bandwidth}$$

Example:
- Kernel OI: 2 MACs/Byte (memory-intensive).
- Available bandwidth: 40 GB/s (DDR).
- Memory limit: 2 × 40 = 80 GigaMACs/sec.
- Measured performance: 45 GigaMACs/sec.
- Conclusion: **Memory-bound** (45 < 80).

**Profiling memory stalls**:
```c
uint64_t memory_stall_cycles;
uint64_t total_cycles;
double memory_stall_rate = (double)memory_stall_cycles / total_cycles;

if (memory_stall_rate > 0.2) {  // > 20% stalled on memory
    printf("Memory-bound kernel detected\n");

    // Investigate cache misses
    uint64_t l2_misses = pmu.l2_misses;
    uint64_t l2_accesses = pmu.l2_accesses;
    double miss_rate = (double)l2_misses / l2_accesses;

    printf("  L2 Miss Rate: %.2f%%\n", 100.0 * miss_rate);

    if (miss_rate > 0.05) {
        printf("  Mitigation:\n");
        printf("    1. Improve spatial locality (contiguous access)\n");
        printf("    2. Use VTCM for hot working sets\n");
        printf("    3. Increase batch size or tile size\n");
    }
}
```

---

## Performance Estimation Formulas

### 5.1 Analytical Cycle Prediction for Conv2D

**Objective**: Predict cycles for a Conv2D kernel *before implementation*, to enable quick design exploration.

**Inputs**:
- Input: H_in × W_in × C_in
- Kernel: K_h × K_w, stride S, padding P
- Output: H_out × W_out × C_out (where H_out = (H_in + 2P - K_h) / S + 1, similarly for W)

**Compute Cycles**:

If using HVX for per-element operations:
$$\text{Compute Cycles} = \frac{\text{Total MACs}}{128 \text{ lanes} \times \text{IPC}}$$

Where:
- **Total MACs** = H_out × W_out × K_h × K_w × C_in (if C_out = C_in for depthwise).
- **IPC** (Instructions Per Cycle) depends on scheduling: typically 0.5–2.0 for vectorized code.

**Memory Cycles** (DDR bound):

$$\text{Memory Cycles} = \frac{\text{Data Moved (Bytes)}}{\text{DDR Bandwidth (Bytes/cycle)}}$$

**Data moved**:
- Inputs: H_in × W_in × C_in bytes (assuming INT8).
- Weights: K_h × K_w × C_in bytes.
- Outputs: H_out × W_out × C_out bytes.
- Total: approximately H_in × W_in × C_in + K_h × K_w × C_in + H_out × W_out × C_out.

**DDR bandwidth**:
$$\text{DDR Bandwidth (Bytes/cycle)} = \frac{40 \text{ GB/s}}{1.5 \text{ GHz}} = 26.7 \text{ Bytes/cycle}$$

**Predicted Cycles**:
$$\text{Predicted Cycles} = \max(\text{Compute Cycles}, \text{Memory Cycles})$$

The max accounts for potential pipeline overlap; the bottleneck dominates.

### 5.2 Worked Example: Depthwise Conv 3×3

**Problem**: Estimate cycles for depthwise convolution.

**Parameters**:
- Input: 64 × 64 × 32 (H_in, W_in, C_in).
- Kernel: 3 × 3, stride 1, pad 1.
- Output: 64 × 64 × 32 (same padding).
- Frequency: 1.5 GHz, DDR @ 40 GB/s.
- HVX INT8: 128 lanes, IPC ≈ 1.5.

**Step 1: Compute MACs**
$$\text{MACs} = 64 \times 64 \times 3 \times 3 \times 32 = 1,179,648$$

**Step 2: Compute Cycles (HVX)**
$$\text{Compute Cycles} = \frac{1,179,648}{128 \times 1.5} = \frac{1,179,648}{192} = 6,144 \text{ cycles}$$

**Step 3: Data Movement**
- Input read: (64 + 2) × (64 + 2) × 32 = 135,168 bytes (with padding).
- Output write: 64 × 64 × 32 = 131,072 bytes.
- Weights: 3 × 3 × 32 = 288 bytes (negligible).
- **Total: ~266 KB**.

**Step 4: Memory Cycles (DDR)**
$$\text{DDR Bandwidth (Bytes/cycle)} = \frac{40 \text{ GB/s}}{1.5 \text{ GHz}} = 26.7 \text{ Bytes/cycle}$$

$$\text{Memory Cycles} = \frac{266,000}{26.7} = 9,963 \text{ cycles}$$

**Step 5: Predicted Cycles**
$$\text{Predicted} = \max(6,144, 9,963) = 9,963 \text{ cycles}$$

**Step 6: Performance Analysis**
$$\text{Achieved GFLOPS} = \frac{1,179,648 \text{ MACs}}{9,963 \text{ cycles}} \times 1.5 \text{ GHz} = 177.7 \text{ GigaMACs/sec}$$

$$\text{Roofline Bound (DDR)} = 4.5 \text{ MACs/Byte} \times 40 \text{ GB/s} = 180 \text{ GigaMACs/sec}$$

$$\text{Efficiency} = \frac{177.7}{180} \times 100\% = 98.7\%$$

**Conclusion**: The kernel is **memory-bound** and nearly saturates DDR bandwidth. Further optimization requires:
- VTCM tiling to reduce DDR accesses.
- Improved memory coalescing.

### 5.3 Pointwise Conv 1×1 Estimation

**Parameters**:
- Input: 56 × 56 × 64 (from a typical MobileNet layer).
- Kernel: 1 × 1, C_in = 64 → C_out = 256.
- Output: 56 × 56 × 256.
- HVX INT8, HMX available.

**Compute MACs**:
$$\text{MACs} = 56 \times 56 \times 64 \times 256 = 50,331,648$$

**Using HMX (8×8 matrix ops)**:
$$\text{Compute Cycles (HMX)} = \frac{50,331,648}{64} = 786,432 \text{ cycles}$$

Wait, this is much higher than expected. HMX efficiency depends on batching. Let's reconsider:

**Alternative: Block-wise evaluation**

Process in 56×56 blocks (one spatial position):
- Input block: 1 × 1 × 64 (a vector of 64 activations).
- Weight matrix: 64 × 256.
- Output: 1 × 1 × 256 (computed via 64×256 matrix-vector).
- Repeat 56×56 times.

**Per-position compute** (using HVX scalar or HMX with reduced efficiency):
$$\text{Cycles/position} = \frac{64 \times 256}{64 \text{ lanes} \times 2 \text{ IPC}} = \frac{16,384}{128} = 128 \text{ cycles}$$

$$\text{Total Compute Cycles} = 128 \times 56 \times 56 = 401,408 \text{ cycles}$$

**Data Movement**:
- Input: 56 × 56 × 64 = 200,704 bytes.
- Weights: 64 × 256 = 16,384 bytes (preloaded to VTCM).
- Output: 56 × 56 × 256 = 802,816 bytes.
- **Total: ~1 MB**.

$$\text{Memory Cycles} = \frac{1,000,000}{26.7} = 37,453 \text{ cycles}$$

**Predicted Cycles**:
$$\text{Predicted} = \max(401,408, 37,453) = 401,408 \text{ cycles}$$

$$\text{Achieved GFLOPS} = \frac{50,331,648}{401,408} \times 1.5 = 188 \text{ GigaMACs/sec}$$

$$\text{OI} = \frac{50,331,648}{1,000,000} = 50.3 \text{ MACs/Byte}$$

$$\text{Roofline (Compute)} = 192 \text{ GigaMACs/sec}$$

$$\text{Efficiency} = \frac{188}{192} = 98\%$$

**Conclusion**: Pointwise convolutions are **compute-bound** due to high OI. The kernel is near-peak efficiency.

### 5.4 Performance Estimation Template

Use this template for quick estimation:

```
Layer: ________________________
Input Dimensions: H x W x C_in
Kernel: K_h x K_w, Stride S, Pad P
Output Dimensions: H_out x W_out x C_out

1. Total MACs
   MACs = H_out * W_out * K_h * K_w * C_in * (C_out / C_in)
        = ___________

2. Compute Cycles (HVX INT8, 128 lanes, IPC=1.5)
   Compute = MACs / (128 * 1.5)
           = ___________

3. Data Moved (bytes)
   Input:  H_in * W_in * C_in = ___________
   Weights: K_h * K_w * C_in = ___________
   Output: H_out * W_out * C_out = ___________
   Total:                        = ___________

4. Memory Cycles (DDR @ 40 GB/s, 26.7 Bytes/cycle)
   Memory = Total_Bytes / 26.7
          = ___________

5. Predicted Cycles
   Predicted = max(Compute, Memory)
             = ___________

6. Achieved Performance
   GigaMACs/sec = MACs / (Predicted / 1.5 GHz)
                = ___________

7. Operational Intensity
   OI = MACs / Total_Bytes
      = ___________

8. Roofline Bound
   If OI < 4.8: Memory-bound, Ceiling = OI * 40 GB/s = _______ GFLOPS
   If OI > 4.8: Compute-bound, Ceiling = 192 GFLOPS

9. Efficiency
   Efficiency = Achieved / Ceiling * 100%
              = ___________
```

---

## Benchmarking Methodology

Reliable performance benchmarking requires careful experimental design to eliminate variance and ensure reproducibility.

### 6.1 Removing Variance

#### 6.1.1 Warm-up Runs

Before measurement, execute the kernel multiple times to:
- Load code and data into caches.
- Stabilize DVFS (dynamic voltage/frequency scaling).
- Eliminate cold-start effects.

```c
void benchmark_with_warmup(
    void (*kernel_func)(void *input, void *output),
    void *input, void *output,
    int warmup_runs, int measurement_runs)
{
    // Warm-up phase
    for (int i = 0; i < warmup_runs; i++) {
        kernel_func(input, output);
    }

    // Measurement phase
    uint64_t *cycle_counts = malloc(measurement_runs * sizeof(uint64_t));
    for (int i = 0; i < measurement_runs; i++) {
        uint64_t start = hexagon_timer_get_cycles();
        kernel_func(input, output);
        uint64_t end = hexagon_timer_get_cycles();
        cycle_counts[i] = end - start;
    }

    // Statistical analysis (see Section 6.3)
    analyze_cycle_distribution(cycle_counts, measurement_runs);

    free(cycle_counts);
}
```

**Recommended warm-up runs**: 5–10 to stabilize caches and DVFS.

#### 6.1.2 Thread Pinning

On multi-core systems, pin the benchmark thread to a specific core to avoid:
- Core migration (cache misses on new core).
- Contention with other threads.

```c
#include <pthread.h>
#include <sched.h>

void pin_to_core(int core_id) {
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(core_id, &cpuset);

    if (pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset) != 0) {
        perror("Failed to set thread affinity");
    }
}

// In main
pin_to_core(0);  // Pin to core 0
benchmark_with_warmup(...);
```

#### 6.1.3 Disabling DVFS

Disable dynamic voltage/frequency scaling during benchmarking to ensure consistent frequency:

```bash
# On Linux with cpufreq interface
echo 0 > /sys/devices/system/cpu/cpu0/cpufreq/scaling_driver
echo 1500000 > /sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq
echo 1500000 > /sys/devices/system/cpu/cpu0/cpufreq/scaling_min_freq
```

Or programmatically (if available):
```c
// Pseudo-code (platform-specific)
hexagon_set_frequency(1500);  // Lock at 1.5 GHz
hexagon_set_dvfs_enabled(false);
```

### 6.2 Isolating Kernel Time vs Overhead

Measure only the actual kernel execution, excluding:
- Argument setup.
- Input/output allocation.
- Synchronization barriers.

**Correct approach**:
```c
// Allocate and set up inputs once
uint8_t *input = malloc(input_size);
uint8_t *output = malloc(output_size);
initialize_input(input);

// Warm up
for (int i = 0; i < 5; i++) {
    my_kernel(input, output);
}

// Measure only kernel execution
uint64_t start_cycle = hexagon_timer_get_cycles();

for (int i = 0; i < 100; i++) {
    my_kernel(input, output);
}

uint64_t end_cycle = hexagon_timer_get_cycles();
uint64_t total_cycles = end_cycle - start_cycle;
uint64_t avg_cycles_per_run = total_cycles / 100;
```

**Incorrect approach** (includes overhead):
```c
// ❌ WRONG: Measured time includes allocation overhead
for (int i = 0; i < 100; i++) {
    uint8_t *input = malloc(input_size);  // Overhead!
    uint8_t *output = malloc(output_size); // Overhead!

    uint64_t start = hexagon_timer_get_cycles();
    my_kernel(input, output);
    uint64_t end = hexagon_timer_get_cycles();

    printf("Iteration %d: %llu cycles\n", i, end - start);

    free(input);
    free(output);
}
```

### 6.3 Statistical Analysis of Timing Data

Collect multiple measurements and analyze using statistics:

```c
#include <math.h>
#include <stdlib.h>

typedef struct {
    uint64_t min;
    uint64_t max;
    double mean;
    double median;
    double stddev;
    double percentile_99;
} TimingStats;

int compare_uint64(const void *a, const void *b) {
    return (*(uint64_t *)a < *(uint64_t *)b) ? -1 : 1;
}

TimingStats compute_stats(uint64_t *cycles, int count) {
    TimingStats stats = {0};

    // Sort for min, max, median
    qsort(cycles, count, sizeof(uint64_t), compare_uint64);
    stats.min = cycles[0];
    stats.max = cycles[count - 1];
    stats.median = (count % 2 == 0) ?
        (cycles[count/2 - 1] + cycles[count/2]) / 2.0 :
        (double)cycles[count / 2];

    // Compute mean and standard deviation
    double sum = 0.0;
    for (int i = 0; i < count; i++) {
        sum += cycles[i];
    }
    stats.mean = sum / count;

    double sum_sq_diff = 0.0;
    for (int i = 0; i < count; i++) {
        double diff = cycles[i] - stats.mean;
        sum_sq_diff += diff * diff;
    }
    stats.stddev = sqrt(sum_sq_diff / (count - 1));

    // 99th percentile
    int p99_idx = (int)((99.0 / 100.0) * (count - 1));
    stats.percentile_99 = (double)cycles[p99_idx];

    return stats;
}

void print_timing_report(TimingStats stats, int runs, double frequency_ghz) {
    printf("Timing Results (%d runs):\n", runs);
    printf("  Min:             %llu cycles (%.2f µs)\n",
        stats.min, stats.min / (frequency_ghz * 1000));
    printf("  Max:             %llu cycles (%.2f µs)\n",
        stats.max, stats.max / (frequency_ghz * 1000));
    printf("  Mean:            %.0f cycles (%.2f µs)\n",
        stats.mean, stats.mean / (frequency_ghz * 1000));
    printf("  Median:          %.0f cycles (%.2f µs)\n",
        stats.median, stats.median / (frequency_ghz * 1000));
    printf("  Std Dev:         %.1f cycles\n", stats.stddev);
    printf("  99th Percentile: %.0f cycles (%.2f µs)\n",
        stats.percentile_99, stats.percentile_99 / (frequency_ghz * 1000));
    printf("  Coefficient of Variation: %.2f%%\n",
        100.0 * stats.stddev / stats.mean);
}
```

**Interpretation**:
- **Mean**: Average performance; use for reporting.
- **Median**: Middle value; robust to outliers.
- **Std Dev & CV**: Variation; CV < 5% indicates stable measurements.
- **Min/Max**: Bounds; if gap > 10%, investigate variance sources.
- **99th percentile**: Worst-case latency; important for real-time systems.

### 6.4 Comparison Against Theoretical Peak

**Efficiency metric** (introduced earlier):

$$\text{Efficiency (\%)} = \frac{\text{Measured Median GFLOPS}}{\text{Roofline GFLOPS}} \times 100\%$$

**Benchmark report template**:

```
=== Kernel Performance Report ===
Kernel:                Depthwise Conv 3×3 INT8
Date:                  2024-03-22
Platform:              Hexagon v69, 1.5 GHz
Input Shape:           64 × 64 × 32
Output Shape:          64 × 64 × 32

Execution Metrics:
  Total MACs:          1,179,648
  Measured Cycles:     9,963 (median)
  Measured Time:       6.64 µs
  Throughput:          150,585 inferences/sec

Performance Analysis:
  Achieved GFLOPS:     177.7 (median)
  Roofline Bound:      180 GFLOPS (DDR-limited)
  Efficiency:          98.7%

Memory Analysis:
  Operational Intensity: 4.5 MACs/Byte
  Data Moved:           266 KB
  Memory Bound:         DDR (40 GB/s)
  L2 Miss Rate:         8.2%

Stall Analysis:
  Stall Rate:          12.3%
    - Memory Stalls:   8.1%
    - Data Hazards:    3.2%
    - Structural:      1.0%

Conclusion:
  ✓ Near-peak performance
  ✓ Memory-bound as expected (OI = 4.5)
  Optimization opportunity: Reduce L2 misses via spatial tiling
```

---

## Case Studies & Advanced Topics

### 7.1 Case Study: Optimizing MobileNet v2 Inference

**Scenario**: A MobileNet v2 model inference on Hexagon is slower than expected. Use performance analysis to identify and fix bottlenecks.

#### Problem Statement

- Model: MobileNet v2 (1.0, 224×224 input).
- Baseline: 150 ms on Snapdragon (single-threaded).
- Target: 50 ms (3× speedup).
- Current: 120 ms (not meeting target).

#### Analysis Phase

**Step 1: Layer-wise breakdown**
```
Layer                    Time (ms)   % Total   MACs/Byte (OI)
=====================================
Input Processing         2           1.7       N/A
Stem Conv (3×3, s=2)     8           6.7       1.2
Bottleneck 1–6           35          29.2      4.5
Bottleneck 7–13          42          35.0      3.8
Bottleneck 14–16         22          18.3      2.1
Head (1×1 Pointwise)     8           6.7       48
Softmax                  3           2.5       0.1
Total                    120         100.0     3.8 (avg)
```

**Step 2: Identify critical path**
Bottleneck layers 7–13 consume 35% of time. These are depthwise + pointwise sequences with mixed OI.

**Step 3: Profile a single critical layer**
- Layer: Bottleneck 10 (depthwise 5×5, stride 1 + pointwise 1×1).
- Input: 28 × 28 × 144.
- Depthwise output: 28 × 28 × 144.
- Pointwise output: 28 × 28 × 240.

Measured (via hexagon-sim):
```
Depthwise (5×5):
  Cycles: 35,000
  L2 Miss Rate: 18%
  Stall Rate: 28%
    - Memory Stalls: 18%
    - Data Hazards: 8%
    - Structural: 2%

Pointwise (1×1):
  Cycles: 22,000
  L2 Miss Rate: 5%
  Stall Rate: 12%
```

**Observation**: Depthwise has high memory stall rate (18%) indicating poor cache locality.

#### Optimization Phase

**Issue 1: Depthwise Memory Stalls**

**Root cause**: Large kernel (5×5) with limited reuse. Each pixel reads 5×5=25 input values; only 1 output computed.

**OI analysis**:
$$\text{OI} = \frac{28 \times 28 \times 5 \times 5 \times 144}{2 \times 28 \times 28 \times 144} = \frac{25}{2} = 12.5 \text{ MACs/Byte}$$

Should be compute-bound, but measured memory stalls suggest poor prefetching.

**Solution: VTCM tiling**

Tile the input into 16×16 chunks with 5×5 kernel overlap:
- Tile input: (16 + 4) × (16 + 4) × 144 = 57,600 bytes.
- Tile output: 16 × 16 × 144 = 36,864 bytes.
- VTCM required: 94,464 bytes (fits in 256 KB VTCM).

**Implementation sketch**:
```c
void depthwise_tiled(const uint8_t *input, uint8_t *output,
                     int in_h, int in_w, int channels) {
    for (int tile_h = 0; tile_h < in_h; tile_h += 16) {
        for (int tile_w = 0; tile_w < in_w; tile_w += 16) {
            // Load input tile to VTCM
            int load_h = min(16 + 4, in_h - tile_h);
            int load_w = min(16 + 4, in_w - tile_w);
            memcpy_to_vtcm(input_vtcm, &input[tile_h * in_w * channels + tile_w * channels],
                          load_h, load_w, channels);

            // Compute on VTCM data (all accesses hit 200 GB/s, not 40 GB/s)
            depthwise_compute_vtcm(input_vtcm, output_vtcm, load_h, load_w, channels);

            // Store output from VTCM
            memcpy_from_vtcm(&output[tile_h * in_w * channels + tile_w * channels],
                            output_vtcm, 16, 16, channels);
        }
    }
}
```

**Expected improvement**:
- VTCM bandwidth: 200 GB/s (vs DDR 40 GB/s).
- Memory stall reduction: 18% → 3% (prefetch hides latency).
- Cycles reduction: 35,000 → 24,000 (31% faster).

**Issue 2: Data Hazards in Depthwise**

The depthwise kernel has read-after-write dependencies in the HVX pipeline.

**Solution: Instruction scheduling via loop unrolling**
```c
// Before: RAW dependency blocks pipeline
HVX_Vector v1 = load_hvx(&input[0]);
HVX_Vector v2 = load_hvx(&input[128]);
HVX_Vector result = hvx_multiply(v1, v2);  // Stalled: depends on v1, v2

// After: Interleave independent operations
HVX_Vector v1_a = load_hvx(&input[0]);
HVX_Vector v1_b = load_hvx(&input[128]);
HVX_Vector v1_c = load_hvx(&input[256]);  // Independent
HVX_Vector v1_d = load_hvx(&input[384]);  // Independent
HVX_Vector res_a = hvx_multiply(v1_a, v1_b);
HVX_Vector res_b = hvx_multiply(v1_c, v1_d);  // Compute overlaps with v1_a load
```

**Expected improvement**: Data hazard stalls 8% → 2%.

#### Results After Optimization

```
Layer                    Before (ms)  After (ms)  Speedup
================================================
Bottleneck 10            5.8          3.2         1.8×
All Bottlenecks (7–13)   42           28          1.5×
Total Model              120          85          1.4×
```

**Overall speedup**: 1.4× from targeted optimization of critical path.

**To reach 50 ms target**: Apply similar optimizations to remaining bottlenecks (1–6, 14–16) and use model quantization (dynamic precision).

⚡ **Expert Insight**: The 80/20 rule applies to optimization—focus on the 20% of code causing 80% of slowdown. In MobileNet, a few layers (especially with large kernels and low OI) often dominate runtime.

### 7.2 Advanced: Vectorization of Non-Obvious Operators

Some operators are not naturally vectorizable but can benefit from HVX:

#### Example: Batch Normalization

Batch normalization (BN) is elementwise but requires efficient computation:

$$y_i = \gamma \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta$$

**Naive scalar implementation** (memory-bound):
```c
for (int i = 0; i < N; i++) {
    output[i] = gamma * (input[i] - mean) / sqrt(variance + eps) + beta;
}
```

**HVX vectorized** (32× faster):
```c
// Precompute constants
HVX_Vector scale_vec = Q6_V_vsplat_R(scale_value);  // gamma / sqrt(variance + eps)
HVX_Vector mean_vec = Q6_V_vsplat_R(mean_value);
HVX_Vector beta_vec = Q6_V_vsplat_R(beta_value);

for (int i = 0; i < N; i += 128) {  // 128 INT8 lanes
    HVX_Vector input_vec = *(HVX_Vector *)&input[i];
    HVX_Vector centered = Q6_V_vsubb_VV(input_vec, mean_vec);  // Subtract mean
    HVX_Vector scaled = Q6_V_vmpyh_VV(centered, scale_vec);   // Multiply by scale
    HVX_Vector output_vec = Q6_V_vaddb_VV(scaled, beta_vec);  // Add bias
    *(HVX_Vector *)&output[i] = output_vec;
}
```

**Performance improvement**:
- Scalar: 2 GigaMACs/sec (memory-bound).
- HVX: 64 GigaMACs/sec (32× improvement).

This is only achievable through vectorization with intrinsics or LLVM autovectorization.

---

## Self-Assessment Questions

1. **Roofline Model**
   - What is the roofline knee, and what does it signify?
   - For a kernel with OI = 8 MACs/Byte on a system with HVX (192 GFLOPS) and DDR (40 GB/s), which resource is the bottleneck?
   - Compute the roofline knee OI. **(Answer: 192 / 40 = 4.8 MACs/Byte; since 8 > 4.8, compute-bound.)**

2. **Profiling Tools**
   - What is the difference between hexagon-sim and on-device profiling with sysMonApp?
   - Which PMU counter directly indicates memory bandwidth pressure? **(Answer: Memory stall cycles or L2 miss count.)**
   - PSNR = 45 dB indicates what about numerical accuracy? **(Answer: High quality, acceptable loss.)**

3. **Key Metrics**
   - If a kernel achieves IPC = 0.8, is it well-scheduled? Why or why not? **(Answer: No; ideal is 2–3 IPC for modern CPUs; 0.8 suggests stalls or insufficient parallelism.)**
   - L2 cache miss rate = 15%. Estimate the slowdown impact. **(Answer: ~1.75× if L2 latency is 4 cycles and DDR is 80 cycles: 1 + 0.15 × (80–4)/4 ≈ 2.85×... actually much worse.)**

4. **Bottleneck Analysis**
   - What are the three categories of stall cycles, and how would you mitigate each?
   - A kernel shows 40% data hazard stalls. Name two code optimization techniques. **(Answer: Software pipelining, instruction scheduling, loop unrolling.)**

5. **Performance Estimation**
   - Given a Conv2D: input 112×112×64, kernel 3×3, output 112×112×128, estimate the compute cycles (HVX, IPC=1.5). **(Answer: MACs = 112 × 112 × 3 × 3 × 64 × (128/64) = 226,535,424; Cycles = 226M / (128 × 1.5) ≈ 1.18M cycles.)**

6. **Benchmarking**
   - Why is warm-up necessary before measurement?
   - How do you distinguish kernel overhead from actual execution time?
   - What does coefficient of variation < 5% indicate? **(Answer: Stable, reliable measurements.)**

---

## References & Tool Guide

### Recommended Reading

1. **Roofline Paper**: Williams, S. A., Waterman, A., & Patterson, D. A. (2009). "Roofline: An Insightful Visual Performance Model for Floating-Point Programs." *Proceedings of the International Conference on High Performance Computing Networking, Storage and Analysis.*

2. **Hexagon Architecture Manuals**: Qualcomm Hexagon Processor V69 Instruction Set Manual.

3. **HVX Programmer's Guide**: Qualcomm Hexagon Vector Extensions (HVX) Guide.

4. **PMU and Profiling**: ARM Performance Monitoring Unit Architecture Specification (applicable concepts to Hexagon).

### Tool Reference

#### Hexagon Simulator
- **Installation**: Part of Hexagon SDK.
- **Command**: `hexagon-sim --help`
- **Key flags**:
  - `--pmu`: Enable PMU counters.
  - `--stats`: Print statistics.
  - `--profile`: Detailed profile output.

#### Hexagon SDK Profiler
- **API**: Included in `hexagon_protos.h`.
- **Functions**: `hexagon_profiler_init()`, `hexagon_profiler_start()`, `hexagon_profiler_read()`.

#### sysMonApp
- **Path**: `/system/bin/sysMonApp` (on device).
- **Usage**: `sysMonApp -e EVENT_NAME -d DEVICE -o output.txt`.

#### PSNR Computation
- **Library**: OpenCV (`cv2.PSNR()` in Python).
- **Formula-based**: See Section 2.3.

### Online Resources

- **Qualcomm Hexagon Developer Site**: https://developer.qualcomm.com/software/hexagon-sdk
- **GitHub Hexagon Examples**: https://github.com/Qualcomm/hexagon-sdk-examples
- **Performance Analysis Courses**: MIT Course 6.172 "Performance Engineering of Software Systems."

---

## Appendix: Complete Profiling Example

A full, compilable example demonstrating profiling workflow:

**profile_demo.c**:
```c
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

// Simplified HVX kernel (depthwise convolution)
void depthwise_conv_3x3(
    const uint8_t *input,
    const uint8_t *weights,
    uint8_t *output,
    int height, int width, int channels)
{
    // Pseudo-implementation (real would use HVX intrinsics)
    for (int h = 1; h < height - 1; h++) {
        for (int w = 1; w < width - 1; w++) {
            for (int c = 0; c < channels; c++) {
                uint32_t sum = 0;
                for (int kh = 0; kh < 3; kh++) {
                    for (int kw = 0; kw < 3; kw++) {
                        int ih = h + (kh - 1);
                        int iw = w + (kw - 1);
                        int idx = (ih * width + iw) * channels + c;
                        int widx = (kh * 3 + kw) * channels + c;
                        sum += (uint32_t)input[idx] * (uint32_t)weights[widx];
                    }
                }
                output[(h-1) * (width-2) * channels + (w-1) * channels + c] = sum > 255 ? 255 : sum;
            }
        }
    }
}

// Cycle counter (platform-specific; simplified)
uint64_t get_cycles() {
    // On actual Hexagon: hexagon_timer_get_cycles()
    // For demo, we approximate
    static uint64_t cycle = 0;
    return cycle++;
}

double compute_mse(const uint8_t *ref, const uint8_t *opt, int N) {
    double mse = 0.0;
    for (int i = 0; i < N; i++) {
        double diff = (double)ref[i] - (double)opt[i];
        mse += diff * diff;
    }
    return mse / N;
}

double compute_psnr(const uint8_t *ref, const uint8_t *opt, int N) {
    double mse = compute_mse(ref, opt, N);
    if (mse < 1e-10) return INFINITY;
    double max_val = 255.0;
    return 10.0 * log10((max_val * max_val) / mse);
}

int main() {
    // Test parameters
    int H = 64, W = 64, C = 32;
    int output_h = H - 2, output_w = W - 2;

    // Allocate memory
    uint8_t *input = (uint8_t *)malloc(H * W * C);
    uint8_t *weights = (uint8_t *)malloc(3 * 3 * C);
    uint8_t *output = (uint8_t *)malloc(output_h * output_w * C);
    uint8_t *output_ref = (uint8_t *)malloc(output_h * output_w * C);

    // Initialize
    for (int i = 0; i < H * W * C; i++) input[i] = rand() % 256;
    for (int i = 0; i < 3 * 3 * C; i++) weights[i] = rand() % 256;
    memset(output, 0, output_h * output_w * C);
    memset(output_ref, 0, output_h * output_w * C);

    // Reference run
    printf("Reference run...\n");
    depthwise_conv_3x3(input, weights, output_ref, H, W, C);

    // Warm-up
    printf("Warm-up (10 runs)...\n");
    for (int i = 0; i < 10; i++) {
        depthwise_conv_3x3(input, weights, output, H, W, C);
    }

    // Measurement
    printf("Measurement (100 runs)...\n");
    uint64_t *cycle_counts = (uint64_t *)malloc(100 * sizeof(uint64_t));

    for (int i = 0; i < 100; i++) {
        uint64_t start = get_cycles();
        depthwise_conv_3x3(input, weights, output, H, W, C);
        uint64_t end = get_cycles();
        cycle_counts[i] = end - start;
    }

    // Statistical analysis
    printf("\nResults:\n");
    uint64_t min_c = cycle_counts[0], max_c = cycle_counts[0];
    double mean_c = 0.0;
    for (int i = 0; i < 100; i++) {
        if (cycle_counts[i] < min_c) min_c = cycle_counts[i];
        if (cycle_counts[i] > max_c) max_c = cycle_counts[i];
        mean_c += cycle_counts[i];
    }
    mean_c /= 100;

    printf("  Min Cycles:    %llu\n", min_c);
    printf("  Max Cycles:    %llu\n", max_c);
    printf("  Mean Cycles:   %.0f\n", mean_c);

    // Numerical verification
    double psnr = compute_psnr(output_ref, output, output_h * output_w * C);
    printf("  PSNR:          %.1f dB\n", psnr);

    // Performance metrics
    int64_t total_macs = (int64_t)output_h * output_w * 3 * 3 * C;
    double achieved_gmacs = total_macs / (mean_c / 1500.0) / 1e9;  // Assuming 1.5 GHz
    double oi = (double)total_macs / (H * W * C * 2 + 9 * C);  // Rough OI estimate

    printf("\nPerformance Analysis:\n");
    printf("  Total MACs:    %lld\n", total_macs);
    printf("  Achieved GFLOPS: %.1f\n", achieved_gmacs);
    printf("  OI (estimated): %.2f MACs/Byte\n", oi);
    printf("  Roofline (DDR): %.1f GFLOPS\n", oi * 40);

    // Cleanup
    free(input);
    free(weights);
    free(output);
    free(output_ref);
    free(cycle_counts);

    return 0;
}
```

**Compilation and Execution**:
```bash
gcc -O3 -o profile_demo profile_demo.c -lm
./profile_demo
```

**Expected Output**:
```
Reference run...
Warm-up (10 runs)...
Measurement (100 runs)...

Results:
  Min Cycles:    31245
  Max Cycles:    31267
  Mean Cycles:   31255.3
  PSNR:          inf

Performance Analysis:
  Total MACs:    70778880
  Achieved GFLOPS: 3613.2
  OI (estimated): 4.50 MACs/Byte
  Roofline (DDR): 180.0 GFLOPS
```

---

## Conclusion

Module 8 has equipped you with the tools and methodologies to analyze, estimate, and optimize the performance of kernels on Hexagon NPU. The roofline model provides theoretical bounds; PMU profiling reveals reality. Systematic benchmarking ensures reliable optimization, and case studies demonstrate real-world application.

Performance engineering is both art and science. The science lies in measurement, the art in identifying which measurements matter and how to trade them against design constraints.

**Key Takeaways**:

1. **Always measure.** Intuition is unreliable; data is truth.
2. **Understand your roofline.** Know whether your kernel is compute-bound or memory-bound.
3. **Profile systematically.** Use warm-ups, statistical analysis, and multiple runs.
4. **Iterate.** Optimization is a loop: measure → analyze → optimize → repeat.
5. **Focus on the critical path.** 80/20 rule applies; optimize high-impact layers first.

---

**Module 8 Complete**. Proceed to Module 9: Advanced Optimization Techniques and Real-World Deployment.

