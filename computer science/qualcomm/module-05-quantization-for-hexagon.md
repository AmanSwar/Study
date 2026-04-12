# Module 5: Quantization for Hexagon NPU

**Curriculum Level:** PhD
**Version:** 1.0
**Last Updated:** March 2026
**Prerequisite Modules:** Module 1-4 (Architecture, HVX, HMX, Compiler Basics)

---

## Table of Contents

1. [Introduction](#introduction)
2. [Section 1: Why INT8/INT4 Quantization is Essential](#section-1-why-int8int4-quantization-is-essential)
3. [Section 2: Per-Tensor vs Per-Channel Quantization](#section-2-per-tensor-vs-per-channel-quantization)
4. [Section 3: Symmetric vs Asymmetric Quantization](#section-3-symmetric-vs-asymmetric-quantization)
5. [Section 4: Requantization](#section-4-requantization)
6. [Section 5: QDQ Node Fusion](#section-5-qdq-node-fusion)
7. [Section 6: Mixed Precision](#section-6-mixed-precision)
8. [Section 7: Practical Pipeline - PyTorch/ONNX to Quantized Hexagon](#section-7-practical-pipeline)
9. [Advanced Topics](#advanced-topics)
10. [Self-Assessment](#self-assessment)
11. [References](#references)

---

## Introduction

Quantization is the cornerstone of efficient deep learning inference on edge NPUs. The Hexagon processor, when paired with HVX (Hexagon Vector eXtensions) and HMX (Hexagon Matrix eXtensions), achieves massive throughput improvements through INT8 and INT4 quantization. This module covers the mathematical foundations, hardware-level optimizations, and end-to-end pipeline for deploying quantized models on Hexagon.

**Why Quantization Matters for Hexagon:**
- INT8 multiply-accumulate (MAC) operates natively on HVX
- **2–4× throughput** improvement vs FP32 on same hardware
- **4× memory bandwidth** reduction (4 bytes → 1 byte per value)
- **75% power reduction** in data movement
- Critical for mobile/edge deployment with thermal constraints

By the end of this module, you will:
- Understand the mathematical theory behind quantization schemes
- Implement per-channel and per-tensor quantization
- Write HVX kernels that perform requantization in fixed-point
- Fuse QDQ nodes during compilation
- Deploy mixed-precision models
- Build a complete PyTorch → Quantized Hexagon pipeline

---

## Section 1: Why INT8/INT4 Quantization is Essential

### 1.1 Throughput Comparison on Hexagon

Hexagon's HVX and HMX architectures differ fundamentally in their handling of data widths. Understanding these differences is critical for quantization strategy.

#### HVX (Vector Extensions)

HVX operates on 128-byte (1024-bit) vectors, with native support for operations on multiple data widths:

**INT8 Operations:**
- **Width:** 128 bytes of INT8 = 128 elements
- **Peak Throughput:** vmpyi (multiply with fixed-point) @ 1 MAC per cycle per lane
- **Effective Throughput:** 128 MACs/cycle for matrix operations
- **Latency:** 5 cycles (vmpyi) + 2 cycles (accumulate)

**FP16 Operations:**
- **Width:** 128 bytes / 2 = 64 elements
- **Peak Throughput:** vmpyh (FP16 multiply) @ ~0.5 MACs per cycle per lane
- **Effective Throughput:** 64 FP16 MACs/cycle
- **Latency:** 6 cycles + variable accumulate overhead

**FP32 Operations:**
- **Width:** 128 bytes / 4 = 32 elements
- **Peak Throughput:** vmpye+vmpyo @ 0.5 MACs per cycle
- **Effective Throughput:** 32 FP32 MACs/cycle
- **Latency:** 8 cycles

**Throughput Ratio (Theoretical):**
```
INT8 : FP16 : FP32 = 128 : 64 : 32 = 4x : 2x : 1x
```

#### HMX (Matrix Extensions)

HMX provides specialized matrix multiplication units that accelerate INT8 matrix products:

**INT8 GEMM Performance:**
- 512 MACs per cycle @ up to 800 MHz
- Peak: 409.6 GOPS (INT8)
- 2D matrix accumulation with shape optimization
- Automatic tiling for large matrices

**Matrix Shapes Supported:**
- 128×4 × 4×128 → 128×128 (most efficient)
- Variable shapes via multi-iteration tiling

**Bandwidth vs Compute:**
- INT8: requires 1/4 the bandwidth of FP32 for same compute
- Memory-compute ratio: 2 bytes per MAC (vs 8 for FP32)

### 1.2 Memory Bandwidth Savings

**Bandwidth Analysis (ResNet-50 inference):**

| Precision | Weight Size | Activation Size | Cache Pressure | Bus Traffic |
|-----------|------------|-----------------|-----------------|------------|
| FP32      | 102 MB     | 54 MB (batch=1) | High (TLB miss) | 156 MB/frame |
| FP16      | 51 MB      | 27 MB           | Medium          | 78 MB/frame |
| INT8      | 25.5 MB    | 13.5 MB         | Low             | 39 MB/frame |

**Bandwidth Utilization @ 800 MHz, 64-bit DDR4:**
- FP32: 156 MB/frame × 30 fps = 4.68 GB/s (out of 12.8 GB/s = 36% utilization)
- INT8: 39 MB/frame × 30 fps = 1.17 GB/s (out of 12.8 GB/s = 9% utilization)
- **Headroom for other tasks: 91%** (vs 64% for FP32)

### 1.3 Power Efficiency

Power consumption in inference breaks down as:

$$P_{total} = P_{compute} + P_{memory} + P_{interconnect}$$

**For a single convolution layer (224×224→56×56, 64 channels):**

| Component | FP32 | INT8 |
|-----------|------|------|
| Compute (nJ/MAC) | 0.8 | 0.3 |
| Memory (nJ/byte) | 1.2 | 1.2 |
| Total (μJ/frame) | 18.4 | 4.6 |

**Key Insight:** Memory energy dominates. INT8 reduces memory footprint by 4×, yielding ~75% power reduction when memory+compute are balanced.

### 1.4 Hexagon's Native INT8 Multiply-Accumulate

Hexagon's integer ISA is optimized for INT8 operations:

**HVX Intrinsics:**
```c
// INT8 × INT8 → INT16 (with saturation)
HVX_Vector vmpyi(HVX_Vector a, HVX_Vector b)
  // a, b: 128 bytes of INT8 each
  // returns: 64 INT16 accumulation results
  // Latency: 5 cycles
```

**Fused Operations:**
```c
// Multiply-accumulate: acc += a × b
HVX_Vector vmpyiacc(HVX_Vector acc, HVX_Vector a, HVX_Vector b)
  // Fused operation, saves one register read
  // Latency: 6 cycles (overlap with previous instruction)
```

**Why This Matters:**
- No floating-point unit overhead
- All arithmetic in integer ALU (lower power)
- Native support for scale+shift post-processing
- Perfect alignment with quantized neural networks

---

## Section 2: Per-Tensor vs Per-Channel Quantization

### 2.1 Mathematical Formulation

Quantization maps a floating-point tensor to an integer tensor via a linear transformation.

#### Definition: Uniform Quantization

Given a float tensor **x** ∈ ℝ^{d₁×d₂×...}, we compute:

$$\tilde{x}_{q} = \text{round}\left(\frac{x}{s} - z\right)$$

where:
- **s** = scale (float, typically 1e-3 to 1e-2)
- **z** = zero-point (int, typically 0 for symmetric, -128 to 127 for asymmetric)
- **x̃ₑ** = quantized value ∈ {-128, ..., 127} for INT8

**Dequantization:**
$$\hat{x} = s(\tilde{x}_{q} + z)$$

#### Per-Tensor Quantization

Single scale and zero-point for entire tensor:

$$\tilde{X}^{(per\text{-}tensor)} = \text{round}\left(\frac{X}{s_{global}} - z_{global}\right)$$

**Calibration Strategy (min-max):**
$$s = \frac{\max(X) - \min(X)}{255}$$
$$z = \text{round}\left(-\frac{\min(X)}{s}\right)$$

**Advantages:**
- Minimal memory overhead (one scale, one zero-point)
- No per-channel scale arrays
- Simpler HVX kernel implementation
- Faster quantization/dequantization

**Disadvantages:**
- **Poor accuracy** for layers with high variance across channels
- Example: A conv layer may have activation ranges [0.1, 2.5] — one channel dominates
- Typically causes **1–3% accuracy loss** on ResNet-50

#### Per-Channel Quantization

Independent scale for each output channel (for weights) or input channel (for activations):

$$\tilde{X}^{(per\text{-}channel)} = \text{round}\left(\frac{X}{s_{c}} - z_{c}\right), \quad c = 1, \ldots, C$$

where each channel c has its own scale **s_c** and zero-point **z_c**.

**Calibration:**
```python
# For weight tensor of shape (out_channels, in_channels, kh, kw)
for c in range(out_channels):
    X_c = weights[c, :, :, :]  # Shape: (in_channels, kh, kw)
    s_c = (max(X_c) - min(X_c)) / 255
    z_c = round(-min(X_c) / s_c)
    scales[c] = s_c
    zero_points[c] = z_c
```

**Advantages:**
- **Preserves per-channel characteristics**
- Typical accuracy loss: **0.1–0.5%** (vs 1–3% for per-tensor)
- Standard in modern frameworks (PyTorch, TensorFlow)

**Disadvantages:**
- **Scale/zero-point arrays required:** O(C) memory
- HVX kernel must apply per-channel scales during inference
- More complex post-processing

### 2.2 Impact on HVX Kernel Design

#### Per-Tensor Kernel (Simple)

```c
// Simple per-tensor INT8 matrix multiplication
// C += A × B (all INT8)
// with single scale factor

#define BLOCKSIZE 128  // HVX vector size

void gemm_int8_per_tensor(
    int8_t *A,      // M × K matrix
    int8_t *B,      // K × N matrix
    int8_t *C,      // M × N output
    int M, int K, int N,
    float scale     // Single scale for all elements
) {
    for (int i = 0; i < M; i += BLOCKSIZE) {
        for (int j = 0; j < N; j += BLOCKSIZE) {
            for (int k = 0; k < K; k += BLOCKSIZE) {
                // Load vectors
                HVX_Vector *A_ptr = (HVX_Vector*)(A + i*K + k);
                HVX_Vector *B_ptr = (HVX_Vector*)(B + k*N + j);
                HVX_Vector acc = Q6_V_vsplat_R(0);  // Clear accumulator

                // Multiply-accumulate
                for (int kk = 0; kk < BLOCKSIZE; kk++) {
                    HVX_Vector a = A_ptr[kk];
                    HVX_Vector b = B_ptr[kk];
                    acc = Q6_Vhw_vacc_VxVxV(acc, a, b);  // 128 MACs
                }

                // Store result (no per-channel scale overhead)
                HVX_Vector *C_ptr = (HVX_Vector*)(C + i*N + j);
                *C_ptr = acc;
            }
        }
    }
}
```

**Kernel Characteristics:**
- **Register count:** 3 (A, B, acc)
- **Latency:** 5 cycles per iteration
- **Throughput:** 128 MACs/cycle (full HVX utilization)

#### Per-Channel Kernel (Complex)

```c
// Per-channel quantized convolution
// For each output channel, apply independent scale

void conv2d_int8_per_channel(
    int8_t *input,          // H × W × C_in
    int8_t *weights,        // C_out × C_in × kh × kw
    int8_t *output,         // H' × W' × C_out
    float *scales,          // C_out scale factors
    int8_t *zero_points,    // C_out zero-points
    int H, int W, int C_in, int C_out, int kh, int kw
) {
    for (int c_out = 0; c_out < C_out; c_out++) {
        float scale = scales[c_out];
        int8_t zp = zero_points[c_out];

        for (int h = 0; h < H - kh + 1; h += 8) {  // Process 8 rows at a time
            for (int w = 0; w < W - kw + 1; w += 8) {
                // Load scale into vector register (replicate across 128 bytes)
                HVX_Vector scale_vec = Q6_Vw_vasr_VwR(
                    Q6_V_vsplat_R(*(int32_t*)&scale),  // Bitcast float to int32
                    8
                );

                // Convolution for this output channel
                int32_t acc = 0;
                for (int kh_idx = 0; kh_idx < kh; kh_idx++) {
                    for (int kw_idx = 0; kw_idx < kw; kw_idx++) {
                        for (int c_in = 0; c_in < C_in; c_in++) {
                            int8_t w_val = weights[c_out * C_in * kh * kw +
                                                    kh_idx * kw * C_in +
                                                    kw_idx * C_in + c_in];
                            int8_t in_val = input[(h + kh_idx) * W * C_in +
                                                  (w + kw_idx) * C_in + c_in];
                            acc += w_val * in_val;  // INT8 × INT8 → INT32
                        }
                    }
                }

                // Apply per-channel scale
                // Requantize: acc × scale → INT8
                // (Covered in Section 4)
                output[h * W * C_out + w * C_out + c_out] =
                    (int8_t)CLAMP(acc * scale / 256, -128, 127);
            }
        }
    }
}
```

**Kernel Characteristics:**
- **Register count:** 5+ (scale_vec, zp_vec, acc, weight, input)
- **Memory bandwidth:** Must load scales/zero-points per channel
- **Throughput loss:** ~10–15% due to scalar scale lookups
- **Cache pressure:** Scale arrays (C_out × 4 bytes) may miss L1

### 2.3 Accuracy Tradeoffs: Theoretical Analysis

**Quantization Error Bound (per-tensor):**

For uniform quantization with range [min(x), max(x)]:

$$\epsilon_{per\text{-}tensor} = \max_i |x_i - \hat{x}_i| \leq \frac{s}{2}$$

where s is the quantization step. For high-variance channels:

$$s = \frac{\text{max}_c \max(X_c) - \text{min}_c \min(X_c)}{255}$$

This single scale is determined by the **worst-case** channel, forcing all other channels to use coarser quantization.

**Quantization Error Bound (per-channel):**

$$\epsilon_{per\text{-}channel} = \max_c \max_i |x_{c,i} - \hat{x}_{c,i}| \leq \frac{s_c}{2}$$

where each channel has:

$$s_c = \frac{\max(X_c) - \min(X_c)}{255}$$

**Relative Error Reduction:**

$$\frac{\epsilon_{per\text{-}channel}}}{\epsilon_{per\text{-}tensor}} = \frac{\text{avg range per channel}}{\max range across channels}$$

**Typical values:**
- ConvNets: 0.2–0.5 (per-channel is 2–5× better)
- Transformers: 0.1–0.3 (per-channel is 3–10× better due to attention)

### 2.4 When to Use Each

**Per-Tensor:**
- First/last layers (accuracy < 0.5% loss acceptable)
- Highly symmetric weight distributions (rare)
- Extremely latency-critical kernels

**Per-Channel:**
- Production models (best practice)
- Fine-tuning with QAT
- All intermediate layers of deep networks

---

## Section 3: Symmetric vs Asymmetric Quantization

### 3.1 Zero-Point Handling

#### Symmetric Quantization

Symmetric quantization assumes the distribution is centered around zero:

$$\tilde{x}_{q} = \text{round}\left(\frac{x}{s}\right), \quad z = 0$$

**Range:** ℤ ∩ [-127, 127]

**Calibration:**
$$s = \frac{\max(|x|)}{127}$$

**Dequantization:**
$$\hat{x} = s \cdot \tilde{x}_{q}$$

**Advantages:**
- **Zero-point = 0:** No subtraction needed before multiply
- **Mathematical simplicity:** Odd symmetry in quantization
- **HVX efficiency:** vmpyi can directly multiply quantized values

**Disadvantages:**
- Assumes symmetric distribution
- Poor for naturally asymmetric activations (ReLU outputs ≥ 0)

#### Asymmetric Quantization

Asymmetric quantization handles arbitrary distributions:

$$\tilde{x}_{q} = \text{round}\left(\frac{x}{s} - z\right), \quad z \neq 0$$

**Range:** ℤ ∩ [-128, 127] (full INT8 range)

**Calibration:**
$$s = \frac{\max(x) - \min(x)}{255}$$
$$z = \text{round}\left(\frac{-\min(x)}{s}\right)$$

**Dequantization:**
$$\hat{x} = s(\tilde{x}_{q} + z)$$

**Advantages:**
- **Handles asymmetric distributions** perfectly
- **Reduced clipping:** Full INT8 range utilized
- **Better accuracy:** Especially for post-ReLU activations

**Disadvantages:**
- **Zero-point subtraction:** Adds computation in matrix multiply
- **Extra storage:** Need to store z per-tensor or per-channel
- **HVX complexity:** Requires zero-point adjustment

### 3.2 Why Asymmetric for Activations

**ReLU Post-Activation Distribution:**

After ReLU in Conv2D:
- **Min value:** 0 (negative values clipped)
- **Max value:** varies (e.g., 4.3)
- **Distribution:** Skewed toward 0

**Symmetric Quantization (Bad):**
$$s_{sym} = \frac{\max(|x|)}{127} = \frac{4.3}{127} \approx 0.0338$$

Range covered: [-4.3, 4.3]
**Problem:** Negative range unused! (No negative activations post-ReLU)

**Effective resolution:** Only [0, 4.3] covered by quantizer
- Only ~64 distinct quantization levels for positive values (out of 128)
- Relative error: **2× worse** than potential

**Asymmetric Quantization (Good):**
$$s_{asym} = \frac{\max(x) - \min(x)}{255} = \frac{4.3 - 0}{255} \approx 0.0169$$

Range covered: [0, 4.3]
**Benefit:** All 256 quantization levels used efficiently

**Relative error improvement:** 2× (i.e., **half the quantization error**)

### 3.3 Zero-Point Effects in HVX Kernels

#### Symmetric Kernel (Simple)

```c
// Simple vmpyi without zero-point adjustment
// output = (A * B) >> scale_shift

HVX_Vector output = Q6_Vh_vmpyacc_VhVhVh_sat(
    init_acc,   // Accumulator
    a,          // INT8 weights
    b           // INT8 activations
    // No zero-point subtraction!
);
```

**Cycles:** 5 (vmpyi latency)

#### Asymmetric Kernel (Complex)

For asymmetric quantization, we must compute:

$$\tilde{Y} = \tilde{X} \times \tilde{W} - z_x \sum_k \tilde{W}_k - z_w \sum_k \tilde{X}_k + z_x z_w N$$

where:
- **z_x** = activation zero-point
- **z_w** = weight zero-point
- **N** = number of summands (e.g., input channels)

**Implementation Strategy:**

```c
// Asymmetric INT8 matrix multiply with zero-point adjustment
// C = A * B - z_a * sum(B) - z_b * sum(A)

void gemm_int8_asymmetric(
    int8_t *A,                  // M × K weights
    int8_t *B,                  // K × N activations
    int32_t *C,                 // M × N output (int32 accumulator)
    float scale_a, float scale_b, // Quantization scales
    int8_t zero_a, int8_t zero_b, // Zero-points
    int M, int K, int N
) {
    // Precompute column sums: sum_B[j] = sum_k B[k,j]
    int32_t *sum_B = malloc(N * sizeof(int32_t));
    for (int j = 0; j < N; j++) {
        int32_t sum = 0;
        for (int k = 0; k < K; k++) {
            sum += B[k * N + j];
        }
        sum_B[j] = sum;
    }

    // Main multiply-accumulate
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            int32_t acc = 0;

            // A[i,k] * B[k,j] summed over k
            for (int k = 0; k < K; k++) {
                acc += (int32_t)A[i * K + k] * (int32_t)B[k * N + j];
            }

            // Subtract zero-point contributions
            int32_t sum_A_row = 0;
            for (int k = 0; k < K; k++) {
                sum_A_row += A[i * K + k];
            }

            acc -= zero_b * sum_A_row;    // z_b * sum(A_i)
            acc -= zero_a * sum_B[j];     // z_a * sum(B_j)
            acc += zero_a * zero_b * K;   // z_a * z_b * K

            C[i * N + j] = acc;
        }
    }

    free(sum_B);
}
```

**HVX Implementation with VMPYI:**

```c
// Vectorized asymmetric matrix multiply (HVX 128-byte vectors)
// Process 128 elements of A and B in parallel

#define VLEN 128  // HVX vector length in bytes

void gemm_hvx_int8_asymmetric(
    int8_t *A,              // Weights (M × K)
    int8_t *B,              // Activations (K × N)
    int32_t *C,             // Output (M × N)
    int8_t zero_a, int8_t zero_b,
    int M, int K, int N
) {
    // Precompute zero-point correction vectors
    HVX_Vector zero_a_vec = Q6_V_vsplat_B(zero_a);
    HVX_Vector zero_b_vec = Q6_V_vsplat_B(zero_b);

    // Row-wise computation
    for (int i = 0; i < M; i++) {
        // Precompute sum of A_i (contribution to zero-point correction)
        int32_t sum_A_i = 0;
        for (int k = 0; k < K; k++) {
            sum_A_i += A[i * K + k];
        }

        for (int j = 0; j < N; j += VLEN) {
            HVX_Vector acc = Q6_V_vsplat_R(0);

            // Main multiply-accumulate loop
            for (int k = 0; k < K; k++) {
                HVX_Vector a_val = Q6_V_vsplat_B(A[i * K + k]);  // Broadcast A[i,k]
                HVX_Vector b_vec = *(HVX_Vector*)(B + k * N + j);  // Load B[k, j:j+128]

                // Multiply: result is 128 INT8s becoming 64 INT16s
                HVX_Vector prod = Q6_Vh_vmpy_VbVb(a_val, b_vec);

                // Accumulate
                acc = Q6_Vw_vacc_VwVh(acc, prod);
            }

            // Zero-point correction: acc -= zero_b * sum(A_i)
            int32_t corr_1 = zero_b * sum_A_i;
            HVX_Vector corr_vec_1 = Q6_V_vsplat_R(corr_1);
            acc = Q6_Vw_vsub_VwVw(acc, corr_vec_1);

            // Zero-point correction: acc -= zero_a * sum(B[k,j:j+128])
            // (sum_B precomputed or computed dynamically)

            // Store result
            *(HVX_Vector*)(C + i * N + j) = acc;
        }
    }
}
```

**Latency Analysis:**
- Main multiply: 5 cycles (vmpyi)
- Zero-point correction: 2 cycles (vector subtraction)
- Memory: 3 cycles (load B)
- **Total latency: 10 cycles** (vs 5 for symmetric)
- **Throughput loss: 2×** due to zero-point arithmetic

### 3.4 Symmetric Simplifications for Weights

**Observation:** Weights have symmetric or near-symmetric distributions.

Why?
- Weight initialization (Kaiming, Xavier) is typically symmetric
- Training with SGD preserves symmetry
- No ReLU applied to weights (unlike activations)

**Strategy:** Use symmetric quantization for weights, asymmetric for activations.

**Benefits:**
- Weights: No zero-point computation (s_w multiplies by zero)
- Activations: Full asymmetric precision
- **Latency:** Only need zero-point correction once per (activation channel) × (weight channel)
- **Typical speedup: 1.5–2×** vs fully asymmetric

**Standard Configuration:**
```
Weights:     INT8 symmetric (scale only)
Activations: INT8 asymmetric (scale + zero-point)
```

---

## Section 4: Requantization

### 4.1 Full Mathematics of Requantization

Requantization is the process of converting accumulator output from one quantization scale to another.

**Context:** After multiplying INT8 values, the result is INT16 (or wider). We must scale back to INT8 for next layer.

#### Problem Setup

- **Accumulator:** 32-bit INT32 accumulation of INT8 × INT8 products
- **Input scale:** s_in (from previous layer quantization)
- **Output scale:** s_out (desired quantization scale)
- **Scale ratio:** M = s_out / s_in

**Dequantization (INT32 → FP32):**
$$\hat{y} = s_{in} \cdot acc$$

**Requantization (FP32 → INT8 with new scale):**
$$\tilde{y}_{out} = \text{round}\left(\frac{\hat{y}}{s_{out}}\right) = \text{round}\left(\frac{s_{in}}{s_{out}} \cdot acc\right) = \text{round}(M \cdot acc)$$

where **M = s_in / s_out** (note: often M < 1).

#### Fixed-Point Representation

Since M is a float, but we need integer arithmetic:

$$M = \frac{m}{2^{n}} = m \cdot 2^{-n}$$

where:
- **m** = mantissa (typically 30–32 bits)
- **n** = exponent (number of right-shifts)

**Example:**
- M = 0.0078 (typical INT8 scale for ResNet)
- m = 502 (32-bit), n = 16
- Verify: 502 / 2^16 = 502 / 65536 ≈ 0.00766 ✓

#### Requantization in Fixed-Point

Given accumulator acc (INT32) and (m, n):

$$\text{requant}(acc, m, n) = \text{round}\left(\frac{acc \times m}{2^n}\right)$$

**Rounding Strategy (Banker's Rounding):**
$$\text{result} = \left\lfloor \frac{acc \times m}{2^n} + 0.5 \right\rfloor$$

In binary:
$$\text{result} = \frac{(acc \times m) + (1 \ll (n-1))}{2^n}$$

where `(1 << (n-1))` is the rounding correction.

### 4.2 Hexagon HVX Implementation: vmpye + vmpyo + vasr

Hexagon's HVX provides specialized instructions for efficient fixed-point requantization:

- **vmpye:** Vector multiply producing even-indexed results
- **vmpyo:** Vector multiply producing odd-indexed results
- **vasr:** Vector arithmetic right shift

#### Multiplication Decomposition

For 32-bit × 32-bit multiply to preserve full precision, HVX uses:

```
vmpyei(a, b) = (a[0]*b[0]) >> 32, (a[2]*b[2]) >> 32, ...
vmpyoi(a, b) = (a[1]*b[1]) >> 32, (a[3]*b[3]) >> 32, ...
```

This allows computing acc × m in 64-bit precision while storing results separately.

#### Step-by-Step Requantization

```c
// Requantize: convert INT32 acc (with scale s_in) to INT8 (with scale s_out)
// Given: m = mantissa, n = exponent such that M = m / 2^n

HVX_Vector requantize_hvx(
    HVX_Vector acc_even,  // 64 INT32 values (even indices)
    HVX_Vector acc_odd,   // 64 INT32 values (odd indices)
    uint32_t m,           // Mantissa
    int n                 // Right-shift amount
) {
    // Step 1: Multiply by mantissa (even lanes)
    // Result is 64-bit, keep upper 32 bits
    HVX_Vector prod_even_hi = Q6_Vw_vmpye_VwVw(acc_even, m);
    HVX_Vector prod_even_lo = Q6_Vw_vmpyo_VwVw(acc_even, m);

    // Step 2: Multiply by mantissa (odd lanes)
    HVX_Vector prod_odd_hi = Q6_Vw_vmpye_VwVw(acc_odd, m);
    HVX_Vector prod_odd_lo = Q6_Vw_vmpyo_VwVw(acc_odd, m);

    // Step 3: Add rounding correction (0.5 in fixed-point = 1 << (n-1))
    HVX_Vector round_corr = Q6_V_vsplat_R(1 << (n - 1));
    prod_even_hi = Q6_Vw_vadd_VwVw(prod_even_hi, round_corr);
    prod_odd_hi = Q6_Vw_vadd_VwVw(prod_odd_hi, round_corr);

    // Step 4: Arithmetic right shift by n bits
    HVX_Vector shifted_even = Q6_Vw_vasr_VwR(prod_even_hi, n);
    HVX_Vector shifted_odd = Q6_Vw_vasr_VwR(prod_odd_hi, n);

    // Step 5: Pack INT32 back to INT8 (with saturation)
    // 128 INT32 values → 128 INT8 values
    HVX_Vector result = Q6_Vb_vpack_VwVw_sat(shifted_even, shifted_odd);

    return result;
}
```

**Latency:**
- vmpye: 5 cycles
- vmpyo: 5 cycles
- vadd: 1 cycle
- vasr: 1 cycle
- vpack: 1 cycle
- **Total: ~10 cycles** for 128 INT8 elements

**Throughput: 12.8 INT8 MACs per cycle** (128 elements / 10 cycles)

### 4.3 Complete Code Example: Requantization in Action

```c
#include "hexagon_types.h"
#include "hvx_hexagon_protos.h"
#include "stdint.h"
#include "string.h"

// Structure to hold requantization parameters
typedef struct {
    uint32_t scale_mantissa;   // m in M = m / 2^n
    int scale_shift;           // n in M = m / 2^n
    int8_t zero_point_in;      // Zero-point of input quantization
    int8_t zero_point_out;     // Zero-point of output quantization
    int32_t accum_offset;      // Offset added after accumulation
} RequantParams;

// Quantized convolution with requantization
// Input: INT8 weights and activations
// Output: INT8 (requantized)
// Process:
//   1. Multiply INT8 weights × INT8 activations → INT32 accumulation
//   2. Subtract zero-point effects
//   3. Apply requantization scale+shift
//   4. Clamp to INT8 range
//   5. Output INT8

void conv_int8_requant_hvx(
    const int8_t *weights,           // Shape: (out_ch, in_ch, kh, kw)
    const int8_t *input,             // Shape: (H, W, in_ch)
    int8_t *output,                  // Shape: (H', W', out_ch)
    const RequantParams *requant,    // Requantization parameters
    int H, int W, int in_ch,
    int out_ch, int kh, int kw,
    int stride, int padding
) {
    int H_out = (H - kh + 2 * padding) / stride + 1;
    int W_out = (W - kw + 2 * padding) / stride + 1;

    // Process output by tiles (8×8 output elements)
    for (int out_h = 0; out_h < H_out; out_h++) {
        for (int out_w = 0; out_w < W_out; out_w++) {
            for (int out_c = 0; out_c < out_ch; out_c++) {
                // Accumulate convolution
                int32_t acc = 0;

                for (int kh_idx = 0; kh_idx < kh; kh_idx++) {
                    for (int kw_idx = 0; kw_idx < kw; kw_idx++) {
                        for (int in_c = 0; in_c < in_ch; in_c++) {
                            int in_h = out_h * stride + kh_idx - padding;
                            int in_w = out_w * stride + kw_idx - padding;

                            if (in_h >= 0 && in_h < H && in_w >= 0 && in_w < W) {
                                int8_t w = weights[out_c * in_ch * kh * kw +
                                                    kh_idx * kw * in_ch +
                                                    kw_idx * in_ch + in_c];
                                int8_t x = input[in_h * W * in_ch +
                                               in_w * in_ch + in_c];
                                acc += (int32_t)w * (int32_t)x;
                            }
                        }
                    }
                }

                // --- REQUANTIZATION SECTION ---
                // Step 1: Subtract accumulated zero-point effects
                // (This is simplified; full asymmetric would have more terms)
                acc += requant->accum_offset;

                // Step 2: Apply scale mantissa (fixed-point multiply)
                // M = mantissa / 2^shift
                // result = (acc * mantissa) >> shift + rounding
                int64_t scaled = (int64_t)acc * requant->scale_mantissa;
                int32_t rounded = (int32_t)((scaled + (1 << (requant->scale_shift - 1)))
                                            >> requant->scale_shift);

                // Step 3: Clamp to INT8 range and store
                int8_t out_val = rounded > 127 ? 127 : (rounded < -128 ? -128 : rounded);
                output[out_h * W_out * out_ch + out_w * out_ch + out_c] = out_val;
            }
        }
    }
}

// Vectorized version using HVX intrinsics (inner loop)
void gemm_int8_requant_hvx_inner(
    const int8_t *A,                 // Weights (1D batch)
    const int8_t *B,                 // Activations (1D batch)
    int8_t *C,                       // Output (1D batch)
    int K,                           // Number of multiply-accumulate terms
    const RequantParams *requant
) {
    // Process 128 INT8 elements at a time (HVX vector width)
    HVX_Vector zero = Q6_V_vsplat_R(0);
    HVX_Vector acc_even = zero;
    HVX_Vector acc_odd = zero;

    // Multiply-accumulate
    for (int k = 0; k < K; k++) {
        HVX_Vector a_vec = *(HVX_Vector*)(A + k * 128);
        HVX_Vector b_vec = *(HVX_Vector*)(B + k * 128);

        // INT8 × INT8 → INT16 with extended precision
        HVX_Vector prod = Q6_Vh_vmpy_VbVb(a_vec, b_vec);  // 64 INT16s

        // Extend to INT32 for accumulation
        // Split into even and odd INT32 accumulators
        HVX_Vector prod_even = Q6_Vw_vunpack_VhV_sat(prod);
        HVX_Vector prod_odd = Q6_Vw_vunpack_Vh_sat(prod);

        acc_even = Q6_Vw_vacc_VwVw(acc_even, prod_even);
        acc_odd = Q6_Vw_vacc_VwVw(acc_odd, prod_odd);
    }

    // --- REQUANTIZATION IN HVX ---
    // Multiply by mantissa
    HVX_Vector m_vec = Q6_V_vsplat_R(requant->scale_mantissa);
    HVX_Vector scaled_even = Q6_Vw_vmpye_VwVw(acc_even, m_vec);
    HVX_Vector scaled_odd = Q6_Vw_vmpyo_VwVw(acc_even, m_vec);

    // Add rounding correction
    HVX_Vector round = Q6_V_vsplat_R(1 << (requant->scale_shift - 1));
    scaled_even = Q6_Vw_vadd_VwVw(scaled_even, round);
    scaled_odd = Q6_Vw_vadd_VwVw(scaled_odd, round);

    // Right shift
    HVX_Vector shifted_even = Q6_Vw_vasr_VwR(scaled_even, requant->scale_shift);
    HVX_Vector shifted_odd = Q6_Vw_vasr_VwR(scaled_odd, requant->scale_shift);

    // Pack to INT8 with saturation
    HVX_Vector result = Q6_Vb_vpack_VwVw_sat(shifted_even, shifted_odd);

    // Store result
    *(HVX_Vector*)C = result;
}
```

**Performance Analysis:**
- **Multiply-accumulate:** 5 cycles per K iteration
- **Requantization:** ~10 cycles
- **Memory:** 3 cycles (load/store)
- **Total for 128-element batch:** 18 cycles
- **Throughput:** 128 / 18 ≈ 7 INT8 ops per cycle (out of peak 128 in main loop, limited by requant bandwidth)

---

## Section 5: QDQ Node Fusion

### 5.1 Quantize-Dequantize Patterns in ONNX

Neural networks exported to ONNX can include **QuantizeLinear** (Q) and **DequantizeLinear** (DQ) nodes that explicitly represent quantization.

#### ONNX QuantizeLinear Node

```
QuantizeLinear(X, scale, zero_point) → Y_int8

where:
  X ∈ ℝ            (float input)
  scale ∈ ℝ        (quantization scale)
  zero_point ∈ ℤ   (zero-point, typically -128 to 127)
  Y_int8 ∈ ℤ8      (quantized output)

formula: Y_int8 = round(X / scale + zero_point)
```

#### ONNX DequantizeLinear Node

```
DequantizeLinear(Y_int8, scale, zero_point) → X_float

formula: X_float = (Y_int8 - zero_point) * scale
```

#### Example: QDQ Pattern in ResNet-50

```
graph {
  input_img [H, W, 3] FP32
    ↓ (QuantizeLinear)
  input_q [H, W, 3] INT8        {scale: 0.0078, zp: 0}
    ↓ (Conv2d_1)
  conv1_q [56, 56, 64] INT32    {accum scale: 0.0156}
    ↓ (DequantizeLinear)
  conv1_f [56, 56, 64] FP32     {scale: 0.0156}
    ↓ (QuantizeLinear)
  conv1_q [56, 56, 64] INT8     {scale: 0.0078, zp: 0}
    ↓ (Conv2d_2)
  conv2_q [56, 56, 64] INT32
    ↓ (DequantizeLinear)
  conv2_f [56, 56, 64] FP32
    ... (repeat for many layers)
}
```

**Problem:** Every DQ→Q pair is unnecessary!
- Dequantizes to float: **4 bytes × N elements**
- Requantizes back to INT8: **1 byte × N elements**
- **3N bytes of memory bandwidth waste** per intermediate layer

### 5.2 Graph Compilation and QDQ Fusion

**Fusion Strategy:** Replace a DequantizeLinear → QuantizeLinear pair with a Requantize operation.

#### Fusion Rule

```
Input (INT8) → DequantizeLinear(scale_1, zp_1) →
               QuantizeLinear(scale_2, zp_2) → Output (INT8)

Fused:
Input (INT8) → Requantize(scale_1, scale_2, zp_1, zp_2) → Output (INT8)
```

**Requantize Formula:**

Given input x_q with scale s₁ and zero-point z₁:

$$\hat{x} = (x_q - z_1) \cdot s_1$$  (Dequantize)

$$y_q = \text{round}\left(\frac{\hat{x}}{s_2} + z_2\right)$$  (Requantize)

Substituting:

$$y_q = \text{round}\left(\frac{(x_q - z_1) \cdot s_1}{s_2} + z_2\right)$$

$$= \text{round}\left(x_q \cdot \frac{s_1}{s_2} - z_1 \cdot \frac{s_1}{s_2} + z_2\right)$$

Let M = s₁/s₂ (scale factor), then:

$$y_q = \text{round}\left(M \cdot x_q + (z_2 - M \cdot z_1)\right)$$

**Offset term:** b = z₂ - M·z₁

#### Compiler Implementation

```python
# Simplified QDQ fusion in a compiler pass

def fuse_qdq_nodes(graph):
    """
    Remove DequantizeLinear → QuantizeLinear pairs from graph
    """
    nodes_to_remove = []

    for node in graph.nodes:
        if node.op_type == "QuantizeLinear":
            # Check if input comes from a DequantizeLinear node
            input_producer = graph.get_producer(node.input[0])

            if input_producer and input_producer.op_type == "DequantizeLinear":
                # Found a QDQ pair!
                dq_node = input_producer
                q_node = node

                # Extract quantization parameters
                scale_in = graph.get_constant(dq_node.input[1])      # DQ scale
                zp_in = graph.get_constant(dq_node.input[2])         # DQ zp
                scale_out = graph.get_constant(q_node.input[1])      # Q scale
                zp_out = graph.get_constant(q_node.input[2])         # Q zp

                # Create Requantize node
                requant_node = ONNXNode(
                    op_type="Requantize",
                    inputs=[dq_node.input[0]],  # Direct input to DQ
                    outputs=[q_node.output[0]],  # Output of Q
                    attributes={
                        "scale_in": scale_in,
                        "zero_point_in": zp_in,
                        "scale_out": scale_out,
                        "zero_point_out": zp_out,
                    }
                )

                # Replace Q node with Requantize
                graph.replace_node(q_node, requant_node)

                # Mark DQ for removal (input will be used directly)
                nodes_to_remove.append(dq_node)

    # Remove unused DQ nodes
    for dq_node in nodes_to_remove:
        if len(graph.get_consumers(dq_node.output[0])) == 0:
            graph.remove_node(dq_node)

    return graph
```

### 5.3 Folding Quantization into Operator Kernels

Rather than explicit DQ→Q, many operators can fold quantization directly:

#### Conv2D with Implicit Quantization

**Traditional:** Conv(x_q) → y_int32 → DQ → y_f32 → Q → y_q

**Folded:** Conv_Requant(x_q) → y_q (single kernel)

**Implementation:**

```c
// Conv2D kernel with built-in requantization
// Fuses: Conv + DQ(scale_1) + Q(scale_2)

void conv2d_int8_fused_requant(
    const int8_t *input,           // H × W × C_in (quantized)
    const int8_t *weights,         // C_out × C_in × kh × kw (quantized)
    int8_t *output,                // H' × W' × C_out (quantized)
    const float *scales_in,        // Per-layer quantization scales
    const float *scales_out,
    const int8_t *zero_points_in,
    const int8_t *zero_points_out,
    int H, int W, int C_in, int C_out, int kh, int kw
) {
    // Precompute fixed-point requantization factors
    // M[c] = scales_in[c] / scales_out[c]
    uint32_t *mantissa = malloc(C_out * sizeof(uint32_t));
    int *shift = malloc(C_out * sizeof(int));

    for (int c = 0; c < C_out; c++) {
        float M = scales_in[c] / scales_out[c];
        // Convert to fixed-point (m, n) such that M ≈ m / 2^n
        int n = 0;
        uint32_t m = (uint32_t)((1LL << 32) * M);  // Estimate
        while (m > (1u << 31) && n < 32) {
            m >>= 1;
            n++;
        }
        mantissa[c] = m;
        shift[c] = n;
    }

    // Main convolution loop
    for (int oh = 0; oh < H - kh + 1; oh++) {
        for (int ow = 0; ow < W - kw + 1; ow++) {
            for (int c_out = 0; c_out < C_out; c_out++) {
                // Accumulate convolution
                int32_t acc = 0;

                for (int kh_idx = 0; kh_idx < kh; kh_idx++) {
                    for (int kw_idx = 0; kw_idx < kw; kw_idx++) {
                        for (int c_in = 0; c_in < C_in; c_in++) {
                            int8_t w_val = weights[c_out * C_in * kh * kw +
                                                    kh_idx * kw * C_in +
                                                    kw_idx * C_in + c_in];
                            int8_t in_val = input[(oh + kh_idx) * W * C_in +
                                                  (ow + kw_idx) * C_in + c_in];
                            acc += (int32_t)w_val * (int32_t)in_val;
                        }
                    }
                }

                // Fused DQ + Requant + Q
                // dq_val = (acc - zp_in) * scale_in
                // q_val = round(dq_val / scale_out + zp_out)
                //       = round(acc * (scale_in/scale_out) - zp_in*(scale_in/scale_out) + zp_out)

                int32_t m = mantissa[c_out];
                int n = shift[c_out];
                int8_t zp_in = zero_points_in[c_out];
                int8_t zp_out = zero_points_out[c_out];

                // Fixed-point arithmetic:
                // offset = zp_out - zp_in * M (in fixed point)
                int32_t offset = ((int32_t)zp_out << n) - zp_in * m;

                // Requantized value
                int64_t scaled = (int64_t)acc * m + offset;
                int32_t result = (scaled + (1 << (n - 1))) >> n;  // Round

                // Clamp to INT8
                int8_t out_val = (result > 127) ? 127 : ((result < -128) ? -128 : result);
                output[oh * W * C_out + ow * C_out + c_out] = out_val;
            }
        }
    }

    free(mantissa);
    free(shift);
}
```

### 5.4 Handling QDQ Across Operator Boundaries

Not all operators can fuse quantization. Example: Skip connections in ResNets.

```
Path 1:                    Path 2:
Conv → DQ → Q              Conv → (no quantization)
  ↓                          ↓
 [Need requantize]       (may have different scale)
  ↓
Add (with implicit resize)
```

**Strategy:** Insert requantization nodes at operator boundaries where scales mismatch.

**Detection Algorithm:**

```python
def insert_requant_for_incompatible_ops(graph):
    """
    Insert Requantize nodes between operators with different output scales
    """
    for node in graph.nodes:
        # For each input of this node
        for input_idx, input_name in enumerate(node.inputs):
            producer = graph.get_producer(input_name)

            if producer is None:
                continue  # Input is a graph input

            # Check if scales are compatible
            producer_scale = get_output_scale(producer)
            consumer_scale = get_input_scale(node, input_idx)

            if producer_scale != consumer_scale:
                # Insert Requantize node
                requant_node = ONNXNode(
                    op_type="Requantize",
                    inputs=[input_name],
                    outputs=[f"{input_name}_requant"],
                    attributes={
                        "scale_in": producer_scale,
                        "scale_out": consumer_scale,
                    }
                )

                # Update graph connections
                graph.insert_node(requant_node)
                node.inputs[input_idx] = f"{input_name}_requant"
```

---

## Section 6: Mixed Precision

### 6.1 INT8 Weights + INT16 Activations Strategy

Mixed precision uses different bit-widths for different data types:

**Configuration:**
- **Weights:** INT8 (4 bytes per weight, easy to store/load)
- **Activations:** INT16 (2 bytes per activation, better precision)
- **Multiply:** INT8 × INT16 → INT32 (accumulation)

**Motivation:** Some layers are more sensitive to quantization errors.

#### When to Use Mixed Precision

1. **First layer:** Processes raw image data (high bit-depth)
   - INT8 weight quantization acceptable
   - INT16 activation more robust

2. **Attention layers (Transformers):**
   - Softmax attention is sensitive to small differences
   - INT16 activations preserve attention shape
   - Example: BERT layer_norm output typically needs higher precision

3. **Skip connections with mismatched scales:**
   - Add operation: x + y (different scales)
   - INT16 provides more headroom for scale adjustment

4. **Last layer before task head:**
   - Classification requires careful feature representation
   - INT16 activations improve final accuracy

### 6.2 HVX Support for Mixed-Width Multiply

Hexagon HVX provides instructions for INT8 × INT16 → INT32:

**Instruction: vmpy (mixed-width)**

```c
// INT8 × INT16 → INT32 multiply-accumulate

HVX_Vector vmpy_int8_int16(
    HVX_Vector a_int8,   // 128 INT8 values (weights)
    HVX_Vector b_int16   // 64 INT16 values (activations)
    // Note: a_int8 has 128 elements, b_int16 has 64
    // Implicitly, adjacent pairs of INT8 multiply with single INT16
)
```

**Usage Pattern:**

```c
// Weights: INT8, 128 elements per vector
// Activations: INT16, 64 elements per vector
// Output: INT32, 64 elements per vector

void gemm_int8_int16_mixed_precision(
    const int8_t *A,             // Weights (M × K)
    const int16_t *B,            // Activations (K × N)
    int32_t *C,                  // Output accumulator (M × N)
    int M, int K, int N
) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j += 64) {  // Process 64 elements (INT16 width)
            HVX_Vector acc = Q6_V_vsplat_R(0);

            for (int k = 0; k < K; k += 128) {  // Process 128 INT8 at a time
                // Load 128 INT8 weights
                HVX_Vector a = *(HVX_Vector*)(A + i * K + k);

                // Load 64 INT16 activations (note: k must align to 64 INT16s)
                // Since each INT16 is 2 bytes: k * 2 bytes per INT16
                HVX_Vector b_0 = *(HVX_Vector*)(B + (k + 0) * N + j);
                HVX_Vector b_1 = *(HVX_Vector*)(B + (k + 64) * N + j);

                // Multiply INT8 × INT16 for first half of a
                HVX_Vector a_lo = Q6_Vb_vlut8_VbRb(a, 0x0F);  // Low byte indices
                HVX_Vector a_hi = Q6_Vb_vlut8_VbRb(a, 0xF0);  // High byte indices

                HVX_Vector prod_lo = Q6_Vw_vmpy_VhVh(
                    Q6_Vh_vsub_VhVh(a_lo, 0),  // Convert INT8 to INT16
                    b_0
                );
                HVX_Vector prod_hi = Q6_Vw_vmpy_VhVh(a_hi, b_1);

                // Accumulate
                acc = Q6_Vw_vacc_VwVw(acc, prod_lo);
                acc = Q6_Vw_vacc_VwVw(acc, prod_hi);
            }

            // Store result
            *(HVX_Vector*)(C + i * N + j) = acc;
        }
    }
}
```

**HVX Latency:**
- Load: 3 cycles
- vmpy (INT8 × INT16): 6 cycles (more complex than INT8 × INT8)
- Accumulate: 1 cycle
- **Total: 10 cycles** (vs 5 for INT8 × INT8)

### 6.3 Throughput Implications

**INT8 × INT8 (per 128-element vector):**
- Latency: 5 cycles
- Throughput: 128 MACs / 5 cycles = 25.6 MACs/cycle

**INT8 × INT16 (per 64-element output vector):**
- Latency: 6 cycles (mixed-width more complex)
- Throughput: 64 MACs / 6 cycles = 10.7 MACs/cycle

**Throughput Reduction: ~2.4×** (due to width mismatch)

**Memory Bandwidth:**
- INT8 × INT8: weights 1 byte, activations 1 byte → 2 bytes/MAC
- INT8 × INT16: weights 1 byte, activations 2 bytes → 3 bytes/MAC

**Trade-off:** Accept 2× throughput loss for better accuracy (typically 0.5–1% improvement on sensitive layers).

---

## Section 7: Practical Pipeline - PyTorch/ONNX to Quantized Hexagon

### 7.1 Step-by-Step Walkthrough

This section provides a complete, production-grade pipeline for quantizing models and deploying on Hexagon.

#### Step 1: Model Definition and Training (PyTorch)

```python
# Define a simple quantization-aware model
import torch
import torch.nn as nn
from torch.quantization import QConfig, default_weight_observer, default_activation_observer
from torch.quantization import prepare_qat, convert

class SimpleResNet(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleResNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            # Residual blocks (simplified)
            self._build_layer(64, 64, 3),
            self._build_layer(64, 128, 3, stride=2),
            self._build_layer(128, 256, 3, stride=2),
            self._build_layer(256, 512, 3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

        # Quantization stubs
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()

    def _build_layer(self, in_channels, out_channels, blocks, stride=1):
        layers = []
        layers.append(nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1))
        layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(blocks - 1):
            layers.append(nn.Conv2d(out_channels, out_channels, 3, padding=1))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.quant(x)  # Insert quantization at input
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        x = self.dequant(x)  # Insert dequantization at output
        return x

# Instantiate and load pretrained weights
model = SimpleResNet(num_classes=1000)
# model.load_state_dict(torch.load('pretrained_resnet.pth'))
model.eval()

print("Model defined. Ready for quantization.")
```

#### Step 2: Post-Training Quantization (PTQ) – Min-Max Calibration

```python
import torch
from torch.quantization import get_default_qconfig
from torch.quantization import prepare, convert

def calibrate_model_minmax(model, calibration_loader, num_batches=10):
    """
    Calibrate model statistics using min-max on calibration data.
    This is Post-Training Quantization (PTQ).
    """
    model.eval()

    # Use default quantization config (symmetric INT8 for weights, asymmetric for activations)
    qconfig = get_default_qconfig('qnnpack')  # Or 'fbgemm' for CPU
    model.qconfig = qconfig

    # Prepare model for calibration (insert quantization placeholders)
    model = prepare(model, inplace=True)

    # Run calibration on subset of data
    print("Calibrating model on {} batches...".format(num_batches))
    with torch.no_grad():
        for batch_idx, (images, _) in enumerate(calibration_loader):
            if batch_idx >= num_batches:
                break
            if torch.cuda.is_available():
                images = images.cuda()
            _ = model(images)
            if batch_idx % 5 == 0:
                print(f"  Calibrated batch {batch_idx}/{num_batches}")

    # Convert to quantized model
    model = convert(model, inplace=True)
    print("Calibration complete. Model converted to INT8.")

    return model

# Example usage
# from torchvision import datasets, transforms
# transform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
# ])
# cal_dataset = datasets.ImageNet(root='data/', split='train', transform=transform)
# cal_loader = torch.utils.data.DataLoader(cal_dataset, batch_size=32, shuffle=True)
# model = calibrate_model_minmax(model, cal_loader, num_batches=20)
```

#### Step 3: Percentile Calibration Strategy

```python
def calibrate_model_percentile(model, calibration_loader, percentile=99.9, num_batches=10):
    """
    Calibrate using percentile-based range to reduce outlier impact.

    Instead of using min/max (sensitive to outliers),
    use percentile range: [percentile_low, percentile_high]
    """

    model.eval()

    # Collect activation statistics
    activation_stats = {}  # layer_name -> [values]
    hooks = []

    def capture_activations(name):
        def hook(module, input, output):
            if isinstance(output, torch.Tensor):
                if name not in activation_stats:
                    activation_stats[name] = []
                activation_stats[name].extend(output.detach().cpu().numpy().flatten().tolist())
        return hook

    # Register hooks on all activations
    for name, module in model.named_modules():
        if isinstance(module, (nn.ReLU, nn.BatchNorm2d)):
            hook = module.register_forward_hook(capture_activations(name))
            hooks.append(hook)

    # Run calibration
    print(f"Collecting activation statistics on {num_batches} batches...")
    with torch.no_grad():
        for batch_idx, (images, _) in enumerate(calibration_loader):
            if batch_idx >= num_batches:
                break
            if torch.cuda.is_available():
                images = images.cuda()
            _ = model(images)
            if batch_idx % 5 == 0:
                print(f"  Processed batch {batch_idx}/{num_batches}")

    # Remove hooks
    for hook in hooks:
        hook.remove()

    # Compute percentile ranges
    print("Computing percentile ranges...")
    percentile_ranges = {}
    for name, values in activation_stats.items():
        low = np.percentile(values, 100 - percentile)
        high = np.percentile(values, percentile)
        percentile_ranges[name] = (low, high)
        print(f"  {name}: [{low:.4f}, {high:.4f}]")

    # Set quantization ranges manually
    # (In practice, use a custom observer that applies percentile)
    return model, percentile_ranges

import numpy as np

# model, ranges = calibrate_model_percentile(model, cal_loader, percentile=99.9)
```

#### Step 4: Quantization-Aware Training (QAT)

```python
def train_qat(model, train_loader, val_loader, epochs=10, learning_rate=1e-4):
    """
    Fine-tune quantized model (QAT) to recover accuracy.

    During QAT:
    - Weights are quantized: w_q = round(w / s_w)
    - Forward pass uses dequantized values: w_hat = s_w * w_q
    - Backprop goes through dequantization
    """

    from torch.quantization import prepare_qat, convert

    model.train()

    # Prepare model for QAT
    qconfig = get_default_qconfig('qnnpack')
    model.qconfig = qconfig
    model = prepare_qat(model, inplace=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0

    for epoch in range(epochs):
        # Training phase
        total_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (images, labels) in enumerate(train_loader):
            if torch.cuda.is_available():
                images, labels = images.cuda(), labels.cuda()

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

            if batch_idx % 50 == 0:
                print(f"Epoch {epoch} Batch {batch_idx}: Loss={loss.item():.4f}, Acc={100*correct/total:.1f}%")

        # Validation phase
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                if torch.cuda.is_available():
                    images, labels = images.cuda(), labels.cuda()
                outputs = model(images)
                _, predicted = outputs.max(1)
                val_correct += predicted.eq(labels).sum().item()
                val_total += labels.size(0)

        val_acc = 100 * val_correct / val_total
        print(f"Epoch {epoch} Validation Accuracy: {val_acc:.1f}%")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'qat_best.pth')

        model.train()

    # Convert to inference model
    model.eval()
    model = convert(model, inplace=True)

    print(f"QAT complete. Best accuracy: {best_acc:.1f}%")
    return model

# model = train_qat(model, train_loader, val_loader, epochs=10)
```

#### Step 5: Export to ONNX with QDQ Nodes

```python
import onnx
from onnx import helper, TensorProto

def export_to_onnx_with_qdq(model, input_shape, output_path='model_quantized.onnx'):
    """
    Export PyTorch model to ONNX with explicit QuantizeLinear/DequantizeLinear nodes.

    This allows Hexagon compiler to fuse QDQ nodes.
    """

    # Create dummy input
    dummy_input = torch.randn(*input_shape)
    if torch.cuda.is_available():
        dummy_input = dummy_input.cuda()
        model = model.cuda()

    # Export to ONNX
    print(f"Exporting model to {output_path}...")
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        opset_version=13,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        verbose=False,
        dynamic_axes={'input': {0: 'batch_size'},
                      'output': {0: 'batch_size'}}
    )

    print(f"Model exported to {output_path}")

    # Load and inspect ONNX model
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print("ONNX model validated successfully")

    # Print graph summary
    print("\n=== ONNX Graph Summary ===")
    for node in onnx_model.graph.node:
        print(f"  {node.op_type}: {node.input} -> {node.output}")

    return onnx_model

# export_to_onnx_with_qdq(model, input_shape=(1, 3, 224, 224))
```

#### Step 6: Custom Hexagon Quantization Config

```python
import json

class HexagonQuantizationConfig:
    """
    Configuration for Hexagon-specific quantization options.
    """

    def __init__(self):
        self.config = {
            "backend": "hexagon",
            "precision": {
                "weights": "INT8",
                "activations": "INT8",
                "accumulators": "INT32",
            },
            "per_channel": {
                "weights": True,  # Critical for accuracy
                "activations": False,  # Usually per-tensor for speed
            },
            "quantization_scheme": {
                "weights": "symmetric",    # Weights are symmetric
                "activations": "asymmetric",  # Activations are asymmetric
            },
            "calibration": {
                "method": "min-max",  # or "percentile", "entropy", "mse"
                "percentile": 99.9,
                "num_batches": 20,
            },
            "qat": {
                "enabled": True,
                "epochs": 10,
                "learning_rate": 1e-4,
                "momentum": 0.9,
            },
            "hvx_optimizations": {
                "fuse_qdq": True,         # Fuse Q-DQ nodes
                "fuse_requant": True,     # Fuse requantization into conv
                "mixed_precision_layers": [0, 1, 20, 21],  # Layers to use INT16 activations
                "skip_fusion": ["LayerNorm"],  # Layers to skip quantization
            },
            "performance": {
                "target_throughput_ops_per_cycle": 100,  # Target performance
                "max_memory_per_layer_mb": 10,
                "batch_size": 1,
            },
            "validation": {
                "test_set_size": 1000,
                "acceptable_accuracy_drop_percent": 1.0,  # Allow up to 1% accuracy loss
            }
        }

    def save(self, filepath):
        """Save config to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.config, f, indent=2)
        print(f"Config saved to {filepath}")

    def load(self, filepath):
        """Load config from JSON file."""
        with open(filepath, 'r') as f:
            self.config = json.load(f)
        print(f"Config loaded from {filepath}")

    def get(self, key):
        """Get config value by key."""
        return self.config.get(key)

    def __str__(self):
        return json.dumps(self.config, indent=2)

# Create and save config
config = HexagonQuantizationConfig()
config.save('hexagon_quant_config.json')
print(config)
```

#### Step 7: Validation and Accuracy Assessment

```python
def validate_quantized_model(original_model, quantized_model, val_loader, num_batches=None):
    """
    Compare accuracy of original vs quantized model.
    """

    original_model.eval()
    quantized_model.eval()

    orig_correct = 0
    quant_correct = 0
    total = 0

    orig_time = 0.0
    quant_time = 0.0

    import time

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(val_loader):
            if num_batches and batch_idx >= num_batches:
                break

            if torch.cuda.is_available():
                images, labels = images.cuda(), labels.cuda()

            # Original model
            start = time.time()
            orig_outputs = original_model(images)
            orig_time += time.time() - start

            # Quantized model
            start = time.time()
            quant_outputs = quantized_model(images)
            quant_time += time.time() - start

            # Accuracy
            _, orig_pred = orig_outputs.max(1)
            _, quant_pred = quant_outputs.max(1)

            orig_correct += orig_pred.eq(labels).sum().item()
            quant_correct += quant_pred.eq(labels).sum().item()
            total += labels.size(0)

            if batch_idx % 20 == 0:
                print(f"Batch {batch_idx}: "
                      f"Orig Acc={100*orig_correct/total:.1f}%, "
                      f"Quant Acc={100*quant_correct/total:.1f}%")

    orig_acc = 100 * orig_correct / total
    quant_acc = 100 * quant_correct / total
    accuracy_drop = orig_acc - quant_acc

    print("\n=== Validation Results ===")
    print(f"Original Model Accuracy: {orig_acc:.2f}%")
    print(f"Quantized Model Accuracy: {quant_acc:.2f}%")
    print(f"Accuracy Drop: {accuracy_drop:.2f}%")
    print(f"Original Model Time: {orig_time:.2f}s")
    print(f"Quantized Model Time: {quant_time:.2f}s")
    print(f"Speedup: {orig_time / quant_time:.2f}x")

    return {
        "orig_acc": orig_acc,
        "quant_acc": quant_acc,
        "accuracy_drop": accuracy_drop,
        "speedup": orig_time / quant_time,
    }

# results = validate_quantized_model(model_original, model_quantized, val_loader, num_batches=50)
```

### 7.2 End-to-End Pipeline Example

```python
def full_quantization_pipeline(model, train_loader, cal_loader, val_loader, output_dir='./quantized/'):
    """
    Complete pipeline from FP32 model to quantized Hexagon-ready model.
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    # Step 1: Save original model
    print("=" * 60)
    print("STEP 1: Saving original model")
    print("=" * 60)
    torch.save(model.state_dict(), f'{output_dir}/model_original.pth')
    model_original = SimpleResNet()
    model_original.load_state_dict(torch.load(f'{output_dir}/model_original.pth'))

    # Step 2: Post-Training Quantization (PTQ)
    print("\n" + "=" * 60)
    print("STEP 2: Post-Training Quantization (Min-Max)")
    print("=" * 60)
    model_ptq = SimpleResNet()
    model_ptq.load_state_dict(torch.load(f'{output_dir}/model_original.pth'))
    model_ptq = calibrate_model_minmax(model_ptq, cal_loader, num_batches=20)
    torch.save(model_ptq.state_dict(), f'{output_dir}/model_ptq.pth')

    # Step 3: Quantization-Aware Training (QAT)
    print("\n" + "=" * 60)
    print("STEP 3: Quantization-Aware Training (QAT)")
    print("=" * 60)
    model_qat = SimpleResNet()
    model_qat.load_state_dict(torch.load(f'{output_dir}/model_original.pth'))
    model_qat = train_qat(model_qat, train_loader, val_loader, epochs=5)
    torch.save(model_qat.state_dict(), f'{output_dir}/model_qat.pth')

    # Step 4: Export to ONNX
    print("\n" + "=" * 60)
    print("STEP 4: Export to ONNX")
    print("=" * 60)
    export_to_onnx_with_qdq(model_qat, input_shape=(1, 3, 224, 224),
                           output_path=f'{output_dir}/model_qat.onnx')

    # Step 5: Create Hexagon config
    print("\n" + "=" * 60)
    print("STEP 5: Create Hexagon Configuration")
    print("=" * 60)
    config = HexagonQuantizationConfig()
    config.save(f'{output_dir}/hexagon_config.json')

    # Step 6: Validate
    print("\n" + "=" * 60)
    print("STEP 6: Validation")
    print("=" * 60)
    results = validate_quantized_model(model_original, model_qat, val_loader, num_batches=100)

    # Summary
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"Output directory: {output_dir}")
    print(f"Files generated:")
    print(f"  - model_original.pth")
    print(f"  - model_ptq.pth")
    print(f"  - model_qat.pth")
    print(f"  - model_qat.onnx  <-- Ready for Hexagon compiler")
    print(f"  - hexagon_config.json")
    print(f"\nAccuracy Summary:")
    print(f"  Original: {results['orig_acc']:.2f}%")
    print(f"  Quantized: {results['quant_acc']:.2f}%")
    print(f"  Drop: {results['accuracy_drop']:.2f}%")
    print(f"  Speedup: {results['speedup']:.2f}x")

# full_quantization_pipeline(model, train_loader, cal_loader, val_loader)
```

---

## Advanced Topics

### 8.1 Entropy-Based Calibration

```python
def calibrate_model_entropy(model, calibration_loader, num_batches=10):
    """
    Use KL divergence to find optimal quantization range.

    Approach:
    1. Collect FP32 activation distribution
    2. For each possible INT8 range, compute KL divergence vs FP32
    3. Choose range that minimizes KL divergence
    """

    model.eval()

    # Collect distributions
    activation_distributions = {}

    def capture_stats(name):
        def hook(module, input, output):
            if isinstance(output, torch.Tensor):
                if name not in activation_distributions:
                    activation_distributions[name] = []
                activation_distributions[name].append(output.detach().cpu().numpy())
        return hook

    hooks = []
    for name, module in model.named_modules():
        if isinstance(module, nn.ReLU):
            hook = module.register_forward_hook(capture_stats(name))
            hooks.append(hook)

    # Collect statistics
    print(f"Collecting distributions on {num_batches} batches...")
    with torch.no_grad():
        for batch_idx, (images, _) in enumerate(calibration_loader):
            if batch_idx >= num_batches:
                break
            if torch.cuda.is_available():
                images = images.cuda()
            _ = model(images)

    for hook in hooks:
        hook.remove()

    # Compute optimal ranges using KL divergence
    print("Computing entropy-based ranges...")
    optimal_ranges = {}

    for name, stats_list in activation_distributions.items():
        data = np.concatenate(stats_list).flatten()
        data = np.clip(data, 0, None)  # ReLU is non-negative

        # Create histogram of FP32 data (256 bins)
        fp32_hist, bins = np.histogram(data, bins=256)
        fp32_hist = fp32_hist + 1e-10  # Avoid log(0)
        fp32_hist = fp32_hist / fp32_hist.sum()

        # Try different quantization ranges
        min_kl = float('inf')
        best_range = (0, data.max())

        for percentile in range(80, 100):
            high = np.percentile(data, percentile)
            low = 0

            # Quantize to INT8
            quant_data = np.round(data / (high / 127))
            quant_data = np.clip(quant_data, 0, 127)

            # Create INT8 histogram
            int8_hist, _ = np.histogram(quant_data, bins=128)
            int8_hist = int8_hist + 1e-10
            int8_hist = int8_hist / int8_hist.sum()

            # Compute KL divergence (rebin FP32 to INT8)
            fp32_rebinned = np.zeros(128)
            for i in range(256):
                fp32_rebinned[i // 2] += fp32_hist[i]

            kl_div = np.sum(fp32_rebinned * (np.log(fp32_rebinned) - np.log(int8_hist)))

            if kl_div < min_kl:
                min_kl = kl_div
                best_range = (low, high)

        optimal_ranges[name] = best_range
        print(f"  {name}: range={best_range}, KL={min_kl:.4f}")

    return optimal_ranges
```

### 8.2 HMX Optimizations for Matrix Multiplication

```c
// Optimized INT8 GEMM using HMX (Hexagon Matrix Extensions)
// HMX provides 512 MACs per cycle for INT8 operations

#include "hvx_hexagon_protos.h"
#include "hmx_hexagon_protos.h"

void gemm_int8_hmx_optimized(
    const int8_t *A,           // M × K matrix (weights)
    const int8_t *B,           // K × N matrix (activations)
    int8_t *C,                 // M × N output
    int M, int K, int N,
    float scale_in, float scale_out
) {
    // HMX operates on 128×128 matrices with padding
    const int BLOCK_SIZE = 128;

    int8_t *C_int32 = malloc(M * N * sizeof(int32_t));

    // Process in blocks
    for (int m = 0; m < M; m += BLOCK_SIZE) {
        for (int n = 0; n < N; n += BLOCK_SIZE) {
            int m_block = (m + BLOCK_SIZE <= M) ? BLOCK_SIZE : (M - m);
            int n_block = (n + BLOCK_SIZE <= N) ? BLOCK_SIZE : (N - n);

            // Zero accumulator
            for (int i = 0; i < m_block * n_block; i++) {
                C_int32[i] = 0;
            }

            // K-loop
            for (int k = 0; k < K; k += BLOCK_SIZE) {
                int k_block = (k + BLOCK_SIZE <= K) ? BLOCK_SIZE : (K - k);

                // Extract submatrices
                int8_t *A_block = (int8_t*)(A + m * K + k);
                int8_t *B_block = (int8_t*)(B + k * N + n);
                int32_t *C_block = (int32_t*)(C_int32);

                // HMX gemm: C_block += A_block × B_block
                // Peak: 512 INT8 MACs per cycle @ 800 MHz = 409.6 GOPS
                hmx_gemm_int8(A_block, B_block, C_block,
                             m_block, k_block, n_block);
            }

            // Requantize and store
            for (int i = 0; i < m_block; i++) {
                for (int j = 0; j < n_block; j++) {
                    int32_t acc = C_int32[i * n_block + j];
                    int32_t scaled = (int32_t)(((int64_t)acc * scale_in) / scale_out);
                    C[(m + i) * N + (n + j)] = CLAMP(scaled, -128, 127);
                }
            }
        }
    }

    free(C_int32);
}
```

### 8.3 Layer Fusion Strategies

Common layer fusion patterns for Hexagon:

1. **Conv + BatchNorm Folding:**
   - Merge BN parameters into Conv weights
   - Eliminates separate BN kernel

2. **Conv + ReLU Fusion:**
   - ReLU applied during requantization (clamp)
   - No separate ReLU instruction

3. **Conv + Add (Skip Connection):**
   - Add performed with same scale
   - Requantization handles mismatched scales

4. **Depthwise + Pointwise:**
   - Grouped convolutions
   - More efficient on HVX with per-group scale

---

## Self-Assessment

### Questions

1. **Throughput Comparison:**
   - A Hexagon HVX operates at 800 MHz. How many FP32 MACs per second can it achieve?
   - How does this compare to INT8 MACs per second?
   - What is the bandwidth requirement for FP32 vs INT8 at peak throughput?

2. **Per-Channel Quantization:**
   - Explain why per-channel quantization is superior to per-tensor for ResNet-50.
   - What is the memory overhead of storing per-channel scales?

3. **Asymmetric Zero-Point:**
   - For a ReLU-activated layer with range [0, 4.3], compare:
     - Symmetric quantization error
     - Asymmetric quantization error
   - Why is asymmetric preferred?

4. **Requantization Fixed-Point:**
   - Given M = 0.00781 (s_in / s_out), decompose into (m, n) such that M ≈ m / 2^n
   - Implement the formula: result = round((acc × m) / 2^n) in pseudocode using vmpye + vasr

5. **QDQ Fusion:**
   - Show the mathematical equivalence of:
     - Dequant(scale_1, zp_1) → Quant(scale_2, zp_2)
     - Requant(scale_1, scale_2, zp_1, zp_2)
   - What is saved by fusing?

6. **Mixed Precision:**
   - For a model with INT8 weights and INT16 activations, what is the throughput vs INT8×INT8?
   - When should mixed precision be used?

7. **PyTorch → Hexagon:**
   - Outline the full pipeline from FP32 PyTorch model to quantized ONNX with QDQ nodes
   - What is the purpose of each step?

### Expected Answers (Sketches)

1. **Throughput:**
   - FP32 @ 800 MHz: 32 MACs × 800M = 25.6 GFLOPS
   - INT8 @ 800 MHz: 128 MACs × 800M = 102.4 GOPS (4× better)
   - Bandwidth @ 64-bit DDR4: 12.8 GB/s (FP32 needs 3.2 GB/s, INT8 needs 0.8 GB/s)

2. **Per-Channel:**
   - Per-channel allows each of the 64 output channels to have independent scale
   - Reduces quantization error by 3–5× for typical layers
   - Memory overhead: 64 × 4 bytes = 256 bytes (minimal vs layer size)

3. **Zero-Point:**
   - Symmetric: needs 64 levels for [0, 4.3], loses negative range
   - Asymmetric: uses full 256 levels for [0, 4.3], 2× better error
   - Reason: ReLU output never negative, so asymmetric is optimal

4. **Requantization:**
   - M = 0.00781 ≈ 2^31 × 0.00781 / 2^31 = 3353 / 2^32 + rounding
   - Or: m = 505, n = 16 (505 / 2^16 ≈ 0.00771)
   - Formula: `result = (acc * 505 + (1 << 15)) >> 16`

5. **QDQ Fusion:**
   - y_q = round((x_q - z1) × s1 / s2 + z2)
   - = round(x_q × (s1/s2) + (z2 - z1 × (s1/s2)))
   - Saves: 2 tensor memory allocations and 2 kernel launches

6. **Mixed Precision:**
   - Throughput: 64 INT8 elements × 1 MAC/cycle vs 128 INT8 (2× reduction)
   - Use for: first/last layers, attention, skip connections with large mismatch
   - Typically 0.5–1% accuracy improvement for small throughput cost

7. **PyTorch → Hexagon:**
   - FP32 training → PTQ calibration → Export ONNX → Hexagon compiler → Quantized HVX code
   - Each step reduces: model size, memory bandwidth, latency

---

## References

### Hexagon SDK Documentation

- **Hexagon V66/V68 HVX Programmer's Guide**
  - Vector instruction set, latency tables
  - Located: `<HEXAGON_SDK>/docs/HVX_Programming.pdf`

- **Hexagon DSP Audio Optimization**
  - HVX best practices
  - Located: `<HEXAGON_SDK>/examples/audio/`

- **HMX (Hexagon Matrix eXtensions) Specification**
  - Matrix operation patterns
  - Located: `<HEXAGON_SDK>/docs/HMX_spec.pdf`

### PyTorch Quantization Documentation

- **PyTorch Quantization Docs**
  - https://pytorch.org/docs/stable/quantization.html

- **QAT Tutorial**
  - https://pytorch.org/tutorials/intermediate/quantization_aware_training.html

### ONNX Quantization

- **ONNX QuantizeLinear/DequantizeLinear**
  - https://github.com/onnx/onnx/blob/main/docs/Operators.md#quantizelinear

- **ONNX QDQ Graph Optimization**
  - https://github.com/microsoft/onnxruntime/blob/master/docs/execution_providers/QNN-ExecutionProvider.md

### Research Papers

- **Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference**
  - Jacob et al., 2018
  - Seminal work on post-training quantization

- **A White Paper on Neural Network Quantization**
  - ARMv8 NEON quantization (similar to HVX)
  - Details on fixed-point arithmetic

### Expert Insight Boxes

**⚡ Key Takeaway: Why INT8 Dominates Edge Inference**

The fundamental advantage of INT8 quantization is the 4× reduction in memory bandwidth. For mobile/edge inference, bandwidth is often the bottleneck (not compute). A convolution layer processes each activation once (O(N)) but uses each weight once (O(N/batch)). Thus:

- FP32: 4 bytes/value × N activations + 4 bytes/value × M weights = ~8N bytes
- INT8: 1 byte/value × N activations + 1 byte/value × M weights = ~2N bytes

This 4× savings directly translates to:
1. **4× speedup** on memory-limited hardware
2. **75% power reduction** (memory >> compute in power budget)
3. **Better thermal profile** (less heat → sustained performance)

The trade-off (accuracy loss) is recoverable via QAT and proper calibration.

---

## Conclusion

Quantization is the bridge between high-precision training and efficient edge deployment. Hexagon's specialized INT8 units (HVX, HMX) make this bridge optimal:

- **Throughput**: 4× over FP32
- **Memory**: 4× reduction
- **Power**: 75% savings
- **Latency**: 2–3× speedup in realistic workloads

The complete pipeline (PyTorch → PTQ/QAT → ONNX QDQ → Hexagon Compiler → HVX Kernels) is now standard in production. Master each step, and you can deploy state-of-the-art models on Qualcomm Snapdragon processors with minimal accuracy loss.

**Next Module:** Compiler Techniques and Graph Optimization (Module 6)

---

**Document Statistics:**
- **Total Lines:** 1,847
- **Code Examples:** 22 (Python + C/HVX intrinsics)
- **Mathematical Derivations:** 14
- **ASCII Diagrams:** 3
- **References:** 10+ SDK/research papers
- **Self-Assessment Questions:** 7 with sketches

