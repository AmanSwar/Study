# Module 6: Operator Kernel Design & Optimization for Hexagon NPU

**Target Audience**: PhD-level ML systems researchers, low-level optimization engineers
**Prerequisites**: Modules 1-5 (HVX architecture, intrinsics, VTCM management, quantization)
**Duration**: 8-10 weeks intensive study
**Practical Outcomes**: Production-grade operator implementations; ability to optimize any custom operator for Hexagon

---

## Table of Contents

1. [Introduction & Design Principles](#introduction--design-principles)
2. [Convolution Operators (Conv2D)](#convolution-operators-conv2d)
3. [Fully Connected / Linear Layers (GEMM)](#fully-connected--linear-layers-gemm)
4. [Activation Functions](#activation-functions)
5. [Softmax & Normalization](#softmax--normalization)
6. [Layer Normalization & Fused Operations](#layer-normalization--fused-operations)
7. [Element-wise Operations](#element-wise-operations)
8. [Pooling Operations](#pooling-operations)
9. [Tensor Layout Transformations](#tensor-layout-transformations)
10. [Attention Mechanisms](#attention-mechanisms)
11. [Performance Tuning & Benchmarking](#performance-tuning--benchmarking)
12. [Real-World Case Studies](#real-world-case-studies)

---

## Introduction & Design Principles

### 1.1 The Hexagon Optimization Landscape

Modern deep neural network inference on edge devices requires an unprecedented level of fine-grained optimization. The Hexagon Vector Extension (HVX) on Qualcomm Snapdragon processors provides:

- **128-bit or 256-bit wide vector operations** (configurable)
- **Virtual Tiling Cache Memory (VTCM)**: 256 KB fast scratchpad memory per HVX unit
- **HMX (Heavy Matrix Extension)**: AI-accelerated matrix operations (if available)
- **Dual-issue execution**: Multiple independent HVX operations per cycle
- **Permutation network**: Low-cost shuffle/transpose operations

However, naive use of these features yields only 10-20% of peak performance. Achieving 60-80% requires:

1. **Hierarchical tiling**: Decompose work to fit in VTCM
2. **Memory-aware computation**: Match data layout to HVX instruction patterns
3. **Dependency chains**: Minimize pipeline stalls
4. **Instruction-level parallelism**: Keep all execution units busy
5. **Quantization-aware design**: Exploit INT8 throughput

### 1.2 Core Design Patterns

#### Pattern 1: Tile-by-Tile Computation

```
// Pseudocode structure for all operators
for (int tile_h = 0; tile_h < H; tile_h += TILE_H) {
    for (int tile_w = 0; tile_w < W; tile_w += TILE_W) {
        for (int tile_c = 0; tile_c < C; tile_c += TILE_C) {

            // Load tile into VTCM (via DMA or explicit load)
            // Process: compute results in registers/VTCM
            // Store tile to output memory
        }
    }
}
```

#### Pattern 2: Reuse-Driven Computation

Data reuse hierarchy (from best to worst):
1. **Registers**: 32 HVX registers × (128 or 256) bits
2. **VTCM L1**: 256 KB, ~1 cycle latency
3. **L2 Cache**: ~4-8 KB per core, ~4 cycle latency
4. **L3 Cache**: Shared, variable latency
5. **DDR**: ~100+ cycle latency

Arithmetic intensity = FLOPs / (bytes loaded from memory). Target >10 for efficient compute.

#### Pattern 3: Quantization-First Design

All operators discussed assume INT8 weights and activations where applicable, with careful tracking of:
- Scale factors (per-tensor, per-channel)
- Clipping ranges
- Accumulation bit-width

### 1.3 Performance Modeling

On Hexagon with HVX (256-bit mode):
- **Peak throughput**: 256-bit operations every cycle
- **Peak data bandwidth**: 8 bytes/cycle (DDR3) to ~50 GB/s (VTCM)
- **Instruction latency**: 1 cycle for simple ops; 3+ for complex ops

**Roofline Model**: For a given operator, compute:
- Arithmetic intensity: I = total_ops / total_bytes
- Memory-bound if I < I_roof = peak_bw / peak_throughput
- Compute-bound if I > I_roof

For INT8 ops with HVX:
- Peak throughput ≈ 256 × clock_GHz INT8 ops/sec
- Peak memory bandwidth ≈ 20-30 GB/s from main memory

---

## Convolution Operators (Conv2D)

### 2.1 Convolution Fundamentals & Hexagon Considerations

Convolution is the bottleneck in most CNN inference on Hexagon. A (H, W, C_in) × (K, K, C_in, C_out) convolution performs:

```
output[n, h, w, c_out] = sum_{kh, kw, c_in} (
    input[n, h*stride + kh, w*stride + kw, c_in] * kernel[kh, kw, c_in, c_out]
)
```

**Challenge on Hexagon**:
- Cannot hold all weights in VTCM for large C_out (e.g., mobilenet with 512+ channels)
- Input reuse is 9× (for 3×3 kernel) but weights must be streamed
- Quantization introduces scale factor handling

**Key Optimizations**:
1. **Weight layout**: Pack weights into format optimal for vrmpyacc (matrix multiply accumulate)
2. **Tiling strategy**: Process H and W in strips; outer-loop over K_out; stream weights
3. **Padding handling**: Fuse zero-padding into load logic or use boundary guards
4. **Im2col preprocessing**: For small kernels, convert to GEMM (sometimes faster)

### 2.2 Weight Packing for Convolution

For INT8 convolution, weights must be laid out to maximize throughput of `vrmpyacc` (vector matrix multiply accumulate):

```
// Naive layout (BAD):
weight[kh * K + kw, c_in, c_out]  // Poor cache locality for dot product

// Optimized layout (GOOD):
// Pack into 32-byte chunks aligned for vrmpyacc
// For INT8: 32 bytes = 32 weights
// Outer: output channel (c_out)
// Middle: spatial positions (kh, kw)
// Inner: input channel (c_in) with padding to 32-byte alignment
```

**Example packing for 3×3 convolution, INT8**:

```c
// Original: weight[3, 3, 32, 64] (kernel 3×3, 32 input channels, 64 output)
// Packed: weight_packed[64, 32, 36]
//   - 64: output channels (c_out, vectorized)
//   - 32: input channels (c_in)
//   - 36: spatial (9) × padding factor; stored as rows for vrmpyacc

void pack_conv_weights(
    const int8_t *weight,  // [Kh, Kw, C_in, C_out]
    int Kh, int Kw, int C_in, int C_out,
    int8_t *weight_packed   // [C_out, C_in, Kh*Kw] (simplified)
) {
    for (int c_out = 0; c_out < C_out; c_out++) {
        for (int c_in = 0; c_in < C_in; c_in++) {
            for (int kh = 0; kh < Kh; kh++) {
                for (int kw = 0; kw < Kw; kw++) {
                    int idx = c_out * (C_in * Kh * Kw)
                            + c_in * (Kh * Kw)
                            + kh * Kw + kw;
                    weight_packed[idx] = weight[kh * (Kw * C_in * C_out)
                                               + kw * (C_in * C_out)
                                               + c_in * C_out
                                               + c_out];
                }
            }
        }
    }
}
```

### 2.3 Tiling Strategy for Convolution

Given VTCM of 256 KB, allocate:
- ~100 KB for input tile
- ~100 KB for output tile
- ~50 KB for weights (if fitting a C_out subset)

**Two-level tiling approach**:

```
// Outer loop: stream different weight channels
for (int c_out_tile = 0; c_out_tile < C_out; c_out_tile += TILE_C_OUT) {
    // Load TILE_C_OUT weight planes into VTCM

    // Inner loop: tile input spatial dimensions
    for (int h_tile = 0; h_tile < H_out; h_tile += TILE_H) {
        for (int w_tile = 0; w_tile < W_out; w_tile += TILE_W) {

            // Load input tile + padding from DDR
            // Compute: partial results for [h_tile:h_tile+TILE_H, w_tile:w_tile+TILE_W]
            // Accumulate into output buffer
            // Store results
        }
    }
}
```

**Typical tile sizes** (256-bit HVX, INT8):
- TILE_H = 16-32 (rows of output)
- TILE_W = 32-64 (cols of output)
- TILE_C_OUT = 16-32 (output channels per weight load)

### 2.4 Handling Padding, Stride, Dilation

**Padding**: Three approaches:
1. **Pad in-place**: Load input with padding region zeroed (simple, wastes VTCM)
2. **Guard logic in loop**: Check boundary conditions during compute (complex)
3. **Fused load**: Special VTCM load functions that zero-pad automatically

**Stride & Dilation**: Handled at index computation:

```c
// For stride=S, dilation=D:
int in_h = h_out * S + kh * D;  // Map output position to input

// Ensure in_h, in_w within valid range [0, H), [0, W)
if (in_h >= 0 && in_h < H && in_w >= 0 && in_w < W) {
    // Valid position: contribute to dot product
} else {
    // Out of bounds: skip or use padding value (usually 0)
}
```

### 2.5 Direct Convolution vs Im2Col

**Direct convolution**: Implement loop-nest directly, computing dot products on-the-fly.

```
for (int oh = 0; oh < OH; oh++) {
    for (int ow = 0; ow < OW; ow++) {
        for (int c_out = 0; c_out < C_out; c_out++) {
            int32_t acc = 0;
            for (int kh = 0; kh < K; kh++) {
                for (int kw = 0; kw < K; kw++) {
                    for (int c_in = 0; c_in < C_in; c_in++) {
                        int ih = oh*stride + kh;
                        int iw = ow*stride + kw;
                        if (valid(ih, iw)) {
                            acc += input[ih, iw, c_in] * weight[kh, kw, c_in, c_out];
                        }
                    }
                }
            }
            output[oh, ow, c_out] = requantize(acc);
        }
    }
}
```

**Im2Col**: Convert input to a 2D matrix, then use GEMM:
- Input [H, W, C_in] → Im2Col matrix [K*K*C_in, H_out*W_out]
- Weight [K, K, C_in, C_out] → matrix [C_out, K*K*C_in]
- Output via GEMM: [C_out, H_out*W_out]

**When to use each**:
- **Direct**: Small kernels (1×1, 3×3), HVX-friendly memory patterns
- **Im2Col**: Very large kernels (7×7+), benefits from GEMM library

**Hexagon reality**: Highly-tuned direct convolution almost always wins due to better VTCM reuse.

### 2.6 Complete HVX Implementation: INT8 3×3 Depthwise Convolution

Depthwise separable convolution (used in MobileNet) has C_in = C_out = channels, kernel 3×3.

```c
#include <hexagon_types.h>
#include <hvx_hexagon_protos.h>
#include <string.h>

#define VTCM_IOBUF_SIZE (256 * 1024)
#define TILE_H 16
#define TILE_W 32
#define PAD 1
#define STRIDE 1

typedef struct {
    int height, width, channels;
    int out_h, out_w;
    const int8_t *input;
    const int8_t *weight;
    const int32_t *bias;
    int32_t input_scale, weight_scale, output_scale;
    int8_t *output;
} Conv2DParams;

/**
 * Depthwise 3×3 INT8 convolution with stride=1, padding=1
 *
 * Key features:
 *  - Direct implementation (no im2col)
 *  - Per-channel quantization support
 *  - Tiled to fit in VTCM
 *  - Uses HVX vector instructions for 8-wide parallel dot products
 *
 * Assumptions:
 *  - input/output are tightly packed [H, W, C] format
 *  - weight is [3, 3, C] (depthwise, so C_in = C_out = C)
 *  - VTCM is available for buffering
 */
void conv2d_3x3_depthwise_int8_hvx(const Conv2DParams *params) {
    int H = params->height;
    int W = params->width;
    int C = params->channels;
    int OH = params->out_h;
    int OW = params->out_w;

    const int8_t *input = params->input;
    const int8_t *weight = params->weight;
    const int32_t *bias = params->bias;
    int8_t *output = params->output;

    // Allocate VTCM buffers
    int8_t *vtcm_input = (int8_t *)memalign(128, VTCM_IOBUF_SIZE / 2);
    int32_t *vtcm_accum = (int32_t *)(vtcm_input + VTCM_IOBUF_SIZE / 4);

    // Tiling parameters
    int tile_h = TILE_H;
    int tile_w = TILE_W;

    // Outer loop: tile over output height
    for (int oh_start = 0; oh_start < OH; oh_start += tile_h) {
        int oh_end = (oh_start + tile_h < OH) ? oh_start + tile_h : OH;
        int oh_len = oh_end - oh_start;

        // Tile over output width
        for (int ow_start = 0; ow_start < OW; ow_start += tile_w) {
            int ow_end = (ow_start + tile_w < OW) ? ow_start + tile_w : OW;
            int ow_len = ow_end - ow_start;

            // Input tile spans: [oh_start : oh_end + 2*PAD] × [ow_start : ow_end + 2*PAD] × C
            // (because 3×3 kernel needs padding)
            int ih_start = oh_start * STRIDE - PAD;  // 0 with PAD=1, stride=1
            int ih_end = oh_end * STRIDE + PAD + 1;   // oh_end + PAD
            int iw_start = ow_start * STRIDE - PAD;
            int iw_end = ow_end * STRIDE + PAD + 1;

            // Clamp to valid input ranges
            int ih_fetch_start = (ih_start < 0) ? 0 : ih_start;
            int ih_fetch_end = (ih_end > H) ? H : ih_end;
            int iw_fetch_start = (iw_start < 0) ? 0 : iw_start;
            int iw_fetch_end = (iw_end > W) ? W : iw_end;

            // Load input tile into VTCM
            // For simplicity: load entire padded region with zero-padding
            int32_t *accum = vtcm_accum;  // Accumulator in VTCM
            memset(accum, 0, oh_len * ow_len * C * sizeof(int32_t));

            // Compute kernel
            // For each spatial position in output tile
            for (int oh = oh_start; oh < oh_end; oh++) {
                for (int ow = ow_start; ow < ow_end; ow++) {
                    // For each channel (depthwise)
                    for (int c = 0; c < C; c += 8) {  // Process 8 channels at a time
                        HVX_Vector vacc0 = Q6_V_vsplat_R(0);  // Accum for channels [c, c+8)

                        // Convolve 3×3 kernel
                        for (int kh = 0; kh < 3; kh++) {
                            for (int kw = 0; kw < 3; kw++) {
                                int ih = oh * STRIDE + kh - PAD;
                                int iw = ow * STRIDE + kw - PAD;

                                if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                                    // Load 8 input values [ih, iw, c:c+8]
                                    const int8_t *in_ptr = &input[ih * W * C + iw * C + c];
                                    HVX_Vector vin = Q6_V_vldu_A((HVX_Vector *)in_ptr);

                                    // Load 8 weight values [kh, kw, c:c+8]
                                    const int8_t *w_ptr = &weight[kh * 3 * C + kw * C + c];
                                    HVX_Vector vw = Q6_V_vldu_A((HVX_Vector *)w_ptr);

                                    // Multiply and accumulate (int8 × int8 → int32)
                                    // HVX doesn't have native 8×8→32, use 16×16 with unpacking
                                    HVX_VectorPair p = Q6_Wh_vmpa_WbWb(
                                        Q6_Wb_vunpack_V(vin),  // Unpack to 16-bit
                                        Q6_Wb_vunpack_V(vw)    // Unpack to 16-bit
                                    );
                                    HVX_Vector hi = Q6_V_hi_W(p);
                                    HVX_Vector lo = Q6_V_lo_W(p);

                                    // Accumulate (note: this is simplified)
                                    vacc0 = Q6_Vw_vadd_VwVw(
                                        Q6_Vw_vmpy_VhVh(hi, Q6_V_vsplat_R(1)),
                                        vacc0
                                    );
                                } else {
                                    // Out of bounds: input is 0 (padding)
                                    // 0 * weight = 0, no contribution
                                }
                            }
                        }

                        // Store accumulator
                        int out_idx = (oh - oh_start) * ow_len * C + (ow - ow_start) * C + c;
                        Q6_V_vstu_A((HVX_Vector *)&accum[out_idx], vacc0);
                    }
                }
            }

            // Requantize and store output
            for (int oh = oh_start; oh < oh_end; oh++) {
                for (int ow = ow_start; ow < ow_end; ow++) {
                    for (int c = 0; c < C; c++) {
                        int out_idx_tile = (oh - oh_start) * ow_len * C + (ow - ow_start) * C + c;
                        int32_t acc_val = accum[out_idx_tile];

                        // Add bias
                        acc_val += bias[c];

                        // Requantize: convert back to INT8 range
                        // acc_val is in arbitrary scale; divide by input_scale * weight_scale
                        int32_t scaled = acc_val / (params->input_scale * params->weight_scale);
                        scaled = (scaled * params->output_scale) >> 8;

                        // Clip to [-128, 127]
                        int8_t out_val = (int8_t)Q6_R_clip_R(scaled, 127);
                        out_val = (int8_t)Q6_R_max_RR(out_val, -128);

                        output[oh * OW * C + ow * C + c] = out_val;
                    }
                }
            }
        }
    }

    free(vtcm_input);
}

/**
 * Higher-level wrapper for standard and grouped convolution
 */
void conv2d_int8_hvx(
    const int8_t *input, int H, int W, int C_in,
    const int8_t *weight, int K, int C_out, int groups,
    int stride, int padding, int dilation,
    const int32_t *bias,
    int32_t input_scale, int32_t weight_scale, int32_t output_scale,
    int8_t *output
) {
    int OH = (H + 2*padding - dilation*(K-1) - 1) / stride + 1;
    int OW = (W + 2*padding - dilation*(K-1) - 1) / stride + 1;

    if (groups == 1) {
        // Standard convolution
        // Weight shape: [K, K, C_in, C_out]
        // Use GEMM-based approach or direct convolution depending on K

        if (K == 3 && stride == 1 && padding == 1) {
            // Optimized path for 3×3 convolution (depthwise variant)
            Conv2DParams params = {
                .height = H,
                .width = W,
                .channels = C_in,
                .out_h = OH,
                .out_w = OW,
                .input = input,
                .weight = weight,
                .bias = bias,
                .input_scale = input_scale,
                .weight_scale = weight_scale,
                .output_scale = output_scale,
                .output = output,
            };
            conv2d_3x3_depthwise_int8_hvx(&params);
        } else {
            // General case: use direct convolution or im2col
            // [Implementation would follow similar pattern]
        }
    } else {
        // Grouped convolution: C_in and C_out are divided by groups
        // Process each group independently
        int C_in_group = C_in / groups;
        int C_out_group = C_out / groups;

        for (int g = 0; g < groups; g++) {
            const int8_t *input_g = &input[0];  // [No offset in spatial, but offset in channels]
            const int8_t *weight_g = &weight[K*K*C_in_group*g];
            int32_t *bias_g = (int32_t *)&bias[C_out_group * g];
            int8_t *output_g = &output[0];

            // Recursively call conv2d_int8_hvx with groups=1
            // [Simplified; full implementation would slice tensors]
        }
    }
}
```

**Key points in the implementation**:

1. **VTCM allocation**: `memalign(128, ...)` for cache-line alignment
2. **Tiling bounds**: Carefully handle edge cases where tiles don't align with output dimensions
3. **Padding logic**: Zero out-of-bounds positions (if ih < 0 || iw >= H, etc.)
4. **HVX intrinsics**:
   - `Q6_V_vldu_A()`: Load unaligned vector
   - `Q6_Wb_vunpack_V()`: Unpack INT8 to INT16 (for INT8 multiply)
   - `Q6_Wh_vmpa_WbWb()`: Vector matrix multiply-accumulate (16×16→32)
   - `Q6_R_clip_R()`: Saturate clip to range
5. **Requantization**: Divide by scale factors and apply output scale; saturate clip

**Performance analysis**:
- **Peak throughput**: 32 INT8 multiplies per cycle (256 bits ÷ 8 bits) × 2 execution units
- **Arithmetic intensity**: For 3×3 conv with C_in=32, C_out=32: I ≈ (9×32×32) / (9×32×2 + 32×32) ≈ 2.8 (memory-bound)
- **Typical bandwidth**: ~20 GB/s from DDR; VTCM: ~200+ GB/s
- **Expected performance**: 60-70% of peak for well-tuned 3×3 conv

---

### 2.7 Standard and Grouped Convolution Variants

**Standard convolution** (weight shape [K, K, C_in, C_out]):
- C_in and C_out independent
- Each output channel computes a dot product over all input channels

**Depthwise convolution** (weight shape [K, K, C]):
- C_in = C_out = C
- Each output channel only depends on one input channel
- 1/8 to 1/32 the computation vs standard conv (depending on C)
- Used in MobileNet, EfficientNet

**Grouped convolution** (groups > 1, weight shape [K, K, C_in, C_out]):
- Input and output channels divided into groups
- Each group processes independently
- Intermediate between standard and depthwise

**Separable convolution** (Depthwise + Pointwise):
1. Depthwise 3×3 (per-channel)
2. Pointwise 1×1 convolution (across channels)
- Total compute ≈ 1/9 of standard 3×3 + cost of 1×1
- Heavily used in mobile architectures

### 2.8 Expert Insight & Pitfalls

⚡ **Expert Insight**:
- On Hexagon, direct convolution beats im2col for most kernels ≤7×7 due to superior VTCM reuse. Im2col shines for very large kernels (11×11+) where padding dominates, or when leveraging highly-optimized GEMM libraries.
- **Weight layout is everything**: A poorly-packed weight tensor can reduce peak throughput by 3×. Transpose weights offline during model compilation, not at inference time.
- **Padding overhead**: If padding is significant (e.g., "same" padding with stride>1), fuse it into the load logic rather than pre-padding the input (saves memory bandwidth).
- **Dilation requires special care**: Dilated convolutions have poor memory patterns (sparse strided access). Consider unfurling the loop or using bit-manipulation tricks to compute strides.

⚠ **Common Pitfalls**:
1. **Integer overflow in accumulation**: For INT8×INT8 dot products over 256+ channels, INT32 accumulator overflows. Solution: Accumulate in INT32 or INT64; requantize frequently.
2. **Quantization underflow**: Small scale factors (< 0.01) cause rounding errors. Ensure input_scale × weight_scale ≈ 1 through proper calibration.
3. **Cache thrashing**: Loading weights from DDR each time = 100+ cycle latency. Always load weights into VTCM first.
4. **Stride-induced aliasing**: Stride > kernel size causes input regions to be skipped. Verify stride ≤ kernel size for standard CNN designs.
5. **Unaligned memory access**: HVX requires aligned loads. If input dimensions aren't multiples of vector width, pad or use `Q6_V_vldu_A()` (unaligned, slower).

---

## Fully Connected / Linear Layers (GEMM)

### 3.1 GEMM on Hexagon: Fundamentals

A fully connected (FC) layer is a matrix multiplication:

```
output [N, C_out] = input [N, C_in] @ weight [C_in, C_out]ᵀ
```

For INT8 quantized networks:
- input: INT8, shape [N, C_in], scale factor S_in
- weight: INT8, shape [C_out, C_in], scale factor S_w (per-channel or per-tensor)
- bias: INT32, shape [C_out]
- output: INT8, shape [N, C_out], scale factor S_out

The computation is a series of dot products:

```
for n in range(N):
    for c_out in range(C_out):
        acc = 0
        for c_in in range(C_in):
            acc += input[n, c_in] * weight[c_out, c_in]
        output[n, c_out] = requantize(acc)
```

**Hexagon considerations**:
1. **Weight packing**: Reorder weights to maximize cache locality and HVX utilization
2. **Tiling over N**: Process multiple input rows in parallel (outer loop)
3. **Tiling over C_in**: Stream weight rows, process multiple output columns in parallel
4. **VTCM reuse**: Load N rows of input and C_out columns of weight; compute N×C_out output tile

### 3.2 Weight Packing for GEMM

For INT8 GEMM, weights should be laid out such that dot products are computed efficiently.

**Layout strategy**:
```
Original:     weight[C_out, C_in] (row-major)
Packed:       weight_packed[C_out_packed, C_in_packed]
  where packed means:
    - C_out grouped into chunks (e.g., 16 channels per group)
    - C_in padded to multiple of HVX vector width (32 for 256-bit mode with INT8)
```

**Example packing code**:

```c
#define C_IN_PACK_WIDTH 32  // HVX 256-bit ÷ 8 bits = 32 elements
#define C_OUT_PACK_HEIGHT 16

void pack_gemm_weights(
    const int8_t *weight,         // [C_out, C_in]
    int C_out, int C_in,
    int8_t *weight_packed         // [C_out, C_in_padded]
) {
    int C_in_padded = ((C_in + C_IN_PACK_WIDTH - 1) / C_IN_PACK_WIDTH) * C_IN_PACK_WIDTH;

    for (int c_out = 0; c_out < C_out; c_out++) {
        for (int c_in = 0; c_in < C_in; c_in++) {
            weight_packed[c_out * C_in_padded + c_in] = weight[c_out * C_in + c_in];
        }
        // Pad with zeros
        for (int c_in = C_in; c_in < C_in_padded; c_in++) {
            weight_packed[c_out * C_in_padded + c_in] = 0;
        }
    }
}
```

### 3.3 Tiling Strategy for GEMM

**Three-dimensional tiling**: M (rows) × N (columns) × K (reduction dimension)

```
for (m_tile = 0; m_tile < M; m_tile += TILE_M) {
    for (n_tile = 0; n_tile < N; n_tile += TILE_N) {

        // Initialize output accumulator: [TILE_M, TILE_N] in registers/VTCM
        int32_t acc[TILE_M * TILE_N] = {0};

        for (k_tile = 0; k_tile < K; k_tile += TILE_K) {
            // Load input tile:  [TILE_M, TILE_K] from input matrix
            // Load weight tile: [TILE_N, TILE_K] from weight matrix
            // Compute partial product: acc += input_tile @ weight_tileᵀ
        }

        // Requantize and store output tile
        store_output_tile(acc, m_tile, n_tile);
    }
}
```

**Typical tile sizes** (for 256-bit HVX):
- TILE_M = 4-8 (input rows)
- TILE_N = 8-16 (output columns, corresponding to weight rows)
- TILE_K = 256-512 (reduction dimension, processed in blocks)

**VTCM allocation**:
- Input tile: TILE_M × TILE_K × 1 byte = 4 KB (for TILE_M=8, TILE_K=512)
- Weight tile: TILE_N × TILE_K × 1 byte = 4-8 KB (for TILE_N=16, TILE_K=512)
- Accumulator: TILE_M × TILE_N × 4 bytes = 256 B (for TILE_M=8, TILE_N=16)
- **Total**: ~10-12 KB, leaving plenty of VTCM for other uses

### 3.4 Complete HVX Implementation: INT8 GEMM

```c
#include <hexagon_types.h>
#include <hvx_hexagon_protos.h>
#include <string.h>

#define VTCM_WORKSPACE (256 * 1024)
#define TILE_M 8
#define TILE_N 16
#define TILE_K 256
#define C_IN_PACK_WIDTH 32

typedef struct {
    const int8_t *input;        // [N, K] (K = C_in)
    const int8_t *weight;       // [M, K] (M = C_out, K = C_in, row-major, packed)
    const int32_t *bias;        // [M]
    int N, M, K;                // Dimensions
    int32_t input_scale, weight_scale, output_scale;
    int8_t *output;             // [N, M]
} GEMMParams;

/**
 * INT8 GEMM (matrix multiplication) optimized for Hexagon HVX
 *
 * Computes: output[N, M] = input[N, K] @ weight[M, K]ᵀ + bias[M]
 *
 * Key optimizations:
 *  - 3-level tiling: M (outputs) × N (batch) × K (dot product)
 *  - HVX vector multiplication (INT8 × INT8 → INT32)
 *  - VTCM buffering for input and weight tiles
 *  - Per-channel quantization (via scale factors)
 *  - Dual-issue parallel operations
 *
 * Assumptions:
 *  - weight is pre-packed with padding: weight[M, K_padded] where K_padded ≥ K
 *  - M is multiple of TILE_N (or handled with boundary logic)
 *  - N is multiple of TILE_M (or handled with boundary logic)
 */
void gemm_int8_hvx(const GEMMParams *params) {
    const int8_t *input = params->input;
    const int8_t *weight = params->weight;
    const int32_t *bias = params->bias;
    int N = params->N;
    int M = params->M;
    int K = params->K;
    int8_t *output = params->output;

    // Allocate VTCM for tiles
    int8_t *vtcm_input_tile = (int8_t *)memalign(128, TILE_M * TILE_K);
    int8_t *vtcm_weight_tile = (int8_t *)memalign(128, TILE_N * TILE_K);
    int32_t *vtcm_accum = (int32_t *)memalign(128, TILE_M * TILE_N * sizeof(int32_t));

    // Compute K padded (must be multiple of C_IN_PACK_WIDTH for packed weights)
    int K_padded = ((K + C_IN_PACK_WIDTH - 1) / C_IN_PACK_WIDTH) * C_IN_PACK_WIDTH;

    // Outer loops: tile over output dimensions
    for (int m_tile = 0; m_tile < M; m_tile += TILE_N) {
        int m_end = (m_tile + TILE_N < M) ? m_tile + TILE_N : M;
        int m_len = m_end - m_tile;

        for (int n_tile = 0; n_tile < N; n_tile += TILE_M) {
            int n_end = (n_tile + TILE_M < N) ? n_tile + TILE_M : N;
            int n_len = n_end - n_tile;

            // Initialize accumulator to zero
            memset(vtcm_accum, 0, n_len * m_len * sizeof(int32_t));

            // Inner loop: tile over reduction dimension K
            for (int k_tile = 0; k_tile < K; k_tile += TILE_K) {
                int k_end = (k_tile + TILE_K < K) ? k_tile + TILE_K : K;
                int k_len = k_end - k_tile;

                // ===== Load input tile [n_len, k_len] into VTCM =====
                for (int n = 0; n < n_len; n++) {
                    memcpy(&vtcm_input_tile[n * k_len],
                           &input[(n_tile + n) * K + k_tile],
                           k_len);
                }

                // Pad with zeros to k_len to TILE_K
                for (int n = 0; n < n_len; n++) {
                    for (int k = k_len; k < TILE_K; k++) {
                        vtcm_input_tile[n * TILE_K + k] = 0;
                    }
                }

                // ===== Load weight tile [m_len, k_len] into VTCM =====
                for (int m = 0; m < m_len; m++) {
                    memcpy(&vtcm_weight_tile[m * k_len],
                           &weight[(m_tile + m) * K_padded + k_tile],
                           k_len);
                }

                // Pad with zeros to k_len to TILE_K
                for (int m = 0; m < m_len; m++) {
                    for (int k = k_len; k < TILE_K; k++) {
                        vtcm_weight_tile[m * TILE_K + k] = 0;
                    }
                }

                // ===== Compute partial product: acc += input_tile @ weight_tileᵀ =====
                // For each output position [n, m] in the tile
                for (int n = 0; n < n_len; n++) {
                    for (int m = 0; m < m_len; m++) {

                        // Dot product: input[n, :] · weight[m, :]
                        HVX_Vector vacc = Q6_V_vsplat_R(0);  // Accumulator for this (n, m)

                        // Process in HVX vector chunks (32 INT8 elements per vector)
                        for (int k = 0; k < TILE_K; k += 32) {
                            // Load 32 input bytes
                            const int8_t *in_ptr = &vtcm_input_tile[n * TILE_K + k];
                            HVX_Vector vin = Q6_V_vldu_A((HVX_Vector *)in_ptr);

                            // Load 32 weight bytes
                            const int8_t *w_ptr = &vtcm_weight_tile[m * TILE_K + k];
                            HVX_Vector vw = Q6_V_vldu_A((HVX_Vector *)w_ptr);

                            // Unpack INT8 to INT16 and multiply
                            // INT8 × INT8 → INT16 with sign extension
                            HVX_VectorPair p = Q6_Wh_vmpa_WbWb(
                                Q6_Wb_vunpack_V(vin),  // Unpack to INT16 (low 16 of 32 bytes)
                                Q6_Wb_vunpack_V(vw)    // Unpack to INT16
                            );

                            // Extract high/low 32-bit results
                            HVX_Vector prod_hi = Q6_V_hi_W(p);  // [INT32, INT32, ...]
                            HVX_Vector prod_lo = Q6_V_lo_W(p);

                            // Add to accumulator (converting back to scalar)
                            // In practice, this is more complex; simplified here
                            vacc = Q6_Vw_vadd_VwVw(
                                Q6_Vw_vmpy_VhVh(prod_hi, Q6_V_vsplat_R(1)),
                                vacc
                            );
                        }

                        // Extract scalar from vector (sum-reduce)
                        // For now, simplified: accumulator is a vector, need to sum lanes
                        int32_t dot_product = 0;  // Placeholder
                        // In a real implementation, use horizontal sum

                        vtcm_accum[n * m_len + m] += dot_product;
                    }
                }
            }

            // ===== Requantize and store output tile =====
            for (int n = 0; n < n_len; n++) {
                for (int m = 0; m < m_len; m++) {
                    int32_t acc = vtcm_accum[n * m_len + m];

                    // Add bias
                    acc += bias[m_tile + m];

                    // Requantize: input_scale × weight_scale is baked into acc
                    // Convert to output scale
                    int32_t scaled = acc / (params->input_scale * params->weight_scale);
                    scaled = (scaled * params->output_scale) >> 8;

                    // Saturate clip to [-128, 127]
                    int8_t out_val = (int8_t)Q6_R_clip_R(scaled, 127);
                    out_val = (int8_t)Q6_R_max_RR(out_val, -128);

                    output[(n_tile + n) * M + (m_tile + m)] = out_val;
                }
            }
        }
    }

    // Cleanup
    free(vtcm_input_tile);
    free(vtcm_weight_tile);
    free(vtcm_accum);
}

/**
 * Wrapper function for standard FC layer
 */
void fc_layer_int8_hvx(
    const int8_t *input,     // [batch_size, C_in]
    const int8_t *weight,    // [C_out, C_in] (pre-packed)
    const int32_t *bias,     // [C_out]
    int batch_size, int C_in, int C_out,
    int32_t input_scale, int32_t weight_scale, int32_t output_scale,
    int8_t *output           // [batch_size, C_out]
) {
    GEMMParams params = {
        .input = input,
        .weight = weight,
        .bias = bias,
        .N = batch_size,
        .M = C_out,
        .K = C_in,
        .input_scale = input_scale,
        .weight_scale = weight_scale,
        .output_scale = output_scale,
        .output = output,
    };

    gemm_int8_hvx(&params);
}
```

**Key HVX intrinsics used**:
- `Q6_V_vsplat_R(x)`: Replicate scalar to all vector lanes
- `Q6_Wb_vunpack_V(v)`: Unpack INT8 to INT16 (sign-extend)
- `Q6_Wh_vmpa_WbWb(a, b)`: Vector matrix product (INT16 × INT16 → INT32)
- `Q6_V_hi_W(p)`, `Q6_V_lo_W(p)`: Extract high/low from vector pair

**Optimization notes**:
1. **Horizontal sum for scalar reduction**: After accumulating in `vacc`, extract scalar:
   ```c
   // Sum all 8 lanes of 32-bit vector (256-bit mode, 8 × 32 bits)
   HVX_Vector vdup = Q6_V_vror_VR(vacc, 4);  // Rotate right by 4 bytes (32 bits)
   vacc = Q6_Vw_vadd_VwVw(vacc, vdup);
   // Repeat for other lanes...
   ```

2. **Fused bias addition**: Add bias during requantization to save a memory access.

3. **Per-channel quantization**: If weight scale varies per output channel:
   ```c
   int32_t scaled = acc / (input_scale * weight_scale[m_tile + m]);
   ```

### 3.5 Performance Analysis

For INT8 GEMM on Hexagon with 256-bit HVX:

- **Peak throughput**: 32 INT8 operations per cycle × 4 cycles per multiply ≈ 2 billion INT8×INT8 ops/sec (at 1 GHz)
- **Memory bandwidth**: ~20-30 GB/s from DDR
- **Arithmetic intensity** (typical): I = 2×M×N×K / (M×K + N×K + M×N) ≈ 2 (for square GEMM)
- **Roofline analysis**: If I < 2, memory-bound; if I > 2, compute-bound

**Typical performance**:
- **Small GEMM** (M=16, N=256, K=128): ~40-50% of peak (memory-bound due to poor cache locality)
- **Large GEMM** (M=512, N=256, K=512): ~65-75% of peak (better cache utilization)
- **Batch processing**: Process multiple examples in parallel (N dimension) to improve cache hit rate

### 3.6 Expert Insight & Pitfalls

⚡ **Expert Insight**:
- **Vectorization strategy**: Explicitly unroll the K loop and use vector registers for parallel processing. A loop-unrolled K dimension (e.g., process 4 accumulator lanes in parallel) can hide latency and improve ILP.
- **Weight format**: Pre-transposed weights (store as [C_out, C_in] rather than [C_in, C_out]) improve cache locality for dot products. Do this during model serialization.
- **Batch dimension**: Hexagon excels at batch processing (e.g., N > 1). For mobile inference, consider batching multiple inference requests when possible.
- **Fused operations**: Combine GEMM with bias addition, activation, and requantization into a single kernel to reduce memory traffic.

⚠ **Common Pitfalls**:
1. **Register spilling**: Overly complex loop bodies cause spilling to DDR. Keep inner loops simple.
2. **Dependency chains**: Accumulating into a single scalar variable creates a dependency chain (e.g., ACC += A*B forces sequential operations). Use multiple independent accumulators.
3. **Alignment issues**: INT8 loads must align on 16-byte boundaries for HVX. Misaligned loads trigger exceptions or use slow unaligned paths.
4. **Quantization mismatch**: Input and weight scales must match training calibration. Recalibrate if accuracy drops.

---

## Activation Functions

### 4.1 ReLU and ReLU6

**ReLU**: Thresholding at zero.

```
relu(x) = max(0, x)
```

**Implementation (trivial on HVX)**:

```c
HVX_Vector vec_relu(HVX_Vector v) {
    HVX_Vector vzero = Q6_V_vsplat_R(0);
    // For INT8, max is unsigned comparison
    // Assuming INT8 with signed range [-128, 127]
    return Q6_Vb_vmax_VbVb(v, vzero);
}
```

**ReLU6**: Clamped ReLU.

```
relu6(x) = min(6, max(0, x))
```

**Implementation**:

```c
HVX_Vector vec_relu6(HVX_Vector v, int8_t max_val) {
    HVX_Vector vzero = Q6_V_vsplat_R(0);
    HVX_Vector vmax = Q6_V_vsplat_R(max_val);  // = 6 (or scaled)

    // First, clamp to [0, max_val]
    HVX_Vector relu = Q6_Vb_vmax_VbVb(v, vzero);          // max(0, v)
    return Q6_Vb_vmin_VbVb(relu, vmax);                   // min(6, relu)
}
```

**Performance**: 1 cycle per vector (two independent max/min operations).

### 4.2 GELU Activation

**GELU**: Gaussian Error Linear Unit, used in transformers.

```
gelu(x) = x × Φ(x)  where Φ is the standard normal CDF
```

**Exact computation**: Requires error function, impractical on fixed-point.

**Approximation 1: Polynomial** (tanh-based):

```
gelu(x) ≈ 0.5 × x × (1 + tanh(√(2/π) × (x + 0.044715 × x³)))
```

**Approximation 2: Piecewise Linear** (quantization-friendly):

For INT8 GELU:
- Map [-128, 127] input range to GELU output
- Use piecewise linear segments (e.g., 8-16 segments)
- Precompute slopes and intercepts

**Implementation (lookup table)**:

```c
#define GELU_LUT_SIZE 256  // 1 entry per INT8 value

int8_t gelu_lut[256] = {
    // Precomputed GELU values for input range [INT8_MIN, INT8_MAX]
    // Example: input=0 → output ≈ 0, input=64 → output ≈ 50, etc.
    -64, -62, -60, -58, ..., 0, ..., 58, 60, 62, 64
};

HVX_Vector vec_gelu_lut(HVX_Vector v) {
    // Load GELU LUT into VTCM once at kernel start
    // For each element in v, lookup gelu_lut[v[i] + 128]

    HVX_Vector voffset = Q6_Vb_vadd_VbVb(v, Q6_V_vsplat_R(128));

    // Gather 32 bytes from LUT
    // HVX doesn't have direct gather; use scalar loop or permutation
    // (Simplified: assuming we have LUT-lookup intrinsic)
    return Q6_Vb_vlut32_R(v, Q6_V_vsplat_R((uint32_t)&gelu_lut[0]));
}
```

**Implementation (polynomial approximation)**:

```c
HVX_Vector vec_gelu_poly(HVX_Vector v) {
    // gelu(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
    //
    // For INT8: scale all values to fixed-point
    // Assume v is INT8 in range [-4, 4] (scaled)

    // Compute x³
    HVX_Vector v2 = Q6_Vh_vmpy_VbVb(v, v);         // x²
    HVX_Vector v3 = Q6_Vh_vmpy_VhVb(v2, v);        // x³

    // Compute 0.044715 * x³
    // 0.044715 ≈ 5730/128000 in Q15 fixed-point
    HVX_Vector coeff = Q6_Vh_vmpy_VhR(v3, Q6_R_combine_RR(5730, 5730));

    // Compute x + 0.044715 * x³
    HVX_Vector arg = Q6_Vh_vadd_VhVh(v, coeff);

    // tanh approximation: tanh(x) ≈ x * (3 + x²) / (3 + 3*x²)
    // (simpler than full polynomial, still accurate)
    HVX_Vector arg2 = Q6_Vh_vmpy_VhVh(arg, arg);
    HVX_Vector num = Q6_Vh_vadd_VhVh(arg2, Q6_V_vsplat_R(3));
    HVX_Vector denom = Q6_Vh_vadd_VhVh(Q6_Vh_vmpy_VhVh(arg2, Q6_V_vsplat_R(3)),
                                        Q6_V_vsplat_R(3));
    HVX_Vector tanh_arg = Q6_Vh_vmpy_VhVh(arg, num);
    // Division via reciprocal (use Newton-Raphson or LUT)
    HVX_Vector tanh_approx = Q6_Vh_vmpy_VhVh(tanh_arg, Q6_V_vrecip_Vh(denom));

    // 1 + tanh(...)
    HVX_Vector one = Q6_V_vsplat_R(1);
    HVX_Vector one_plus_tanh = Q6_Vh_vadd_VhVh(one, tanh_approx);

    // 0.5 * x * (1 + tanh(...))
    HVX_Vector half_v = Q6_Vh_vmpy_VhR(v, Q6_R_combine_RR(16384, 16384));  // 0.5 in Q15
    HVX_Vector result = Q6_Vh_vmpy_VhVh(half_v, one_plus_tanh);

    return Q6_Vb_vsaturate_Vh(result);
}
```

**Performance**:
- **LUT**: 1-2 cycles per vector (load from VTCM)
- **Polynomial**: 10-15 cycles per vector (multiple multiplications + approximations)

**Recommendation**: Use LUT for inference (speed), polynomial for training (flexibility).

### 4.3 Sigmoid Activation

**Sigmoid**: Used in RNNs, attention gates.

```
sigmoid(x) = 1 / (1 + exp(-x))
```

**Piecewise linear approximation**:

```
sigmoid(x) ≈ {
    0,                    if x < -4
    0.05 + 0.0625*x,      if -4 ≤ x < 0
    0.5 + 0.0625*x,       if 0 ≤ x < 4
    1,                    if x ≥ 4
}
```

**Implementation**:

```c
HVX_Vector vec_sigmoid_pwl(HVX_Vector v) {
    // Piecewise linear sigmoid
    HVX_Vector v_neg4 = Q6_V_vsplat_R(-4);
    HVX_Vector v_pos4 = Q6_V_vsplat_R(4);
    HVX_Vector vzero = Q6_V_vsplat_R(0);

    // Compare branches
    HVX_Vector cmp_less_neg4 = Q6_Vb_vcmp_gt_VbVb(v, v_neg4);  // NOT(v < -4)
    HVX_Vector cmp_less_pos4 = Q6_Vb_vcmp_gt_VbVb(v, v_pos4);
    HVX_Vector cmp_less_zero = Q6_Vb_vcmp_gt_VbVb(v, vzero);

    // Compute piecewise results
    HVX_Vector result = Q6_V_vsplat_R(0);  // x < -4

    // Branch 1: -4 ≤ x < 0 → 0.05 + 0.0625*x
    HVX_Vector res1 = Q6_Vh_vadd_VhVh(
        Q6_V_vsplat_R(13),  // 0.05 in Q8
        Q6_Vh_vmpy_VhR(v, Q6_R_combine_RR(16, 16))  // 0.0625 in Q8
    );

    // Branch 2: 0 ≤ x < 4 → 0.5 + 0.0625*x
    HVX_Vector res2 = Q6_Vh_vadd_VhVh(
        Q6_V_vsplat_R(128),  // 0.5 in Q8
        Q6_Vh_vmpy_VhR(v, Q6_R_combine_RR(16, 16))  // 0.0625 in Q8
    );

    // Branch 3: x ≥ 4 → 1.0
    HVX_Vector res3 = Q6_V_vsplat_R(255);

    // Select result based on comparisons (conditional move)
    // Simplified: use masks
    // (In practice, use Q6_Vb_vmux_V or vector conditional select)

    return result;
}
```

**LUT approach** (preferred for inference):

```c
int8_t sigmoid_lut[256] = {
    // Precomputed sigmoid values for INT8 input [-128, 127]
    // Maps to output range [0, 127] (approximating 0-1)
    0, 1, 2, 4, 6, 10, 15, 21, 28, 35, 43, 50, 57, 63, 68, 71,
    // ... etc
    127, 127, 127, 127, 127, 127, 127, 127  // Saturation region
};

HVX_Vector vec_sigmoid_lut(HVX_Vector v) {
    // Use LUT: output[i] = sigmoid_lut[v[i] + 128]
    // Similar to GELU LUT
    return Q6_Vb_vlut32_R(v, Q6_R_combine_RR((uint32_t)sigmoid_lut, (uint32_t)sigmoid_lut));
}
```

**Performance**: 1-2 cycles per vector (LUT), 5-10 cycles (polynomial).

### 4.4 Expert Insight & Pitfalls

⚡ **Expert Insight**:
- **LUT quantization**: Ensure LUT entries cover the full input range expected from preceding layers. If activations are clipped (e.g., ReLU6), the LUT can be smaller.
- **Fused activations**: Combine activation with preceding operation (e.g., Conv+ReLU in a single kernel) to save memory bandwidth.
- **Approximation error**: For GELU, a 4-segment piecewise linear has ~3% max error; 8 segments reduces to <1%. Choose based on accuracy requirements.

⚠ **Common Pitfalls**:
1. **Quantization artifacts**: Poor calibration of activation ranges leads to clipping or poor resolution. Use statistics from representative inference data.
2. **Fixed-point overflow**: Intermediate values in polynomial approximation can overflow. Scale carefully or use higher bit-widths temporarily.
3. **LUT initialization**: Ensure LUT is loaded into VTCM once, not for every inference call. Use a static initialization.

---

## Softmax & Normalization

### 5.1 Softmax: Numerically Stable Implementation

**Softmax**: Normalizes logits to a probability distribution.

```
softmax(x)[i] = exp(x[i]) / Σⱼ exp(x[j])
```

**Numerical issue**: exp(x) overflows for large x. Solution: subtract max.

```
softmax(x)[i] = exp(x[i] - max(x)) / Σⱼ exp(x[j] - max(x))
```

**Steps**:
1. Find max(x)
2. Compute y[i] = exp(x[i] - max)
3. Sum all y[i]
4. Divide y[i] by sum

**Hexagon implementation**:

```c
#include <math.h>  // For exp, log

#define SOFTMAX_LUT_BITS 8
#define SOFTMAX_LUT_SIZE (1 << SOFTMAX_LUT_BITS)  // 256

// Precomputed exp LUT for exp(x - max) where x, max ∈ [-128, 127]
// Store as Q16 fixed-point (multiply by 2^16 for integer arithmetic)
int32_t exp_lut[SOFTMAX_LUT_SIZE] = {
    // exp(-256) ≈ 0, exp(-128) ≈ tiny, ..., exp(0) = 1, ..., exp(127) ≈ huge
    // Map to fixed-point for accuracy
};

// Precomputed division LUT for 1/sum
int16_t inv_lut[SOFTMAX_LUT_SIZE] = {
    // inv_lut[i] = 1/i (with scaling)
};

void precompute_softmax_lut() {
    for (int i = 0; i < SOFTMAX_LUT_SIZE; i++) {
        // exp_lut[i] = exp(i - 128) in Q16 fixed-point
        float exp_val = expf(i - 128);
        exp_lut[i] = (int32_t)(exp_val * (1 << 16));

        // Prevent division by zero
        if (i > 0) {
            inv_lut[i] = (int16_t)((1 << 16) / i);
        }
    }
}

void softmax_int8_hvx(
    const int8_t *input,  // [vocab_size]
    int vocab_size,
    int8_t *output        // [vocab_size], values in [0, 127] (approximating probabilities)
) {
    // Step 1: Find max
    int8_t max_val = input[0];
    for (int i = 1; i < vocab_size; i++) {
        if (input[i] > max_val) {
            max_val = input[i];
        }
    }

    // Step 2: Compute exp(x - max) and sum
    int64_t sum = 0;
    int32_t exp_vals[1024];  // Assuming vocab_size ≤ 1024

    for (int i = 0; i < vocab_size; i++) {
        int diff = input[i] - max_val;

        // Clamp diff to [-128, 127] for LUT lookup
        if (diff < -128) diff = -128;
        if (diff > 127) diff = 127;

        // Lookup exp value
        int32_t exp_val = exp_lut[diff + 128];
        exp_vals[i] = exp_val;
        sum += exp_val;
    }

    // Step 3: Divide by sum
    for (int i = 0; i < vocab_size; i++) {
        // output[i] = exp_vals[i] / sum (in range [0, 1])
        // Convert to INT8: multiply by 127

        int32_t prob = (exp_vals[i] << 7) / sum;  // Shift by 7 for Q7 fixed-point
        int8_t out_val = (int8_t)(prob >> 16);

        // Saturate clip to [0, 127]
        if (out_val < 0) out_val = 0;
        if (out_val > 127) out_val = 127;

        output[i] = out_val;
    }
}

/**
 * Vectorized softmax using HVX
 *
 * Key optimizations:
 *  - Horizontal reduction for max (using HVX shuffles)
 *  - Vectorized exp computation via LUT
 *  - Vectorized division
 */
void softmax_int8_hvx_vectorized(
    const int8_t *input,
    int vocab_size,
    int8_t *output
) {
    // Step 1: Find max using horizontal reduction
    HVX_Vector max_vec = Q6_V_vldu_A((HVX_Vector *)&input[0]);

    for (int i = 32; i < vocab_size; i += 32) {
        HVX_Vector v = Q6_V_vldu_A((HVX_Vector *)&input[i]);
        max_vec = Q6_Vb_vmax_VbVb(max_vec, v);
    }

    // Horizontal max (reduce across all lanes)
    // Simplified: perform pairwise max reductions
    HVX_Vector max_h = Q6_V_vror_VR(max_vec, 16);  // Rotate by 2 bytes
    max_vec = Q6_Vb_vmax_VbVb(max_vec, max_h);
    max_h = Q6_V_vror_VR(max_vec, 8);
    max_vec = Q6_Vb_vmax_VbVb(max_vec, max_h);
    max_h = Q6_V_vror_VR(max_vec, 4);
    max_vec = Q6_Vb_vmax_VbVb(max_vec, max_h);

    // Extract scalar max (from lane 0)
    int8_t max_val = ((int8_t *)&max_vec)[0];

    // Step 2 & 3: Compute exp, sum, and divide in vectorized loop
    HVX_Vector sum_vec = Q6_V_vsplat_R(0);

    for (int i = 0; i < vocab_size; i += 32) {
        HVX_Vector v = Q6_V_vldu_A((HVX_Vector *)&input[i]);
        HVX_Vector v_offset = Q6_Vb_vadd_VbVb(v, Q6_V_vsplat_R(-max_val));  // x - max

        // Clamp to [-128, 127]
        HVX_Vector v_clamped = Q6_Vb_vmax_VbVb(v_offset, Q6_V_vsplat_R(-128));
        v_clamped = Q6_Vb_vmin_VbVb(v_clamped, Q6_V_vsplat_R(127));

        // Lookup exp values (gather from exp_lut)
        // Simplified: use intrinsic that doesn't exist in real HVX
        // In practice: loop over lanes or use permutation-based gather

        // sum_vec += exp_vals;
        // (Store intermediate results, then sum at end)
    }

    // Final division and store
    // (Similar to scalar version)
}
```

**Complexity**:
- **Time**: O(vocab_size) for max, O(vocab_size) for exp & sum, O(vocab_size) for divide → O(vocab_size) total
- **Space**: O(vocab_size) for storing exp values
- **Typical vocab size**: 10K-100K (for NLP tasks)

### 5.2 Flash Attention: Efficient Softmax with Long Sequences

For transformer attention with long sequences (e.g., 4K tokens), standard softmax doesn't fit in VTCM:
- Attention scores: [seq_len, seq_len] = [4096, 4096] = 16 MB (if FP32) → exceeds VTCM

**Flash Attention idea**: Process in tiles, with online softmax normalization (avoid storing all scores).

```
# Pseudocode for Flash Attention on Hexagon
for block_k in range(0, seq_len, BLOCK_K):
    for block_q in range(0, seq_len, BLOCK_Q):
        # Load Q[BLOCK_Q, d], K[BLOCK_K, d], V[BLOCK_K, d] into VTCM
        # Compute S = Q @ K^T → [BLOCK_Q, BLOCK_K] (fits in VTCM)
        # Apply softmax (online, row-wise)
        # Multiply by V → output[BLOCK_Q, d] (accumulate)
```

**Advantages**:
- **Memory bound**: Only load Q, K, V once (optimal)
- **VTCM-friendly**: Fits blocks of attention scores

---

## Layer Normalization & Fused Operations

### 6.1 Layer Normalization: Two-Pass vs Fused

**Layer Norm**:

```
y = (x - mean) / sqrt(var + eps) * gamma + beta
```

**Two-pass algorithm**:
1. **Pass 1**: Compute mean and variance
2. **Pass 2**: Normalize and scale

**Fused single-pass (Welford's algorithm)**:
- Compute mean and variance in a single pass
- Update mean and variance incrementally

**Two-Pass Implementation**:

```c
void layer_norm_two_pass(
    const int8_t *input,      // [seq_len, hidden_size]
    int seq_len, int hidden_size,
    const int16_t *gamma,     // Scale [hidden_size]
    const int16_t *beta,      // Shift [hidden_size]
    int8_t *output
) {
    // Pass 1: Compute mean and variance
    int32_t sum = 0;
    int32_t sum_sq = 0;

    for (int i = 0; i < seq_len * hidden_size; i++) {
        int32_t x = input[i];
        sum += x;
        sum_sq += x * x;
    }

    float mean = (float)sum / (seq_len * hidden_size);
    float var = (float)sum_sq / (seq_len * hidden_size) - mean * mean;
    float std = sqrtf(var + 1e-6);

    // Pass 2: Normalize and scale
    for (int i = 0; i < seq_len * hidden_size; i++) {
        float norm = ((float)input[i] - mean) / std;
        float scaled = norm * gamma[i % hidden_size] + beta[i % hidden_size];

        // Convert back to INT8
        output[i] = (int8_t)Q6_R_clip_R((int32_t)scaled, 127);
    }
}
```

**Fused Single-Pass (Welford's)**:

```c
void layer_norm_fused_welford(
    const int8_t *input,
    int seq_len, int hidden_size,
    const int16_t *gamma,
    const int16_t *beta,
    int8_t *output
) {
    // Welford's algorithm: compute mean and variance in a single pass

    double mean = 0.0;
    double M2 = 0.0;  // Variance accumulator
    int n = seq_len * hidden_size;

    for (int i = 0; i < n; i++) {
        double x = input[i];
        double delta = x - mean;
        mean += delta / (i + 1);
        double delta2 = x - mean;
        M2 += delta * delta2;
    }

    double var = M2 / n;
    double std = sqrtf(var + 1e-6);

    // Normalize and scale
    for (int i = 0; i < seq_len * hidden_size; i++) {
        float norm = ((float)input[i] - mean) / std;
        float scaled = norm * gamma[i % hidden_size] + beta[i % hidden_size];
        output[i] = (int8_t)Q6_R_clip_R((int32_t)scaled, 127);
    }
}
```

**Hexagon Vectorization**:

```c
void layer_norm_hvx_vectorized(
    const int8_t *input,
    int seq_len, int hidden_size,
    const int16_t *gamma,
    const int16_t *beta,
    int8_t *output
) {
    // Vectorized mean computation
    HVX_Vector vsum = Q6_V_vsplat_R(0);
    HVX_Vector vsum_sq = Q6_V_vsplat_R(0);

    for (int i = 0; i < seq_len * hidden_size; i += 32) {
        HVX_Vector v = Q6_V_vldu_A((HVX_Vector *)&input[i]);

        // Accumulate sum (note: this wraps INT8 values; use INT32 for wider accumulation)
        vsum = Q6_Vb_vadd_VbVb(vsum, v);

        // Accumulate sum of squares (v × v)
        HVX_VectorPair p = Q6_Wh_vmpy_VbVb(v, v);  // INT8 × INT8 → INT16 (packed)
        // ... extract INT32 results from p and accumulate
    }

    // Horizontal sum (reduce vsum and vsum_sq across all lanes)
    // Then compute mean and var

    // Vectorized normalization
    for (int i = 0; i < seq_len * hidden_size; i += 32) {
        HVX_Vector v = Q6_V_vldu_A((HVX_Vector *)&input[i]);

        // Subtract mean (scalar operation)
        HVX_Vector v_centered = Q6_Vb_vsub_VbVb(v, Q6_V_vsplat_R((int8_t)mean));

        // Divide by std (via reciprocal or LUT)
        // (Simplified; real implementation more complex)

        // Multiply by gamma and add beta
        // ...

        Q6_V_vstu_A((HVX_Vector *)&output[i], v_centered);
    }
}
```

### 6.2 Expert Insight & Pitfalls

⚡ **Expert Insight**:
- **Fused operations**: Combine LayerNorm with the preceding operation (e.g., Linear + LayerNorm) to avoid writing intermediate results to memory.
- **Fixed-point arithmetic**: Ensure mean and variance are computed with sufficient precision (use INT32 or INT64 accumulators, then convert to float for sqrt).
- **Channel-wise normalization**: Some architectures use channel-wise (per-channel) or group-wise normalization. Adapt the implementation accordingly.

⚠ **Common Pitfalls**:
1. **Numerical instability**: Variance can become negative due to floating-point errors (rare but catastrophic). Always use `max(var, eps)` before sqrt.
2. **Quantization drift**: If gamma/beta are quantized, accuracy degrades. Keep them in FP32 or use higher bit-widths.
3. **Single-pass approximations**: Welford's method is more stable than computing mean then variance, but still assumes sequential access patterns.

---

## Element-wise Operations

### 7.1 Element-wise Addition with Broadcasting

**Broadcasting rule**: When shapes don't match, expand smaller dims.

```
# Example: add [1, 128] + [256, 128] → [256, 128]
# Broadcast the first tensor from (1, 128) to (256, 128)
```

**Implementation**:

```c
void elemwise_add_int8_hvx(
    const int8_t *a,      // [shape_a]
    const int8_t *b,      // [shape_b]
    int8_t *output,       // [output_shape]
    int size_a, int size_b, int size_out,
    int a_stride_inner, int b_stride_inner
) {
    for (int i = 0; i < size_out; i++) {
        // Compute indices with broadcasting
        int idx_a = (i / a_stride_inner) % (size_a / a_stride_inner);
        int idx_b = (i / b_stride_inner) % (size_b / b_stride_inner);

        int8_t va = a[idx_a * a_stride_inner];
        int8_t vb = b[idx_b * b_stride_inner];

        output[i] = va + vb;  // Simplified; handle overflow
    }
}

/**
 * Vectorized version for common case: broadcast last dimension
 * e.g., [H, W, C] + [C] → [H, W, C]
 */
void elemwise_add_broadcast_int8_hvx(
    const int8_t *input,     // [H, W, C]
    const int8_t *bias,      // [C]
    int H, int W, int C,
    int8_t *output
) {
    for (int hw = 0; hw < H * W; hw++) {
        for (int c = 0; c < C; c += 32) {  // Process 32 channels at a time
            HVX_Vector vin = Q6_V_vldu_A((HVX_Vector *)&input[hw * C + c]);
            HVX_Vector vbias = Q6_V_vldu_A((HVX_Vector *)&bias[c]);

            HVX_Vector vout = Q6_Vb_vadd_VbVb(vin, vbias);

            // Saturate clip if needed
            // vout = Q6_Vb_vsat_VhVh(...);

            Q6_V_vstu_A((HVX_Vector *)&output[hw * C + c], vout);
        }
    }
}
```

### 7.2 Element-wise Multiplication and Fused Operations

**Element-wise multiplication** (e.g., element-wise attention weights):

```c
void elemwise_mul_int8_hvx(
    const int8_t *a,
    const int8_t *b,
    int size,
    int8_t *output
) {
    for (int i = 0; i < size; i += 32) {
        HVX_Vector va = Q6_V_vldu_A((HVX_Vector *)&a[i]);
        HVX_Vector vb = Q6_V_vldu_A((HVX_Vector *)&b[i]);

        // INT8 × INT8 → INT16
        HVX_VectorPair p = Q6_Wh_vmpy_VbVb(va, vb);
        HVX_Vector hi = Q6_V_hi_W(p);
        HVX_Vector lo = Q6_V_lo_W(p);

        // Pack back to INT8 (with rounding/saturation)
        HVX_Vector vout = Q6_Vb_vsaturate_Vh(hi);  // Take high lane, saturate

        Q6_V_vstu_A((HVX_Vector *)&output[i], vout);
    }
}
```

**Fused operations** (e.g., Add + ReLU):

```c
void elemwise_add_relu_int8_hvx(
    const int8_t *a,
    const int8_t *b,
    int size,
    int8_t *output
) {
    HVX_Vector vzero = Q6_V_vsplat_R(0);

    for (int i = 0; i < size; i += 32) {
        HVX_Vector va = Q6_V_vldu_A((HVX_Vector *)&a[i]);
        HVX_Vector vb = Q6_V_vldu_A((HVX_Vector *)&b[i]);

        HVX_Vector vsum = Q6_Vb_vadd_VbVb(va, vb);
        HVX_Vector vout = Q6_Vb_vmax_VbVb(vsum, vzero);  // ReLU

        Q6_V_vstu_A((HVX_Vector *)&output[i], vout);
    }
}
```

---

## Pooling Operations

### 8.1 Max Pooling and Average Pooling

**Max pooling**: Compute maximum over each K×K window.

```
max_pool(input)[h, w, c] = max_{kh, kw} input[h*stride + kh, w*stride + kw, c]
```

**Implementation**:

```c
void maxpool_int8_hvx(
    const int8_t *input,      // [H, W, C]
    int H, int W, int C,
    int K, int stride, int padding,
    int8_t *output            // [H_out, W_out, C]
) {
    int H_out = (H + 2*padding - K) / stride + 1;
    int W_out = (W + 2*padding - K) / stride + 1;

    for (int oh = 0; oh < H_out; oh++) {
        for (int ow = 0; ow < W_out; ow++) {
            for (int c = 0; c < C; c += 32) {  // Process 32 channels at a time
                HVX_Vector vmax = Q6_V_vsplat_R(-128);  // MIN_INT8

                for (int kh = 0; kh < K; kh++) {
                    for (int kw = 0; kw < K; kw++) {
                        int ih = oh * stride + kh - padding;
                        int iw = ow * stride + kw - padding;

                        if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                            HVX_Vector v = Q6_V_vldu_A((HVX_Vector *)&input[ih * W * C + iw * C + c]);
                            vmax = Q6_Vb_vmax_VbVb(vmax, v);
                        }
                    }
                }

                Q6_V_vstu_A((HVX_Vector *)&output[oh * W_out * C + ow * C + c], vmax);
            }
        }
    }
}
```

**Average pooling**:

```c
void avgpool_int8_hvx(
    const int8_t *input,
    int H, int W, int C,
    int K, int stride, int padding,
    int8_t *output
) {
    int H_out = (H + 2*padding - K) / stride + 1;
    int W_out = (W + 2*padding - K) / stride + 1;

    int count = K * K;
    int count_inv = (1 << 16) / count;  // Fixed-point reciprocal

    for (int oh = 0; oh < H_out; oh++) {
        for (int ow = 0; ow < W_out; ow++) {
            for (int c = 0; c < C; c += 32) {
                HVX_Vector vsum = Q6_V_vsplat_R(0);

                for (int kh = 0; kh < K; kh++) {
                    for (int kw = 0; kw < K; kw++) {
                        int ih = oh * stride + kh - padding;
                        int iw = ow * stride + kw - padding;

                        if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                            HVX_Vector v = Q6_V_vldu_A((HVX_Vector *)&input[ih * W * C + iw * C + c]);
                            // For proper averaging, need to accumulate in higher precision
                            // Simplified: just sum
                            vsum = Q6_Vb_vadd_VbVb(vsum, v);
                        }
                    }
                }

                // Divide by count (shift or use reciprocal)
                HVX_Vector vavg = Q6_Vh_vmpy_VhR(Q6_Wb_vunpack_V(vsum), Q6_R_combine_RR(count_inv, count_inv));
                vavg = Q6_Vb_vsaturate_Vh(vavg);

                Q6_V_vstu_A((HVX_Vector *)&output[oh * W_out * C + ow * C + c], vavg);
            }
        }
    }
}
```

### 8.2 Global Pooling

**Global average pooling**: Average over entire spatial dimensions.

```
global_avgpool(input)[c] = mean_{h, w} input[h, w, c]
```

**Implementation**:

```c
void global_avgpool_int8_hvx(
    const int8_t *input,   // [H, W, C]
    int H, int W, int C,
    int8_t *output         // [C]
) {
    for (int c = 0; c < C; c++) {
        int64_t sum = 0;

        for (int h = 0; h < H; h++) {
            for (int w = 0; w < W; w++) {
                sum += input[h * W * C + w * C + c];
            }
        }

        int32_t avg = sum / (H * W);
        output[c] = (int8_t)Q6_R_clip_R(avg, 127);
    }
}

/**
 * Vectorized global average pooling (process multiple channels in parallel)
 */
void global_avgpool_hvx_vectorized(
    const int8_t *input,
    int H, int W, int C,
    int8_t *output
) {
    // Initialize accumulators for all channels
    int32_t *accums = (int32_t *)malloc(C * sizeof(int32_t));
    memset(accums, 0, C * sizeof(int32_t));

    // Sum all spatial positions
    for (int hw = 0; hw < H * W; hw++) {
        for (int c = 0; c < C; c++) {
            accums[c] += input[hw * C + c];
        }
    }

    // Average and store
    int count = H * W;
    for (int c = 0; c < C; c++) {
        int32_t avg = accums[c] / count;
        output[c] = (int8_t)Q6_R_clip_R(avg, 127);
    }

    free(accums);
}
```

---

## Tensor Layout Transformations

### 9.1 Transpose, Reshape, and Slice Operations

**Challenge**: These operations are often memory-bound (few arithmetic operations, high bandwidth).

**Strategy**: Use zero-copy when possible; only materialize data when necessary.

### 9.1.1 Transpose

**Naive approach**: Read input in row-major, write output in column-major.

```c
void transpose_int8(
    const int8_t *input,   // [H, W]
    int H, int W,
    int8_t *output         // [W, H]
) {
    for (int h = 0; h < H; h++) {
        for (int w = 0; w < W; w++) {
            output[w * H + h] = input[h * W + w];
        }
    }
}
```

**Cache-oblivious recursive transpose** (avoids unnecessary memory traffic):

```c
void transpose_recursive(
    const int8_t *input, int input_stride,
    int8_t *output, int output_stride,
    int H, int W, int base_case_size
) {
    if (H <= base_case_size && W <= base_case_size) {
        // Base case: perform naive transpose
        for (int h = 0; h < H; h++) {
            for (int w = 0; w < W; w++) {
                output[w * output_stride + h] = input[h * input_stride + w];
            }
        }
    } else if (H > W) {
        // Divide height in half
        int H_half = H / 2;
        transpose_recursive(input, input_stride,
                           output, output_stride,
                           H_half, W, base_case_size);
        transpose_recursive(input + H_half * input_stride, input_stride,
                           output + H_half, output_stride,
                           H - H_half, W, base_case_size);
    } else {
        // Divide width in half
        int W_half = W / 2;
        transpose_recursive(input, input_stride,
                           output, output_stride,
                           H, W_half, base_case_size);
        transpose_recursive(input + W_half, input_stride,
                           output + W_half * output_stride, output_stride,
                           H, W - W_half, base_case_size);
    }
}
```

**HVX-accelerated transpose** (for small matrices):

```c
void transpose_hvx(
    const int8_t *input,   // [H, W]
    int H, int W,
    int8_t *output         // [W, H]
) {
    // HVX shuffle/deal instructions for fast transpose of 32×32 blocks
    // (Pseudo-code; real implementation uses Q6_Vb_vdeal/vshuff)

    for (int bh = 0; bh < H; bh += 32) {
        for (int bw = 0; bw < W; bw += 32) {
            // Load 32×32 block from input
            HVX_Vector block[32];
            for (int h = 0; h < 32; h++) {
                block[h] = Q6_V_vldu_A((HVX_Vector *)&input[(bh + h) * W + bw]);
            }

            // Transpose block in-place using vshuff
            // (Details depend on HVX version and instruction set)

            // Store transposed block to output
            for (int w = 0; w < 32; w++) {
                Q6_V_vstu_A((HVX_Vector *)&output[(bw + w) * H + bh], block[w]);
            }
        }
    }
}
```

### 9.1.2 Reshape

**Zero-copy**: If only reinterpreting data layout, no computation needed.

```c
// Zero-copy reshape
int8_t *reshape_zero_copy(
    int8_t *input,
    int old_shape[4], int old_strides[4],  // Shape and strides of input
    int new_shape[4], int new_strides[4]   // Target shape and strides
) {
    // Just return the same pointer; reinterpret strides
    // Example: [2, 3, 4, 5] with shape [6, 20] still points to same memory

    // Verification: ensure old and new shapes have same total size
    if (old_shape[0] * old_shape[1] * old_shape[2] * old_shape[3] ==
        new_shape[0] * new_shape[1] * new_shape[2] * new_shape[3]) {
        return input;  // Zero-copy!
    }

    return NULL;  // Requires data movement
}
```

**Data-copy reshape** (if zero-copy impossible):

```c
void reshape_with_copy(
    const int8_t *input,
    int input_numel,
    int8_t *output
) {
    memcpy(output, input, input_numel);
}
```

### 9.1.3 Slice

**Zero-copy slicing** (common case):

```c
// Slicing is usually zero-copy if striding properly
int8_t *slice_zero_copy(
    int8_t *input,
    int old_strides[4],
    int slice_start[4], int slice_size[4]
) {
    // Return pointer offset + new strides
    int8_t *sliced_ptr = input;
    for (int i = 0; i < 4; i++) {
        sliced_ptr += slice_start[i] * old_strides[i];
    }

    // New strides remain the same as old strides
    return sliced_ptr;
}
```

### 9.2 Expert Insight & Pitfalls

⚡ **Expert Insight**:
- **Fusion is key**: Always fuse tensor layout transformations with preceding/following compute. E.g., if Conv2D is followed by transpose, implement it within the Conv2D kernel to avoid extra memory I/O.
- **Strided layouts**: Modern frameworks (TensorFlow, PyTorch) support strided tensors, allowing zero-copy slicing, transposing, and reshaping. Hexagon operators should exploit this.
- **Tiling for large transposes**: For very large matrices that don't fit in VTCM, tile into 256×256 (or smaller) blocks and transpose each block independently.

⚠ **Common Pitfalls**:
1. **Assuming contiguous memory**: Many operators assume input is contiguous (stride[i] = shape[i+1] * stride[i+1]). Verify this; if not, copy to contiguous buffer or adjust loops.
2. **Transpose thrashing**: Naively transposing large matrices in DDR causes severe cache misses. Always use blocked/recursive transpose.
3. **Reshape side effects**: Reshaping can affect data layout assumptions downstream. Document which operators support non-contiguous inputs.

---

## Attention Mechanisms

### 10.1 Multi-Head Self-Attention: Architecture

**Multi-head self-attention** is the core of transformer inference:

```
MultiHeadAttention(Q, K, V):
  for h in 1..num_heads:
    Q_h = Q @ W_q[h]
    K_h = K @ W_k[h]
    V_h = V @ W_v[h]

    scores = softmax(Q_h @ K_h^T / sqrt(d_k))
    output_h = scores @ V_h

  return concat(output_1, ..., output_num_heads) @ W_o
```

**Hexagon challenge**: Attention scores [seq_len, seq_len] can be enormous for long sequences.

**Example dimensions**:
- seq_len = 4096 (tokens)
- d_model = 768 (hidden dimension)
- num_heads = 12
- d_k = d_model / num_heads = 64

**Attention scores**: [4096, 4096] = 16 MB (if FP32) → exceeds VTCM

### 10.2 Complete Attention Implementation

**Approach 1: Tile both Q and K**

```c
#include <hexagon_types.h>
#include <hvx_hexagon_protos.h>
#include <math.h>

typedef struct {
    // Input tensors
    const int8_t *Q;            // [seq_len, d_model] (query)
    const int8_t *K;            // [seq_len, d_model] (key)
    const int8_t *V;            // [seq_len, d_model] (value)

    // Weights
    const int8_t *W_q, *W_k, *W_v, *W_o;  // Weight matrices

    // Dimensions
    int seq_len;
    int d_model;
    int num_heads;
    int d_k, d_v;

    // Quantization
    int32_t q_scale, k_scale, v_scale;
    int32_t output_scale;

    // Output
    int8_t *output;  // [seq_len, d_model]
} AttentionParams;

/**
 * Flash Attention on Hexagon
 *
 * Key idea: Process attention in blocks that fit in VTCM
 *  - Block size: ~32 tokens × d_model = ~24 KB (manageable)
 *  - Maintain online softmax statistics per query position
 *
 * Steps:
 *  1. Project Q, K, V using linear layers
 *  2. Split into multiple heads
 *  3. For each query block and key block:
 *     - Compute Q_block @ K_block^T
 *     - Apply softmax (online)
 *     - Accumulate weighted V
 *  4. Concatenate heads, project to output
 */
void attention_flash_int8_hvx(const AttentionParams *params) {
    int seq_len = params->seq_len;
    int d_model = params->d_model;
    int num_heads = params->num_heads;
    int d_k = params->d_k;
    int d_v = params->d_v;

    // Tile sizes (must fit in VTCM)
    int BLOCK_Q = 32;   // Process 32 queries at a time
    int BLOCK_K = 64;   // Process 64 keys at a time

    // Allocate VTCM buffers
    int8_t *vtcm_Q_proj = (int8_t *)memalign(128, BLOCK_Q * d_model);
    int8_t *vtcm_K_proj = (int8_t *)memalign(128, BLOCK_K * d_model);
    int8_t *vtcm_V_proj = (int8_t *)memalign(128, BLOCK_K * d_model);

    int32_t *vtcm_scores = (int32_t *)memalign(128, BLOCK_Q * BLOCK_K * sizeof(int32_t));
    int32_t *vtcm_accum = (int32_t *)memalign(128, BLOCK_Q * d_model * sizeof(int32_t));

    // For each head
    for (int head = 0; head < num_heads; head++) {
        int head_q_offset = head * d_k;
        int head_k_offset = head * d_k;
        int head_v_offset = head * d_v;

        // For each query block
        for (int q_start = 0; q_start < seq_len; q_start += BLOCK_Q) {
            int q_end = (q_start + BLOCK_Q < seq_len) ? q_start + BLOCK_Q : seq_len;
            int q_len = q_end - q_start;

            // Initialize attention accumulator
            memset(vtcm_accum, 0, q_len * d_model * sizeof(int32_t));

            // For each key block
            for (int k_start = 0; k_start < seq_len; k_start += BLOCK_K) {
                int k_end = (k_start + BLOCK_K < seq_len) ? k_start + BLOCK_K : seq_len;
                int k_len = k_end - k_start;

                // Load Q block: project Q_raw[q_start:q_end, :] @ W_q
                // Simplified: assume pre-projected Q
                // In practice: load from Q, multiply by W_q[head]
                for (int q = 0; q < q_len; q++) {
                    memcpy(&vtcm_Q_proj[q * d_k],
                           &params->Q[(q_start + q) * d_model + head_q_offset],
                           d_k);
                }

                // Load K block
                for (int k = 0; k < k_len; k++) {
                    memcpy(&vtcm_K_proj[k * d_k],
                           &params->K[(k_start + k) * d_model + head_k_offset],
                           d_k);
                }

                // Load V block
                for (int k = 0; k < k_len; k++) {
                    memcpy(&vtcm_V_proj[k * d_v],
                           &params->V[(k_start + k) * d_model + head_v_offset],
                           d_v);
                }

                // Compute attention scores: Q_block @ K_block^T / sqrt(d_k)
                // Shape: [q_len, d_k] @ [d_k, k_len] → [q_len, k_len]
                float sqrt_dk = sqrtf(d_k);

                for (int q = 0; q < q_len; q++) {
                    for (int k = 0; k < k_len; k++) {
                        // Dot product: Q[q, :] · K[k, :]
                        int32_t dot = 0;
                        for (int d = 0; d < d_k; d++) {
                            int8_t q_val = vtcm_Q_proj[q * d_k + d];
                            int8_t k_val = vtcm_K_proj[k * d_k + d];
                            dot += q_val * k_val;
                        }

                        // Scale by 1/sqrt(d_k)
                        int32_t scaled = (int32_t)(dot / sqrt_dk);
                        vtcm_scores[q * k_len + k] = scaled;
                    }
                }

                // Apply softmax to each row of scores
                // (Online softmax to avoid storing full score matrix)
                for (int q = 0; q < q_len; q++) {
                    // Find max in this row
                    int32_t max_score = vtcm_scores[q * k_len];
                    for (int k = 1; k < k_len; k++) {
                        if (vtcm_scores[q * k_len + k] > max_score) {
                            max_score = vtcm_scores[q * k_len + k];
                        }
                    }

                    // Compute exp(score - max) and sum
                    float sum_exp = 0.0f;
                    float exp_scores[k_len];
                    for (int k = 0; k < k_len; k++) {
                        float exp_val = expf((float)(vtcm_scores[q * k_len + k] - max_score) / 128.0f);
                        exp_scores[k] = exp_val;
                        sum_exp += exp_val;
                    }

                    // Accumulate weighted V values
                    for (int k = 0; k < k_len; k++) {
                        float weight = exp_scores[k] / sum_exp;

                        for (int d = 0; d < d_v; d++) {
                            int8_t v_val = vtcm_V_proj[k * d_v + d];
                            vtcm_accum[q * d_model + head_v_offset + d] +=
                                (int32_t)(weight * v_val * 256);  // Fixed-point scaling
                        }
                    }
                }
            }

            // Store attention output for this query block
            for (int q = 0; q < q_len; q++) {
                int out_row = q_start + q;
                for (int d = 0; d < d_model; d++) {
                    int32_t val = vtcm_accum[q * d_model + d];
                    int8_t out_val = (int8_t)(val >> 8);  // Rescale
                    params->output[out_row * d_model + d] =
                        (int8_t)Q6_R_clip_R(out_val, 127);
                }
            }
        }
    }

    // Cleanup
    free(vtcm_Q_proj);
    free(vtcm_K_proj);
    free(vtcm_V_proj);
    free(vtcm_scores);
    free(vtcm_accum);
}

/**
 * High-level attention wrapper
 */
void attention_int8_hvx(
    const int8_t *input,       // [seq_len, d_model]
    const int8_t *W_q,         // [d_model, d_model]
    const int8_t *W_k,         // [d_model, d_model]
    const int8_t *W_v,         // [d_model, d_model]
    const int8_t *W_o,         // [d_model, d_model]
    int seq_len, int d_model, int num_heads,
    int32_t input_scale, int32_t weight_scale, int32_t output_scale,
    int8_t *output             // [seq_len, d_model]
) {
    // Project Q, K, V
    // (Simplified: assume projections already done; in practice, use GEMM)

    AttentionParams params = {
        .Q = input,
        .K = input,
        .V = input,
        .W_q = W_q,
        .W_k = W_k,
        .W_v = W_v,
        .W_o = W_o,
        .seq_len = seq_len,
        .d_model = d_model,
        .num_heads = num_heads,
        .d_k = d_model / num_heads,
        .d_v = d_model / num_heads,
        .q_scale = input_scale,
        .k_scale = input_scale,
        .v_scale = input_scale,
        .output_scale = output_scale,
        .output = output,
    };

    attention_flash_int8_hvx(&params);
}
```

### 10.3 Optimization Strategies for Attention

**Strategy 1: Grouped Query Attention (GQA)**
- Multiple query heads share a single key/value head
- Reduces memory traffic for K and V
- Configuration: num_q_heads = 12, num_kv_heads = 3 (ratio 4:1)

**Strategy 2: Multi-Query Attention (MQA)**
- All query heads share a single key/value head
- Extreme compression; slight accuracy loss
- Configuration: num_q_heads = 12, num_kv_heads = 1

**Strategy 3: Sparse Attention**
- Only compute attention for nearby tokens (local attention)
- E.g., each token only attends to ±32 neighbors
- Reduces complexity from O(n²) to O(n)

**Implementation sketch for sparse attention**:

```c
void sparse_attention_hvx(
    const int8_t *Q, const int8_t *K, const int8_t *V,
    int seq_len, int d_model, int num_heads,
    int window_size,  // Attend to ±window_size neighbors
    int8_t *output
) {
    for (int q = 0; q < seq_len; q++) {
        int k_start = max(0, q - window_size);
        int k_end = min(seq_len, q + window_size + 1);

        // Compute attention only for keys in [k_start, k_end)
        // ... (similar to full attention, but restricted K range)
    }
}
```

### 10.4 Expert Insight & Pitfalls

⚡ **Expert Insight**:
- **Online softmax**: Maintain running max, sum, and intermediate results; avoid storing full score matrix. Reduces memory by 4×.
- **KV cache**: In autoregressive generation, key-value projections don't change. Cache them to avoid recomputation. Store in VTCM or L3 if sequence length is moderate.
- **Quantization in attention**: Attention scores are typically in range [-30, 30] (before softmax). Quantize to INT8 for memory, but be careful with clipping regions (values far from 0).
- **Head parallelism**: Process multiple heads in parallel (different execution units, shared K/V memory). Significant speedup for num_heads > 4.

⚠ **Common Pitfalls**:
1. **Softmax numerical issues**: Avoid overflow by subtracting max. Do this per query row.
2. **Attention collapse**: If temperature is too low (large attention scores), softmax becomes near one-hot, losing gradient information. Watch for this in quantized models.
3. **Sequence length limitations**: Online softmax still requires loading full V matrix, which can be memory-bound for very long sequences (>32K tokens).
4. **Causal masking**: For decoder attention, apply masking before softmax (set future positions to -inf). Implement via selective computation or masking flag.

---

## Performance Tuning & Benchmarking

### 11.1 Hexagon Performance Counters

Enable and read HVX performance counters:

```c
#include <hexagon_standalone.h>

typedef struct {
    uint64_t cycles;
    uint64_t instructions;
    uint64_t hvx_ops;
    uint64_t memory_stalls;
    uint64_t cache_misses;
} PerfCounters;

PerfCounters read_perf_counters() {
    PerfCounters pc = {
        .cycles = hexagon_perf_get_event(0),      // Cycle counter
        .instructions = hexagon_perf_get_event(1), // Instruction count
        .hvx_ops = hexagon_perf_get_event(2),      // HVX vector ops
        .memory_stalls = hexagon_perf_get_event(3), // Memory stalls
        .cache_misses = hexagon_perf_get_event(4),  // L2 cache misses
    };
    return pc;
}

void benchmark_operator(
    const char *op_name,
    void (*op_func)(void),
    int iterations
) {
    hexagon_perf_enable();

    for (int i = 0; i < iterations; i++) {
        op_func();
    }

    PerfCounters pc = read_perf_counters();

    printf("Operator: %s\n", op_name);
    printf("  Cycles: %llu\n", pc.cycles);
    printf("  Instructions: %llu\n", pc.instructions);
    printf("  HVX ops: %llu\n", pc.hvx_ops);
    printf("  Memory stalls: %llu\n", pc.memory_stalls);
    printf("  IPC (inst/cycle): %.2f\n", (float)pc.instructions / pc.cycles);
    printf("  HVX utilization: %.2f%%\n", (float)pc.hvx_ops / pc.instructions * 100);

    hexagon_perf_disable();
}
```

### 11.2 Roofline Analysis

```c
void roofline_analysis(
    const char *op_name,
    uint64_t total_ops,
    uint64_t total_bytes,
    uint64_t cycles
) {
    // Peak throughput: X ops/cycle
    float peak_throughput = 256 / 8;  // 32 INT8 ops per cycle (256-bit ÷ 8)

    // Memory bandwidth: Y GB/s (from DDR or VTCM)
    float peak_bandwidth = 20.0f;  // GB/s from DDR

    // Arithmetic intensity
    float arithmetic_intensity = (float)total_ops / total_bytes;

    // Roofline: achieved throughput = min(peak_throughput, arithmetic_intensity * peak_bandwidth)
    float achieved_throughput = fminf(peak_throughput, arithmetic_intensity * peak_bandwidth);

    // Achieved ops/cycle
    float achieved_ops_per_cycle = (float)total_ops / cycles;

    // Percentage of peak
    float pct_of_peak = (achieved_ops_per_cycle / peak_throughput) * 100;

    printf("Roofline Analysis: %s\n", op_name);
    printf("  Total ops: %llu\n", total_ops);
    printf("  Total bytes: %llu\n", total_bytes);
    printf("  Cycles: %llu\n", cycles);
    printf("  Arithmetic intensity: %.2f ops/byte\n", arithmetic_intensity);
    printf("  Roofline (peak): %.2f ops/cycle\n", peak_throughput);
    printf("  Achieved: %.2f ops/cycle (%.1f%% of peak)\n", achieved_ops_per_cycle, pct_of_peak);
    printf("  Memory-bound: %s\n", arithmetic_intensity < peak_throughput / peak_bandwidth ? "yes" : "no");
}
```

### 11.3 Optimization Workflow

```
1. Profile baseline implementation
   - Identify bottleneck (memory vs compute)
   - Measure current throughput vs roofline

2. Bottleneck analysis
   - If memory-bound: improve data locality, reduce reloads, fuse operations
   - If compute-bound: increase ILP, reduce dependencies, enable dual-issue

3. Micro-optimization
   - Inline functions to reduce call overhead
   - Use intrinsics instead of C code
   - Unroll loops to expose parallelism

4. Measure again
   - Check if performance improved
   - Iterate until 70%+ of roofline achieved
```

---

## Real-World Case Studies

### 12.1 Case Study: MobileNetV2 on Hexagon

**Model**: MobileNetV2, typical variant with 224×224 input, 1000 classes

**Architecture**:
- Initial 3×3 conv (32 filters)
- 17 inverted residual blocks (depthwise separable convs)
- Final 1×1 conv (1280 filters)
- Global average pooling
- FC layer (1000 classes)

**Hexagon optimization approach**:

```
1. Profile bottlenecks:
   - Depthwise convolutions: memory-bound (high arithmetic intensity)
   - 1×1 convolutions: compute-bound (few memory reads per op)
   - FC layer: memory-bound (N=1 batch)

2. Optimization by layer type:
   - Depthwise 3×3: Use tiled direct convolution with careful padding handling
   - 1×1 conv: Fuse with bias and ReLU6 activation
   - FC: Pre-pack weights, use blocked GEMM

3. Quantization strategy:
   - Calibrate on representative data (ImageNet validation set)
   - Use per-channel quantization for weights (better than per-tensor)
   - Track activation ranges; apply clipping where needed

4. Memory layout:
   - Pack weights at model preparation time
   - Input: [224, 224, 3] stored NHWC (channel-last)
   - Weights: [K, K, C_in, C_out] stored optimally for HVX ops

5. Latency target: <100 ms on Snapdragon 888
   - Peak throughput: ~32 INT8 ops/cycle
   - Model: ~500M operations
   - Time: 500M / (1GHz × 32 ops/cycle) ≈ 15.6 ms (theoretical)
   - With 70% efficiency: ~22 ms (achievable)
```

### 12.2 Case Study: BERT Inference (DistilBERT)

**Model**: DistilBERT, 6-layer transformer, 768 hidden dimension

**Architecture**:
- Embedding layer (vocab lookup)
- 6 transformer blocks, each with:
  - Multi-head self-attention (12 heads)
  - Feed-forward (3072 hidden)
  - LayerNorm (×2)
- Classification head

**Hexagon optimization approach**:

```
1. Sequence length challenge:
   - Typical: seq_len = 128-512 tokens
   - Attention scores: [128, 128] = 16 KB (manageable)
   - But for longer sequences (2K): [2048, 2048] = 4 MB (exceeds VTCM)
   - Solution: Use online softmax + flash attention

2. GEMM optimization:
   - Attention: Q @ K^T ([seq_len, d_k] @ [d_k, seq_len])
     - Tile: BLOCK_Q=32, BLOCK_K=64
   - FC in FFN: [seq_len, hidden] @ [hidden, 4×hidden]
     - Tile: BLOCK_M=8, BLOCK_N=16, BLOCK_K=256
   - Output: [seq_len, 4×hidden] @ [4×hidden, hidden]
     - Same tiling as above

3. Fused kernels:
   - GEMM + bias + ReLU (in FF layer)
   - LayerNorm + projection
   - Softmax + GEMM (in attention)

4. Quantization strategy:
   - Per-layer calibration (BertFormatter script)
   - Special handling for attention scores (different range from activations)
   - Keep LayerNorm in FP32 or high-precision INT16 (critical for stability)

5. Latency target: <500 ms per inference on Snapdragon 888
   - 6 layers × 2 attention + FF per layer = ~24 GEMM ops
   - Each GEMM: ~100M ops
   - Total: ~2.4B ops
   - Time: 2.4B / 32 ops/cycle = 75M cycles = 75 ms at 1 GHz (theoretical)
   - With 65% efficiency: ~115 ms (reasonable)
```

---

## Conclusion & Further Learning

### Key Takeaways

1. **Tiling is essential**: Decompose work to fit in VTCM; reduces bandwidth by 10×
2. **Weight packing matters**: Proper layout can improve throughput by 2-3×
3. **Quantization requires care**: INT8 saves memory but demands precise scale factors
4. **Fusion reduces memory traffic**: Combine ops to avoid writing intermediate results
5. **Benchmark thoroughly**: Roofline analysis reveals headroom for optimization

### Advanced Topics Not Covered

- **HMX (Heavy Matrix Extension)**: Specialized hardware for matrix ops (available in newer SoCs)
- **Custom ASIC kernels**: For very specialized operations, consider fixed hardware blocks
- **Multi-threading**: Parallelize across multiple HVX units or ARM cores
- **Energy profiling**: Measure power consumption; optimize for energy-efficiency (critical for mobile)
- **Compiler support**: Leverage auto-vectorization in tools like HVX2 compiler extensions

### References & Resources

- Qualcomm Hexagon SDK documentation
- Hexagon HVX intrinsics reference
- "Flash Attention" paper (Dao et al., 2022)
- PyTorch/TensorFlow quantization guides
- ARM Performance Engineering guides (applicable to Hexagon)

---

**End of Module 6**
**Estimated study time: 60-80 hours**
**Practical projects: Implement 3+ operators; benchmark on real hardware**
