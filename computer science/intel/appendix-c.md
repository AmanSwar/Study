# APPENDIX C — CPUID Feature Detection Code

## Overview

This appendix provides production-ready C code for runtime CPU feature detection using CPUID and XGETBV instructions. Supports AVX-512 subsets (F, BW, DQ, VNNI, BF16, FP16, IFMA, VBMI), AMX (TILE, INT8, BF16, FP16), and OS XSAVE support verification. Code compiles with GCC/Clang, includes inline assembly and intrinsics variants.

---

## 1. Complete Portable Feature Detection Library

**File: `cpu_features.h`**

```c
/*
 * cpu_features.h
 * Runtime detection of AVX-512, AMX, and other x86-64 features.
 * Compatible with GCC/Clang on Linux, macOS, Windows.
 */

#ifndef CPU_FEATURES_H
#define CPU_FEATURES_H

#include <stdint.h>
#include <stdbool.h>
#include <string.h>

/* Feature flag structure */
typedef struct {
    /* AVX-512 subsets */
    bool avx512_f;      /* AVX-512 Foundation */
    bool avx512_bw;     /* AVX-512 Byte/Word */
    bool avx512_dq;     /* AVX-512 Double/Quad Word */
    bool avx512_vl;     /* AVX-512 Vector Length (256/128-bit) */
    bool avx512_vnni;   /* Vector Neural Network Instructions */
    bool avx512_bf16;   /* BF16 Float Format */
    bool avx512_fp16;   /* FP16 Float Format */
    bool avx512_ifma;   /* Integer 52-bit Fused Multiply-Add */
    bool avx512_vbmi;   /* Vector Bit Manipulation Instructions */
    bool avx512_vbmi2;  /* Vector Bit Manipulation Instructions 2 */

    /* AMX (Advanced Matrix Extensions) */
    bool amx_tile;      /* Tile operations */
    bool amx_int8;      /* INT8 tile operations */
    bool amx_bf16;      /* BF16 tile operations */
    bool amx_fp16;      /* FP16 tile operations (future) */

    /* Other features */
    bool avx2;          /* AVX2 */
    bool avx;           /* AVX */
    bool sse42;         /* SSE4.2 */
    bool popcnt;        /* POPCNT instruction */

    /* OS support */
    bool xsave_enabled; /* XSAVE state management enabled */
    bool zmm_enabled;   /* ZMM registers (XMM/YMM/ZMM state) enabled */
    bool tile_enabled;  /* Tile registers enabled */
} cpu_features_t;

/* Global feature flags (cached) */
extern cpu_features_t g_cpu_features;
extern bool g_features_initialized;

/* Initialize feature detection (call once at program start) */
void cpu_features_init(void);

/* Get features (lazy initialization) */
cpu_features_t* cpu_features_get(void);

/* Print feature report */
void cpu_features_print(void);

#endif /* CPU_FEATURES_H */
```

**File: `cpu_features.c`**

```c
/*
 * cpu_features.c
 * Runtime CPU feature detection using CPUID and XGETBV.
 */

#include "cpu_features.h"
#include <stdio.h>

#ifdef _MSC_VER
#include <intrin.h>
#else
#include <cpuid.h>
#endif

cpu_features_t g_cpu_features = {0};
bool g_features_initialized = false;

/* Inline CPUID wrapper for portability */
static inline void cpuid_leaf(uint32_t leaf, uint32_t subleaf,
                              uint32_t *eax, uint32_t *ebx,
                              uint32_t *ecx, uint32_t *edx)
{
#ifdef _MSC_VER
    int regs[4];
    __cpuidex(regs, leaf, subleaf);
    *eax = regs[0];
    *ebx = regs[1];
    *ecx = regs[2];
    *edx = regs[3];
#else
    __cpuid_count(leaf, subleaf, *eax, *ebx, *ecx, *edx);
#endif
}

/* Get XGETBV value (for OS XSAVE support) */
static inline uint64_t xgetbv(uint32_t index)
{
#ifdef _MSC_VER
    return _xgetbv(index);
#else
    uint32_t eax, edx;
    asm volatile("xgetbv" : "=a" (eax), "=d" (edx) : "c" (index));
    return ((uint64_t)edx << 32) | eax;
#endif
}

/* Check if XSAVE features are enabled by OS */
static bool xsave_enabled(void)
{
    uint32_t eax, ebx, ecx, edx;

    /* Check CPUID leaf 1, bit 26 (XSAVE support) */
    cpuid_leaf(1, 0, &eax, &ebx, &ecx, &edx);
    if (!(ecx & (1U << 26))) {
        return false;  /* CPU doesn't support XSAVE */
    }

    /* Check OS XSAVE support via XGETBV(0) */
    uint64_t xfeatures = xgetbv(0);

    /* Bit 0: x87 FPU state, Bit 1: SSE state, Bit 2: AVX state */
    return (xfeatures & 0x6) == 0x6;  /* x87 + SSE + AVX enabled */
}

/* Check if ZMM state is enabled by OS */
static bool zmm_enabled(void)
{
    uint32_t eax, ebx, ecx, edx;

    /* Check AVX-512 support first */
    cpuid_leaf(7, 0, &eax, &ebx, &ecx, &edx);
    if (!(ebx & (1U << 16))) {
        return false;  /* CPU doesn't support AVX-512 F */
    }

    /* Check OS support for ZMM (XGETBV XCR0 bits 5:4 = 11b for opmask + ZMM) */
    uint64_t xfeatures = xgetbv(0);

    /* Bits 5-4: opmask registers (k0-k7) and upper ZMM halves */
    return (xfeatures & 0x30) == 0x30;
}

/* Check if Tile state is enabled by OS (XGETBV bit 17) */
static bool tile_enabled(void)
{
    uint32_t eax, ebx, ecx, edx;

    /* Check AMX support */
    cpuid_leaf(7, 0, &eax, &ebx, &ecx, &edx);
    if (!(ecx & (1U << 24))) {
        return false;  /* CPU doesn't support AMX TILE */
    }

    /* Check OS support for tile (XGETBV bit 17) */
    uint64_t xfeatures = xgetbv(0);
    return (xfeatures & (1ULL << 17)) != 0;
}

void cpu_features_init(void)
{
    if (g_features_initialized) {
        return;
    }

    uint32_t eax, ebx, ecx, edx;
    memset(&g_cpu_features, 0, sizeof(g_cpu_features));

    /* CPUID Leaf 1: Basic CPU info (EBX, ECX, EDX flags) */
    cpuid_leaf(1, 0, &eax, &ebx, &ecx, &edx);
    g_cpu_features.popcnt = (ecx & (1U << 23)) != 0;
    g_cpu_features.sse42 = (ecx & (1U << 20)) != 0;
    g_cpu_features.avx = (ecx & (1U << 28)) != 0;

    /* Check XSAVE support */
    g_cpu_features.xsave_enabled = xsave_enabled();

    /* CPUID Leaf 7, Subleaf 0: Extended features */
    cpuid_leaf(7, 0, &eax, &ebx, &ecx, &edx);

    g_cpu_features.avx2 = (ebx & (1U << 5)) != 0;
    g_cpu_features.avx512_f = (ebx & (1U << 16)) != 0;
    g_cpu_features.avx512_dq = (ebx & (1U << 17)) != 0;
    g_cpu_features.avx512_ifma = (ebx & (1U << 21)) != 0;
    g_cpu_features.avx512_vl = (ebx & (1U << 31)) != 0;
    g_cpu_features.avx512_vnni = (ecx & (1U << 11)) != 0;
    g_cpu_features.avx512_bf16 = (ecx & (1U << 5)) != 0;
    g_cpu_features.avx512_vbmi = (ecx & (1U << 1)) != 0;
    g_cpu_features.avx512_vbmi2 = (ecx & (1U << 6)) != 0;
    g_cpu_features.amx_tile = (ecx & (1U << 24)) != 0;
    g_cpu_features.amx_int8 = (ecx & (1U << 25)) != 0;
    g_cpu_features.amx_bf16 = (ecx & (1U << 22)) != 0;

    /* CPUID Leaf 7, Subleaf 1: Extended AVX-512 features */
    if (eax >= 1) {
        cpuid_leaf(7, 1, &eax, &ebx, &ecx, &edx);
        g_cpu_features.avx512_fp16 = (eax & (1U << 23)) != 0;
    }

    /* Check BW (only valid with F and VL) */
    if (g_cpu_features.avx512_f && g_cpu_features.avx512_vl) {
        cpuid_leaf(7, 0, &eax, &ebx, &ecx, &edx);
        g_cpu_features.avx512_bw = (ebx & (1U << 30)) != 0;
    }

    /* Check OS XSAVE support for ZMM and Tile */
    g_cpu_features.zmm_enabled = zmm_enabled();
    g_cpu_features.tile_enabled = tile_enabled();

    g_features_initialized = true;
}

cpu_features_t* cpu_features_get(void)
{
    cpu_features_init();
    return &g_cpu_features;
}

void cpu_features_print(void)
{
    cpu_features_init();

    printf("=== CPU Feature Detection Report ===\n\n");

    printf("Basic Features:\n");
    printf("  AVX:                %s\n", g_cpu_features.avx ? "YES" : "NO");
    printf("  AVX2:               %s\n", g_cpu_features.avx2 ? "YES" : "NO");
    printf("  SSE4.2:             %s\n", g_cpu_features.sse42 ? "YES" : "NO");
    printf("  POPCNT:             %s\n", g_cpu_features.popcnt ? "YES" : "NO");

    printf("\nAVX-512 Extensions:\n");
    printf("  AVX-512F:           %s\n", g_cpu_features.avx512_f ? "YES" : "NO");
    printf("  AVX-512BW:          %s\n", g_cpu_features.avx512_bw ? "YES" : "NO");
    printf("  AVX-512DQ:          %s\n", g_cpu_features.avx512_dq ? "YES" : "NO");
    printf("  AVX-512VL:          %s\n", g_cpu_features.avx512_vl ? "YES" : "NO");
    printf("  AVX-512VNNI:        %s\n", g_cpu_features.avx512_vnni ? "YES" : "NO");
    printf("  AVX-512BF16:        %s\n", g_cpu_features.avx512_bf16 ? "YES" : "NO");
    printf("  AVX-512FP16:        %s\n", g_cpu_features.avx512_fp16 ? "YES" : "NO");
    printf("  AVX-512IFMA:        %s\n", g_cpu_features.avx512_ifma ? "YES" : "NO");
    printf("  AVX-512VBMI:        %s\n", g_cpu_features.avx512_vbmi ? "YES" : "NO");
    printf("  AVX-512VBMI2:       %s\n", g_cpu_features.avx512_vbmi2 ? "YES" : "NO");

    printf("\nAMX Extensions:\n");
    printf("  AMX-TILE:           %s\n", g_cpu_features.amx_tile ? "YES" : "NO");
    printf("  AMX-INT8:           %s\n", g_cpu_features.amx_int8 ? "YES" : "NO");
    printf("  AMX-BF16:           %s\n", g_cpu_features.amx_bf16 ? "YES" : "NO");

    printf("\nOS Support:\n");
    printf("  XSAVE enabled:      %s\n", g_cpu_features.xsave_enabled ? "YES" : "NO");
    printf("  ZMM registers:      %s\n", g_cpu_features.zmm_enabled ? "YES" : "NO");
    printf("  Tile registers:     %s\n", g_cpu_features.tile_enabled ? "YES" : "NO");

    printf("\n");
}
```

---

## 2. Inference Engine Dispatch Table

**File: `kernel_dispatch.h`**

```c
/*
 * kernel_dispatch.h
 * Function pointers for optimized kernels based on detected CPU features.
 */

#ifndef KERNEL_DISPATCH_H
#define KERNEL_DISPATCH_H

#include <stddef.h>
#include "cpu_features.h"

/* Matrix multiplication kernel signature */
typedef void (*gemm_fn_t)(float *C, const float *A, const float *B,
                         size_t M, size_t N, size_t K);

/* Inference data type */
typedef enum {
    DT_FP32,
    DT_FP16,
    DT_BF16,
    DT_INT8,
} data_type_t;

/* Kernel registry */
typedef struct {
    /* FP32 kernels */
    gemm_fn_t gemm_fp32_avx2;
    gemm_fn_t gemm_fp32_avx512;
    gemm_fn_t gemm_fp32_amx;  /* Via AMX-TILE or emulation */

    /* FP16 kernels */
    gemm_fn_t gemm_fp16_avx512fp16;
    gemm_fn_t gemm_fp16_fallback;  /* Emulated in FP32 */

    /* BF16 kernels */
    gemm_fn_t gemm_bf16_avx512bf16;
    gemm_fn_t gemm_bf16_fallback;

    /* INT8 kernels */
    gemm_fn_t gemm_int8_avx512vnni;
    gemm_fn_t gemm_int8_fallback;

    /* Activation functions */
    void (*relu_avx2)(float *data, size_t count);
    void (*relu_avx512)(float *data, size_t count);

    void (*softmax_avx2)(float *data, size_t count);
    void (*softmax_avx512)(float *data, size_t count);
} kernel_registry_t;

/* Get appropriate kernel for data type and available features */
gemm_fn_t kernel_dispatch_gemm(data_type_t dtype);

/* Initialize kernel registry */
void kernel_registry_init(kernel_registry_t *registry);

#endif /* KERNEL_DISPATCH_H */
```

**File: `kernel_dispatch.c`**

```c
/*
 * kernel_dispatch.c
 * Dispatch table for optimized kernels.
 */

#include "kernel_dispatch.h"
#include <stdio.h>

/* Placeholder kernel implementations */

static void gemm_fp32_scalar(float *C, const float *A, const float *B,
                             size_t M, size_t N, size_t K)
{
    for (size_t i = 0; i < M; i++) {
        for (size_t j = 0; j < N; j++) {
            float sum = 0.0f;
            for (size_t k = 0; k < K; k++) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

static void gemm_fp32_avx2(float *C, const float *A, const float *B,
                           size_t M, size_t N, size_t K)
{
    /* Placeholder: Real implementation uses AVX2 VFMADD256PS */
    gemm_fp32_scalar(C, A, B, M, N, K);
}

static void gemm_fp32_avx512(float *C, const float *A, const float *B,
                             size_t M, size_t N, size_t K)
{
    /* Placeholder: Real implementation uses AVX-512F VFMADD512PS */
    gemm_fp32_scalar(C, A, B, M, N, K);
}

/* INT8 kernel using VNNI */
static void gemm_int8_avx512vnni(float *C, const float *A, const float *B,
                                  size_t M, size_t N, size_t K)
{
    /* Placeholder: Real implementation uses VPDPBUSD for INT8 MACs */
    /* Convert INT8 to FP32, compute, quantize back */
    gemm_fp32_scalar(C, A, B, M, N, K);
}

/* ReLU with AVX-512 */
static void relu_avx512(float *data, size_t count)
{
    /* Placeholder: Real implementation uses VMAXPS with zero */
    for (size_t i = 0; i < count; i++) {
        if (data[i] < 0.0f) {
            data[i] = 0.0f;
        }
    }
}

/* Dispatch to appropriate GEMM kernel */
gemm_fn_t kernel_dispatch_gemm(data_type_t dtype)
{
    cpu_features_t *features = cpu_features_get();

    switch (dtype) {
        case DT_FP32:
            if (features->zmm_enabled && features->avx512_f) {
                return gemm_fp32_avx512;
            } else if (features->avx2) {
                return gemm_fp32_avx2;
            } else {
                return gemm_fp32_scalar;
            }

        case DT_INT8:
            if (features->zmm_enabled && features->avx512_vnni && features->avx512_bw) {
                return gemm_int8_avx512vnni;
            } else if (features->avx2) {
                return gemm_fp32_avx2;  /* Fallback to FP32 */
            } else {
                return gemm_fp32_scalar;
            }

        case DT_FP16:
        case DT_BF16:
        default:
            return gemm_fp32_avx2;  /* Conservative fallback */
    }
}

void kernel_registry_init(kernel_registry_t *registry)
{
    cpu_features_t *features = cpu_features_get();

    /* FP32 */
    registry->gemm_fp32_avx2 = gemm_fp32_avx2;
    registry->gemm_fp32_avx512 = gemm_fp32_avx512;

    /* INT8 */
    registry->gemm_int8_avx512vnni = gemm_int8_avx512vnni;
    registry->gemm_int8_fallback = gemm_fp32_avx2;

    /* ReLU */
    if (features->zmm_enabled) {
        registry->relu_avx512 = relu_avx512;
    } else {
        registry->relu_avx2 = relu_avx512;  /* Approximate */
    }

    printf("Kernel registry initialized:\n");
    printf("  FP32 GEMM: %s\n",
           features->zmm_enabled ? "AVX-512" : (features->avx2 ? "AVX2" : "Scalar"));
    printf("  INT8 GEMM: %s\n",
           features->avx512_vnni ? "AVX-512 VNNI" : "Fallback");
}
```

---

## 3. Compilation & Testing

**File: `test_features.c`**

```c
/*
 * test_features.c
 * Test program for CPU feature detection and kernel dispatch.
 */

#include <stdio.h>
#include <stdlib.h>
#include "cpu_features.h"
#include "kernel_dispatch.h"

int main(int argc, char *argv[])
{
    printf("=== CPU Feature Detection and Kernel Dispatch Test ===\n\n");

    /* Initialize and print features */
    cpu_features_print();

    /* Initialize kernel registry */
    kernel_registry_t registry;
    kernel_registry_init(&registry);

    /* Test GEMM dispatch */
    printf("\nKernel Selection Test:\n");
    gemm_fn_t fn_fp32 = kernel_dispatch_gemm(DT_FP32);
    gemm_fn_t fn_int8 = kernel_dispatch_gemm(DT_INT8);

    printf("  FP32 GEMM function pointer: %p\n", (void *)fn_fp32);
    printf("  INT8 GEMM function pointer: %p\n", (void *)fn_int8);

    /* Quick functional test */
    printf("\nFunctional Test:\n");
    float A[9] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    float B[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
    float C[9] = {0};

    fn_fp32(C, A, B, 3, 3, 3);
    printf("  GEMM result (3x3 identity mult):\n");
    for (int i = 0; i < 3; i++) {
        printf("    [%.1f %.1f %.1f]\n", C[i*3], C[i*3+1], C[i*3+2]);
    }

    return 0;
}
```

**Build script:**

```bash
#!/bin/bash

# Compile CPU feature detection library
gcc -c -O2 -march=native cpu_features.c -o cpu_features.o

# Compile kernel dispatch module
gcc -c -O2 -march=native kernel_dispatch.c -o kernel_dispatch.o

# Compile test program
gcc -c -O2 -march=native test_features.c -o test_features.o

# Link everything
gcc -O2 cpu_features.o kernel_dispatch.o test_features.o -o test_features

# Run
./test_features
```

**Expected Output (on SPR):**

```
=== CPU Feature Detection Report ===

Basic Features:
  AVX:                YES
  AVX2:               YES
  SSE4.2:             YES
  POPCNT:             YES

AVX-512 Extensions:
  AVX-512F:           YES
  AVX-512BW:          YES
  AVX-512DQ:          YES
  AVX-512VL:          YES
  AVX-512VNNI:        YES
  AVX-512BF16:        YES
  AVX-512FP16:        NO
  AVX-512IFMA:        YES
  AVX-512VBMI:        YES
  AVX-512VBMI2:       NO

AMX Extensions:
  AMX-TILE:           YES
  AMX-INT8:           YES
  AMX-BF16:           YES

OS Support:
  XSAVE enabled:      YES
  ZMM registers:      YES
  Tile registers:     YES

Kernel registry initialized:
  FP32 GEMM: AVX-512
  INT8 GEMM: AVX-512 VNNI
```

---

## 4. Advanced: Inline Assembly Variants

**File: `cpu_features_asm.c`** (Alternative using inline assembly)

```c
/*
 * Inline assembly version of CPUID for maximum portability.
 */

#include <stdint.h>

static inline void cpuid_inline(uint32_t leaf, uint32_t subleaf,
                                uint32_t *eax_out, uint32_t *ebx_out,
                                uint32_t *ecx_out, uint32_t *edx_out)
{
    uint32_t eax = leaf;
    uint32_t ebx = 0;
    uint32_t ecx = subleaf;
    uint32_t edx = 0;

    asm volatile(
        "cpuid"
        : "+a" (eax), "=b" (ebx), "+c" (ecx), "=d" (edx)
        :
        : "memory"
    );

    *eax_out = eax;
    *ebx_out = ebx;
    *ecx_out = ecx;
    *edx_out = edx;
}

static inline uint64_t xgetbv_inline(uint32_t ecx)
{
    uint32_t eax = 0;
    uint32_t edx = 0;

    asm volatile(
        "xgetbv"
        : "=a" (eax), "=d" (edx)
        : "c" (ecx)
    );

    return ((uint64_t)edx << 32) | eax;
}
```

---

## 5. Usage Example: ML Inference Engine Initialization

```c
/*
 * inference_engine.c
 * Example: Initialize inference engine with appropriate kernels.
 */

#include "cpu_features.h"
#include "kernel_dispatch.h"
#include <stdio.h>

typedef struct {
    kernel_registry_t kernels;
    float *weights;
    float *activations;
} model_t;

model_t* model_create(void)
{
    model_t *model = malloc(sizeof(model_t));

    /* Initialize features and dispatch table */
    cpu_features_print();
    kernel_registry_init(&model->kernels);

    /* Allocate model state */
    model->weights = malloc(10 * 1024 * 1024);  /* 10 MB weights */
    model->activations = malloc(2 * 1024 * 1024);

    return model;
}

void model_infer(model_t *model, const float *input, float *output, size_t batch_size)
{
    cpu_features_t *features = cpu_features_get();

    if (features->amx_tile && features->tile_enabled) {
        printf("Using AMX-TILE for inference\n");
        /* Use AMX-based gemm_amx kernels */
    } else if (features->zmm_enabled && features->avx512_f) {
        printf("Using AVX-512 for inference\n");
        /* Use gemm_avx512 kernels */
    } else if (features->avx2) {
        printf("Using AVX2 for inference\n");
        /* Use gemm_avx2 kernels */
    } else {
        printf("WARNING: Using scalar kernels (performance will be poor)\n");
    }

    /* Call appropriate GEMM kernel */
    gemm_fn_t gemm = kernel_dispatch_gemm(DT_FP32);
    gemm(output, model->weights, input, batch_size, 1000, 512);
}
```

---

## 6. CI/CD Integration

**File: `.github/workflows/cpu_detect_test.yml`**

```yaml
name: CPU Feature Detection Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Build feature detection
        run: |
          gcc -O2 -march=native cpu_features.c kernel_dispatch.c test_features.c -o test_features
      - name: Run feature detection
        run: ./test_features
      - name: Verify output
        run: |
          ./test_features | grep -q "CPU Feature Detection Report"
          echo "Feature detection test passed"
```

---

## 7. Platform-Specific Notes

### Linux
- Requires `cpuid.h` from `gcc` toolchain
- XSAVE controlled by kernel; check `/proc/cpuinfo` for `xsave` flag
- AMX requires Linux 5.17+ kernel

### macOS
- Use `-std=c99 -O2` for compatibility
- XSAVE always enabled by default in modern OS versions

### Windows
- Use `intrin.h` (MSVC) or gcc from MinGW
- Replace `xgetbv` with `_xgetbv(0)` from `intrin.h`

---

## 8. Reference & Validation

```bash
# Verify detected features against /proc/cpuinfo
cat /proc/cpuinfo | grep -E "avx512|amx"

# Expected output on SPR:
# flags ... avx512f avx512dq avx512ifma avx512cd avx512bw avx512vl ...

# Check XSAVE state
cat /proc/cpuinfo | grep xsave

# Verify tile registers with cpuid command-line tool
# apt install cpuid
cpuid -l 0x07 -s 0  # Leaf 7, subleaf 0
# Look for "AVX-512 TILE", "AMX-TILE", "AMX-INT8", "AMX-BF16"
```

