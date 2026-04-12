# MODULE 16 — Compiler Optimization & Code Generation

## 1. CONCEPTUAL FOUNDATION

Compilers are not magical optimization engines—they are **constrained solvers** working within strict rules:
- Preserve program semantics (never change output for correct inputs)
- Respect pragma directives and compiler flags
- Generate code that meets timing/latency targets

Modern optimizing compilers (GCC 13+, LLVM 17+) integrate **cost models** that estimate instruction latency, throughput, and memory bandwidth. These models guide optimization decisions. Understanding what the compiler *can* and *cannot* do is prerequisite for expert optimization.

### Optimization Levels and Their Semantics

**-O0 (default)**: No optimization. Every C statement maps directly to assembly. Useful for debugging but performance is ~10× worse than -O3.

**-O1**: Enables conservative optimizations:
- Dead code elimination
- Common subexpression elimination
- Loop-invariant code motion
- ~30-40% speedup over -O0

**-O2**: Enables aggressive but "safe" optimizations:
- Inlining (within reasonable limits)
- Loop unrolling (conservative: up to 4× or 8×)
- Vectorization (simple patterns only)
- Software prefetch insertion
- Branch prediction optimization
- ~50-70% speedup over -O1, baseline for production
- **Maintains strict IEEE semantics** (no reordering of floating-point operations)

**-O3**: Aggressive optimizations:
- Aggressive inlining (32× code expansion possible)
- Loop unrolling (up to 64×)
- Vectorization of complex patterns
- Profile-guided optimizations (if PGO data available)
- Cross-module inlining (Link Time Optimization)
- **May violate IEEE semantics** (fused multiply-add, reordering FP ops)
- ~5-15% speedup over -O2 on CPU-intensive code, but risk of code bloat

**-Ofast** (GCC specific): Like -O3 but also enables:
- `-ffast-math`: violates IEEE 754 (NaN, Inf handling)
- `-funsafe-math-optimizations`: associative reordering of FP expressions
- ~5% more speedup over -O3, **dangerous** for numerical applications

**-Os / -Oz**: Optimize for code size instead of speed. Useful for embedded systems and instruction-cache pressure, but generally slower for compute kernels.

Reference: GCC manual, Section "Optimize Options". Agner Fog *Optimizing Software in C++*, Ch. 2.

### Auto-Vectorization: Compiler Capabilities and Limits

Modern compilers auto-vectorize loops under specific conditions:

**Loop must be**:
1. Innermost loop (doesn't contain nested loops)
2. Trip count must be statically determinable (or at least > vector width)
3. Data types must be uniform (no mixed int/float)
4. No data dependencies across iterations (or only forward dependencies)
5. Aligned memory access (or compiler inserts alignment checks)

**Example: Vectorizable**
```c
for (int i = 0; i < N; i++) {
    c[i] = a[i] + b[i];  // ✓ No dependencies, simple operation
}
// Compiler generates: AVX-512 VADD (8 doubles per iteration)
```

**Example: Non-vectorizable**
```c
for (int i = 1; i < N; i++) {
    c[i] = c[i-1] + a[i];  // ✗ Loop-carried dependency on c[i-1]
}
// Must execute serially
```

**Pragmas to guide compiler**:
- `#pragma omp simd` (OpenMP SIMD): assert the loop is vectorizable
- `#pragma GCC ivdep` (GCC): ignore vector dependencies (programmer asserts correctness)
- `#pragma omp simd collapse(2)` : vectorize multiple nested loops
- `#pragma GCC unroll(4)` : specify unroll factor

**Compiler flags**:
- `-fopt-info-vec-all` (GCC) : explain vectorization decisions
- `-march=sapphirerapids` (or `znver4`, `icelake-server`): target ISA

Reference: Agner Fog *Optimizing Software in C++*, Ch. 8-9.

### Profile-Guided Optimization (PGO)

**PGO workflow**:
1. Compile with `-fprofile-generate` → binary with instrumentation
2. Run on representative workload → generates `.profdata` files
3. Recompile with `-fprofile-use` and the `.profdata` → optimized binary

**What PGO enables**:
- **Branch prediction hinting**: compiler moves likely paths to cache-friendly locations (false branches often pushed to distant code)
- **Inlining decisions**: inline only frequently-called functions (reduces code size)
- **Loop unrolling**: unroll only hot loops (saves instruction cache)
- **Register allocation**: prioritize registers for frequently-used variables

**Typical speedup**: 5-20% over -O3 on branch-heavy workloads (inference graphs, dispatch loops).

**Cost**: 2-3× build time (instrument + profile + recompile).

**For ML systems**: If your inference engine dispatches operations based on operator type (conv → gemm → softmax → etc.), PGO learns the dispatch pattern and inlines frequently-called operations. This is particularly effective for autoregressive decoding (same token-by-token pattern repeated).

Reference: GCC manual, Section "Instrumentation Options". LLVM documentation on PGO.

### Link-Time Optimization (LTO)

**Traditional compilation**: each `.o` file optimized in isolation. `main.o` doesn't know the implementation of `library.a::foo()`, so can't inline it.

**LTO workflow**:
1. Compile each `.cpp` with `-flto` → emits LLVM IR (not native code)
2. Link with `-flto` → LTO linker combines all IR, optimizes across module boundaries, generates native code

**What LTO enables**:
- **Interprocedural inlining**: inline function calls across files
- **Dead code elimination**: remove unused functions (only visible after linking)
- **IPO (Interprocedural Optimization)**: propagate constants across functions

**Typical speedup**: 10-20% for inference engines (hot dispatch loop often inlines across module boundaries).

**Cost**: 3-5× link time (IR optimization + code generation).

**For ML systems**: If your inference engine calls DGEMM from a separate library, LTO can inline the dispatch logic and GEMM epilogue, reducing per-operation overhead.

Reference: LLVM documentation on LTO. Agner Fog *Optimizing Software in C++*, Ch. 3.

### LLVM IR: Reading and Debugging Auto-Vectorization

LLVM IR is a machine-independent intermediate representation. Examining it reveals why the compiler made a decision.

**Generate IR**:
```bash
clang -O3 -emit-llvm -S kernel.c -o kernel.ll
# kernel.ll is human-readable LLVM IR
```

**Example IR**: Vectorized add loop
```llvm
%3 = load <8 x double>, <8 x double>* %2, align 64  ; vector load, 64-byte aligned
%4 = load <8 x double>, <8 x double>* %5, align 64
%6 = fadd <8 x double> %3, %4                        ; vector add (8 doubles)
store <8 x double> %6, <8 x double>* %7, align 64
```

**Example IR**: Failed vectorization (loop-carried dependency)
```llvm
%3 = load double, double* %2, align 8  ; scalar load
%4 = fadd double %3, 1.0                ; scalar add (no vectorization)
```

**Reading IR to diagnose**:
- `<8 x double>` = vector type (8 elements)
- `vector load`/`vector add` = vectorized operation
- `scalar load`/`scalar add` = no vectorization (check why)

Use `-fopt-info-vec-all` flag to see compiler's explanation:
```bash
gcc -O3 -fopt-info-vec-all kernel.c 2>&1 | grep -i "note:"
# Output: "note: loop vectorized" or "note: loop not vectorized because..."
```

Reference: LLVM Language Reference Manual. Dendibakh *Performance Analysis and Tuning*, Ch. 8.

### Intrinsics vs Builtins vs Inline Assembly

Three mechanisms to control code generation:

**Intrinsics**: Compiler-provided functions that map directly to CPU instructions. Portable across GCC/Clang.

```c
#include <immintrin.h>

// Intrinsic: _mm512_add_pd
__m512d a = _mm512_set1_pd(1.0);
__m512d b = _mm512_set1_pd(2.0);
__m512d c = _mm512_add_pd(a, b);  // Generates VADDPD instruction
```

**Advantages**:
- Type-safe
- Compiler optimizes register allocation
- Portable across GCC/Clang

**Disadvantages**:
- Limited to predefined operations
- Compiler may insert unnecessary moves

**Builtins**: Compiler-specific functions (usually GCC or Clang only).

```c
// GCC builtin: __builtin_popcountll (count set bits)
long count = __builtin_popcountll(x);  // May compile to POPCNT instruction

// GCC builtin: __builtin_prefetch
__builtin_prefetch(&a[i], 0, 3);  // Hint to prefetch a[i]
```

**Advantages**:
- More flexible than intrinsics
- Access to compiler internals

**Disadvantages**:
- Non-portable (GCC vs Clang syntax differs)
- May not map to instructions on older CPUs

**Inline assembly**: Raw assembly embedded in C.

```c
#include <stdint.h>

// Inline assembly: compute popcount using POPCNT instruction
uint64_t popcount(uint64_t x) {
    uint64_t result;
    asm volatile("popcntq %1, %0" : "=r"(result) : "r"(x));
    return result;
}
```

**Constraints**:
- `"=r"`: output operand in register
- `"r"`: input operand in register
- `"=m"`: output in memory
- `"m"`: input from memory

**Advantages**:
- Maximum control
- Lowest overhead (no compiler interference)

**Disadvantages**:
- Architecture-specific (x86 vs ARM syntax differs)
- Compiler can't optimize across asm boundary
- Risk of register clobbering

**When to use each**:
1. Intrinsics (99%): Default choice, portable, optimizable
2. Builtins (0.9%): Compiler-specific optimizations (__builtin_prefetch, branch hints)
3. Inline asm (0.1%): Final resort when intrinsics can't express the operation

Reference: Agner Fog *Optimizing Software in C++*, Ch. 6.

### Target-Specific Tuning and Function Multiversioning

Modern CPUs have different microarchitectures and ISA extensions. Optimize for the target:

**-march flag selects CPU type**:
- `-march=sapphirerapids` (Intel 4th Gen Xeon Scalable)
- `-march=znver4` (AMD EPYC Genoa)
- `-march=icelake-server` (Intel 3rd Gen Xeon Scalable)

**Each implies**:
- Supported ISAs (AVX-512 for Sapphire Rapids, not for older CPUs)
- Pipeline widths and latencies
- Cache sizes and latencies

**Example**:
```bash
gcc -O3 -march=sapphirerapids kernel.c
# Generates AVX-512 instructions (not available on Haswell)

gcc -O3 -march=haswell kernel.c
# Falls back to AVX2 (no AVX-512)
```

**Function multiversioning**: Compile same function multiple times for different ISAs, select at runtime.

```c
// GCC attribute: create multiple versions
__attribute__((target("avx-512f")))
void gemm_avx512(double *C, double *A, double *B, int M, int N, int K) {
    // Use AVX-512
    for (int i = 0; i < M; i++) {
        __m512d c = _mm512_loadu_pd(&C[i*N]);
        // ... AVX-512 operations
    }
}

__attribute__((target("avx2")))
void gemm_avx2(double *C, double *A, double *B, int M, int N, int K) {
    // Use AVX2
    for (int i = 0; i < M; i++) {
        __m256d c = _mm256_loadu_pd(&C[i*N]);
        // ... AVX2 operations
    }
}

// Wrapper: select at runtime (usually via CPUID)
void gemm(double *C, double *A, double *B, int M, int N, int K) {
    if (__builtin_cpu_supports("avx512f")) {
        return gemm_avx512(C, A, B, M, N, K);
    } else if (__builtin_cpu_supports("avx2")) {
        return gemm_avx2(C, A, B, M, N, K);
    }
}
```

**Alternative: Compiler-supported multiversioning** (GCC 6+, Clang 5+):

```c
__attribute__((target_clones("avx512f", "avx2", "default")))
void gemm_auto(double *C, double *A, double *B, int M, int N, int K) {
    // Compiler generates 3 versions automatically
}
```

**Cost**: 2-3× binary size (multiple versions of hot functions). **Benefit**: single binary works on multiple CPUs, exploits AVX-512 on capable hardware.

Reference: GCC documentation on target attributes. Agner Fog Vol. 1.

---

## 2. MENTAL MODEL

```
┌──────────────────────────────────────────────────────────────────┐
│            OPTIMIZATION LEVEL DECISION TREE                       │
├──────────────────────────────────────────────────────────────────┤
│                                                                    │
│                    Starting Code (semantics correct)              │
│                           │                                       │
│                    -O0 (No Optimization)                          │
│         │ add: -O1     │ add: -O2     │ add: -O3                │
│         │ flags        │ flags        │ flags                    │
│         ▼              ▼              ▼                           │
│    ~1 GB/s         ~3 GB/s          ~4 GB/s      (example GEMM)  │
│    (baseline)   (conservative)   (aggressive)                    │
│                                                                    │
│  Which to choose:                                                │
│  • Development: -O0 + symbols (fast build, easy debug)           │
│  • Testing: -O2 (realistic perf, safe semantics)                 │
│  • Benchmarking: -O3 (peak perf, production-representative)      │
│  • Numerical work: -O2 only (avoid -Ofast due to IEEE violations)│
│                                                                    │
│  Special flags:                                                   │
│  • -fprofile-generate: instrument for PGO                        │
│  • -flto: enable cross-module optimization                       │
│  • -march=sapphirerapids: target specific CPU                    │
│  • -fopt-info-vec-all: explain vectorization decisions           │
│                                                                    │
└──────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────┐
│     AUTO-VECTORIZATION DECISION FLOW (GCC/Clang)                 │
├──────────────────────────────────────────────────────────────────┤
│                                                                    │
│  Loop detected (for, while, do-while)                           │
│           │                                                      │
│           ├─ Innermost? ──No──→ Skip (nested loops hard)        │
│           │                                                      │
│           └─ Yes                                                │
│              │                                                   │
│              ├─ Trip count known? ──No──→ Check if > vec_width │
│              │                                                   │
│              └─ Yes                                              │
│                 │                                                │
│                 ├─ Data types uniform? ──No──→ Skip            │
│                 │                                                │
│                 └─ Yes                                           │
│                    │                                             │
│                    ├─ Loop-carried dependencies? ──Yes──→ Skip  │
│                    │                                             │
│                    └─ No (or only forward deps)                 │
│                       │                                          │
│                       ├─ Memory access aligned? ──No──→ Add     │
│                       │                           check logic    │
│                       ├─ Reduction? ──Yes──→ Check if supported │
│                       │                                          │
│                       └─ Yes, vectorizable!                     │
│                          │                                       │
│                          └─ Generate vector code                │
│                             (e.g., VADD <8 x double>)          │
│                                                                  │
│  Compiler output:                                               │
│    gcc -O3 -fopt-info-vec-all kernel.c                         │
│    note: loop vectorized for SIMD width 8                      │
│    note: loop not vectorized: too large trip count             │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────┐
│         PROFILE-GUIDED OPTIMIZATION (PGO) WORKFLOW                │
├──────────────────────────────────────────────────────────────────┤
│                                                                    │
│  Step 1: Compile with instrumentation                           │
│  $ gcc -O3 -fprofile-generate=profdata kernel.c -o kernel_instr │
│                                                                    │
│  Step 2: Run on representative workload                         │
│  $ ./kernel_instr < input_data.txt                              │
│  → Generates profdata/kernel.gcda, profdata/kernel.gcno         │
│                                                                    │
│  Step 3: Recompile with profile data                            │
│  $ gcc -O3 -fprofile-use=profdata -flto kernel.c -o kernel_opt  │
│                                                                    │
│  Compiler learns:                                                │
│  • Which branches are taken 99% of the time                     │
│  • Which functions are called in hot paths                      │
│  • Which loops iterate > 100,000 times                          │
│                                                                    │
│  Optimization decisions:                                         │
│  • Hot branches: move to cache-friendly alignment                │
│  • Hot functions: inline them (reduces call overhead)            │
│  • Hot loops: unroll aggressively                                │
│                                                                    │
│  Result: 5-20% speedup on branch-heavy code                     │
│                                                                    │
│  Cost: 3× build time (instrument + profile + recompile)         │
│                                                                    │
└──────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────┐
│         INTRINSIC vs BUILTIN vs INLINE ASSEMBLY                  │
├──────────────────────────────────────────────────────────────────┤
│                                                                    │
│  Use Case              First Choice       Second Choice  Avoid   │
│  ──────────────────────────────────────────────────────────────  │
│  Vector add            _mm512_add_pd      __builtin_vadd inline  │
│  Prefetch              _mm_prefetch       __builtin_prefetch asm │
│  Population count      __builtin_popcountll (varies)  asm        │
│  POPCOUNT instruction  intrinsic if exists, else asm             │
│  Complex operation     inline asm         (careful!)             │
│                                                                    │
│  Selection rationale:                                            │
│  • Intrinsics: type-safe, portable, compiler optimizes regalloc │
│  • Builtins: compiler-specific, access to compiler optimizations│
│  • Inline asm: maximum control, final resort                     │
│                                                                    │
│  Example: Vector multiply-add                                    │
│  ┌─ Intrinsic (preferred) ──────────────────────────────────┐   │
│  │ __m512d a = _mm512_set1_pd(1.0);                         │   │
│  │ __m512d b = _mm512_set1_pd(2.0);                         │   │
│  │ __m512d c = _mm512_set1_pd(3.0);                         │   │
│  │ __m512d result = _mm512_fmadd_pd(a, b, c);  // a*b + c   │   │
│  │ // Compiler: register allocation, CSE, loop unrolling    │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                    │
│  Example: Inline assembly (fallback)                             │
│  ┌─ Inline asm ──────────────────────────────────────────────┐   │
│  │ __m512d result;                                            │   │
│  │ asm volatile("vfmadd213pd %2, %1, %0"                      │   │
│  │             : "+x"(result)  // output: a*b+c in result    │   │
│  │             : "x"(a), "x"(b), "x"(c));  // inputs         │   │
│  │ // Compiler: minimal optimization, exact control          │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                    │
└──────────────────────────────────────────────────────────────────┘
```

---

## 3. PERFORMANCE LENS

**Optimization Level Trade-offs**:
- `-O2`: Safe baseline. 50-70% speedup from -O0. Maintains IEEE FP semantics. Recommended for production.
- `-O3`: Aggressive. Additional 5-15% over -O2, but risks code bloat (instruction cache misses) and IEEE violations.
- PGO: 5-20% over -O3 on branch-heavy code (inference dispatch, autoregressive decoding).
- LTO: 10-20% over -O3 on code with heavy cross-module calls.

**Auto-Vectorization**:
If compiler vectorizes a loop, you get **6-8× throughput improvement** (8 double-precision elements per SIMD instruction). This is often the largest single optimization available.

If vectorization fails, manual vectorization with intrinsics is the next opportunity (similar speedup).

**Target-Specific Compilation**:
- AVX-512 on Sapphire Rapids: 2× throughput vs AVX2 (8 vs 4 double elements per instruction).
- AVX2 on older CPUs: 4× throughput vs scalar (4 double elements).
- Multiversioning allows single binary to exploit AVX-512 on capable hardware while remaining compatible.

**Code Profiling with Compiler Output**:
- `-fopt-info-vec-all`: tells you why vectorization failed (loop-carried dependency? unaligned memory? unknown trip count?)
- `-S -masm=intel`: generates readable assembly (inspect what compiler actually generated)
- LLVM IR inspection: debug optimization failures at IR level (before register allocation)

---

## 4. ANNOTATED CODE

### Example 1: Vectorization Pragmas and Compiler Output

```c
// File: vectorization_example.c
// Demonstrate compiler vectorization decisions and pragmas

#include <stdio.h>
#include <math.h>

// Case 1: Simple loop, compiler should vectorize automatically
void simple_add(double *c, const double *a, const double *b, int n) {
    for (int i = 0; i < n; i++) {
        c[i] = a[i] + b[i];  // ✓ Vectorizable: no dependencies
    }
}

// Case 2: Loop with assumed aliasing, compiler may not vectorize
void aliased_add(double *c, const double *a, const double *b, int n) {
    // ✗ Without __restrict__, compiler assumes c might overlap a or b
    // Example: c = a; then aliased_add(a, a, b, n) would be incorrect
    for (int i = 0; i < n; i++) {
        c[i] = a[i] + b[i];
    }
}

// Case 3: Same loop, but with __restrict, compiler vectorizes
void restricted_add(double *restrict c, const double *restrict a,
                    const double *restrict b, int n) {
    // ✓ __restrict tells compiler: c, a, b don't overlap
    for (int i = 0; i < n; i++) {
        c[i] = a[i] + b[i];
    }
}

// Case 4: Loop-carried dependency, not vectorizable
void reduction_sum(double *result, const double *a, int n) {
    double sum = 0.0;
    for (int i = 0; i < n; i++) {
        sum += a[i];  // ✗ Loop-carried: iteration i+1 depends on sum from iteration i
    }
    *result = sum;
}

// Case 5: Reduction with pragma override (compiler parallelizes via OpenMP)
#pragma omp declare simd
double reduction_sum_omp(const double *a, int n) {
    double sum = 0.0;
    #pragma omp simd reduction(+:sum)
    for (int i = 0; i < n; i++) {
        sum += a[i];  // ✓ Pragma tells compiler: handle reduction (uses SIMD reduction operation)
    }
    return sum;
}

// Case 6: GEMM (matrix multiply), vectorizable in inner loop
void gemm_simple(double *C, const double *A, const double *B,
                 int M, int N, int K) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            double sum = 0.0;
            for (int k = 0; k < K; k++) {
                sum += A[i*K + k] * B[k*N + j];  // ✗ Innermost loop has reduction
            }
            C[i*N + j] = sum;
        }
    }
}

// Case 7: GEMM with pragma, vectorizable
void gemm_vectorized(double *C, const double *A, const double *B,
                     int M, int N, int K) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            double sum = 0.0;
            #pragma omp simd reduction(+:sum)
            for (int k = 0; k < K; k++) {
                sum += A[i*K + k] * B[k*N + j];  // ✓ Pragma enables SIMD reduction
            }
            C[i*N + j] = sum;
        }
    }
}

// Case 8: Manual vectorization with intrinsics
#include <immintrin.h>

void gemm_intrinsics(double *C, const double *A, const double *B,
                     int M, int N, int K) {
    // Compute in 8×8 tiles using AVX-512
    for (int i = 0; i < M; i += 8) {
        for (int j = 0; j < N; j += 8) {
            // Initialize C[i:i+8, j:j+8] to zero
            __m512d sum[8];
            for (int ii = 0; ii < 8; ii++) {
                sum[ii] = _mm512_setzero_pd();
            }

            // Inner product loop (vectorized over K)
            for (int k = 0; k < K; k++) {
                // Load A[i:i+8, k] (8 elements)
                __m512d a = _mm512_loadu_pd(&A[(i)*K + k]);

                // FMA: sum += a * B[k, j:j+8]
                for (int ii = 0; ii < 8; ii++) {
                    __m512d b = _mm512_set1_pd(B[k*N + (j+ii)]);
                    sum[ii] = _mm512_fmadd_pd(a, b, sum[ii]);  // 8 FMA per iteration
                }
            }

            // Store result
            for (int ii = 0; ii < 8; ii++) {
                _mm512_storeu_pd(&C[(i+ii)*N + j], sum[ii]);
            }
        }
    }
}

// Compilation guide:
//
// Observe compiler decisions:
// $ gcc -O3 -fopt-info-vec-all vectorization_example.c 2>&1 | grep -E "vectorized|not vectorized"
//
// Output:
//   note: loop vectorized for SIMD width 4, cost model says it's profitable
//   note: loop not vectorized: loop with multiple exits
//
// Generate assembly to inspect:
// $ gcc -O3 -S -masm=intel vectorization_example.c
// $ grep -A 5 "simple_add:" vectorization_example.s
//
// Expected (vectorized):
//   vmovupd ymm0, [rdi]      # load 4 doubles from c
//   vaddpd ymm0, [rsi]       # add a, result in ymm0
//   vmovupd [rdx], ymm0      # store to c
//   add rdi, 32              # advance pointers by 32 bytes (4 * 8 bytes)
//   cmp rdi, rcx
//   jne loop_start            # repeat until n=0
//
// Generate LLVM IR to debug vectorization failure:
// $ clang -O3 -emit-llvm -S vectorization_example.c -o vectorization_example.ll
// $ grep -A 3 "simple_add" vectorization_example.ll
//
// Expected (vectorized):
//   <4 x double>* → SIMD type found
//   @llvm.fmadd.v4f64 → vector operation found
```

**Compilation and analysis**:
```bash
# Compile with -fopt-info-vec to see vectorization decisions
gcc -O3 -fopt-info-vec-all vectorization_example.c -c 2>&1

# Output shows which loops vectorized:
#   note: loop vectorized for SIMD width 4
#   note: loop not vectorized: loop with multiple exits
#   note: loop not vectorized: loop with unresolved mem references

# Generate assembly to verify
gcc -O3 -S -masm=intel vectorization_example.c
grep -B2 "vmovupd\|vaddpd\|vfmadd" vectorization_example.s

# Generate LLVM IR (Clang)
clang -O3 -emit-llvm -S vectorization_example.c
# Read the .ll file to see vectorized operations (<8 x double> types)
```

### Example 2: Profile-Guided Optimization (PGO) Workflow

```c
// File: inference_dispatch.c
// Simulate inference engine with operator dispatch
// Ideal candidate for PGO (branch prediction matters heavily)

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// Operator types
enum op_type {
    OP_CONV = 0,      // Conv2D
    OP_GEMM = 1,      // Dense matrix multiply
    OP_SOFTMAX = 2,   // Softmax
    OP_RELU = 3,      // ReLU activation
    OP_NORM = 4,      // Layer norm
};

// Simulate operator implementations
static void conv_kernel(double *out, const double *in, const double *weights,
                        int H, int W, int C) {
    // Compute H×W×C×K operations (expensive)
    for (int i = 0; i < H*W*C; i++) {
        out[i] = in[i] * weights[i % 64];  // simplification
    }
}

static void gemm_kernel(double *out, const double *A, const double *B,
                        int M, int N, int K) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            double sum = 0.0;
            for (int k = 0; k < K; k++) {
                sum += A[i*K + k] * B[k*N + j];
            }
            out[i*N + j] = sum;
        }
    }
}

static void softmax_kernel(double *out, const double *in, int n) {
    // Simplified softmax
    double max_val = in[0];
    for (int i = 1; i < n; i++) {
        if (in[i] > max_val) max_val = in[i];
    }
    double sum = 0.0;
    for (int i = 0; i < n; i++) {
        out[i] = exp(in[i] - max_val);
        sum += out[i];
    }
    for (int i = 0; i < n; i++) {
        out[i] /= sum;
    }
}

static void relu_kernel(double *out, const double *in, int n) {
    for (int i = 0; i < n; i++) {
        out[i] = (in[i] > 0.0) ? in[i] : 0.0;
    }
}

static void norm_kernel(double *out, const double *in, int n) {
    double mean = 0.0;
    for (int i = 0; i < n; i++) mean += in[i];
    mean /= n;

    double var = 0.0;
    for (int i = 0; i < n; i++) {
        double delta = in[i] - mean;
        var += delta * delta;
    }
    var /= n;

    for (int i = 0; i < n; i++) {
        out[i] = (in[i] - mean) / sqrt(var + 1e-5);
    }
}

// Dispatch function: branch for each operator type
// WITHOUT PGO: all branches are equally likely (compiler predicts poorly)
// WITH PGO: compiler learns operator frequency and predicts better
void dispatch_operator(double *out, const double *in, const double *weights,
                       enum op_type op, int H, int W, int C, int K) {
    // After PGO profile run on transformer inference,
    // compiler learns: GEMM is 60%, SOFTMAX 20%, others 20%
    // → moves GEMM branch code to hot cache location
    // → moves cold branches (NORM, RELU) to distant code

    if (op == OP_CONV) {
        conv_kernel(out, in, weights, H, W, C);
    } else if (op == OP_GEMM) {
        gemm_kernel(out, in, weights, H, W, C);
    } else if (op == OP_SOFTMAX) {
        softmax_kernel(out, in, H*W);
    } else if (op == OP_RELU) {
        relu_kernel(out, in, H*W*C);
    } else if (op == OP_NORM) {
        norm_kernel(out, in, H*W*C);
    }
}

// Simulate a forward pass (typical operator sequence)
void inference_forward(double *data, int iterations) {
    double *temp1 = malloc(sizeof(double) * 512 * 512);
    double *temp2 = malloc(sizeof(double) * 512 * 512);
    double *weights = malloc(sizeof(double) * 512 * 512);

    // Typical inference pattern (from real transformer profiling):
    // 60% of dispatch calls are GEMM
    // 20% are SOFTMAX
    // 15% are NORM
    // 5% are RELU, CONV (rare)

    for (int iter = 0; iter < iterations; iter++) {
        // Layer 0: Conv (once)
        if (iter == 0) {
            dispatch_operator(data, data, weights, OP_CONV, 256, 256, 3, 64);
        }

        // Attention block (30 times per forward pass)
        for (int head = 0; head < 12; head++) {
            // Q, K, V projection: GEMM (3×)
            dispatch_operator(temp1, data, weights, OP_GEMM, 512, 64, 512, 512);
            dispatch_operator(temp2, data, weights, OP_GEMM, 512, 64, 512, 512);
            dispatch_operator(data, data, weights, OP_GEMM, 512, 512, 512, 512);

            // Attention softmax: SOFTMAX (1×)
            dispatch_operator(temp1, temp1, NULL, OP_SOFTMAX, 512, 512, 1, 1);

            // Output projection: GEMM (1×)
            dispatch_operator(data, temp1, weights, OP_GEMM, 512, 512, 512, 512);
        }

        // FFN layers (2 per block)
        // Mostly GEMM, some NORM, some RELU
        for (int i = 0; i < 2; i++) {
            dispatch_operator(temp1, data, weights, OP_GEMM, 512, 2048, 512, 512);
            dispatch_operator(temp1, temp1, NULL, OP_RELU, 512, 2048, 1, 1);
            dispatch_operator(data, temp1, weights, OP_GEMM, 512, 512, 2048, 512);
            dispatch_operator(data, data, NULL, OP_NORM, 512, 512, 1, 1);
        }
    }

    free(temp1);
    free(temp2);
    free(weights);
}

// Main: run inference forward pass
int main(int argc, char *argv[]) {
    // When compiled WITHOUT PGO:
    // $ gcc -O3 inference_dispatch.c -o inference_no_pgo
    // Performance: ~100 Gops/s (branch misprediction cost)
    //
    // When compiled WITH PGO:
    // Step 1: Compile with profiling
    // $ gcc -O3 -fprofile-generate=profdata inference_dispatch.c -o inference_profile
    //
    // Step 2: Run on representative input (this forward pass)
    // $ ./inference_profile
    // → Generates profdata/*.gcda files
    //
    // Step 3: Recompile with profile data
    // $ gcc -O3 -fprofile-use=profdata inference_dispatch.c -o inference_pgo
    //
    // Performance: ~115 Gops/s (compiler inlines hot GEMM dispatch)
    // Speedup: ~15% over non-PGO
    //
    // Inspection:
    // $ gcc -O3 -fprofile-use=profdata -S inference_dispatch.c
    // $ grep -A 20 "dispatch_operator:" inference_dispatch.s
    // You'll see GEMM code inlined, other branches moved to cold cache locations

    double *data = malloc(sizeof(double) * 512 * 512);
    inference_forward(data, 1);  // One forward pass
    free(data);

    return 0;
}
```

**PGO Workflow**:
```bash
# Step 1: Compile with profiling instrumentation
gcc -O3 -fprofile-generate=profdata inference_dispatch.c -o inference_profile

# Step 2: Run on representative workload (generates .gcda files)
./inference_profile
# This creates profdata/inference_dispatch.gcda with branch frequencies

# Step 3: Recompile with profile data
gcc -O3 -fprofile-use=profdata inference_dispatch.c -o inference_pgo

# Step 4: Verify speedup
time ./inference_no_pgo
time ./inference_pgo
# PGO version should be 10-15% faster (on branch-heavy code)

# Step 5: Inspect assembly to see optimizations
gcc -O3 -fprofile-use=profdata -S -masm=intel inference_dispatch.c
grep -B5 -A20 "dispatch_operator:" inference_dispatch.s
# You'll see hot branch (OP_GEMM) code inlined with less overhead
```

### Example 3: LTO (Link-Time Optimization)

```c
// File: library.c (compiled into .a archive)
// Contains hot functions called from main

double compute_kernel(const double *a, const double *b, int n) {
    double sum = 0.0;
    for (int i = 0; i < n; i++) {
        sum += a[i] * b[i];
    }
    return sum;
}

// File: main.c
// Main function that calls library

#include <stdio.h>

double compute_kernel(const double *a, const double *b, int n);

void process_batch(const double *data, int batch_size, int dim) {
    for (int b = 0; b < batch_size; b++) {
        // Without LTO: compiler can't inline compute_kernel
        // (it doesn't have the source, only the .o file)
        // With LTO: linker has LLVM IR, can inline
        double result = compute_kernel(&data[b*dim], &data[0], dim);
        printf("Result %d: %f\n", b, result);
    }
}

int main() {
    double data[10000];
    process_batch(data, 100, 100);
    return 0;
}

// Compilation without LTO:
// $ gcc -O3 -c library.c -o library.o
// $ gcc -O3 -c main.c -o main.o
// $ gcc -O3 library.o main.o -o no_lto
//
// Result: compute_kernel is NOT inlined (compiler doesn't see source)
// ~5% of time spent in function call overhead
//
// Compilation with LTO:
// $ gcc -O3 -flto -c library.c -o library.o
// $ gcc -O3 -flto -c main.c -o main.o
// $ gcc -O3 -flto library.o main.o -o with_lto
//
// Result: compute_kernel IS inlined during link stage
// No function call overhead, loop fully unrolled in context of caller
// ~3-5% speedup for small leaf functions
//
// LTO cost: ~5× link time (link stage now includes full IR optimization)
```

---

## 5. EXPERT INSIGHT

### Non-Obvious Truth #1: -O3 Is Not Always Better Than -O2

Case study: Dense GEMM kernel

**-O2**: 85 GFLOPs (baseline)
**-O3**: 78 GFLOPs (-8% slower!)

**Why?** -O3 aggressive loop unrolling (64×) causes:
1. Instruction cache misses (8 KB unrolled loop doesn't fit in 32 KB L1I)
2. Register spilling (64× unrolled loop needs more registers)
3. Poor code locality (hot path is fragmented)

**Expert move**: Profile both -O2 and -O3. If -O3 is slower, stick with -O2 and enable specific optimizations (`-finline-limit=200` to control inlining depth).

### Non-Obvious Truth #2: __restrict Is More Powerful Than Pragma

Compiler's aliasing analysis is conservative. Without `__restrict`, compiler assumes:
- `c` and `a` might overlap (can't reorder loads and stores)
- `c` and `b` might overlap

Example:
```c
void add_unsafe(double *c, const double *a, const double *b, int n) {
    // Compiler: can't vectorize (might have aliasing)
    for (int i = 0; i < n; i++) {
        c[i] = a[i] + b[i];
    }
}

void add_safe(double *restrict c, const double *restrict a,
              const double *restrict b, int n) {
    // Compiler: guaranteed no aliasing, vectorizes
    for (int i = 0; i < n; i++) {
        c[i] = a[i] + b[i];
    }
}
```

**Result**: With `__restrict`, 4× speedup (vectorization kicks in).

**Lesson**: Always use `__restrict` on non-overlapping pointers. It's the single most impactful annotation for auto-vectorization.

### Non-Obvious Truth #3: LLVM IR Inspection Beats Compiler Output Messages

Compiler says "loop not vectorized", but IR might show why it tried and failed:

```llvm
; Compiler says: "loop not vectorized"
; But IR shows:
%cmp = icmp slt i64 %i, %n
br i1 %cmp, label %loop.body, label %loop.end
; Loop condition is a scalar comparison (cmp), not vectorized branch
```

If you see scalar operations in the loop body IR, **the loop wasn't vectorized**. If you see `<8 x double>` operations, **it was vectorized**.

### Non-Obvious Truth #4: PGO Helps with Inference More Than Training

**Training**: Gradient flows backward, all branches are equally likely (data is randomized).
**Inference**: Same operator graph runs millions of times, branch pattern is deterministic.

Example: Inference dispatcher on transformer:
- 60% GEMM calls
- 20% softmax calls
- 20% other

After 1 million inferences with PGO, compiler learns this distribution. GEMM branch is inlined, others are moved to cold cache.

**Result**: 10-15% speedup on inference, 1-2% on training.

### Non-Obvious Truth #5: Function Multiversioning Adds Complexity But Preserves Portability

Binary containing both AVX-512 and AVX2 code is slightly larger but runs on any CPU:

```c
__attribute__((target_clones("avx512f", "avx2", "default")))
void gemm(double *C, double *A, double *B, int n) {
    // Compiler generates 3 versions
}
```

This is the **only way to get AVX-512 speedup while remaining portable**. Deploy one binary, let CPU capabilities decide which version runs.

**Expert move**: Always use multiversioning for performance-critical kernels in production. The cost (20-30% binary size) is negligible compared to the benefit (2× speedup on capable hardware).

---

## 6. BENCHMARK / MEASUREMENT

### Procedure: Profile-Guided Optimization for ML Inference

```bash
#!/bin/bash
# File: pgo_inference_pipeline.sh

set -e

MODEL_PATH="./model.onnx"
INFERENCE_BINARY="./inference_engine"

echo "=== PGO Pipeline for Inference Engine ==="

# Step 1: Compile with profiling instrumentation
echo "Step 1: Compile with -fprofile-generate"
gcc -O3 -fprofile-generate=profdata_dir \
    -I./include \
    src/*.c src/inference/*.c \
    -o "$INFERENCE_BINARY"_profile \
    -lm -lpthread

# Step 2: Generate representative workload (e.g., 1000 inferences on validation set)
echo "Step 2: Run on validation dataset (generates profile data)"
time "$INFERENCE_BINARY"_profile \
    --model "$MODEL_PATH" \
    --input-file ./data/validation.bin \
    --num-inferences 1000 \
    --batch-size 8
# This creates profdata_dir/inference*.gcda files with branch frequencies

# Step 3: Merge profile data (if multi-threaded)
echo "Step 3: Merge profile data"
llvm-cov merge-overlapping profdata_dir/*.gcda -o profdata_dir/merged.gcda 2>/dev/null || true

# Step 4: Recompile with profile data
echo "Step 4: Recompile with -fprofile-use"
gcc -O3 -fprofile-use=profdata_dir \
    -I./include \
    src/*.c src/inference/*.c \
    -o "$INFERENCE_BINARY"_pgo \
    -lm -lpthread

# Step 5: Benchmark both versions
echo "Step 5: Benchmark comparison"
echo "Without PGO:"
time "$INFERENCE_BINARY"_profile \
    --model "$MODEL_PATH" \
    --input-file ./data/test.bin \
    --num-inferences 100 \
    --batch-size 16

echo "With PGO:"
time "$INFERENCE_BINARY"_pgo \
    --model "$MODEL_PATH" \
    --input-file ./data/test.bin \
    --num-inferences 100 \
    --batch-size 16

# Step 6: Measure exact improvement
echo "Step 6: Calculate speedup"
perf stat "$INFERENCE_BINARY"_profile --num-inferences 100
perf stat "$INFERENCE_BINARY"_pgo --num-inferences 100
```

### Procedure: Analyzing Vectorization Performance

```bash
#!/bin/bash
# File: analyze_vectorization.sh

KERNEL_SOURCE="kernel.c"

echo "=== Vectorization Analysis ==="

# Step 1: Compile with vectorization reporting
echo "Step 1: Examine compiler vectorization decisions"
gcc -O3 -fopt-info-vec-all "$KERNEL_SOURCE" 2>&1 | tee vectorization.log

# Step 2: Generate assembly
echo "Step 2: Generate assembly"
gcc -O3 -S -masm=intel "$KERNEL_SOURCE" -o kernel.s

# Step 3: Look for SIMD instructions
echo "Step 3: Check for SIMD instructions"
grep -E "vmov|vadd|vfma|vload|vstore" kernel.s || echo "No SIMD instructions found"

# Step 4: Generate LLVM IR
echo "Step 4: Generate LLVM IR for inspection"
clang -O3 -emit-llvm -S "$KERNEL_SOURCE" -o kernel.ll

# Step 5: Check IR for vector operations
echo "Step 5: Look for vector operations in IR"
grep -E "<[0-9]+ x (double|float)>" kernel.ll || echo "No vector types in IR"

# Step 6: Profile the kernel
echo "Step 6: Profile to measure actual performance"
gcc -O3 -pg "$KERNEL_SOURCE" -o kernel_profile
./kernel_profile
gprof kernel_profile gmon.out | head -20

# Step 7: Compare with manual vectorization
echo "Step 7: Compare scalar vs vectorized"
gcc -O0 "$KERNEL_SOURCE" -o kernel_scalar
gcc -O3 "$KERNEL_SOURCE" -o kernel_vector
time ./kernel_scalar
time ./kernel_vector
```

### Validating Compiler Optimizations via Roofline

```python
#!/usr/bin/env python3
# File: validate_compiler_opts.py

import subprocess
import re
import statistics

def measure_performance(binary, iterations=10):
    """Measure kernel performance using perf"""
    cmd = f"perf stat -e cycles,instructions,LLC-loads,LLC-load-misses {binary}"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

    cycles = None
    instructions = None
    for line in result.stderr.split('\n'):
        if 'cycles' in line:
            cycles = int(line.split()[0].replace(',', ''))
        elif 'instructions' in line:
            instructions = int(line.split()[0].replace(',', ''))

    if cycles and instructions:
        # Assume 3 GHz
        gflops = (instructions / 2) / (cycles / 3e9) / 1e9
        ipc = instructions / cycles
        return {'gflops': gflops, 'ipc': ipc, 'cycles': cycles}
    return None

# Compile with different optimization levels
binaries = [
    ("kernel_O0", "-O0"),
    ("kernel_O2", "-O2"),
    ("kernel_O3", "-O3"),
    ("kernel_pgo", "-O3 -fprofile-use=profdata"),
]

print("Optimization Level Comparison (GEMM 1024×1024×1024):")
print("Opt Level | GFLOPs | IPC  | Speedup")
print("----------|--------|------|--------")

baseline_gflops = None
for binary, flags in binaries:
    # Compile
    subprocess.run(f"gcc {flags} kernel.c -o {binary} -lm", shell=True, check=True)

    # Measure
    perf = measure_performance(binary)
    if perf:
        speedup = "baseline" if baseline_gflops is None else f"{perf['gflops']/baseline_gflops:.1f}×"
        print(f"{flags:8} | {perf['gflops']:6.0f} | {perf['ipc']:4.2f} | {speedup}")

        if baseline_gflops is None:
            baseline_gflops = perf['gflops']

print("\nAnalysis:")
print("• -O2: baseline + loop unrolling + basic vectorization")
print("• -O3: aggressive inlining + advanced vectorization + prefetch")
print("• PGO: branch prediction + hot function inlining + cache-aware scheduling")
```

---

## 7. ML SYSTEMS RELEVANCE

### Inference Engine Dispatch Loop

Most inference engines have a hot dispatch loop:

```c
for (int i = 0; i < graph.num_ops; i++) {
    op_t *op = &graph.ops[i];
    switch (op->type) {
        case OP_GEMM: gemm_dispatch(op); break;
        case OP_CONV: conv_dispatch(op); break;
        case OP_SOFTMAX: softmax_dispatch(op); break;
        // ...
    }
}
```

**Without PGO**: Compiler can't predict which branch is taken → branch misses, pipeline stalls.
**With PGO**: Compiler learns dispatch frequency, inlines hot branch, optimizes code layout.

**Result**: 10-20% speedup on inference servers (dispatch is a small but frequent operation).

### Quantization and Compiler Code Generation

INT8 inference requires conversion operators (float → int8, int8 → float). Compiler optimization impacts performance:

```c
// Convert FP32 weights to INT8 (once at startup)
void quantize_weights(int8_t *out, const float *in, int n, float scale) {
    for (int i = 0; i < n; i++) {
        int32_t val = (int32_t)(in[i] / scale);
        out[i] = (int8_t)((val > 127) ? 127 : (val < -128) ? -128 : val);
    }
}

// With -O3: compiler vectorizes with VPSLLD (shift + pack)
// With -O2: scalar code, slower
```

**Expert move**: Enable PGO on quantization kernels (they're called once per weight, but many times across batches). Compiler learns to inline frequently-used conversions.

### Attention Layer Compilation

Flash Attention (Dao et al., 2022) compiles to very different code depending on optimization level:

```c
// Forward pass attention (simplified)
void attention_fwd(float *out, const float *Q, const float *K, const float *V, int N) {
    // With -O3: loops unroll, GEMM inlined, vectorization kicks in
    // With -O2: moderate unrolling, may not inline heavy GEMM
    // With -O0: loops don't unroll, no vectorization

    // Typical speedup -O0 → -O3: 6-10× on attention
}
```

Use `-O3` + function multiversioning for attention kernels. The vectorization speedup (6-8×) outweighs instruction cache cost.

---

## 8. PhD QUALIFIER QUESTIONS

**Q1**: Explain the difference between auto-vectorization at -O2 vs -O3. Why might -O3 be slower on some kernels despite more aggressive optimizations?

**Q2**: Design a PGO profiling strategy for a transformer inference engine. What operations should appear in the profiling workload, and how would you validate that the profiling is representative?

**Q3**: A GEMM kernel achieves 200 GFLOPs with -O2 and 180 GFLOPs with -O3. Use LLVM IR and assembly inspection to diagnose the performance regression. Propose a fix (pragma, flag, or algorithm change).

**Q4**: Explain when to use intrinsics, builtins, and inline assembly. For each category, provide a concrete example and justify the choice.

**Q5**: Implement function multiversioning for a 4×4 matrix multiply kernel supporting AVX-512, AVX2, and scalar code. Measure the binary size overhead and performance speedup on CPUs that support each ISA.

---

## 9. READING LIST

1. **Agner Fog** (2023). *Optimizing Software in C++*. Online publication.
   - See: Chapter 1 (assembly basics), Chapter 2 (optimization levels), Chapter 6 (intrinsics), Chapter 8-9 (vectorization)

2. **Fog, A.** (2023). *Instruction Tables: Lists of Instruction Latencies, Throughputs, and Micro-operation Breakdowns*. Vol. 1-2.
   - See: Appendix A (Sapphire Rapids latencies/throughputs), Appendix B (vectorization)

3. **Dendibakh, D.** (2024). *Performance Analysis and Tuning on Modern CPUs*. O'Reilly Media.
   - See: Chapter 8 (auto-vectorization), Chapter 9 (inline assembly)

4. **GCC Manual**. Free Software Foundation.
   - See: Section 3.13 "Optimize Options", Section 3.14 "Option Summary"
   - https://gcc.gnu.org/onlinedocs/gcc-13.2.0/gcc/

5. **LLVM Language Reference Manual**. https://llvm.org/docs/LangRef/
   - See: Vector types, vectorization passes

6. **Intel Software Developer Manual Vol. 1: Basic Architecture**. Intel Corp.
   - See: Section 3.5 (instruction fetch), Section 12 (floating-point operations)

7. **LLVM Documentation on PGO**. https://llvm.org/docs/PGO/
   - Comprehensive guide to profile-guided optimization workflow

8. **Dao, T. et al.** (2022). "Flash-Attention: Fast and Memory-Efficient Exact Attention with IO-Awareness." NIPS 2022.
   - See: Section 3 (compilation strategy for fused attention kernels)

---

**End of Module 16**

*Compiler mastery is the final frontier of performance optimization. After profiling (Module 15), understanding the compiler's decision-making closes the feedback loop from measurement to code generation.*

*Modules 14-16 form a complete PhD-level systems curriculum on performance engineering: theory (Module 14) → measurement (Module 15) → optimization (Module 16). Mastery requires proficiency in all three domains.*
