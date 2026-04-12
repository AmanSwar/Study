# APPENDIX E — Benchmark Baselines

## Overview

This appendix provides setup, compilation, interpretation, and expected numbers for standard benchmarks (STREAM, LINPACK, Intel MLC, custom RDTSC microbenchmarks) on Intel Sapphire Rapids (SPR) and AMD Zen 4 (EPYC 9004).

---

## 1. STREAM Benchmark

### Purpose
Memory bandwidth benchmark for various access patterns: Copy, Scale, Add, Triad.

### Formulas (per operation)

| Kernel | Formula | Memory Traffic | Bytes/Operation |
|---|---|---|---|
| Copy | C[i] = A[i] | Read A, write C | 16 B (2×8 B) |
| Scale | B[i] = scalar × C[i] | Read C, write B | 16 B |
| Add | C[i] = A[i] + B[i] | Read A, B; write C | 24 B |
| Triad | A[i] = B[i] + scalar × C[i] | Read B, C; write A | 24 B |

### Compilation & Flags (SPR)

```bash
# Download STREAM
wget https://www.cs.virginia.edu/stream/FTP/Code/stream.c

# Compile with Intel compiler (best performance)
icc -O3 -qopt-prefetch -march=native -mtune=native stream.c -o stream

# Or GCC (portable)
gcc -O3 -march=native -ffast-math -fno-alias stream.c -o stream

# Or GCC with AVX-512
gcc -O3 -march=skylake-avx512 -mtune=skylake-avx512 -ffast-math stream.c -o stream

# Clang variant
clang -O3 -march=native -ffast-math stream.c -o stream
```

### Tuning Parameters

```c
/* In stream.c, adjust array size for your system */
#define N 134217728   /* 128 M elements = 1 GB per array (for 1 socket) */
#define NTIMES 100    /* Number of repetitions */

/* For dual-socket NUMA, increase to 2 GB per array */
#define N 268435456   /* 256 M elements */
```

### Running STREAM

```bash
# Single-socket (bind to node 0)
numactl --membind=0 --cpunodebind=0 ./stream

# Dual-socket (interleaved, uses both)
numactl --interleave=all ./stream

# With pinning (OMP_PLACES)
OMP_NUM_THREADS=24 OMP_PLACES=cores OMP_PROC_BIND=close ./stream
```

### Expected Output (SPR, Single-Socket)

```
-------------------------------------------------------------
STREAM version $Revision: 5.10 $
-------------------------------------------------------------
This system uses 8 bytes per DOUBLE PRECISION word.
-------------------------------------------------------------
Array size = 134,217,728 (elements), Offset = 0 (elements)
Total memory required = 3,076.0 MiB (3.0 GiB).
Each test is run 100 times
   The best(?) results are as follows for each test:
Function    Best Rate MB/s  Avg time     Min time     Max time
Copy:          230000.0     0.0059       0.0058       0.0063
Scale:         226000.0     0.0060       0.0059       0.0065
Add:           315000.0     0.0064       0.0063       0.0068
Triad:         314000.0     0.0064       0.0063       0.0069
-------------------------------------------------------------
```

### Interpretation (SPR)

**Peak Single-Socket Bandwidth:**
- Dual-channel DDR5-4800: 60 GB/s
- 12-channel DDR5-4800: 230 GB/s (actual!)
- Expected: 230–250 GB/s (matching STREAM above)

**Multi-socket (Dual DDR5-4800 interfaces per socket):**
- 2 sockets × 230 GB/s = ~460 GB/s peak
- STREAM Triad with `--interleave=all`: 450–480 GB/s

### Expected Output (Zen 4, Single-Socket)

```
Function    Best Rate MB/s  Avg time     Min time     Max time
Copy:          240000.0     0.0056       0.0056       0.0060
Scale:         235000.0     0.0057       0.0057       0.0061
Add:           330000.0     0.0060       0.0060       0.0065
Triad:         328000.0     0.0061       0.0061       0.0066
```

**Notes:** Zen 4 slightly faster (larger L3, more efficient HW prefetcher).

---

## 2. LINPACK (HPL) Benchmark

### Purpose
Measure peak floating-point performance (FLOPS) via LU decomposition.

### Theoretical Peak (SPR)

```
24 cores/socket × 2 sockets × 4.2 GHz × 1 FMA/cycle × 2 ops/FMA = 806.4 GFLOPS/socket
Dual-socket: 1.61 TFLOPS
```

### Actual Achievable Performance (SPR, 1-socket)

| Configuration | Peak GFLOPS | Efficiency | Notes |
|---|---|---|---|
| Scalar FP64 | 100 | 12% | No vectorization |
| AVX-512F FP64 | 600 | 75% | 8 FMA per cycle |
| DGEMM tuned | 750 | 93% | Cache-optimal |
| Theoretical | 806 | 100% | Peak frequency + IPC |

### HPL Setup (Single-Socket)

```bash
# Download HPL
wget http://www.netlib.org/benchmark/hpl/hpl-2.3.tar.gz
tar xzf hpl-2.3.tar.gz
cd hpl-2.3

# Create Make.Linux_PII_CBLAS for SPR (edit Make.UNKNOWN)
cp Make.UNKNOWN Make.Linux_SPR

# Edit Make.Linux_SPR
cat > Make.Linux_SPR << 'EOF'
ARCH = Linux_SPR
TOPdir = $(HOME)/hpl-2.3
INCdir = $(TOPdir)/include
BINdir = $(TOPdir)/bin/$(ARCH)
LIBdir = $(TOPdir)/lib/$(ARCH)

# Use Intel MKL for BLAS
LAdir = /opt/intel/mkl
LAinc = -I$(LAdir)/include
LAlib = -L$(LAdir)/lib/intel64 -lmkl_intel_lp64 -lmkl_core -lmkl_sequential -lpthread -lm

# Compiler
CC = gcc
CCFLAGS = -O3 -march=skylake-avx512 -mtune=skylake-avx512 -ffast-math
EOF

# Build
make arch=Linux_SPR
```

### HPL.dat Configuration (SPR, 1-socket, 256 GB memory)

```
# 256 GB available, use 192 GB for test
N = 524288      # Matrix size (NxN float64 = 2 GB)
NB = 384        # Block size (divisor of N)
PMAP = 0        # Process mapping
P = 1           # 1 row of processes
Q = 1           # 1 column
PFACT = 0       # Panel factorization (BLAS)
NBMIN = 4       # Minimum panel size
NDIV = 2        # Panel division
RFACT = 0       # Recursive factor
BCAST = 0       # Broadcast method
DEPTH = 0       # Lookahead depth
SWAP = 1        # Swap method
L1 = 1          # L1 transposition
U = 1           # Transposition of U
Timing Sincronization = 6
```

### Running HPL (Dual-Socket)

```bash
# Single-process (use 1 socket)
numactl --cpunodebind=0 --membind=0 ./hpl

# Output excerpt:
# ============================================================================
# Linpack 2.0 -- High-Performance Linpack benchmark
# ============================================================================
# N = 524288
# ...
# Performance = 780.5 Gflops
# Efficiency = 96.8%
# Time = 22.45 seconds
```

### Interpretation

| Benchmark | SPR (1-Socket) | Zen 4 (1-Socket) | Ratio |
|---|---|---|---|
| Peak Theoretical | 806 GFLOPS | 336 GFLOPS* | 2.4× |
| LINPACK Achievable | 750 GFLOPS | 320 GFLOPS | 2.3× |
| Efficiency | 93% | 95% | — |

*Zen 4: 12 cores × 3.5 GHz × 2 FMA/cycle × 2 ops/FMA = 168 GFLOPS per core, but scaling limited to ~320 due to memory.

---

## 3. Intel Memory Latency Checker (MLC)

### Installation

```bash
# Download from Intel (registration required)
# https://www.intel.com/content/www/en/en/download/736659.html

cd /opt/mlc
```

### Key Commands

#### Latency Matrix

```bash
./mlc --latency -T
```

**Output (SPR, dual-socket):**

```
Intel(R) Memory Latency Checker - v3.11
Command line: mlc --latency

Measuring idle latencies (in nanoseconds) on the system.
Inject Delay = 0 usec (default)

      Numa Node
Numa Node  0      1
    0    63.6   168.2
    1   171.8    63.5
```

**Interpretation:**

| Latency | Value | Note |
|---|---|---|
| Local (node 0 → 0) | 63.6 ns | L3 miss → main memory |
| Remote (0 → 1) | 168.2 ns | Cross-socket + interconnect |
| Remote/Local Ratio | 2.6× | NUMA penalty |

#### Memory Bandwidth (Peak Injection)

```bash
./mlc --peak_injection_bandwidth -T
```

**Output:**

```
Measuring peak bandwidth (in MB/sec) on the system.

      Numa Node
Numa Node  0        1
    0    231000    115000
    1    115000    231000
```

**Interpretation:**

| Scenario | Bandwidth | Note |
|---|---|---|
| Local write (0 → 0) | 231 GB/s | Full 12-channel bandwidth |
| Remote write (0 → 1) | 115 GB/s | Shared interconnect |
| Total dual-socket | ~360 GB/s | Aggregated across sockets |

#### Loaded Latency (with cache contention)

```bash
./mlc --loaded_latency -d0 -T
```

**Output:**

```
Measuring latencies with background load injected.

      Numa Node
Numa Node  0      1
    0    128.4   212.5
    1   215.3   129.1
```

**Differences:**
- Latencies increase due to memory bus contention
- Remote penalty increases (168 → 212 ns)

### Zen 4 Expected Output

```
# Zen 4 (EPYC 9004) Latency
      Numa Node
Numa Node  0      1
    0    50.2   145.7
    1   147.5    49.8

# Zen 4 (EPYC 9004) Bandwidth
      Numa Node
Numa Node  0        1
    0    260000    120000
    1    120000    260000
```

**Differences from SPR:**
- **Latency:** ~13 ns faster local (better memory controller)
- **Bandwidth:** ~13% higher local (16 channels on some Genoa SKUs)
- **Remote latency:** Slightly better due to superior interconnect (16 vs 20 ratio)

---

## 4. Custom RDTSC Microbenchmark: Instruction Latency & Throughput

### Template (Compatible GCC/Clang)

**File: `microbench.c`**

```c
/*
 * microbench.c
 * Measure instruction latency and throughput using RDTSC.
 */

#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <omp.h>

/* Read timestamp counter */
static inline uint64_t rdtsc(void)
{
    uint32_t low, high;
    asm volatile("rdtsc" : "=a" (low), "=d" (high));
    return ((uint64_t)high << 32) | low;
}

/* Serialize execution (prevent out-of-order) */
static inline void lfence(void)
{
    asm volatile("lfence" ::: "memory");
}

/* Measure latency of dependent chain */
uint64_t measure_latency_fma(size_t iterations)
{
    float x = 1.5f;
    float a = 2.0f;
    float b = 3.0f;
    uint64_t start, end;

    lfence();
    start = rdtsc();

    for (size_t i = 0; i < iterations; i++) {
        x = x * a + b;  /* FMA: latency-bound */
    }

    lfence();
    end = rdtsc();

    /* Prevent compiler optimization */
    asm volatile("" : "+x" (x));

    return (end - start) / iterations;
}

/* Measure throughput (independent operations) */
uint64_t measure_throughput_fma(size_t iterations)
{
    float x1 = 1.5f, x2 = 1.6f, x3 = 1.7f, x4 = 1.8f;
    float a = 2.0f, b = 3.0f;
    uint64_t start, end;

    lfence();
    start = rdtsc();

    for (size_t i = 0; i < iterations; i++) {
        x1 = x1 * a + b;
        x2 = x2 * a + b;
        x3 = x3 * a + b;
        x4 = x4 * a + b;
    }

    lfence();
    end = rdtsc();

    asm volatile("" : "+x" (x1), "+x" (x2), "+x" (x3), "+x" (x4));

    /* 4 independent ops per iteration */
    return (end - start) / (iterations * 4);
}

/* Measure L1D latency */
uint64_t measure_latency_l1d(size_t iterations)
{
    volatile uint64_t *data = malloc(8 * 1024 * 1024);  /* 64 MB */
    memset((void *)data, 0, 8 * 1024 * 1024);

    uint64_t sum = 0;
    uint64_t start, end;

    lfence();
    start = rdtsc();

    for (size_t i = 0; i < iterations; i++) {
        sum += data[i & 0xFFF];  /* Stride within L1: 8 KB */
    }

    lfence();
    end = rdtsc();

    free((void *)data);
    asm volatile("" : "+r" (sum));

    return (end - start) / iterations;
}

/* Measure memory bandwidth (128-bit loads, unrolled) */
uint64_t measure_bw_memory(size_t mb)
{
    float *data = malloc(mb * 1024 * 1024);
    size_t count = (mb * 1024 * 1024) / sizeof(float);
    float sum = 0.0f;
    uint64_t start, end;

    lfence();
    start = rdtsc();

    #pragma omp parallel for reduction(+:sum)
    for (size_t i = 0; i < count; i += 4) {
        sum += data[i] + data[i+1] + data[i+2] + data[i+3];
    }

    lfence();
    end = rdtsc();

    free(data);
    asm volatile("" : "+x" (sum));

    uint64_t cycles = end - start;
    uint64_t bytes_read = mb * 1024 * 1024;

    return bytes_read / cycles;  /* Bytes per cycle */
}

int main(void)
{
    printf("=== RDTSC Microbenchmark Results ===\n\n");

    /* CPU frequency (approximate) */
    uint64_t freq_mhz = 4200;  /* SPR turbo */
    printf("Assumed CPU frequency: %llu MHz\n\n", freq_mhz);

    /* FMA Latency */
    uint64_t lat = measure_latency_fma(1000000);
    printf("FMA Latency:          %llu cycles (%.1f ns)\n", lat, lat * 1000.0 / freq_mhz);

    /* FMA Throughput */
    uint64_t tp = measure_throughput_fma(1000000);
    printf("FMA Throughput:       %.2f cycles per op (%.1f ops/cycle)\n",
           (float)tp, 1.0f / tp);

    /* L1D Latency */
    uint64_t lat_l1d = measure_latency_l1d(1000000);
    printf("L1D Latency (8 KB):   %llu cycles (%.1f ns)\n", lat_l1d, lat_l1d * 1000.0 / freq_mhz);

    /* Memory Bandwidth */
    uint64_t bw = measure_bw_memory(256);  /* 256 MB test */
    printf("Memory Bandwidth:     %.1f bytes/cycle (%llu GB/s at %llu MHz)\n",
           (float)bw, bw * freq_mhz / 1000, freq_mhz);

    return 0;
}
```

### Compilation

```bash
# Compile with optimization and AVX-512
gcc -O3 -march=skylake-avx512 -mtune=skylake-avx512 -fopenmp -fno-alias microbench.c -o microbench

# Run
numactl --membind=0 --cpunodebind=0 ./microbench
```

### Expected Output (SPR, Single-Socket)

```
=== RDTSC Microbenchmark Results ===

Assumed CPU frequency: 4200 MHz

FMA Latency:          4 cycles (0.9 ns)
FMA Throughput:       0.25 cycles per op (4.0 ops/cycle)
L1D Latency (8 KB):   4 cycles (1.0 ns)
Memory Bandwidth:     57 bytes/cycle (238 GB/s at 4200 MHz)
```

### Expected Output (Zen 4, Single-Socket)

```
=== RDTSC Microbenchmark Results ===

Assumed CPU frequency: 3500 MHz

FMA Latency:          3 cycles (0.9 ns)
FMA Throughput:       0.25 cycles per op (4.0 ops/cycle)
L1D Latency (8 KB):   4 cycles (1.1 ns)
Memory Bandwidth:     74 bytes/cycle (259 GB/s at 3500 MHz)
```

---

## 5. Expected Performance Numbers Reference Table

### Single-Socket FP64 Performance

| Benchmark | SPR | Zen 4 | Units |
|---|---|---|---|
| Theoretical Peak | 806 | 168 | GFLOPS |
| LINPACK | 750 | 160 | GFLOPS |
| STREAM Triad | 314 | 328 | GB/s |
| FMA Throughput | 4.0 | 4.0 | ops/cycle |
| FMA Latency | 4 | 3 | cycles |

### Memory Subsystem

| Metric | SPR | Zen 4 | Units |
|---|---|---|---|
| Peak Memory Bandwidth | 230 | 260 | GB/s |
| L1D Latency (hit) | 4 | 4 | cycles |
| L2 Latency (hit) | 12 | 12 | cycles |
| L3 Latency (hit) | 45 | 35 | cycles |
| Main Memory Latency (local) | 63 | 50 | ns |
| Main Memory Latency (remote) | 168 | 146 | ns |

### Dual-Socket Aggregated

| Metric | SPR | Zen 4 | Units |
|---|---|---|---|
| Total FP64 Peak | 1.61 | 0.34 | TFLOPS |
| Total Memory Bandwidth | 460 | 520 | GB/s |
| Intra-socket | 0.92 | 0.49 | ns latency |
| Inter-socket | 2.6× | 2.9× | latency ratio |

---

## 6. Benchmark Interpretation Guide

### When STREAM << Peak Bandwidth

**Causes:**
1. Insufficient parallelism (low thread count)
2. NUMA locality issues (remote access penalty)
3. TLB misses (missing huge pages)
4. Small working set (fits in L3, no memory access)

**Solutions:**
```bash
# Verify NUMA locality
numactl --hardware
numactl --show

# Enable huge pages and rebind
echo 128 > /proc/sys/vm/nr_hugepages
./stream --use_hugepages

# Pin to single socket
numactl --cpunodebind=0 --membind=0 ./stream
```

### When LINPACK << Peak FLOPS

**Causes:**
1. Memory bandwidth limit (AI < 0.87 FLOP/byte)
2. Suboptimal blocking (poor L3 reuse)
3. False sharing (multiple threads touching same cache line)
4. TLB pressure (large matrices without proper allocation)

**Solutions:**
```c
/* Increase block size for better cache reuse */
#define NB 512  /* vs default 256 */

/* Use MKL DGEMM with tuned parameters */
mkl_set_num_threads_local(48);
cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
            N, N, N, 1.0, A, K, B, N, 0.0, C, N);
```

### When MLC Remote Latency > 2.5× Local

**Indicates:** Inefficient NUMA binding or interconnect contention.

**Debug:**
```bash
perf stat -e cpu_clock,cycles,instructions,cache-misses,dTLB-loads,dTLB-load-misses ./myapp
# Look for high dTLB-load-misses (NUMA page table walk)
```

---

## 7. Roofline Model Application

### Roofline Equation

```
Performance = min(Peak_FLOPS, Peak_BW × AI)
```

Where **AI (Arithmetic Intensity)** = FLOPS / Bytes_Transferred.

### SPR Roofline Example

```
Peak FLOPS (1-socket):  806 GFLOPS
Peak BW (1-socket):     230 GB/s
Breakeven AI:           806 / 230 = 3.5 FLOP/byte
```

**Categorize workload:**

| AI | Bound | Example |
|---|---|---|
| < 3.5 | Memory | Matrix copy, streaming |
| 3.5–7 | Transition | DGEMM with K=128 |
| > 7 | Compute | DGEMM with K=4096 |

### Example: Matrix Multiply Arithmetic Intensity

```
DGEMM: C = A × B   (N×K times K×N = N×N output)

FLOPs:   2 × N × K × N = 2N²K
Bytes:   (N×K) read A + (K×N) read B + (N×N) write C
       = 2NK + N² bytes (ignoring cache effects)

AI = 2N²K / (2NK + N²) ≈ 2N²K / 2NK ≈ N  (for large N, K)

For N=4096, K=4096: AI ≈ 4096 → Fully compute-bound
For N=256, K=128:   AI ≈ 256 → Memory-bound on SPR
```

---

## 8. Benchmark Automation Script

**File: `run_benchmarks.sh`**

```bash
#!/bin/bash

# Comprehensive benchmark runner

OUTDIR="/tmp/benchmark_results_$(date +%s)"
mkdir -p $OUTDIR

echo "=== Running Benchmarks ===" | tee $OUTDIR/summary.txt

# STREAM
echo "STREAM benchmark..." | tee -a $OUTDIR/summary.txt
numactl --interleave=all ./stream 2>&1 | tee $OUTDIR/stream.txt | grep "Best Rate"

# HPL
echo "HPL (LINPACK) benchmark..." | tee -a $OUTDIR/summary.txt
timeout 120 numactl --cpunodebind=0 ./hpl 2>&1 | tee $OUTDIR/hpl.txt | grep "Performance"

# Intel MLC
echo "Intel MLC latency..." | tee -a $OUTDIR/summary.txt
/opt/mlc/mlc --latency -T 2>&1 | tee $OUTDIR/mlc_latency.txt

echo "Intel MLC bandwidth..." | tee -a $OUTDIR/summary.txt
/opt/mlc/mlc --peak_injection_bandwidth -T 2>&1 | tee $OUTDIR/mlc_bandwidth.txt

# Microbenchmark
echo "RDTSC Microbenchmark..." | tee -a $OUTDIR/summary.txt
./microbench 2>&1 | tee $OUTDIR/microbench.txt

echo ""
echo "Results saved to: $OUTDIR"
```

---

## 9. Performance Regression Detection

```bash
#!/bin/bash
# Check if current performance within tolerance of baseline

BASELINE_FLOPS=750000  # LINPACK baseline
TOLERANCE=0.95         # 95% of baseline

# Run LINPACK
CURRENT=$(hpl 2>&1 | grep Performance | awk '{print $NF}')

if (( $(echo "$CURRENT >= $BASELINE_FLOPS * $TOLERANCE" | bc -l) )); then
    echo "✓ Performance acceptable: $CURRENT GFLOPS"
    exit 0
else
    echo "✗ Performance degraded: $CURRENT GFLOPS (expected >= $(echo "$BASELINE_FLOPS * $TOLERANCE" | bc) GFLOPS)"
    exit 1
fi
```

---

## 10. Comparison: SPR vs Zen 4 Head-to-Head

| Metric | SPR | Zen 4 | Winner |
|---|---|---|---|
| **Peak FP64 FLOPS** | 806 GFLOPS | 168 GFLOPS | **SPR (4.8×)** |
| **Memory BW** | 230 GB/s | 260 GB/s | **Zen 4 (1.1×)** |
| **AI Breakeven** | 3.5 | 6.5 | SPR (lower = more memory-bound) |
| **L3 Cache** | 32 MB | 96 MB | **Zen 4 (3×)** |
| **Local Mem Latency** | 63.6 ns | 50.2 ns | **Zen 4 (1.3×)** |
| **Remote Mem Latency** | 168.2 ns | 145.7 ns | **Zen 4 (1.2×)** |
| **Multi-socket Agg** | 1.61 TFLOPS | 0.34 TFLOPS | **SPR (4.7×)** |

### Use Case Recommendation

| Workload | Recommendation |
|---|---|
| **Dense linear algebra (GEMM, LAPACK)** | **SPR** — 5× higher peak FLOPS |
| **Memory-heavy streaming** | **Zen 4** — 13% higher BW, better latency |
| **Graph algorithms** | **Zen 4** — 3× L3 cache advantage |
| **Single-precision ML inference** | **SPR** — Higher power efficiency per FLOP |
| **NUMA-sensitive workloads** | **Zen 4** — Better latency & interconnect |

