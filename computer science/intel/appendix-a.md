# APPENDIX A — Instruction Latency & Throughput Reference Tables

## Overview

This appendix provides cycle counts for critical instruction classes on Intel Sapphire Rapids (SPR, 4th Gen Xeon Scalable) and AMD Zen 4 (EPYC 7004 series). These numbers drive roofline modeling and performance bottleneck analysis. All latencies/throughputs measured under ideal conditions with out-of-order execution enabled.

**Key Sources:**
- Agner Fog's Instruction Tables (fogleman.com/cpu_schedule.html)
- Intel 64 and IA-32 Architectures Optimization Reference Manual (March 2023)
- AMD EPYC Processors Optimization Guide (October 2022)
- Internal benchmark validation on actual hardware

---

## 1. Arithmetic Instructions — FP32

### Intel Sapphire Rapids (10nm, ICX+)

| Instruction | Latency | Throughput (instructions/cycle) | Port | Notes |
|---|---|---|---|---|
| VMULPS (scalar) | 4 | 1 | Port 0,1 | 128-bit operands |
| VMULPS (256-bit) | 4 | 1 | Port 0,1 | AVX |
| VMULPS (512-bit) | 4 | 1 | Port 0,1 | AVX-512F |
| VADDPS (scalar) | 3 | 1 | Port 0,1 | 128-bit operands |
| VADDPS (256-bit) | 3 | 1 | Port 0,1 | AVX |
| VADDPS (512-bit) | 3 | 1 | Port 0,1 | AVX-512F |
| VDIVPS (scalar) | 11 | 0.5 | Port 0,1 | ~22 cycles for 512-bit |
| VDIVPS (256-bit) | 11 | 0.5 | Port 0,1 | Division unit shared |
| VDIVPS (512-bit) | 22 | 0.25 | Port 0,1 | Serialized on SPR |
| VSQRTPS (scalar) | 11 | 0.5 | Port 0,1 | 128-bit |
| VSQRTPS (256-bit) | 11 | 0.5 | Port 0,1 | AVX |
| VSQRTPS (512-bit) | 21 | 0.25 | Port 0,1 | AVX-512F, serialized |
| VFMADD132PS | 4 | 1 | Port 0,1 | Fused multiply-add |
| VFMADD213PS | 4 | 1 | Port 0,1 | 3-operand form |
| VFMADD231PS | 4 | 1 | Port 0,1 | Accumulate form |

### AMD Zen 4 (5nm, EPYC 7004)

| Instruction | Latency | Throughput | Port | Notes |
|---|---|---|---|---|
| VMULPS (scalar) | 3 | 1 | FP-multiply unit | Lower latency vs SPR |
| VMULPS (256-bit) | 3 | 1 | FP-multiply unit | AVX native |
| VADDPS (scalar) | 3 | 1 | FP-add unit | Parallel multiplier |
| VADDPS (256-bit) | 3 | 1 | FP-add unit | Independent pipes |
| VDIVPS (scalar) | 10-13 | 0.5 | FP-multiply unit | Latency depends on exponent |
| VDIVPS (256-bit) | 10-13 | 0.5 | FP-multiply unit | Serialized |
| VSQRTPS (scalar) | 10-15 | 0.25 | FP-multiply unit | Latency variable |
| VFMADD132PS | 4 | 2 | FP-multiply + FP-add | **Two FMAs per cycle** |
| VFMADD213PS | 4 | 2 | Parallel execution | Zen 4 advantage |
| VFMADD231PS | 4 | 2 | Both units in flight | Unpipelined |

**Insight:** Zen 4 FMA throughput is 2x SPR due to separate add/multiply pipelines. SPR serializes on Port 0/1.

---

## 2. Arithmetic Instructions — FP16 (AVX-512 BF16, FP16 extensions)

### Intel Sapphire Rapids (with AVX-512 BF16)

| Instruction | Latency | Throughput | Notes |
|---|---|---|---|
| VMULPH (512-bit, FP16) | 5 | 2 | Requires AVX-512FP16 extension |
| VADDPH (512-bit, FP16) | 4 | 2 | Similar to V256 but 2x width |
| VFMADD132PH | 5 | 2 | Native FP16 FMA |
| VCVTPH2PS (512-bit → 4×256) | 5 | 2 | Upsample and convert |
| VCVTPS2PH (4×256 → 512) | 5 | 2 | Downsample and convert |
| VBFMADDN2PS (BF16) | 4 | 2 | Simplified FMA for BF16 pairs |

### AMD Zen 4 (no native BF16/FP16 FMA)

| Instruction | Latency | Throughput | Notes |
|---|---|---|---|
| Convert FP32 → BF16 manually | 2 shifts + 1 store | Variable | Must emulate with FP32 ops |
| FMA in FP32 → BF16 | 4 + 2 (convert) | 1 | Inefficient, use FP32 native |

**Insight:** SPR's BF16 advantage is significant for inference; Zen 4 lacks dedicated units.

---

## 3. Arithmetic Instructions — INT8 (AVX-512VNNI on SPR, AVX-512IFMA on both)

### Intel Sapphire Rapids (AVX-512 VNNI)

| Instruction | Latency | Throughput | Notes |
|---|---|---|---|
| VPDPBUSD (INT8 dot product) | 5 | 1 | 4×INT8 → INT32 |
| VPDPWSSD (INT16 dot product) | 5 | 1 | 2×INT16 → INT32 |
| VPMADDUBSW (mixed sign) | 3 | 1 | Multiply-add INT8 pairs |
| VPMADD52LUQ (IFMA, 52-bit) | 5 | 2 | Integer FMA, lower precision |
| VPMADD52HUQ (High 52 bits) | 5 | 2 | Pair with LOW for 104-bit accumulation |

### AMD Zen 4 (AVX-512 VNNI from EPYC 9004)

| Instruction | Latency | Throughput | Notes |
|---|---|---|---|
| VPDPBUSD (INT8 dot product) | 4 | 1 | Faster latency than SPR |
| VPDPWSSD (INT16 dot product) | 4 | 1 | Lower latency |
| VPMADD52LUQ (IFMA) | 4 | 1 | Single issue per cycle |

**Insight:** Zen 4 has lower latency on INT8 ops; both platforms saturate at 1 issue/cycle.

---

## 4. Load & Store Instructions

### Intel Sapphire Rapids

| Instruction | Latency | Throughput | Notes |
|---|---|---|---|
| MOV reg, [mem] (L1$) | 4 | 2 loads/cycle | Pipelined load units |
| MOV [mem], reg | 0 (AGU) | 1 store/cycle | Store buffer latency ~400 cycles |
| VMOVUPS [mem], ymm | 0 (AGU) | 0.5 stores/cycle | 256-bit write |
| VMOVUPS [mem], zmm | 0 (AGU) | 0.25 stores/cycle | 512-bit write, serialized |
| VGATHERDPS (gather) | 12 | 0.5 | Scalar operations hidden |
| VSCATTERDPS (scatter) | 12 | 0.25 | Very serialized |

### AMD Zen 4

| Instruction | Latency | Throughput | Notes |
|---|---|---|---|
| MOV reg, [mem] (L1$) | 4 | 2 loads/cycle | Two parallel load units |
| MOV [mem], reg | 0 (AGU) | 2 stores/cycle | **2x better throughput** |
| VMOVUPS [mem], ymm | 0 (AGU) | 1 store/cycle | Better write bandwidth |
| VGATHERDPS (gather) | 12 | 0.5 | Similar to SPR |

**Insight:** Zen 4 has superior store bandwidth (2 vs 1 store/cycle). Important for write-heavy kernels.

---

## 5. Shuffle & Permutation Instructions

### Intel Sapphire Rapids

| Instruction | Latency | Throughput | Notes |
|---|---|---|---|
| VPSHUFB (in-lane shuffle) | 1 | 1 | 128-bit control, fast |
| VPERMD (32-bit lane shuffle) | 3 | 1 | 8 lanes (256-bit) |
| VPERMD (512-bit) | 4 | 1 | 16 lanes, slightly slower |
| VPERMQ (64-bit lane shuffle) | 3 | 1 | 4 lanes (256-bit) |
| VPERMQ (512-bit) | 4 | 1 | 8 lanes |
| VSHUFPS | 1 | 1 | In-lane shuffle |
| VSHUFPD | 1 | 1 | In-lane shuffle |
| VPUNPCKLDQ (unpack low) | 1 | 1 | Basic shuffle |
| VPERM2F128 (256-bit) | 3 | 1 | Cross-lane permute |
| VPERMI2D (two-source shuffle) | 3 | 1 | Full 512-bit routing |

### AMD Zen 4

| Instruction | Latency | Throughput | Notes |
|---|---|---|---|
| VPSHUFB (in-lane shuffle) | 1 | 2 | **Pipelined** |
| VPERMD (32-bit) | 3 | 1 | Same as SPR |
| VPERMQ (64-bit) | 3 | 1 | Same |
| VPERMI2D (two-source) | 3 | 1 | Similar |

**Insight:** Zen 4 pipelines VPSHUFB (2/cycle vs 1/cycle), useful for data-shuffling kernels.

---

## 6. Comparison & Mask Instructions

### Intel Sapphire Rapids

| Instruction | Latency | Throughput | Notes |
|---|---|---|---|
| VCMPPD (compare, FP64) | 3 | 1 | Sets mask register (k0–k7) |
| VCMPPS (compare, FP32) | 3 | 1 | Sets mask register |
| VPCMPD (signed INT32) | 1 | 1 | Integer comparison faster |
| VPCMPUD (unsigned INT32) | 1 | 1 | Zero-latency after flag set |
| VPCOMPRESSB (compress via mask) | 7 | 0.5 | Variable-latency store |
| VPEXPANDD (expand via mask) | 7 | 0.5 | Variable-latency load |
| VCOMPRESS (to contiguous) | 7 | 0.5 | Store non-matching to separate location |
| KORW (mask OR) | 1 | 1 | Mask-on-mask ops |

### AMD Zen 4

| Instruction | Latency | Throughput | Notes |
|---|---|---|---|
| VCMPPD (FP64 compare) | 3 | 1 | Same latency |
| VPCMPD (INT32 compare) | 1 | 1 | Same |
| VPCOMPRESSB (compress) | 10 | 0.33 | Slower than SPR |
| KORW (mask OR) | 1 | 1 | Same |

**Insight:** SPR slightly faster on compress/expand operations.

---

## 7. AMX (Advanced Matrix Extensions) — Intel Sapphire Rapids Only

| Instruction | Latency | Throughput | Tile Size | Notes |
|---|---|---|---|---|
| TILELOADD | 6 | 1 | 16×16 INT32 or smaller | Load via memory |
| TILELOADDT1 | 6 | 1 | 16×16 INT32 | Transpose variant |
| TILESTORED | 0 (write) | 1 | 16×16 INT32 | Store via memory |
| TDPBF16PS (BF16 FMA) | 9 | 1 | 16×16 (accumulator) | 256 BF16 MACs per instr |
| TDPBUSD (INT8 FMA) | 9 | 1 | 16×16 (accumulator) | 1024 INT8 MACs per instr |
| TDPBUUS (unsigned) | 9 | 1 | 16×16 (accumulator) | 1024 MACs |
| TDPBSUS (mixed) | 9 | 1 | 16×16 (accumulator) | 1024 MACs |
| TDPBSSD (INT16 FMA) | 9 | 1 | 16×16 (accumulator) | 256 INT16 MACs |

**Theoretical Throughput:**
- TDPBF16PS: 16×16 tiles, 256 BF16×BF16→FP32 MACs, 9 cycles = **28.4 GFLOPS per tile** on 1-core (with 48 cores @ 2.1 GHz = 1.36 TFLOPS peak).
- TDPBUSD: 1024 INT8 MACs per tile, 9 cycles = **113.8 GOPS per tile**.

**AMD Zen 4:** No AMX equivalent; uses scalar or AVX-512 only.

---

## 8. Comparison Table: SPR vs Zen 4 Advantages

| Operation Class | SPR Advantage | Zen 4 Advantage | Impact |
|---|---|---|---|
| **FP32 FMA** | — | 2× throughput (2 per cycle) | Major for ML inference |
| **FP16 (BF16)** | Native BF16 FMA | Requires emulation | Critical for inference |
| **INT8 Dot Product** | — | Lower latency (4 vs 5 cycles) | Marginal |
| **Store Bandwidth** | — | 2× (2 stores/cycle) | Critical for ETL, writes |
| **Permutation (VPSHUFB)** | — | 2× pipelined | Useful for data shuffling |
| **Gather/Scatter** | Similar | Similar | Both slow; avoid if possible |
| **AMX (matrix)** | Dedicated unit | None | Transformative for linear algebra |
| **L1D Bandwidth** | 2 loads/cycle | 2 loads/cycle | Parity |

---

## 9. Port Binding on SPR (µ-architecture view)

Intel Sapphire Rapids has **12 execution units** distributed across 4 ports:

- **Port 0:** FP multiply, INT multiply, some shuffles
- **Port 1:** FP add, INT add, some shuffles
- **Port 2:** Load AGU
- **Port 3:** Load AGU
- **Port 4:** Store AGU
- **Port 5:** INT ALU, logic, shifts
- **Port 6:** INT ALU, branches
- **Port 7:** Store data (write buffer)

**Impact:** Unrelated operations (e.g., INT multiply on Port 0 + FP multiply on Port 0) compete for same port. Careful pairing required in tight loops.

---

## 10. Example: Roofline Calculation

### FP32 FMA on Sapphire Rapids (dual-socket)

**Peak FLOPs (1-socket):**
- 24 cores × 4.2 GHz (turbo) × 1 FMA/cycle × 2 FP32 ops/FMA = **403.2 GFLOPS**
- Dual-socket: **806.4 GFLOPS**

**Peak Memory Bandwidth (1-socket):**
- 12-channel DDR5-4800, 64-byte bus per channel
- 4800 MT/s × 64 bytes × 12 channels / 8 (bits per byte) = **460.8 GB/s per socket**
- Dual-socket: **921.6 GB/s** (but NUMA-aware code needed)

**Arithmetic Intensity Threshold:**
- Break-even: 806.4 GFLOPS / 921.6 GB/s = **0.87 FLOP/byte**
- Kernels below 0.87 FLOP/byte are memory-bound.

### FP32 FMA on Zen 4 (dual-socket)

**Peak FLOPs (1-socket):**
- 12 cores × 3.5 GHz (boost) × 2 FMA/cycle × 2 FP32 ops/FMA = **168 GFLOPS** (conservative)
- Dual-socket: **336 GFLOPS**

**Note:** Zen 4 achieves higher FMA throughput (2/cycle) but lower clock + fewer cores = lower absolute FLOPS. Advantage is in specific workloads (e.g., parallel FMA chains).

---

## 11. Instruction Encoding Sizes (useful for code cache)

| Instruction | Encoding | Notes |
|---|---|---|
| VMULPS r/m, reg | 6–8 bytes | EVEX prefix (4 bytes) + opcode (2) + modrm (1) + displacement (1–4) |
| VMULPS 512-bit | 6–8 bytes | EVEX-encoded with Z masking possible |
| TDPBF16PS tmm, tmm, tmm | 4 bytes | Compact tile opcode |
| VGATHERDPS (zmm, mem, zmm) | 8–12 bytes | VSIB (vector scaled index byte) + EVEX |

**Impact:** Tight FMA loops can fit in L1-I (32 KB) for 1000+ iterations; gather loops are larger, may evict code cache.

---

## 12. Latency Chain Example: Matrix Multiply

For `C[i][j] += A[i][k] * B[k][j]`, the critical path (assuming sequential loads):

**SPR:**
```
Load A[i][k]      (4 cycles latency)
  └─ Multiply    (4 cycles, starts at cycle 4, finishes at 8)
    └─ FMA       (4 cycles, starts at 8, finishes at 12)
      └─ Store C (finishes at 12 + 0 AGU = 12 cycles minimum)
```
**Minimum 12 cycles between independent accumulation steps.**

**Zen 4:**
```
Load A[i][k]      (4 cycles)
  └─ Multiply    (3 cycles, finishes at 7)
    └─ FMA       (4 cycles, finishes at 11)
      └─ Store C (11 cycles minimum)
```
**Zen 4 slightly faster on dependent chains due to lower multiply latency.**

---

## References & Tools

1. **Agner Fog's Instruction Tables:** Download from `agner.org/optimize/`; contains µ-op counts, µ-code, and latency for all Intel/AMD instructions.
2. **Intel Optimization Manual:** `intel.com/content/dam/www/public/us/en/documents/manuals/64-ia-32-architectures-optimization-reference-manual.pdf`
3. **AMD EPYC Optimization Guide:** `amd.com/en/technologies/epyc-7004-genoa`
4. **uops.info:** Interactive tool for latency/throughput lookup by instruction.
5. **Microbenchmark validation:** Use `likwid`, `perf`, or custom RDTSC-based code (Appendix E) to verify on actual hardware.

