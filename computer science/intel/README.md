# Computer Systems & Architecture for ML Systems Engineers
## A PhD-Level Study Curriculum

**Target Reader:** Experienced ML Systems Engineer with strong CUDA background seeking mastery of CPU architecture, performance optimization, and inference engine design on Intel Xeon and AMD EPYC hardware.

**Pedagogical Standard:** Graduate-level rigor modeled after MIT 6.004/6.823, Stanford CS149, UC Berkeley CS152/252, CMU 15-213/15-418, and Onur Mutlu's ETH Zürich lectures.

---

## Curriculum Structure

### Part I — Computer Systems Foundations (The "CS:APP + OS" Layer)

| Module | Title | File |
|--------|-------|------|
| 1 | The Execution Model Every Programmer Must Internalize | [module-01.md](module-01.md) |
| 2 | Memory Hierarchy: The Single Most Important Topic | [module-02.md](module-02.md) |
| 3 | The Memory Consistency Model & Synchronization | [module-03.md](module-03.md) |
| 4 | Operating System Internals for Performance Engineers | [module-04.md](module-04.md) |

### Part II — Computer Architecture Deep Dive (The "Hennessy-Patterson + Mutlu" Layer)

| Module | Title | File |
|--------|-------|------|
| 5 | Pipelining & Hazards: The Foundation | [module-05.md](module-05.md) |
| 6 | Out-of-Order Execution: The Heart of Modern CPUs | [module-06.md](module-06.md) |
| 7 | Branch Prediction: Modern State of the Art | [module-07.md](module-07.md) |
| 8 | Instruction-Level Parallelism & Superscalar Execution | [module-08.md](module-08.md) |
| 9 | SIMD & Vector Computing: AVX-512 Mastery | [module-09.md](module-09.md) |
| 10 | Caches in Microarchitectural Detail | [module-10.md](module-10.md) |
| 11 | Memory Subsystem: DRAM to DIMM to Controller | [module-11.md](module-11.md) |

### Part III — Parallelism & Concurrency Architecture

| Module | Title | File |
|--------|-------|------|
| 12 | Multi-Core & Thread-Level Parallelism | [module-12.md](module-12.md) |
| 13 | Coherence, Consistency, and Interconnects | [module-13.md](module-13.md) |

### Part IV — Performance Engineering Methodology

| Module | Title | File |
|--------|-------|------|
| 14 | The Performance Engineering Framework | [module-14.md](module-14.md) |
| 15 | Profiling Tools Mastery | [module-15.md](module-15.md) |
| 16 | Compiler Optimization & Code Generation | [module-16.md](module-16.md) |

### Part V — Intel Xeon Scalable: Complete Deep Dive

| Module | Title | File |
|--------|-------|------|
| 17 | Intel Xeon Microarchitecture Evolution | [module-17.md](module-17.md) |
| 18 | Sapphire Rapids (SPR) Microarchitecture In Depth | [module-18.md](module-18.md) |
| 19 | Intel Xeon Memory Subsystem | [module-19.md](module-19.md) |
| 20 | Intel-Specific Performance Optimization Techniques | [module-20.md](module-20.md) |

### Part VI — AMD EPYC: Complete Deep Dive

| Module | Title | File |
|--------|-------|------|
| 21 | AMD Zen Microarchitecture Evolution | [module-21.md](module-21.md) |
| 22 | Zen 4 (Genoa) Microarchitecture In Depth | [module-22.md](module-22.md) |
| 23 | AMD EPYC Memory Subsystem | [module-23.md](module-23.md) |
| 24 | AMD-Specific Optimization Techniques | [module-24.md](module-24.md) |

### Part VII — Systems Design for ML Inference Engines

| Module | Title | File |
|--------|-------|------|
| 25 | CPU Inference Engine Architecture | [module-25.md](module-25.md) |
| 26 | Kernel Implementation for CPU Inference | [module-26.md](module-26.md) |
| 27 | End-to-End Performance Optimization Workflow | [module-27.md](module-27.md) |

### Appendices

| Appendix | Title | File |
|----------|-------|------|
| A | Instruction Latency & Throughput Reference Tables | [appendix-a.md](appendix-a.md) |
| B | Hardware Performance Counter Reference | [appendix-b.md](appendix-b.md) |
| C | CPUID Feature Detection Code | [appendix-c.md](appendix-c.md) |
| D | NUMA Topology Interrogation | [appendix-d.md](appendix-d.md) |
| E | Benchmark Baselines | [appendix-e.md](appendix-e.md) |

---

## Primary References

1. **CS:APP** — Bryant & O'Hallaron, *Computer Systems: A Programmer's Perspective*, 3rd Ed.
2. **H&P** — Hennessy & Patterson, *Computer Architecture: A Quantitative Approach*, 6th Ed.
3. **P&H** — Patterson & Hennessy, *Computer Organization and Design*, RISC-V Ed.
4. **McKenney** — Paul McKenney, *Is Parallel Programming Hard, And, If So, What Can You Do About It?*
5. **Intel Opt Manual** — Intel® 64 and IA-32 Architectures Optimization Reference Manual
6. **AMD Opt Guide** — AMD Software Optimization Guide for AMD EPYC™ 9004 Series Processors
7. **Agner Fog** — Optimization Manuals, Vols 1–4 (microarchitecture, instruction tables, calling conventions, C++ optimization)
8. **Mutlu Lectures** — Onur Mutlu, ETH Zürich / CMU Computer Architecture Lectures (YouTube)
9. **Dendibakh** — Denis Bakhvalov, *Performance Analysis and Tuning on Modern CPUs*

---

## Module Format

Every module follows this structure:

1. **CONCEPTUAL FOUNDATION** — Deep theory with cited sources
2. **MENTAL MODEL** — ASCII diagrams of hardware/software mechanisms
3. **PERFORMANCE LENS** — Impact on code performance
4. **ANNOTATED CODE** — C/C++ with intrinsics, assembly, line-by-line explanation
5. **EXPERT INSIGHT** — Non-obvious truths that separate senior from junior engineers
6. **BENCHMARK / MEASUREMENT** — Tools, commands, what to look for
7. **ML SYSTEMS RELEVANCE** — Direct application to inference engine design
8. **PhD QUALIFIER QUESTIONS** — 5 exam-grade questions per module
9. **READING LIST** — Exact chapters/sections from references
