# Qualcomm Hexagon NPU — PhD-Level Study Curriculum

## Building a Custom Inference Engine on Hexagon from First Principles

---

**Target Audience:** Graduate researchers and senior engineers who want to deeply understand the Qualcomm Hexagon NPU hardware and build a custom ML inference engine on top of it.

**Prerequisites:**
- Strong C/C++ programming skills
- Solid understanding of computer architecture (pipelines, caches, SIMD)
- Experience building ML models in PyTorch/TensorFlow
- No prior Hexagon or DSP programming experience required

**Scope:** 10 modules, ~20,000+ lines of technical content, covering everything from silicon architecture to production inference engine design.

---

## Curriculum Map

### Foundation Layer

| Module | Title | Focus | Lines |
|--------|-------|-------|-------|
| [Module 1](module-01-hexagon-soc-architecture.md) | **Hexagon SoC Architecture Internals** | Hardware anatomy, threading, memory hierarchy, ISA, SoC integration | 2200+ |
| [Module 2](module-02-hvx-programming-model.md) | **HVX Programming Model** | Vector registers, intrinsics, convolution kernels, software pipelining | 2300+ |
| [Module 3](module-03-hta-hmx-accelerators.md) | **HTA / HMX Accelerators** | Systolic arrays, tensor operations, HVX+HMX orchestration, double-buffering | 1870+ |

### Systems Layer

| Module | Title | Focus | Lines |
|--------|-------|-------|-------|
| [Module 4](module-04-memory-subsystem.md) | **Memory Subsystem Mastery** | VTCM, DMA engines, cache management, scratchpad allocators | 2400+ |
| [Module 5](module-05-quantization-for-hexagon.md) | **Quantization for Hexagon** | INT8/INT4, per-channel, requantization in HVX, PyTorch→Hexagon pipeline | 1850+ |
| [Module 6](module-06-operator-kernel-design.md) | **Operator Kernel Design** | Conv2D, GEMM, Softmax, LayerNorm, Attention — all with HVX code | 2450+ |

### Engine Layer

| Module | Title | Focus | Lines |
|--------|-------|-------|-------|
| [Module 7](module-07-inference-engine-architecture.md) | **Inference Engine Architecture** | Graph IR, operator fusion, memory planning, execution scheduling | 2520+ |
| [Module 8](module-08-performance-analysis.md) | **Performance Analysis & Profiling** | Roofline model, profiling tools, bottleneck identification, benchmarking | 1700+ |

### Practice Layer

| Module | Title | Focus | Lines |
|--------|-------|-------|-------|
| [Module 9](module-09-toolchain-development.md) | **Toolchain & Development Environment** | SDK setup, cross-compilation, FastRPC, debugging, Android integration | 2900+ |
| [Module 10](module-10-research-frontiers.md) | **Research Frontiers & Advanced Topics** | Sparsity, INT4, LLM inference, multi-DSP, beyond QNN/SNPE | 2100+ |

---

## Recommended Study Path

```
                    ┌─────────────┐
                    │  Module 1   │  ← Start here: understand the hardware
                    │  SoC Arch   │
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │  Module 9   │  ← Set up your dev environment early
                    │  Toolchain  │
                    └──────┬──────┘
                           │
              ┌────────────┼────────────┐
              │            │            │
       ┌──────▼──────┐    │     ┌──────▼──────┐
       │  Module 2   │    │     │  Module 4   │
       │  HVX Prog   │    │     │  Memory     │
       └──────┬──────┘    │     └──────┬──────┘
              │            │            │
              └────────────┼────────────┘
                           │
                    ┌──────▼──────┐
                    │  Module 3   │  ← Now you can understand HMX
                    │  HTA / HMX  │
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │  Module 5   │  ← Quantization theory
                    │  Quant      │
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │  Module 6   │  ← Write real kernels
                    │  Operators  │
                    └──────┬──────┘
                           │
              ┌────────────┼────────────┐
              │                         │
       ┌──────▼──────┐          ┌──────▼──────┐
       │  Module 7   │          │  Module 8   │
       │  Engine     │          │  Profiling  │
       └──────┬──────┘          └──────┬──────┘
              │                         │
              └────────────┬────────────┘
                           │
                    ┌──────▼──────┐
                    │  Module 10  │  ← Research frontiers
                    │  Advanced   │
                    └─────────────┘
```

---

## What Each Module Contains

Every module follows a consistent format:

1. **Theory Section** — Deep, precise explanations with no hand-waving. Mathematical derivations where applicable.
2. **Working Code Examples** — Complete, annotated C/C++ with Hexagon intrinsics (and Python where relevant). Line-by-line explanations.
3. **Expert Insight Callouts (⚡)** — Non-obvious tips that separate good engineers from great ones.
4. **ASCII Architecture Diagrams** — Visual representations of data flow, memory layouts, pipeline stages.
5. **Self-Assessment Questions** — PhD qualifier exam style. Test deep understanding, not memorization.
6. **SDK Documentation Pointers** — Exact references to Qualcomm Hexagon SDK docs and whitepapers.

---

## Key Hexagon Generations Covered

| Generation | Key Features | Primary Coverage |
|-----------|-------------|-----------------|
| **v65** | Baseline HVX-128 | Modules 1, 2 |
| **v66** | Improved HVX, audio DSP focus | Modules 1, 2, 9 |
| **v68** | HTA introduction, 256-bit HVX option | Modules 1, 3, 4 |
| **v69** | Incremental improvements | Module 1 |
| **v73** | Enhanced caches, better HVX | Modules 1, 4, 8 |
| **v75** | HMX introduction, larger VTCM | Modules 1, 3, 6, 10 |

---

## How to Use This Curriculum

**If you're building an inference engine from scratch:**
Follow the study path above sequentially. By Module 7, you'll have enough knowledge to architect the full system.

**If you're optimizing an existing engine (e.g., extending QNN):**
Start with Modules 1, 2, 3 for hardware understanding, then jump to Module 6 for kernel optimization and Module 8 for profiling.

**If you're researching LLM deployment on mobile NPUs:**
Modules 1, 3, 5, 6 (Attention section), 7, and 10 (LLM section) form a focused reading path.

**If you're preparing for a Qualcomm NPU engineering interview:**
The self-assessment questions across all modules form a comprehensive preparation set.

---

## Essential External References

- **Qualcomm Hexagon SDK Documentation** — Primary reference for APIs and tools
- **Hexagon V66/V68 Programmer's Reference Manual** — ISA details
- **Qualcomm AI Engine Direct SDK** — HTA/HMX programming
- **QNN SDK Documentation** — Production inference framework reference
- **Hexagon Vector Extensions (HVX) Programmer's Reference** — Intrinsics catalog

---

*This curriculum was designed to be so complete that the reader never needs another source. Every concept is explained from first principles, every kernel is fully annotated, and every architectural decision is justified.*
