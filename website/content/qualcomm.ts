import { Track } from '@/lib/types'

export const qualcommTrack: Track = {
  id: 'qualcomm',
  title: 'Qualcomm Hexagon NPU',
  shortTitle: 'Qualcomm',
  description: 'Complete reference for building custom inference engines on Hexagon NPU. Covers VLIW architecture, HVX vector programming, HTA/HMX tensor accelerators, and on-device inference optimization.',
  moduleCount: 10,
  appendixCount: 0,
  icon: 'Smartphone',
  color: 'orange',
  modules: [
    { id: 'module-01', number: 1, title: 'Hexagon SoC Architecture Internals', shortTitle: 'Hexagon Architecture', description: 'Hardware anatomy, VLIW threading model, ISA overview, SoC integration', readingTime: '35 min', sourceFile: 'module-01-hexagon-soc-architecture.md', href: '/qualcomm/module-01' },
    { id: 'module-02', number: 2, title: 'HVX Programming Model', shortTitle: 'HVX Programming', description: 'Vector operations, HVX intrinsics, 128-byte vector kernels', readingTime: '35 min', sourceFile: 'module-02-hvx-programming-model.md', href: '/qualcomm/module-02' },
    { id: 'module-03', number: 3, title: 'HTA/HMX Tensor Accelerators', shortTitle: 'HTA/HMX', description: 'Systolic arrays, tensor operations, HMX programming model', readingTime: '30 min', sourceFile: 'module-03-hta-hmx-accelerators.md', href: '/qualcomm/module-03' },
    { id: 'module-04', number: 4, title: 'Memory Subsystem Mastery', shortTitle: 'Memory Subsystem', description: 'VTCM, DMA engines, cache management, double buffering', readingTime: '35 min', sourceFile: 'module-04-memory-subsystem.md', href: '/qualcomm/module-04' },
    { id: 'module-05', number: 5, title: 'Quantization for Hexagon', shortTitle: 'Hexagon Quantization', description: 'INT8/INT4 quantization, requantization pipelines, accuracy preservation', readingTime: '30 min', sourceFile: 'module-05-quantization-for-hexagon.md', href: '/qualcomm/module-05' },
    { id: 'module-06', number: 6, title: 'Operator Kernel Design', shortTitle: 'Kernel Design', description: 'Conv2D, GEMM, attention kernels optimized for HVX', readingTime: '35 min', sourceFile: 'module-06-operator-kernel-design.md', href: '/qualcomm/module-06' },
    { id: 'module-07', number: 7, title: 'Inference Engine Architecture', shortTitle: 'Engine Architecture', description: 'Graph IR, operator fusion, scheduling, runtime design', readingTime: '35 min', sourceFile: 'module-07-inference-engine-architecture.md', href: '/qualcomm/module-07' },
    { id: 'module-08', number: 8, title: 'Performance Analysis & Profiling', shortTitle: 'Performance Analysis', description: 'Roofline model for Hexagon, bottleneck identification, profiling tools', readingTime: '30 min', sourceFile: 'module-08-performance-analysis.md', href: '/qualcomm/module-08' },
    { id: 'module-09', number: 9, title: 'Toolchain & Development Environment', shortTitle: 'Toolchain', description: 'Hexagon SDK, cross-compilation, debugging, simulation', readingTime: '30 min', sourceFile: 'module-09-toolchain-development.md', href: '/qualcomm/module-09' },
    { id: 'module-10', number: 10, title: 'Research Frontiers & Advanced Topics', shortTitle: 'Research Frontiers', description: 'Sparsity exploitation, INT4 inference, LLM on Hexagon, future directions', readingTime: '25 min', sourceFile: 'module-10-research-frontiers.md', href: '/qualcomm/module-10' },
  ],
}
