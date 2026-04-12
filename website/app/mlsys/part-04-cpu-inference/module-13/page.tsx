import { MarkdownModulePage } from '@/components/content/MarkdownModulePage'

export const metadata = { title: 'Module 13: GEMM Optimization for CPU Inference' }

export default function Module13Page() {
  return (
    <MarkdownModulePage
      sourcePath="MLsys/part_04_cpu_inference/module_13_gemm_optimization_cpu.md"
      breadcrumbs={[
        { label: 'MLsys', href: '/mlsys' },
        { label: 'Part 4: CPU Inference', href: '/mlsys/part-04-cpu-inference' },
        { label: 'Module 13' },
      ]}
      moduleNumber={13}
      title="GEMM Optimization for CPU Inference"
      track="mlsys"
      part={4}
      readingTime="45 min"
      description="SIMD intrinsics, cache tiling, AMX/VNNI, micro-kernel design"
      prev={{ href: '/mlsys/part-04-cpu-inference/module-12', label: 'CPU Inference Engine' }}
      next={{ href: '/mlsys/part-04-cpu-inference/module-14', label: 'Attention for CPU' }}
    />
  )
}
