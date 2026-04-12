import { MarkdownModulePage } from '@/components/content/MarkdownModulePage'

export const metadata = { title: 'Module 14: Attention Implementation for CPU' }

export default function Module14Page() {
  return (
    <MarkdownModulePage
      sourcePath="MLsys/part_04_cpu_inference/module_14_attention_cpu.md"
      breadcrumbs={[
        { label: 'MLsys', href: '/mlsys' },
        { label: 'Part 4: CPU Inference', href: '/mlsys/part-04-cpu-inference' },
        { label: 'Module 14' },
      ]}
      moduleNumber={14}
      title="Attention Implementation for CPU"
      track="mlsys"
      part={4}
      readingTime="35 min"
      description="CPU-optimized attention kernels, SIMD attention, memory access patterns"
      prev={{ href: '/mlsys/part-04-cpu-inference/module-13', label: 'GEMM Optimization' }}
      next={{ href: '/mlsys/part-04-cpu-inference/module-15', label: 'Full LLM Pipeline on CPU' }}
    />
  )
}
