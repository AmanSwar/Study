import { MarkdownModulePage } from '@/components/content/MarkdownModulePage'

export const metadata = { title: 'Module 15: Full LLM Inference Pipeline on CPU' }

export default function Module15Page() {
  return (
    <MarkdownModulePage
      sourcePath="MLsys/part_04_cpu_inference/module_15_full_pipeline_cpu.md"
      breadcrumbs={[
        { label: 'MLsys', href: '/mlsys' },
        { label: 'Part 4: CPU Inference', href: '/mlsys/part-04-cpu-inference' },
        { label: 'Module 15' },
      ]}
      moduleNumber={15}
      title="Full LLM Inference Pipeline on CPU"
      track="mlsys"
      part={4}
      readingTime="45 min"
      description="End-to-end CPU inference, NUMA optimization, thread scheduling, deployment"
      prev={{ href: '/mlsys/part-04-cpu-inference/module-14', label: 'Attention for CPU' }}
      next={{ href: '/mlsys/part-05-apple-silicon/module-16', label: 'Apple Silicon Architecture' }}
    />
  )
}
