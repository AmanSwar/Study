import { MarkdownModulePage } from '@/components/content/MarkdownModulePage'

export const metadata = { title: 'Module 11: Why CPU Inference Matters & When It Wins' }

export default function Module11Page() {
  return (
    <MarkdownModulePage
      sourcePath="MLsys/part_04_cpu_inference/module_11_why_cpu_inference.md"
      breadcrumbs={[
        { label: 'MLsys', href: '/mlsys' },
        { label: 'Part 4: CPU Inference', href: '/mlsys/part-04-cpu-inference' },
        { label: 'Module 11' },
      ]}
      moduleNumber={11}
      title="Why CPU Inference Matters & When It Wins"
      track="mlsys"
      part={4}
      readingTime="25 min"
      description="Cost analysis, latency profiles, use cases where CPU beats GPU"
      prev={{ href: '/mlsys/part-03-distributed-inference/module-10', label: 'Communication Primitives' }}
      next={{ href: '/mlsys/part-04-cpu-inference/module-12', label: 'CPU Inference Engine Internals' }}
    />
  )
}
