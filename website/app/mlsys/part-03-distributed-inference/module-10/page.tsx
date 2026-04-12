import { MarkdownModulePage } from '@/components/content/MarkdownModulePage'

export const metadata = { title: 'Module 10: Communication Primitives & Collective Operations' }

export default function Module10Page() {
  return (
    <MarkdownModulePage
      sourcePath="MLsys/part_03_distributed_inference/module_10_communication_primitives.md"
      breadcrumbs={[
        { label: 'MLsys', href: '/mlsys' },
        { label: 'Part 3: Distributed Inference', href: '/mlsys/part-03-distributed-inference' },
        { label: 'Module 10' },
      ]}
      moduleNumber={10}
      title="Communication Primitives & Collective Operations"
      track="mlsys"
      part={3}
      readingTime="35 min"
      description="AllReduce, AllGather, NCCL, bandwidth optimization, latency hiding"
      prev={{ href: '/mlsys/part-03-distributed-inference/module-09', label: 'Parallelism Strategies' }}
      next={{ href: '/mlsys/part-04-cpu-inference/module-11', label: 'Why CPU Inference Matters' }}
    />
  )
}
