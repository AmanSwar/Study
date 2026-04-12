import { MarkdownModulePage } from '@/components/content/MarkdownModulePage'

export const metadata = { title: 'Module 9: Parallelism Strategies for Inference' }

export default function Module09Page() {
  return (
    <MarkdownModulePage
      sourcePath="MLsys/part_03_distributed_inference/module_09_parallelism_strategies.md"
      breadcrumbs={[
        { label: 'MLsys', href: '/mlsys' },
        { label: 'Part 3: Distributed Inference', href: '/mlsys/part-03-distributed-inference' },
        { label: 'Module 9' },
      ]}
      moduleNumber={9}
      title="Parallelism Strategies for Inference"
      track="mlsys"
      part={3}
      readingTime="35 min"
      description="Tensor parallelism, pipeline parallelism, expert parallelism for MoE"
      prev={{ href: '/mlsys/part-02-transformer-architecture/module-08', label: 'Inference Serving Systems' }}
      next={{ href: '/mlsys/part-03-distributed-inference/module-10', label: 'Communication Primitives' }}
    />
  )
}
