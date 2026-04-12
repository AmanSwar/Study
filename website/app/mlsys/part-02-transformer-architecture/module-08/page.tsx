import { MarkdownModulePage } from '@/components/content/MarkdownModulePage'

export const metadata = { title: 'Module 8: Inference Serving Systems Architecture' }

export default function Module08Page() {
  return (
    <MarkdownModulePage
      sourcePath="MLsys/part_02_transformer_architecture/module_08_inference_serving_systems.md"
      breadcrumbs={[
        { label: 'MLsys', href: '/mlsys' },
        { label: 'Part 2: Transformer Architecture', href: '/mlsys/part-02-transformer-architecture' },
        { label: 'Module 8' },
      ]}
      moduleNumber={8}
      title="Inference Serving Systems Architecture"
      track="mlsys"
      part={2}
      readingTime="40 min"
      description="vLLM, TensorRT-LLM, request scheduling, SLA management"
      prev={{ href: '/mlsys/part-02-transformer-architecture/module-07', label: 'KV-Cache Management' }}
      next={{ href: '/mlsys/part-03-distributed-inference/module-09', label: 'Parallelism Strategies' }}
    />
  )
}
