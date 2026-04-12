import { MarkdownModulePage } from '@/components/content/MarkdownModulePage'

export const metadata = { title: 'Module 6: Attention Mechanism — Systems Optimization' }

export default function Module06Page() {
  return (
    <MarkdownModulePage
      sourcePath="MLsys/part_02_transformer_architecture/module_06_attention_optimization.md"
      breadcrumbs={[
        { label: 'MLsys', href: '/mlsys' },
        { label: 'Part 2: Transformer Architecture', href: '/mlsys/part-02-transformer-architecture' },
        { label: 'Module 6' },
      ]}
      moduleNumber={6}
      title="Attention Mechanism — Systems Optimization"
      track="mlsys"
      part={2}
      readingTime="45 min"
      description="FlashAttention, multi-query attention, online softmax, hardware mapping"
      prev={{ href: '/mlsys/part-01-inference-fundamentals/module-05', label: 'Graph Optimization' }}
      next={{ href: '/mlsys/part-02-transformer-architecture/module-07', label: 'KV-Cache Management' }}
    />
  )
}
