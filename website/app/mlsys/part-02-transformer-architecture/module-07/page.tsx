import { MarkdownModulePage } from '@/components/content/MarkdownModulePage'

export const metadata = { title: 'Module 7: KV-Cache Management Systems' }

export default function Module07Page() {
  return (
    <MarkdownModulePage
      sourcePath="MLsys/part_02_transformer_architecture/module_07_kv_cache_management.md"
      breadcrumbs={[
        { label: 'MLsys', href: '/mlsys' },
        { label: 'Part 2: Transformer Architecture', href: '/mlsys/part-02-transformer-architecture' },
        { label: 'Module 7' },
      ]}
      moduleNumber={7}
      title="KV-Cache Management Systems"
      track="mlsys"
      part={2}
      readingTime="35 min"
      description="PagedAttention, vLLM, KV-cache compression, continuous batching"
      prev={{ href: '/mlsys/part-02-transformer-architecture/module-06', label: 'Attention Optimization' }}
      next={{ href: '/mlsys/part-02-transformer-architecture/module-08', label: 'Inference Serving Systems' }}
    />
  )
}
