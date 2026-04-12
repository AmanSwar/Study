import { MarkdownModulePage } from '@/components/content/MarkdownModulePage'

export const metadata = { title: 'Module 24: Edge LLM Inference' }

export default function Module24Page() {
  return (
    <MarkdownModulePage
      sourcePath="MLsys/part_06_edge_ai/module_24_edge_llm.md"
      breadcrumbs={[
        { label: 'MLsys', href: '/mlsys' },
        { label: 'Part 6: Edge AI', href: '/mlsys/part-06-edge-ai' },
        { label: 'Module 24' },
      ]}
      moduleNumber={24}
      title="Edge LLM Inference"
      track="mlsys"
      part={6}
      readingTime="30 min"
      description="Running LLMs on mobile/edge, model compression, speculative decoding"
      prev={{ href: '/mlsys/part-06-edge-ai/module-23', label: 'Mobile Inference' }}
      next={{ href: '/mlsys/part-07-gpu-inference/module-25', label: 'Modern GPU Stack' }}
    />
  )
}
