import { MarkdownModulePage } from '@/components/content/MarkdownModulePage'

export const metadata = { title: 'Module 38: Multimodal Inference' }

export default function Module38Page() {
  return (
    <MarkdownModulePage
      sourcePath="MLsys/part_10_rag_infrastructure/module_38_multimodal_inference.md"
      breadcrumbs={[
        { label: 'MLsys', href: '/mlsys' },
        { label: 'Part 10: RAG & Infrastructure', href: '/mlsys/part-10-rag-infrastructure' },
        { label: 'Module 38' },
      ]}
      moduleNumber={38}
      title="Multimodal Inference"
      track="mlsys"
      part={10}
      readingTime="30 min"
      description="Vision-language models, audio-text, multimodal serving architecture"
      prev={{ href: '/mlsys/part-10-rag-infrastructure/module-37', label: 'Deployment Toolchain' }}
      next={{ href: '/mlsys/part-10-rag-infrastructure/module-39', label: 'Observability' }}
    />
  )
}
