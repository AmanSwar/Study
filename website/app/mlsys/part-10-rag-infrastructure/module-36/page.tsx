import { MarkdownModulePage } from '@/components/content/MarkdownModulePage'

export const metadata = { title: 'Module 36: RAG Systems Architecture' }

export default function Module36Page() {
  return (
    <MarkdownModulePage
      sourcePath="MLsys/part_10_rag_infrastructure/module_36_rag_systems.md"
      breadcrumbs={[
        { label: 'MLsys', href: '/mlsys' },
        { label: 'Part 10: RAG & Infrastructure', href: '/mlsys/part-10-rag-infrastructure' },
        { label: 'Module 36' },
      ]}
      moduleNumber={36}
      title="RAG Systems Architecture"
      track="mlsys"
      part={10}
      readingTime="40 min"
      description="Retrieval pipelines, vector databases, chunking strategies, hybrid search"
      prev={{ href: '/mlsys/part-09-voice-ai/module-35', label: 'E2E Voice' }}
      next={{ href: '/mlsys/part-10-rag-infrastructure/module-37', label: 'Deployment Toolchain' }}
    />
  )
}
