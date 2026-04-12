import { MarkdownModulePage } from '@/components/content/MarkdownModulePage'

export const metadata = { title: 'Module 37: Deployment Toolchain' }

export default function Module37Page() {
  return (
    <MarkdownModulePage
      sourcePath="MLsys/part_10_rag_infrastructure/module_37_deployment_toolchain.md"
      breadcrumbs={[
        { label: 'MLsys', href: '/mlsys' },
        { label: 'Part 10: RAG & Infrastructure', href: '/mlsys/part-10-rag-infrastructure' },
        { label: 'Module 37' },
      ]}
      moduleNumber={37}
      title="Deployment Toolchain"
      track="mlsys"
      part={10}
      readingTime="30 min"
      description="Model serving frameworks, containerization, orchestration, A/B testing"
      prev={{ href: '/mlsys/part-10-rag-infrastructure/module-36', label: 'RAG Architecture' }}
      next={{ href: '/mlsys/part-10-rag-infrastructure/module-38', label: 'Multimodal Inference' }}
    />
  )
}
