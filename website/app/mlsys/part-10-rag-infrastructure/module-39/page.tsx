import { MarkdownModulePage } from '@/components/content/MarkdownModulePage'

export const metadata = { title: 'Module 39: Observability & Monitoring' }

export default function Module39Page() {
  return (
    <MarkdownModulePage
      sourcePath="MLsys/part_10_rag_infrastructure/module_39_observability.md"
      breadcrumbs={[
        { label: 'MLsys', href: '/mlsys' },
        { label: 'Part 10: RAG & Infrastructure', href: '/mlsys/part-10-rag-infrastructure' },
        { label: 'Module 39' },
      ]}
      moduleNumber={39}
      title="Observability & Monitoring"
      track="mlsys"
      part={10}
      readingTime="25 min"
      description="ML inference monitoring, latency tracking, model quality, alerting"
      prev={{ href: '/mlsys/part-10-rag-infrastructure/module-38', label: 'Multimodal' }}

    />
  )
}
