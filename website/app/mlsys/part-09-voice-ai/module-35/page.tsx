import { MarkdownModulePage } from '@/components/content/MarkdownModulePage'

export const metadata = { title: 'Module 35: End-to-End Voice AI Optimization' }

export default function Module35Page() {
  return (
    <MarkdownModulePage
      sourcePath="MLsys/part_09_voice_ai/module_35_e2e_voice_optimization.md"
      breadcrumbs={[
        { label: 'MLsys', href: '/mlsys' },
        { label: 'Part 9: Voice AI', href: '/mlsys/part-09-voice-ai' },
        { label: 'Module 35' },
      ]}
      moduleNumber={35}
      title="End-to-End Voice AI Optimization"
      track="mlsys"
      part={9}
      readingTime="30 min"
      description="Full pipeline optimization, latency reduction, production deployment"
      prev={{ href: '/mlsys/part-09-voice-ai/module-34', label: 'Audio Processing' }}
      next={{ href: '/mlsys/part-10-rag-infrastructure/module-36', label: 'RAG Architecture' }}
    />
  )
}
