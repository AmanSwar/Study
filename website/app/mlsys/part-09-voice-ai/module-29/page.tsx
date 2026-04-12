import { MarkdownModulePage } from '@/components/content/MarkdownModulePage'

export const metadata = { title: 'Module 29: Voice AI Pipeline Architecture' }

export default function Module29Page() {
  return (
    <MarkdownModulePage
      sourcePath="MLsys/part_09_voice_ai/module_29_voice_pipeline_architecture.md"
      breadcrumbs={[
        { label: 'MLsys', href: '/mlsys' },
        { label: 'Part 9: Voice AI', href: '/mlsys/part-09-voice-ai' },
        { label: 'Module 29' },
      ]}
      moduleNumber={29}
      title="Voice AI Pipeline Architecture"
      track="mlsys"
      part={9}
      readingTime="40 min"
      description="End-to-end voice pipeline, latency budgets, streaming architecture"
      prev={{ href: '/mlsys/part-08-fine-tuning/module-28', label: 'RLHF & Alignment' }}
      next={{ href: '/mlsys/part-09-voice-ai/module-30', label: 'VAD Systems' }}
    />
  )
}
