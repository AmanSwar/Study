import { MarkdownModulePage } from '@/components/content/MarkdownModulePage'

export const metadata = { title: 'Module 34: Audio Processing & Codec Systems' }

export default function Module34Page() {
  return (
    <MarkdownModulePage
      sourcePath="MLsys/part_09_voice_ai/module_34_audio_processing.md"
      breadcrumbs={[
        { label: 'MLsys', href: '/mlsys' },
        { label: 'Part 9: Voice AI', href: '/mlsys/part-09-voice-ai' },
        { label: 'Module 34' },
      ]}
      moduleNumber={34}
      title="Audio Processing & Codec Systems"
      track="mlsys"
      part={9}
      readingTime="25 min"
      description="Audio codecs, noise suppression, echo cancellation, audio pipeline"
      prev={{ href: '/mlsys/part-09-voice-ai/module-33', label: 'TTS' }}
      next={{ href: '/mlsys/part-09-voice-ai/module-35', label: 'E2E Voice Optimization' }}
    />
  )
}
