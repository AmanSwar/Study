import { MarkdownModulePage } from '@/components/content/MarkdownModulePage'

export const metadata = { title: 'Module 30: VAD (Voice Activity Detection) Systems' }

export default function Module30Page() {
  return (
    <MarkdownModulePage
      sourcePath="MLsys/part_09_voice_ai/module_30_vad.md"
      breadcrumbs={[
        { label: 'MLsys', href: '/mlsys' },
        { label: 'Part 9: Voice AI', href: '/mlsys/part-09-voice-ai' },
        { label: 'Module 30' },
      ]}
      moduleNumber={30}
      title="VAD (Voice Activity Detection) Systems"
      track="mlsys"
      part={9}
      readingTime="25 min"
      description="Silero VAD, WebRTC VAD, streaming VAD architectures"
      prev={{ href: '/mlsys/part-09-voice-ai/module-29', label: 'Voice Pipeline' }}
      next={{ href: '/mlsys/part-09-voice-ai/module-31', label: 'Streaming ASR' }}
    />
  )
}
