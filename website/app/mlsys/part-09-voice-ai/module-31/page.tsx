import { MarkdownModulePage } from '@/components/content/MarkdownModulePage'

export const metadata = { title: 'Module 31: Streaming ASR' }

export default function Module31Page() {
  return (
    <MarkdownModulePage
      sourcePath="MLsys/part_09_voice_ai/module_31_streaming_asr.md"
      breadcrumbs={[
        { label: 'MLsys', href: '/mlsys' },
        { label: 'Part 9: Voice AI', href: '/mlsys/part-09-voice-ai' },
        { label: 'Module 31' },
      ]}
      moduleNumber={31}
      title="Streaming ASR"
      track="mlsys"
      part={9}
      readingTime="30 min"
      description="Whisper, CTC/RNN-T, streaming recognition, word-level timestamps"
      prev={{ href: '/mlsys/part-09-voice-ai/module-30', label: 'VAD' }}
      next={{ href: '/mlsys/part-09-voice-ai/module-32', label: 'LLM Voice Integration' }}
    />
  )
}
