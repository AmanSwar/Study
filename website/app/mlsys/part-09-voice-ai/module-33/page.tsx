import { MarkdownModulePage } from '@/components/content/MarkdownModulePage'

export const metadata = { title: 'Module 33: TTS (Text-to-Speech) Systems Engineering' }

export default function Module33Page() {
  return (
    <MarkdownModulePage
      sourcePath="MLsys/part_09_voice_ai/module_33_tts.md"
      breadcrumbs={[
        { label: 'MLsys', href: '/mlsys' },
        { label: 'Part 9: Voice AI', href: '/mlsys/part-09-voice-ai' },
        { label: 'Module 33' },
      ]}
      moduleNumber={33}
      title="TTS (Text-to-Speech) Systems Engineering"
      track="mlsys"
      part={9}
      readingTime="30 min"
      description="Neural TTS, streaming synthesis, voice cloning, latency optimization"
      prev={{ href: '/mlsys/part-09-voice-ai/module-32', label: 'LLM Voice' }}
      next={{ href: '/mlsys/part-09-voice-ai/module-34', label: 'Audio Processing' }}
    />
  )
}
