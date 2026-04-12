import { MarkdownModulePage } from '@/components/content/MarkdownModulePage'

export const metadata = { title: 'Module 32: LLM Integration in Voice Pipeline' }

export default function Module32Page() {
  return (
    <MarkdownModulePage
      sourcePath="MLsys/part_09_voice_ai/module_32_llm_voice_integration.md"
      breadcrumbs={[
        { label: 'MLsys', href: '/mlsys' },
        { label: 'Part 9: Voice AI', href: '/mlsys/part-09-voice-ai' },
        { label: 'Module 32' },
      ]}
      moduleNumber={32}
      title="LLM Integration in Voice Pipeline"
      track="mlsys"
      part={9}
      readingTime="30 min"
      description="Streaming LLM responses, turn-taking, interruption handling"
      prev={{ href: '/mlsys/part-09-voice-ai/module-31', label: 'Streaming ASR' }}
      next={{ href: '/mlsys/part-09-voice-ai/module-33', label: 'TTS Systems' }}
    />
  )
}
