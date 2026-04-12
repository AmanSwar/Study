import { MarkdownModulePage } from '@/components/content/MarkdownModulePage'

export const metadata = { title: 'Module 19: llama.cpp on Apple Silicon' }

export default function Module19Page() {
  return (
    <MarkdownModulePage
      sourcePath="MLsys/part_05_apple_silicon/module_19_llamacpp_apple.md"
      breadcrumbs={[
        { label: 'MLsys', href: '/mlsys' },
        { label: 'Part 5: Apple Silicon', href: '/mlsys/part-05-apple-silicon' },
        { label: 'Module 19' },
      ]}
      moduleNumber={19}
      title="llama.cpp on Apple Silicon"
      track="mlsys"
      part={5}
      readingTime="30 min"
      description="llama.cpp Metal backend, quantization on Apple Silicon, performance"
      prev={{ href: '/mlsys/part-05-apple-silicon/module-18', label: 'Metal GPU Inference' }}
      next={{ href: '/mlsys/part-05-apple-silicon/module-20', label: 'Full Stack on Apple' }}
    />
  )
}
