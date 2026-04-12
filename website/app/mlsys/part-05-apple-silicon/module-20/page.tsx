import { MarkdownModulePage } from '@/components/content/MarkdownModulePage'

export const metadata = { title: 'Module 20: Full Stack Inference on Apple Silicon' }

export default function Module20Page() {
  return (
    <MarkdownModulePage
      sourcePath="MLsys/part_05_apple_silicon/module_20_full_stack_apple.md"
      breadcrumbs={[
        { label: 'MLsys', href: '/mlsys' },
        { label: 'Part 5: Apple Silicon', href: '/mlsys/part-05-apple-silicon' },
        { label: 'Module 20' },
      ]}
      moduleNumber={20}
      title="Full Stack Inference on Apple Silicon"
      track="mlsys"
      part={5}
      readingTime="30 min"
      description="Production deployment on Mac, ANE+GPU hybrid, memory management"
      prev={{ href: '/mlsys/part-05-apple-silicon/module-19', label: 'llama.cpp Apple' }}
      next={{ href: '/mlsys/part-06-edge-ai/module-21', label: 'Edge Hardware Landscape' }}
    />
  )
}
