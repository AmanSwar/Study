import { MarkdownModulePage } from '@/components/content/MarkdownModulePage'

export const metadata = { title: 'Module 16: Apple Silicon Architecture for ML Engineers' }

export default function Module16Page() {
  return (
    <MarkdownModulePage
      sourcePath="MLsys/part_05_apple_silicon/module_16_apple_silicon_architecture.md"
      breadcrumbs={[
        { label: 'MLsys', href: '/mlsys' },
        { label: 'Part 5: Apple Silicon', href: '/mlsys/part-05-apple-silicon' },
        { label: 'Module 16' },
      ]}
      moduleNumber={16}
      title="Apple Silicon Architecture for ML Engineers"
      track="mlsys"
      part={5}
      readingTime="30 min"
      description="M-series SoC, unified memory, ANE, GPU architecture overview"
      prev={{ href: '/mlsys/part-04-cpu-inference/module-15', label: 'Full CPU Pipeline' }}
      next={{ href: '/mlsys/part-05-apple-silicon/module-17', label: 'CoreML & ANE' }}
    />
  )
}
