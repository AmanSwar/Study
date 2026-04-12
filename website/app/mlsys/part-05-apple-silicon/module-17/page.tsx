import { MarkdownModulePage } from '@/components/content/MarkdownModulePage'

export const metadata = { title: 'Module 17: CoreML & ANE Programming' }

export default function Module17Page() {
  return (
    <MarkdownModulePage
      sourcePath="MLsys/part_05_apple_silicon/module_17_coreml_ane.md"
      breadcrumbs={[
        { label: 'MLsys', href: '/mlsys' },
        { label: 'Part 5: Apple Silicon', href: '/mlsys/part-05-apple-silicon' },
        { label: 'Module 17' },
      ]}
      moduleNumber={17}
      title="CoreML & ANE Programming"
      track="mlsys"
      part={5}
      readingTime="30 min"
      description="CoreML optimization, ANE constraints, model conversion"
      prev={{ href: '/mlsys/part-05-apple-silicon/module-16', label: 'Apple Silicon Architecture' }}
      next={{ href: '/mlsys/part-05-apple-silicon/module-18', label: 'Metal & GPU Inference' }}
    />
  )
}
