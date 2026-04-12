import { MarkdownModulePage } from '@/components/content/MarkdownModulePage'

export const metadata = { title: 'Module 8: Instruction-Level Parallelism' }

export default function Module08Page() {
  return (
    <MarkdownModulePage
      sourcePath="intel/module-08.md"
      breadcrumbs={[
        { label: 'Intel/AMD', href: '/intel' },
        { label: 'Part 2: Architecture', href: '/intel/part-02-architecture' },
        { label: 'Module 8' },
      ]}
      moduleNumber={8}
      title="Instruction-Level Parallelism"
      track="intel"
      part={2}
      readingTime="30 min"
      description="ILP limits, superscalar width, instruction windows, dispatch"
      prev={{ href: '/intel/part-02-architecture/module-07', label: 'Branch Prediction' }}
      next={{ href: '/intel/part-02-architecture/module-09', label: 'SIMD' }}
    />
  )
}
