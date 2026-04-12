import { MarkdownModulePage } from '@/components/content/MarkdownModulePage'

export const metadata = { title: 'Module 7: Branch Prediction' }

export default function Module07Page() {
  return (
    <MarkdownModulePage
      sourcePath="intel/module-07.md"
      breadcrumbs={[
        { label: 'Intel/AMD', href: '/intel' },
        { label: 'Part 2: Architecture', href: '/intel/part-02-architecture' },
        { label: 'Module 7' },
      ]}
      moduleNumber={7}
      title="Branch Prediction"
      track="intel"
      part={2}
      readingTime="25 min"
      description="Two-level predictors, TAGE, BTB, speculative execution"
      prev={{ href: '/intel/part-02-architecture/module-06', label: 'OOO Execution' }}
      next={{ href: '/intel/part-02-architecture/module-08', label: 'ILP' }}
    />
  )
}
