import { MarkdownModulePage } from '@/components/content/MarkdownModulePage'

export const metadata = { title: 'Module 6: Out-of-Order Execution' }

export default function Module06Page() {
  return (
    <MarkdownModulePage
      sourcePath="intel/module-06.md"
      breadcrumbs={[
        { label: 'Intel/AMD', href: '/intel' },
        { label: 'Part 2: Architecture', href: '/intel/part-02-architecture' },
        { label: 'Module 6' },
      ]}
      moduleNumber={6}
      title="Out-of-Order Execution"
      track="intel"
      part={2}
      readingTime="35 min"
      description="Tomasulo algorithm, ROB, reservation stations, register renaming"
      prev={{ href: '/intel/part-02-architecture/module-05', label: 'Pipelining' }}
      next={{ href: '/intel/part-02-architecture/module-07', label: 'Branch Prediction' }}
    />
  )
}
