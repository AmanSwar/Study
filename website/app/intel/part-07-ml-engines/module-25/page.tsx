import { MarkdownModulePage } from '@/components/content/MarkdownModulePage'

export const metadata = { title: 'Module 25: CPU Inference Engine Architecture' }

export default function Module25Page() {
  return (
    <MarkdownModulePage
      sourcePath="intel/module-25.md"
      breadcrumbs={[
        { label: 'Intel/AMD', href: '/intel' },
        { label: 'Part 7: ML Engines', href: '/intel/part-07-ml-engines' },
        { label: 'Module 25' },
      ]}
      moduleNumber={25}
      title="CPU Inference Engine Architecture"
      track="intel"
      part={7}
      readingTime="35 min"
      description="Graph IR, operator scheduling, memory planning, thread pools"
      prev={{ href: '/intel/part-06-epyc/module-24', label: 'AMD Optimization' }}
      next={{ href: '/intel/part-07-ml-engines/module-26', label: 'Kernel Implementation' }}
    />
  )
}
