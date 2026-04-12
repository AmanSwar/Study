import { MarkdownModulePage } from '@/components/content/MarkdownModulePage'

export const metadata = { title: 'Module 17: Intel Xeon Microarchitecture Evolution' }

export default function Module17Page() {
  return (
    <MarkdownModulePage
      sourcePath="intel/module-17.md"
      breadcrumbs={[
        { label: 'Intel/AMD', href: '/intel' },
        { label: 'Part 5: Intel Xeon', href: '/intel/part-05-xeon' },
        { label: 'Module 17' },
      ]}
      moduleNumber={17}
      title="Intel Xeon Microarchitecture Evolution"
      track="intel"
      part={5}
      readingTime="30 min"
      description="Skylake through Emerald Rapids progression"
      prev={{ href: '/intel/part-04-performance/module-16', label: 'Compiler Optimization' }}
      next={{ href: '/intel/part-05-xeon/module-18', label: 'Sapphire Rapids' }}
    />
  )
}
