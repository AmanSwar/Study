import { MarkdownModulePage } from '@/components/content/MarkdownModulePage'

export const metadata = { title: 'Module 19: Intel Xeon Memory Subsystem' }

export default function Module19Page() {
  return (
    <MarkdownModulePage
      sourcePath="intel/module-19.md"
      breadcrumbs={[
        { label: 'Intel/AMD', href: '/intel' },
        { label: 'Part 5: Intel Xeon', href: '/intel/part-05-xeon' },
        { label: 'Module 19' },
      ]}
      moduleNumber={19}
      title="Intel Xeon Memory Subsystem"
      track="intel"
      part={5}
      readingTime="30 min"
      description="DDR5, CXL, HBM integration, sub-NUMA clustering"
      prev={{ href: '/intel/part-05-xeon/module-18', label: 'Sapphire Rapids' }}
      next={{ href: '/intel/part-05-xeon/module-20', label: 'Intel Optimization' }}
    />
  )
}
