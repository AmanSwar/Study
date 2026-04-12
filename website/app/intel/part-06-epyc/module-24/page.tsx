import { MarkdownModulePage } from '@/components/content/MarkdownModulePage'

export const metadata = { title: 'Module 24: AMD-Specific Optimization Techniques' }

export default function Module24Page() {
  return (
    <MarkdownModulePage
      sourcePath="intel/module-24.md"
      breadcrumbs={[
        { label: 'Intel/AMD', href: '/intel' },
        { label: 'Part 6: AMD EPYC', href: '/intel/part-06-epyc' },
        { label: 'Module 24' },
      ]}
      moduleNumber={24}
      title="AMD-Specific Optimization Techniques"
      track="intel"
      part={6}
      readingTime="30 min"
      description="NUMA-aware allocation, IF bandwidth tuning, AMD uProf"
      prev={{ href: '/intel/part-06-epyc/module-23', label: 'EPYC Memory' }}
      next={{ href: '/intel/part-07-ml-engines/module-25', label: 'Engine Architecture' }}
    />
  )
}
