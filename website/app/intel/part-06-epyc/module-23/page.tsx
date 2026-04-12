import { MarkdownModulePage } from '@/components/content/MarkdownModulePage'

export const metadata = { title: 'Module 23: AMD EPYC Memory Subsystem' }

export default function Module23Page() {
  return (
    <MarkdownModulePage
      sourcePath="intel/module-23.md"
      breadcrumbs={[
        { label: 'Intel/AMD', href: '/intel' },
        { label: 'Part 6: AMD EPYC', href: '/intel/part-06-epyc' },
        { label: 'Module 23' },
      ]}
      moduleNumber={23}
      title="AMD EPYC Memory Subsystem"
      track="intel"
      part={6}
      readingTime="30 min"
      description="Infinity Fabric, NUMA topology, memory bandwidth optimization"
      prev={{ href: '/intel/part-06-epyc/module-22', label: 'Zen 4 Genoa' }}
      next={{ href: '/intel/part-06-epyc/module-24', label: 'AMD Optimization' }}
    />
  )
}
