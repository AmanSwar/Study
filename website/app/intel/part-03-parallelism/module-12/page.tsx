import { MarkdownModulePage } from '@/components/content/MarkdownModulePage'

export const metadata = { title: 'Module 12: Multi-Core & Thread-Level Parallelism' }

export default function Module12Page() {
  return (
    <MarkdownModulePage
      sourcePath="intel/module-12.md"
      breadcrumbs={[
        { label: 'Intel/AMD', href: '/intel' },
        { label: 'Part 3: Parallelism', href: '/intel/part-03-parallelism' },
        { label: 'Module 12' },
      ]}
      moduleNumber={12}
      title="Multi-Core & Thread-Level Parallelism"
      track="intel"
      part={3}
      readingTime="30 min"
      description="SMT, CMP, Amdahl's law, thread scheduling, affinity"
      prev={{ href: '/intel/part-02-architecture/module-11', label: 'Memory Subsystem' }}
      next={{ href: '/intel/part-03-parallelism/module-13', label: 'Coherence' }}
    />
  )
}
