import { MarkdownModulePage } from '@/components/content/MarkdownModulePage'

export const metadata = { title: 'Module 4: Operating System Internals for Performance Engineers' }

export default function Module04Page() {
  return (
    <MarkdownModulePage
      sourcePath="intel/module-04.md"
      breadcrumbs={[
        { label: 'Intel/AMD', href: '/intel' },
        { label: 'Part 1: Systems Foundations', href: '/intel/part-01-foundations' },
        { label: 'Module 4' },
      ]}
      moduleNumber={4}
      title="Operating System Internals for Performance Engineers"
      track="intel"
      part={1}
      readingTime="30 min"
      description="Scheduling, interrupts, page tables, huge pages, cgroups"
      prev={{ href: '/intel/part-01-foundations/module-03', label: 'Memory Consistency' }}
      next={{ href: '/intel/part-02-architecture/module-05', label: 'Pipelining' }}
    />
  )
}
