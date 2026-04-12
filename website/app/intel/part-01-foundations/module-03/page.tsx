import { MarkdownModulePage } from '@/components/content/MarkdownModulePage'

export const metadata = { title: 'Module 3: Memory Consistency Model & Synchronization' }

export default function Module03Page() {
  return (
    <MarkdownModulePage
      sourcePath="intel/module-03.md"
      breadcrumbs={[
        { label: 'Intel/AMD', href: '/intel' },
        { label: 'Part 1: Systems Foundations', href: '/intel/part-01-foundations' },
        { label: 'Module 3' },
      ]}
      moduleNumber={3}
      title="Memory Consistency Model & Synchronization"
      track="intel"
      part={1}
      readingTime="30 min"
      description="TSO, sequential consistency, atomic operations, lock-free programming"
      prev={{ href: '/intel/part-01-foundations/module-02', label: 'Memory Hierarchy' }}
      next={{ href: '/intel/part-01-foundations/module-04', label: 'OS Internals' }}
    />
  )
}
