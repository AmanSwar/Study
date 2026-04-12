import { MarkdownModulePage } from '@/components/content/MarkdownModulePage'

export const metadata = { title: 'Module 2: Memory Hierarchy' }

export default function Module02Page() {
  return (
    <MarkdownModulePage
      sourcePath="intel/module-02.md"
      breadcrumbs={[
        { label: 'Intel/AMD', href: '/intel' },
        { label: 'Part 1: Systems Foundations', href: '/intel/part-01-foundations' },
        { label: 'Module 2' },
      ]}
      moduleNumber={2}
      title="Memory Hierarchy"
      track="intel"
      part={1}
      readingTime="35 min"
      description="Cache organization, replacement policies, prefetching, NUMA topology"
      prev={{ href: '/intel/part-01-foundations/module-01', label: 'Execution Model' }}
      next={{ href: '/intel/part-01-foundations/module-03', label: 'Memory Consistency' }}
    />
  )
}
