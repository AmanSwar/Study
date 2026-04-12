import { MarkdownModulePage } from '@/components/content/MarkdownModulePage'

export const metadata = { title: 'Module 11: Memory Subsystem — DRAM to DIMM to Controller' }

export default function Module11Page() {
  return (
    <MarkdownModulePage
      sourcePath="intel/module-11.md"
      breadcrumbs={[
        { label: 'Intel/AMD', href: '/intel' },
        { label: 'Part 2: Architecture', href: '/intel/part-02-architecture' },
        { label: 'Module 11' },
      ]}
      moduleNumber={11}
      title="Memory Subsystem — DRAM to DIMM to Controller"
      track="intel"
      part={2}
      readingTime="30 min"
      description="DRAM timing, rank/bank parallelism, memory controller scheduling"
      prev={{ href: '/intel/part-02-architecture/module-10', label: 'Caches Microarch' }}
      next={{ href: '/intel/part-03-parallelism/module-12', label: 'Multi-Core' }}
    />
  )
}
