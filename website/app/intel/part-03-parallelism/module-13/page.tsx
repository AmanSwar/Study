import { MarkdownModulePage } from '@/components/content/MarkdownModulePage'

export const metadata = { title: 'Module 13: Coherence, Consistency, and Interconnects' }

export default function Module13Page() {
  return (
    <MarkdownModulePage
      sourcePath="intel/module-13.md"
      breadcrumbs={[
        { label: 'Intel/AMD', href: '/intel' },
        { label: 'Part 3: Parallelism', href: '/intel/part-03-parallelism' },
        { label: 'Module 13' },
      ]}
      moduleNumber={13}
      title="Coherence, Consistency, and Interconnects"
      track="intel"
      part={3}
      readingTime="30 min"
      description="MESI/MOESI, snooping, directory protocols, mesh interconnects"
      prev={{ href: '/intel/part-03-parallelism/module-12', label: 'Multi-Core' }}
      next={{ href: '/intel/part-04-performance/module-14', label: 'Performance Framework' }}
    />
  )
}
