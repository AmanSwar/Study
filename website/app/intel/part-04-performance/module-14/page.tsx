import { MarkdownModulePage } from '@/components/content/MarkdownModulePage'

export const metadata = { title: 'Module 14: Performance Engineering Framework' }

export default function Module14Page() {
  return (
    <MarkdownModulePage
      sourcePath="intel/module-14.md"
      breadcrumbs={[
        { label: 'Intel/AMD', href: '/intel' },
        { label: 'Part 4: Performance Engineering', href: '/intel/part-04-performance' },
        { label: 'Module 14' },
      ]}
      moduleNumber={14}
      title="Performance Engineering Framework"
      track="intel"
      part={4}
      readingTime="25 min"
      description="Top-down analysis, bottleneck identification, roofline methodology"
      prev={{ href: '/intel/part-03-parallelism/module-13', label: 'Coherence' }}
      next={{ href: '/intel/part-04-performance/module-15', label: 'Profiling Tools' }}
    />
  )
}
