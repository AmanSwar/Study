import { MarkdownModulePage } from '@/components/content/MarkdownModulePage'

export const metadata = { title: 'Module 15: Profiling Tools Mastery' }

export default function Module15Page() {
  return (
    <MarkdownModulePage
      sourcePath="intel/module-15.md"
      breadcrumbs={[
        { label: 'Intel/AMD', href: '/intel' },
        { label: 'Part 4: Performance Engineering', href: '/intel/part-04-performance' },
        { label: 'Module 15' },
      ]}
      moduleNumber={15}
      title="Profiling Tools Mastery"
      track="intel"
      part={4}
      readingTime="30 min"
      description="perf, VTune, uProf, hardware performance counters"
      prev={{ href: '/intel/part-04-performance/module-14', label: 'Performance Framework' }}
      next={{ href: '/intel/part-04-performance/module-16', label: 'Compiler Optimization' }}
    />
  )
}
