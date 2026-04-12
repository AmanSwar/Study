import { MarkdownModulePage } from '@/components/content/MarkdownModulePage'

export const metadata = { title: 'Module 22: Zen 4 (Genoa) Microarchitecture In Depth' }

export default function Module22Page() {
  return (
    <MarkdownModulePage
      sourcePath="intel/module-22.md"
      breadcrumbs={[
        { label: 'Intel/AMD', href: '/intel' },
        { label: 'Part 6: AMD EPYC', href: '/intel/part-06-epyc' },
        { label: 'Module 22' },
      ]}
      moduleNumber={22}
      title="Zen 4 (Genoa) Microarchitecture In Depth"
      track="intel"
      part={6}
      readingTime="35 min"
      description="CCD/IOD architecture, AVX-512, cache hierarchy changes"
      prev={{ href: '/intel/part-06-epyc/module-21', label: 'Zen Evolution' }}
      next={{ href: '/intel/part-06-epyc/module-23', label: 'EPYC Memory' }}
    />
  )
}
