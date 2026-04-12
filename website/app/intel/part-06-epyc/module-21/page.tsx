import { MarkdownModulePage } from '@/components/content/MarkdownModulePage'

export const metadata = { title: 'Module 21: AMD Zen Microarchitecture Evolution' }

export default function Module21Page() {
  return (
    <MarkdownModulePage
      sourcePath="intel/module-21.md"
      breadcrumbs={[
        { label: 'Intel/AMD', href: '/intel' },
        { label: 'Part 6: AMD EPYC', href: '/intel/part-06-epyc' },
        { label: 'Module 21' },
      ]}
      moduleNumber={21}
      title="AMD Zen Microarchitecture Evolution"
      track="intel"
      part={6}
      readingTime="30 min"
      description="Zen 1 through Zen 4, chiplet design, infinity fabric"
      prev={{ href: '/intel/part-05-xeon/module-20', label: 'Intel Optimization' }}
      next={{ href: '/intel/part-06-epyc/module-22', label: 'Zen 4 Genoa' }}
    />
  )
}
