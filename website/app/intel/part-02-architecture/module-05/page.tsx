import { MarkdownModulePage } from '@/components/content/MarkdownModulePage'

export const metadata = { title: 'Module 5: Pipelining & Hazards' }

export default function Module05Page() {
  return (
    <MarkdownModulePage
      sourcePath="intel/module-05.md"
      breadcrumbs={[
        { label: 'Intel/AMD', href: '/intel' },
        { label: 'Part 2: Architecture', href: '/intel/part-02-architecture' },
        { label: 'Module 5' },
      ]}
      moduleNumber={5}
      title="Pipelining & Hazards"
      track="intel"
      part={2}
      readingTime="30 min"
      description="Pipeline stages, data/control hazards, forwarding, stalls"
      prev={{ href: '/intel/part-01-foundations/module-04', label: 'OS Internals' }}
      next={{ href: '/intel/part-02-architecture/module-06', label: 'OOO Execution' }}
    />
  )
}
