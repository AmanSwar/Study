import { MarkdownModulePage } from '@/components/content/MarkdownModulePage'

export const metadata = { title: 'Module 18: Sapphire Rapids (SPR) Microarchitecture In Depth' }

export default function Module18Page() {
  return (
    <MarkdownModulePage
      sourcePath="intel/module-18.md"
      breadcrumbs={[
        { label: 'Intel/AMD', href: '/intel' },
        { label: 'Part 5: Intel Xeon', href: '/intel/part-05-xeon' },
        { label: 'Module 18' },
      ]}
      moduleNumber={18}
      title="Sapphire Rapids (SPR) Microarchitecture In Depth"
      track="intel"
      part={5}
      readingTime="35 min"
      description="AMX, DSA, IAA, tile architecture, new instructions"
      prev={{ href: '/intel/part-05-xeon/module-17', label: 'Xeon Evolution' }}
      next={{ href: '/intel/part-05-xeon/module-19', label: 'Xeon Memory' }}
    />
  )
}
