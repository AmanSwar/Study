import { MarkdownModulePage } from '@/components/content/MarkdownModulePage'

export const metadata = { title: 'Module 10: Caches in Microarchitectural Detail' }

export default function Module10Page() {
  return (
    <MarkdownModulePage
      sourcePath="intel/module-10.md"
      breadcrumbs={[
        { label: 'Intel/AMD', href: '/intel' },
        { label: 'Part 2: Architecture', href: '/intel/part-02-architecture' },
        { label: 'Module 10' },
      ]}
      moduleNumber={10}
      title="Caches in Microarchitectural Detail"
      track="intel"
      part={2}
      readingTime="30 min"
      description="Set-associative caches, MSHR, non-blocking caches, prefetch streams"
      prev={{ href: '/intel/part-02-architecture/module-09', label: 'SIMD' }}
      next={{ href: '/intel/part-02-architecture/module-11', label: 'Memory Subsystem' }}
    />
  )
}
