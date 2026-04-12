import { MarkdownModulePage } from '@/components/content/MarkdownModulePage'

export const metadata = { title: 'Module 16: Compiler Optimization & Code Generation' }

export default function Module16Page() {
  return (
    <MarkdownModulePage
      sourcePath="intel/module-16.md"
      breadcrumbs={[
        { label: 'Intel/AMD', href: '/intel' },
        { label: 'Part 4: Performance Engineering', href: '/intel/part-04-performance' },
        { label: 'Module 16' },
      ]}
      moduleNumber={16}
      title="Compiler Optimization & Code Generation"
      track="intel"
      part={4}
      readingTime="30 min"
      description="ICC/GCC/LLVM optimization passes, PGO, LTO, vectorization reports"
      prev={{ href: '/intel/part-04-performance/module-15', label: 'Profiling Tools' }}
      next={{ href: '/intel/part-05-xeon/module-17', label: 'Xeon Evolution' }}
    />
  )
}
