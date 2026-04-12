import { MarkdownModulePage } from '@/components/content/MarkdownModulePage'

export const metadata = { title: 'Module 9: SIMD & Vector Computing — AVX-512 Mastery' }

export default function Module09Page() {
  return (
    <MarkdownModulePage
      sourcePath="intel/module-09.md"
      breadcrumbs={[
        { label: 'Intel/AMD', href: '/intel' },
        { label: 'Part 2: Architecture', href: '/intel/part-02-architecture' },
        { label: 'Module 9' },
      ]}
      moduleNumber={9}
      title="SIMD & Vector Computing — AVX-512 Mastery"
      track="intel"
      part={2}
      readingTime="35 min"
      description="SSE through AVX-512, vector intrinsics, auto-vectorization"
      prev={{ href: '/intel/part-02-architecture/module-08', label: 'ILP' }}
      next={{ href: '/intel/part-02-architecture/module-10', label: 'Caches Microarch' }}
    />
  )
}
