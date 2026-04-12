import { MarkdownModulePage } from '@/components/content/MarkdownModulePage'

export const metadata = { title: 'Module 26: Kernel Implementation for CPU Inference' }

export default function Module26Page() {
  return (
    <MarkdownModulePage
      sourcePath="intel/module-26.md"
      breadcrumbs={[
        { label: 'Intel/AMD', href: '/intel' },
        { label: 'Part 7: ML Engines', href: '/intel/part-07-ml-engines' },
        { label: 'Module 26' },
      ]}
      moduleNumber={26}
      title="Kernel Implementation for CPU Inference"
      track="intel"
      part={7}
      readingTime="35 min"
      description="GEMM micro-kernels, activation functions, fused operators"
      prev={{ href: '/intel/part-07-ml-engines/module-25', label: 'Engine Architecture' }}
      next={{ href: '/intel/part-07-ml-engines/module-27', label: 'E2E Optimization' }}
    />
  )
}
