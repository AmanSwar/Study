import { MarkdownModulePage } from '@/components/content/MarkdownModulePage'

export const metadata = { title: 'Module 4: Memory Subsystem Mastery' }

export default function Module04Page() {
  return (
    <MarkdownModulePage
      sourcePath="qualcomm/module-04-memory-subsystem.md"
      breadcrumbs={[
        { label: 'Qualcomm', href: '/qualcomm' },
        { label: 'Module 4' },
      ]}
      moduleNumber={4}
      title="Memory Subsystem Mastery"
      track="qualcomm"
      part={0}
      readingTime="4-5 hours"
      prerequisites={['Module 1', 'Cache hierarchy concepts', 'DMA programming']}
      description="Deep dive into Hexagon's memory hierarchy: VTCM (Vector TCM), DMA engines, cache management, double-buffering strategies, and bandwidth optimization for ML workloads."
      prev={{ href: '/qualcomm/module-03', label: 'HTA/HMX Tensor Accelerators' }}
      next={{ href: '/qualcomm/module-05', label: 'Quantization for Hexagon' }}
    />
  )
}
