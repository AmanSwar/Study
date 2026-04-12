import { MarkdownModulePage } from '@/components/content/MarkdownModulePage'

export const metadata = { title: 'Module 22: TinyML — Inference on Microcontrollers' }

export default function Module22Page() {
  return (
    <MarkdownModulePage
      sourcePath="MLsys/part_06_edge_ai/module_22_tinyml.md"
      breadcrumbs={[
        { label: 'MLsys', href: '/mlsys' },
        { label: 'Part 6: Edge AI', href: '/mlsys/part-06-edge-ai' },
        { label: 'Module 22' },
      ]}
      moduleNumber={22}
      title="TinyML — Inference on Microcontrollers"
      track="mlsys"
      part={6}
      readingTime="25 min"
      description="TFLite Micro, memory-constrained inference, quantization for MCUs"
      prev={{ href: '/mlsys/part-06-edge-ai/module-21', label: 'Edge Hardware' }}
      next={{ href: '/mlsys/part-06-edge-ai/module-23', label: 'Mobile Inference' }}
    />
  )
}
