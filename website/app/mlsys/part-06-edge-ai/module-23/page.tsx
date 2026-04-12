import { MarkdownModulePage } from '@/components/content/MarkdownModulePage'

export const metadata = { title: 'Module 23: Mobile Inference — Android & iOS' }

export default function Module23Page() {
  return (
    <MarkdownModulePage
      sourcePath="MLsys/part_06_edge_ai/module_23_mobile_inference.md"
      breadcrumbs={[
        { label: 'MLsys', href: '/mlsys' },
        { label: 'Part 6: Edge AI', href: '/mlsys/part-06-edge-ai' },
        { label: 'Module 23' },
      ]}
      moduleNumber={23}
      title="Mobile Inference — Android & iOS"
      track="mlsys"
      part={6}
      readingTime="30 min"
      description="TFLite, CoreML, NNAPI, mobile GPU delegates, on-device optimization"
      prev={{ href: '/mlsys/part-06-edge-ai/module-22', label: 'TinyML' }}
      next={{ href: '/mlsys/part-06-edge-ai/module-24', label: 'Edge LLM Inference' }}
    />
  )
}
