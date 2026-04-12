import { MarkdownModulePage } from '@/components/content/MarkdownModulePage'

export const metadata = { title: 'Module 21: Edge AI Hardware Landscape' }

export default function Module21Page() {
  return (
    <MarkdownModulePage
      sourcePath="MLsys/part_06_edge_ai/module_21_edge_hardware_landscape.md"
      breadcrumbs={[
        { label: 'MLsys', href: '/mlsys' },
        { label: 'Part 6: Edge AI', href: '/mlsys/part-06-edge-ai' },
        { label: 'Module 21' },
      ]}
      moduleNumber={21}
      title="Edge AI Hardware Landscape"
      track="mlsys"
      part={6}
      readingTime="30 min"
      description="Mobile SoCs, NPUs, microcontrollers, hardware comparison matrix"
      prev={{ href: '/mlsys/part-05-apple-silicon/module-20', label: 'Full Stack Apple' }}
      next={{ href: '/mlsys/part-06-edge-ai/module-22', label: 'TinyML' }}
    />
  )
}
