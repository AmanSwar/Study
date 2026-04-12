import { MarkdownModulePage } from '@/components/content/MarkdownModulePage'

export const metadata = { title: 'Module 5: Quantization for Hexagon' }

export default function Module05Page() {
  return (
    <MarkdownModulePage
      sourcePath="qualcomm/module-05-quantization-for-hexagon.md"
      breadcrumbs={[
        { label: 'Qualcomm', href: '/qualcomm' },
        { label: 'Module 5' },
      ]}
      moduleNumber={5}
      title="Quantization for Hexagon"
      track="qualcomm"
      part={0}
      readingTime="4 hours"
      prerequisites={['Module 2: HVX Programming', 'INT8 quantization basics']}
      description="INT8/INT4/INT2 quantization strategies for Hexagon: requantization pipelines, accuracy preservation, per-channel/per-tensor quantization, and hardware-accelerated requantize operations."
      prev={{ href: '/qualcomm/module-04', label: 'Memory Subsystem Mastery' }}
      next={{ href: '/qualcomm/module-06', label: 'Operator Kernel Design' }}
    />
  )
}
