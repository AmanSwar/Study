import { MarkdownModulePage } from '@/components/content/MarkdownModulePage'

export const metadata = { title: 'Module 6: Operator Kernel Design' }

export default function Module06Page() {
  return (
    <MarkdownModulePage
      sourcePath="qualcomm/module-06-operator-kernel-design.md"
      breadcrumbs={[
        { label: 'Qualcomm', href: '/qualcomm' },
        { label: 'Module 6' },
      ]}
      moduleNumber={6}
      title="Operator Kernel Design"
      track="qualcomm"
      part={0}
      readingTime="5 hours"
      prerequisites={['Modules 2-5', 'Linear algebra', 'Convolution arithmetic']}
      description="Designing high-performance ML operator kernels for Hexagon: Conv2D, GEMM, attention, layer norm, and softmax kernels optimized for HVX and HMX."
      prev={{ href: '/qualcomm/module-05', label: 'Quantization for Hexagon' }}
      next={{ href: '/qualcomm/module-07', label: 'Inference Engine Architecture' }}
    />
  )
}
