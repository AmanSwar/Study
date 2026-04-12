import { MarkdownModulePage } from '@/components/content/MarkdownModulePage'

export const metadata = { title: 'Module 3: HTA / HMX (Tensor & Matrix Accelerators)' }

export default function Module03Page() {
  return (
    <MarkdownModulePage
      sourcePath="qualcomm/module-03-hta-hmx-accelerators.md"
      breadcrumbs={[
        { label: 'Qualcomm', href: '/qualcomm' },
        { label: 'Module 3' },
      ]}
      moduleNumber={3}
      title="HTA / HMX (Tensor & Matrix Accelerators)"
      track="qualcomm"
      part={0}
      readingTime="4-5 hours"
      prerequisites={['Module 1: Hexagon Architecture', 'Module 2: HVX Programming', 'Systolic array concepts']}
      description="The Hexagon Tensor Accelerator (HTA) and Matrix Accelerator (HMX) — descriptor-driven programming model, 32x32 systolic array architecture, INT8/INT16/FP16/INT4 data formats, HTA+HVX pipeline orchestration, and triple-buffering optimization."
      prev={{ href: '/qualcomm/module-02', label: 'HVX Programming Model' }}
      next={{ href: '/qualcomm/module-04', label: 'Memory Subsystem Mastery' }}
    />
  )
}
