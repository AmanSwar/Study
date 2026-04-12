import { MarkdownModulePage } from '@/components/content/MarkdownModulePage'

export const metadata = { title: 'Module 7: Inference Engine Architecture' }

export default function Module07Page() {
  return (
    <MarkdownModulePage
      sourcePath="qualcomm/module-07-inference-engine-architecture.md"
      breadcrumbs={[
        { label: 'Qualcomm', href: '/qualcomm' },
        { label: 'Module 7' },
      ]}
      moduleNumber={7}
      title="Inference Engine Architecture"
      track="qualcomm"
      part={0}
      readingTime="5 hours"
      prerequisites={['Module 6', 'Graph IR concepts', 'Compiler design basics']}
      description="Building production inference engines on Hexagon: graph IR, operator fusion, memory planning, scheduling, and runtime architecture for end-to-end model execution."
      prev={{ href: '/qualcomm/module-06', label: 'Operator Kernel Design' }}
      next={{ href: '/qualcomm/module-08', label: 'Performance Analysis & Profiling' }}
    />
  )
}
