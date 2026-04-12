import { MarkdownModulePage } from '@/components/content/MarkdownModulePage'

export const metadata = { title: 'Module 8: Performance Analysis & Profiling' }

export default function Module08Page() {
  return (
    <MarkdownModulePage
      sourcePath="qualcomm/module-08-performance-analysis.md"
      breadcrumbs={[
        { label: 'Qualcomm', href: '/qualcomm' },
        { label: 'Module 8' },
      ]}
      moduleNumber={8}
      title="Performance Analysis & Profiling"
      track="qualcomm"
      part={0}
      readingTime="4 hours"
      prerequisites={['Modules 1-7', 'Roofline model basics']}
      description="Performance methodology for Hexagon: roofline analysis, bottleneck identification, profiling tools (PMU, Hexagon Profiler), and systematic optimization workflow."
      prev={{ href: '/qualcomm/module-07', label: 'Inference Engine Architecture' }}
      next={{ href: '/qualcomm/module-09', label: 'Toolchain & Development' }}
    />
  )
}
