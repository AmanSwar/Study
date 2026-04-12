import { MarkdownModulePage } from '@/components/content/MarkdownModulePage'

export const metadata = { title: 'Module 9: Toolchain & Development Environment' }

export default function Module09Page() {
  return (
    <MarkdownModulePage
      sourcePath="qualcomm/module-09-toolchain-development.md"
      breadcrumbs={[
        { label: 'Qualcomm', href: '/qualcomm' },
        { label: 'Module 9' },
      ]}
      moduleNumber={9}
      title="Toolchain & Development Environment"
      track="qualcomm"
      part={0}
      readingTime="5-6 hours"
      prerequisites={['Module 1', 'Build systems', 'Cross-compilation basics']}
      description="Hexagon SDK, cross-compilation, debugging, simulation, deployment workflows, and CI/CD integration for embedded ML inference."
      prev={{ href: '/qualcomm/module-08', label: 'Performance Analysis & Profiling' }}
      next={{ href: '/qualcomm/module-10', label: 'Research Frontiers' }}
    />
  )
}
