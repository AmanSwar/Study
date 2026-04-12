import { MarkdownModulePage } from '@/components/content/MarkdownModulePage'

export const metadata = { title: 'Module 10: Research Frontiers & Advanced Topics' }

export default function Module10Page() {
  return (
    <MarkdownModulePage
      sourcePath="qualcomm/module-10-research-frontiers.md"
      breadcrumbs={[
        { label: 'Qualcomm', href: '/qualcomm' },
        { label: 'Module 10' },
      ]}
      moduleNumber={10}
      title="Research Frontiers & Advanced Topics"
      track="qualcomm"
      part={0}
      readingTime="4 hours"
      prerequisites={['All prior modules']}
      description="Cutting-edge research directions for Hexagon: sparsity exploitation, INT4 inference, LLM on mobile NPUs, neural architecture search for Hexagon, and future hardware trends."
      prev={{ href: '/qualcomm/module-09', label: 'Toolchain & Development' }}
    />
  )
}
