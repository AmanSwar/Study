import { MarkdownModulePage } from '@/components/content/MarkdownModulePage'

export const metadata = { title: 'Module 27: End-to-End Performance Optimization Workflow' }

export default function Module27Page() {
  return (
    <MarkdownModulePage
      sourcePath="intel/module-27.md"
      breadcrumbs={[
        { label: 'Intel/AMD', href: '/intel' },
        { label: 'Part 7: ML Engines', href: '/intel/part-07-ml-engines' },
        { label: 'Module 27' },
      ]}
      moduleNumber={27}
      title="End-to-End Performance Optimization Workflow"
      track="intel"
      part={7}
      readingTime="30 min"
      description="Full optimization workflow, benchmarking, deployment checklist"
      prev={{ href: '/intel/part-07-ml-engines/module-26', label: 'Kernel Implementation' }}

    />
  )
}
