import { MarkdownModulePage } from '@/components/content/MarkdownModulePage'

export const metadata = { title: 'Module 1: Execution Model' }

export default function Module01Page() {
  return (
    <MarkdownModulePage
      sourcePath="intel/module-01.md"
      breadcrumbs={[
        { label: 'Intel/AMD', href: '/intel' },
        { label: 'Part 1: Systems Foundations', href: '/intel/part-01-foundations' },
        { label: 'Module 1' },
      ]}
      moduleNumber={1}
      title="Execution Model"
      track="intel"
      part={1}
      readingTime="30 min"
      description="Process abstraction, virtual memory, context switching, system calls"


    />
  )
}
