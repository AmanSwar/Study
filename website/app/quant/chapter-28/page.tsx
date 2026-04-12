import { MarkdownModulePage } from '@/components/content/MarkdownModulePage'

export const metadata = { title: 'Chapter 28: Continuous Improvement' }

export default function Chapter28Page() {
  return (
    <MarkdownModulePage
      sourcePath="From_Zero_to_Quant/Chapter_28_Continuous_Improvement.md"
      breadcrumbs={[
        { label: 'Quant', href: '/quant' },
        { label: 'Chapter 28' },
      ]}
      moduleNumber={28}
      title="Continuous Improvement"
      track="quant"
      part={0}
      readingTime="40 min"
      description="Alpha decay, strategy rotation, research pipeline, scaling up"
      prev={{ href: '/quant/chapter-27', label: 'Live Performance Analysis' }}

    />
  )
}
