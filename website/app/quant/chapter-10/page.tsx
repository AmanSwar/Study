import { MarkdownModulePage } from '@/components/content/MarkdownModulePage'

export const metadata = { title: 'Chapter 10: Feature Engineering for Finance' }

export default function Chapter10Page() {
  return (
    <MarkdownModulePage
      sourcePath="From_Zero_to_Quant/Chapter_10_Feature_Engineering_Finance.md"
      breadcrumbs={[
        { label: 'Quant', href: '/quant' },
        { label: 'Chapter 10' },
      ]}
      moduleNumber={10}
      title="Feature Engineering for Finance"
      track="quant"
      part={0}
      readingTime="45 min"
      description="Technical indicators, fundamental features, label construction"
      prev={{ href: '/quant/chapter-09', label: 'Financial Data Engineering' }}
      next={{ href: '/quant/chapter-11', label: 'Alpha Research Methodology' }}
    />
  )
}
