import { MarkdownModulePage } from '@/components/content/MarkdownModulePage'

export const metadata = { title: 'Chapter 14: Alpha Combination & Signal Processing' }

export default function Chapter14Page() {
  return (
    <MarkdownModulePage
      sourcePath="From_Zero_to_Quant/Chapter_14_Alpha_Combination_Signal_Processing.md"
      breadcrumbs={[
        { label: 'Quant', href: '/quant' },
        { label: 'Chapter 14' },
      ]}
      moduleNumber={14}
      title="Alpha Combination & Signal Processing"
      track="quant"
      part={0}
      readingTime="45 min"
      description="Combining alphas, signal blending, decay modeling"
      prev={{ href: '/quant/chapter-13', label: 'Alternative Data & NLP Signals' }}
      next={{ href: '/quant/chapter-15', label: 'Why Financial ML is Different' }}
    />
  )
}
