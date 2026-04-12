import { MarkdownModulePage } from '@/components/content/MarkdownModulePage'

export const metadata = { title: 'Chapter 12: Classic Alpha Factors' }

export default function Chapter12Page() {
  return (
    <MarkdownModulePage
      sourcePath="From_Zero_to_Quant/Chapter_12_Classic_Alpha_Factors.md"
      breadcrumbs={[
        { label: 'Quant', href: '/quant' },
        { label: 'Chapter 12' },
      ]}
      moduleNumber={12}
      title="Classic Alpha Factors"
      track="quant"
      part={0}
      readingTime="50 min"
      description="Value, momentum, quality, size, low-volatility factor strategies"
      prev={{ href: '/quant/chapter-11', label: 'Alpha Research Methodology' }}
      next={{ href: '/quant/chapter-13', label: 'Alternative Data & NLP Signals' }}
    />
  )
}
