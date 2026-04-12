import { MarkdownModulePage } from '@/components/content/MarkdownModulePage'

export const metadata = { title: 'Chapter 13: Alternative Data & NLP Signals' }

export default function Chapter13Page() {
  return (
    <MarkdownModulePage
      sourcePath="From_Zero_to_Quant/Chapter_13_Alternative_Data_NLP_Signals.md"
      breadcrumbs={[
        { label: 'Quant', href: '/quant' },
        { label: 'Chapter 13' },
      ]}
      moduleNumber={13}
      title="Alternative Data & NLP Signals"
      track="quant"
      part={0}
      readingTime="55 min"
      description="Sentiment analysis, news signals, web scraping, NLP for finance"
      prev={{ href: '/quant/chapter-12', label: 'Classic Alpha Factors' }}
      next={{ href: '/quant/chapter-14', label: 'Alpha Combination & Signal Processing' }}
    />
  )
}
