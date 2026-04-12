import { MarkdownModulePage } from '@/components/content/MarkdownModulePage'

export const metadata = { title: 'Chapter 5: Probability & Statistics for Finance' }

export default function Chapter05Page() {
  return (
    <MarkdownModulePage
      sourcePath="From_Zero_to_Quant/Chapter_05_Probability_Statistics_Finance.md"
      breadcrumbs={[
        { label: 'Quant', href: '/quant' },
        { label: 'Chapter 5' },
      ]}
      moduleNumber={5}
      title="Probability & Statistics for Finance"
      track="quant"
      part={0}
      readingTime="60 min"
      description="Distributions, hypothesis testing, Bayesian methods, copulas"
      prev={{ href: '/quant/chapter-04', label: 'Market Efficiency and Edge' }}
      next={{ href: '/quant/chapter-06', label: 'Time Series Analysis' }}
    />
  )
}
