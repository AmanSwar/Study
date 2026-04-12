import { MarkdownModulePage } from '@/components/content/MarkdownModulePage'

export const metadata = { title: 'Chapter 3: Returns, Risk & Performance' }

export default function Chapter03Page() {
  return (
    <MarkdownModulePage
      sourcePath="From_Zero_to_Quant/Chapter_03_Returns_Risk_Performance.md"
      breadcrumbs={[
        { label: 'Quant', href: '/quant' },
        { label: 'Chapter 3' },
      ]}
      moduleNumber={3}
      title="Returns, Risk & Performance"
      track="quant"
      part={0}
      readingTime="45 min"
      description="Return calculations, risk metrics, Sharpe, Sortino, drawdown"
      prev={{ href: '/quant/chapter-02', label: 'Market Microstructure' }}
      next={{ href: '/quant/chapter-04', label: 'Market Efficiency and Edge' }}
    />
  )
}
