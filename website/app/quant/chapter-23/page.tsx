import { MarkdownModulePage } from '@/components/content/MarkdownModulePage'

export const metadata = { title: 'Chapter 23: Transaction Cost Analysis' }

export default function Chapter23Page() {
  return (
    <MarkdownModulePage
      sourcePath="From_Zero_to_Quant/Chapter_23_Transaction_Cost_Analysis.md"
      breadcrumbs={[
        { label: 'Quant', href: '/quant' },
        { label: 'Chapter 23' },
      ]}
      moduleNumber={23}
      title="Transaction Cost Analysis"
      track="quant"
      part={0}
      readingTime="40 min"
      description="Spread costs, market impact, VWAP/TWAP, implementation shortfall"
      prev={{ href: '/quant/chapter-22', label: 'Risk Management' }}
      next={{ href: '/quant/chapter-24', label: 'Trading System Architecture' }}
    />
  )
}
