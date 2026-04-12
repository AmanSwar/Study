import { MarkdownModulePage } from '@/components/content/MarkdownModulePage'

export const metadata = { title: 'Chapter 19: Building a Realistic Backtester' }

export default function Chapter19Page() {
  return (
    <MarkdownModulePage
      sourcePath="From_Zero_to_Quant/Chapter_19_Building_Realistic_Backtester.md"
      breadcrumbs={[
        { label: 'Quant', href: '/quant' },
        { label: 'Chapter 19' },
      ]}
      moduleNumber={19}
      title="Building a Realistic Backtester"
      track="quant"
      part={0}
      readingTime="50 min"
      description="Event-driven backtesting, realistic fills, slippage, market impact"
      prev={{ href: '/quant/chapter-18', label: 'Advanced ML Topics' }}
      next={{ href: '/quant/chapter-20', label: 'Backtest Evaluation & Validation' }}
    />
  )
}
