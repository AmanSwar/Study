import { MarkdownModulePage } from '@/components/content/MarkdownModulePage'

export const metadata = { title: 'Chapter 20: Backtest Evaluation & Validation' }

export default function Chapter20Page() {
  return (
    <MarkdownModulePage
      sourcePath="From_Zero_to_Quant/Chapter_20_Backtest_Evaluation_Validation.md"
      breadcrumbs={[
        { label: 'Quant', href: '/quant' },
        { label: 'Chapter 20' },
      ]}
      moduleNumber={20}
      title="Backtest Evaluation & Validation"
      track="quant"
      part={0}
      readingTime="45 min"
      description="Deflated Sharpe, Monte Carlo, walk-forward, detecting overfitting"
      prev={{ href: '/quant/chapter-19', label: 'Building a Realistic Backtester' }}
      next={{ href: '/quant/chapter-21', label: 'Portfolio Construction' }}
    />
  )
}
