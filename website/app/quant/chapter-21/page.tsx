import { MarkdownModulePage } from '@/components/content/MarkdownModulePage'

export const metadata = { title: 'Chapter 21: Portfolio Construction' }

export default function Chapter21Page() {
  return (
    <MarkdownModulePage
      sourcePath="From_Zero_to_Quant/Chapter_21_Portfolio_Construction.md"
      breadcrumbs={[
        { label: 'Quant', href: '/quant' },
        { label: 'Chapter 21' },
      ]}
      moduleNumber={21}
      title="Portfolio Construction"
      track="quant"
      part={0}
      readingTime="45 min"
      description="Position sizing, risk budgeting, sector constraints, turnover"
      prev={{ href: '/quant/chapter-20', label: 'Backtest Evaluation & Validation' }}
      next={{ href: '/quant/chapter-22', label: 'Risk Management' }}
    />
  )
}
