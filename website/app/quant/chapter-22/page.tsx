import { MarkdownModulePage } from '@/components/content/MarkdownModulePage'

export const metadata = { title: 'Chapter 22: Risk Management' }

export default function Chapter22Page() {
  return (
    <MarkdownModulePage
      sourcePath="From_Zero_to_Quant/Chapter_22_Risk_Management.md"
      breadcrumbs={[
        { label: 'Quant', href: '/quant' },
        { label: 'Chapter 22' },
      ]}
      moduleNumber={22}
      title="Risk Management"
      track="quant"
      part={0}
      readingTime="45 min"
      description="VaR, CVaR, stress testing, drawdown control, tail risk"
      prev={{ href: '/quant/chapter-21', label: 'Portfolio Construction' }}
      next={{ href: '/quant/chapter-23', label: 'Transaction Cost Analysis' }}
    />
  )
}
