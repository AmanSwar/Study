import { MarkdownModulePage } from '@/components/content/MarkdownModulePage'

export const metadata = { title: 'Chapter 8: Optimization & Portfolio Management' }

export default function Chapter08Page() {
  return (
    <MarkdownModulePage
      sourcePath="From_Zero_to_Quant/Chapter_08_Optimization_Portfolio_Management.md"
      breadcrumbs={[
        { label: 'Quant', href: '/quant' },
        { label: 'Chapter 8' },
      ]}
      moduleNumber={8}
      title="Optimization & Portfolio Management"
      track="quant"
      part={0}
      readingTime="50 min"
      description="Mean-variance, Black-Litterman, risk parity, convex optimization"
      prev={{ href: '/quant/chapter-07', label: 'Regression & Predictive Modeling' }}
      next={{ href: '/quant/chapter-09', label: 'Financial Data Engineering' }}
    />
  )
}
