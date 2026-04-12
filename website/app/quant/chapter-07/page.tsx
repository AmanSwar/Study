import { MarkdownModulePage } from '@/components/content/MarkdownModulePage'

export const metadata = { title: 'Chapter 7: Regression & Predictive Modeling' }

export default function Chapter07Page() {
  return (
    <MarkdownModulePage
      sourcePath="From_Zero_to_Quant/Chapter_07_Regression_Predictive_Modeling.md"
      breadcrumbs={[
        { label: 'Quant', href: '/quant' },
        { label: 'Chapter 7' },
      ]}
      moduleNumber={7}
      title="Regression & Predictive Modeling"
      track="quant"
      part={0}
      readingTime="45 min"
      description="Linear/ridge/lasso regression, panel data, Fama-MacBeth"
      prev={{ href: '/quant/chapter-06', label: 'Time Series Analysis' }}
      next={{ href: '/quant/chapter-08', label: 'Optimization & Portfolio Management' }}
    />
  )
}
