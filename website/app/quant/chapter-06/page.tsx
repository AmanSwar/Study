import { MarkdownModulePage } from '@/components/content/MarkdownModulePage'

export const metadata = { title: 'Chapter 6: Time Series Analysis' }

export default function Chapter06Page() {
  return (
    <MarkdownModulePage
      sourcePath="From_Zero_to_Quant/Chapter_06_Time_Series_Analysis.md"
      breadcrumbs={[
        { label: 'Quant', href: '/quant' },
        { label: 'Chapter 6' },
      ]}
      moduleNumber={6}
      title="Time Series Analysis"
      track="quant"
      part={0}
      readingTime="50 min"
      description="Stationarity, ARIMA, GARCH, cointegration, regime detection"
      prev={{ href: '/quant/chapter-05', label: 'Probability & Statistics for Finance' }}
      next={{ href: '/quant/chapter-07', label: 'Regression & Predictive Modeling' }}
    />
  )
}
