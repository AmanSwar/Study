import { MarkdownModulePage } from '@/components/content/MarkdownModulePage'

export const metadata = { title: 'Chapter 15: Why Financial ML is Different' }

export default function Chapter15Page() {
  return (
    <MarkdownModulePage
      sourcePath="From_Zero_to_Quant/Chapter_15_Why_Financial_ML_Different.md"
      breadcrumbs={[
        { label: 'Quant', href: '/quant' },
        { label: 'Chapter 15' },
      ]}
      moduleNumber={15}
      title="Why Financial ML is Different"
      track="quant"
      part={0}
      readingTime="40 min"
      description="Non-stationarity, low SNR, regime changes, overfitting"
      prev={{ href: '/quant/chapter-14', label: 'Alpha Combination & Signal Processing' }}
      next={{ href: '/quant/chapter-16', label: 'Cross-Validation & Model Selection' }}
    />
  )
}
