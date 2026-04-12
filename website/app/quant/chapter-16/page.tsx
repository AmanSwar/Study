import { MarkdownModulePage } from '@/components/content/MarkdownModulePage'

export const metadata = { title: 'Chapter 16: Cross-Validation & Model Selection' }

export default function Chapter16Page() {
  return (
    <MarkdownModulePage
      sourcePath="From_Zero_to_Quant/Chapter_16_Cross_Validation_Model_Selection.md"
      breadcrumbs={[
        { label: 'Quant', href: '/quant' },
        { label: 'Chapter 16' },
      ]}
      moduleNumber={16}
      title="Cross-Validation & Model Selection"
      track="quant"
      part={0}
      readingTime="45 min"
      description="Purged/embargo CV, walk-forward, model selection for finance"
      prev={{ href: '/quant/chapter-15', label: 'Why Financial ML is Different' }}
      next={{ href: '/quant/chapter-17', label: 'Models for Return Prediction' }}
    />
  )
}
