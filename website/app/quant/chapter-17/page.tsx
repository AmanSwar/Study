import { MarkdownModulePage } from '@/components/content/MarkdownModulePage'

export const metadata = { title: 'Chapter 17: Models for Return Prediction' }

export default function Chapter17Page() {
  return (
    <MarkdownModulePage
      sourcePath="From_Zero_to_Quant/Chapter_17_Models_Return_Prediction.md"
      breadcrumbs={[
        { label: 'Quant', href: '/quant' },
        { label: 'Chapter 17' },
      ]}
      moduleNumber={17}
      title="Models for Return Prediction"
      track="quant"
      part={0}
      readingTime="50 min"
      description="Tree models, neural nets, ensemble methods, meta-labeling"
      prev={{ href: '/quant/chapter-16', label: 'Cross-Validation & Model Selection' }}
      next={{ href: '/quant/chapter-18', label: 'Advanced ML Topics' }}
    />
  )
}
