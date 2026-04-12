import { MarkdownModulePage } from '@/components/content/MarkdownModulePage'

export const metadata = { title: 'Chapter 18: Advanced ML Topics' }

export default function Chapter18Page() {
  return (
    <MarkdownModulePage
      sourcePath="From_Zero_to_Quant/Chapter_18_Advanced_ML_Topics.md"
      breadcrumbs={[
        { label: 'Quant', href: '/quant' },
        { label: 'Chapter 18' },
      ]}
      moduleNumber={18}
      title="Advanced ML Topics"
      track="quant"
      part={0}
      readingTime="55 min"
      description="RL for trading, GANs for synthetic data, transformers for sequences"
      prev={{ href: '/quant/chapter-17', label: 'Models for Return Prediction' }}
      next={{ href: '/quant/chapter-19', label: 'Building a Realistic Backtester' }}
    />
  )
}
