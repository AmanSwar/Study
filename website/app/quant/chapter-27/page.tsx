import { MarkdownModulePage } from '@/components/content/MarkdownModulePage'

export const metadata = { title: 'Chapter 27: Live Performance Analysis' }

export default function Chapter27Page() {
  return (
    <MarkdownModulePage
      sourcePath="From_Zero_to_Quant/Chapter_27_Live_Performance_Analysis.md"
      breadcrumbs={[
        { label: 'Quant', href: '/quant' },
        { label: 'Chapter 27' },
      ]}
      moduleNumber={27}
      title="Live Performance Analysis"
      track="quant"
      part={0}
      readingTime="35 min"
      description="Attribution, slippage analysis, strategy health, when to stop"
      prev={{ href: '/quant/chapter-26', label: 'Going Live — First 30 Days' }}
      next={{ href: '/quant/chapter-28', label: 'Continuous Improvement' }}
    />
  )
}
