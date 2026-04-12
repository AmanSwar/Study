import { MarkdownModulePage } from '@/components/content/MarkdownModulePage'

export const metadata = { title: 'Chapter 11: Alpha Research Methodology' }

export default function Chapter11Page() {
  return (
    <MarkdownModulePage
      sourcePath="From_Zero_to_Quant/Chapter_11_Alpha_Research_Methodology.md"
      breadcrumbs={[
        { label: 'Quant', href: '/quant' },
        { label: 'Chapter 11' },
      ]}
      moduleNumber={11}
      title="Alpha Research Methodology"
      track="quant"
      part={0}
      readingTime="45 min"
      description="Systematic alpha discovery, hypothesis generation, signal testing"
      prev={{ href: '/quant/chapter-10', label: 'Feature Engineering for Finance' }}
      next={{ href: '/quant/chapter-12', label: 'Classic Alpha Factors' }}
    />
  )
}
