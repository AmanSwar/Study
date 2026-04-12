import { MarkdownModulePage } from '@/components/content/MarkdownModulePage'

export const metadata = { title: 'Chapter 4: Market Efficiency and Edge' }

export default function Chapter04Page() {
  return (
    <MarkdownModulePage
      sourcePath="From_Zero_to_Quant/Chapter_04_Market_Efficiency_and_Edge.md"
      breadcrumbs={[
        { label: 'Quant', href: '/quant' },
        { label: 'Chapter 4' },
      ]}
      moduleNumber={4}
      title="Market Efficiency and Edge"
      track="quant"
      part={0}
      readingTime="40 min"
      description="EMH, anomalies, behavioral finance, alpha sources"
      prev={{ href: '/quant/chapter-03', label: 'Returns, Risk & Performance' }}
      next={{ href: '/quant/chapter-05', label: 'Probability & Statistics for Finance' }}
    />
  )
}
