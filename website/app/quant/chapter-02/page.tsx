import { MarkdownModulePage } from '@/components/content/MarkdownModulePage'

export const metadata = { title: 'Chapter 2: Market Microstructure' }

export default function Chapter02Page() {
  return (
    <MarkdownModulePage
      sourcePath="From_Zero_to_Quant/Chapter_02_Market_Microstructure.md"
      breadcrumbs={[
        { label: 'Quant', href: '/quant' },
        { label: 'Chapter 2' },
      ]}
      moduleNumber={2}
      title="Market Microstructure"
      track="quant"
      part={0}
      readingTime="60 min"
      description="Order books, bid-ask spreads, market makers, exchange matching"
      prev={{ href: '/quant/chapter-01', label: 'Financial Markets Fundamentals' }}
      next={{ href: '/quant/chapter-03', label: 'Returns, Risk & Performance' }}
    />
  )
}
