import { MarkdownModulePage } from '@/components/content/MarkdownModulePage'

export const metadata = { title: 'Chapter 25: Live Execution with Zerodha' }

export default function Chapter25Page() {
  return (
    <MarkdownModulePage
      sourcePath="From_Zero_to_Quant/Chapter_25_Live_Execution_Zerodha.md"
      breadcrumbs={[
        { label: 'Quant', href: '/quant' },
        { label: 'Chapter 25' },
      ]}
      moduleNumber={25}
      title="Live Execution with Zerodha"
      track="quant"
      part={0}
      readingTime="50 min"
      description="Kite Connect API, order management, position tracking"
      prev={{ href: '/quant/chapter-24', label: 'Trading System Architecture' }}
      next={{ href: '/quant/chapter-26', label: 'Going Live — First 30 Days' }}
    />
  )
}
