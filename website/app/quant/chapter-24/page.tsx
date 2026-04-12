import { MarkdownModulePage } from '@/components/content/MarkdownModulePage'

export const metadata = { title: 'Chapter 24: Trading System Architecture' }

export default function Chapter24Page() {
  return (
    <MarkdownModulePage
      sourcePath="From_Zero_to_Quant/Chapter_24_Trading_System_Architecture.md"
      breadcrumbs={[
        { label: 'Quant', href: '/quant' },
        { label: 'Chapter 24' },
      ]}
      moduleNumber={24}
      title="Trading System Architecture"
      track="quant"
      part={0}
      readingTime="50 min"
      description="Production systems, message queues, fault tolerance, monitoring"
      prev={{ href: '/quant/chapter-23', label: 'Transaction Cost Analysis' }}
      next={{ href: '/quant/chapter-25', label: 'Live Execution with Zerodha' }}
    />
  )
}
