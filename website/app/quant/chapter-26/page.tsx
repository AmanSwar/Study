import { MarkdownModulePage } from '@/components/content/MarkdownModulePage'

export const metadata = { title: 'Chapter 26: Going Live — First 30 Days' }

export default function Chapter26Page() {
  return (
    <MarkdownModulePage
      sourcePath="From_Zero_to_Quant/Chapter_26_Going_Live_First_30_Days.md"
      breadcrumbs={[
        { label: 'Quant', href: '/quant' },
        { label: 'Chapter 26' },
      ]}
      moduleNumber={26}
      title="Going Live — First 30 Days"
      track="quant"
      part={0}
      readingTime="55 min"
      description="Paper trading, gradual deployment, monitoring, common mistakes"
      prev={{ href: '/quant/chapter-25', label: 'Live Execution with Zerodha' }}
      next={{ href: '/quant/chapter-27', label: 'Live Performance Analysis' }}
    />
  )
}
