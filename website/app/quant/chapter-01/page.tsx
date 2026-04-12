import { MarkdownModulePage } from '@/components/content/MarkdownModulePage'

export const metadata = { title: 'Chapter 1: Financial Markets Fundamentals' }

export default function Chapter01Page() {
  return (
    <MarkdownModulePage
      sourcePath="From_Zero_to_Quant/Chapter_01_Financial_Markets_Fundamentals.md"
      breadcrumbs={[
        { label: 'Quant', href: '/quant' },
        { label: 'Chapter 1' },
      ]}
      moduleNumber={1}
      title="Financial Markets Fundamentals"
      track="quant"
      part={0}
      readingTime="60 min"
      description="How markets price risk, instruments, derivatives, Indian ecosystem"

      next={{ href: '/quant/chapter-02', label: 'Market Microstructure' }}
    />
  )
}
