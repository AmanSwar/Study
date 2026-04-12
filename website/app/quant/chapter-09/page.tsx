import { MarkdownModulePage } from '@/components/content/MarkdownModulePage'

export const metadata = { title: 'Chapter 9: Financial Data Engineering' }

export default function Chapter09Page() {
  return (
    <MarkdownModulePage
      sourcePath="From_Zero_to_Quant/Chapter_09_Financial_Data_Engineering.md"
      breadcrumbs={[
        { label: 'Quant', href: '/quant' },
        { label: 'Chapter 9' },
      ]}
      moduleNumber={9}
      title="Financial Data Engineering"
      track="quant"
      part={0}
      readingTime="50 min"
      description="Data sources, cleaning, corporate actions, survivorship bias"
      prev={{ href: '/quant/chapter-08', label: 'Optimization & Portfolio Management' }}
      next={{ href: '/quant/chapter-10', label: 'Feature Engineering for Finance' }}
    />
  )
}
