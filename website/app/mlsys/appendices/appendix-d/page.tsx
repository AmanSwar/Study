import { Breadcrumbs } from '@/components/layout/Breadcrumbs'
import { MarkdownRenderer } from '@/components/content/MarkdownRenderer'
import { TableOfContents } from '@/components/content/TableOfContents'
import { loadMarkdown, extractHeadings, stripFirstH1 } from '@/lib/markdown'

export const metadata = { title: 'Appendix D: Key Paper Reading List' }

export default async function AppendixPage() {
  const raw = await loadMarkdown('MLsys/appendices/appendix_d_paper_reading_list.md')
  const content = stripFirstH1(raw)
  const tocItems = extractHeadings(content).filter(h => h.level === 2)

  return (
    <>
      <Breadcrumbs items={[
        { label: 'MLsys', href: '/mlsys' },
        { label: 'Appendices', href: '/mlsys/appendices' },
        { label: 'Appendix D' },
      ]} />
      <div className="flex gap-0">
        <article className="flex-1 min-w-0">
          <div className="mb-10 pb-8 border-b border-border-primary">
            <div className="text-xs text-text-tertiary uppercase tracking-wider mb-2">Appendix D</div>
            <h1 className="text-3xl sm:text-4xl font-extrabold tracking-tight text-text-primary">Key Paper Reading List</h1>
          </div>
          <MarkdownRenderer content={content} />
        </article>
        <TableOfContents items={tocItems} />
      </div>
    </>
  )
}
