import { Breadcrumbs, BreadcrumbItem } from '@/components/layout/Breadcrumbs'
import { MarkdownRenderer } from '@/components/content/MarkdownRenderer'
import { TableOfContents } from '@/components/content/TableOfContents'
import { StudyRuntime } from '@/components/content/StudyRuntime'
import { AppendixCards } from '@/components/content/TrackIndex'
import { loadMarkdown, extractHeadings, stripFirstH1 } from '@/lib/markdown'
import { loadHtmlModule } from '@/lib/html-module'
import { colorClasses } from '@/lib/track-theme'
import type { Appendix, Track } from '@/lib/registry-types'

export function AppendixIndexPage({ track, breadcrumbs }: { track: Track; breadcrumbs: BreadcrumbItem[] }) {
  return (
    <>
      <Breadcrumbs items={breadcrumbs} />
      <h1 className="text-3xl font-extrabold tracking-tight text-text-primary mb-2">Appendices</h1>
      <p className="text-text-secondary mb-6">{track.title}</p>
      <AppendixCards appendices={track.appendices} theme={colorClasses(track.color)} title="" />
    </>
  )
}

export async function AppendixPage({ appendix, breadcrumbs }: { track: Track; appendix: Appendix; breadcrumbs: BreadcrumbItem[] }) {
  if (appendix.format === 'html') {
    const { html, headings } = await loadHtmlModule(appendix.absPath)
    return (
      <>
        <link rel="stylesheet" href="/study/study.css" precedence="study" />
        <Breadcrumbs items={breadcrumbs} />
        <div className="flex gap-0">
          <article className="flex-1 min-w-0">
            <StudyRuntime html={html} />
          </article>
          <TableOfContents items={headings.filter((h) => h.level === 2)} />
        </div>
      </>
    )
  }
  const raw = await loadMarkdown(appendix.absPath)
  const content = stripFirstH1(raw)
  const tocItems = extractHeadings(content).filter((h) => h.level === 2)
  return (
    <>
      <Breadcrumbs items={breadcrumbs} />
      <div className="flex gap-0">
        <article className="flex-1 min-w-0">
          <div className="mb-10 pb-8 border-b border-border-primary">
            <div className="text-xs text-text-tertiary uppercase tracking-wider mb-2">Appendix {appendix.letter}</div>
            <h1 className="text-3xl sm:text-4xl font-extrabold tracking-tight text-text-primary">{appendix.title}</h1>
          </div>
          <MarkdownRenderer content={content} />
        </article>
        <TableOfContents items={tocItems} />
      </div>
    </>
  )
}
