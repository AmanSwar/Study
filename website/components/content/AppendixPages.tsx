import { MarkdownRenderer } from '@/components/content/MarkdownRenderer'
import { TableOfContents } from '@/components/content/TableOfContents'
import { StudyRuntime } from '@/components/content/StudyRuntime'
import { AppendixCards } from '@/components/content/TrackIndex'
import { ReadingProgress } from '@/components/layout/ReadingProgress'
import { loadMarkdown, extractHeadings, stripFirstH1 } from '@/lib/markdown'
import { loadHtmlModule } from '@/lib/html-module'
import type { Appendix, Track } from '@/lib/registry-types'

export function AppendixIndexPage({ track }: { track: Track }) {
  return (
    <div className="max-w-[46rem] mx-auto px-6 sm:px-8 pt-10 pb-24">
      <header className="mb-4">
        <p className="ui font-sans text-[12px] font-medium uppercase tracking-[0.06em] text-text-tertiary mb-3">{track.shortTitle}</p>
        <h1 className="font-serif text-[2.125rem] font-semibold tracking-[-0.02em] leading-[1.15] text-text-primary">Appendices</h1>
      </header>
      <AppendixCards appendices={track.appendices} title="" />
    </div>
  )
}

export async function AppendixPage({ track, appendix }: { track: Track; appendix: Appendix }) {
  if (appendix.format === 'html') {
    const { html, headings } = await loadHtmlModule(appendix.absPath)
    return (
      <>
        <ReadingProgress />
        <div className="reading-page">
          <div className="reading-article">
            <article><StudyRuntime html={html} /></article>
          </div>
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
      <ReadingProgress />
      <div className="reading-page">
        <div className="reading-article">
          <article className="study markdown reading-narrow">
            <header className="study-header">
              <p className="kicker">{track.shortTitle} · Appendix {appendix.letter}</p>
              <h1>{appendix.title}</h1>
            </header>
            <MarkdownRenderer content={content} />
          </article>
        </div>
        <TableOfContents items={tocItems} />
      </div>
    </>
  )
}
