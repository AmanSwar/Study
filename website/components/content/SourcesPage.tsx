import { MarkdownRenderer } from '@/components/content/MarkdownRenderer'
import { TableOfContents } from '@/components/content/TableOfContents'
import { ReadingProgress } from '@/components/layout/ReadingProgress'
import { loadMarkdown, extractHeadings, stripFirstH1 } from '@/lib/markdown'
import type { Track } from '@/lib/registry-types'

/** Renders a track's sources.md (the research stage's source map). */
export async function SourcesPage({ track }: { track: Track }) {
  const raw = await loadMarkdown(track.sourcesAbsPath!)
  const content = stripFirstH1(raw)
  const tocItems = extractHeadings(content).filter((h) => h.level === 2)
  return (
    <>
      <ReadingProgress />
      <div className="reading-page">
        <div className="reading-article">
          <article className="study markdown reading-narrow">
            <header className="study-header">
              <p className="kicker">{track.shortTitle} · Source map</p>
              <h1>Sources</h1>
              <p className="lede">Every primary source the material was researched from, with the facts extracted from each and where sources disagree.</p>
            </header>
            <MarkdownRenderer content={content} />
          </article>
        </div>
        <TableOfContents items={tocItems} />
      </div>
    </>
  )
}
