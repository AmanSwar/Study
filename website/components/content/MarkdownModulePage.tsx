import { ModuleHeader } from '@/components/content/ModuleHeader'
import { MarkdownRenderer } from '@/components/content/MarkdownRenderer'
import { TableOfContents } from '@/components/content/TableOfContents'
import { ModuleNav } from '@/components/layout/ModuleNav'
import { ReadingProgress } from '@/components/layout/ReadingProgress'
import { ReadingPosition } from '@/components/layout/ReadingPosition'
import type { ModulePageProps } from '@/components/content/HtmlModulePage'
import { loadMarkdown, extractHeadings, stripFirstH1 } from '@/lib/markdown'

/**
 * A markdown module in the reading layout: the site supplies the header (same
 * markup as HTML modules), the body renders through MarkdownRenderer onto the
 * study.css components, plus progress, on-this-page rail, position memory and
 * prev/next.
 */
export async function MarkdownModulePage({
  absPath, href, moduleNumber, title, track, trackTitle, unitLabel, part, readingTime, prerequisites = [], description, prev, next, index, count,
}: ModulePageProps) {
  const rawContent = await loadMarkdown(absPath)
  const content = stripFirstH1(rawContent)
  const sections = extractHeadings(content).filter((h) => h.level === 2)

  return (
    <>
      <ReadingProgress />
      <div className="reading-page">
        <div className="reading-article">
          <article className="study markdown reading-narrow">
            <ModuleHeader
              number={moduleNumber}
              title={title}
              trackTitle={trackTitle}
              unitLabel={unitLabel}
              part={part}
              readingTime={readingTime}
              prerequisites={prerequisites}
              description={description}
            />
            <MarkdownRenderer content={content} />
          </article>
          <ModuleNav prev={prev} next={next} unitLabel={unitLabel} />
        </div>
        <TableOfContents
          items={sections}
          meta={
            <>
              {readingTime && <><b className="font-medium text-text-secondary">{readingTime}</b> read<br /></>}
              {unitLabel} {index + 1} of {count}
            </>
          }
        />
      </div>
      <ReadingPosition href={href} title={title} track={track} sections={sections} />
    </>
  )
}
