import { TableOfContents } from '@/components/content/TableOfContents'
import { ModuleNav, type ModuleLink } from '@/components/layout/ModuleNav'
import { ReadingProgress } from '@/components/layout/ReadingProgress'
import { ReadingPosition } from '@/components/layout/ReadingPosition'
import { StudyRuntime } from '@/components/content/StudyRuntime'
import { loadHtmlModule } from '@/lib/html-module'

export interface ModulePageProps {
  absPath: string
  href: string
  moduleNumber: number
  title: string
  track: string
  trackTitle: string
  unitLabel: string
  part: number
  readingTime: string
  prerequisites?: string[]
  description?: string
  prev?: ModuleLink
  next?: ModuleLink
  /** position in the track's reading order */
  index: number
  count: number
}

/**
 * A self-contained HTML module inside the reading layout. The module carries its
 * own header (kicker, title, lede, meta); the site adds the progress hairline,
 * the on-this-page rail, position memory and prev/next. study.css is loaded
 * globally; study.js runs through StudyRuntime.
 */
export async function HtmlModulePage({ absPath, href, title, track, unitLabel, readingTime, prev, next, index, count }: ModulePageProps) {
  const { html, headings } = await loadHtmlModule(absPath)
  const sections = headings.filter((h) => h.level === 2)

  return (
    <>
      <ReadingProgress />
      <div className="reading-page">
        <div className="reading-article">
          <article>
            <StudyRuntime html={html} />
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
