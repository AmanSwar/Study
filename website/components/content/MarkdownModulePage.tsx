import { Breadcrumbs, BreadcrumbItem } from '@/components/layout/Breadcrumbs'
import { ModuleHeader } from '@/components/content/ModuleHeader'
import { MarkdownRenderer } from '@/components/content/MarkdownRenderer'
import { TableOfContents } from '@/components/content/TableOfContents'
import { ModuleNav } from '@/components/layout/ModuleNav'
import { ReadingProgress } from '@/components/layout/ReadingProgress'
import { loadMarkdown, extractHeadings, stripFirstH1 } from '@/lib/markdown'

interface MarkdownModulePageProps {
  /** Path to source markdown, relative to "computer science/" directory */
  sourcePath: string
  /** Breadcrumb items leading to this module */
  breadcrumbs: BreadcrumbItem[]
  /** Module number for the header */
  moduleNumber: number
  /** Module title */
  title: string
  /** Track ID (mlsys, intel, qualcomm) */
  track: string
  /** Part number (0 for flat tracks like Qualcomm) */
  part: number
  /** Estimated reading time */
  readingTime: string
  /** Prerequisites array */
  prerequisites?: string[]
  /** Optional description */
  description?: string
  /** Previous module link */
  prev?: { href: string; label: string }
  /** Next module link */
  next?: { href: string; label: string }
}

/**
 * MarkdownModulePage — a server component that loads a source markdown file
 * and renders it inside the standard module page layout:
 *
 *   [ReadingProgress bar at top]
 *   [Breadcrumbs]
 *   [ModuleHeader with number, title, description, metadata]
 *   [MarkdownRenderer — full content]
 *   [TableOfContents — right rail on wide screens]
 *   [ModuleNav — rich prev/next cards with arrow key shortcuts]
 */
export async function MarkdownModulePage({
  sourcePath,
  breadcrumbs,
  moduleNumber,
  title,
  track,
  part,
  readingTime,
  prerequisites = [],
  description,
  prev,
  next,
}: MarkdownModulePageProps) {
  const rawContent = await loadMarkdown(sourcePath)
  const content = stripFirstH1(rawContent)
  const tocItems = extractHeadings(content).filter((h) => h.level === 2)

  return (
    <>
      <ReadingProgress />
      <Breadcrumbs items={breadcrumbs} />

      <div className="flex gap-0">
        <article className="flex-1 min-w-0">
          <ModuleHeader
            number={moduleNumber}
            title={title}
            track={track}
            part={part}
            readingTime={readingTime}
            prerequisites={prerequisites}
            description={description}
          />

          <MarkdownRenderer content={content} />

          <ModuleNav prev={prev} next={next} />
        </article>

        <TableOfContents items={tocItems} />
      </div>
    </>
  )
}
