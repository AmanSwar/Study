import { Breadcrumbs, BreadcrumbItem } from '@/components/layout/Breadcrumbs'
import { TableOfContents } from '@/components/content/TableOfContents'
import { ModuleNav } from '@/components/layout/ModuleNav'
import { ReadingProgress } from '@/components/layout/ReadingProgress'
import { StudyRuntime } from '@/components/content/StudyRuntime'
import { loadHtmlModule } from '@/lib/html-module'

export interface ModulePageProps {
  absPath: string
  breadcrumbs: BreadcrumbItem[]
  moduleNumber: number
  title: string
  track: string
  part: number
  readingTime: string
  prerequisites?: string[]
  description?: string
  prev?: { href: string; label: string }
  next?: { href: string; label: string }
}

/**
 * HtmlModulePage — same chrome as MarkdownModulePage, but the body is a
 * self-contained HTML module (design system in /study/). The module's own
 * <head> is ignored; the site loads study.css here (React hoists and dedupes
 * the link) and study.js through StudyRuntime. The module carries its own
 * header (kicker, title, lede, meta chips), so the site's ModuleHeader is not
 * rendered — the manifest's title/readingTime still feed <title>, cards and
 * the sidebar.
 */
export async function HtmlModulePage({ absPath, breadcrumbs, prev, next }: ModulePageProps) {
  const { html, headings } = await loadHtmlModule(absPath)
  const tocItems = headings.filter((h) => h.level === 2)

  return (
    <>
      <link rel="stylesheet" href="/study/study.css" precedence="study" />
      <ReadingProgress />
      <Breadcrumbs items={breadcrumbs} />

      <div className="flex gap-0">
        <article className="flex-1 min-w-0">
          <StudyRuntime html={html} />
          <ModuleNav prev={prev} next={next} />
        </article>

        <TableOfContents items={tocItems} />
      </div>
    </>
  )
}
