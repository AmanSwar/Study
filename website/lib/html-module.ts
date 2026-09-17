import 'server-only'
import { promises as fs } from 'fs'

/**
 * Loader for self-contained HTML study modules (written by the study skills).
 * The module file is a full standalone document; the site keeps only the inner
 * HTML of <main class="study"> and supplies its own layout, stylesheet and
 * runtime. Regex is sufficient because the skill's linter guarantees the shape
 * (exactly one <main class="study">, ids on every h2/h3).
 */
export interface HtmlModule {
  html: string
  headings: { id: string; title: string; level: number }[]
  /** first <h1> text, if any */
  title?: string
}

export async function loadHtmlModule(absPath: string): Promise<HtmlModule> {
  const raw = await fs.readFile(absPath, 'utf-8')
  const html = extractStudyMain(raw)
  return { html, headings: extractHeadings(html), title: decode((html.match(/<h1\b[^>]*>([\s\S]*?)<\/h1>/i)?.[1] ?? '').replace(/<[^>]+>/g, '')).trim() || undefined }
}

function extractStudyMain(doc: string): string {
  const open = doc.match(/<main\b[^>]*\bclass="[^"]*\bstudy\b[^"]*"[^>]*>/i)
  if (open && open.index !== undefined) {
    const start = open.index + open[0].length
    const end = doc.lastIndexOf('</main>')
    if (end > start) return stripHeadOnlyTags(doc.slice(start, end))
  }
  const body = doc.match(/<body\b[^>]*>([\s\S]*?)<\/body>/i)
  return stripHeadOnlyTags(body ? body[1] : doc)
}

// A standalone file may carry its stylesheet/runtime links inside the body; the site provides both.
const stripHeadOnlyTags = (s: string) =>
  s.replace(/<link\b[^>]*rel=["']stylesheet["'][^>]*>/gi, '').replace(/<script\b[^>]*\bsrc=["'][^"']*study\.js["'][^>]*>\s*<\/script>/gi, '')

export function extractHeadings(html: string): HtmlModule['headings'] {
  const out: HtmlModule['headings'] = []
  const re = /<(h2|h3)\b([^>]*)>([\s\S]*?)<\/\1>/gi
  let m: RegExpExecArray | null
  while ((m = re.exec(html))) {
    const id = m[2].match(/\bid="([^"]+)"/)?.[1]
    if (!id) continue
    out.push({ level: m[1].toLowerCase() === 'h2' ? 2 : 3, id, title: decode(m[3].replace(/<[^>]+>/g, '')).trim() })
  }
  return out
}

const decode = (s: string) =>
  s.replace(/&amp;/g, '&').replace(/&lt;/g, '<').replace(/&gt;/g, '>').replace(/&quot;/g, '"').replace(/&#39;/g, "'").replace(/&nbsp;/g, ' ')

/** Plain text of a module (for the search index keywords). */
export function extractText(html: string, limit = 4000): string {
  return decode(
    html
      .replace(/<section\b[^>]*id="references"[\s\S]*?<\/section>/i, ' ')
      .replace(/<svg[\s\S]*?<\/svg>/gi, ' ')
      .replace(/<script[\s\S]*?<\/script>/gi, ' ')
      .replace(/<[^>]+>/g, ' '),
  ).replace(/\s+/g, ' ').trim().slice(0, limit)
}
