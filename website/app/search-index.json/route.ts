import { getRegistry } from '@/lib/registry'
import { buildSearchIndex } from '@/lib/search-index'
import { loadHtmlModule, extractText } from '@/lib/html-module'

// Prerendered once at build; served as a static file. The client fetches it
// lazily the first time the search dialog opens.
export const dynamic = 'force-static'

export async function GET() {
  const { tracks } = await getRegistry()
  // HTML modules contribute their headings + opening text to the keyword field.
  const extra: Record<string, string> = {}
  for (const t of tracks) {
    for (const m of t.allModules) {
      if (m.format !== 'html') continue
      try {
        const { html, headings } = await loadHtmlModule(m.absPath)
        extra[`${t.id}-${m.id}`] = [headings.map((h) => h.title).join(' '), extractText(html, 1500)].join(' ')
      } catch { /* a module that fails to load simply has no extra keywords */ }
    }
  }
  return Response.json(buildSearchIndex(tracks, extra))
}
