import type { Track } from './registry-types'

export interface SearchItem {
  id: string
  title: string
  subtitle: string          // e.g. "MLsys · Part 1: Fundamentals · Module 3"
  href: string
  track: string             // track id
  trackLabel: string
  color: string
  keywords: string          // combined searchable text, lowercased
}

/**
 * Flatten every visible module across all tracks into search items. Runs on
 * the server (route handler) — `extraKeywords` lets HTML modules contribute
 * their headings/body text keyed by `${track.id}-${module.id}`.
 */
export function buildSearchIndex(tracks: Track[], extraKeywords: Record<string, string> = {}): SearchItem[] {
  const items: SearchItem[] = []
  for (const track of tracks) {
    for (const mod of track.allModules) {
      const part = mod.partId ? track.parts?.find((p) => p.id === mod.partId) : undefined
      const id = `${track.id}-${mod.id}`
      items.push({
        id,
        title: mod.title,
        subtitle: `${track.shortTitle} · ${part ? `Part ${part.number}: ${part.shortTitle} · ` : ''}${track.unitLabel} ${mod.number}`,
        href: mod.href,
        track: track.id,
        trackLabel: track.shortTitle,
        color: track.color,
        keywords: [mod.title, mod.shortTitle, mod.description, part?.title, part?.shortTitle, track.shortTitle, track.title, track.category, extraKeywords[id]]
          .filter(Boolean)
          .join(' ')
          .toLowerCase(),
      })
    }
  }
  return items
}

/**
 * Simple fuzzy search: splits query into tokens and returns items where
 * all tokens are substrings of the keywords field (case-insensitive).
 * Returns items ranked by: title prefix match > title contains > keyword match.
 */
export function searchItems(items: SearchItem[], query: string): SearchItem[] {
  const q = query.trim().toLowerCase()
  if (!q) return []

  const tokens = q.split(/\s+/).filter(Boolean)

  type Scored = { item: SearchItem; score: number }
  const scored: Scored[] = []

  for (const item of items) {
    const titleLower = item.title.toLowerCase()
    const keywords = item.keywords

    if (!tokens.every((t) => keywords.includes(t))) continue

    let score = 0
    if (titleLower.startsWith(q)) score += 1000
    else if (titleLower.includes(q)) score += 500
    for (const t of tokens) {
      if (titleLower.includes(t)) score += 100
    }
    score -= item.title.length * 0.1

    scored.push({ item, score })
  }

  scored.sort((a, b) => b.score - a.score)
  return scored.slice(0, 20).map((s) => s.item)
}
