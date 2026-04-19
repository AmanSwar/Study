import { allTracks } from '@/content/tracks'

export interface SearchItem {
  id: string
  title: string
  subtitle: string          // e.g., "MLsys · Part 1" or "Quant · Chapter 5"
  href: string
  track: string             // 'mlsys' | 'intel' | 'qualcomm' | 'quant'
  keywords: string          // combined searchable text
}

/**
 * Build a flat list of all modules across all tracks for client-side search.
 * Computed once at build time since it's static.
 */
export function buildSearchIndex(): SearchItem[] {
  const items: SearchItem[] = []

  for (const track of allTracks) {
    // Flat tracks (Qualcomm, Quant) with direct modules
    if (track.modules) {
      for (const module of track.modules) {
        items.push({
          id: `${track.id}-${module.id}`,
          title: module.title,
          subtitle: `${track.shortTitle} · ${module.id.startsWith('chapter') ? `Chapter ${module.number}` : `Module ${module.number}`}`,
          href: module.href,
          track: track.id,
          keywords: [
            module.title,
            module.shortTitle,
            module.description,
            track.shortTitle,
            track.title,
          ].join(' ').toLowerCase(),
        })
      }
    }

    // Hierarchical tracks (MLsys, Intel) with parts → modules
    if (track.parts) {
      for (const part of track.parts) {
        for (const module of part.modules) {
          items.push({
            id: `${track.id}-${module.id}`,
            title: module.title,
            subtitle: `${track.shortTitle} · Part ${part.number}: ${part.shortTitle} · Module ${module.number}`,
            href: module.href,
            track: track.id,
            keywords: [
              module.title,
              module.shortTitle,
              module.description,
              part.title,
              part.shortTitle,
              track.shortTitle,
              track.title,
            ].join(' ').toLowerCase(),
          })
        }
      }
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

    // All tokens must match
    if (!tokens.every((t) => keywords.includes(t))) continue

    let score = 0
    // Title starts with query → highest
    if (titleLower.startsWith(q)) score += 1000
    // Title contains full query
    else if (titleLower.includes(q)) score += 500
    // Any token matches title
    for (const t of tokens) {
      if (titleLower.includes(t)) score += 100
    }
    // Shorter titles rank higher among equal-score items
    score -= item.title.length * 0.1

    scored.push({ item, score })
  }

  scored.sort((a, b) => b.score - a.score)
  return scored.slice(0, 20).map((s) => s.item)
}
