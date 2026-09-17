/**
 * Where the reader is in each module, kept in localStorage. Read by the module
 * page (resume pill, read marks), the home page (continue reading) and the
 * syllabus (read state). Client-only; every call is safe when storage throws.
 */
export interface ReadingPosition {
  href: string
  title: string
  track: string
  /** 0–100, how far down the page */
  pct: number
  /** id and title of the h2 the reader was in */
  section?: { id: string; title: string }
  /** reached the end of the module at least once */
  done: boolean
  /** last update, ms since epoch */
  ts: number
}

const KEY = 'aman.study:positions'
const MAX = 60
const EVENT = 'aman.study:positions-changed'

export function readPositions(): Record<string, ReadingPosition> {
  try {
    const raw = localStorage.getItem(KEY)
    return raw ? (JSON.parse(raw) as Record<string, ReadingPosition>) : {}
  } catch {
    return {}
  }
}

export function getPosition(href: string): ReadingPosition | undefined {
  return readPositions()[href]
}

export function savePosition(next: Omit<ReadingPosition, 'ts' | 'done'> & { done?: boolean }) {
  const all = readPositions()
  const prev = all[next.href]
  all[next.href] = { ...next, done: next.done ?? prev?.done ?? false, ts: Date.now() }
  const entries = Object.values(all).sort((a, b) => b.ts - a.ts).slice(0, MAX)
  try {
    localStorage.setItem(KEY, JSON.stringify(Object.fromEntries(entries.map((p) => [p.href, p]))))
    window.dispatchEvent(new Event(EVENT))
  } catch {}
}

export function recentPositions(limit = 3): ReadingPosition[] {
  return Object.values(readPositions()).sort((a, b) => b.ts - a.ts).slice(0, limit)
}

/** Modules of a track marked done, and the one most recently open. */
export function trackProgress(hrefPrefix: string) {
  const list = Object.values(readPositions()).filter((p) => p.href.startsWith(hrefPrefix + '/'))
  return { done: list.filter((p) => p.done).length, latest: list.sort((a, b) => b.ts - a.ts)[0] }
}

export function onPositionsChange(cb: () => void) {
  window.addEventListener(EVENT, cb)
  window.addEventListener('storage', cb)
  return () => {
    window.removeEventListener(EVENT, cb)
    window.removeEventListener('storage', cb)
  }
}

export function relativeTime(ts: number) {
  const s = Math.round((Date.now() - ts) / 1000)
  if (s < 60) return 'just now'
  const m = Math.round(s / 60)
  if (m < 60) return `${m} min ago`
  const h = Math.round(m / 60)
  if (h < 24) return `${h} h ago`
  const d = Math.round(h / 24)
  if (d === 1) return 'yesterday'
  if (d < 30) return `${d} days ago`
  return new Date(ts).toLocaleDateString(undefined, { month: 'short', day: 'numeric' })
}
