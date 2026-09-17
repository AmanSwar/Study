'use client'

import { useEffect, useState, type ReactNode } from 'react'
import { usePathname } from 'next/navigation'
import { X } from 'lucide-react'
import { TopNav, type Crumb } from '@/components/layout/TopNav'
import { TrackSidebar } from '@/components/layout/TrackSidebar'
import type { NavTrack } from '@/lib/registry-types'

/**
 * Track page chrome: the reading bar with breadcrumb, the course contents as a
 * drawer (Contents button, `c`), the page, the footer. Nothing else stays on
 * screen while reading. The footer is rendered by the server layout and passed
 * in so this component never needs the registry.
 */
export function TrackShell({ track, footer, children }: { track: NavTrack; footer: ReactNode; children: ReactNode }) {
  const pathname = usePathname()
  const [open, setOpen] = useState(false)

  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      const t = e.target as HTMLElement | null
      if (t && (t.tagName === 'INPUT' || t.tagName === 'TEXTAREA' || t.tagName === 'SELECT' || t.isContentEditable)) return
      if (e.metaKey || e.ctrlKey || e.altKey) return
      if (e.key === 'c') { e.preventDefault(); setOpen((o) => !o) }
      else if (e.key === 'Escape') setOpen(false)
    }
    window.addEventListener('keydown', handler)
    return () => window.removeEventListener('keydown', handler)
  }, [])

  useEffect(() => {
    document.body.style.overflow = open ? 'hidden' : ''
    return () => { document.body.style.overflow = '' }
  }, [open])

  return (
    <>
      <TopNav crumbs={crumbsFor(track, pathname)} onContents={() => setOpen((o) => !o)} contentsOpen={open} />

      {open && (
        <div className="fixed inset-0 z-50 no-print" role="dialog" aria-modal="true" aria-label="Course contents">
          <div className="absolute inset-0 bg-bg-primary/60 backdrop-blur-[2px]" onClick={() => setOpen(false)} />
          <aside
            className="absolute inset-y-0 left-0 w-[22rem] max-w-[90vw] overflow-y-auto bg-bg-surface border-r border-border-primary shadow-lg animate-fade-in"
            onClick={(e) => { if ((e.target as HTMLElement).closest('a')) setOpen(false) }}
          >
            <button
              type="button"
              onClick={() => setOpen(false)}
              aria-label="Close contents"
              className="absolute top-3 right-3 w-8 h-8 inline-flex items-center justify-center rounded-md text-text-tertiary hover:text-text-primary hover:bg-bg-surface-hover"
            >
              <X className="w-4 h-4" />
            </button>
            <TrackSidebar track={track} />
          </aside>
        </div>
      )}

      <main className="flex-1 min-w-0 pt-12">
        {children}
      </main>
      {footer}
    </>
  )
}

function crumbsFor(track: NavTrack, pathname: string): Crumb[] {
  const root: Crumb = { label: track.shortTitle, href: track.href }
  const rest = pathname.replace(/\/+$/, '').split('/').filter(Boolean).slice(1)
  if (rest.length === 0) return [{ label: track.shortTitle }]
  const [a, b] = rest
  if (a === 'appendices') {
    if (!b) return [root, { label: 'Appendices' }]
    const ap = track.appendices.find((x) => x.id === b)
    return [root, { label: 'Appendices', href: `${track.href}/appendices` }, { label: ap ? `Appendix ${ap.letter}` : b }]
  }
  if (a === 'sources') return [root, { label: 'Sources' }]
  const part = track.parts?.find((p) => p.id === a)
  if (part) {
    const mod = b ? part.modules.find((m) => m.id === b) : undefined
    return mod
      ? [root, { label: `Part ${part.number}`, href: part.href }, { label: `${track.unitLabel} ${mod.number}` }]
      : [root, { label: `Part ${part.number}` }]
  }
  const mod = track.modules?.find((m) => m.id === a)
  return mod ? [root, { label: `${track.unitLabel} ${mod.number}` }] : [root]
}
