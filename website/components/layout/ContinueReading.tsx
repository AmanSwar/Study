'use client'

import Link from 'next/link'
import { useEffect, useState } from 'react'
import { onPositionsChange, recentPositions, relativeTime, type ReadingPosition } from '@/lib/reading-state'

/**
 * The modules most recently open on this device, with how far the reader got.
 * Renders nothing until mounted, and nothing at all for a first visit.
 */
export function ContinueReading({ trackLabels }: { trackLabels: Record<string, string> }) {
  const [items, setItems] = useState<ReadingPosition[]>([])

  useEffect(() => {
    const read = () => setItems(recentPositions(3).filter((p) => p.pct >= 3))
    read()
    return onPositionsChange(read)
  }, [])

  if (items.length === 0) return null

  return (
    <section className="mb-4">
      <h2 className="ui font-sans text-[11px] font-semibold uppercase tracking-[0.1em] text-text-tertiary pb-2 mb-2 border-b border-border-primary">Continue reading</h2>
      <div className="grid gap-2">
        {items.map((p) => (
          <Link
            key={p.href}
            href={p.done ? p.href : p.section ? `${p.href}#${p.section.id}` : p.href}
            className="group grid grid-cols-[1fr_auto] gap-x-6 gap-y-0.5 items-center px-4 py-3 rounded-lg border border-border-primary bg-bg-surface hover:border-border-secondary transition-colors"
          >
            <span className="font-serif text-[1.05rem] font-semibold tracking-tight leading-snug text-text-primary group-hover:text-accent-blue transition-colors">{p.title}</span>
            <span className="ui row-span-2 inline-flex items-center gap-2.5 font-sans text-[12px] text-text-tertiary whitespace-nowrap">
              {p.done ? 'Read' : `${p.pct}%`}
              <span className="relative inline-block w-20 h-[3px] rounded-full bg-border-primary overflow-hidden" aria-hidden="true">
                <span className={`absolute inset-y-0 left-0 ${p.done ? 'bg-accent-green' : 'bg-accent-blue'}`} style={{ width: `${p.done ? 100 : p.pct}%` }} />
              </span>
            </span>
            <span className="ui font-sans text-[12px] text-text-tertiary truncate">
              {trackLabels[p.track] ?? p.track}
              {p.section && !p.done && <> · § {p.section.title}</>}
              {' · '}{relativeTime(p.ts)}
            </span>
          </Link>
        ))}
      </div>
    </section>
  )
}
