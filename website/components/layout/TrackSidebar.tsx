'use client'

import Link from 'next/link'
import { usePathname } from 'next/navigation'
import { useEffect, useRef } from 'react'
import type { NavTrack, NavModule } from '@/lib/registry-types'
import { ReadState } from '@/components/content/ReadState'

/** Course contents, shown inside the drawer: parts, modules, reading time, read state. */
export function TrackSidebar({ track }: { track: NavTrack }) {
  const pathname = usePathname()
  const activeRef = useRef<HTMLAnchorElement | null>(null)

  useEffect(() => {
    activeRef.current?.scrollIntoView({ block: 'center', behavior: 'instant' })
  }, [])

  const row = (m: NavModule) => {
    const active = pathname === m.href
    return (
      <Link
        key={m.id}
        href={m.href}
        ref={active ? activeRef : null}
        aria-current={active ? 'page' : undefined}
        className={`flex items-baseline gap-2.5 px-2 py-1.5 rounded-md text-[13.5px] leading-snug transition-colors
          ${active ? 'bg-accent-blue-subtle text-accent-blue' : 'text-text-secondary hover:text-text-primary hover:bg-bg-surface-hover'}`}
      >
        <span className={`font-mono text-[11px] w-5 text-right shrink-0 ${active ? 'text-accent-blue' : 'text-text-tertiary'}`}>
          {String(m.number).padStart(2, '0')}
        </span>
        <span className="flex-1 min-w-0">{m.shortTitle}</span>
        {m.readingTime && <span className="text-[11px] text-text-tertiary shrink-0">{m.readingTime}</span>}
        <ReadState href={m.href} />
      </Link>
    )
  }

  return (
    <nav className="ui font-sans px-4 pt-4 pb-8" aria-label="Course contents">
      <Link href={track.href} className="block pr-8 mb-3 font-serif font-semibold text-[15px] leading-snug tracking-tight text-text-primary hover:text-accent-blue transition-colors">
        {track.title}
      </Link>

      {track.parts?.map((part) => (
        <section key={part.id} className="mt-3">
          <Link
            href={part.href}
            className={`block px-2 pb-1 text-[10.5px] font-semibold uppercase tracking-[0.08em] transition-colors
              ${pathname.startsWith(part.href) ? 'text-text-secondary' : 'text-text-tertiary hover:text-text-secondary'}`}
          >
            Part {part.number} · {part.shortTitle}
          </Link>
          <div className="space-y-px">{part.modules.map(row)}</div>
        </section>
      ))}

      {track.modules && <div className="space-y-px mt-1">{track.modules.map(row)}</div>}

      {(track.appendices.length > 0 || track.hasSources) && (
        <section className="mt-4 pt-3 border-t border-border-primary">
          <div className="px-2 pb-1 text-[10.5px] font-semibold uppercase tracking-[0.08em] text-text-tertiary">Appendices</div>
          <div className="space-y-px">
            {track.appendices.map((a) => (
              <Link
                key={a.id}
                href={a.href}
                aria-current={pathname === a.href ? 'page' : undefined}
                className={`flex items-baseline gap-2.5 px-2 py-1.5 rounded-md text-[13.5px] leading-snug transition-colors
                  ${pathname === a.href ? 'bg-accent-blue-subtle text-accent-blue' : 'text-text-secondary hover:text-text-primary hover:bg-bg-surface-hover'}`}
              >
                <span className="font-mono text-[11px] w-5 text-right shrink-0 text-text-tertiary">{a.letter}</span>
                <span className="flex-1 min-w-0">{a.title}</span>
              </Link>
            ))}
            {track.hasSources && (
              <Link
                href={`${track.href}/sources`}
                className={`flex items-baseline gap-2.5 px-2 py-1.5 rounded-md text-[13.5px] leading-snug transition-colors
                  ${pathname === `${track.href}/sources` ? 'bg-accent-blue-subtle text-accent-blue' : 'text-text-secondary hover:text-text-primary hover:bg-bg-surface-hover'}`}
              >
                <span className="font-mono text-[11px] w-5 text-right shrink-0 text-text-tertiary">§</span>
                <span className="flex-1 min-w-0">Sources</span>
              </Link>
            )}
          </div>
        </section>
      )}
    </nav>
  )
}
