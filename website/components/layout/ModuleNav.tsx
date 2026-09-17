'use client'

import Link from 'next/link'
import { useRouter } from 'next/navigation'
import { useEffect } from 'react'

export interface ModuleLink { href: string; label: string; number?: number }

interface ModuleNavProps {
  prev?: ModuleLink
  next?: ModuleLink
  unitLabel?: string
}

/**
 * Previous / next at the end of a module, as plain typographic links with the
 * full title. Also binds ← and → (skipped while typing or with modifiers held).
 */
export function ModuleNav({ prev, next, unitLabel = 'Module' }: ModuleNavProps) {
  const router = useRouter()

  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      const target = e.target as HTMLElement | null
      if (!target) return
      const tag = target.tagName
      if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT' || target.isContentEditable) return
      if (e.metaKey || e.ctrlKey || e.altKey) return
      // steppers own the arrow keys while focused
      if (target.closest('.stepper')) return
      if (e.key === 'ArrowLeft' && prev) { e.preventDefault(); router.push(prev.href) }
      else if (e.key === 'ArrowRight' && next) { e.preventDefault(); router.push(next.href) }
    }
    window.addEventListener('keydown', handler)
    return () => window.removeEventListener('keydown', handler)
  }, [prev, next, router])

  if (!prev && !next) return null

  const item = (m: ModuleLink | undefined, dir: 'prev' | 'next') =>
    m ? (
      <Link href={m.href} className={`group block min-w-0 ${dir === 'next' ? 'sm:text-right' : ''}`}>
        <span className="block font-sans text-[11px] font-medium uppercase tracking-[0.08em] text-text-tertiary mb-1.5">
          {dir === 'prev' ? <><kbd className="mr-1.5">←</kbd>Previous{m.number !== undefined && ` · ${unitLabel} ${m.number}`}</> : <>{m.number !== undefined && `${unitLabel} ${m.number} · `}Next<kbd className="ml-1.5">→</kbd></>}
        </span>
        <span className="block font-serif text-[1.05rem] font-semibold leading-snug tracking-tight text-text-primary group-hover:text-accent-blue transition-colors text-balance">
          {m.label}
        </span>
      </Link>
    ) : (
      <div />
    )

  return (
    <nav className="ui no-print reading-narrow mt-16 pt-6 border-t border-border-primary grid gap-6 sm:grid-cols-2" aria-label="Module navigation">
      {item(prev, 'prev')}
      {item(next, 'next')}
    </nav>
  )
}
