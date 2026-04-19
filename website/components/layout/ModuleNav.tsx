'use client'

import Link from 'next/link'
import { useRouter } from 'next/navigation'
import { useEffect } from 'react'
import { ArrowLeft, ArrowRight } from 'lucide-react'

interface ModuleNavProps {
  prev?: { href: string; label: string }
  next?: { href: string; label: string }
}

/**
 * Rich prev/next navigation at the bottom of each module page.
 * Clickable cards with arrows, labels, and directional hints.
 * Also binds ← and → arrow keys to navigate between modules (non-invasive —
 * skips if the user is in an input, textarea, or contentEditable).
 */
export function ModuleNav({ prev, next }: ModuleNavProps) {
  const router = useRouter()

  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      // Don't hijack when typing into inputs, textareas, or contentEditable elements
      const target = e.target as HTMLElement | null
      if (!target) return
      const tag = target.tagName
      if (
        tag === 'INPUT' ||
        tag === 'TEXTAREA' ||
        tag === 'SELECT' ||
        target.isContentEditable
      ) {
        return
      }

      // Don't hijack if modifiers are held (Cmd/Ctrl/Alt)
      if (e.metaKey || e.ctrlKey || e.altKey) return

      if (e.key === 'ArrowLeft' && prev) {
        e.preventDefault()
        router.push(prev.href)
      } else if (e.key === 'ArrowRight' && next) {
        e.preventDefault()
        router.push(next.href)
      }
    }

    window.addEventListener('keydown', handler)
    return () => window.removeEventListener('keydown', handler)
  }, [prev, next, router])

  if (!prev && !next) return null

  return (
    <nav
      className="mt-16 pt-8 border-t border-border-primary grid gap-4 sm:grid-cols-2 no-print"
      aria-label="Module navigation"
    >
      {prev ? (
        <Link
          href={prev.href}
          className="group relative flex flex-col gap-1 p-5 rounded-xl border border-border-primary bg-bg-surface hover:border-accent-blue/40 hover:bg-bg-surface-hover transition-all"
        >
          <div className="flex items-center gap-1.5 text-xs font-medium text-text-tertiary">
            <ArrowLeft className="w-3.5 h-3.5 transition-transform group-hover:-translate-x-0.5" />
            <span>Previous</span>
            <kbd className="ml-1 text-[10px] px-1 py-0.5 rounded border border-border-primary text-text-tertiary/70">
              ←
            </kbd>
          </div>
          <span className="text-sm font-semibold text-text-primary group-hover:text-accent-blue transition-colors line-clamp-2">
            {prev.label}
          </span>
        </Link>
      ) : (
        <div />
      )}

      {next ? (
        <Link
          href={next.href}
          className="group relative flex flex-col gap-1 p-5 rounded-xl border border-border-primary bg-bg-surface hover:border-accent-blue/40 hover:bg-bg-surface-hover transition-all sm:text-right"
        >
          <div className="flex items-center gap-1.5 text-xs font-medium text-text-tertiary sm:justify-end">
            <kbd className="mr-1 text-[10px] px-1 py-0.5 rounded border border-border-primary text-text-tertiary/70">
              →
            </kbd>
            <span>Next</span>
            <ArrowRight className="w-3.5 h-3.5 transition-transform group-hover:translate-x-0.5" />
          </div>
          <span className="text-sm font-semibold text-text-primary group-hover:text-accent-blue transition-colors line-clamp-2">
            {next.label}
          </span>
        </Link>
      ) : (
        <div />
      )}
    </nav>
  )
}
