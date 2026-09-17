'use client'

import Link from 'next/link'
import { Search, ListTree } from 'lucide-react'
import { useEffect, useState } from 'react'
import { ThemeToggle } from './ThemeToggle'
import { ReaderSettings } from './ReaderSettings'
import { useSearch } from '@/components/search/SearchProvider'

export interface Crumb { label: string; href?: string }

interface TopNavProps {
  crumbs?: Crumb[]
  /** when set, a Contents button opens the course drawer */
  onContents?: () => void
  contentsOpen?: boolean
}

/**
 * 48px reading bar: wordmark and breadcrumb on the left, tools on the right.
 * It hides while scrolling down and returns on the first scroll up, so during
 * reading the only chrome on screen is the progress hairline.
 */
export function TopNav({ crumbs = [], onContents, contentsOpen }: TopNavProps) {
  const { open: openSearch } = useSearch()
  const [hidden, setHidden] = useState(false)

  useEffect(() => {
    let last = window.scrollY
    let raf: number | null = null
    const onScroll = () => {
      if (raf !== null) return
      raf = requestAnimationFrame(() => {
        const y = window.scrollY
        setHidden(y > 160 && y > last)
        last = y
        raf = null
      })
    }
    window.addEventListener('scroll', onScroll, { passive: true })
    return () => {
      window.removeEventListener('scroll', onScroll)
      if (raf !== null) cancelAnimationFrame(raf)
    }
  }, [])

  return (
    <header
      className={`ui no-print fixed inset-x-0 top-0 z-40 h-12 flex items-center justify-between gap-4 px-4 font-sans text-[13px] text-text-tertiary
        bg-bg-primary/85 backdrop-blur-md transition-[transform,opacity] duration-200 ease-out
        ${hidden ? '-translate-y-full opacity-0' : 'translate-y-0 opacity-100'}`}
    >
      <nav className="flex items-center gap-2 min-w-0 whitespace-nowrap overflow-hidden" aria-label="Breadcrumb">
        <Link href="/" className="font-semibold tracking-tight text-text-secondary hover:text-text-primary transition-colors">
          aman.study
        </Link>
        {crumbs.map((c, i) => (
          <span key={i} className={`items-center gap-2 min-w-0 ${i < crumbs.length - 1 ? 'hidden sm:flex' : 'flex'}`}>
            <span className="opacity-40" aria-hidden="true">/</span>
            {c.href ? (
              <Link href={c.href} className="hover:text-text-primary transition-colors truncate">{c.label}</Link>
            ) : (
              <span className="text-text-primary truncate">{c.label}</span>
            )}
          </span>
        ))}
      </nav>

      <div className="flex items-center gap-0.5 shrink-0">
        {onContents && (
          <button
            type="button"
            onClick={onContents}
            aria-expanded={contentsOpen}
            className="inline-flex items-center gap-1.5 h-8 px-2.5 rounded-md hover:bg-bg-surface-hover hover:text-text-primary transition-colors"
          >
            <ListTree className="w-[15px] h-[15px]" />
            <span className="hidden sm:inline">Contents</span>
            <kbd className="hidden md:inline">c</kbd>
          </button>
        )}
        <button
          type="button"
          onClick={openSearch}
          className="inline-flex items-center gap-1.5 h-8 px-2.5 rounded-md hover:bg-bg-surface-hover hover:text-text-primary transition-colors"
          aria-label="Search"
        >
          <Search className="w-[15px] h-[15px]" />
          <kbd className="hidden md:inline">⌘K</kbd>
        </button>
        <ReaderSettings />
        <ThemeToggle />
      </div>
    </header>
  )
}
