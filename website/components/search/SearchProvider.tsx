'use client'

import { createContext, useContext, useState, useEffect, ReactNode } from 'react'
import { SearchDialog } from './SearchDialog'
import type { SearchItem } from '@/lib/search-index'

// The index is prerendered at build time (app/search-index.json/route.ts) and
// fetched once, lazily, the first time search opens.
let indexPromise: Promise<SearchItem[]> | null = null
function loadIndex(): Promise<SearchItem[]> {
  return (indexPromise ??= fetch('/search-index.json').then((r) => (r.ok ? r.json() : [])).catch(() => { indexPromise = null; return [] }))
}

interface SearchContextValue {
  open: () => void
  close: () => void
  isOpen: boolean
}

const SearchContext = createContext<SearchContextValue | null>(null)

/**
 * Provides global search state + Cmd+K / Ctrl+K keyboard shortcut.
 * Wrap the root layout with this; use `useSearch()` anywhere to open the
 * search dialog.
 */
export function SearchProvider({ children }: { children: ReactNode }) {
  const [isOpen, setIsOpen] = useState(false)
  const [items, setItems] = useState<SearchItem[]>([])

  useEffect(() => {
    if (!isOpen || items.length) return
    let alive = true
    loadIndex().then((data) => { if (alive) setItems(data) })
    return () => { alive = false }
  }, [isOpen, items.length])

  const open = () => setIsOpen(true)
  const close = () => setIsOpen(false)

  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      // Cmd+K on Mac, Ctrl+K on Windows/Linux
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'k') {
        e.preventDefault()
        setIsOpen((prev) => !prev)
      } else if (e.key === '/' && !isOpen) {
        // Slash also opens search, skip if in input
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
        e.preventDefault()
        setIsOpen(true)
      }
    }

    window.addEventListener('keydown', handler)
    return () => window.removeEventListener('keydown', handler)
  }, [isOpen])

  return (
    <SearchContext.Provider value={{ open, close, isOpen }}>
      {children}
      <SearchDialog open={isOpen} onClose={close} items={items} />
    </SearchContext.Provider>
  )
}

export function useSearch() {
  const ctx = useContext(SearchContext)
  if (!ctx) throw new Error('useSearch must be used inside <SearchProvider>')
  return ctx
}
