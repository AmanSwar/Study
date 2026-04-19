'use client'

import { createContext, useContext, useState, useEffect, ReactNode, useMemo } from 'react'
import { SearchDialog } from './SearchDialog'
import { buildSearchIndex } from '@/lib/search-index'

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
  const items = useMemo(() => buildSearchIndex(), [])

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
