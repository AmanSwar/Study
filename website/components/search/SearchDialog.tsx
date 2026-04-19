'use client'

import { useEffect, useRef, useState, useCallback } from 'react'
import { useRouter } from 'next/navigation'
import { Search, X, ArrowRight, CornerDownLeft } from 'lucide-react'
import { SearchItem, searchItems } from '@/lib/search-index'

interface SearchDialogProps {
  open: boolean
  onClose: () => void
  items: SearchItem[]
}

export function SearchDialog({ open, onClose, items }: SearchDialogProps) {
  const router = useRouter()
  const inputRef = useRef<HTMLInputElement>(null)
  const listRef = useRef<HTMLDivElement>(null)
  const [query, setQuery] = useState('')
  const [activeIndex, setActiveIndex] = useState(0)

  // Results: if query, filtered; else recent/all tracks grouped
  const results = query ? searchItems(items, query) : items.slice(0, 12)

  // Focus input when dialog opens
  useEffect(() => {
    if (open && inputRef.current) {
      inputRef.current.focus()
    }
  }, [open])

  // Reset state when dialog closes
  useEffect(() => {
    if (!open) {
      setQuery('')
      setActiveIndex(0)
    }
  }, [open])

  // Keep active index in bounds
  useEffect(() => {
    setActiveIndex((i) => Math.min(i, Math.max(0, results.length - 1)))
  }, [results.length])

  // Scroll active item into view
  useEffect(() => {
    const list = listRef.current
    if (!list) return
    const activeEl = list.querySelector<HTMLElement>(`[data-index="${activeIndex}"]`)
    if (activeEl) {
      activeEl.scrollIntoView({ block: 'nearest' })
    }
  }, [activeIndex])

  const handleSelect = useCallback(
    (item: SearchItem) => {
      router.push(item.href)
      onClose()
    },
    [router, onClose]
  )

  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent) => {
      if (e.key === 'ArrowDown') {
        e.preventDefault()
        setActiveIndex((i) => Math.min(i + 1, results.length - 1))
      } else if (e.key === 'ArrowUp') {
        e.preventDefault()
        setActiveIndex((i) => Math.max(i - 1, 0))
      } else if (e.key === 'Enter') {
        e.preventDefault()
        const selected = results[activeIndex]
        if (selected) handleSelect(selected)
      } else if (e.key === 'Escape') {
        e.preventDefault()
        onClose()
      }
    },
    [results, activeIndex, handleSelect, onClose]
  )

  if (!open) return null

  return (
    <div
      className="fixed inset-0 z-[100] flex items-start justify-center p-4 pt-[15vh] bg-black/50 backdrop-blur-sm animate-fade-in"
      onClick={onClose}
      role="dialog"
      aria-label="Search"
    >
      <div
        className="w-full max-w-xl rounded-xl bg-bg-elevated border border-border-primary shadow-2xl overflow-hidden"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Search input */}
        <div className="flex items-center gap-3 px-4 py-3 border-b border-border-primary">
          <Search className="w-5 h-5 text-text-tertiary shrink-0" />
          <input
            ref={inputRef}
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Search modules, chapters, topics..."
            className="flex-1 bg-transparent text-base text-text-primary placeholder:text-text-tertiary outline-none"
            autoComplete="off"
            spellCheck="false"
          />
          <button
            onClick={onClose}
            className="flex items-center justify-center w-6 h-6 rounded hover:bg-bg-surface-hover text-text-tertiary hover:text-text-secondary transition-colors"
            aria-label="Close search"
          >
            <X className="w-4 h-4" />
          </button>
        </div>

        {/* Results */}
        <div ref={listRef} className="max-h-[60vh] overflow-y-auto">
          {results.length === 0 ? (
            <div className="px-4 py-12 text-center text-sm text-text-tertiary">
              {query ? (
                <>
                  No results for <span className="text-text-primary font-medium">&quot;{query}&quot;</span>
                </>
              ) : (
                'Start typing to search'
              )}
            </div>
          ) : (
            <div className="py-2">
              {!query && (
                <div className="px-4 py-1.5 text-[10px] font-semibold uppercase tracking-wider text-text-tertiary">
                  Browse
                </div>
              )}
              {results.map((item, idx) => (
                <button
                  key={item.id}
                  data-index={idx}
                  onClick={() => handleSelect(item)}
                  onMouseEnter={() => setActiveIndex(idx)}
                  className={`w-full flex items-center gap-3 px-4 py-2.5 text-left transition-colors ${
                    activeIndex === idx
                      ? 'bg-accent-blue-subtle'
                      : 'hover:bg-bg-surface-hover'
                  }`}
                >
                  <TrackBadge track={item.track} />
                  <div className="flex-1 min-w-0">
                    <div className={`text-sm font-medium truncate ${
                      activeIndex === idx ? 'text-accent-blue' : 'text-text-primary'
                    }`}>
                      {item.title}
                    </div>
                    <div className="text-xs text-text-tertiary truncate">
                      {item.subtitle}
                    </div>
                  </div>
                  {activeIndex === idx && (
                    <ArrowRight className="w-4 h-4 text-accent-blue shrink-0" />
                  )}
                </button>
              ))}
            </div>
          )}
        </div>

        {/* Footer with shortcuts */}
        <div className="flex items-center justify-between gap-3 px-4 py-2.5 border-t border-border-primary bg-bg-surface/50 text-xs text-text-tertiary">
          <div className="flex items-center gap-4">
            <span className="flex items-center gap-1.5">
              <kbd>↑</kbd>
              <kbd>↓</kbd>
              <span>navigate</span>
            </span>
            <span className="flex items-center gap-1.5">
              <kbd className="inline-flex items-center gap-0.5"><CornerDownLeft className="w-2.5 h-2.5" /></kbd>
              <span>select</span>
            </span>
            <span className="flex items-center gap-1.5">
              <kbd>esc</kbd>
              <span>close</span>
            </span>
          </div>
          <span>{results.length} {results.length === 1 ? 'result' : 'results'}</span>
        </div>
      </div>
    </div>
  )
}

function TrackBadge({ track }: { track: string }) {
  const config: Record<string, { label: string; color: string }> = {
    mlsys: { label: 'MLsys', color: 'bg-blue-500/20 text-blue-500' },
    intel: { label: 'Intel', color: 'bg-cyan-500/20 text-cyan-600' },
    qualcomm: { label: 'Qualcomm', color: 'bg-orange-500/20 text-orange-500' },
    quant: { label: 'Quant', color: 'bg-green-500/20 text-green-600' },
  }
  const c = config[track] || { label: track, color: 'bg-gray-500/20 text-gray-500' }
  return (
    <span className={`text-[10px] font-semibold px-2 py-1 rounded-md uppercase tracking-wider shrink-0 ${c.color}`}>
      {c.label}
    </span>
  )
}
