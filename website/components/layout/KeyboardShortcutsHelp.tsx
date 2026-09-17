'use client'

import { useEffect, useState } from 'react'
import { X } from 'lucide-react'

interface Shortcut {
  keys: string[]
  description: string
}

const shortcuts: { group: string; items: Shortcut[] }[] = [
  {
    group: 'Navigation',
    items: [
      { keys: ['⌘', 'K'], description: 'Search' },
      { keys: ['/'], description: 'Search' },
      { keys: ['c'], description: 'Course contents' },
      { keys: ['←'], description: 'Previous module' },
      { keys: ['→'], description: 'Next module' },
    ],
  },
  {
    group: 'Reading',
    items: [
      { keys: ['Aa'], description: 'Typeface, size, width, theme (top right)' },
      { keys: ['Esc'], description: 'Close a dialog or a zoomed figure' },
      { keys: ['?'], description: 'This list' },
    ],
  },
]

/** Keyboard shortcuts overlay, on `?`. */
export function KeyboardShortcutsHelp() {
  const [open, setOpen] = useState(false)

  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      const target = e.target as HTMLElement | null
      if (!target) return
      const tag = target.tagName
      if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT' || target.isContentEditable) return
      if (e.key === '?' && !e.metaKey && !e.ctrlKey) {
        e.preventDefault()
        setOpen(true)
      } else if (e.key === 'Escape') {
        setOpen(false)
      }
    }
    window.addEventListener('keydown', handler)
    return () => window.removeEventListener('keydown', handler)
  }, [])

  if (!open) return null

  return (
    <div
      className="ui fixed inset-0 z-[90] flex items-center justify-center p-4 bg-bg-primary/60 backdrop-blur-sm animate-fade-in font-sans"
      onClick={() => setOpen(false)}
    >
      <div
        role="dialog"
        aria-label="Keyboard shortcuts"
        className="w-full max-w-sm rounded-lg bg-bg-elevated border border-border-primary shadow-lg overflow-hidden"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-center justify-between px-5 py-3 border-b border-border-primary">
          <h2 className="text-[13px] font-semibold text-text-primary">Keyboard shortcuts</h2>
          <button
            type="button"
            onClick={() => setOpen(false)}
            className="flex items-center justify-center w-6 h-6 rounded hover:bg-bg-surface-hover text-text-tertiary hover:text-text-primary transition-colors"
            aria-label="Close"
          >
            <X className="w-4 h-4" />
          </button>
        </div>

        <div className="p-5 space-y-5">
          {shortcuts.map((group) => (
            <div key={group.group}>
              <div className="text-[10.5px] font-semibold uppercase tracking-[0.1em] text-text-tertiary mb-2">{group.group}</div>
              <div className="space-y-1.5">
                {group.items.map((shortcut, i) => (
                  <div key={i} className="flex items-center justify-between gap-4">
                    <span className="text-[13px] text-text-secondary">{shortcut.description}</span>
                    <div className="flex items-center gap-1">
                      {shortcut.keys.map((key, j) => <kbd key={j}>{key}</kbd>)}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}
