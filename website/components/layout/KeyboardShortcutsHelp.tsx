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
      { keys: ['⌘', 'K'], description: 'Open search' },
      { keys: ['/'], description: 'Open search' },
      { keys: ['←'], description: 'Previous module' },
      { keys: ['→'], description: 'Next module' },
    ],
  },
  {
    group: 'General',
    items: [
      { keys: ['?'], description: 'Show keyboard shortcuts' },
      { keys: ['Esc'], description: 'Close dialog' },
    ],
  },
]

/**
 * Keyboard shortcuts help overlay. Triggered by pressing `?`.
 * Shows a list of all available shortcuts in a modal.
 */
export function KeyboardShortcutsHelp() {
  const [open, setOpen] = useState(false)

  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      // Skip if user is typing in an input
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

      if (e.key === '?' && !e.metaKey && !e.ctrlKey) {
        e.preventDefault()
        setOpen(true)
      } else if (e.key === 'Escape' && open) {
        e.preventDefault()
        setOpen(false)
      }
    }

    window.addEventListener('keydown', handler)
    return () => window.removeEventListener('keydown', handler)
  }, [open])

  if (!open) return null

  return (
    <div
      className="fixed inset-0 z-[90] flex items-center justify-center p-4 bg-black/50 backdrop-blur-sm animate-fade-in"
      onClick={() => setOpen(false)}
    >
      <div
        className="w-full max-w-md rounded-xl bg-bg-elevated border border-border-primary shadow-2xl overflow-hidden"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-center justify-between px-5 py-3.5 border-b border-border-primary">
          <h2 className="text-sm font-semibold text-text-primary">Keyboard Shortcuts</h2>
          <button
            onClick={() => setOpen(false)}
            className="flex items-center justify-center w-6 h-6 rounded hover:bg-bg-surface-hover text-text-tertiary hover:text-text-secondary transition-colors"
            aria-label="Close"
          >
            <X className="w-4 h-4" />
          </button>
        </div>

        <div className="p-5 space-y-5">
          {shortcuts.map((group) => (
            <div key={group.group}>
              <div className="text-[10px] font-semibold uppercase tracking-[0.1em] text-text-tertiary mb-2.5">
                {group.group}
              </div>
              <div className="space-y-2">
                {group.items.map((shortcut, i) => (
                  <div key={i} className="flex items-center justify-between">
                    <span className="text-sm text-text-secondary">{shortcut.description}</span>
                    <div className="flex items-center gap-1">
                      {shortcut.keys.map((key, j) => (
                        <kbd key={j}>{key}</kbd>
                      ))}
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
