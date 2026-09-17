'use client'

import { useState, type ReactNode } from 'react'
import { TopNav } from '@/components/layout/TopNav'

/**
 * Track page chrome: top nav, sticky sidebar (desktop) / overlay (mobile),
 * centred content column, footer. The sidebar and footer are rendered by the
 * server layout and passed in as nodes so this client component never needs
 * the registry.
 */
export function TrackShell({ sidebar, footer, children }: { sidebar: ReactNode; footer: ReactNode; children: ReactNode }) {
  const [menuOpen, setMenuOpen] = useState(false)

  return (
    <>
      <TopNav onMenuToggle={() => setMenuOpen(!menuOpen)} menuOpen={menuOpen} />
      <div className="flex flex-1 min-h-0">
        <aside className="hidden lg:block w-72 shrink-0 border-r border-border-primary bg-bg-sidebar overflow-y-auto sticky top-14 h-[calc(100vh-3.5rem)]">
          {sidebar}
        </aside>

        {menuOpen && (
          <div className="lg:hidden fixed inset-0 z-40 bg-black/50" onClick={() => setMenuOpen(false)}>
            <aside
              className="w-72 h-full bg-bg-sidebar border-r border-border-primary overflow-y-auto"
              onClick={(e) => e.stopPropagation()}
            >
              {sidebar}
            </aside>
          </div>
        )}

        <main className="flex-1 min-w-0 overflow-y-auto">
          <div className="max-w-4xl mx-auto px-4 lg:px-8 py-6">
            {children}
          </div>
          {footer}
        </main>
      </div>
    </>
  )
}
