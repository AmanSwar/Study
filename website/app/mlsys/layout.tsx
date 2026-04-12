'use client'

import { useState } from 'react'
import { TopNav } from '@/components/layout/TopNav'
import { TrackSidebar } from '@/components/layout/TrackSidebar'
import { Footer } from '@/components/layout/Footer'
import { mlsysTrack } from '@/content/mlsys'

export default function MlsysLayout({ children }: { children: React.ReactNode }) {
  const [menuOpen, setMenuOpen] = useState(false)

  return (
    <>
      <TopNav onMenuToggle={() => setMenuOpen(!menuOpen)} menuOpen={menuOpen} />
      <div className="flex flex-1 min-h-0">
        {/* Sidebar - desktop */}
        <aside className="hidden lg:block w-72 shrink-0 border-r border-border-primary bg-bg-sidebar overflow-y-auto sticky top-14 h-[calc(100vh-3.5rem)]">
          <TrackSidebar track={mlsysTrack} />
        </aside>

        {/* Sidebar - mobile overlay */}
        {menuOpen && (
          <div className="lg:hidden fixed inset-0 z-40 bg-black/50" onClick={() => setMenuOpen(false)}>
            <aside
              className="w-72 h-full bg-bg-sidebar border-r border-border-primary overflow-y-auto"
              onClick={(e) => e.stopPropagation()}
            >
              <TrackSidebar track={mlsysTrack} />
            </aside>
          </div>
        )}

        {/* Main content */}
        <main className="flex-1 min-w-0 overflow-y-auto">
          <div className="max-w-4xl mx-auto px-4 lg:px-8 py-6">
            {children}
          </div>
          <Footer />
        </main>
      </div>
    </>
  )
}
