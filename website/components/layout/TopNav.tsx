'use client'

import Link from 'next/link'
import { Search, Menu, X, BookOpen } from 'lucide-react'
import { useEffect, useState } from 'react'
import { ThemeToggle } from './ThemeToggle'
import { useSearch } from '@/components/search/SearchProvider'

interface TopNavProps {
  onMenuToggle?: () => void
  menuOpen?: boolean
}

export function TopNav({ onMenuToggle, menuOpen }: TopNavProps) {
  const { open: openSearch } = useSearch()
  const [isMac, setIsMac] = useState(true)

  useEffect(() => {
    // Detect OS for the keyboard shortcut hint
    setIsMac(/Mac|iPod|iPhone|iPad/.test(navigator.platform))
  }, [])

  return (
    <header className="sticky top-0 z-50 h-14 border-b border-border-primary bg-bg-primary/75 backdrop-blur-xl">
      <div className="flex items-center justify-between h-full px-4 lg:px-6">
        {/* Left: Menu button + Logo */}
        <div className="flex items-center gap-3">
          <button
            onClick={onMenuToggle}
            className="lg:hidden w-9 h-9 flex items-center justify-center rounded-lg
              hover:bg-bg-surface-hover transition-colors"
            aria-label="Toggle menu"
          >
            {menuOpen ? (
              <X className="w-5 h-5 text-text-secondary" />
            ) : (
              <Menu className="w-5 h-5 text-text-secondary" />
            )}
          </button>

          <Link href="/" className="flex items-center gap-2 group">
            <div className="w-7 h-7 rounded-md bg-gradient-to-br from-accent-blue to-accent-cyan flex items-center justify-center
              group-hover:scale-105 transition-transform shadow-sm">
              <BookOpen className="w-3.5 h-3.5 text-white" />
            </div>
            <span className="font-semibold text-base text-text-primary tracking-tight hidden sm:block">
              aman.study
            </span>
          </Link>
        </div>

        {/* Center: Search button (triggers the global Cmd+K dialog) */}
        <button
          onClick={openSearch}
          className="hidden md:flex items-center gap-2.5 px-3 py-1.5 rounded-lg
            bg-bg-surface border border-border-primary text-text-tertiary text-sm
            hover:border-border-secondary hover:text-text-secondary transition-colors
            min-w-[260px] group"
        >
          <Search className="w-4 h-4 shrink-0" />
          <span>Search modules...</span>
          <kbd className="ml-auto bg-bg-surface-hover text-[11px] leading-none">
            {isMac ? '⌘' : 'Ctrl'} K
          </kbd>
        </button>

        {/* Mobile: search icon only */}
        <button
          onClick={openSearch}
          className="md:hidden w-9 h-9 flex items-center justify-center rounded-lg
            hover:bg-bg-surface-hover text-text-secondary transition-colors"
          aria-label="Search"
        >
          <Search className="w-4 h-4" />
        </button>

        {/* Right: Theme toggle */}
        <div className="flex items-center gap-1">
          <ThemeToggle />
        </div>
      </div>
    </header>
  )
}
