'use client'

import Link from 'next/link'
import { Search, Menu, X, BookOpen } from 'lucide-react'
import { ThemeToggle } from './ThemeToggle'
import { useState } from 'react'

interface TopNavProps {
  onMenuToggle?: () => void
  menuOpen?: boolean
}

export function TopNav({ onMenuToggle, menuOpen }: TopNavProps) {
  return (
    <header className="sticky top-0 z-50 h-14 border-b border-border-primary bg-bg-primary/80 backdrop-blur-xl">
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

          <Link href="/" className="flex items-center gap-2.5 group">
            <div className="w-8 h-8 rounded-lg bg-accent-blue flex items-center justify-center
              group-hover:scale-105 transition-transform">
              <BookOpen className="w-4 h-4 text-white" />
            </div>
            <span className="font-bold text-lg text-text-primary tracking-tight hidden sm:block">
              CS Study
            </span>
          </Link>
        </div>

        {/* Center: Search (placeholder for now) */}
        <button
          className="hidden md:flex items-center gap-2 px-3 py-1.5 rounded-lg
            bg-bg-surface border border-border-primary text-text-tertiary text-sm
            hover:border-border-secondary hover:text-text-secondary transition-colors
            min-w-[240px]"
          onClick={() => {/* TODO: open search dialog */}}
        >
          <Search className="w-4 h-4" />
          <span>Search modules...</span>
          <kbd className="ml-auto text-xs bg-bg-surface-hover px-1.5 py-0.5 rounded border border-border-primary font-mono">
            ⌘K
          </kbd>
        </button>

        {/* Right: Theme toggle */}
        <div className="flex items-center gap-2">
          <ThemeToggle />
        </div>
      </div>
    </header>
  )
}
