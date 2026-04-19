'use client'

import { useEffect, useState } from 'react'

interface TocItem {
  id: string
  title: string
  level: number
}

interface TableOfContentsProps {
  items: TocItem[]
}

/**
 * Floating table of contents shown on wide screens (≥1280px).
 * Uses IntersectionObserver to highlight the currently-visible section.
 */
export function TableOfContents({ items }: TableOfContentsProps) {
  const [activeId, setActiveId] = useState<string>('')

  useEffect(() => {
    if (items.length === 0) return

    // Track visible headings. As they enter/leave the viewport, we pick the
    // topmost visible one as active.
    const visible = new Set<string>()

    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) visible.add(entry.target.id)
          else visible.delete(entry.target.id)
        })

        // Pick the first item in the document that's currently visible
        const firstVisible = items.find((item) => visible.has(item.id))
        if (firstVisible) {
          setActiveId(firstVisible.id)
        }
      },
      {
        rootMargin: '-80px 0px -60% 0px',
        threshold: 0,
      }
    )

    items.forEach((item) => {
      const el = document.getElementById(item.id)
      if (el) observer.observe(el)
    })

    return () => observer.disconnect()
  }, [items])

  if (items.length === 0) return null

  return (
    <nav className="hidden xl:block sticky top-20 w-56 shrink-0 ml-10 self-start max-h-[calc(100vh-6rem)] overflow-y-auto">
      <div className="mb-4 text-[10px] font-semibold uppercase tracking-[0.12em] text-text-tertiary">
        On this page
      </div>
      <ul className="space-y-0.5">
        {items.map((item) => {
          const isActive = activeId === item.id
          return (
            <li key={item.id}>
              <a
                href={`#${item.id}`}
                className={`group block py-1.5 text-[13px] leading-snug transition-colors relative ${
                  item.level === 3 ? 'pl-5' : 'pl-3'
                } ${
                  isActive
                    ? 'text-accent-blue'
                    : 'text-text-tertiary hover:text-text-primary'
                }`}
              >
                {/* Active indicator */}
                <span
                  className={`absolute left-0 top-1/2 -translate-y-1/2 w-0.5 rounded-full transition-all ${
                    isActive
                      ? 'h-5 bg-accent-blue'
                      : 'h-0 bg-transparent group-hover:h-3 group-hover:bg-text-tertiary/40'
                  }`}
                  aria-hidden="true"
                />
                <span className="line-clamp-2">{item.title}</span>
              </a>
            </li>
          )
        })}
      </ul>
    </nav>
  )
}
