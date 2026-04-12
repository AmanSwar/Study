'use client'

import { useEffect, useState } from 'react'
import { List } from 'lucide-react'

interface TocItem {
  id: string
  title: string
  level: number
}

interface TableOfContentsProps {
  items: TocItem[]
}

export function TableOfContents({ items }: TableOfContentsProps) {
  const [activeId, setActiveId] = useState<string>('')

  useEffect(() => {
    const observers: IntersectionObserver[] = []

    items.forEach((item) => {
      const el = document.getElementById(item.id)
      if (!el) return

      const observer = new IntersectionObserver(
        (entries) => {
          entries.forEach((entry) => {
            if (entry.isIntersecting) {
              setActiveId(item.id)
            }
          })
        },
        {
          rootMargin: '-80px 0px -70% 0px',
          threshold: 0,
        }
      )

      observer.observe(el)
      observers.push(observer)
    })

    return () => observers.forEach((o) => o.disconnect())
  }, [items])

  if (items.length === 0) return null

  return (
    <nav className="hidden xl:block sticky top-20 w-56 shrink-0 ml-8">
      <div className="flex items-center gap-2 mb-3 text-xs font-semibold uppercase tracking-wider text-text-tertiary">
        <List className="w-3.5 h-3.5" />
        On this page
      </div>
      <ul className="space-y-1 border-l border-border-primary">
        {items.map((item) => (
          <li key={item.id}>
            <a
              href={`#${item.id}`}
              className={`block text-xs leading-relaxed py-1 transition-colors
                ${item.level === 2 ? 'pl-3' : 'pl-6'}
                ${activeId === item.id
                  ? 'text-accent-blue border-l-2 border-accent-blue -ml-px font-medium'
                  : 'text-text-tertiary hover:text-text-secondary'
                }`}
            >
              {item.title}
            </a>
          </li>
        ))}
      </ul>
    </nav>
  )
}
