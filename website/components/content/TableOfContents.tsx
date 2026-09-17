'use client'

import { useEffect, useState } from 'react'

export interface TocItem { id: string; title: string; level: number }

interface TableOfContentsProps {
  items: TocItem[]
  /** small print under the list: reading time, sources, position in the track */
  meta?: React.ReactNode
}

/**
 * "On this page" rail (≥ 80rem). The current section is the last heading above
 * the reading line; earlier sections are dimmed as read.
 */
export function TableOfContents({ items, meta }: TableOfContentsProps) {
  const [current, setCurrent] = useState(-1)

  useEffect(() => {
    if (items.length === 0) return
    let raf: number | null = null
    const mark = () => {
      raf = null
      const line = window.scrollY + 120
      let cur = -1
      items.forEach((item, i) => {
        const el = document.getElementById(item.id)
        if (el && el.getBoundingClientRect().top + window.scrollY <= line) cur = i
      })
      setCurrent(cur)
    }
    const onScroll = () => { if (raf === null) raf = requestAnimationFrame(mark) }
    mark()
    window.addEventListener('scroll', onScroll, { passive: true })
    window.addEventListener('resize', onScroll, { passive: true })
    return () => {
      window.removeEventListener('scroll', onScroll)
      window.removeEventListener('resize', onScroll)
      if (raf !== null) cancelAnimationFrame(raf)
    }
  }, [items])

  if (items.length === 0 && !meta) return null

  return (
    <nav className="ui reading-rail no-print font-sans text-[12.5px] leading-[1.4]" aria-label="On this page">
      {items.length > 0 && (
        <>
          <p className="mb-3 text-[11px] font-semibold uppercase tracking-[0.08em] text-text-tertiary">On this page</p>
          <ol>
            {items.map((item, i) => {
              const active = i === current
              const past = i < current
              return (
                <li key={item.id}>
                  <a
                    href={`#${item.id}`}
                    aria-current={active ? 'location' : undefined}
                    className={`block py-[0.3rem] pr-2 border-l-[1.5px] transition-colors ${item.level === 3 ? 'pl-6 text-[11.5px]' : 'pl-3.5'} ${
                      active
                        ? 'border-text-primary text-text-primary'
                        : past
                          ? 'border-border-primary text-text-secondary hover:text-text-primary'
                          : 'border-border-primary text-text-tertiary hover:text-text-primary'
                    }`}
                  >
                    {item.title}
                  </a>
                </li>
              )
            })}
          </ol>
        </>
      )}
      {meta && <div className="mt-5 pt-4 border-t border-border-primary text-[11.5px] leading-relaxed text-text-tertiary">{meta}</div>}
    </nav>
  )
}
