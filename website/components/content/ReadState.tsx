'use client'

import { useEffect, useState } from 'react'
import { getPosition, onPositionsChange } from '@/lib/reading-state'

/**
 * A small ring after a module title: empty (unread), half (in progress), filled (read).
 * Renders nothing until mounted so server and client markup agree.
 */
export function ReadState({ href, className = '' }: { href: string; className?: string }) {
  const [state, setState] = useState<'none' | 'part' | 'done'>('none')

  useEffect(() => {
    const read = () => {
      const p = getPosition(href)
      setState(!p ? 'none' : p.done ? 'done' : p.pct >= 3 ? 'part' : 'none')
    }
    read()
    return onPositionsChange(read)
  }, [href])

  const title = state === 'done' ? 'Read' : state === 'part' ? 'In progress' : 'Unread'
  return (
    <span
      role="img"
      aria-label={title}
      title={title}
      className={`inline-block w-2 h-2 rounded-full border-[1.5px] align-middle shrink-0 ${className} ${
        state === 'done'
          ? 'bg-accent-green border-accent-green'
          : state === 'part'
            ? 'border-accent-blue [background:linear-gradient(90deg,var(--accent-blue)_50%,transparent_50%)]'
            : 'border-border-secondary'
      }`}
    />
  )
}
