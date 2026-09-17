'use client'

import { useEffect, useState } from 'react'
import { onPositionsChange, trackProgress } from '@/lib/reading-state'

/** "3 of 39 read" with a thin bar, from what the reader has actually finished. Nothing until mounted. */
export function TrackProgress({ href, total, unit = 'read' }: { href: string; total: number; unit?: string }) {
  const [done, setDone] = useState<number | null>(null)

  useEffect(() => {
    const read = () => setDone(trackProgress(href).done)
    read()
    return onPositionsChange(read)
  }, [href])

  if (!done) return null
  return (
    <span className="ui inline-flex items-center gap-2.5 font-sans text-[12px] text-text-tertiary">
      <span className="relative inline-block w-24 h-[3px] rounded-full bg-border-primary overflow-hidden" aria-hidden="true">
        <span className="absolute inset-y-0 left-0 bg-accent-blue" style={{ width: `${Math.round((done / total) * 100)}%` }} />
      </span>
      {done} of {total} {unit}
    </span>
  )
}
