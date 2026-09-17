'use client'

import { useEffect, useState } from 'react'

/** A 2px hairline at the very top of the viewport: how far through the page you are. */
export function ReadingProgress() {
  const [progress, setProgress] = useState(0)

  useEffect(() => {
    let rafId: number | null = null
    const update = () => {
      const scrollTop = window.scrollY
      const scrollHeight = document.documentElement.scrollHeight - window.innerHeight
      setProgress(scrollHeight > 0 ? Math.min(100, (scrollTop / scrollHeight) * 100) : 0)
      rafId = null
    }
    const onScroll = () => { if (rafId === null) rafId = requestAnimationFrame(update) }
    update()
    window.addEventListener('scroll', onScroll, { passive: true })
    window.addEventListener('resize', onScroll, { passive: true })
    return () => {
      window.removeEventListener('scroll', onScroll)
      window.removeEventListener('resize', onScroll)
      if (rafId !== null) cancelAnimationFrame(rafId)
    }
  }, [])

  return (
    <div className="no-print fixed top-0 left-0 right-0 h-[2px] z-[60] pointer-events-none" aria-hidden="true">
      <div className="h-full bg-accent-blue transition-[width] duration-100 ease-linear" style={{ width: `${progress}%` }} />
    </div>
  )
}
