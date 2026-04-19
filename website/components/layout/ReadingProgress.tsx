'use client'

import { useEffect, useState } from 'react'

/**
 * A thin progress bar fixed to the top of the viewport showing how far the
 * user has scrolled through the page. Subtle and non-intrusive.
 */
export function ReadingProgress() {
  const [progress, setProgress] = useState(0)

  useEffect(() => {
    let rafId: number | null = null

    const update = () => {
      const scrollTop = window.scrollY
      const scrollHeight = document.documentElement.scrollHeight - window.innerHeight
      const pct = scrollHeight > 0 ? Math.min(100, (scrollTop / scrollHeight) * 100) : 0
      setProgress(pct)
      rafId = null
    }

    const onScroll = () => {
      if (rafId !== null) return
      rafId = requestAnimationFrame(update)
    }

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
    <div
      className="fixed top-0 left-0 right-0 h-[2px] z-[60] pointer-events-none"
      style={{ background: 'transparent' }}
    >
      <div
        className="h-full transition-[width] duration-150 ease-out"
        style={{
          width: `${progress}%`,
          background: 'linear-gradient(90deg, var(--accent-blue), var(--accent-cyan))',
        }}
      />
    </div>
  )
}
