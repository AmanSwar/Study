'use client'

import { useState, useEffect, useCallback, useRef } from 'react'

interface UseHighlightSequenceOptions {
  elements: string[]
  interval?: number  // ms between highlights
  active: boolean
  loop?: boolean
}

export function useHighlightSequence({
  elements,
  interval = 1500,
  active,
  loop = true,
}: UseHighlightSequenceOptions) {
  const [activeIndex, setActiveIndex] = useState(0)
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null)

  useEffect(() => {
    if (!active || elements.length === 0) return

    timerRef.current = setInterval(() => {
      setActiveIndex((i) => {
        const next = i + 1
        if (next >= elements.length) {
          return loop ? 0 : i
        }
        return next
      })
    }, interval)

    return () => {
      if (timerRef.current) clearInterval(timerRef.current)
    }
  }, [active, elements.length, interval, loop])

  const activeElement = elements[activeIndex] || null

  const reset = useCallback(() => setActiveIndex(0), [])

  return { activeIndex, activeElement, reset }
}
