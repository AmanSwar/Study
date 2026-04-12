'use client'

import { useState, useEffect, useCallback, useRef } from 'react'
import { AnimationStep } from '../core/AnimationController'

interface UsePipelineAnimationReturn {
  currentStep: number
  isPlaying: boolean
  activeElements: Set<string>
  activeArrows: Set<string>
  play: () => void
  pause: () => void
  togglePlayPause: () => void
  goToStep: (step: number) => void
  next: () => void
  prev: () => void
}

export function usePipelineAnimation(steps: AnimationStep[]): UsePipelineAnimationReturn {
  const [currentStep, setCurrentStep] = useState(0)
  const [isPlaying, setIsPlaying] = useState(false)
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null)

  const step = steps[currentStep]
  const activeElements = new Set(step?.highlightElements ?? [])
  const activeArrows = new Set(step?.animateArrows ?? [])

  const clearTimer = useCallback(() => {
    if (timerRef.current) {
      clearTimeout(timerRef.current)
      timerRef.current = null
    }
  }, [])

  const play = useCallback(() => setIsPlaying(true), [])
  const pause = useCallback(() => {
    setIsPlaying(false)
    clearTimer()
  }, [clearTimer])
  const togglePlayPause = useCallback(() => {
    setIsPlaying((p) => !p)
  }, [])

  const goToStep = useCallback(
    (s: number) => {
      const clamped = Math.max(0, Math.min(steps.length - 1, s))
      setCurrentStep(clamped)
    },
    [steps.length]
  )

  const next = useCallback(() => {
    setCurrentStep((s) => {
      if (s >= steps.length - 1) {
        setIsPlaying(false)
        return s
      }
      return s + 1
    })
  }, [steps.length])

  const prev = useCallback(() => {
    setCurrentStep((s) => Math.max(0, s - 1))
  }, [])

  // Auto-advance when playing
  useEffect(() => {
    if (!isPlaying) return
    if (currentStep >= steps.length - 1) {
      setIsPlaying(false)
      return
    }

    const duration = steps[currentStep]?.duration ?? 2000
    timerRef.current = setTimeout(() => {
      setCurrentStep((s) => s + 1)
    }, duration)

    return () => clearTimer()
  }, [isPlaying, currentStep, steps, clearTimer])

  return {
    currentStep,
    isPlaying,
    activeElements,
    activeArrows,
    play,
    pause,
    togglePlayPause,
    goToStep,
    next,
    prev,
  }
}
