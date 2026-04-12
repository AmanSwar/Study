'use client'

import { Play, Pause, SkipBack, SkipForward, ChevronLeft, ChevronRight } from 'lucide-react'

export interface AnimationStep {
  id: string
  title: string
  description: string
  duration: number // ms to auto-advance
  highlightElements: string[]
  animateArrows: string[]
  annotations?: Array<{
    elementId: string
    text: string
    position: 'top' | 'right' | 'bottom' | 'left'
  }>
}

interface AnimationControllerProps {
  steps: AnimationStep[]
  currentStep: number
  isPlaying: boolean
  onStepChange: (step: number) => void
  onPlayPause: () => void
}

export function AnimationController({
  steps,
  currentStep,
  isPlaying,
  onStepChange,
  onPlayPause,
}: AnimationControllerProps) {
  const step = steps[currentStep]
  const progress = steps.length > 1 ? (currentStep / (steps.length - 1)) * 100 : 100

  return (
    <div className="border-t border-border-primary bg-bg-surface/80 backdrop-blur-sm">
      {/* Progress bar */}
      <div className="h-1 bg-bg-surface-hover relative">
        <div
          className="h-full bg-accent-blue transition-all duration-300 ease-out"
          style={{ width: `${progress}%` }}
        />
        {/* Step markers */}
        {steps.map((_, i) => (
          <button
            key={i}
            onClick={() => onStepChange(i)}
            className="absolute top-1/2 -translate-y-1/2 w-2.5 h-2.5 rounded-full border-2 transition-all
              hover:scale-125"
            style={{
              left: `${steps.length > 1 ? (i / (steps.length - 1)) * 100 : 50}%`,
              transform: 'translate(-50%, -50%)',
              backgroundColor: i <= currentStep ? 'var(--accent-blue)' : 'var(--bg-surface-hover)',
              borderColor: i <= currentStep ? 'var(--accent-blue)' : 'var(--border-primary)',
            }}
          />
        ))}
      </div>

      {/* Controls */}
      <div className="flex items-center gap-3 px-4 py-3">
        {/* Transport controls */}
        <div className="flex items-center gap-1">
          <button
            onClick={() => onStepChange(0)}
            disabled={currentStep === 0}
            className="w-7 h-7 flex items-center justify-center rounded-md
              text-text-tertiary hover:text-text-primary hover:bg-bg-surface-hover
              disabled:opacity-30 disabled:cursor-not-allowed transition-colors"
          >
            <SkipBack className="w-3.5 h-3.5" />
          </button>
          <button
            onClick={() => onStepChange(Math.max(0, currentStep - 1))}
            disabled={currentStep === 0}
            className="w-7 h-7 flex items-center justify-center rounded-md
              text-text-tertiary hover:text-text-primary hover:bg-bg-surface-hover
              disabled:opacity-30 disabled:cursor-not-allowed transition-colors"
          >
            <ChevronLeft className="w-4 h-4" />
          </button>
          <button
            onClick={onPlayPause}
            className="w-9 h-9 flex items-center justify-center rounded-lg
              bg-accent-blue text-white hover:bg-accent-blue-hover transition-colors"
          >
            {isPlaying ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4 ml-0.5" />}
          </button>
          <button
            onClick={() => onStepChange(Math.min(steps.length - 1, currentStep + 1))}
            disabled={currentStep === steps.length - 1}
            className="w-7 h-7 flex items-center justify-center rounded-md
              text-text-tertiary hover:text-text-primary hover:bg-bg-surface-hover
              disabled:opacity-30 disabled:cursor-not-allowed transition-colors"
          >
            <ChevronRight className="w-4 h-4" />
          </button>
          <button
            onClick={() => onStepChange(steps.length - 1)}
            disabled={currentStep === steps.length - 1}
            className="w-7 h-7 flex items-center justify-center rounded-md
              text-text-tertiary hover:text-text-primary hover:bg-bg-surface-hover
              disabled:opacity-30 disabled:cursor-not-allowed transition-colors"
          >
            <SkipForward className="w-3.5 h-3.5" />
          </button>
        </div>

        {/* Step info */}
        <div className="flex-1 min-w-0 ml-2">
          <div className="flex items-center gap-2">
            <span className="text-xs font-medium text-accent-blue">
              Step {currentStep + 1}/{steps.length}
            </span>
            <span className="text-sm font-medium text-text-primary truncate">
              {step?.title}
            </span>
          </div>
          {step?.description && (
            <p className="text-xs text-text-tertiary mt-0.5 truncate">{step.description}</p>
          )}
        </div>
      </div>
    </div>
  )
}
