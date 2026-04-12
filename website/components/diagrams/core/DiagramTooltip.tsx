'use client'

import { useState, useCallback, ReactNode } from 'react'
import { createPortal } from 'react-dom'
import { useDiagramTheme } from './DiagramTheme'

interface TooltipState {
  visible: boolean
  x: number
  y: number
  content: ReactNode
}

export function useDiagramTooltip() {
  const [tooltip, setTooltip] = useState<TooltipState>({
    visible: false,
    x: 0,
    y: 0,
    content: null,
  })

  const showTooltip = useCallback((e: React.MouseEvent, content: ReactNode) => {
    const rect = (e.currentTarget as Element).closest('svg')?.getBoundingClientRect()
    if (!rect) return
    setTooltip({
      visible: true,
      x: e.clientX,
      y: e.clientY - 10,
      content,
    })
  }, [])

  const moveTooltip = useCallback((e: React.MouseEvent) => {
    setTooltip((prev) => ({
      ...prev,
      x: e.clientX,
      y: e.clientY - 10,
    }))
  }, [])

  const hideTooltip = useCallback(() => {
    setTooltip((prev) => ({ ...prev, visible: false }))
  }, [])

  return { tooltip, showTooltip, moveTooltip, hideTooltip }
}

interface DiagramTooltipPortalProps {
  tooltip: TooltipState
}

export function DiagramTooltipPortal({ tooltip }: DiagramTooltipPortalProps) {
  const colors = useDiagramTheme()

  if (!tooltip.visible || !tooltip.content) return null
  if (typeof document === 'undefined') return null

  return createPortal(
    <div
      className="fixed z-[100] pointer-events-none animate-fade-in"
      style={{
        left: tooltip.x + 12,
        top: tooltip.y,
        transform: 'translateY(-100%)',
      }}
    >
      <div
        className="px-3 py-2 rounded-lg text-xs leading-relaxed max-w-xs shadow-lg border"
        style={{
          backgroundColor: colors.tooltipBg,
          borderColor: colors.tooltipBorder,
          color: colors.text,
        }}
      >
        {tooltip.content}
      </div>
    </div>,
    document.body
  )
}
