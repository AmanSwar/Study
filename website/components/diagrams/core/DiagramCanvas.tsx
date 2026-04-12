'use client'

import { ReactNode, useRef, useState, useCallback } from 'react'
import { useDiagramTheme } from './DiagramTheme'

interface DiagramCanvasProps {
  width: number
  height: number
  children: ReactNode
  className?: string
  title?: string
  zoomable?: boolean
  minZoom?: number
  maxZoom?: number
}

export function DiagramCanvas({
  width,
  height,
  children,
  className = '',
  title,
  zoomable = true,
  minZoom = 0.5,
  maxZoom = 3,
}: DiagramCanvasProps) {
  const colors = useDiagramTheme()
  const svgRef = useRef<SVGSVGElement>(null)
  const [zoom, setZoom] = useState(1)
  const [pan, setPan] = useState({ x: 0, y: 0 })
  const [isPanning, setIsPanning] = useState(false)
  const [panStart, setPanStart] = useState({ x: 0, y: 0 })

  const handleWheel = useCallback(
    (e: React.WheelEvent) => {
      if (!zoomable) return
      e.preventDefault()
      const delta = e.deltaY > 0 ? 0.9 : 1.1
      setZoom((z) => Math.min(maxZoom, Math.max(minZoom, z * delta)))
    },
    [zoomable, minZoom, maxZoom]
  )

  const handleMouseDown = useCallback((e: React.MouseEvent) => {
    if (!zoomable || e.button !== 1) return // middle click only for pan
    setIsPanning(true)
    setPanStart({ x: e.clientX - pan.x, y: e.clientY - pan.y })
  }, [zoomable, pan])

  const handleMouseMove = useCallback((e: React.MouseEvent) => {
    if (!isPanning) return
    setPan({ x: e.clientX - panStart.x, y: e.clientY - panStart.y })
  }, [isPanning, panStart])

  const handleMouseUp = useCallback(() => {
    setIsPanning(false)
  }, [])

  const resetView = useCallback(() => {
    setZoom(1)
    setPan({ x: 0, y: 0 })
  }, [])

  return (
    <div className={`my-8 rounded-xl border border-border-primary overflow-hidden ${className}`}>
      {/* Title bar */}
      {title && (
        <div className="flex items-center justify-between px-4 py-2 border-b border-border-primary bg-bg-surface/50">
          <span className="text-xs font-semibold text-text-tertiary uppercase tracking-wider">{title}</span>
          {zoomable && zoom !== 1 && (
            <button
              onClick={resetView}
              className="text-xs text-text-tertiary hover:text-accent-blue transition-colors"
            >
              Reset view
            </button>
          )}
        </div>
      )}

      {/* SVG Container */}
      <div
        className="relative overflow-hidden"
        style={{ backgroundColor: colors.bg }}
      >
        <svg
          ref={svgRef}
          viewBox={`0 0 ${width} ${height}`}
          className="w-full h-auto select-none"
          style={{
            transform: `scale(${zoom}) translate(${pan.x / zoom}px, ${pan.y / zoom}px)`,
            transformOrigin: 'center center',
            cursor: isPanning ? 'grabbing' : zoomable ? 'default' : 'default',
          }}
          onWheel={handleWheel}
          onMouseDown={handleMouseDown}
          onMouseMove={handleMouseMove}
          onMouseUp={handleMouseUp}
          onMouseLeave={handleMouseUp}
        >
          {children}
        </svg>
      </div>
    </div>
  )
}
