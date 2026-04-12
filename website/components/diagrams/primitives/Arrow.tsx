'use client'

import { useDiagramTheme } from '../core/DiagramTheme'
import { motion } from 'framer-motion'

interface ArrowProps {
  from: { x: number; y: number }
  to: { x: number; y: number }
  label?: string
  color?: string
  animated?: boolean
  highlighted?: boolean
  dimmed?: boolean
  dashed?: boolean
  strokeWidth?: number
  id?: string
  bidirectional?: boolean
  onMouseEnter?: (e: React.MouseEvent) => void
  onMouseMove?: (e: React.MouseEvent) => void
  onMouseLeave?: () => void
}

export function Arrow({
  from,
  to,
  label,
  color,
  animated = false,
  highlighted = false,
  dimmed = false,
  dashed = false,
  strokeWidth = 1.5,
  id,
  bidirectional = false,
  onMouseEnter,
  onMouseMove,
  onMouseLeave,
}: ArrowProps) {
  const colors = useDiagramTheme()
  const c = color || (highlighted ? colors.arrowActive : colors.arrow)
  const opacity = dimmed ? 0.2 : 1

  // Calculate arrowhead
  const dx = to.x - from.x
  const dy = to.y - from.y
  const len = Math.sqrt(dx * dx + dy * dy)
  const ux = dx / len
  const uy = dy / len

  // Shorten line to not overlap arrowhead
  const headSize = 8
  const lineEnd = { x: to.x - ux * headSize, y: to.y - uy * headSize }
  const lineStart = bidirectional
    ? { x: from.x + ux * headSize, y: from.y + uy * headSize }
    : from

  // Arrowhead points
  const perpX = -uy
  const perpY = ux
  const headPoints = [
    `${to.x},${to.y}`,
    `${to.x - ux * headSize + perpX * headSize * 0.4},${to.y - uy * headSize + perpY * headSize * 0.4}`,
    `${to.x - ux * headSize - perpX * headSize * 0.4},${to.y - uy * headSize - perpY * headSize * 0.4}`,
  ].join(' ')

  // Reverse arrowhead for bidirectional
  const reverseHeadPoints = bidirectional ? [
    `${from.x},${from.y}`,
    `${from.x + ux * headSize + perpX * headSize * 0.4},${from.y + uy * headSize + perpY * headSize * 0.4}`,
    `${from.x + ux * headSize - perpX * headSize * 0.4},${from.y + uy * headSize - perpY * headSize * 0.4}`,
  ].join(' ') : ''

  // Label position (midpoint)
  const midX = (from.x + to.x) / 2
  const midY = (from.y + to.y) / 2

  return (
    <g
      id={id}
      style={{ opacity }}
      onMouseEnter={onMouseEnter}
      onMouseMove={onMouseMove}
      onMouseLeave={onMouseLeave}
    >
      {/* Line */}
      <motion.line
        x1={lineStart.x}
        y1={lineStart.y}
        x2={lineEnd.x}
        y2={lineEnd.y}
        stroke={c}
        strokeWidth={highlighted ? strokeWidth + 1 : strokeWidth}
        strokeDasharray={dashed ? '6 4' : undefined}
        initial={false}
        animate={{ stroke: c }}
        transition={{ duration: 0.3 }}
      />

      {/* Animated flow dots along line */}
      {animated && (
        <>
          <motion.circle
            r={3}
            fill={c}
            initial={{ opacity: 0 }}
            animate={{
              cx: [from.x, to.x],
              cy: [from.y, to.y],
              opacity: [0, 1, 1, 0],
            }}
            transition={{
              duration: 1.5,
              repeat: Infinity,
              ease: 'linear',
            }}
          />
          <motion.circle
            r={3}
            fill={c}
            initial={{ opacity: 0 }}
            animate={{
              cx: [from.x, to.x],
              cy: [from.y, to.y],
              opacity: [0, 1, 1, 0],
            }}
            transition={{
              duration: 1.5,
              repeat: Infinity,
              ease: 'linear',
              delay: 0.5,
            }}
          />
          <motion.circle
            r={3}
            fill={c}
            initial={{ opacity: 0 }}
            animate={{
              cx: [from.x, to.x],
              cy: [from.y, to.y],
              opacity: [0, 1, 1, 0],
            }}
            transition={{
              duration: 1.5,
              repeat: Infinity,
              ease: 'linear',
              delay: 1.0,
            }}
          />
        </>
      )}

      {/* Forward arrowhead */}
      <polygon points={headPoints} fill={c} />

      {/* Reverse arrowhead */}
      {bidirectional && <polygon points={reverseHeadPoints} fill={c} />}

      {/* Label */}
      {label && (
        <g>
          <rect
            x={midX - label.length * 3.2}
            y={midY - 16}
            width={label.length * 6.4}
            height={14}
            rx={3}
            fill={colors.bg}
            fillOpacity={0.9}
          />
          <text
            x={midX}
            y={midY - 9}
            textAnchor="middle"
            dominantBaseline="central"
            fill={colors.textLight}
            fontSize={9}
            fontWeight={500}
            fontFamily="JetBrains Mono, monospace"
          >
            {label}
          </text>
        </g>
      )}
    </g>
  )
}
