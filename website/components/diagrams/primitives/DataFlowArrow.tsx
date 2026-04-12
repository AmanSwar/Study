'use client'

import { useDiagramTheme } from '../core/DiagramTheme'
import { motion } from 'framer-motion'

interface DataFlowArrowProps {
  from: { x: number; y: number }
  to: { x: number; y: number }
  label?: string
  bandwidth?: string
  color?: string
  active?: boolean
  dimmed?: boolean
  id?: string
  packetCount?: number
  speed?: number
}

export function DataFlowArrow({
  from,
  to,
  label,
  bandwidth,
  color,
  active = true,
  dimmed = false,
  id,
  packetCount = 3,
  speed = 1.2,
}: DataFlowArrowProps) {
  const colors = useDiagramTheme()
  const c = color || colors.arrow
  const opacity = dimmed ? 0.2 : 1

  const dx = to.x - from.x
  const dy = to.y - from.y
  const len = Math.sqrt(dx * dx + dy * dy)
  const ux = dx / len
  const uy = dy / len

  // Arrowhead
  const headSize = 8
  const perpX = -uy
  const perpY = ux
  const headPoints = [
    `${to.x},${to.y}`,
    `${to.x - ux * headSize + perpX * headSize * 0.4},${to.y - uy * headSize + perpY * headSize * 0.4}`,
    `${to.x - ux * headSize - perpX * headSize * 0.4},${to.y - uy * headSize - perpY * headSize * 0.4}`,
  ].join(' ')

  const midX = (from.x + to.x) / 2
  const midY = (from.y + to.y) / 2

  return (
    <g id={id} style={{ opacity }}>
      {/* Main line */}
      <line
        x1={from.x}
        y1={from.y}
        x2={to.x - ux * headSize}
        y2={to.y - uy * headSize}
        stroke={c}
        strokeWidth={2}
        strokeOpacity={0.4}
      />

      {/* Arrowhead */}
      <polygon points={headPoints} fill={c} fillOpacity={0.6} />

      {/* Animated data packets */}
      {active &&
        Array.from({ length: packetCount }, (_, i) => (
          <motion.circle
            key={i}
            r={4}
            fill={c}
            filter="url(#glow)"
            initial={{ opacity: 0 }}
            animate={{
              cx: [from.x, to.x],
              cy: [from.y, to.y],
              opacity: [0, 0.9, 0.9, 0],
            }}
            transition={{
              duration: speed,
              repeat: Infinity,
              ease: 'linear',
              delay: (i / packetCount) * speed,
            }}
          />
        ))}

      {/* Bandwidth label */}
      {bandwidth && (
        <g>
          <rect
            x={midX - bandwidth.length * 3}
            y={midY + 6}
            width={bandwidth.length * 6 + 8}
            height={14}
            rx={3}
            fill={colors.bg}
            fillOpacity={0.85}
            stroke={c}
            strokeWidth={0.5}
            strokeOpacity={0.3}
          />
          <text
            x={midX + 4}
            y={midY + 13}
            textAnchor="middle"
            dominantBaseline="central"
            fill={c}
            fontSize={8}
            fontWeight={600}
            fontFamily="JetBrains Mono, monospace"
          >
            {bandwidth}
          </text>
        </g>
      )}

      {/* Label */}
      {label && (
        <text
          x={midX}
          y={midY - 8}
          textAnchor="middle"
          dominantBaseline="central"
          fill={colors.textLight}
          fontSize={9}
          fontWeight={500}
          fontFamily="Inter, system-ui, sans-serif"
        >
          {label}
        </text>
      )}
    </g>
  )
}
