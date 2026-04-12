'use client'

import { useDiagramTheme } from '../core/DiagramTheme'
import { motion } from 'framer-motion'

export type BlockVariant = 'primary' | 'secondary' | 'accent' | 'muted' | 'compute' | 'memory' | 'success' | 'warning' | 'danger'

interface BlockProps {
  x: number
  y: number
  width: number
  height: number
  label: string
  sublabel?: string
  variant?: BlockVariant
  highlighted?: boolean
  dimmed?: boolean
  id?: string
  rx?: number
  fontSize?: number
  sublabelSize?: number
  onClick?: () => void
  onMouseEnter?: (e: React.MouseEvent) => void
  onMouseMove?: (e: React.MouseEvent) => void
  onMouseLeave?: () => void
  icon?: React.ReactNode
}

export function Block({
  x,
  y,
  width,
  height,
  label,
  sublabel,
  variant = 'primary',
  highlighted = false,
  dimmed = false,
  id,
  rx = 8,
  fontSize = 12,
  sublabelSize = 10,
  onClick,
  onMouseEnter,
  onMouseMove,
  onMouseLeave,
}: BlockProps) {
  const colors = useDiagramTheme()

  const variantColors: Record<BlockVariant, { fill: string; stroke: string }> = {
    primary: { fill: colors.blockPrimary, stroke: colors.blockPrimary },
    secondary: { fill: colors.blockSecondary, stroke: colors.blockSecondary },
    accent: { fill: colors.blockAccent, stroke: colors.blockAccent },
    muted: { fill: colors.blockMuted, stroke: colors.blockMuted },
    compute: { fill: colors.compute, stroke: colors.compute },
    memory: { fill: colors.memory, stroke: colors.memory },
    success: { fill: colors.success, stroke: colors.success },
    warning: { fill: colors.warning, stroke: colors.warning },
    danger: { fill: colors.danger, stroke: colors.danger },
  }

  const { fill, stroke } = variantColors[variant]
  const opacity = dimmed ? 0.3 : 1

  return (
    <motion.g
      id={id}
      style={{ cursor: onClick ? 'pointer' : 'default', opacity }}
      onClick={onClick}
      onMouseEnter={onMouseEnter}
      onMouseMove={onMouseMove}
      onMouseLeave={onMouseLeave}
      initial={false}
      animate={{
        filter: highlighted ? `drop-shadow(0 0 8px ${colors.highlight})` : 'none',
      }}
      transition={{ duration: 0.3 }}
    >
      {/* Block background */}
      <motion.rect
        x={x}
        y={y}
        width={width}
        height={height}
        rx={rx}
        fill={fill}
        fillOpacity={0.15}
        stroke={stroke}
        strokeWidth={highlighted ? 2.5 : 1.5}
        strokeOpacity={highlighted ? 1 : 0.6}
        initial={false}
        animate={{
          strokeOpacity: highlighted ? 1 : 0.6,
          strokeWidth: highlighted ? 2.5 : 1.5,
        }}
        transition={{ duration: 0.3 }}
      />

      {/* Highlight glow */}
      {highlighted && (
        <rect
          x={x}
          y={y}
          width={width}
          height={height}
          rx={rx}
          fill={colors.highlight}
          fillOpacity={0.08}
        />
      )}

      {/* Label */}
      <text
        x={x + width / 2}
        y={sublabel ? y + height / 2 - 4 : y + height / 2}
        textAnchor="middle"
        dominantBaseline="central"
        fill={colors.text}
        fontSize={fontSize}
        fontWeight={600}
        fontFamily="Inter, system-ui, sans-serif"
      >
        {label}
      </text>

      {/* Sublabel */}
      {sublabel && (
        <text
          x={x + width / 2}
          y={y + height / 2 + 12}
          textAnchor="middle"
          dominantBaseline="central"
          fill={colors.textLight}
          fontSize={sublabelSize}
          fontWeight={400}
          fontFamily="Inter, system-ui, sans-serif"
        >
          {sublabel}
        </text>
      )}
    </motion.g>
  )
}
