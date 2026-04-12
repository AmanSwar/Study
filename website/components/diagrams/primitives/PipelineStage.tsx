'use client'

import { useDiagramTheme } from '../core/DiagramTheme'
import { motion } from 'framer-motion'

type StageStatus = 'idle' | 'active' | 'stalled' | 'flushed'

interface PipelineStageProps {
  x: number
  y: number
  width: number
  height: number
  label: string
  sublabel?: string
  status?: StageStatus
  id?: string
  onMouseEnter?: (e: React.MouseEvent) => void
  onMouseMove?: (e: React.MouseEvent) => void
  onMouseLeave?: () => void
}

export function PipelineStage({
  x,
  y,
  width,
  height,
  label,
  sublabel,
  status = 'idle',
  id,
  onMouseEnter,
  onMouseMove,
  onMouseLeave,
}: PipelineStageProps) {
  const colors = useDiagramTheme()

  const statusColors: Record<StageStatus, { fill: string; stroke: string; textColor: string }> = {
    idle: { fill: colors.blockMuted, stroke: colors.blockMuted, textColor: colors.textLight },
    active: { fill: colors.compute, stroke: colors.compute, textColor: colors.text },
    stalled: { fill: colors.danger, stroke: colors.danger, textColor: colors.text },
    flushed: { fill: colors.warning, stroke: colors.warning, textColor: colors.text },
  }

  const { fill, stroke, textColor } = statusColors[status]

  return (
    <motion.g
      id={id}
      onMouseEnter={onMouseEnter}
      onMouseMove={onMouseMove}
      onMouseLeave={onMouseLeave}
      initial={false}
      animate={{ opacity: status === 'idle' ? 0.5 : 1 }}
      transition={{ duration: 0.3 }}
    >
      <motion.rect
        x={x}
        y={y}
        width={width}
        height={height}
        rx={6}
        fill={fill}
        fillOpacity={status === 'active' ? 0.25 : 0.1}
        stroke={stroke}
        strokeWidth={status === 'active' ? 2 : 1}
        strokeOpacity={status === 'active' ? 0.8 : 0.3}
        initial={false}
        animate={{
          fillOpacity: status === 'active' ? 0.25 : status === 'stalled' ? 0.2 : 0.1,
        }}
        transition={{ duration: 0.3 }}
      />

      {/* Active pulse effect */}
      {status === 'active' && (
        <motion.rect
          x={x}
          y={y}
          width={width}
          height={height}
          rx={6}
          fill="none"
          stroke={fill}
          strokeWidth={2}
          initial={{ opacity: 0.6 }}
          animate={{ opacity: [0.6, 0, 0.6] }}
          transition={{ duration: 2, repeat: Infinity }}
        />
      )}

      {/* Stalled X pattern */}
      {status === 'stalled' && (
        <>
          <line
            x1={x + 4}
            y1={y + 4}
            x2={x + width - 4}
            y2={y + height - 4}
            stroke={colors.danger}
            strokeWidth={1}
            strokeOpacity={0.3}
          />
          <line
            x1={x + width - 4}
            y1={y + 4}
            x2={x + 4}
            y2={y + height - 4}
            stroke={colors.danger}
            strokeWidth={1}
            strokeOpacity={0.3}
          />
        </>
      )}

      <text
        x={x + width / 2}
        y={sublabel ? y + height / 2 - 5 : y + height / 2}
        textAnchor="middle"
        dominantBaseline="central"
        fill={textColor}
        fontSize={11}
        fontWeight={600}
        fontFamily="Inter, system-ui, sans-serif"
      >
        {label}
      </text>

      {sublabel && (
        <text
          x={x + width / 2}
          y={y + height / 2 + 10}
          textAnchor="middle"
          dominantBaseline="central"
          fill={colors.textLight}
          fontSize={8}
          fontWeight={400}
          fontFamily="JetBrains Mono, monospace"
        >
          {sublabel}
        </text>
      )}
    </motion.g>
  )
}
