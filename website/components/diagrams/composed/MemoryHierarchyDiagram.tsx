'use client'

import { DiagramCanvas } from '../core/DiagramCanvas'
import { DiagramThemeProvider, useDiagramTheme } from '../core/DiagramTheme'
import { DiagramTooltipPortal, useDiagramTooltip } from '../core/DiagramTooltip'
import { AnimationController, AnimationStep } from '../core/AnimationController'
import { usePipelineAnimation } from '../animations/usePipelineAnimation'
import { motion } from 'framer-motion'

interface MemoryLevel {
  name: string
  size: string
  latency: string
  bandwidth: string
  color: string
}

interface MemoryHierarchyDiagramProps {
  levels?: MemoryLevel[]
  title?: string
}

const defaultLevels: MemoryLevel[] = [
  { name: 'Registers', size: '~10 KB', latency: '<1 ns', bandwidth: '~50 TB/s', color: '#22c55e' },
  { name: 'L1 Cache', size: '48 KB/core', latency: '~1 ns', bandwidth: '~15 TB/s', color: '#3b82f6' },
  { name: 'L2 Cache', size: '1-2 MB/core', latency: '~5 ns', bandwidth: '~5 TB/s', color: '#06b6d4' },
  { name: 'L3 / SLC', size: '32-96 MB', latency: '~15 ns', bandwidth: '~2 TB/s', color: '#8b5cf6' },
  { name: 'HBM3 / DRAM', size: '80-141 GB', latency: '~80 ns', bandwidth: '2 TB/s', color: '#f97316' },
  { name: 'SSD / NVMe', size: '1-8 TB', latency: '~100 μs', bandwidth: '7 GB/s', color: '#ef4444' },
]

const animSteps: AnimationStep[] = [
  { id: 'reg', title: 'Register File', description: 'Fastest storage — operands for ALU, ~10KB total across all registers', duration: 2500, highlightElements: ['level-0'], animateArrows: [] },
  { id: 'l1', title: 'L1 Cache Hit', description: 'Per-core, 48KB, ~1ns access. Hot data stays here for repeated access', duration: 2500, highlightElements: ['level-1'], animateArrows: ['flow-0'] },
  { id: 'l2', title: 'L2 Cache Hit', description: 'Per-core, 1-2MB, ~5ns. Activations and small weight tiles', duration: 2500, highlightElements: ['level-2'], animateArrows: ['flow-1'] },
  { id: 'l3', title: 'L3 / Last-Level Cache', description: 'Shared across cores, 32-96MB, ~15ns. KV-cache for short sequences', duration: 2500, highlightElements: ['level-3'], animateArrows: ['flow-2'] },
  { id: 'dram', title: 'HBM3 / DRAM — The Bottleneck', description: 'Model weights live here. 80-141GB, ~80ns, 2TB/s. This is THE bottleneck for decode.', duration: 3000, highlightElements: ['level-4'], animateArrows: ['flow-3'] },
  { id: 'ssd', title: 'SSD / Host Memory', description: 'Offloading target. 7 GB/s PCIe — 285× slower than HBM3. Avoid at all costs.', duration: 2500, highlightElements: ['level-5'], animateArrows: ['flow-4'] },
]

function MemoryHierarchyInner({ levels = defaultLevels, title }: MemoryHierarchyDiagramProps) {
  const colors = useDiagramTheme()
  const { tooltip, showTooltip, moveTooltip, hideTooltip } = useDiagramTooltip()
  const { currentStep, isPlaying, activeElements, activeArrows, togglePlayPause, goToStep } =
    usePipelineAnimation(animSteps)

  const W = 750
  const H = 480
  const pyramidTop = 35
  const pyramidBottom = H - 60
  const pyramidHeight = pyramidBottom - pyramidTop
  const layerH = pyramidHeight / levels.length

  // Trapezoid: narrow at top, wide at bottom
  const maxW = 550
  const minW = 140

  return (
    <>
      <div className="my-8 rounded-xl border border-border-primary overflow-hidden">
        <div className="px-4 py-2 border-b border-border-primary bg-bg-surface/50">
          <span className="text-xs font-semibold text-text-tertiary uppercase tracking-wider">
            {title || 'Memory Hierarchy — Size vs Speed Tradeoff'}
          </span>
        </div>

        <DiagramCanvas width={W} height={H} zoomable={false} className="!my-0 !rounded-none !border-0">
          <defs>
            <filter id="glow">
              <feGaussianBlur stdDeviation="3" result="coloredBlur" />
              <feMerge>
                <feMergeNode in="coloredBlur" />
                <feMergeNode in="SourceGraphic" />
              </feMerge>
            </filter>
          </defs>

          {/* Speed / Size arrows on sides */}
          <g>
            <text x={30} y={pyramidTop + pyramidHeight * 0.3} fill={colors.success} fontSize={10} fontWeight={700} textAnchor="middle" fontFamily="Inter, system-ui, sans-serif"
              transform={`rotate(-90, 30, ${pyramidTop + pyramidHeight / 2})`}>
              ← FASTER
            </text>
            <text x={W - 30} y={pyramidTop + pyramidHeight * 0.7} fill={colors.memory} fontSize={10} fontWeight={700} textAnchor="middle" fontFamily="Inter, system-ui, sans-serif"
              transform={`rotate(90, ${W - 30}, ${pyramidTop + pyramidHeight / 2})`}>
              ← LARGER
            </text>
          </g>

          {/* Pyramid layers */}
          {levels.map((level, i) => {
            const progress = i / (levels.length - 1)
            const w = minW + (maxW - minW) * progress
            const x = (W - w) / 2
            const y = pyramidTop + i * layerH
            const isActive = activeElements.has(`level-${i}`)
            const isDimmed = activeElements.size > 0 && !isActive

            return (
              <motion.g
                key={level.name}
                id={`level-${i}`}
                initial={false}
                animate={{ opacity: isDimmed ? 0.25 : 1 }}
                transition={{ duration: 0.3 }}
              >
                {/* Trapezoid shape */}
                <motion.path
                  d={(() => {
                    const nextProgress = (i + 1) / (levels.length - 1)
                    const nextW = i < levels.length - 1 ? minW + (maxW - minW) * nextProgress : w + 30
                    const nextX = (W - nextW) / 2
                    return `M${x},${y} L${x + w},${y} L${nextX + nextW},${y + layerH} L${nextX},${y + layerH} Z`
                  })()}
                  fill={level.color}
                  fillOpacity={isActive ? 0.3 : 0.12}
                  stroke={level.color}
                  strokeWidth={isActive ? 2 : 1}
                  strokeOpacity={isActive ? 0.9 : 0.4}
                  style={{ cursor: 'pointer' }}
                  onMouseEnter={(e: React.MouseEvent) =>
                    showTooltip(e, (
                      <div>
                        <div className="font-semibold" style={{ color: level.color }}>{level.name}</div>
                        <div>Size: {level.size}</div>
                        <div>Latency: {level.latency}</div>
                        <div>Bandwidth: {level.bandwidth}</div>
                      </div>
                    ))
                  }
                  onMouseMove={(e: React.MouseEvent) => moveTooltip(e)}
                  onMouseLeave={hideTooltip}
                />

                {/* Level label */}
                <text
                  x={W / 2}
                  y={y + layerH / 2 - 6}
                  textAnchor="middle"
                  dominantBaseline="central"
                  fill={colors.text}
                  fontSize={12}
                  fontWeight={600}
                  fontFamily="Inter, system-ui, sans-serif"
                >
                  {level.name}
                </text>

                {/* Stats line */}
                <text
                  x={W / 2}
                  y={y + layerH / 2 + 10}
                  textAnchor="middle"
                  dominantBaseline="central"
                  fill={colors.textLight}
                  fontSize={9}
                  fontFamily="JetBrains Mono, monospace"
                >
                  {level.size} · {level.latency} · {level.bandwidth}
                </text>

                {/* Animated flow between levels */}
                {i < levels.length - 1 && activeArrows.has(`flow-${i}`) && (
                  <>
                    {[0, 1, 2].map((j) => (
                      <motion.circle
                        key={j}
                        r={4}
                        fill={level.color}
                        filter="url(#glow)"
                        initial={{ opacity: 0 }}
                        animate={{
                          cy: [y + layerH, y + layerH * 2],
                          cx: [W / 2 - 20 + j * 20, W / 2 - 20 + j * 20],
                          opacity: [0, 0.8, 0.8, 0],
                        }}
                        transition={{
                          duration: 1,
                          repeat: Infinity,
                          delay: j * 0.3,
                          ease: 'linear',
                        }}
                      />
                    ))}
                  </>
                )}
              </motion.g>
            )
          })}

          {/* Bottom label */}
          <text x={W / 2} y={H - 15} textAnchor="middle" fill={colors.textLight} fontSize={10} fontFamily="Inter, system-ui, sans-serif">
            LLM Inference bottleneck: 140GB model weights loaded from HBM3 every decode step
          </text>
        </DiagramCanvas>

        <AnimationController
          steps={animSteps}
          currentStep={currentStep}
          isPlaying={isPlaying}
          onStepChange={goToStep}
          onPlayPause={togglePlayPause}
        />
      </div>
      <DiagramTooltipPortal tooltip={tooltip} />
    </>
  )
}

export function MemoryHierarchyDiagram(props: MemoryHierarchyDiagramProps) {
  return (
    <DiagramThemeProvider>
      <MemoryHierarchyInner {...props} />
    </DiagramThemeProvider>
  )
}
