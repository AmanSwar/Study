'use client'

import { useState } from 'react'
import { DiagramCanvas } from '../core/DiagramCanvas'
import { DiagramThemeProvider, useDiagramTheme } from '../core/DiagramTheme'
import { DiagramTooltipPortal, useDiagramTooltip } from '../core/DiagramTooltip'

interface HardwareRoofline {
  name: string
  peakFlops: number   // TFLOPS
  bandwidth: number   // GB/s
  color: string
}

interface WorkloadPoint {
  name: string
  intensity: number     // FLOPs/byte
  phase: 'prefill' | 'decode'
}

interface RooflinePlotProps {
  hardware: HardwareRoofline[]
  workloads: WorkloadPoint[]
}

function RooflinePlotInner({ hardware, workloads }: RooflinePlotProps) {
  const colors = useDiagramTheme()
  const { tooltip, showTooltip, moveTooltip, hideTooltip } = useDiagramTooltip()
  const [hoveredHw, setHoveredHw] = useState<string | null>(null)

  // Plot dimensions
  const W = 800
  const H = 450
  const pad = { top: 40, right: 30, bottom: 70, left: 70 }
  const plotW = W - pad.left - pad.right
  const plotH = H - pad.top - pad.bottom

  // Log scale ranges
  const minI = 0.1
  const maxI = 10000
  const minP = 0.01
  const maxP = 200

  const logScale = (val: number, min: number, max: number, size: number) => {
    const logMin = Math.log10(min)
    const logMax = Math.log10(max)
    return ((Math.log10(val) - logMin) / (logMax - logMin)) * size
  }

  const toX = (intensity: number) => pad.left + logScale(intensity, minI, maxI, plotW)
  const toY = (perf: number) => pad.top + plotH - logScale(perf, minP, maxP, plotH)

  // Grid lines
  const xTicks = [0.1, 1, 10, 100, 1000, 10000]
  const yTicks = [0.01, 0.1, 1, 10, 100]

  // Calculate roofline performance
  const rooflinePerf = (hw: HardwareRoofline, intensity: number) => {
    const memBound = (intensity * hw.bandwidth) / 1000 // TFLOPS
    return Math.min(hw.peakFlops, memBound)
  }

  // Ridge point
  const ridgePoint = (hw: HardwareRoofline) => (hw.peakFlops * 1000) / hw.bandwidth

  return (
    <>
      <DiagramCanvas width={W} height={H} title="Interactive Roofline Model — Log-Log Scale" zoomable={false}>
        {/* SVG defs for glow */}
        <defs>
          <filter id="glow">
            <feGaussianBlur stdDeviation="3" result="coloredBlur" />
            <feMerge>
              <feMergeNode in="coloredBlur" />
              <feMergeNode in="SourceGraphic" />
            </feMerge>
          </filter>
          <filter id="pointGlow">
            <feGaussianBlur stdDeviation="4" result="coloredBlur" />
            <feMerge>
              <feMergeNode in="coloredBlur" />
              <feMergeNode in="SourceGraphic" />
            </feMerge>
          </filter>
        </defs>

        {/* Background */}
        <rect x={pad.left} y={pad.top} width={plotW} height={plotH} fill={colors.grid} fillOpacity={0.3} rx={4} />

        {/* Grid lines */}
        {xTicks.map((v) => (
          <g key={`xg-${v}`}>
            <line x1={toX(v)} y1={pad.top} x2={toX(v)} y2={pad.top + plotH} stroke={colors.grid} strokeWidth={1} strokeOpacity={0.5} />
            <text x={toX(v)} y={pad.top + plotH + 18} textAnchor="middle" fill={colors.textLight} fontSize={10} fontFamily="JetBrains Mono, monospace">
              {v >= 1 ? v : v}
            </text>
          </g>
        ))}
        {yTicks.map((v) => (
          <g key={`yg-${v}`}>
            <line x1={pad.left} y1={toY(v)} x2={pad.left + plotW} y2={toY(v)} stroke={colors.grid} strokeWidth={1} strokeOpacity={0.5} />
            <text x={pad.left - 10} y={toY(v)} textAnchor="end" dominantBaseline="central" fill={colors.textLight} fontSize={10} fontFamily="JetBrains Mono, monospace">
              {v >= 1 ? v : v}
            </text>
          </g>
        ))}

        {/* Axis labels */}
        <text x={pad.left + plotW / 2} y={H - 10} textAnchor="middle" fill={colors.text} fontSize={12} fontWeight={600} fontFamily="Inter, system-ui, sans-serif">
          Arithmetic Intensity (FLOPs/byte)
        </text>
        <text x={16} y={pad.top + plotH / 2} textAnchor="middle" dominantBaseline="central" fill={colors.text} fontSize={12} fontWeight={600} fontFamily="Inter, system-ui, sans-serif"
          transform={`rotate(-90, 16, ${pad.top + plotH / 2})`}>
          Performance (TFLOPS)
        </text>

        {/* Memory-bound / Compute-bound labels */}
        <text x={toX(0.5)} y={pad.top + 16} textAnchor="middle" fill={colors.memory} fontSize={10} fontWeight={600} fontFamily="Inter, system-ui, sans-serif" opacity={0.7}>
          MEMORY-BOUND
        </text>
        <text x={toX(3000)} y={pad.top + 16} textAnchor="middle" fill={colors.compute} fontSize={10} fontWeight={600} fontFamily="Inter, system-ui, sans-serif" opacity={0.7}>
          COMPUTE-BOUND
        </text>

        {/* Rooflines for each hardware */}
        {hardware.map((hw) => {
          const ridge = ridgePoint(hw)
          const isHovered = hoveredHw === hw.name
          const lineOpacity = hoveredHw === null ? 0.8 : isHovered ? 1 : 0.2

          // Build roofline path: bandwidth slope up to ridge, then flat at peak
          const pts: string[] = []
          // Start from leftmost visible point
          const startI = minI
          const startPerf = rooflinePerf(hw, startI)
          pts.push(`${toX(startI)},${toY(startPerf)}`)

          // Ridge point
          pts.push(`${toX(ridge)},${toY(hw.peakFlops)}`)

          // End at rightmost
          pts.push(`${toX(maxI)},${toY(hw.peakFlops)}`)

          return (
            <g key={hw.name}>
              {/* Roofline path */}
              <polyline
                points={pts.join(' ')}
                fill="none"
                stroke={hw.color}
                strokeWidth={isHovered ? 3 : 2}
                strokeOpacity={lineOpacity}
                strokeLinecap="round"
                strokeLinejoin="round"
              />

              {/* Ridge point marker */}
              <circle
                cx={toX(ridge)}
                cy={toY(hw.peakFlops)}
                r={isHovered ? 5 : 3}
                fill={hw.color}
                fillOpacity={lineOpacity}
              />

              {/* Hardware label */}
              <text
                x={toX(maxI) - 5}
                y={toY(hw.peakFlops) - 8}
                textAnchor="end"
                fill={hw.color}
                fontSize={10}
                fontWeight={600}
                fontFamily="Inter, system-ui, sans-serif"
                opacity={lineOpacity}
                onMouseEnter={(e) => {
                  setHoveredHw(hw.name)
                  showTooltip(e, (
                    <div>
                      <div className="font-semibold mb-1">{hw.name}</div>
                      <div>Peak: {hw.peakFlops} TFLOPS (FP16)</div>
                      <div>Bandwidth: {hw.bandwidth} GB/s</div>
                      <div>Ridge: {ridge.toFixed(1)} FLOPs/byte</div>
                    </div>
                  ))
                }}
                onMouseMove={moveTooltip}
                onMouseLeave={() => { setHoveredHw(null); hideTooltip() }}
                style={{ cursor: 'pointer' }}
              >
                {hw.name}
              </text>
            </g>
          )
        })}

        {/* Workload points */}
        {workloads.map((wl, i) => {
          // Find best hardware performance for this workload
          const bestHw = hardware[0]
          const perf = rooflinePerf(bestHw, wl.intensity)
          const color = wl.phase === 'prefill' ? colors.compute : colors.memory
          const px = toX(wl.intensity)
          const py = toY(perf)

          return (
            <g key={wl.name}>
              {/* Glow */}
              <circle cx={px} cy={py} r={10} fill={color} fillOpacity={0.15} filter="url(#pointGlow)" />
              {/* Point */}
              <circle
                cx={px}
                cy={py}
                r={6}
                fill={color}
                stroke={colors.bg}
                strokeWidth={2}
                style={{ cursor: 'pointer' }}
                onMouseEnter={(e) =>
                  showTooltip(e, (
                    <div>
                      <div className="font-semibold">{wl.name}</div>
                      <div>Intensity: {wl.intensity} FLOPs/byte</div>
                      <div>Phase: {wl.phase}</div>
                      <div>Perf on {bestHw.name}: {perf.toFixed(2)} TFLOPS</div>
                    </div>
                  ))
                }
                onMouseMove={moveTooltip}
                onMouseLeave={hideTooltip}
              />
              {/* Label */}
              <text
                x={px}
                y={py - 14}
                textAnchor="middle"
                fill={color}
                fontSize={9}
                fontWeight={600}
                fontFamily="Inter, system-ui, sans-serif"
              >
                {wl.name}
              </text>
            </g>
          )
        })}
      </DiagramCanvas>
      <DiagramTooltipPortal tooltip={tooltip} />
    </>
  )
}

export function RooflinePlot(props: RooflinePlotProps) {
  return (
    <DiagramThemeProvider>
      <RooflinePlotInner {...props} />
    </DiagramThemeProvider>
  )
}
