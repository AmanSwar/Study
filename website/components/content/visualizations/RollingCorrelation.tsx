'use client'

import { useMemo, useState } from 'react'
import { VizFrame } from './VizFrame'

const W = 760
const H = 380
const PAD = { l: 56, r: 24, t: 24, b: 50 }

// Months from Jan 2007 to Dec 2024 (216 points). We synthesise a plausible
// rolling-correlation series between two large-cap stocks (e.g. HDFC Bank ↔
// Reliance) — calm in normal regimes, spiking toward 1 during global stress.
function buildSeries() {
  const months: { date: Date; corr: number }[] = []
  const start = new Date(2007, 0, 1)
  for (let i = 0; i < 216; i++) {
    const d = new Date(start.getFullYear(), start.getMonth() + i, 1)
    let base = 0.45 + 0.18 * Math.sin(i / 11) + 0.05 * Math.cos(i / 4)
    // Crisis periods bump correlation hard
    if (i >= 20 && i <= 30) base = 0.85 + 0.05 * Math.sin(i)         // GFC 2008–09
    if (i >= 60 && i <= 66) base = 0.72                              // Eurozone 2012
    if (i >= 110 && i <= 118) base = 0.78                            // Aug 2015 / China crash
    if (i >= 158 && i <= 167) base = 0.92                            // COVID Mar 2020
    if (i >= 188 && i <= 196) base = 0.74                            // Russia/Ukraine + rates 2022
    // Mild noise, clamp
    base += (Math.sin(i * 1.7) + Math.cos(i * 2.3)) * 0.04
    months.push({ date: d, corr: Math.max(-0.1, Math.min(0.99, base)) })
  }
  return months
}

const CRISES = [
  { start: 19, end: 30, label: 'GFC' },
  { start: 158, end: 167, label: 'COVID' },
  { start: 188, end: 196, label: '2022 rates' },
]

export function RollingCorrelation() {
  const series = useMemo(buildSeries, [])
  const [hover, setHover] = useState<{ idx: number } | null>(null)

  const xScale = (i: number) => PAD.l + (i / (series.length - 1)) * (W - PAD.l - PAD.r)
  const yScale = (v: number) => H - PAD.b - ((v + 0.2) / 1.2) * (H - PAD.t - PAD.b)

  const yTicks = [-0.2, 0, 0.2, 0.4, 0.6, 0.8, 1.0]
  const yearTicks: number[] = []
  for (let i = 0; i < series.length; i += 24) yearTicks.push(i)

  const path = 'M ' + series.map((p, i) => `${xScale(i)},${yScale(p.corr)}`).join(' L ')

  // Mean line
  const mean = series.reduce((a, b) => a + b.corr, 0) / series.length

  return (
    <VizFrame
      title="60-day rolling correlation: HDFC Bank vs Reliance"
      caption="In normal regimes correlation drifts around 0.4–0.5. During global stress it jumps toward 1.0 — diversification evaporates exactly when you need it. Treating correlation as a constant ignores this."
    >
      <svg viewBox={`0 0 ${W} ${H}`} width="100%" style={{ maxWidth: W, height: 'auto' }} role="img"
           onMouseLeave={() => setHover(null)}
           onMouseMove={(e) => {
             const rect = (e.currentTarget as SVGSVGElement).getBoundingClientRect()
             const xRatio = (e.clientX - rect.left) / rect.width
             const xPx = xRatio * W
             const idx = Math.round(((xPx - PAD.l) / (W - PAD.l - PAD.r)) * (series.length - 1))
             if (idx >= 0 && idx < series.length) setHover({ idx })
           }}
      >
        {/* crisis bands */}
        {CRISES.map((c) => (
          <g key={c.label}>
            <rect
              x={xScale(c.start)} y={PAD.t}
              width={xScale(c.end) - xScale(c.start)}
              height={H - PAD.t - PAD.b}
              fill="var(--accent-red)" opacity={0.08}
            />
            <text x={(xScale(c.start) + xScale(c.end)) / 2} y={PAD.t + 12} textAnchor="middle" fontSize={10.5} fontWeight={600} fill="var(--accent-red)" opacity={0.9}>
              {c.label}
            </text>
          </g>
        ))}

        {/* y grid + labels */}
        {yTicks.map((t) => (
          <g key={t}>
            <line x1={PAD.l} x2={W - PAD.r} y1={yScale(t)} y2={yScale(t)} stroke="var(--border-subtle)" strokeDasharray="2 4" />
            <text x={PAD.l - 8} y={yScale(t) + 4} textAnchor="end" fontSize={11} fill="var(--text-tertiary)">{t.toFixed(1)}</text>
          </g>
        ))}

        {/* x year labels */}
        {yearTicks.map((i) => (
          <text key={i} x={xScale(i)} y={H - PAD.b + 16} textAnchor="middle" fontSize={11} fill="var(--text-tertiary)">
            {series[i].date.getFullYear()}
          </text>
        ))}

        {/* zero axis */}
        <line x1={PAD.l} x2={W - PAD.r} y1={yScale(0)} y2={yScale(0)} stroke="var(--text-tertiary)" strokeWidth={1} />

        {/* mean line */}
        <line x1={PAD.l} x2={W - PAD.r} y1={yScale(mean)} y2={yScale(mean)} stroke="var(--accent-orange)" strokeDasharray="5 4" strokeWidth={1.5} />
        <text x={W - PAD.r - 4} y={yScale(mean) - 5} textAnchor="end" fontSize={10.5} fontWeight={600} fill="var(--accent-orange)">
          Mean = {mean.toFixed(2)}
        </text>

        {/* main series */}
        <path d={path} fill="none" stroke="var(--accent-cyan)" strokeWidth={2} />

        {/* hover crosshair */}
        {hover && (
          <g pointerEvents="none">
            <line x1={xScale(hover.idx)} x2={xScale(hover.idx)} y1={PAD.t} y2={H - PAD.b} stroke="var(--text-tertiary)" strokeDasharray="3 3" />
            <circle cx={xScale(hover.idx)} cy={yScale(series[hover.idx].corr)} r={5} fill="var(--accent-cyan)" stroke="var(--bg-code)" strokeWidth={2} />
            <g transform={`translate(${Math.min(xScale(hover.idx) + 10, W - 140)}, ${PAD.t + 8})`}>
              <rect width={130} height={36} rx={4} fill="var(--bg-surface)" stroke="var(--border-primary)" />
              <text x={8} y={15} fontSize={11} fontWeight={600} fill="var(--text-primary)">
                {series[hover.idx].date.toLocaleDateString('en-US', { year: 'numeric', month: 'short' })}
              </text>
              <text x={8} y={29} fontSize={11} fill="var(--text-secondary)">
                ρ = {series[hover.idx].corr.toFixed(2)}
              </text>
            </g>
          </g>
        )}

        {/* axis labels */}
        <text x={W / 2} y={H - 8} textAnchor="middle" fontSize={12} fontWeight={600} fill="var(--text-secondary)">
          Date
        </text>
        <text x={14} y={H / 2} textAnchor="middle" fontSize={12} fontWeight={600} fill="var(--text-secondary)" transform={`rotate(-90 14 ${H / 2})`}>
          60-day rolling ρ
        </text>
      </svg>

      <div className="mt-3 grid grid-cols-2 sm:grid-cols-4 gap-2 text-[11px]">
        {[
          { label: 'Min', val: Math.min(...series.map((s) => s.corr)).toFixed(2), color: 'text-accent-blue' },
          { label: 'Max', val: Math.max(...series.map((s) => s.corr)).toFixed(2), color: 'text-accent-red' },
          { label: 'Mean', val: mean.toFixed(2), color: 'text-accent-orange' },
          { label: 'σ', val: Math.sqrt(series.reduce((a, b) => a + (b.corr - mean) ** 2, 0) / series.length).toFixed(2), color: 'text-text-secondary' },
        ].map((s) => (
          <div key={s.label} className="rounded-md border border-border-primary bg-bg-surface/40 px-3 py-2">
            <div className="text-text-tertiary uppercase tracking-wider text-[9px]">{s.label}</div>
            <div className={`font-mono font-semibold ${s.color}`}>{s.val}</div>
          </div>
        ))}
      </div>
    </VizFrame>
  )
}
