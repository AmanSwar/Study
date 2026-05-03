'use client'

import { useMemo } from 'react'
import { VizFrame } from './VizFrame'

const W = 720
const H = 420
const PAD = { l: 60, r: 24, t: 24, b: 50 }

// Mulberry32 — deterministic seeded PRNG so the chart is reproducible.
function mulberry32(seed: number) {
  let t = seed
  return () => {
    t |= 0
    t = (t + 0x6D2B79F5) | 0
    let r = Math.imul(t ^ (t >>> 15), 1 | t)
    r = (r + Math.imul(r ^ (r >>> 7), 61 | r)) ^ r
    return ((r ^ (r >>> 14)) >>> 0) / 4294967296
  }
}

// Box–Muller for one normal sample
function normalSample(rand: () => number) {
  const u1 = Math.max(rand(), 1e-12)
  const u2 = rand()
  return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2)
}

// Inverse standard normal CDF (Acklam approximation)
function probit(p: number) {
  const a = [-3.969683028665376e1, 2.209460984245205e2, -2.759285104469687e2, 1.383577518672690e2, -3.066479806614716e1, 2.506628277459239]
  const b = [-5.447609879822406e1, 1.615858368580409e2, -1.556989798598866e2, 6.680131188771972e1, -1.328068155288572e1]
  const c = [-7.784894002430293e-3, -3.223964580411365e-1, -2.400758277161838, -2.549732539343734, 4.374664141464968, 2.938163982698783]
  const d = [7.784695709041462e-3, 3.224671290700398e-1, 2.445134137142996, 3.754408661907416]
  const pl = 0.02425
  const ph = 1 - pl
  let q: number, r: number
  if (p < pl) {
    q = Math.sqrt(-2 * Math.log(p))
    return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
           ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
  }
  if (p <= ph) {
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q /
           (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)
  }
  q = Math.sqrt(-2 * Math.log(1 - p))
  return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
         ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
}

export function QQPlot() {
  const { points, axMax } = useMemo(() => {
    const N = 500
    const rand = mulberry32(20240517)
    // Stock returns ~ Student-t(df=4) — fat tails. Approx by mixture: normal + occasional shock.
    const samples: number[] = []
    for (let i = 0; i < N; i++) {
      let x = normalSample(rand)
      // Tail shock 6% of the time
      if (rand() < 0.06) x *= 3 + rand() * 2
      samples.push(x)
    }
    samples.sort((a, b) => a - b)
    const pts: { x: number; y: number; outlier: boolean }[] = samples.map((y, i) => {
      const p = (i + 0.5) / N
      const x = probit(p)
      return { x, y, outlier: Math.abs(y) > Math.abs(x) * 1.5 + 1.0 }
    })
    const m = Math.max(Math.abs(pts[0].y), Math.abs(pts[N - 1].y), 4)
    return { points: pts, axMax: Math.ceil(m) }
  }, [])

  const xScale = (v: number) => PAD.l + ((v + axMax) / (2 * axMax)) * (W - PAD.l - PAD.r)
  const yScale = (v: number) => H - PAD.b - ((v + axMax) / (2 * axMax)) * (H - PAD.t - PAD.b)

  const ticks: number[] = []
  for (let v = -axMax; v <= axMax; v += axMax >= 8 ? 2 : 1) ticks.push(v)

  return (
    <VizFrame
      title="Q-Q plot — daily stock returns vs Normal"
      caption="Quantile-quantile plot. If returns were Gaussian the points would hug the dashed 45° line. Real returns curl away in both tails — markets crash and rally far more often than the normal distribution predicts."
    >
      <svg viewBox={`0 0 ${W} ${H}`} width="100%" style={{ maxWidth: W, height: 'auto' }} role="img">
        {/* shaded tail bands */}
        <rect x={PAD.l} y={PAD.t} width={xScale(-2) - PAD.l} height={H - PAD.t - PAD.b} fill="var(--accent-red)" opacity={0.05} />
        <rect x={xScale(2)} y={PAD.t} width={W - PAD.r - xScale(2)} height={H - PAD.t - PAD.b} fill="var(--accent-red)" opacity={0.05} />

        {/* grid + ticks */}
        {ticks.map((t) => (
          <g key={`tx${t}`}>
            <line x1={xScale(t)} x2={xScale(t)} y1={PAD.t} y2={H - PAD.b} stroke="var(--border-subtle)" strokeDasharray="2 4" />
            <text x={xScale(t)} y={H - PAD.b + 16} textAnchor="middle" fontSize={11} fill="var(--text-tertiary)">{t}σ</text>
          </g>
        ))}
        {ticks.map((t) => (
          <g key={`ty${t}`}>
            <line x1={PAD.l} x2={W - PAD.r} y1={yScale(t)} y2={yScale(t)} stroke="var(--border-subtle)" strokeDasharray="2 4" />
            <text x={PAD.l - 8} y={yScale(t) + 4} textAnchor="end" fontSize={11} fill="var(--text-tertiary)">{t}σ</text>
          </g>
        ))}

        {/* labels */}
        <text x={W / 2} y={H - 8} textAnchor="middle" fontSize={12} fontWeight={600} fill="var(--text-secondary)">
          Theoretical quantiles (Normal)
        </text>
        <text x={14} y={H / 2} textAnchor="middle" fontSize={12} fontWeight={600} fill="var(--text-secondary)" transform={`rotate(-90 14 ${H / 2})`}>
          Sample quantiles (returns)
        </text>

        {/* 45° reference line */}
        <line x1={xScale(-axMax)} y1={yScale(-axMax)} x2={xScale(axMax)} y2={yScale(axMax)} stroke="var(--accent-blue)" strokeDasharray="6 5" strokeWidth={1.8} />

        {/* points */}
        {points.map((p, i) => (
          <circle
            key={i}
            cx={xScale(p.x)} cy={yScale(p.y)}
            r={p.outlier ? 3.2 : 2.4}
            fill={p.outlier ? 'var(--accent-red)' : 'var(--accent-cyan)'}
            opacity={p.outlier ? 0.95 : 0.78}
          />
        ))}

        {/* annotations */}
        <g>
          <text x={xScale(-axMax + 0.4)} y={PAD.t + 14} fontSize={11} fontWeight={600} fill="var(--accent-red)">Left tail: extra crashes</text>
          <text x={xScale(axMax - 0.4)} y={PAD.t + 14} textAnchor="end" fontSize={11} fontWeight={600} fill="var(--accent-red)">Right tail: extra rallies</text>
          <text x={xScale(0)} y={yScale(0) - 8} textAnchor="middle" fontSize={11} fill="var(--accent-blue)">Normal fits well near the center</text>
        </g>
      </svg>

      <div className="mt-3 flex flex-wrap gap-x-5 gap-y-1.5 text-[11px] text-text-tertiary">
        <span><span className="inline-block w-2 h-2 rounded-full bg-accent-cyan align-middle mr-1.5" />Empirical quantiles</span>
        <span><span className="inline-block w-2 h-2 rounded-full align-middle mr-1.5" style={{ backgroundColor: 'var(--accent-red)' }} />Tail outliers (|y| ≫ |x|)</span>
        <span><span className="inline-block w-3 h-[2px] align-middle mr-1.5" style={{ background: 'repeating-linear-gradient(90deg, var(--accent-blue) 0 4px, transparent 4px 8px)' }} />y = x reference</span>
      </div>
    </VizFrame>
  )
}
