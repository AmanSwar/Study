'use client'

import { useMemo, useState } from 'react'
import { VizFrame } from './VizFrame'

const W = 740
const H = 400
const PAD = { l: 56, r: 24, t: 28, b: 50 }

// Marchenko–Pastur density for ratio q = N/T (assets / observations).
// Support: [(1-sqrt(q))^2, (1+sqrt(q))^2] for sample eigenvalues of pure noise.
function mpDensity(lambda: number, q: number) {
  const a = (1 - Math.sqrt(q)) ** 2
  const b = (1 + Math.sqrt(q)) ** 2
  if (lambda < a || lambda > b) return 0
  return Math.sqrt((b - lambda) * (lambda - a)) / (2 * Math.PI * q * lambda)
}

// Build a synthetic eigenvalue spectrum: N noise eigenvalues drawn from MP +
// k "signal" eigenvalues that stick out above the bulk.
function buildSpectrum(N: number, T: number, signals: number[]) {
  const q = N / T
  const a = (1 - Math.sqrt(q)) ** 2
  const b = (1 + Math.sqrt(q)) ** 2
  const eigs: number[] = []
  // Inverse-CDF sample from MP via rejection on a deterministic grid (no RNG needed)
  const noiseCount = N - signals.length
  for (let i = 0; i < noiseCount; i++) {
    const u = (i + 0.5) / noiseCount
    // Linear interp across [a,b], pulled toward density mass — quick approx.
    const lambda = a + (b - a) * (1 - Math.cos(u * Math.PI)) / 2
    eigs.push(lambda)
  }
  for (const s of signals) eigs.push(s)
  return eigs.sort((x, y) => y - x)
}

const SIGNALS = [9.4, 3.2, 2.1]
const N_ASSETS = 60
const T_OBS = 200
const Q = N_ASSETS / T_OBS

export function CovarianceEigenvalues() {
  const [showDenoised, setShowDenoised] = useState(true)
  const sample = useMemo(() => buildSpectrum(N_ASSETS, T_OBS, SIGNALS), [])
  const a = (1 - Math.sqrt(Q)) ** 2
  const b = (1 + Math.sqrt(Q)) ** 2

  // x range covers the bulk + signal eigenvalues
  const xMax = Math.max(...sample) * 1.05
  const xMin = 0
  // Histogram bins for the bulk
  const NBINS = 28
  const hist = useMemo(() => {
    const bins = Array.from({ length: NBINS }, () => 0)
    const binW = (b - a) / NBINS
    for (const e of sample) {
      if (e >= a && e <= b) {
        const idx = Math.min(NBINS - 1, Math.floor((e - a) / binW))
        bins[idx]++
      }
    }
    const total = bins.reduce((p, c) => p + c, 0)
    return bins.map((count, i) => ({
      lo: a + i * binW,
      hi: a + (i + 1) * binW,
      density: total > 0 ? (count / total) / binW : 0,
    }))
  }, [sample, a, b])

  // MP density curve, sampled
  const mpCurve = useMemo(() => {
    const pts: { x: number; y: number }[] = []
    const STEPS = 120
    for (let i = 0; i <= STEPS; i++) {
      const lambda = a + (b - a) * (i / STEPS)
      pts.push({ x: lambda, y: mpDensity(lambda, Q) })
    }
    return pts
  }, [a, b])

  const yMax = Math.max(...mpCurve.map((p) => p.y), ...hist.map((h) => h.density)) * 1.15

  const xScale = (v: number) => PAD.l + ((v - xMin) / (xMax - xMin)) * (W - PAD.l - PAD.r)
  const yScale = (v: number) => H - PAD.b - (v / yMax) * (H - PAD.t - PAD.b)
  const barW = (xScale(b) - xScale(a)) / NBINS - 1

  const xTicks: number[] = []
  for (let v = 0; v <= xMax; v += xMax > 8 ? 2 : 1) xTicks.push(v)

  return (
    <VizFrame
      title="Eigenvalue spectrum vs Marchenko–Pastur (N=60 assets, T=200 obs)"
      caption="Most eigenvalues of a sample covariance matrix lie inside the Marchenko–Pastur bulk — pure noise from finite-sample estimation. The eigenvalues that punch through the upper edge are the real signal (market factor, sector factors). PCA denoising clips the bulk back to its mean and keeps the spikes."
    >
      <div className="flex items-center gap-3 mb-3">
        <label className="flex items-center gap-1.5 text-xs text-text-secondary cursor-pointer">
          <input type="checkbox" checked={showDenoised} onChange={(e) => setShowDenoised(e.target.checked)} className="accent-accent-blue" />
          Show PCA-denoised spectrum
        </label>
      </div>

      <svg viewBox={`0 0 ${W} ${H}`} width="100%" style={{ maxWidth: W, height: 'auto' }} role="img">
        {/* MP support shading */}
        <rect x={xScale(a)} y={PAD.t} width={xScale(b) - xScale(a)} height={H - PAD.t - PAD.b} fill="var(--accent-blue)" opacity={0.06} />
        <text x={(xScale(a) + xScale(b)) / 2} y={PAD.t + 14} textAnchor="middle" fontSize={11} fontWeight={600} fill="var(--accent-blue)">
          MP bulk: noise eigenvalues
        </text>

        {/* x grid + ticks */}
        {xTicks.map((t) => (
          <g key={`tx${t}`}>
            <line x1={xScale(t)} x2={xScale(t)} y1={PAD.t} y2={H - PAD.b} stroke="var(--border-subtle)" strokeDasharray="2 4" />
            <text x={xScale(t)} y={H - PAD.b + 16} textAnchor="middle" fontSize={11} fill="var(--text-tertiary)">{t}</text>
          </g>
        ))}

        {/* histogram bars */}
        {hist.map((h, i) => (
          <rect
            key={i}
            x={xScale(h.lo) + 0.5}
            y={yScale(h.density)}
            width={Math.max(2, barW)}
            height={H - PAD.b - yScale(h.density)}
            fill="var(--accent-cyan)"
            opacity={0.62}
          />
        ))}

        {/* MP density curve */}
        <path
          d={'M ' + mpCurve.map((p) => `${xScale(p.x)},${yScale(p.y)}`).join(' L ')}
          fill="none" stroke="var(--accent-blue)" strokeWidth={2.5}
        />

        {/* signal eigenvalues — vertical sticks */}
        {SIGNALS.map((s, i) => (
          <g key={i}>
            <line
              x1={xScale(s)} x2={xScale(s)}
              y1={yScale(0)} y2={yScale(yMax * 0.85)}
              stroke="var(--accent-orange)" strokeWidth={2.5}
            />
            <circle cx={xScale(s)} cy={yScale(yMax * 0.85)} r={5} fill="var(--accent-orange)" />
            <text x={xScale(s)} y={yScale(yMax * 0.85) - 10} textAnchor="middle" fontSize={11} fontWeight={600} fill="var(--accent-orange)">
              λ = {s.toFixed(1)}
            </text>
          </g>
        ))}

        {/* denoised: replace bulk with mean line */}
        {showDenoised && (
          <line
            x1={xScale(a)} x2={xScale(b)}
            y1={yScale(0.04)} y2={yScale(0.04)}
            stroke="var(--accent-green)" strokeWidth={4} strokeLinecap="round"
          />
        )}
        {showDenoised && (
          <text x={xScale((a + b) / 2)} y={yScale(0.04) + 18} textAnchor="middle" fontSize={11} fontWeight={600} fill="var(--accent-green)">
            ↓ flatten bulk to mean λ̄
          </text>
        )}

        {/* axis labels */}
        <text x={W / 2} y={H - 8} textAnchor="middle" fontSize={12} fontWeight={600} fill="var(--text-secondary)">
          Eigenvalue λ
        </text>
        <text x={14} y={H / 2} textAnchor="middle" fontSize={12} fontWeight={600} fill="var(--text-secondary)" transform={`rotate(-90 14 ${H / 2})`}>
          Density
        </text>
      </svg>

      <div className="mt-3 flex flex-wrap gap-x-5 gap-y-1.5 text-[11px] text-text-tertiary">
        <span><span className="inline-block w-3 h-2 align-middle mr-1.5" style={{ backgroundColor: 'var(--accent-cyan)', opacity: 0.62 }} />Sample histogram</span>
        <span><span className="inline-block w-3 h-[2px] bg-accent-blue align-middle mr-1.5" />Marchenko–Pastur theory</span>
        <span><span className="inline-block w-3 h-[2px] bg-accent-orange align-middle mr-1.5" />Signal eigenvalues</span>
        {showDenoised && (
          <span><span className="inline-block w-3 h-[2px] bg-accent-green align-middle mr-1.5" />Denoised (bulk flattened)</span>
        )}
      </div>
    </VizFrame>
  )
}
