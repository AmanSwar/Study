'use client'

import { useState, useMemo } from 'react'
import { VizFrame } from './VizFrame'

const W = 720
const H = 420
const PAD = { l: 60, r: 24, t: 24, b: 48 }

const ASSETS = [
  { name: 'Bonds',    sigma: 0.04, mu: 0.045 },
  { name: 'Gold',     sigma: 0.16, mu: 0.075 },
  { name: 'NIFTY',    sigma: 0.21, mu: 0.115 },
  { name: 'IT Index', sigma: 0.27, mu: 0.135 },
  { name: 'Smallcap', sigma: 0.34, mu: 0.155 },
  { name: 'Crypto',   sigma: 0.62, mu: 0.180 },
]

const RF = 0.04

// Bullet (efficient frontier hyperbola): for a 2-fund spanning, sigma^2 = a*(mu - mu0)^2 + sigma_min^2
const SIGMA_MIN = 0.045
const MU_MIN = 0.06     // GMVP return
const FRONTIER_K = 6.0  // shape parameter

function frontierSigma(mu: number) {
  const dm = mu - MU_MIN
  return Math.sqrt(SIGMA_MIN * SIGMA_MIN + dm * dm * FRONTIER_K)
}

// Tangency portfolio: where the line from RF is tangent to the bullet
const MU_TAN = 0.13
const SIGMA_TAN = frontierSigma(MU_TAN)

const X_MAX = 0.7
const Y_MAX = 0.22
const Y_MIN = 0

const xScale = (s: number) => PAD.l + (s / X_MAX) * (W - PAD.l - PAD.r)
const yScale = (m: number) => H - PAD.b - ((m - Y_MIN) / (Y_MAX - Y_MIN)) * (H - PAD.t - PAD.b)

export function EfficientFrontier() {
  const [showCAL, setShowCAL] = useState(true)
  const [showLongOnly, setShowLongOnly] = useState(true)
  const [hover, setHover] = useState<{ x: number; y: number; label: string } | null>(null)

  const frontierPoints = useMemo(() => {
    const pts: { sigma: number; mu: number }[] = []
    for (let m = MU_MIN; m <= Y_MAX; m += 0.003) {
      pts.push({ sigma: frontierSigma(m), mu: m })
    }
    return pts
  }, [])

  const inefficientPoints = useMemo(() => {
    const pts: { sigma: number; mu: number }[] = []
    for (let m = 0.005; m <= MU_MIN; m += 0.003) {
      pts.push({ sigma: frontierSigma(m), mu: m })
    }
    return pts
  }, [])

  // Long-only constrained frontier: shifted right (higher risk for same return)
  const longOnlyFrontier = useMemo(() => {
    const pts: { sigma: number; mu: number }[] = []
    for (let m = MU_MIN + 0.005; m <= Y_MAX - 0.02; m += 0.003) {
      pts.push({ sigma: frontierSigma(m) * 1.18 + 0.01, mu: m })
    }
    return pts
  }, [])

  return (
    <VizFrame
      title="Efficient Frontier (mean-variance)"
      caption="The efficient frontier traces minimum-variance portfolios for each target return. Adding the risk-free rate and drawing the tangent gives the Capital Allocation Line: every rational investor holds a mix of cash and the tangency portfolio."
    >
      <div className="flex items-center gap-3 mb-3">
        <label className="flex items-center gap-1.5 text-xs text-text-secondary cursor-pointer">
          <input type="checkbox" checked={showCAL} onChange={(e) => setShowCAL(e.target.checked)} className="accent-accent-blue" />
          Capital Allocation Line
        </label>
        <label className="flex items-center gap-1.5 text-xs text-text-secondary cursor-pointer">
          <input type="checkbox" checked={showLongOnly} onChange={(e) => setShowLongOnly(e.target.checked)} className="accent-accent-orange" />
          Long-only constrained frontier
        </label>
      </div>

      <svg viewBox={`0 0 ${W} ${H}`} width="100%" style={{ maxWidth: W, height: 'auto' }} role="img">
        {/* grid */}
        {[0, 0.05, 0.10, 0.15, 0.20].map((y) => (
          <g key={`gy${y}`}>
            <line
              x1={PAD.l} x2={W - PAD.r}
              y1={yScale(y)} y2={yScale(y)}
              stroke="var(--border-subtle)" strokeDasharray="2 4"
            />
            <text x={PAD.l - 8} y={yScale(y) + 4} textAnchor="end" fontSize={11} fill="var(--text-tertiary)">
              {(y * 100).toFixed(0)}%
            </text>
          </g>
        ))}
        {[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7].map((x) => (
          <g key={`gx${x}`}>
            <line
              x1={xScale(x)} x2={xScale(x)}
              y1={PAD.t} y2={H - PAD.b}
              stroke="var(--border-subtle)" strokeDasharray="2 4"
            />
            <text x={xScale(x)} y={H - PAD.b + 16} textAnchor="middle" fontSize={11} fill="var(--text-tertiary)">
              {(x * 100).toFixed(0)}%
            </text>
          </g>
        ))}

        {/* axis labels */}
        <text x={W / 2} y={H - 8} textAnchor="middle" fontSize={12} fontWeight={600} fill="var(--text-secondary)">
          Volatility (σ, annualised)
        </text>
        <text x={14} y={H / 2} textAnchor="middle" fontSize={12} fontWeight={600} fill="var(--text-secondary)" transform={`rotate(-90 14 ${H / 2})`}>
          Expected return (μ, annualised)
        </text>

        {/* inefficient half (lower) */}
        <path
          d={'M ' + inefficientPoints.map((p) => `${xScale(p.sigma)},${yScale(p.mu)}`).join(' L ')}
          fill="none" stroke="var(--text-tertiary)" strokeWidth={1.5} strokeDasharray="3 4" opacity={0.6}
        />

        {/* efficient frontier */}
        <path
          d={'M ' + frontierPoints.map((p) => `${xScale(p.sigma)},${yScale(p.mu)}`).join(' L ')}
          fill="none" stroke="var(--accent-blue)" strokeWidth={2.5}
        />

        {/* long-only constrained frontier */}
        {showLongOnly && (
          <path
            d={'M ' + longOnlyFrontier.map((p) => `${xScale(p.sigma)},${yScale(p.mu)}`).join(' L ')}
            fill="none" stroke="var(--accent-orange)" strokeWidth={2} strokeDasharray="6 4"
          />
        )}

        {/* CAL: line from (0, RF) through tangency portfolio, extended */}
        {showCAL && (
          <>
            <line
              x1={xScale(0)} y1={yScale(RF)}
              x2={xScale(X_MAX)} y2={yScale(RF + (X_MAX / SIGMA_TAN) * (MU_TAN - RF))}
              stroke="var(--accent-cyan)" strokeWidth={2}
            />
            <circle cx={xScale(0)} cy={yScale(RF)} r={5} fill="var(--accent-cyan)" />
            <text x={xScale(0) + 10} y={yScale(RF) - 6} fontSize={11} fontWeight={600} fill="var(--accent-cyan)">
              R_f = {(RF * 100).toFixed(1)}%
            </text>
          </>
        )}

        {/* GMVP marker */}
        <circle cx={xScale(SIGMA_MIN)} cy={yScale(MU_MIN)} r={6} fill="var(--accent-blue)" stroke="var(--bg-code)" strokeWidth={2} />
        <text x={xScale(SIGMA_MIN) - 10} y={yScale(MU_MIN) + 4} textAnchor="end" fontSize={11} fontWeight={600} fill="var(--accent-blue)">
          GMVP
        </text>

        {/* Tangency portfolio */}
        {showCAL && (
          <>
            <circle cx={xScale(SIGMA_TAN)} cy={yScale(MU_TAN)} r={7} fill="var(--accent-cyan)" stroke="var(--bg-code)" strokeWidth={2} />
            <text x={xScale(SIGMA_TAN) + 10} y={yScale(MU_TAN) - 6} fontSize={11} fontWeight={600} fill="var(--accent-cyan)">
              Tangency
            </text>
          </>
        )}

        {/* asset markers */}
        {ASSETS.map((a) => (
          <g
            key={a.name}
            onMouseEnter={() => setHover({ x: xScale(a.sigma), y: yScale(a.mu), label: `${a.name}  σ=${(a.sigma * 100).toFixed(0)}% μ=${(a.mu * 100).toFixed(1)}%` })}
            onMouseLeave={() => setHover(null)}
            style={{ cursor: 'pointer' }}
          >
            <circle cx={xScale(a.sigma)} cy={yScale(a.mu)} r={5} fill="var(--accent-orange)" stroke="var(--bg-code)" strokeWidth={1.5} opacity={0.9} />
            <text x={xScale(a.sigma) + 8} y={yScale(a.mu) + 4} fontSize={10.5} fill="var(--text-secondary)">
              {a.name}
            </text>
          </g>
        ))}

        {/* hover tooltip */}
        {hover && (
          <g pointerEvents="none">
            <rect x={hover.x - 70} y={hover.y - 32} width={140} height={22} rx={4} fill="var(--bg-surface)" stroke="var(--border-primary)" />
            <text x={hover.x} y={hover.y - 18} textAnchor="middle" fontSize={11} fontWeight={600} fill="var(--text-primary)">
              {hover.label}
            </text>
          </g>
        )}
      </svg>

      <div className="mt-3 flex flex-wrap gap-x-5 gap-y-1.5 text-[11px] text-text-tertiary">
        <span><span className="inline-block w-3 h-[2px] bg-accent-blue align-middle mr-1.5" />Efficient frontier</span>
        <span><span className="inline-block w-3 h-[2px] align-middle mr-1.5" style={{ backgroundColor: 'var(--accent-cyan)' }} />Capital Allocation Line</span>
        <span><span className="inline-block w-3 h-[2px] align-middle mr-1.5" style={{ background: 'repeating-linear-gradient(90deg, var(--accent-orange) 0 4px, transparent 4px 8px)' }} />Long-only constraint</span>
        <span><span className="inline-block w-2 h-2 rounded-full bg-accent-orange align-middle mr-1.5" />Individual assets</span>
      </div>
    </VizFrame>
  )
}
