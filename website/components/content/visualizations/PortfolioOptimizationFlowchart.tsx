'use client'

import { VizFrame } from './VizFrame'

const W = 760
const H = 520

interface Box {
  x: number
  y: number
  w: number
  h: number
  title: string
  subtitle?: string
  color: 'blue' | 'cyan' | 'orange' | 'green' | 'gray'
}

const COLOR: Record<Box['color'], { stroke: string; fill: string; text: string }> = {
  blue:   { stroke: 'var(--accent-blue)',   fill: 'rgba(59,130,246,0.08)',  text: 'var(--accent-blue)'   },
  cyan:   { stroke: 'var(--accent-cyan)',   fill: 'rgba(34,211,238,0.08)',  text: 'var(--accent-cyan)'   },
  orange: { stroke: 'var(--accent-orange)', fill: 'rgba(249,115,22,0.08)',  text: 'var(--accent-orange)' },
  green:  { stroke: 'var(--accent-green)',  fill: 'rgba(34,197,94,0.08)',   text: 'var(--accent-green)'  },
  gray:   { stroke: 'var(--text-tertiary)', fill: 'var(--bg-surface)',      text: 'var(--text-secondary)' },
}

// Layout: 3-column flowchart
//   Row 1 (y=20):   [Price data] [Universe metadata] [Macro factors]
//   Row 2 (y=110):  [Covariance Σ̂]  ←  [Returns μ̂]
//   Row 3 (y=200):  [MVO]   [Risk Parity]   [HRP]
//   Row 4 (y=320):  [Backtest + selection]
//   Row 5 (y=410):  [Production weights]

const BOXES: Box[] = [
  { x:  20, y:  20, w: 220, h: 64, title: 'Price history', subtitle: 'Daily OHLCV, dividends', color: 'gray' },
  { x: 270, y:  20, w: 220, h: 64, title: 'Universe metadata', subtitle: 'Sectors, market cap, free float', color: 'gray' },
  { x: 520, y:  20, w: 220, h: 64, title: 'Macro / factor data', subtitle: 'Rates, FX, sentiment', color: 'gray' },

  { x:  60, y: 140, w: 280, h: 70, title: 'Covariance estimation Σ̂', subtitle: 'Sample → Ledoit–Wolf shrinkage → PCA denoise', color: 'cyan' },
  { x: 420, y: 140, w: 280, h: 70, title: 'Expected return μ̂', subtitle: 'Equilibrium · Black–Litterman · model views', color: 'cyan' },

  { x:  20, y: 270, w: 220, h: 90, title: 'MVO', subtitle: 'Markowitz mean–variance\n(needs both Σ̂ and μ̂)', color: 'blue' },
  { x: 270, y: 270, w: 220, h: 90, title: 'Risk Parity', subtitle: 'Equal risk contribution\n(Σ̂ only — μ-free)', color: 'orange' },
  { x: 520, y: 270, w: 220, h: 90, title: 'HRP', subtitle: 'Hierarchical risk parity\n(robust to ill-conditioned Σ̂)', color: 'green' },

  { x: 130, y: 400, w: 500, h: 64, title: 'Backtest + selection', subtitle: 'Rolling out-of-sample · turnover/cost adjusted Sharpe', color: 'cyan' },
  { x: 230, y: 480, w: 300, h: 30, title: 'Production weights', color: 'blue' },
]

interface Arrow {
  from: number
  to: number
  // optional offset on the line midpoint label
  label?: string
}

const ARROWS: Arrow[] = [
  { from: 0, to: 3 },
  { from: 1, to: 3 },
  { from: 2, to: 4 },
  { from: 0, to: 4, label: 'estimate' },
  { from: 3, to: 5 },
  { from: 4, to: 5 },
  { from: 3, to: 6 },
  { from: 3, to: 7 },
  { from: 5, to: 8 },
  { from: 6, to: 8 },
  { from: 7, to: 8 },
  { from: 8, to: 9 },
]

function boxCenter(b: Box) {
  return { cx: b.x + b.w / 2, cy: b.y + b.h / 2 }
}

// Connect with a path that exits the bottom of `from` and enters the top of `to`
// (or sides when same row). Returns an SVG path string + label position.
function connectorPath(from: Box, to: Box): { d: string; midX: number; midY: number; arrowAngle: number } {
  const fc = boxCenter(from)
  const tc = boxCenter(to)

  // Default: bottom of `from` → top of `to`
  let x1 = fc.cx
  let y1 = from.y + from.h
  let x2 = tc.cx
  let y2 = to.y

  // If on roughly the same row (same y), connect side-to-side
  if (Math.abs(fc.cy - tc.cy) < 30) {
    if (fc.cx < tc.cx) {
      x1 = from.x + from.w
      y1 = fc.cy
      x2 = to.x
      y2 = tc.cy
    } else {
      x1 = from.x
      y1 = fc.cy
      x2 = to.x + to.w
      y2 = tc.cy
    }
    return {
      d: `M ${x1} ${y1} L ${x2} ${y2}`,
      midX: (x1 + x2) / 2, midY: (y1 + y2) / 2,
      arrowAngle: x2 > x1 ? 0 : Math.PI,
    }
  }

  // Vertical S-curve
  const midY = (y1 + y2) / 2
  const d = `M ${x1} ${y1} C ${x1} ${midY}, ${x2} ${midY}, ${x2} ${y2}`
  return {
    d, midX: (x1 + x2) / 2, midY,
    arrowAngle: Math.PI / 2, // pointing down
  }
}

export function PortfolioOptimizationFlowchart() {
  return (
    <VizFrame
      title="Portfolio optimization pipeline"
      caption="From raw data → covariance & return estimation → three competing optimisers (MVO, Risk Parity, HRP) → backtest selection → production weights. Each optimiser handles estimation noise differently — pick the one whose assumptions match your data quality."
    >
      <svg viewBox={`0 0 ${W} ${H}`} width="100%" style={{ maxWidth: W, height: 'auto' }} role="img">
        <defs>
          <marker id="poflow-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
            <path d="M 0 0 L 10 5 L 0 10 z" fill="var(--accent-cyan)" />
          </marker>
        </defs>

        {/* arrows first so boxes overlap them */}
        {ARROWS.map((a, i) => {
          const path = connectorPath(BOXES[a.from], BOXES[a.to])
          return (
            <g key={i}>
              <path d={path.d} fill="none" stroke="var(--accent-cyan)" strokeWidth={1.6} markerEnd="url(#poflow-arrow)" opacity={0.85} />
              {a.label && (
                <g transform={`translate(${path.midX}, ${path.midY})`}>
                  <rect x={-26} y={-9} width={52} height={16} rx={3} fill="var(--bg-code)" />
                  <text textAnchor="middle" y={3} fontSize={10} fill="var(--text-tertiary)">{a.label}</text>
                </g>
              )}
            </g>
          )
        })}

        {/* boxes */}
        {BOXES.map((b, i) => {
          const c = COLOR[b.color]
          const { cx, cy } = boxCenter(b)
          const subtitleLines = b.subtitle?.split('\n') ?? []
          const subtitleStartY = cy + (b.subtitle ? 4 : 0)
          return (
            <g key={i}>
              <rect x={b.x} y={b.y} width={b.w} height={b.h} rx={8} fill={c.fill} stroke={c.stroke} strokeWidth={1.6} />
              <text
                x={cx} y={subtitleLines.length > 0 ? cy - 6 : cy + 4}
                textAnchor="middle" fontSize={13} fontWeight={700} fill={c.text}
              >
                {b.title}
              </text>
              {subtitleLines.map((line, j) => (
                <text
                  key={j}
                  x={cx} y={subtitleStartY + 12 + j * 14}
                  textAnchor="middle" fontSize={11} fill="var(--text-secondary)"
                >
                  {line}
                </text>
              ))}
            </g>
          )
        })}
      </svg>

      <div className="mt-3 grid grid-cols-1 sm:grid-cols-3 gap-2 text-[11px]">
        <div className="rounded-md border border-border-primary bg-bg-surface/40 px-3 py-2">
          <div className="font-semibold text-accent-blue mb-0.5">MVO</div>
          <div className="text-text-tertiary">Best when μ̂ is reliable. Brittle on small samples.</div>
        </div>
        <div className="rounded-md border border-border-primary bg-bg-surface/40 px-3 py-2">
          <div className="font-semibold text-accent-orange mb-0.5">Risk Parity</div>
          <div className="text-text-tertiary">Skip the unreliable μ̂. Equalises risk contribution.</div>
        </div>
        <div className="rounded-md border border-border-primary bg-bg-surface/40 px-3 py-2">
          <div className="font-semibold text-accent-green mb-0.5">HRP</div>
          <div className="text-text-tertiary">Cluster first, then allocate. Robust on ill-conditioned Σ̂.</div>
        </div>
      </div>
    </VizFrame>
  )
}
