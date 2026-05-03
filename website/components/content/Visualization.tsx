'use client'

import { EfficientFrontier } from './visualizations/EfficientFrontier'
import { QQPlot } from './visualizations/QQPlot'
import { RollingCorrelation } from './visualizations/RollingCorrelation'
import { CovarianceEigenvalues } from './visualizations/CovarianceEigenvalues'
import { PortfolioOptimizationFlowchart } from './visualizations/PortfolioOptimizationFlowchart'

interface VisualizationProps {
  /** Slug of the visualization to render. */
  id: string
}

const REGISTRY: Record<string, () => React.ReactElement> = {
  'efficient-frontier': () => <EfficientFrontier />,
  'qq-plot-of-real-stock-returns-vs-normal-showing-tail-deviation': () => <QQPlot />,
  'rolling-60-day-correlation-showing-instability': () => <RollingCorrelation />,
  'covariance-estimation-methods': () => <CovarianceEigenvalues />,
  'portfolio-optimization-flowchart': () => <PortfolioOptimizationFlowchart />,
}

export function Visualization({ id }: VisualizationProps) {
  const slug = id.trim()
  const renderer = REGISTRY[slug]
  if (renderer) return renderer()

  // Unknown id — render a friendly placeholder so the gap is obvious in dev
  // but the page still renders cleanly in prod.
  return (
    <div className="my-6 rounded-xl border border-dashed border-border-primary bg-bg-code/50 p-5 not-prose">
      <div className="text-[10px] font-semibold uppercase tracking-wider text-text-tertiary mb-1">
        Visualization placeholder
      </div>
      <div className="text-sm text-text-secondary">
        No interactive visualization registered for <code className="font-mono">{slug || '(empty)'}</code>.
      </div>
    </div>
  )
}
