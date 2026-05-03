'use client'

import { ReactNode } from 'react'

interface VizFrameProps {
  title: string
  caption?: string
  children: ReactNode
}

export function VizFrame({ title, caption, children }: VizFrameProps) {
  return (
    <div className="my-6 rounded-xl border border-border-primary bg-bg-code overflow-hidden not-prose">
      <div className="flex items-center justify-between px-4 py-2 border-b border-border-primary bg-bg-surface/50">
        <span className="text-[10px] font-semibold px-2 py-0.5 rounded-md uppercase tracking-wider bg-amber-500/20 text-amber-400">
          Visualization
        </span>
        <span className="text-[11px] text-text-tertiary tracking-tight">{title}</span>
      </div>
      <div className="p-4">
        <div className="overflow-x-auto">{children}</div>
        {caption && (
          <p className="mt-3 text-xs text-text-tertiary leading-relaxed">{caption}</p>
        )}
      </div>
    </div>
  )
}
