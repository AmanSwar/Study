'use client'

import { ReactNode } from 'react'

interface VizFrameProps {
  title: string
  caption?: string
  children: ReactNode
}

export function VizFrame({ title, caption, children }: VizFrameProps) {
  return (
    <div className="code-block">
      <div className="code-head">
        <span className="lang">Visualization</span>
        <span className="code-title">{title}</span>
      </div>
      <div className="p-4">
        <div className="overflow-x-auto">{children}</div>
        {caption && (
          <p className="ui mt-3 mb-0 font-sans text-[13px] text-text-secondary leading-relaxed">{caption}</p>
        )}
      </div>
    </div>
  )
}
