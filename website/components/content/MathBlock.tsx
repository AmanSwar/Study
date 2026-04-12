'use client'

import 'katex/dist/katex.min.css'
import { InlineMath, BlockMath } from 'react-katex'

interface MathBlockProps {
  children: string
  inline?: boolean
}

export function MathBlock({ children, inline = false }: MathBlockProps) {
  if (inline) {
    return (
      <span className="mx-0.5">
        <InlineMath math={children} />
      </span>
    )
  }

  return (
    <div className="my-6 py-4 px-6 rounded-xl bg-bg-surface border border-border-primary overflow-x-auto">
      <div className="flex items-center justify-center text-lg">
        <BlockMath math={children} />
      </div>
    </div>
  )
}
