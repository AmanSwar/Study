import { Clock } from 'lucide-react'

interface ModuleHeaderProps {
  number: number
  title: string
  track: string
  part: number
  readingTime: string
  prerequisites?: string[]
  description?: string
}

/**
 * The hero block at the top of each module page.
 * Refined layout: small meta row on top, prominent title, description,
 * optional prereq chips.
 */
export function ModuleHeader({
  number,
  title,
  part,
  readingTime,
  prerequisites = [],
  description,
}: ModuleHeaderProps) {
  return (
    <header className="mb-12 pb-10 border-b border-border-primary">
      {/* Meta row: module number + reading time + part */}
      <div className="flex items-center gap-2.5 text-xs mb-5">
        <span className="inline-flex items-center px-2 py-0.5 rounded-md bg-accent-blue-subtle text-accent-blue font-mono font-semibold tabular-nums">
          {part > 0 ? `${String(part).padStart(2, '0')}.${String(number).padStart(2, '0')}` : String(number).padStart(2, '0')}
        </span>
        <span className="inline-flex items-center gap-1 text-text-tertiary">
          <Clock className="w-3 h-3" />
          {readingTime}
        </span>
      </div>

      {/* Title — large, tight, readable */}
      <h1 className="text-4xl sm:text-5xl font-extrabold tracking-[-0.03em] leading-[1.05] text-text-primary mb-5 text-balance">
        {title}
      </h1>

      {/* Description / deck */}
      {description && (
        <p className="text-lg text-text-secondary leading-relaxed max-w-3xl mb-6 text-pretty">
          {description}
        </p>
      )}

      {/* Prerequisites */}
      {prerequisites.length > 0 && (
        <div className="flex flex-wrap items-center gap-1.5">
          <span className="text-[11px] font-semibold text-text-tertiary uppercase tracking-[0.08em] mr-1">
            Prerequisites
          </span>
          {prerequisites.map((prereq) => (
            <span
              key={prereq}
              className="text-xs px-2 py-0.5 rounded-md bg-bg-surface border border-border-primary text-text-secondary"
            >
              {prereq}
            </span>
          ))}
        </div>
      )}
    </header>
  )
}
