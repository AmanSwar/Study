import { Clock, BookOpen } from 'lucide-react'

interface ModuleHeaderProps {
  number: number
  title: string
  track: string
  part: number
  readingTime: string
  prerequisites?: string[]
  description?: string
}

export function ModuleHeader({
  number,
  title,
  track,
  part,
  readingTime,
  prerequisites = [],
  description,
}: ModuleHeaderProps) {
  return (
    <div className="mb-10 pb-8 border-b border-border-primary">
      {/* Module number large watermark */}
      <div className="flex items-start gap-4 mb-4">
        <span className="text-6xl sm:text-7xl font-black text-accent-blue/15 leading-none select-none">
          {String(number).padStart(2, '0')}
        </span>
        <div className="pt-1">
          <div className="flex items-center gap-3 text-xs text-text-tertiary mb-2">
            <span className="flex items-center gap-1">
              <Clock className="w-3 h-3" />
              {readingTime}
            </span>
            <span className="flex items-center gap-1">
              <BookOpen className="w-3 h-3" />
              Part {part}
            </span>
          </div>
          <h1 className="text-2xl sm:text-3xl lg:text-4xl font-extrabold tracking-tight text-text-primary leading-tight">
            {title}
          </h1>
        </div>
      </div>

      {description && (
        <p className="text-lg text-text-secondary leading-relaxed mb-4 max-w-3xl">
          {description}
        </p>
      )}

      {/* Prerequisites */}
      {prerequisites.length > 0 && (
        <div className="flex flex-wrap items-center gap-2">
          <span className="text-xs font-medium text-text-tertiary uppercase tracking-wider">Prerequisites:</span>
          {prerequisites.map((prereq) => (
            <span
              key={prereq}
              className="text-xs px-2.5 py-1 rounded-full bg-bg-surface border border-border-primary
                text-text-secondary"
            >
              {prereq}
            </span>
          ))}
        </div>
      )}
    </div>
  )
}
