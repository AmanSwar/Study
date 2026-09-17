interface ModuleHeaderProps {
  number: number
  title: string
  trackTitle: string
  unitLabel: string
  part: number
  readingTime: string
  prerequisites?: string[]
  description?: string
}

/**
 * Header for markdown modules, in the same markup the HTML modules carry
 * (kicker · title · lede · meta) so both formats read identically.
 */
export function ModuleHeader({ number, title, trackTitle, unitLabel, part, readingTime, prerequisites = [], description }: ModuleHeaderProps) {
  return (
    <header className="study-header">
      <p className="kicker">
        {trackTitle} · {part > 0 && `Part ${part} · `}{unitLabel} {number}
      </p>
      <h1>{title}</h1>
      {description && <p className="lede">{description}</p>}
      <div className="meta">
        {readingTime && <span className="chip"><b>{readingTime}</b> read</span>}
        {prerequisites.length > 0 && <span className="chip">Prerequisites: <b>{prerequisites.join(', ')}</b></span>}
      </div>
    </header>
  )
}
