import Link from 'next/link'
import type { Appendix, Module, Part, Track } from '@/lib/registry-types'
import { ReadState } from '@/components/content/ReadState'
import { TrackProgress } from '@/components/content/TrackProgress'

const unitPlural = (t: Track) => (t.unitLabel === 'Chapter' ? 'chapters' : t.unitLabel === 'Page' ? 'pages' : 'modules')

/** "25 min" → 25; anything else → 0 */
const minutes = (s?: string) => (s ? Number((s.match(/(\d+)\s*min/) ?? [])[1] ?? 0) : 0)

function totalReading(track: Track) {
  const mins = track.allModules.reduce((n, m) => n + minutes(m.readingTime), 0)
  if (!mins) return null
  const h = mins / 60
  return h >= 1 ? `≈ ${h < 10 ? h.toFixed(1).replace(/\.0$/, '') : Math.round(h)} h reading` : `≈ ${mins} min reading`
}

/** Track page: a syllabus — title, description, size, then parts and modules as numbered rows. */
export function TrackHeader({ track }: { track: Track }) {
  const total = totalReading(track)
  return (
    <header className="mb-10">
      <p className="ui font-sans text-[12px] font-medium uppercase tracking-[0.06em] text-text-tertiary mb-3">
        {track.kind === 'deep-dive' ? 'Deep dive' : 'Course'} · {track.category}
      </p>
      <h1 className="font-serif text-[2.125rem] font-semibold tracking-[-0.02em] leading-[1.15] text-text-primary text-balance mb-4">{track.title}</h1>
      <p className="text-[1.1em] leading-[1.5] text-text-secondary text-pretty mb-4">{track.description}</p>
      <p className="ui font-sans text-[13px] text-text-tertiary flex flex-wrap items-center gap-x-2 gap-y-1">
        <span>{track.moduleCount} {unitPlural(track)}</span>
        {track.partCount > 0 && <><span aria-hidden="true">·</span><span>{track.partCount} parts</span></>}
        {track.appendixCount > 0 && <><span aria-hidden="true">·</span><span>{track.appendixCount} {track.appendixCount === 1 ? 'appendix' : 'appendices'}</span></>}
        {total && <><span aria-hidden="true">·</span><span>{total}</span></>}
        {track.sourcesAbsPath && <><span aria-hidden="true">·</span><Link href={`${track.href}/sources`} className="hover:text-text-primary transition-colors underline decoration-border-secondary underline-offset-[3px]">Sources</Link></>}
        <TrackProgress href={track.href} total={track.moduleCount} />
      </p>
    </header>
  )
}

export function ModuleRow({ module }: { module: Module }) {
  return (
    <Link href={module.href} className="group grid grid-cols-[2rem_1fr_auto] items-baseline gap-x-3 py-[0.55rem] border-t border-border-subtle">
      <span className="ui font-mono text-[12px] text-text-tertiary">{String(module.number).padStart(2, '0')}</span>
      <span className="text-[1.02em] leading-snug text-text-primary group-hover:text-accent-blue transition-colors">{module.title}</span>
      <span className="ui font-sans text-[12px] text-text-tertiary whitespace-nowrap flex items-center gap-2.5">
        {module.readingTime}
        <ReadState href={module.href} />
      </span>
    </Link>
  )
}

export function PartsGrid({ parts }: { parts: Part[] }) {
  return (
    <div className="space-y-9">
      {parts.map((part) => (
        <section key={part.id} className="grid grid-cols-[3.5rem_1fr] gap-x-4 max-sm:grid-cols-1">
          <span className="ui font-mono text-[12.5px] text-text-tertiary pt-1.5">Part {part.number}</span>
          <div>
            <h2 className="font-serif text-[1.25rem] font-semibold tracking-[-0.012em] leading-snug text-text-primary mb-1">
              <Link href={part.href} className="hover:text-accent-blue transition-colors">{part.title}</Link>
            </h2>
            <p className="text-[0.95em] text-text-secondary text-pretty mb-3">{part.description}</p>
            {part.modules.map((m) => <ModuleRow key={m.id} module={m} />)}
          </div>
        </section>
      ))}
    </div>
  )
}

export function ModuleCards({ modules }: { modules: Module[] }) {
  return <div>{modules.map((m) => <ModuleRow key={m.id} module={m} />)}</div>
}

export function AppendixCards({ appendices, title = 'Appendices' }: { appendices: Appendix[]; title?: string }) {
  return (
    <section className="mt-10 grid grid-cols-[3.5rem_1fr] gap-x-4 max-sm:grid-cols-1">
      <span className="ui font-mono text-[12.5px] text-text-tertiary pt-1.5" aria-hidden="true" />
      <div>
        {title && <h2 className="font-serif text-[1.25rem] font-semibold tracking-[-0.012em] leading-snug text-text-primary mb-3">{title}</h2>}
        {appendices.map((a) => (
          <Link key={a.id} href={a.href} className="group grid grid-cols-[2rem_1fr] items-baseline gap-x-3 py-[0.55rem] border-t border-border-subtle">
            <span className="ui font-mono text-[12px] text-text-tertiary">{a.letter}</span>
            <span className="text-[1.02em] leading-snug text-text-primary group-hover:text-accent-blue transition-colors">{a.title}</span>
          </Link>
        ))}
      </div>
    </section>
  )
}
