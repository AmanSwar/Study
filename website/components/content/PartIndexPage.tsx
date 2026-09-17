import type { Part, Track } from '@/lib/registry-types'
import { ModuleRow } from '@/components/content/TrackIndex'

/** One part of a course: its modules as numbered rows. */
export function PartIndexPage({ part, track }: { part: Part; track: Track }) {
  return (
    <div className="max-w-[46rem] mx-auto px-6 sm:px-8 pt-10 pb-24">
      <header className="mb-8">
        <p className="ui font-sans text-[12px] font-medium uppercase tracking-[0.06em] text-text-tertiary mb-3">
          {track.shortTitle} · Part {part.number}
        </p>
        <h1 className="font-serif text-[2.125rem] font-semibold tracking-[-0.02em] leading-[1.15] text-text-primary text-balance mb-4">{part.title}</h1>
        <p className="text-[1.1em] leading-[1.5] text-text-secondary text-pretty">{part.description}</p>
      </header>
      <div>{part.modules.map((m) => <ModuleRow key={m.id} module={m} />)}</div>
    </div>
  )
}
