import Link from 'next/link'
import { TopNav } from '@/components/layout/TopNav'
import { Footer } from '@/components/layout/Footer'
import { ContinueReading } from '@/components/layout/ContinueReading'
import { TrackProgress } from '@/components/content/TrackProgress'
import { getRegistry } from '@/lib/registry'
import type { Track } from '@/lib/registry-types'

function groupByCategory(tracks: Track[]) {
  const groups = new Map<string, Track[]>()
  for (const t of tracks) groups.set(t.category, [...(groups.get(t.category) ?? []), t])
  return [...groups.entries()]
}

const unitPlural = (t: Track) => (t.unitLabel === 'Chapter' ? 'chapters' : t.unitLabel === 'Page' ? 'pages' : 'modules')

/** The library: what is here to read, and where you left off. */
export default async function HomePage() {
  const { tracks } = await getRegistry()
  const groups = groupByCategory(tracks)
  const moduleTotal = tracks.reduce((n, t) => n + t.moduleCount, 0)
  const trackLabels = Object.fromEntries(tracks.map((t) => [t.id, t.shortTitle]))

  return (
    <>
      <TopNav />
      <main className="flex-1 pt-12">
        <div className="max-w-[46rem] mx-auto px-6 sm:px-8 pt-12 pb-24">
          <h1 className="font-serif text-[1.5rem] font-semibold tracking-[-0.015em] text-text-primary mb-1">aman.study</h1>
          <p className="ui font-sans text-[14px] text-text-tertiary mb-10">
            A personal library of technical study material — {moduleTotal} modules across {tracks.length} tracks, researched from primary sources. Written to be read.
          </p>

          <ContinueReading trackLabels={trackLabels} />

          {groups.map(([category, list]) => (
            <section key={category} className="mt-10">
              <h2 className="ui font-sans text-[11px] font-semibold uppercase tracking-[0.1em] text-text-tertiary pb-2 border-b border-border-primary flex items-baseline justify-between">
                {category}
                <span className="font-normal normal-case tracking-normal">{list.length} {list.length === 1 ? 'track' : 'tracks'}</span>
              </h2>
              {list.map((track) => (
                <Link key={track.id} href={track.href} className="group grid grid-cols-[1fr_auto] gap-x-8 gap-y-1 py-[1.1rem] border-b border-border-subtle">
                  <span className="font-serif text-[1.25rem] font-semibold tracking-[-0.012em] leading-[1.25] text-text-primary group-hover:text-accent-blue transition-colors">
                    {track.title}
                  </span>
                  <span className="ui font-sans text-[12.5px] text-text-tertiary text-right whitespace-nowrap pt-1">
                    {track.kind === 'deep-dive' ? 'Deep dive' : 'Course'} · {track.moduleCount} {unitPlural(track)}
                    {track.partCount > 0 && ` · ${track.partCount} parts`}
                  </span>
                  <span className="col-span-2 text-[0.95rem] leading-[1.5] text-text-secondary max-w-[40rem] text-pretty">{track.description}</span>
                  <span className="col-span-2 empty:hidden"><TrackProgress href={track.href} total={track.moduleCount} /></span>
                </Link>
              ))}
            </section>
          ))}
        </div>
      </main>
      <Footer />
    </>
  )
}
