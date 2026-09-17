import Link from 'next/link'
import { ArrowRight, BookOpen, Search, Command } from 'lucide-react'
import { TopNav } from '@/components/layout/TopNav'
import { Footer } from '@/components/layout/Footer'
import { getRegistry } from '@/lib/registry'
import { colorClasses, iconFor } from '@/lib/track-theme'
import type { Track } from '@/lib/registry-types'

function groupByCategory(tracks: Track[]) {
  const groups = new Map<string, Track[]>()
  for (const t of tracks) groups.set(t.category, [...(groups.get(t.category) ?? []), t])
  return [...groups.entries()]
}

export default async function HomePage() {
  const { tracks } = await getRegistry()
  const groups = groupByCategory(tracks)
  const moduleTotal = tracks.reduce((n, t) => n + t.moduleCount, 0)
  const firstHref = tracks[0]?.href ?? '/'
  return (
    <>
      <TopNav />
      <main className="flex-1">
        {/* Hero Section */}
        <section className="relative overflow-hidden">
          {/* Subtle grid background */}
          <div
            className="absolute inset-0 pointer-events-none opacity-[0.03] dark:opacity-[0.04]"
            style={{
              backgroundImage: `
                linear-gradient(to right, currentColor 1px, transparent 1px),
                linear-gradient(to bottom, currentColor 1px, transparent 1px)
              `,
              backgroundSize: '48px 48px',
              maskImage: 'radial-gradient(ellipse at top, black 40%, transparent 70%)',
              WebkitMaskImage: 'radial-gradient(ellipse at top, black 40%, transparent 70%)',
            }}
          />

          {/* Gradient glows */}
          <div className="absolute inset-0 overflow-hidden pointer-events-none">
            <div className="absolute top-0 left-1/2 -translate-x-1/2 w-[800px] h-[400px] rounded-full bg-accent-blue/5 blur-3xl" />
            <div className="absolute top-60 left-[20%] w-[500px] h-[400px] rounded-full bg-accent-cyan/5 blur-3xl" />
            <div className="absolute top-40 right-[15%] w-[500px] h-[400px] rounded-full bg-accent-green/5 blur-3xl" />
          </div>

          <div className="relative max-w-6xl mx-auto px-4 lg:px-6 pt-20 pb-24">
            {/* Badge */}
            <div className="flex justify-center mb-8">
              <Link
                href={firstHref}
                className="group inline-flex items-center gap-2 px-3.5 py-1.5 rounded-full
                  bg-bg-surface border border-border-primary text-text-secondary text-xs font-medium
                  hover:border-border-secondary hover:text-text-primary transition-all"
              >
                <span className="inline-flex items-center justify-center w-1.5 h-1.5 rounded-full bg-accent-green animate-pulse" />
                <span>{moduleTotal} modules across {tracks.length} tracks</span>
                <ArrowRight className="w-3 h-3 opacity-50 group-hover:opacity-100 group-hover:translate-x-0.5 transition-all" />
              </Link>
            </div>

            {/* Title — refined weight and spacing */}
            <h1 className="text-center text-5xl sm:text-6xl lg:text-7xl font-extrabold tracking-[-0.035em] leading-[0.95] mb-7 text-text-primary">
              A personal library of
              <br />
              <span className="inline-block mt-2 bg-gradient-to-r from-blue-500 via-cyan-500 to-green-500 bg-clip-text text-transparent">
                deep technical craft
              </span>
            </h1>

            {/* Subtitle */}
            <p className="text-center text-lg sm:text-xl text-text-secondary max-w-2xl mx-auto mb-10 leading-relaxed">
              Expert-level courses and deep dives, researched from primary sources and written to be read.
            </p>

            {/* Search prompt CTA */}
            <div className="flex justify-center mb-20">
              <div className="flex items-center gap-3 text-sm text-text-tertiary">
                <span>Jump to any topic</span>
                <kbd className="inline-flex items-center gap-1 px-2 py-1 text-xs">
                  <Command className="w-3 h-3" />
                  K
                </kbd>
                <span>or</span>
                <kbd className="inline-flex items-center px-2 py-1 text-xs">/</kbd>
              </div>
            </div>

            {/* Track cards, grouped by category */}
            <div className="space-y-12">
              {groups.map(([category, list]) => (
                <section key={category}>
                  <h2 className="text-[11px] font-semibold uppercase tracking-[0.14em] text-text-tertiary mb-4">{category}</h2>
                  <div className="grid md:grid-cols-2 gap-5">
                    {list.map((track) => {
                      const theme = colorClasses(track.color)
                      const Icon = iconFor(track.icon)
                      const unit = track.unitLabel === 'Chapter' ? 'chapters' : track.unitLabel === 'Page' ? 'pages' : 'modules'
                      return (
                        <Link
                          key={track.id}
                          href={track.href}
                          className="group relative rounded-2xl border border-border-primary bg-bg-surface overflow-hidden
                            hover:border-border-secondary transition-all duration-300
                            hover:shadow-md hover:-translate-y-0.5"
                        >
                          <div className={`absolute top-0 left-0 right-0 h-[2px] bg-gradient-to-r ${theme.gradient} opacity-60 group-hover:opacity-100 transition-opacity`} />

                          <div className="p-7">
                            <div className="flex items-start justify-between mb-5">
                              <div className={`w-11 h-11 rounded-xl bg-gradient-to-br ${theme.gradient} flex items-center justify-center shadow-sm`}>
                                <Icon className="w-5 h-5 text-white" strokeWidth={2} />
                              </div>
                              <div className="flex items-baseline gap-1.5 text-text-tertiary">
                                <span className="text-2xl font-semibold tabular-nums text-text-primary">{track.moduleCount}</span>
                                <span className="text-xs uppercase tracking-wider">{unit}</span>
                              </div>
                            </div>

                            <div className="flex items-center gap-2 mb-2">
                              <h3 className="text-lg font-semibold text-text-primary tracking-tight">{track.title}</h3>
                              <span className={`text-[10px] font-semibold uppercase tracking-wider px-1.5 py-0.5 rounded ${theme.subtle} ${theme.text}`}>
                                {track.kind === 'deep-dive' ? 'Deep dive' : 'Course'}
                              </span>
                            </div>

                            <p className="text-sm text-text-secondary leading-relaxed mb-5">{track.description}</p>

                            <div className="flex items-center justify-between text-xs">
                              <div className="flex items-center gap-3 text-text-tertiary">
                                {track.partCount > 0 && <span>{track.partCount} parts</span>}
                                {track.appendixCount > 0 && <span>{track.appendixCount} appendices</span>}
                              </div>
                              <span className={`inline-flex items-center gap-1 text-text-secondary ${theme.hoverText} group-hover:gap-2 transition-all font-medium`}>
                                Open
                                <ArrowRight className="w-3.5 h-3.5" />
                              </span>
                            </div>
                          </div>
                        </Link>
                      )
                    })}
                  </div>
                </section>
              ))}
            </div>

            {/* Bottom helper row */}
            <div className="mt-16 flex flex-col sm:flex-row items-center justify-center gap-4 text-xs text-text-tertiary">
              <span className="inline-flex items-center gap-2">
                <Search className="w-3.5 h-3.5" />
                Use search to jump anywhere
              </span>
              <span className="hidden sm:inline opacity-40">·</span>
              <span className="inline-flex items-center gap-2">
                <BookOpen className="w-3.5 h-3.5" />
                Navigate with <kbd className="text-[10px]">←</kbd> <kbd className="text-[10px]">→</kbd>
              </span>
            </div>
          </div>
        </section>
      </main>
      <Footer tracks={tracks.map((t) => ({ id: t.id, shortTitle: t.shortTitle }))} />
    </>
  )
}
