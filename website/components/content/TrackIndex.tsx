import Link from 'next/link'
import { ArrowRight, BookOpen, Clock, Layers, Library } from 'lucide-react'
import type { Appendix, Module, Part, Track } from '@/lib/registry-types'
import { colorClasses, type ColorClasses } from '@/lib/track-theme'

export function TrackHeader({ track }: { track: Track }) {
  const theme = colorClasses(track.color)
  const unit = track.unitLabel === 'Chapter' ? 'Chapters' : track.unitLabel === 'Page' ? 'Pages' : 'Modules'
  return (
    <div className="mb-10">
      <div className={`inline-flex items-center gap-2 px-3 py-1 rounded-full ${theme.subtle} ${theme.text} text-xs font-medium mb-4`}>
        <Layers className="w-3.5 h-3.5" />
        {track.kind === 'deep-dive' ? 'Deep dive' : 'Course'} &middot; {track.moduleCount} {unit}
        {track.appendixCount > 0 && <>&middot; {track.appendixCount} Appendices</>}
      </div>
      <h1 className="text-3xl sm:text-4xl font-extrabold tracking-tight text-text-primary mb-4">{track.title}</h1>
      <p className="text-lg text-text-secondary leading-relaxed max-w-3xl">{track.description}</p>
    </div>
  )
}

export function PartsGrid({ parts, theme }: { parts: Part[]; theme: ColorClasses }) {
  return (
    <div className="space-y-4">
      {parts.map((part) => (
        <Link
          key={part.id}
          href={part.href}
          className={`group block rounded-xl border border-border-primary bg-bg-surface ${theme.hoverBorder} hover:shadow-md transition-all duration-200 overflow-hidden`}
        >
          <div className="flex items-stretch">
            <div className={`w-16 shrink-0 bg-gradient-to-b ${theme.gradient} flex items-center justify-center`}>
              <span className="text-2xl font-bold text-white">{part.number}</span>
            </div>
            <div className="flex-1 p-5">
              <div className="flex items-start justify-between gap-4">
                <div className="flex-1">
                  <h3 className={`text-lg font-bold text-text-primary ${theme.hoverText} transition-colors mb-1`}>
                    Part {part.number}: {part.title}
                  </h3>
                  <p className="text-sm text-text-secondary mb-3">{part.description}</p>
                  <div className="flex items-center gap-4 text-xs text-text-tertiary">
                    <span className="flex items-center gap-1"><BookOpen className="w-3.5 h-3.5" />{part.modules.length} modules</span>
                  </div>
                </div>
                <ArrowRight className={`w-5 h-5 text-text-tertiary ${theme.hoverText} group-hover:translate-x-1 transition-all shrink-0 mt-1`} />
              </div>
            </div>
          </div>
        </Link>
      ))}
    </div>
  )
}

export function ModuleCards({ modules, theme }: { modules: Module[]; theme: ColorClasses }) {
  return (
    <div className="space-y-3">
      {modules.map((module) => (
        <Link
          key={module.id}
          href={module.href}
          className={`group block rounded-xl border border-border-primary bg-bg-surface ${theme.hoverBorder} hover:shadow-md transition-all duration-200 p-5`}
        >
          <div className="flex items-start gap-4">
            <div className={`w-10 h-10 rounded-lg ${theme.subtle} flex items-center justify-center shrink-0`}>
              <span className={`text-sm font-bold ${theme.text}`}>{module.number}</span>
            </div>
            <div className="flex-1 min-w-0">
              <h3 className={`text-base font-bold text-text-primary ${theme.hoverText} transition-colors mb-1`}>{module.title}</h3>
              <p className="text-sm text-text-secondary leading-relaxed mb-2">{module.description}</p>
              {module.readingTime && (
                <div className="flex items-center gap-3 text-xs text-text-tertiary">
                  <span className="flex items-center gap-1"><Clock className="w-3 h-3" />{module.readingTime}</span>
                </div>
              )}
            </div>
            <ArrowRight className={`w-4 h-4 text-text-tertiary ${theme.hoverText} group-hover:translate-x-1 transition-all shrink-0 mt-2`} />
          </div>
        </Link>
      ))}
    </div>
  )
}

export function AppendixCards({ appendices, theme, title = 'Appendices' }: { appendices: Appendix[]; theme: ColorClasses; title?: string }) {
  return (
    <div className="mt-10">
      <h2 className="text-xl font-bold tracking-tight text-text-primary mb-4 flex items-center gap-2"><Library className="w-4 h-4 text-text-tertiary" />{title}</h2>
      <div className="space-y-3">
        {appendices.map((a) => (
          <Link key={a.id} href={a.href} className={`group block rounded-xl border border-border-primary bg-bg-surface ${theme.hoverBorder} hover:shadow-md transition-all p-5`}>
            <div className="flex items-center gap-4">
              <span className={`w-10 h-10 rounded-lg ${theme.subtle} flex items-center justify-center text-sm font-bold ${theme.text}`}>{a.letter}</span>
              <span className={`text-base font-bold text-text-primary ${theme.hoverText} transition-colors`}>{a.title}</span>
            </div>
          </Link>
        ))}
      </div>
    </div>
  )
}
