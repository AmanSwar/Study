import Link from 'next/link'
import { ArrowRight, BookOpen, Layers } from 'lucide-react'
import { Breadcrumbs } from '@/components/layout/Breadcrumbs'
import { intelTrack } from '@/content/intel'

export const metadata = { title: 'Intel & AMD CPU Architecture Track' }

export default function IntelPage() {
  return (
    <>
      <Breadcrumbs items={[{ label: 'Intel/AMD' }]} />
      <div className="mb-10">
        <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-accent-cyan-subtle text-accent-cyan text-xs font-medium mb-4">
          <Layers className="w-3.5 h-3.5" />
          {intelTrack.moduleCount} Modules &middot; {intelTrack.appendixCount} Appendices
        </div>
        <h1 className="text-3xl sm:text-4xl font-extrabold tracking-tight text-text-primary mb-4">{intelTrack.title}</h1>
        <p className="text-lg text-text-secondary leading-relaxed max-w-3xl">{intelTrack.description}</p>
      </div>
      <div className="space-y-4">
        {intelTrack.parts?.map((part) => (
          <Link key={part.id} href={`/intel/${part.id}`}
            className="group block rounded-xl border border-border-primary bg-bg-surface hover:border-accent-cyan/40 hover:shadow-md transition-all duration-200 overflow-hidden">
            <div className="flex items-stretch">
              <div className="w-16 shrink-0 bg-gradient-to-b from-cyan-500 to-teal-400 flex items-center justify-center">
                <span className="text-2xl font-bold text-white">{part.number}</span>
              </div>
              <div className="flex-1 p-5">
                <div className="flex items-start justify-between gap-4">
                  <div className="flex-1">
                    <h3 className="text-lg font-bold text-text-primary group-hover:text-accent-cyan transition-colors mb-1">
                      Part {part.number}: {part.title}
                    </h3>
                    <p className="text-sm text-text-secondary mb-3">{part.description}</p>
                    <div className="flex items-center gap-4 text-xs text-text-tertiary">
                      <span className="flex items-center gap-1"><BookOpen className="w-3.5 h-3.5" />{part.modules.length} modules</span>
                    </div>
                  </div>
                  <ArrowRight className="w-5 h-5 text-text-tertiary group-hover:text-accent-cyan group-hover:translate-x-1 transition-all shrink-0 mt-1" />
                </div>
              </div>
            </div>
          </Link>
        ))}
      </div>
    </>
  )
}
