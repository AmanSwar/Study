import Link from 'next/link'
import { ArrowRight, BookOpen, Clock, Layers } from 'lucide-react'
import { Breadcrumbs } from '@/components/layout/Breadcrumbs'
import { quantTrack } from '@/content/quant'

export const metadata = { title: 'From Zero to Quant' }

export default function QuantPage() {
  return (
    <>
      <Breadcrumbs items={[{ label: 'Quant' }]} />
      <div className="mb-10">
        <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-accent-green-subtle text-accent-green text-xs font-medium mb-4">
          <Layers className="w-3.5 h-3.5" />
          {quantTrack.moduleCount} Chapters
        </div>
        <h1 className="text-3xl sm:text-4xl font-extrabold tracking-tight text-text-primary mb-4">{quantTrack.title}</h1>
        <p className="text-lg text-text-secondary leading-relaxed max-w-3xl">{quantTrack.description}</p>
      </div>
      <div className="space-y-3">
        {quantTrack.modules?.map((module) => (
          <Link key={module.id} href={module.href}
            className="group block rounded-xl border border-border-primary bg-bg-surface hover:border-accent-green/40 hover:shadow-md transition-all duration-200 p-5">
            <div className="flex items-start gap-4">
              <div className="w-10 h-10 rounded-lg bg-accent-green-subtle flex items-center justify-center shrink-0">
                <span className="text-sm font-bold text-accent-green">{module.number}</span>
              </div>
              <div className="flex-1 min-w-0">
                <h3 className="text-base font-bold text-text-primary group-hover:text-accent-green transition-colors mb-1">{module.title}</h3>
                <p className="text-sm text-text-secondary leading-relaxed mb-2">{module.description}</p>
                <div className="flex items-center gap-3 text-xs text-text-tertiary">
                  <span className="flex items-center gap-1"><Clock className="w-3 h-3" />{module.readingTime}</span>
                </div>
              </div>
              <ArrowRight className="w-4 h-4 text-text-tertiary group-hover:text-accent-green group-hover:translate-x-1 transition-all shrink-0 mt-2" />
            </div>
          </Link>
        ))}
      </div>
    </>
  )
}
