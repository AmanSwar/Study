import Link from 'next/link'
import { ArrowRight, Clock, BookOpen } from 'lucide-react'
import { Breadcrumbs, BreadcrumbItem } from '@/components/layout/Breadcrumbs'
import { Part } from '@/lib/types'

interface PartIndexPageProps {
  part: Part
  trackId: string
  trackTitle: string
  breadcrumbs: BreadcrumbItem[]
}

export function PartIndexPage({ part, trackId, trackTitle, breadcrumbs }: PartIndexPageProps) {
  return (
    <>
      <Breadcrumbs items={breadcrumbs} />

      {/* Part Header */}
      <div className="mb-8">
        <div className="inline-flex items-center gap-2 text-accent-blue text-sm font-semibold mb-2">
          Part {part.number}
        </div>
        <h1 className="text-3xl font-extrabold tracking-tight text-text-primary mb-3">
          {part.title}
        </h1>
        <p className="text-lg text-text-secondary">{part.description}</p>
      </div>

      {/* Module List */}
      <div className="space-y-3">
        {part.modules.map((module, i) => (
          <Link
            key={module.id}
            href={module.href}
            className="group block rounded-xl border border-border-primary bg-bg-surface
              hover:border-accent-blue/40 hover:shadow-md transition-all duration-200 p-5"
            style={{ animationDelay: `${i * 50}ms` }}
          >
            <div className="flex items-start gap-4">
              {/* Module number */}
              <div className="w-10 h-10 rounded-lg bg-accent-blue-subtle
                flex items-center justify-center shrink-0">
                <span className="text-sm font-bold text-accent-blue">{module.number}</span>
              </div>

              <div className="flex-1 min-w-0">
                <h3 className="text-base font-bold text-text-primary group-hover:text-accent-blue transition-colors mb-1">
                  {module.title}
                </h3>
                <p className="text-sm text-text-secondary leading-relaxed mb-2">
                  {module.description}
                </p>
                <div className="flex items-center gap-3 text-xs text-text-tertiary">
                  <span className="flex items-center gap-1">
                    <Clock className="w-3 h-3" />
                    {module.readingTime}
                  </span>
                </div>
              </div>

              <ArrowRight className="w-4 h-4 text-text-tertiary group-hover:text-accent-blue
                group-hover:translate-x-1 transition-all shrink-0 mt-2" />
            </div>
          </Link>
        ))}
      </div>
    </>
  )
}
