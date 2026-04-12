'use client'

import { ReactNode } from 'react'
import { Link2 } from 'lucide-react'

interface TheorySectionProps {
  id: string
  title: string
  children: ReactNode
}

export function TheorySection({ id, title, children }: TheorySectionProps) {
  return (
    <section id={id} className="scroll-mt-20 mb-10">
      <h2 className="group flex items-center gap-2 text-[1.75rem] font-bold tracking-tight text-text-primary
        border-b border-border-primary pb-3 mb-6 mt-12 first:mt-0">
        <a href={`#${id}`} className="hover:text-accent-blue transition-colors">
          {title}
        </a>
        <a
          href={`#${id}`}
          className="opacity-0 group-hover:opacity-100 transition-opacity"
          aria-label={`Link to ${title}`}
        >
          <Link2 className="w-4 h-4 text-text-tertiary hover:text-accent-blue" />
        </a>
      </h2>
      <div className="prose">{children}</div>
    </section>
  )
}

interface SubSectionProps {
  id: string
  title: string
  children: ReactNode
}

export function SubSection({ id, title, children }: SubSectionProps) {
  return (
    <div id={id} className="scroll-mt-20 mb-8">
      <h3 className="group flex items-center gap-2 text-[1.375rem] font-semibold text-text-primary mb-4 mt-8">
        <a href={`#${id}`} className="hover:text-accent-blue transition-colors">
          {title}
        </a>
        <a
          href={`#${id}`}
          className="opacity-0 group-hover:opacity-100 transition-opacity"
          aria-label={`Link to ${title}`}
        >
          <Link2 className="w-3.5 h-3.5 text-text-tertiary hover:text-accent-blue" />
        </a>
      </h3>
      <div className="prose">{children}</div>
    </div>
  )
}
