'use client'

import Link from 'next/link'
import { ChevronRight, Home } from 'lucide-react'

export interface BreadcrumbItem {
  label: string
  href?: string
}

interface BreadcrumbsProps {
  items: BreadcrumbItem[]
}

export function Breadcrumbs({ items }: BreadcrumbsProps) {
  return (
    <nav aria-label="Breadcrumb" className="flex items-center gap-1 text-sm py-3 px-1 overflow-x-auto">
      <Link
        href="/"
        className="text-text-tertiary hover:text-accent-blue transition-colors shrink-0"
      >
        <Home className="w-3.5 h-3.5" />
      </Link>

      {items.map((item, i) => (
        <span key={i} className="flex items-center gap-1 shrink-0">
          <ChevronRight className="w-3.5 h-3.5 text-text-tertiary" />
          {item.href ? (
            <Link
              href={item.href}
              className="text-text-tertiary hover:text-accent-blue transition-colors"
            >
              {item.label}
            </Link>
          ) : (
            <span className="text-text-primary font-medium">{item.label}</span>
          )}
        </span>
      ))}
    </nav>
  )
}
