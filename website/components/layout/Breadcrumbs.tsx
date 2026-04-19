'use client'

import Link from 'next/link'
import { Home } from 'lucide-react'

export interface BreadcrumbItem {
  label: string
  href?: string
}

interface BreadcrumbsProps {
  items: BreadcrumbItem[]
}

export function Breadcrumbs({ items }: BreadcrumbsProps) {
  return (
    <nav
      aria-label="Breadcrumb"
      className="flex items-center gap-1.5 text-xs py-4 overflow-x-auto text-text-tertiary"
    >
      <Link
        href="/"
        className="flex items-center shrink-0 hover:text-text-primary transition-colors"
        aria-label="Home"
      >
        <Home className="w-3.5 h-3.5" />
      </Link>

      {items.map((item, i) => (
        <span key={i} className="flex items-center gap-1.5 shrink-0">
          <span className="text-text-tertiary/50" aria-hidden="true">/</span>
          {item.href ? (
            <Link
              href={item.href}
              className="hover:text-text-primary transition-colors"
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
