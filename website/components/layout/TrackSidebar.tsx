'use client'

import Link from 'next/link'
import { usePathname } from 'next/navigation'
import { useState, useEffect, useRef } from 'react'
import { ChevronRight } from 'lucide-react'
import { Track, Part, ModuleMeta } from '@/lib/types'

interface TrackSidebarProps {
  track: Track
}

export function TrackSidebar({ track }: TrackSidebarProps) {
  const pathname = usePathname()
  const activeItemRef = useRef<HTMLAnchorElement | null>(null)

  // Scroll active item into view on mount
  useEffect(() => {
    if (activeItemRef.current) {
      activeItemRef.current.scrollIntoView({ block: 'nearest', behavior: 'instant' })
    }
  }, [])

  return (
    <nav className="py-4 pr-2">
      {/* Track title */}
      <Link
        href={`/${track.id}`}
        className={`block px-4 py-2 text-[11px] font-semibold uppercase tracking-[0.1em] transition-colors
          ${pathname === `/${track.id}`
            ? 'text-accent-blue'
            : 'text-text-tertiary hover:text-text-secondary'}`}
      >
        {track.shortTitle}
      </Link>

      <div className="mt-2 space-y-0.5">
        {/* Parts */}
        {track.parts?.map((part) => (
          <PartSection
            key={part.id}
            part={part}
            trackId={track.id}
            currentPath={pathname}
            activeItemRef={activeItemRef}
          />
        ))}

        {/* Flat modules (Qualcomm, Quant) */}
        {track.modules?.map((module) => (
          <ModuleLink
            key={module.id}
            module={module}
            isActive={pathname === module.href}
            activeItemRef={activeItemRef}
          />
        ))}
      </div>
    </nav>
  )
}

function PartSection({
  part,
  trackId,
  currentPath,
  activeItemRef,
}: {
  part: Part
  trackId: string
  currentPath: string
  activeItemRef: React.MutableRefObject<HTMLAnchorElement | null>
}) {
  const partPath = `/${trackId}/${part.id}`
  const isInPart = currentPath.startsWith(partPath)
  const [isOpen, setIsOpen] = useState(isInPart)

  return (
    <div>
      <button
        onClick={() => setIsOpen(!isOpen)}
        className={`w-full flex items-center gap-1.5 px-3 py-1.5 text-sm font-medium rounded-md mx-1 transition-all
          ${isInPart ? 'text-text-primary' : 'text-text-secondary hover:text-text-primary'}
          hover:bg-bg-surface-hover`}
      >
        <ChevronRight
          className={`w-3.5 h-3.5 shrink-0 text-text-tertiary transition-transform duration-200 ${isOpen ? 'rotate-90' : ''}`}
        />
        <span className="text-left flex items-baseline gap-1.5 truncate">
          <span className="text-[10px] font-semibold text-text-tertiary tabular-nums">
            {String(part.number).padStart(2, '0')}
          </span>
          <span className="truncate">{part.shortTitle}</span>
        </span>
      </button>

      {isOpen && (
        <div className="ml-4 border-l border-border-primary pl-2 py-1 space-y-0.5 animate-fade-in">
          {part.modules.map((module) => (
            <ModuleLink
              key={module.id}
              module={module}
              isActive={currentPath === module.href}
              activeItemRef={activeItemRef}
            />
          ))}
        </div>
      )}
    </div>
  )
}

function ModuleLink({
  module,
  isActive,
  activeItemRef,
}: {
  module: ModuleMeta
  isActive: boolean
  activeItemRef: React.MutableRefObject<HTMLAnchorElement | null>
}) {
  return (
    <Link
      ref={isActive ? activeItemRef : null}
      href={module.href}
      className={`flex items-center gap-2.5 px-3 py-1.5 text-sm rounded-md mx-1 transition-all
        ${isActive
          ? 'bg-accent-blue-subtle text-accent-blue font-medium'
          : 'text-text-secondary hover:text-text-primary hover:bg-bg-surface-hover'}`}
    >
      <span className={`text-[10px] font-mono tabular-nums shrink-0 ${
        isActive ? 'text-accent-blue' : 'text-text-tertiary'
      }`}>
        {String(module.number).padStart(2, '0')}
      </span>
      <span className="truncate">{module.shortTitle}</span>
    </Link>
  )
}
