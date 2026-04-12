'use client'

import Link from 'next/link'
import { usePathname } from 'next/navigation'
import { useState } from 'react'
import { ChevronRight, ChevronDown, Circle, CheckCircle2 } from 'lucide-react'
import { Track, Part, ModuleMeta } from '@/lib/types'

interface TrackSidebarProps {
  track: Track
}

export function TrackSidebar({ track }: TrackSidebarProps) {
  const pathname = usePathname()

  return (
    <nav className="py-4 space-y-1">
      {/* Track title */}
      <Link
        href={`/${track.id}`}
        className={`block px-4 py-2 text-sm font-bold uppercase tracking-wider
          ${pathname === `/${track.id}`
            ? 'text-accent-blue'
            : 'text-text-tertiary hover:text-text-secondary'
          } transition-colors`}
      >
        {track.shortTitle} Track
      </Link>

      {/* Parts */}
      {track.parts?.map((part) => (
        <PartSection
          key={part.id}
          part={part}
          trackId={track.id}
          currentPath={pathname}
        />
      ))}

      {/* Flat modules (Qualcomm style) */}
      {track.modules?.map((module) => (
        <ModuleLink
          key={module.id}
          module={module}
          isActive={pathname === module.href}
        />
      ))}
    </nav>
  )
}

function PartSection({
  part,
  trackId,
  currentPath,
}: {
  part: Part
  trackId: string
  currentPath: string
}) {
  const partPath = `/${trackId}/${part.id}`
  const isInPart = currentPath.startsWith(partPath)
  const [isOpen, setIsOpen] = useState(isInPart)

  return (
    <div>
      <button
        onClick={() => setIsOpen(!isOpen)}
        className={`w-full flex items-center gap-2 px-4 py-2 text-sm font-medium
          ${isInPart ? 'text-text-primary' : 'text-text-secondary'}
          hover:text-text-primary hover:bg-bg-surface-hover transition-colors rounded-md mx-1`}
      >
        {isOpen ? (
          <ChevronDown className="w-4 h-4 shrink-0 text-text-tertiary" />
        ) : (
          <ChevronRight className="w-4 h-4 shrink-0 text-text-tertiary" />
        )}
        <span className="text-left">
          <span className="text-accent-blue font-semibold">Part {part.number}</span>
          {' '}
          <span className="text-text-secondary">{part.shortTitle}</span>
        </span>
      </button>

      {isOpen && (
        <div className="ml-4 border-l border-border-primary pl-2 space-y-0.5 py-1">
          {part.modules.map((module) => (
            <ModuleLink
              key={module.id}
              module={module}
              isActive={currentPath === module.href}
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
}: {
  module: ModuleMeta
  isActive: boolean
}) {
  return (
    <Link
      href={module.href}
      className={`flex items-center gap-2 px-3 py-1.5 text-sm rounded-md mx-1 transition-all
        ${isActive
          ? 'bg-accent-blue-subtle text-accent-blue font-medium'
          : 'text-text-secondary hover:text-text-primary hover:bg-bg-surface-hover'
        }`}
    >
      <Circle className="w-3 h-3 shrink-0" />
      <span className="truncate">
        <span className="font-medium">{module.number}.</span> {module.shortTitle}
      </span>
    </Link>
  )
}
