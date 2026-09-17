'use client'

import { useEffect, useState } from 'react'
import { X } from 'lucide-react'
import { getPosition, savePosition } from '@/lib/reading-state'

interface Props {
  href: string
  title: string
  track: string
  /** h2s in document order, for "resume at § …" */
  sections: { id: string; title: string }[]
}

/**
 * Remembers how far the reader got in this module and offers to take them back
 * there on the next visit. Position is the scroll percentage; the section is the
 * last h2 above the reading line. Reaching ~92% marks the module read.
 */
export function ReadingPosition({ href, title, track, sections }: Props) {
  const [resume, setResume] = useState<{ pct: number; section?: { id: string; title: string } } | null>(null)

  useEffect(() => {
    const offer = window.setTimeout(() => {
      const saved = getPosition(href)
      if (saved && saved.pct >= 5 && saved.pct < 92 && !window.location.hash && window.scrollY < window.innerHeight * 0.6) {
        setResume({ pct: saved.pct, section: saved.section })
      }
    }, 300)

    let raf: number | null = null
    let last = -1
    let moved = false
    const measure = () => {
      raf = null
      const max = document.documentElement.scrollHeight - window.innerHeight
      const pct = max > 0 ? Math.round(Math.min(100, (window.scrollY / max) * 100)) : 100
      if (pct === last) return
      last = pct
      // opening a module at the top must not erase where the reader had got to
      const saved = getPosition(href)
      if (!moved && saved && pct < saved.pct) return
      const line = window.scrollY + 140
      let section: { id: string; title: string } | undefined
      for (const s of sections) {
        const el = document.getElementById(s.id)
        if (el && el.getBoundingClientRect().top + window.scrollY <= line) section = s
      }
      savePosition({ href, title, track, pct, section, done: (saved?.done ?? false) || pct >= 92 })
    }
    const onScroll = () => { moved = true; if (raf === null) raf = requestAnimationFrame(measure) }
    const timer = window.setTimeout(measure, 1500)
    window.addEventListener('scroll', onScroll, { passive: true })
    return () => {
      window.clearTimeout(offer)
      window.clearTimeout(timer)
      window.removeEventListener('scroll', onScroll)
      if (raf !== null) cancelAnimationFrame(raf)
    }
  }, [href, title, track, sections])

  useEffect(() => {
    if (!resume) return
    const dismiss = () => setResume(null)
    // scrolling past the first screen means they chose to start over
    const onScroll = () => { if (window.scrollY > window.innerHeight * 0.6) dismiss() }
    const t = window.setTimeout(dismiss, 15000)
    window.addEventListener('scroll', onScroll, { passive: true })
    return () => { window.clearTimeout(t); window.removeEventListener('scroll', onScroll) }
  }, [resume])

  if (!resume) return null

  const go = () => {
    const el = resume.section ? document.getElementById(resume.section.id) : null
    if (el) el.scrollIntoView({ block: 'start' })
    else {
      const max = document.documentElement.scrollHeight - window.innerHeight
      window.scrollTo({ top: (resume.pct / 100) * max })
    }
    setResume(null)
  }

  return (
    <div
      role="status"
      className="ui no-print fixed right-5 bottom-5 z-40 flex items-center gap-2 pl-4 pr-1.5 py-1.5 rounded-full bg-bg-elevated border border-border-primary shadow-md font-sans text-[13px] text-text-primary animate-fade-in"
    >
      <span className="max-w-[18rem] truncate">
        Resume {resume.section ? <b className="font-semibold">§ {resume.section.title}</b> : 'where you left off'}
        <span className="text-text-tertiary"> · {resume.pct}%</span>
      </span>
      <button type="button" onClick={go} className="rounded-full bg-accent-blue text-text-inverse px-3 py-1 text-[12.5px] font-medium hover:bg-accent-blue-hover transition-colors">
        Continue
      </button>
      <button type="button" onClick={() => setResume(null)} aria-label="Dismiss" className="w-7 h-7 inline-flex items-center justify-center rounded-full text-text-tertiary hover:text-text-primary hover:bg-bg-surface-hover">
        <X className="w-3.5 h-3.5" />
      </button>
    </div>
  )
}
