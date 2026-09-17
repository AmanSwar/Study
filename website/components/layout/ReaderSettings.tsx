'use client'

import { useEffect, useRef, useState } from 'react'
import { useTheme } from 'next-themes'

export const READER_KEY = 'aman.study:reader'

type Font = 'serif' | 'sans'
type Width = 'narrow' | 'normal' | 'wide'
interface Prefs { font?: Font; size?: number; width?: Width }

const SIZES = [15, 16, 17, 18, 19, 20, 21]
const DEFAULT_SIZE: Record<Font, number> = { serif: 18, sans: 17 }

function readPrefs(): Prefs {
  try { return JSON.parse(localStorage.getItem(READER_KEY) || '{}') } catch { return {} }
}

// The same attributes are set before paint by the bootstrap script in app/layout.tsx.
function apply(p: Prefs) {
  const h = document.documentElement
  if (p.font && p.font !== 'serif') h.dataset.font = p.font; else delete h.dataset.font
  if (p.size) h.dataset.size = String(p.size); else delete h.dataset.size
  if (p.width && p.width !== 'normal') h.dataset.width = p.width; else delete h.dataset.width
  try { localStorage.setItem(READER_KEY, JSON.stringify(p)) } catch {}
}

/** "Aa" popover: typeface, size, measure and theme — remembered on this device. */
export function ReaderSettings() {
  const [open, setOpen] = useState(false)
  // Read once on the client; the popover is closed during hydration so server/client state never disagree on screen.
  const [prefs, setPrefs] = useState<Prefs>(() => (typeof window === 'undefined' ? {} : readPrefs()))
  const { theme, setTheme } = useTheme()
  const ref = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (!open) return
    const onDown = (e: MouseEvent) => { if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false) }
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') setOpen(false) }
    document.addEventListener('mousedown', onDown)
    document.addEventListener('keydown', onKey)
    return () => { document.removeEventListener('mousedown', onDown); document.removeEventListener('keydown', onKey) }
  }, [open])

  const update = (patch: Prefs) => {
    const next = { ...prefs, ...patch }
    setPrefs(next)
    apply(next)
  }

  const font: Font = prefs.font ?? 'serif'
  const size = prefs.size ?? DEFAULT_SIZE[font]
  const width: Width = prefs.width ?? 'normal'
  const step = (d: number) => {
    const i = Math.min(SIZES.length - 1, Math.max(0, SIZES.indexOf(size) + d))
    update({ size: SIZES[i] })
  }

  return (
    <div ref={ref} className="relative">
      <button
        type="button"
        onClick={() => setOpen((o) => !o)}
        aria-expanded={open}
        aria-label="Reader settings"
        className="inline-flex items-center justify-center h-8 px-2.5 rounded-md font-serif font-semibold tracking-tight text-[15px] hover:bg-bg-surface-hover hover:text-text-primary transition-colors"
      >
        Aa
      </button>

      {open && (
        <div
          role="dialog"
          aria-label="Reader settings"
          className="ui absolute right-0 top-10 z-50 w-72 p-4 rounded-lg bg-bg-elevated border border-border-primary shadow-lg font-sans text-[13px] text-text-secondary animate-fade-in"
        >
          <Row label="Font">
            <Seg>
              <SegButton on={font === 'serif'} onClick={() => update({ font: 'serif', size: prefs.size ?? DEFAULT_SIZE.serif })} className="font-serif">Serif</SegButton>
              <SegButton on={font === 'sans'} onClick={() => update({ font: 'sans', size: prefs.size ?? DEFAULT_SIZE.sans })}>Sans</SegButton>
            </Seg>
          </Row>
          <Row label="Size">
            <Seg>
              <SegButton onClick={() => step(-1)} disabled={size === SIZES[0]} aria-label="Smaller text">A−</SegButton>
              <SegButton on className="tabular-nums">{size}</SegButton>
              <SegButton onClick={() => step(1)} disabled={size === SIZES[SIZES.length - 1]} aria-label="Larger text">A+</SegButton>
            </Seg>
          </Row>
          <Row label="Width">
            <Seg>
              {(['narrow', 'normal', 'wide'] as Width[]).map((w) => (
                <SegButton key={w} on={width === w} onClick={() => update({ width: w })}>{w[0].toUpperCase() + w.slice(1)}</SegButton>
              ))}
            </Seg>
          </Row>
          <Row label="Theme">
            <Seg>
              {(['light', 'dark', 'system'] as const).map((t) => (
                <SegButton key={t} on={theme === t} onClick={() => setTheme(t)}>{t[0].toUpperCase() + t.slice(1)}</SegButton>
              ))}
            </Seg>
          </Row>
          <p className="mt-3 text-[11px] leading-relaxed text-text-tertiary">
            Saved on this device. Press <kbd>?</kbd> for keyboard shortcuts.
          </p>
        </div>
      )}
    </div>
  )
}

function Row({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="grid grid-cols-[4rem_1fr] items-center gap-2 my-2">
      <span className="text-[11px] uppercase tracking-[0.06em] text-text-tertiary">{label}</span>
      {children}
    </div>
  )
}

function Seg({ children }: { children: React.ReactNode }) {
  return <div className="flex rounded-md border border-border-primary overflow-hidden">{children}</div>
}

function SegButton({ on, className = '', children, ...rest }: React.ButtonHTMLAttributes<HTMLButtonElement> & { on?: boolean }) {
  return (
    <button
      type="button"
      className={`flex-1 py-1.5 text-center border-l border-border-primary first:border-l-0 transition-colors disabled:opacity-40
        ${on ? 'bg-bg-surface-hover text-text-primary font-semibold' : 'hover:text-text-primary'} ${className}`}
      {...rest}
    >
      {children}
    </button>
  )
}
