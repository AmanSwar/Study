import {
  Cpu, CircuitBoard, Microchip, Smartphone, TrendingUp, BookOpen, Server, Brain, Network, Database, Layers, Zap, Globe, Sigma, type LucideIcon,
} from 'lucide-react'
import type { TrackColor } from './registry-types'

/** Manifests reference icons by lucide name; resolve them here (server components only). */
export const ICONS: Record<string, LucideIcon> = { Cpu, CircuitBoard, Microchip, Smartphone, TrendingUp, BookOpen, Server, Brain, Network, Database, Layers, Zap, Globe, Sigma }
export const iconFor = (name: string): LucideIcon => ICONS[name] ?? BookOpen

/**
 * Full literal class strings per colour — Tailwind v4 only emits classes it can
 * see in source, so these must never be built dynamically.
 */
export const COLORS = {
  blue:   { text: 'text-accent-blue',   subtle: 'bg-accent-blue-subtle',   hoverBorder: 'hover:border-accent-blue/40',   hoverText: 'group-hover:text-accent-blue',   gradient: 'from-blue-600 to-cyan-500',    badge: 'bg-blue-500/20 text-blue-500' },
  cyan:   { text: 'text-accent-cyan',   subtle: 'bg-accent-cyan-subtle',   hoverBorder: 'hover:border-accent-cyan/40',   hoverText: 'group-hover:text-accent-cyan',   gradient: 'from-cyan-500 to-teal-400',    badge: 'bg-cyan-500/20 text-cyan-600' },
  orange: { text: 'text-accent-orange', subtle: 'bg-accent-orange-subtle', hoverBorder: 'hover:border-accent-orange/40', hoverText: 'group-hover:text-accent-orange', gradient: 'from-orange-500 to-amber-400', badge: 'bg-orange-500/20 text-orange-500' },
  green:  { text: 'text-accent-green',  subtle: 'bg-accent-green-subtle',  hoverBorder: 'hover:border-accent-green/40',  hoverText: 'group-hover:text-accent-green',  gradient: 'from-green-500 to-emerald-400', badge: 'bg-green-500/20 text-green-600' },
  amber:  { text: 'text-accent-amber',  subtle: 'bg-accent-amber-subtle',  hoverBorder: 'hover:border-accent-amber/40',  hoverText: 'group-hover:text-accent-amber',  gradient: 'from-amber-500 to-yellow-400',  badge: 'bg-amber-500/20 text-amber-600' },
  red:    { text: 'text-accent-red',    subtle: 'bg-accent-red-subtle',    hoverBorder: 'hover:border-accent-red/40',    hoverText: 'group-hover:text-accent-red',    gradient: 'from-red-500 to-rose-400',      badge: 'bg-red-500/20 text-red-500' },
  purple: { text: 'text-purple-500',    subtle: 'bg-purple-500/10',        hoverBorder: 'hover:border-purple-500/40',    hoverText: 'group-hover:text-purple-500',    gradient: 'from-purple-600 to-fuchsia-500', badge: 'bg-purple-500/20 text-purple-500' },
} as const

export type ColorClasses = (typeof COLORS)[TrackColor]
export const colorClasses = (c: string): ColorClasses => COLORS[c as TrackColor] ?? COLORS.blue
