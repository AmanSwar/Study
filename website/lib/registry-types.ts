/**
 * Types for the content registry. No Node imports here — this file is safe to
 * import from client components. The filesystem side lives in lib/registry.ts.
 */
export type ContentFormat = 'html' | 'md'
export type TrackColor = 'blue' | 'cyan' | 'orange' | 'green' | 'amber' | 'red' | 'purple'
export type TrackKind = 'course' | 'deep-dive'
export type UnitLabel = 'Module' | 'Chapter' | 'Page'

// ---- Raw manifest.json shapes (what the study skills write) ----
export interface RawModule {
  id: string
  number: number
  title: string
  shortTitle: string
  description: string
  readingTime?: string
  file: string
  format: ContentFormat
  prerequisites?: string[]
  status?: 'planned' | 'writing' | 'published' | 'failed'
}
export interface RawPart {
  id: string
  number: number
  title: string
  shortTitle: string
  description: string
  modules: RawModule[]
}
export interface RawAppendix {
  id: string
  letter: string
  title: string
  file: string
  format?: ContentFormat
}
export interface RawManifest {
  version: 1
  id: string
  title: string
  shortTitle: string
  description: string
  category: string
  kind: TrackKind
  color: TrackColor
  icon: string
  unitLabel?: UnitLabel
  status?: 'researching' | 'planned' | 'writing' | 'published'
  /** optional sort key within a category (lower first; default 100) */
  order?: number
  sources?: string
  modules?: RawModule[]
  parts?: RawPart[]
  appendices?: RawAppendix[]
}

// ---- Resolved shapes (what components consume; plain JSON, serialisable) ----
export interface Module extends RawModule {
  href: string
  absPath: string
  partId?: string
  partNumber: number
  readingTime: string
}
export interface Part extends Omit<RawPart, 'modules'> {
  href: string
  modules: Module[]
}
export interface Appendix extends RawAppendix {
  href: string
  absPath: string
  format: ContentFormat
}
export interface Track extends Omit<RawManifest, 'modules' | 'parts' | 'appendices' | 'unitLabel'> {
  href: string
  manifestDir: string
  unitLabel: UnitLabel
  parts?: Part[]
  modules?: Module[]
  appendices: Appendix[]
  /** every visible module in reading order — prev/next source */
  allModules: Module[]
  moduleCount: number
  appendixCount: number
  partCount: number
  sourcesAbsPath?: string
}
export interface Registry {
  tracks: Track[]
  byId: Record<string, Track>
}

/** Sidebar/footer only need the navigational subset — keeps fs paths out of the client payload. */
export interface NavModule { id: string; number: number; shortTitle: string; href: string; readingTime: string }
export interface NavPart { id: string; number: number; shortTitle: string; href: string; modules: NavModule[] }
export interface NavTrack {
  id: string
  title: string
  shortTitle: string
  href: string
  color: TrackColor
  unitLabel: UnitLabel
  parts?: NavPart[]
  modules?: NavModule[]
  appendices: { id: string; letter: string; title: string; href: string }[]
  hasSources: boolean
}
