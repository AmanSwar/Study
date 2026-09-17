import 'server-only'
import { promises as fs } from 'fs'
import path from 'path'
import type { Appendix, Module, NavTrack, Part, RawManifest, RawModule, RawPart, Registry, Track } from './registry-types'

/**
 * Content registry — discovers every `manifest.json` under the repository root
 * (the website lives at <repo>/website/, so content is one level up), validates
 * it, and resolves hrefs/paths. Study skills write manifests; nothing in the
 * site needs editing for a new track to appear.
 */
export const REPO_ROOT = path.resolve(process.cwd(), '..')
const SKIP_DIRS = new Set(['website', 'node_modules', '.git', '.next', 'none'])
const MAX_DEPTH = 3
const RESERVED_IDS = new Set(['study', 'search-index.json', 'favicon.ico', 'api', '_next'])
const ID_RE = /^[a-z0-9][a-z0-9-]*$/

const isVisible = (status?: string) => status === undefined || status === 'published' || process.env.NODE_ENV === 'development'

async function findManifests(dir: string, depth = 0, out: string[] = []): Promise<string[]> {
  if (depth > MAX_DEPTH) return out
  let entries: import('fs').Dirent[]
  try { entries = await fs.readdir(dir, { withFileTypes: true }) } catch { return out }
  for (const e of entries) {
    if (e.isFile() && e.name === 'manifest.json') out.push(path.join(dir, e.name))
    else if (e.isDirectory() && !SKIP_DIRS.has(e.name) && !e.name.startsWith('.')) await findManifests(path.join(dir, e.name), depth + 1, out)
  }
  return out
}

function validate(raw: RawManifest, file: string) {
  const fail = (msg: string) => { throw new Error(`Invalid manifest ${file}: ${msg}`) }
  if (raw.version !== 1) fail('version must be 1')
  for (const k of ['id', 'title', 'shortTitle', 'description', 'category', 'kind', 'color', 'icon'] as const) if (!raw[k]) fail(`missing "${k}"`)
  if (!ID_RE.test(raw.id)) fail(`id "${raw.id}" must match ${ID_RE}`)
  if (RESERVED_IDS.has(raw.id)) fail(`id "${raw.id}" is reserved`)
  if (!!raw.modules === !!raw.parts) fail('exactly one of "modules" or "parts" is required')
  const seen = new Set<string>()
  const checkModule = (m: RawModule, where: string) => {
    for (const k of ['id', 'title', 'shortTitle', 'file', 'format'] as const) if (!m[k]) fail(`${where}: missing "${k}"`)
    if (!ID_RE.test(m.id)) fail(`${where}: id "${m.id}" must match ${ID_RE}`)
    if (seen.has(m.id)) fail(`${where}: duplicate module id "${m.id}"`)
    seen.add(m.id)
    if (m.format !== 'html' && m.format !== 'md') fail(`${where}: format must be html | md`)
  }
  raw.modules?.forEach((m, i) => checkModule(m, `modules[${i}]`))
  raw.parts?.forEach((p, i) => { if (!p.id || !p.modules?.length) fail(`parts[${i}] needs id and modules`); p.modules.forEach((m, j) => checkModule(m, `parts[${i}].modules[${j}]`)) })
}

function resolveTrack(raw: RawManifest, manifestPath: string): Track {
  const manifestDir = path.dirname(manifestPath)
  const href = `/${raw.id}`
  const unitLabel = raw.unitLabel ?? 'Module'
  const mkModule = (m: RawModule, part?: RawPart): Module => ({
    ...m,
    readingTime: m.readingTime ?? '',
    partId: part?.id,
    partNumber: part?.number ?? 0,
    href: part ? `${href}/${part.id}/${m.id}` : `${href}/${m.id}`,
    absPath: path.join(manifestDir, m.file),
  })
  const parts: Part[] | undefined = raw.parts?.map((p) => ({
    ...p,
    href: `${href}/${p.id}`,
    modules: p.modules.filter((m) => isVisible(m.status)).map((m) => mkModule(m, p)),
  })).filter((p) => p.modules.length > 0)
  const modules = raw.modules?.filter((m) => isVisible(m.status)).map((m) => mkModule(m))
  const allModules = parts ? parts.flatMap((p) => p.modules) : modules ?? []
  const appendices: Appendix[] = (raw.appendices ?? []).map((a) => ({
    ...a,
    format: a.format ?? 'md',
    href: `${href}/appendices/${a.id}`,
    absPath: path.join(manifestDir, a.file),
  }))
  const { modules: _m, parts: _p, appendices: _a, unitLabel: _u, ...rest } = raw
  void _m; void _p; void _a; void _u
  return {
    ...rest,
    href,
    manifestDir,
    unitLabel,
    parts,
    modules,
    appendices,
    allModules,
    moduleCount: allModules.length,
    appendixCount: appendices.length,
    partCount: parts?.length ?? 0,
    sourcesAbsPath: raw.sources ? path.join(manifestDir, raw.sources) : undefined,
  }
}

const CATEGORY_ORDER = ['Computer Science', 'Finance', 'Mathematics', 'Business']

async function buildRegistry(): Promise<Registry> {
  const files = (await findManifests(REPO_ROOT)).sort()
  const tracks: Track[] = []
  for (const f of files) {
    const raw = JSON.parse(await fs.readFile(f, 'utf-8')) as RawManifest
    validate(raw, f)
    if (!isVisible(raw.status)) continue
    const track = resolveTrack(raw, f)
    if (track.allModules.length === 0) continue
    tracks.push(track)
  }
  tracks.sort((a, b) => {
    const ca = CATEGORY_ORDER.indexOf(a.category), cb = CATEGORY_ORDER.indexOf(b.category)
    return (ca === -1 ? 99 : ca) - (cb === -1 ? 99 : cb) || a.category.localeCompare(b.category) || (a.order ?? 100) - (b.order ?? 100) || a.title.localeCompare(b.title)
  })
  const byId: Record<string, Track> = {}
  for (const t of tracks) {
    if (byId[t.id]) throw new Error(`Duplicate track id "${t.id}" (${t.manifestDir} and ${byId[t.id].manifestDir})`)
    byId[t.id] = t
  }
  return { tracks, byId }
}

// One directory walk per process (per build worker). Not memoised in dev so a
// newly generated track shows up on refresh.
let cached: Promise<Registry> | null = null
export function getRegistry(): Promise<Registry> {
  if (process.env.NODE_ENV === 'development') return buildRegistry()
  return (cached ??= buildRegistry())
}

export async function getTrack(id: string): Promise<Track | undefined> {
  return (await getRegistry()).byId[id]
}

export function getPrevNext(track: Track, moduleId: string) {
  const i = track.allModules.findIndex((m) => m.id === moduleId)
  const toLink = (m?: Module) => (m ? { href: m.href, label: m.title, number: m.number } : undefined)
  return { prev: toLink(track.allModules[i - 1]), next: toLink(track.allModules[i + 1]), index: i, count: track.allModules.length }
}

/** Strip filesystem paths before handing a track to client components. */
export function toNavTrack(t: Track): NavTrack {
  const nm = (m: Module) => ({ id: m.id, number: m.number, shortTitle: m.shortTitle, href: m.href, readingTime: m.readingTime })
  return {
    id: t.id,
    title: t.title,
    shortTitle: t.shortTitle,
    href: t.href,
    color: t.color,
    unitLabel: t.unitLabel,
    parts: t.parts?.map((p) => ({ id: p.id, number: p.number, shortTitle: p.shortTitle, href: p.href, modules: p.modules.map(nm) })),
    modules: t.modules?.map(nm),
    appendices: t.appendices.map((a) => ({ id: a.id, letter: a.letter, title: a.title, href: a.href })),
    hasSources: !!t.sourcesAbsPath,
  }
}
