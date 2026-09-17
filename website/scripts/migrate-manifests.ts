/**
 * One-off migration: generate manifest.json for the four legacy tracks from
 * content/*.ts (structure) + app/**\/page.tsx (rendered truth: sourcePath, title,
 * readingTime, description, prerequisites) + the local appendix arrays.
 *
 *   cd website && npx -y tsx scripts/migrate-manifests.ts
 *
 * Prints every field where content/*.ts and page.tsx disagree. Safe to re-run.
 *
 * HISTORICAL: it ran once on 2026-09-17; content/*.ts and the per-module pages it
 * reads were deleted afterwards, so it no longer compiles (scripts/ is excluded
 * from tsconfig). Kept as the record of how the four legacy manifests were made.
 */
import { readFileSync, writeFileSync, existsSync } from 'node:fs'
import path from 'node:path'
import { mlsysTrack } from '../content/mlsys'
import { intelTrack } from '../content/intel'
import { qualcommTrack } from '../content/qualcomm'
import { quantTrack } from '../content/quant'
import type { Track } from '../lib/types'

const WEB = process.cwd()
const REPO = path.resolve(WEB, '..')

interface Target {
  track: Track
  dir: string            // repo-relative content dir (manifest goes here)
  prefix: string         // sourcePath prefix to strip → file relative to dir
  category: string
  icon: string
  unitLabel: 'Module' | 'Chapter'
  appendixIndex?: string // app/<track>/appendices/page.tsx
}
const TARGETS: Target[] = [
  { track: mlsysTrack, dir: 'computer science/MLsys', prefix: 'MLsys/', category: 'Computer Science', icon: 'Cpu', unitLabel: 'Module', appendixIndex: 'app/mlsys/appendices/page.tsx' },
  { track: intelTrack, dir: 'computer science/intel', prefix: 'intel/', category: 'Computer Science', icon: 'CircuitBoard', unitLabel: 'Module', appendixIndex: 'app/intel/appendices/page.tsx' },
  { track: qualcommTrack, dir: 'computer science/qualcomm', prefix: 'qualcomm/', category: 'Computer Science', icon: 'Smartphone', unitLabel: 'Module' },
  { track: quantTrack, dir: 'From_Zero_to_Quant', prefix: 'From_Zero_to_Quant/', category: 'Finance', icon: 'TrendingUp', unitLabel: 'Chapter' },
]

const drift: string[] = []
const attr = (src: string, name: string) => src.match(new RegExp(`\\b${name}="([^"]*)"`))?.[1]

function readPage(href: string) {
  const file = path.join(WEB, 'app', href, 'page.tsx')
  const src = readFileSync(file, 'utf8')
  const prereqRaw = src.match(/prerequisites=\{(\[[\s\S]*?\])\}/)?.[1]
  const prerequisites = prereqRaw ? (JSON.parse(prereqRaw.replace(/'/g, '"').replace(/,\s*\]/, ']')) as string[]) : undefined
  return { sourcePath: attr(src, 'sourcePath'), title: attr(src, 'title'), readingTime: attr(src, 'readingTime'), description: attr(src, 'description'), prerequisites }
}

function migrateModule(t: Target, m: Track['modules'] extends (infer M)[] | undefined ? M : never, sourceFallback: string) {
  const p = readPage(m.href)
  const sourcePath = p.sourcePath ?? `${t.prefix}${sourceFallback}`
  if (!sourcePath.startsWith(t.prefix)) throw new Error(`${m.href}: sourcePath "${sourcePath}" does not start with ${t.prefix}`)
  const file = sourcePath.slice(t.prefix.length)
  if (!existsSync(path.join(REPO, t.dir, file))) throw new Error(`${m.href}: file missing on disk: ${t.dir}/${file}`)
  if (p.title && p.title !== m.title) drift.push(`${m.href}: title  manifest="${m.title}"  page="${p.title}"`)
  if (p.readingTime && p.readingTime !== m.readingTime) drift.push(`${m.href}: readingTime  manifest="${m.readingTime}"  page="${p.readingTime}"`)
  if (p.description && p.description !== m.description) drift.push(`${m.href}: description differs`)
  return {
    id: m.id, number: m.number,
    title: p.title ?? m.title, shortTitle: m.shortTitle,
    description: p.description ?? m.description,
    readingTime: p.readingTime ?? m.readingTime,
    file, format: 'md' as const,
    ...(p.prerequisites?.length ? { prerequisites: p.prerequisites } : {}),
    status: 'published' as const,
  }
}

function migrateAppendices(t: Target) {
  if (!t.appendixIndex) return []
  const src = readFileSync(path.join(WEB, t.appendixIndex), 'utf8')
  const out: { id: string; letter: string; title: string; file: string; format: 'md' }[] = []
  for (const m of src.matchAll(/\{\s*letter:\s*'([A-Z])',\s*title:\s*'([^']+)',\s*href:\s*'([^']+)'\s*\}/g)) {
    const [, letter, title, href] = m
    const page = readFileSync(path.join(WEB, 'app', href, 'page.tsx'), 'utf8')
    const lm = page.match(/loadMarkdown\('([^']+)'\)/)?.[1]
    if (!lm || !lm.startsWith(t.prefix)) throw new Error(`${href}: cannot find loadMarkdown path`)
    const file = lm.slice(t.prefix.length)
    if (!existsSync(path.join(REPO, t.dir, file))) throw new Error(`${href}: appendix file missing: ${t.dir}/${file}`)
    out.push({ id: href.split('/').pop()!, letter, title, file, format: 'md' })
  }
  return out
}

for (const t of TARGETS) {
  const tr = t.track
  const base = {
    version: 1, id: tr.id, title: tr.title, shortTitle: tr.shortTitle, description: tr.description,
    category: t.category, kind: 'course', color: tr.color, icon: t.icon, unitLabel: t.unitLabel, status: 'published',
    generator: { skill: 'legacy-migration', request: 'migrated from content/*.ts + page.tsx', createdAt: new Date().toISOString(), updatedAt: new Date().toISOString(), pipeline: { stage: 'done', approved: true } },
  }
  let body: Record<string, unknown>
  if (tr.parts) {
    body = { parts: tr.parts.map((p) => ({ id: p.id, number: p.number, title: p.title, shortTitle: p.shortTitle, description: p.description, modules: p.modules.map((m) => migrateModule(t, m, m.sourceFile)) })) }
  } else {
    body = { modules: (tr.modules ?? []).map((m) => migrateModule(t, m, m.sourceFile)) }
  }
  const appendices = migrateAppendices(t)
  const manifest = { ...base, ...body, ...(appendices.length ? { appendices } : {}) }
  const out = path.join(REPO, t.dir, 'manifest.json')
  writeFileSync(out, JSON.stringify(manifest, null, 2) + '\n')
  const count = tr.parts ? tr.parts.reduce((s, p) => s + p.modules.length, 0) : (tr.modules ?? []).length
  console.log(`wrote ${path.relative(REPO, out)}  (${count} modules, ${appendices.length} appendices)`)
}
console.log(`\n${drift.length} field(s) differ between content/*.ts and page.tsx (page value used):`)
for (const d of drift) console.log('  ' + d)
