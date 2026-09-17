import type { Metadata } from 'next'
import { notFound } from 'next/navigation'
import { getRegistry, getTrack, getPrevNext } from '@/lib/registry'
import type { Appendix, Module, Part, Track } from '@/lib/registry-types'
import { MarkdownModulePage } from '@/components/content/MarkdownModulePage'
import { HtmlModulePage } from '@/components/content/HtmlModulePage'
import { PartIndexPage } from '@/components/content/PartIndexPage'
import { AppendixIndexPage, AppendixPage } from '@/components/content/AppendixPages'
import { SourcesPage } from '@/components/content/SourcesPage'

export const dynamicParams = false

type Params = Promise<{ track: string; slug: string[] }>

type Resolved =
  | { kind: 'part'; part: Part }
  | { kind: 'module'; module: Module; part?: Part }
  | { kind: 'appendix-index' }
  | { kind: 'appendix'; appendix: Appendix }
  | { kind: 'sources' }

/**
 * URL shapes (all pre-existing URLs keep working):
 *   /<track>/<part>                 part index (hierarchical tracks)
 *   /<track>/<part>/<module>        module (hierarchical)
 *   /<track>/<module>               module (flat tracks: deep dives, quant, qualcomm)
 *   /<track>/appendices             appendix index
 *   /<track>/appendices/<id>        appendix
 *   /<track>/sources                source map (generated tracks)
 */
function resolveSlug(track: Track, slug: string[]): Resolved | null {
  const [a, b, ...rest] = slug
  if (!a || rest.length) return null
  if (a === 'appendices') {
    if (!track.appendices.length) return null
    if (!b) return { kind: 'appendix-index' }
    const appendix = track.appendices.find((x) => x.id === b)
    return appendix ? { kind: 'appendix', appendix } : null
  }
  if (a === 'sources' && !b && track.sourcesAbsPath) return { kind: 'sources' }
  if (track.parts) {
    const part = track.parts.find((p) => p.id === a)
    if (!part) return null
    if (!b) return { kind: 'part', part }
    const mod = part.modules.find((m) => m.id === b)
    return mod ? { kind: 'module', module: mod, part } : null
  }
  if (!b) {
    const mod = track.modules?.find((m) => m.id === a)
    return mod ? { kind: 'module', module: mod } : null
  }
  return null
}

export async function generateStaticParams() {
  const { tracks } = await getRegistry()
  const out: { track: string; slug: string[] }[] = []
  for (const t of tracks) {
    for (const p of t.parts ?? []) out.push({ track: t.id, slug: [p.id] })
    for (const m of t.allModules) out.push({ track: t.id, slug: m.href.split('/').slice(2) })
    if (t.appendices.length) {
      out.push({ track: t.id, slug: ['appendices'] })
      for (const a of t.appendices) out.push({ track: t.id, slug: ['appendices', a.id] })
    }
    if (t.sourcesAbsPath) out.push({ track: t.id, slug: ['sources'] })
  }
  return out
}

async function load(params: Params) {
  const { track: id, slug } = await params
  const track = await getTrack(id)
  if (!track) return null
  const r = resolveSlug(track, slug)
  return r && { track, r }
}

export async function generateMetadata({ params }: { params: Params }): Promise<Metadata> {
  const hit = await load(params)
  if (!hit) return {}
  const { track, r } = hit
  switch (r.kind) {
    case 'part': return { title: `Part ${r.part.number}: ${r.part.title}` }
    case 'module': return { title: `${track.unitLabel} ${r.module.number}: ${r.module.title}`, description: r.module.description }
    case 'appendix-index': return { title: `${track.shortTitle} Appendices` }
    case 'appendix': return { title: `Appendix ${r.appendix.letter}: ${r.appendix.title}` }
    case 'sources': return { title: `${track.shortTitle} Sources` }
  }
}

export default async function SlugPage({ params }: { params: Params }) {
  const hit = await load(params)
  if (!hit) notFound()
  const { track, r } = hit
  const root = { label: track.shortTitle, href: track.href }

  switch (r.kind) {
    case 'part':
      return <PartIndexPage part={r.part} color={track.color} breadcrumbs={[root, { label: `Part ${r.part.number}: ${r.part.shortTitle}` }]} />
    case 'module': {
      const breadcrumbs = [
        root,
        ...(r.part ? [{ label: `Part ${r.part.number}: ${r.part.shortTitle}`, href: r.part.href }] : []),
        { label: `${track.unitLabel} ${r.module.number}` },
      ]
      const common = {
        absPath: r.module.absPath,
        breadcrumbs,
        moduleNumber: r.module.number,
        title: r.module.title,
        track: track.id,
        part: r.module.partNumber,
        readingTime: r.module.readingTime,
        prerequisites: r.module.prerequisites,
        description: r.module.description,
        ...getPrevNext(track, r.module.id),
      }
      return r.module.format === 'html' ? <HtmlModulePage {...common} /> : <MarkdownModulePage {...common} />
    }
    case 'appendix-index':
      return <AppendixIndexPage track={track} breadcrumbs={[root, { label: 'Appendices' }]} />
    case 'appendix':
      return <AppendixPage track={track} appendix={r.appendix} breadcrumbs={[root, { label: 'Appendices', href: `${track.href}/appendices` }, { label: `Appendix ${r.appendix.letter}` }]} />
    case 'sources':
      return <SourcesPage track={track} breadcrumbs={[root, { label: 'Sources' }]} />
  }
}
