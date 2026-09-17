import type { Metadata } from 'next'
import { notFound } from 'next/navigation'
import { Breadcrumbs } from '@/components/layout/Breadcrumbs'
import { TrackHeader, PartsGrid, ModuleCards, AppendixCards } from '@/components/content/TrackIndex'
import { getRegistry, getTrack } from '@/lib/registry'
import { colorClasses } from '@/lib/track-theme'

export const dynamicParams = false

export async function generateStaticParams() {
  return (await getRegistry()).tracks.map((t) => ({ track: t.id }))
}

export async function generateMetadata({ params }: { params: Promise<{ track: string }> }): Promise<Metadata> {
  const t = await getTrack((await params).track)
  return t ? { title: t.title, description: t.description } : {}
}

export default async function TrackIndexPage({ params }: { params: Promise<{ track: string }> }) {
  const track = await getTrack((await params).track)
  if (!track) notFound()
  const theme = colorClasses(track.color)
  return (
    <>
      <Breadcrumbs items={[{ label: track.shortTitle }]} />
      <TrackHeader track={track} />
      {track.parts ? <PartsGrid parts={track.parts} theme={theme} /> : <ModuleCards modules={track.modules ?? []} theme={theme} />}
      {track.appendices.length > 0 && <AppendixCards appendices={track.appendices} theme={theme} />}
    </>
  )
}
