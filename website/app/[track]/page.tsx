import type { Metadata } from 'next'
import { notFound } from 'next/navigation'
import { TrackHeader, PartsGrid, ModuleCards, AppendixCards } from '@/components/content/TrackIndex'
import { getRegistry, getTrack } from '@/lib/registry'

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
  return (
    <div className="max-w-[46rem] mx-auto px-6 sm:px-8 pt-10 pb-24">
      <TrackHeader track={track} />
      {track.parts ? <PartsGrid parts={track.parts} /> : <ModuleCards modules={track.modules ?? []} />}
      {track.appendices.length > 0 && <AppendixCards appendices={track.appendices} />}
    </div>
  )
}
