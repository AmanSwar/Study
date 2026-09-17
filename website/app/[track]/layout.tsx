import { notFound } from 'next/navigation'
import { getRegistry, toNavTrack } from '@/lib/registry'
import { TrackShell } from '@/components/layout/TrackShell'
import { TrackSidebar } from '@/components/layout/TrackSidebar'
import { Footer } from '@/components/layout/Footer'

export default async function TrackLayout({ children, params }: { children: React.ReactNode; params: Promise<{ track: string }> }) {
  const { track: id } = await params
  const reg = await getRegistry()
  const track = reg.byId[id]
  if (!track) notFound()
  const footerTracks = reg.tracks.map((t) => ({ id: t.id, shortTitle: t.shortTitle }))
  return (
    <TrackShell sidebar={<TrackSidebar track={toNavTrack(track)} />} footer={<Footer tracks={footerTracks} />}>
      {children}
    </TrackShell>
  )
}
