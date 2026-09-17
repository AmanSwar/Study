import { notFound } from 'next/navigation'
import { getRegistry, toNavTrack } from '@/lib/registry'
import { TrackShell } from '@/components/layout/TrackShell'
import { Footer } from '@/components/layout/Footer'

export default async function TrackLayout({ children, params }: { children: React.ReactNode; params: Promise<{ track: string }> }) {
  const { track: id } = await params
  const reg = await getRegistry()
  const track = reg.byId[id]
  if (!track) notFound()
  return (
    <TrackShell track={toNavTrack(track)} footer={<Footer />}>
      {children}
    </TrackShell>
  )
}
