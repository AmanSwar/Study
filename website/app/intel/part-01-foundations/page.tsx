import { PartIndexPage } from '@/components/content/PartIndexPage'
import { intelTrack } from '@/content/intel'

export const metadata = { title: 'Part 1: Computer Systems Foundations' }

const part = intelTrack.parts![0]

export default function Part01Page() {
  return (
    <PartIndexPage
      part={part}
      trackId="intel"
      trackTitle="Intel/AMD"
      breadcrumbs={[
        { label: 'Intel/AMD', href: '/intel' },
        { label: `Part ${part.number}: ${part.shortTitle}` },
      ]}
    />
  )
}
