import { PartIndexPage } from '@/components/content/PartIndexPage'
import { intelTrack } from '@/content/intel'

export const metadata = { title: 'Part 6: AMD EPYC: Complete Deep Dive' }

const part = intelTrack.parts![5]

export default function Part06Page() {
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
