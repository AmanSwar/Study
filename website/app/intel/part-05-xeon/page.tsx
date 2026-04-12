import { PartIndexPage } from '@/components/content/PartIndexPage'
import { intelTrack } from '@/content/intel'

export const metadata = { title: 'Part 5: Intel Xeon Scalable: Complete Deep Dive' }

const part = intelTrack.parts![4]

export default function Part05Page() {
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
