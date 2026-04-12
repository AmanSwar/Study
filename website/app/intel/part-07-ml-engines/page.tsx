import { PartIndexPage } from '@/components/content/PartIndexPage'
import { intelTrack } from '@/content/intel'

export const metadata = { title: 'Part 7: Systems Design for ML Inference Engines' }

const part = intelTrack.parts![6]

export default function Part07Page() {
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
