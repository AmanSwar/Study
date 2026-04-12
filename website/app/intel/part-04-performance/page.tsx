import { PartIndexPage } from '@/components/content/PartIndexPage'
import { intelTrack } from '@/content/intel'

export const metadata = { title: 'Part 4: Performance Engineering Methodology' }

const part = intelTrack.parts![3]

export default function Part04Page() {
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
