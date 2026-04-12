import { PartIndexPage } from '@/components/content/PartIndexPage'
import { mlsysTrack } from '@/content/mlsys'

export const metadata = { title: 'Part IX: Fast Voice AI Systems' }

const part = mlsysTrack.parts![8]

export default function Part09Page() {
  return (
    <PartIndexPage
      part={part}
      trackId="mlsys"
      trackTitle="MLsys"
      breadcrumbs={[
        { label: 'MLsys', href: '/mlsys' },
        { label: `Part ${part.number}: ${part.shortTitle}` },
      ]}
    />
  )
}
