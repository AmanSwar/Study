import { PartIndexPage } from '@/components/content/PartIndexPage'
import { mlsysTrack } from '@/content/mlsys'

export const metadata = { title: 'Part VI: On-Device & Edge AI Inference' }

const part = mlsysTrack.parts![5]

export default function Part06Page() {
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
