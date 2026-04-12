import { PartIndexPage } from '@/components/content/PartIndexPage'
import { mlsysTrack } from '@/content/mlsys'

export const metadata = { title: 'Part VIII: Fine-Tuning Systems' }

const part = mlsysTrack.parts![7]

export default function Part08Page() {
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
