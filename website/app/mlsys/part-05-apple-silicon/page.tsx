import { PartIndexPage } from '@/components/content/PartIndexPage'
import { mlsysTrack } from '@/content/mlsys'

export const metadata = { title: 'Part V: Apple Silicon Inference' }

const part = mlsysTrack.parts![4]

export default function Part05Page() {
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
