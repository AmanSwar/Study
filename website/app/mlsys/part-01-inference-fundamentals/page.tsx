import { PartIndexPage } from '@/components/content/PartIndexPage'
import { mlsysTrack } from '@/content/mlsys'

export const metadata = { title: 'Part I: Inference Optimization Fundamentals' }

const part = mlsysTrack.parts![0]

export default function Part01Page() {
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
