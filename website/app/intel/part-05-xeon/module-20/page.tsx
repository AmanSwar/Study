import { MarkdownModulePage } from '@/components/content/MarkdownModulePage'

export const metadata = { title: 'Module 20: Intel-Specific Performance Optimization Techniques' }

export default function Module20Page() {
  return (
    <MarkdownModulePage
      sourcePath="intel/module-20.md"
      breadcrumbs={[
        { label: 'Intel/AMD', href: '/intel' },
        { label: 'Part 5: Intel Xeon', href: '/intel/part-05-xeon' },
        { label: 'Module 20' },
      ]}
      moduleNumber={20}
      title="Intel-Specific Performance Optimization Techniques"
      track="intel"
      part={5}
      readingTime="30 min"
      description="AMX programming, VNNI, DL Boost, kernel tuning"
      prev={{ href: '/intel/part-05-xeon/module-19', label: 'Xeon Memory' }}
      next={{ href: '/intel/part-06-epyc/module-21', label: 'Zen Evolution' }}
    />
  )
}
