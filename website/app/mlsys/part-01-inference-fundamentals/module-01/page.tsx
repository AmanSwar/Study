import { MarkdownModulePage } from '@/components/content/MarkdownModulePage'

export const metadata = { title: 'Module 1: The Inference Systems Mental Model' }

export default function Module01Page() {
  return (
    <MarkdownModulePage
      sourcePath="MLsys/part_01_inference_fundamentals/module_01_inference_systems_mental_model.md"
      breadcrumbs={[
        { label: 'MLsys', href: '/mlsys' },
        { label: 'Part 1: Inference Fundamentals', href: '/mlsys/part-01-inference-fundamentals' },
        { label: 'Module 1' },
      ]}
      moduleNumber={1}
      title="The Inference Systems Mental Model"
      track="mlsys"
      part={1}
      readingTime="25 min"
      description="Production LLM inference architecture, roofline model, prefill vs decode phases, and memory wall analysis"


    />
  )
}
