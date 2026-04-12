import { MarkdownModulePage } from '@/components/content/MarkdownModulePage'

export const metadata = { title: 'Module 4: Knowledge Distillation — Systems Perspective' }

export default function Module04Page() {
  return (
    <MarkdownModulePage
      sourcePath="MLsys/part_01_inference_fundamentals/module_04_knowledge_distillation.md"
      breadcrumbs={[
        { label: 'MLsys', href: '/mlsys' },
        { label: 'Part 1: Inference Fundamentals', href: '/mlsys/part-01-inference-fundamentals' },
        { label: 'Module 4' },
      ]}
      moduleNumber={4}
      title="Knowledge Distillation — Systems Perspective"
      track="mlsys"
      part={1}
      readingTime="30 min"
      description="Teacher-student frameworks, distillation losses, production pipelines"
      prev={{ href: '/mlsys/part-01-inference-fundamentals/module-03', label: 'Sparsity' }}
      next={{ href: '/mlsys/part-01-inference-fundamentals/module-05', label: 'Graph Optimization' }}
    />
  )
}
