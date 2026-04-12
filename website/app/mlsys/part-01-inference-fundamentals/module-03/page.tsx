import { MarkdownModulePage } from '@/components/content/MarkdownModulePage'

export const metadata = { title: 'Module 3: Sparsity — Structured & Unstructured' }

export default function Module03Page() {
  return (
    <MarkdownModulePage
      sourcePath="MLsys/part_01_inference_fundamentals/module_03_sparsity.md"
      breadcrumbs={[
        { label: 'MLsys', href: '/mlsys' },
        { label: 'Part 1: Inference Fundamentals', href: '/mlsys/part-01-inference-fundamentals' },
        { label: 'Module 3' },
      ]}
      moduleNumber={3}
      title="Sparsity — Structured & Unstructured"
      track="mlsys"
      part={1}
      readingTime="35 min"
      description="Weight/activation sparsity, NVIDIA 2:4 structured sparsity, SparseGPT"
      prev={{ href: '/mlsys/part-01-inference-fundamentals/module-02', label: 'Quantization' }}
      next={{ href: '/mlsys/part-01-inference-fundamentals/module-04', label: 'Knowledge Distillation' }}
    />
  )
}
