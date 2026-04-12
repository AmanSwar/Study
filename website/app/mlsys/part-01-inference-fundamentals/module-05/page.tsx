import { MarkdownModulePage } from '@/components/content/MarkdownModulePage'

export const metadata = { title: 'Module 5: Graph Optimization & Compilation' }

export default function Module05Page() {
  return (
    <MarkdownModulePage
      sourcePath="MLsys/part_01_inference_fundamentals/module_05_graph_optimization_compilation.md"
      breadcrumbs={[
        { label: 'MLsys', href: '/mlsys' },
        { label: 'Part 1: Inference Fundamentals', href: '/mlsys/part-01-inference-fundamentals' },
        { label: 'Module 5' },
      ]}
      moduleNumber={5}
      title="Graph Optimization & Compilation"
      track="mlsys"
      part={1}
      readingTime="40 min"
      description="Operator fusion, TVM, MLIR, IREE, torch.compile, TensorRT"
      prev={{ href: '/mlsys/part-01-inference-fundamentals/module-04', label: 'Knowledge Distillation' }}
      next={{ href: '/mlsys/part-02-transformer-architecture/module-06', label: 'Attention Optimization' }}
    />
  )
}
