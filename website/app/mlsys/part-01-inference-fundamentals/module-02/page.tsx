import { MarkdownModulePage } from '@/components/content/MarkdownModulePage'

export const metadata = { title: 'Module 2: Quantization — Complete Systems Perspective' }

export default function Module02Page() {
  return (
    <MarkdownModulePage
      sourcePath="MLsys/part_01_inference_fundamentals/module_02_quantization.md"
      breadcrumbs={[
        { label: 'MLsys', href: '/mlsys' },
        { label: 'Part 1: Inference Fundamentals', href: '/mlsys/part-01-inference-fundamentals' },
        { label: 'Module 2' },
      ]}
      moduleNumber={2}
      title="Quantization — Complete Systems Perspective"
      track="mlsys"
      part={1}
      readingTime="40 min"
      description="GPTQ, AWQ, SmoothQuant; hardware mapping for INT8/INT4 quantization"
      prev={{ href: '/mlsys/part-01-inference-fundamentals/module-01', label: 'Inference Mental Model' }}
      next={{ href: '/mlsys/part-01-inference-fundamentals/module-03', label: 'Sparsity' }}
    />
  )
}
