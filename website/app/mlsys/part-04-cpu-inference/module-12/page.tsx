import { MarkdownModulePage } from '@/components/content/MarkdownModulePage'

export const metadata = { title: 'Module 12: CPU Inference Engine Internals' }

export default function Module12Page() {
  return (
    <MarkdownModulePage
      sourcePath="MLsys/part_04_cpu_inference/module_12_cpu_inference_engines.md"
      breadcrumbs={[
        { label: 'MLsys', href: '/mlsys' },
        { label: 'Part 4: CPU Inference', href: '/mlsys/part-04-cpu-inference' },
        { label: 'Module 12' },
      ]}
      moduleNumber={12}
      title="CPU Inference Engine Internals"
      track="mlsys"
      part={4}
      readingTime="40 min"
      description="OpenVINO, ONNX Runtime, llama.cpp architecture and kernel design"
      prev={{ href: '/mlsys/part-04-cpu-inference/module-11', label: 'Why CPU Inference' }}
      next={{ href: '/mlsys/part-04-cpu-inference/module-13', label: 'GEMM Optimization for CPU' }}
    />
  )
}
