import { MarkdownModulePage } from '@/components/content/MarkdownModulePage'

export const metadata = { title: 'Module 25: Modern GPU Inference Stack' }

export default function Module25Page() {
  return (
    <MarkdownModulePage
      sourcePath="MLsys/part_07_gpu_inference/module_25_modern_gpu_stack.md"
      breadcrumbs={[
        { label: 'MLsys', href: '/mlsys' },
        { label: 'Part 7: GPU Inference', href: '/mlsys/part-07-gpu-inference' },
        { label: 'Module 25' },
      ]}
      moduleNumber={25}
      title="Modern GPU Inference Stack"
      track="mlsys"
      part={7}
      readingTime="35 min"
      description="Hopper architecture, TensorRT-LLM, FlashAttention-3, FP8 inference"
      prev={{ href: '/mlsys/part-06-edge-ai/module-24', label: 'Edge LLMs' }}
      next={{ href: '/mlsys/part-08-fine-tuning/module-26', label: 'Fine-Tuning Infrastructure' }}
    />
  )
}
