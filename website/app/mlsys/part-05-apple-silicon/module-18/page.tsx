import { MarkdownModulePage } from '@/components/content/MarkdownModulePage'

export const metadata = { title: 'Module 18: Metal & GPU Inference on Apple Silicon' }

export default function Module18Page() {
  return (
    <MarkdownModulePage
      sourcePath="MLsys/part_05_apple_silicon/module_18_metal_gpu_inference.md"
      breadcrumbs={[
        { label: 'MLsys', href: '/mlsys' },
        { label: 'Part 5: Apple Silicon', href: '/mlsys/part-05-apple-silicon' },
        { label: 'Module 18' },
      ]}
      moduleNumber={18}
      title="Metal & GPU Inference on Apple Silicon"
      track="mlsys"
      part={5}
      readingTime="30 min"
      description="Metal compute shaders, GPU inference kernels, performance tuning"
      prev={{ href: '/mlsys/part-05-apple-silicon/module-17', label: 'CoreML & ANE' }}
      next={{ href: '/mlsys/part-05-apple-silicon/module-19', label: 'llama.cpp on Apple' }}
    />
  )
}
