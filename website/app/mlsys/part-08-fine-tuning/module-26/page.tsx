import { MarkdownModulePage } from '@/components/content/MarkdownModulePage'

export const metadata = { title: 'Module 26: Fine-Tuning Infrastructure' }

export default function Module26Page() {
  return (
    <MarkdownModulePage
      sourcePath="MLsys/part_08_fine_tuning/module_26_finetuning_infrastructure.md"
      breadcrumbs={[
        { label: 'MLsys', href: '/mlsys' },
        { label: 'Part 8: Fine-Tuning', href: '/mlsys/part-08-fine-tuning' },
        { label: 'Module 26' },
      ]}
      moduleNumber={26}
      title="Fine-Tuning Infrastructure"
      track="mlsys"
      part={8}
      readingTime="35 min"
      description="Distributed fine-tuning, DeepSpeed, FSDP, checkpointing"
      prev={{ href: '/mlsys/part-07-gpu-inference/module-25', label: 'GPU Stack' }}
      next={{ href: '/mlsys/part-08-fine-tuning/module-27', label: 'LoRA & PEFT' }}
    />
  )
}
