import { MarkdownModulePage } from '@/components/content/MarkdownModulePage'

export const metadata = { title: 'Module 27: LoRA & Parameter-Efficient Fine-Tuning' }

export default function Module27Page() {
  return (
    <MarkdownModulePage
      sourcePath="MLsys/part_08_fine_tuning/module_27_lora_peft.md"
      breadcrumbs={[
        { label: 'MLsys', href: '/mlsys' },
        { label: 'Part 8: Fine-Tuning', href: '/mlsys/part-08-fine-tuning' },
        { label: 'Module 27' },
      ]}
      moduleNumber={27}
      title="LoRA & Parameter-Efficient Fine-Tuning"
      track="mlsys"
      part={8}
      readingTime="30 min"
      description="Low-rank adaptation, QLoRA, adapter methods, multi-LoRA serving"
      prev={{ href: '/mlsys/part-08-fine-tuning/module-26', label: 'FT Infrastructure' }}
      next={{ href: '/mlsys/part-08-fine-tuning/module-28', label: 'RLHF & Alignment' }}
    />
  )
}
