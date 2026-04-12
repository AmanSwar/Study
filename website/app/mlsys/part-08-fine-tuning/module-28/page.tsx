import { MarkdownModulePage } from '@/components/content/MarkdownModulePage'

export const metadata = { title: 'Module 28: RLHF & Alignment Systems' }

export default function Module28Page() {
  return (
    <MarkdownModulePage
      sourcePath="MLsys/part_08_fine_tuning/module_28_rlhf_alignment.md"
      breadcrumbs={[
        { label: 'MLsys', href: '/mlsys' },
        { label: 'Part 8: Fine-Tuning', href: '/mlsys/part-08-fine-tuning' },
        { label: 'Module 28' },
      ]}
      moduleNumber={28}
      title="RLHF & Alignment Systems"
      track="mlsys"
      part={8}
      readingTime="35 min"
      description="PPO infrastructure, DPO, reward modeling, alignment training"
      prev={{ href: '/mlsys/part-08-fine-tuning/module-27', label: 'LoRA & PEFT' }}
      next={{ href: '/mlsys/part-09-voice-ai/module-29', label: 'Voice Pipeline Architecture' }}
    />
  )
}
