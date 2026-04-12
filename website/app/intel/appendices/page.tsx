import Link from 'next/link'
import { Breadcrumbs } from '@/components/layout/Breadcrumbs'

export const metadata = { title: 'Intel/AMD Appendices' }

const appendices = [
  { letter: 'A', title: 'Instruction Latency & Throughput Reference Tables', href: '/intel/appendices/appendix-a' },
  { letter: 'B', title: 'Hardware Performance Counter Reference', href: '/intel/appendices/appendix-b' },
  { letter: 'C', title: 'CPUID Feature Detection Code', href: '/intel/appendices/appendix-c' },
  { letter: 'D', title: 'NUMA Topology Interrogation', href: '/intel/appendices/appendix-d' },
  { letter: 'E', title: 'Benchmark Baselines', href: '/intel/appendices/appendix-e' },
]

export default function AppendicesPage() {
  return (
    <>
      <Breadcrumbs items={[{ label: 'Intel/AMD', href: '/intel' }, { label: 'Appendices' }]} />
      <h1 className="text-3xl font-extrabold tracking-tight text-text-primary mb-8">Appendices</h1>
      <div className="space-y-3">
        {appendices.map((a) => (
          <Link key={a.letter} href={a.href} className="group block rounded-xl border border-border-primary bg-bg-surface hover:border-accent-cyan/40 hover:shadow-md transition-all p-5">
            <div className="flex items-center gap-4">
              <span className="w-10 h-10 rounded-lg bg-accent-cyan-subtle flex items-center justify-center text-sm font-bold text-accent-cyan">{a.letter}</span>
              <span className="text-base font-bold text-text-primary group-hover:text-accent-cyan transition-colors">{a.title}</span>
            </div>
          </Link>
        ))}
      </div>
    </>
  )
}
