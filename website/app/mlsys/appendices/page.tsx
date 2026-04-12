import Link from 'next/link'
import { Breadcrumbs } from '@/components/layout/Breadcrumbs'

export const metadata = { title: 'MLsys Appendices' }

const appendices = [
  { letter: 'A', title: 'MLPerf Benchmarks Reference', href: '/mlsys/appendices/appendix-a' },
  { letter: 'B', title: 'Model Memory Sizing Calculator', href: '/mlsys/appendices/appendix-b' },
  { letter: 'C', title: 'Latency Budget Templates', href: '/mlsys/appendices/appendix-c' },
  { letter: 'D', title: 'Key Paper Reading List', href: '/mlsys/appendices/appendix-d' },
  { letter: 'E', title: 'Hardware Comparison Matrix', href: '/mlsys/appendices/appendix-e' },
]

export default function AppendicesPage() {
  return (
    <>
      <Breadcrumbs items={[{ label: 'MLsys', href: '/mlsys' }, { label: 'Appendices' }]} />
      <h1 className="text-3xl font-extrabold tracking-tight text-text-primary mb-8">Appendices</h1>
      <div className="space-y-3">
        {appendices.map((a) => (
          <Link key={a.letter} href={a.href} className="group block rounded-xl border border-border-primary bg-bg-surface hover:border-accent-blue/40 hover:shadow-md transition-all p-5">
            <div className="flex items-center gap-4">
              <span className="w-10 h-10 rounded-lg bg-accent-blue-subtle flex items-center justify-center text-sm font-bold text-accent-blue">{a.letter}</span>
              <span className="text-base font-bold text-text-primary group-hover:text-accent-blue transition-colors">{a.title}</span>
            </div>
          </Link>
        ))}
      </div>
    </>
  )
}
