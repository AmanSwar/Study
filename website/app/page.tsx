import Link from 'next/link'
import { Cpu, CircuitBoard, Smartphone, TrendingUp, ArrowRight, BookOpen, Search, Command } from 'lucide-react'
import { TopNav } from '@/components/layout/TopNav'
import { Footer } from '@/components/layout/Footer'

const tracks = [
  {
    id: 'mlsys',
    title: 'ML Systems Engineering',
    description: 'Production inference systems — quantization, transformers, distributed inference, CPU/GPU/Apple/Edge AI, voice AI, and RAG infrastructure.',
    moduleCount: 39,
    appendixCount: 5,
    parts: 10,
    icon: Cpu,
    accent: 'blue',
    gradient: 'from-blue-600 to-cyan-500',
  },
  {
    id: 'intel',
    title: 'Intel & AMD CPU Architecture',
    description: 'CPU design deep dive — x86 architecture, Xeon Scalable, AMD EPYC, performance engineering, and production inference engines.',
    moduleCount: 27,
    appendixCount: 5,
    parts: 7,
    icon: CircuitBoard,
    accent: 'cyan',
    gradient: 'from-cyan-500 to-teal-400',
  },
  {
    id: 'qualcomm',
    title: 'Qualcomm Hexagon NPU',
    description: 'On-device inference mastery — Hexagon VLIW architecture, HVX vector programming, HTA/HMX tensor accelerators, and mobile optimization.',
    moduleCount: 10,
    appendixCount: 0,
    parts: 0,
    icon: Smartphone,
    accent: 'orange',
    gradient: 'from-orange-500 to-amber-400',
  },
  {
    id: 'quant',
    title: 'From Zero to Quant',
    description: 'Quantitative trading for ML engineers — financial markets, alpha research, backtesting, portfolio construction, risk management, and live execution on Indian markets.',
    moduleCount: 28,
    appendixCount: 0,
    parts: 0,
    icon: TrendingUp,
    accent: 'green',
    gradient: 'from-green-500 to-emerald-400',
  },
]

export default function HomePage() {
  return (
    <>
      <TopNav />
      <main className="flex-1">
        {/* Hero Section */}
        <section className="relative overflow-hidden">
          {/* Subtle grid background */}
          <div
            className="absolute inset-0 pointer-events-none opacity-[0.03] dark:opacity-[0.04]"
            style={{
              backgroundImage: `
                linear-gradient(to right, currentColor 1px, transparent 1px),
                linear-gradient(to bottom, currentColor 1px, transparent 1px)
              `,
              backgroundSize: '48px 48px',
              maskImage: 'radial-gradient(ellipse at top, black 40%, transparent 70%)',
              WebkitMaskImage: 'radial-gradient(ellipse at top, black 40%, transparent 70%)',
            }}
          />

          {/* Gradient glows */}
          <div className="absolute inset-0 overflow-hidden pointer-events-none">
            <div className="absolute top-0 left-1/2 -translate-x-1/2 w-[800px] h-[400px] rounded-full bg-accent-blue/5 blur-3xl" />
            <div className="absolute top-60 left-[20%] w-[500px] h-[400px] rounded-full bg-accent-cyan/5 blur-3xl" />
            <div className="absolute top-40 right-[15%] w-[500px] h-[400px] rounded-full bg-accent-green/5 blur-3xl" />
          </div>

          <div className="relative max-w-6xl mx-auto px-4 lg:px-6 pt-20 pb-24">
            {/* Badge */}
            <div className="flex justify-center mb-8">
              <Link
                href="/mlsys"
                className="group inline-flex items-center gap-2 px-3.5 py-1.5 rounded-full
                  bg-bg-surface border border-border-primary text-text-secondary text-xs font-medium
                  hover:border-border-secondary hover:text-text-primary transition-all"
              >
                <span className="inline-flex items-center justify-center w-1.5 h-1.5 rounded-full bg-accent-green animate-pulse" />
                <span>104 modules across 4 tracks</span>
                <ArrowRight className="w-3 h-3 opacity-50 group-hover:opacity-100 group-hover:translate-x-0.5 transition-all" />
              </Link>
            </div>

            {/* Title — refined weight and spacing */}
            <h1 className="text-center text-5xl sm:text-6xl lg:text-7xl font-extrabold tracking-[-0.035em] leading-[0.95] mb-7 text-text-primary">
              A personal library of
              <br />
              <span className="inline-block mt-2 bg-gradient-to-r from-blue-500 via-cyan-500 to-green-500 bg-clip-text text-transparent">
                deep technical craft
              </span>
            </h1>

            {/* Subtitle */}
            <p className="text-center text-lg sm:text-xl text-text-secondary max-w-2xl mx-auto mb-10 leading-relaxed">
              PhD-level notes on ML systems, CPU architecture, mobile NPUs, and quantitative finance.
              Written to be read.
            </p>

            {/* Search prompt CTA */}
            <div className="flex justify-center mb-20">
              <div className="flex items-center gap-3 text-sm text-text-tertiary">
                <span>Jump to any topic</span>
                <kbd className="inline-flex items-center gap-1 px-2 py-1 text-xs">
                  <Command className="w-3 h-3" />
                  K
                </kbd>
                <span>or</span>
                <kbd className="inline-flex items-center px-2 py-1 text-xs">/</kbd>
              </div>
            </div>

            {/* Track Cards */}
            <div className="grid md:grid-cols-2 gap-5">
              {tracks.map((track) => (
                <Link
                  key={track.id}
                  href={`/${track.id}`}
                  className="group relative rounded-2xl border border-border-primary bg-bg-surface overflow-hidden
                    hover:border-border-secondary transition-all duration-300
                    hover:shadow-md hover:-translate-y-0.5"
                >
                  {/* Gradient accent bar */}
                  <div className={`absolute top-0 left-0 right-0 h-[2px] bg-gradient-to-r ${track.gradient} opacity-60 group-hover:opacity-100 transition-opacity`} />

                  <div className="p-7">
                    {/* Icon + counts */}
                    <div className="flex items-start justify-between mb-5">
                      <div className={`w-11 h-11 rounded-xl bg-gradient-to-br ${track.gradient}
                        flex items-center justify-center shadow-sm`}>
                        <track.icon className="w-5 h-5 text-white" strokeWidth={2} />
                      </div>
                      <div className="flex items-baseline gap-1.5 text-text-tertiary">
                        <span className="text-2xl font-semibold tabular-nums text-text-primary">{track.moduleCount}</span>
                        <span className="text-xs uppercase tracking-wider">modules</span>
                      </div>
                    </div>

                    {/* Title */}
                    <h3 className="text-lg font-semibold text-text-primary mb-2 tracking-tight">
                      {track.title}
                    </h3>

                    {/* Description */}
                    <p className="text-sm text-text-secondary leading-relaxed mb-5">
                      {track.description}
                    </p>

                    {/* Meta row */}
                    <div className="flex items-center justify-between text-xs">
                      <div className="flex items-center gap-3 text-text-tertiary">
                        {track.parts > 0 && <span>{track.parts} parts</span>}
                        {track.appendixCount > 0 && <span>{track.appendixCount} appendices</span>}
                      </div>
                      <span className="inline-flex items-center gap-1 text-text-secondary group-hover:text-accent-blue group-hover:gap-2 transition-all font-medium">
                        Open track
                        <ArrowRight className="w-3.5 h-3.5" />
                      </span>
                    </div>
                  </div>
                </Link>
              ))}
            </div>

            {/* Bottom helper row */}
            <div className="mt-16 flex flex-col sm:flex-row items-center justify-center gap-4 text-xs text-text-tertiary">
              <span className="inline-flex items-center gap-2">
                <Search className="w-3.5 h-3.5" />
                Use search to jump anywhere
              </span>
              <span className="hidden sm:inline opacity-40">·</span>
              <span className="inline-flex items-center gap-2">
                <BookOpen className="w-3.5 h-3.5" />
                Navigate with <kbd className="text-[10px]">←</kbd> <kbd className="text-[10px]">→</kbd>
              </span>
            </div>
          </div>
        </section>
      </main>
      <Footer />
    </>
  )
}
