import Link from 'next/link'
import { Cpu, CircuitBoard, Smartphone, TrendingUp, ArrowRight, BookOpen, Layers, Sparkles } from 'lucide-react'
import { TopNav } from '@/components/layout/TopNav'
import { Footer } from '@/components/layout/Footer'

const tracks = [
  {
    id: 'mlsys',
    title: 'ML Systems Engineering',
    shortTitle: 'MLsys',
    description: 'Production inference systems — quantization, transformers, distributed inference, CPU/GPU/Apple Silicon/Edge AI, voice AI, and RAG.',
    moduleCount: 39,
    appendixCount: 5,
    parts: 10,
    icon: Cpu,
    color: 'blue' as const,
    gradient: 'from-blue-600 to-cyan-500',
    bgGlow: 'bg-blue-500/10',
    borderColor: 'border-blue-500/30',
    hoverBorder: 'hover:border-blue-400/60',
    badge: 'bg-blue-500/20 text-blue-400',
    status: 'Active',
  },
  {
    id: 'intel',
    title: 'Intel & AMD CPU Architecture',
    shortTitle: 'Intel/AMD',
    description: 'CPU design deep dive — x86 architecture, Xeon Scalable, AMD EPYC, performance engineering, and production inference engines.',
    moduleCount: 27,
    appendixCount: 5,
    parts: 7,
    icon: CircuitBoard,
    color: 'cyan' as const,
    gradient: 'from-cyan-500 to-teal-400',
    bgGlow: 'bg-cyan-500/10',
    borderColor: 'border-cyan-500/30',
    hoverBorder: 'hover:border-cyan-400/60',
    badge: 'bg-cyan-500/20 text-cyan-400',
    status: 'Coming Soon',
  },
  {
    id: 'qualcomm',
    title: 'Qualcomm Hexagon NPU',
    shortTitle: 'Qualcomm',
    description: 'On-device inference mastery — Hexagon VLIW architecture, HVX vector programming, HTA/HMX tensor accelerators, and mobile optimization.',
    moduleCount: 10,
    appendixCount: 0,
    parts: 0,
    icon: Smartphone,
    color: 'orange' as const,
    gradient: 'from-orange-500 to-amber-400',
    bgGlow: 'bg-orange-500/10',
    borderColor: 'border-orange-500/30',
    hoverBorder: 'hover:border-orange-400/60',
    badge: 'bg-orange-500/20 text-orange-400',
    status: 'Coming Soon',
  },
  {
    id: 'quant',
    title: 'From Zero to Quant',
    shortTitle: 'Quant',
    description: 'Quantitative trading for ML engineers — financial markets, alpha research, backtesting, portfolio construction, risk management, and live execution on Indian markets.',
    moduleCount: 28,
    appendixCount: 0,
    parts: 0,
    icon: TrendingUp,
    color: 'green' as const,
    gradient: 'from-green-500 to-emerald-400',
    bgGlow: 'bg-green-500/10',
    borderColor: 'border-green-500/30',
    hoverBorder: 'hover:border-green-400/60',
    badge: 'bg-green-500/20 text-green-400',
    status: 'Active',
  },
]

export default function HomePage() {
  return (
    <>
      <TopNav />
      <main className="flex-1">
        {/* Hero Section */}
        <section className="relative overflow-hidden">
          {/* Background gradient orbs */}
          <div className="absolute inset-0 overflow-hidden pointer-events-none">
            <div className="absolute -top-40 -right-40 w-96 h-96 rounded-full bg-accent-blue/8 blur-3xl" />
            <div className="absolute -bottom-40 -left-40 w-96 h-96 rounded-full bg-accent-cyan/8 blur-3xl" />
            <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[600px] h-[600px] rounded-full bg-accent-orange/5 blur-3xl" />
          </div>

          <div className="relative max-w-6xl mx-auto px-4 lg:px-6 pt-16 pb-20">
            {/* Badge */}
            <div className="flex justify-center mb-6">
              <div className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full
                bg-accent-blue-subtle border border-accent-blue/20 text-accent-blue text-sm font-medium">
                <Sparkles className="w-4 h-4" />
                PhD-Level Computer Science Curriculum
              </div>
            </div>

            {/* Title */}
            <h1 className="text-center text-4xl sm:text-5xl lg:text-6xl font-extrabold tracking-tight leading-tight mb-6">
              <span className="text-text-primary">Master </span>
              <span className="bg-gradient-to-r from-blue-500 via-cyan-400 to-orange-400 bg-clip-text text-transparent">
                Systems Engineering
              </span>
              <br />
              <span className="text-text-primary">from Silicon to Software</span>
            </h1>

            {/* Subtitle */}
            <p className="text-center text-lg sm:text-xl text-text-secondary max-w-3xl mx-auto mb-10 leading-relaxed">
              104 modules with interactive diagrams, animated data flows, and step-by-step
              visualizations. From inference optimization to quantitative trading.
            </p>

            {/* Stats */}
            <div className="flex flex-wrap justify-center gap-8 mb-16">
              {[
                { label: 'Modules', value: '104', icon: BookOpen },
                { label: 'Tracks', value: '4', icon: Layers },
                { label: 'Words', value: '200K+', icon: Sparkles },
              ].map((stat) => (
                <div key={stat.label} className="flex items-center gap-3">
                  <div className="w-10 h-10 rounded-lg bg-bg-surface border border-border-primary
                    flex items-center justify-center">
                    <stat.icon className="w-5 h-5 text-accent-blue" />
                  </div>
                  <div>
                    <div className="text-2xl font-bold text-text-primary">{stat.value}</div>
                    <div className="text-sm text-text-tertiary">{stat.label}</div>
                  </div>
                </div>
              ))}
            </div>

            {/* Track Cards */}
            <div className="grid md:grid-cols-2 gap-6">
              {tracks.map((track) => (
                <Link
                  key={track.id}
                  href={`/${track.id}`}
                  className={`group relative rounded-xl border ${track.borderColor} ${track.hoverBorder}
                    bg-bg-surface overflow-hidden transition-all duration-300
                    hover:shadow-lg hover:-translate-y-1`}
                >
                  {/* Glow effect */}
                  <div className={`absolute inset-0 ${track.bgGlow} opacity-0 group-hover:opacity-100 transition-opacity`} />

                  {/* Gradient top bar */}
                  <div className={`h-1.5 bg-gradient-to-r ${track.gradient}`} />

                  <div className="relative p-6">
                    {/* Icon + Status */}
                    <div className="flex items-start justify-between mb-4">
                      <div className={`w-12 h-12 rounded-xl bg-gradient-to-br ${track.gradient}
                        flex items-center justify-center shadow-lg`}>
                        <track.icon className="w-6 h-6 text-white" />
                      </div>
                      <span className={`text-xs font-medium px-2.5 py-1 rounded-full ${track.badge}`}>
                        {track.status}
                      </span>
                    </div>

                    {/* Title + Description */}
                    <h3 className="text-lg font-bold text-text-primary mb-2 group-hover:text-accent-blue transition-colors">
                      {track.title}
                    </h3>
                    <p className="text-sm text-text-secondary leading-relaxed mb-4">
                      {track.description}
                    </p>

                    {/* Stats */}
                    <div className="flex items-center gap-4 text-xs text-text-tertiary mb-4">
                      <span>{track.moduleCount} modules</span>
                      {track.appendixCount > 0 && <span>{track.appendixCount} appendices</span>}
                      {track.parts > 0 && <span>{track.parts} parts</span>}
                    </div>

                    {/* CTA */}
                    <div className="flex items-center gap-1.5 text-sm font-medium text-accent-blue
                      group-hover:gap-3 transition-all">
                      Explore track
                      <ArrowRight className="w-4 h-4" />
                    </div>
                  </div>
                </Link>
              ))}
            </div>
          </div>
        </section>
      </main>
      <Footer />
    </>
  )
}
