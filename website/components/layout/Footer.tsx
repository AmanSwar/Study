import Link from 'next/link'
import { Code2 } from 'lucide-react'

export function Footer() {
  return (
    <footer className="border-t border-border-primary bg-bg-primary mt-auto">
      <div className="max-w-7xl mx-auto px-4 lg:px-6 py-10">
        <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-6">
          <div className="space-y-2">
            <div className="text-sm font-semibold text-text-primary">aman.study</div>
            <div className="text-xs text-text-tertiary">
              A personal knowledge base. Press <kbd className="text-[10px]">?</kbd> for shortcuts.
            </div>
          </div>

          <div className="flex items-center gap-6 text-xs text-text-tertiary">
            <div className="flex items-center gap-4">
              <Link href="/mlsys" className="hover:text-text-primary transition-colors">MLsys</Link>
              <Link href="/intel" className="hover:text-text-primary transition-colors">Intel/AMD</Link>
              <Link href="/qualcomm" className="hover:text-text-primary transition-colors">Qualcomm</Link>
              <Link href="/quant" className="hover:text-text-primary transition-colors">Quant</Link>
            </div>
            <a
              href="https://github.com/AmanSwar/Study"
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-center gap-1.5 hover:text-text-primary transition-colors"
              aria-label="GitHub repository"
            >
              <Code2 className="w-3.5 h-3.5" />
              <span>Source</span>
            </a>
          </div>
        </div>
      </div>
    </footer>
  )
}
