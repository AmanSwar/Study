import { BookOpen } from 'lucide-react'

export function Footer() {
  return (
    <footer className="border-t border-border-primary bg-bg-surface py-8 mt-auto">
      <div className="max-w-7xl mx-auto px-4 lg:px-6">
        <div className="flex flex-col sm:flex-row items-center justify-between gap-4">
          <div className="flex items-center gap-2 text-text-tertiary text-sm">
            <BookOpen className="w-4 h-4" />
            <span>CS Study — PhD-Level Computer Science & Quantitative Finance</span>
          </div>
          <div className="text-text-tertiary text-sm">
            39 + 27 + 10 + 28 modules across 4 tracks
          </div>
        </div>
      </div>
    </footer>
  )
}
