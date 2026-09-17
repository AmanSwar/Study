export function Footer() {
  return (
    <footer className="ui no-print mt-auto border-t border-border-primary">
      <div className="max-w-[46rem] mx-auto px-6 sm:px-8 py-8 flex flex-wrap items-center gap-x-5 gap-y-2 font-sans text-[12px] text-text-tertiary">
        <span className="font-semibold text-text-secondary">aman.study</span>
        <span>Press <kbd>?</kbd> for shortcuts</span>
        <span><kbd>⌘K</kbd> search</span>
        <a href="https://github.com/AmanSwar/Study" target="_blank" rel="noopener noreferrer" className="hover:text-text-primary transition-colors">
          Source on GitHub
        </a>
      </div>
    </footer>
  )
}
