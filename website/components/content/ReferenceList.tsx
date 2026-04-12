import { BookMarked } from 'lucide-react'

interface Reference {
  authors: string
  year: number
  title: string
  venue?: string
  url?: string
}

interface ReferenceListProps {
  references: Reference[]
}

export function ReferenceList({ references }: ReferenceListProps) {
  return (
    <div className="my-10 pt-8 border-t border-border-primary">
      <div className="flex items-center gap-2 mb-5">
        <BookMarked className="w-5 h-5 text-text-tertiary" />
        <h2 className="text-lg font-bold text-text-primary">References</h2>
      </div>
      <ol className="space-y-3 list-none">
        {references.map((ref, i) => (
          <li key={i} className="flex items-start gap-3 text-sm">
            <span className="text-xs text-text-tertiary font-mono mt-0.5 shrink-0 w-6 text-right">
              [{i + 1}]
            </span>
            <div className="text-text-secondary leading-relaxed">
              <span className="text-text-primary font-medium">{ref.authors}</span>
              {' '}({ref.year}).{' '}
              {ref.url ? (
                <a
                  href={ref.url}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-accent-blue hover:underline"
                >
                  {ref.title}
                </a>
              ) : (
                <span className="italic">{ref.title}</span>
              )}
              {ref.venue && <span>. {ref.venue}</span>}
              .
            </div>
          </li>
        ))}
      </ol>
    </div>
  )
}
