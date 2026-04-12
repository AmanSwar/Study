import { Zap } from 'lucide-react'

interface KeyTakeawaysProps {
  items: string[]
}

export function KeyTakeaways({ items }: KeyTakeawaysProps) {
  return (
    <div className="my-10 rounded-xl border border-accent-blue/30 bg-accent-blue-subtle/30 p-6">
      <div className="flex items-center gap-2 mb-4">
        <Zap className="w-5 h-5 text-accent-blue" />
        <h2 className="text-lg font-bold text-text-primary">Key Takeaways</h2>
      </div>
      <ul className="space-y-2">
        {items.map((item, i) => (
          <li key={i} className="flex items-start gap-3 text-sm text-text-primary leading-relaxed">
            <span className="w-1.5 h-1.5 rounded-full bg-accent-blue shrink-0 mt-2" />
            {item}
          </li>
        ))}
      </ul>
    </div>
  )
}
