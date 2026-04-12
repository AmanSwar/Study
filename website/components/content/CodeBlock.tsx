'use client'

import { useState } from 'react'
import { Copy, Check, ChevronDown, ChevronUp } from 'lucide-react'

interface CodeBlockProps {
  language: string
  title?: string
  children: string
  highlightLines?: number[]
  collapsible?: boolean
  defaultCollapsed?: boolean
}

const languageLabels: Record<string, string> = {
  python: 'Python',
  py: 'Python',
  c: 'C',
  cpp: 'C++',
  'c++': 'C++',
  asm: 'Assembly',
  assembly: 'Assembly',
  bash: 'Bash',
  shell: 'Shell',
  rust: 'Rust',
  typescript: 'TypeScript',
  ts: 'TypeScript',
  javascript: 'JavaScript',
  js: 'JavaScript',
  go: 'Go',
  cuda: 'CUDA',
  metal: 'Metal',
  json: 'JSON',
  yaml: 'YAML',
  toml: 'TOML',
  sql: 'SQL',
  text: 'Text',
}

const languageColors: Record<string, string> = {
  python: 'bg-yellow-500/20 text-yellow-400',
  py: 'bg-yellow-500/20 text-yellow-400',
  c: 'bg-blue-500/20 text-blue-400',
  cpp: 'bg-blue-500/20 text-blue-400',
  'c++': 'bg-blue-500/20 text-blue-400',
  asm: 'bg-purple-500/20 text-purple-400',
  assembly: 'bg-purple-500/20 text-purple-400',
  bash: 'bg-green-500/20 text-green-400',
  shell: 'bg-green-500/20 text-green-400',
  rust: 'bg-orange-500/20 text-orange-400',
  typescript: 'bg-blue-500/20 text-blue-400',
  ts: 'bg-blue-500/20 text-blue-400',
  cuda: 'bg-green-500/20 text-green-400',
  metal: 'bg-gray-500/20 text-gray-400',
}

export function CodeBlock({
  language,
  title,
  children,
  highlightLines = [],
  collapsible = false,
  defaultCollapsed = false,
}: CodeBlockProps) {
  const [copied, setCopied] = useState(false)
  const [collapsed, setCollapsed] = useState(defaultCollapsed)
  const lines = children.trimEnd().split('\n')
  const displayLines = collapsed ? lines.slice(0, 8) : lines

  const handleCopy = async () => {
    await navigator.clipboard.writeText(children)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  const langLabel = languageLabels[language] || language
  const langColor = languageColors[language] || 'bg-gray-500/20 text-gray-400'

  return (
    <div className="my-6 rounded-xl border border-border-primary bg-bg-code overflow-hidden">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-2 border-b border-border-primary bg-bg-surface/50">
        <div className="flex items-center gap-2">
          <span className={`text-[10px] font-semibold px-2 py-0.5 rounded-md uppercase tracking-wider ${langColor}`}>
            {langLabel}
          </span>
          {title && (
            <span className="text-xs text-text-tertiary font-medium">{title}</span>
          )}
        </div>
        <button
          onClick={handleCopy}
          className="flex items-center gap-1 text-xs text-text-tertiary hover:text-text-secondary
            transition-colors px-2 py-1 rounded-md hover:bg-bg-surface-hover"
          aria-label="Copy code"
        >
          {copied ? (
            <>
              <Check className="w-3.5 h-3.5 text-accent-green" />
              <span className="text-accent-green">Copied</span>
            </>
          ) : (
            <>
              <Copy className="w-3.5 h-3.5" />
              <span>Copy</span>
            </>
          )}
        </button>
      </div>

      {/* Code */}
      <div className="overflow-x-auto">
        <pre className="p-4 text-sm leading-relaxed font-mono">
          <code>
            {displayLines.map((line, i) => {
              const lineNum = i + 1
              const isHighlighted = highlightLines.includes(lineNum)
              return (
                <div
                  key={i}
                  className={`flex ${isHighlighted ? 'bg-accent-amber/10 -mx-4 px-4 border-l-2 border-accent-amber' : ''}`}
                >
                  <span className="select-none w-10 shrink-0 text-right pr-4 text-text-tertiary/50 text-xs leading-relaxed">
                    {lineNum}
                  </span>
                  <span className="flex-1 text-text-primary">{line || ' '}</span>
                </div>
              )
            })}
          </code>
        </pre>
      </div>

      {/* Collapse toggle */}
      {collapsible && lines.length > 8 && (
        <button
          onClick={() => setCollapsed(!collapsed)}
          className="w-full flex items-center justify-center gap-1 py-2 text-xs text-text-tertiary
            hover:text-accent-blue border-t border-border-primary bg-bg-surface/30 transition-colors"
        >
          {collapsed ? (
            <>
              <ChevronDown className="w-3.5 h-3.5" />
              Show all {lines.length} lines
            </>
          ) : (
            <>
              <ChevronUp className="w-3.5 h-3.5" />
              Collapse
            </>
          )}
        </button>
      )}
    </div>
  )
}
