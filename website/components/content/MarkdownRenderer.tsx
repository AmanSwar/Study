'use client'

import 'katex/dist/katex.min.css'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import remarkMath from 'remark-math'
import rehypeKatex from 'rehype-katex'
import rehypeRaw from 'rehype-raw'
import { useState, ReactNode } from 'react'
import dynamic from 'next/dynamic'

// AsciiDiagram parses diagrams into large SVG trees. Rendering that during
// SSR (×60 diagrams × 141 pages) blows the Vercel build budget. Load it
// client-only — server renders the placeholder below, client hydrates with
// the real SVG.
const AsciiDiagram = dynamic(
  () => import('./AsciiDiagram').then((m) => m.AsciiDiagram),
  { ssr: false, loading: () => <Placeholder label="Diagram" /> }
)

// Visualizations carry interactive SVG charts. Same SSR-cost reasoning as
// AsciiDiagram — load on the client.
const Visualization = dynamic(
  () => import('./Visualization').then((m) => m.Visualization),
  { ssr: false, loading: () => <Placeholder label="Visualization" /> }
)

function Placeholder({ label }: { label: string }) {
  return (
    <div className="code-block">
      <div className="code-head"><span className="lang">{label}</span></div>
      <pre className="code"><code className="text-text-tertiary">Loading…</code></pre>
    </div>
  )
}

interface MarkdownRendererProps {
  content: string
}

function slugify(text: string): string {
  return text
    .toLowerCase()
    .replace(/[^\w\s-]/g, '')
    .trim()
    .replace(/\s+/g, '-')
    .replace(/-+/g, '-')
    .substring(0, 60)
}

function childrenToText(children: ReactNode): string {
  if (typeof children === 'string') return children
  if (typeof children === 'number') return String(children)
  if (Array.isArray(children)) return children.map(childrenToText).join('')
  if (children && typeof children === 'object' && 'props' in children) {
    // @ts-expect-error - accessing props on ReactNode
    return childrenToText(children.props.children)
  }
  return ''
}

/**
 * A block is an ASCII diagram only if it contains box-drawing characters
 * (U+2500–257F) — stray arrows in asm/C comments don't count.
 */
function isAsciiDiagram(content: string): boolean {
  return /[\u2500-\u257F]/.test(content)
}

const LANG_LABELS: Record<string, string> = {
  python: 'Python', py: 'Python', c: 'C', cpp: 'C++', 'c++': 'C++',
  asm: 'Assembly', assembly: 'Assembly', bash: 'Shell', shell: 'Shell', sh: 'Shell',
  rust: 'Rust', typescript: 'TypeScript', ts: 'TypeScript', js: 'JavaScript',
  javascript: 'JavaScript', go: 'Go', cuda: 'CUDA', metal: 'Metal',
  json: 'JSON', yaml: 'YAML', sql: 'SQL', text: 'Text', '': 'Text',
  plain: 'Text', ascii: 'Diagram', diagram: 'Diagram',
}

/**
 * Fenced code in the same markup study.css styles for HTML modules
 * (.code-block > .code-head + pre.code). Diagrams and visualisations hand off
 * to their components inside a .md-wide plate.
 */
function MarkdownCodeBlock({ language, children }: { language: string; children: string }) {
  const [copied, setCopied] = useState(false)
  const code = children.replace(/\n$/, '')

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(code)
      setCopied(true)
      setTimeout(() => setCopied(false), 1600)
    } catch {}
  }

  // `viz` blocks are synthesised by the markdown preprocessor from `[VISUALIZATION: …]` lines.
  if (language === 'viz') {
    return <div className="md-wide"><Visualization id={code.trim()} /></div>
  }
  if (isAsciiDiagram(code)) {
    return <div className="md-wide"><AsciiDiagram code={code} language={language} /></div>
  }

  return (
    <div className="code-block">
      <div className="code-head">
        <span className="lang">{LANG_LABELS[language] || language || 'Code'}</span>
        <button type="button" onClick={handleCopy} className={`copy${copied ? ' is-done' : ''}`} aria-label="Copy code">
          {copied ? 'Copied' : 'Copy'}
        </button>
      </div>
      <pre className="code"><code>{code}</code></pre>
    </div>
  )
}

/**
 * MarkdownRenderer — renders source markdown onto the study.css components, so
 * markdown tracks read exactly like HTML modules. Elements are emitted without
 * a wrapper: they must be direct children of the .study column for the wide
 * plate rules to apply.
 */
export function MarkdownRenderer({ content }: MarkdownRendererProps) {
  return (
    <ReactMarkdown
      remarkPlugins={[remarkGfm, remarkMath]}
      rehypePlugins={[
        [
          rehypeKatex,
          {
            // Don't throw on parse errors — render the failing TeX as-is so the rest of the page still renders.
            throwOnError: false,
            errorColor: 'var(--accent-red)',
            strict: false as const,
            trust: true,
          },
        ],
        rehypeRaw,
      ]}
      components={{
        h1: ({ children, ...props }) => <h1 id={slugify(childrenToText(children))} {...props}>{children}</h1>,
        h2: ({ children, ...props }) => {
          const id = slugify(childrenToText(children))
          return (
            <h2 id={id} {...props}>
              {children}
              <a href={`#${id}`} className="anchor" aria-label="Permalink">#</a>
            </h2>
          )
        },
        h3: ({ children, ...props }) => {
          const id = slugify(childrenToText(children))
          return (
            <h3 id={id} {...props}>
              {children}
              <a href={`#${id}`} className="anchor" aria-label="Permalink">#</a>
            </h3>
          )
        },
        a: ({ children, href, ...props }) => (
          <a
            href={href}
            target={href?.startsWith('http') ? '_blank' : undefined}
            rel={href?.startsWith('http') ? 'noopener noreferrer' : undefined}
            {...props}
          >
            {children}
          </a>
        ),
        blockquote: ({ children }) => <aside className="callout callout-note">{children}</aside>,
        code: ({ className, children, ...props }) => {
          const childrenStr = String(children)
          const match = /language-(\w+)/.exec(className || '')
          // A code BLOCK has a language- class (```lang) or contains a newline (bare fence).
          const isBlock = match !== null || childrenStr.includes('\n')
          if (isBlock) {
            return <MarkdownCodeBlock language={match ? match[1] : ''}>{childrenStr.replace(/\n$/, '')}</MarkdownCodeBlock>
          }
          return <code {...props}>{children}</code>
        },
        // the <code> handler above already produced the block; avoid a double <pre>
        pre: ({ children }) => <>{children}</>,
        table: ({ children, ...props }) => (
          <div className="tbl-wrap">
            <table className="tbl" {...props}>{children}</table>
          </div>
        ),
        img: ({ src, alt, ...props }) => <img src={src} alt={alt} loading="lazy" {...props} />,
      }}
    >
      {content}
    </ReactMarkdown>
  )
}
