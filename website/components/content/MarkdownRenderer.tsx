'use client'

import 'katex/dist/katex.min.css'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import remarkMath from 'remark-math'
import rehypeKatex from 'rehype-katex'
import rehypeRaw from 'rehype-raw'
import { useState, ReactNode } from 'react'
import { Copy, Check, Link2 } from 'lucide-react'
import { AsciiDiagram } from './AsciiDiagram'

interface MarkdownRendererProps {
  content: string
}

/**
 * Generate a URL-friendly slug from text content (for heading anchors)
 */
function slugify(text: string): string {
  return text
    .toLowerCase()
    .replace(/[^\w\s-]/g, '')
    .trim()
    .replace(/\s+/g, '-')
    .replace(/-+/g, '-')
    .substring(0, 60)
}

/**
 * Extract plain text from React children (for heading IDs).
 */
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
 * Detect if a block is an ASCII diagram worth trying to render as SVG.
 * We require actual box-drawing characters (U+2500–257F) — stray arrows or
 * geometric glyphs in comments (common in asm/C code) don't count.
 */
function isAsciiDiagram(content: string): boolean {
  // Box-drawing block only (corners, edges, junctions)
  const boxDrawingChars = /[\u2500-\u257F]/
  return boxDrawingChars.test(content)
}

/**
 * CodeBlock component matching the existing site style.
 * - ASCII/text blocks render as clean scrollable <pre> with no line numbers
 *   (line numbers destroy the alignment of box-drawing characters).
 * - Code blocks (with a language) get line numbers via a grid layout
 *   that doesn't constrain content width, so long lines scroll horizontally
 *   instead of wrapping.
 */
function MarkdownCodeBlock({ language, children }: { language: string; children: string }) {
  const [copied, setCopied] = useState(false)
  const code = children.replace(/\n$/, '')
  const lines = code.split('\n')

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(children)
      setCopied(true)
      setTimeout(() => setCopied(false), 2000)
    } catch {}
  }

  const langLabels: Record<string, string> = {
    python: 'Python', py: 'Python', c: 'C', cpp: 'C++', 'c++': 'C++',
    asm: 'Assembly', assembly: 'Assembly', bash: 'Bash', shell: 'Shell',
    rust: 'Rust', typescript: 'TypeScript', ts: 'TypeScript', js: 'JavaScript',
    javascript: 'JavaScript', go: 'Go', cuda: 'CUDA', metal: 'Metal',
    json: 'JSON', yaml: 'YAML', sql: 'SQL', text: 'Diagram', '': 'Diagram',
    plain: 'Diagram', ascii: 'Diagram', diagram: 'Diagram',
  }
  const langColors: Record<string, string> = {
    python: 'bg-yellow-500/20 text-yellow-400', py: 'bg-yellow-500/20 text-yellow-400',
    c: 'bg-blue-500/20 text-blue-400', cpp: 'bg-blue-500/20 text-blue-400', 'c++': 'bg-blue-500/20 text-blue-400',
    asm: 'bg-purple-500/20 text-purple-400', assembly: 'bg-purple-500/20 text-purple-400',
    bash: 'bg-green-500/20 text-green-400', shell: 'bg-green-500/20 text-green-400',
    rust: 'bg-orange-500/20 text-orange-400',
    typescript: 'bg-blue-500/20 text-blue-400', ts: 'bg-blue-500/20 text-blue-400',
    cuda: 'bg-green-500/20 text-green-400',
  }

  // A block is a "diagram" only if it actually contains box-drawing characters.
  // Previously we classified any unlabeled block as a diagram, but many of those
  // are structured text / tables / data — those should render as regular code.
  const isDiagram = isAsciiDiagram(code)

  // For diagram blocks, hand off to the AsciiDiagram component which parses
  // and renders as SVG (with fallback to styled ASCII if parsing fails).
  if (isDiagram) {
    return <AsciiDiagram code={code} language={language} />
  }

  const label = langLabels[language] || language || 'Code'
  const colorClass = langColors[language] || 'bg-gray-500/20 text-gray-400'

  return (
    <div className="my-6 rounded-xl border border-border-primary bg-bg-code overflow-hidden not-prose">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-2 border-b border-border-primary bg-bg-surface/50">
        <span className={`text-[10px] font-semibold px-2 py-0.5 rounded-md uppercase tracking-wider ${colorClass}`}>
          {label}
        </span>
        <button
          onClick={handleCopy}
          className="flex items-center gap-1 text-xs text-text-tertiary hover:text-text-secondary transition-colors px-2 py-1 rounded-md hover:bg-bg-surface-hover"
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

      {/* Real code: line numbers on the left, scrollable code on the right.
          Uses inline-flex so content expands to its natural width (no wrapping),
          and the outer div provides overflow scroll. */}
      <div className="overflow-x-auto">
        <div className="inline-flex min-w-full">
          <div
            className="select-none flex-none py-4 pl-4 pr-3 text-right text-text-tertiary/50 text-xs font-mono bg-bg-code"
            style={{ lineHeight: 1.625 }}
          >
            {lines.map((_, i) => (
              <div key={i}>{i + 1}</div>
            ))}
          </div>
          <pre
            className="flex-none py-4 pr-4 pl-2 text-sm font-mono text-text-primary"
            style={{ whiteSpace: 'pre', lineHeight: 1.625 }}
          >
            <code>{code}</code>
          </pre>
        </div>
      </div>
    </div>
  )
}

/**
 * MarkdownRenderer — renders source markdown with all the styling of the
 * existing curated module components. Maps markdown elements to:
 *   # / ## / ### → styled headings with anchor links
 *   tables → DataTable-styled wrappers
 *   code fences → CodeBlock with copy button + line numbers
 *   blockquotes → CalloutBox style
 *   lists, paragraphs, links, etc → prose styling
 *   $$ math $$ / $math$ → KaTeX
 */
export function MarkdownRenderer({ content }: MarkdownRendererProps) {
  return (
    <div className="markdown-content">
      <ReactMarkdown
        remarkPlugins={[remarkGfm, remarkMath]}
        rehypePlugins={[
          [
            rehypeKatex,
            {
              // Don't throw on parse errors — render the failing TeX as-is
              // (colored/highlighted) so the rest of the page still renders.
              throwOnError: false,
              // Display errors inline (as colored text) rather than popup tooltips
              errorColor: 'var(--accent-red)',
              // Be permissive with non-standard macros
              strict: false as const,
              // Enable common extensions like \begin{cases}, \begin{align*}, etc.
              trust: true,
            },
          ],
          rehypeRaw,
        ]}
        components={{
          h1: ({ children, ...props }) => {
            const text = childrenToText(children)
            const id = slugify(text)
            return (
              <h1
                id={id}
                className="text-3xl sm:text-4xl font-extrabold tracking-tight text-text-primary mb-6 mt-8 scroll-mt-20"
                {...props}
              >
                {children}
              </h1>
            )
          },
          h2: ({ children, ...props }) => {
            const text = childrenToText(children)
            const id = slugify(text)
            return (
              <h2
                id={id}
                className="group flex items-center gap-2 text-[1.75rem] font-bold tracking-tight text-text-primary border-b border-border-primary pb-3 mb-6 mt-12 scroll-mt-20"
                {...props}
              >
                <a href={`#${id}`} className="hover:text-accent-blue transition-colors">{children}</a>
                <a href={`#${id}`} className="opacity-0 group-hover:opacity-100 transition-opacity" aria-label="Permalink">
                  <Link2 className="w-4 h-4 text-text-tertiary hover:text-accent-blue" />
                </a>
              </h2>
            )
          },
          h3: ({ children, ...props }) => {
            const text = childrenToText(children)
            const id = slugify(text)
            return (
              <h3
                id={id}
                className="group flex items-center gap-2 text-[1.375rem] font-semibold text-text-primary mb-4 mt-8 scroll-mt-20"
                {...props}
              >
                <a href={`#${id}`} className="hover:text-accent-blue transition-colors">{children}</a>
                <a href={`#${id}`} className="opacity-0 group-hover:opacity-100 transition-opacity" aria-label="Permalink">
                  <Link2 className="w-3.5 h-3.5 text-text-tertiary hover:text-accent-blue" />
                </a>
              </h3>
            )
          },
          h4: ({ children, ...props }) => (
            <h4 className="text-lg font-semibold text-text-primary mt-6 mb-3" {...props}>
              {children}
            </h4>
          ),
          h5: ({ children, ...props }) => (
            <h5 className="text-base font-semibold text-text-primary mt-4 mb-2" {...props}>
              {children}
            </h5>
          ),
          h6: ({ children, ...props }) => (
            <h6 className="text-sm font-semibold text-text-secondary mt-4 mb-2 uppercase tracking-wider" {...props}>
              {children}
            </h6>
          ),
          p: ({ children, ...props }) => (
            <p className="mb-5 leading-relaxed text-text-primary" {...props}>
              {children}
            </p>
          ),
          a: ({ children, href, ...props }) => (
            <a
              href={href}
              target={href?.startsWith('http') ? '_blank' : undefined}
              rel={href?.startsWith('http') ? 'noopener noreferrer' : undefined}
              className="text-accent-blue hover:text-accent-blue-hover underline underline-offset-2 transition-colors"
              {...props}
            >
              {children}
            </a>
          ),
          strong: ({ children, ...props }) => (
            <strong className="font-bold text-text-primary" {...props}>
              {children}
            </strong>
          ),
          em: ({ children, ...props }) => (
            <em className="italic" {...props}>
              {children}
            </em>
          ),
          ul: ({ children, ...props }) => (
            <ul className="list-disc pl-6 my-5 space-y-1.5 text-text-primary" {...props}>
              {children}
            </ul>
          ),
          ol: ({ children, ...props }) => (
            <ol className="list-decimal pl-6 my-5 space-y-1.5 text-text-primary" {...props}>
              {children}
            </ol>
          ),
          li: ({ children, ...props }) => (
            <li className="leading-relaxed" {...props}>
              {children}
            </li>
          ),
          blockquote: ({ children }) => (
            <div className="my-6 rounded-xl border-l-4 border-accent-blue bg-accent-blue-subtle/30 p-5 not-prose">
              <div className="text-sm leading-relaxed text-text-primary">
                {children}
              </div>
            </div>
          ),
          hr: () => (
            <hr className="my-10 border-t border-border-primary" />
          ),
          code: ({ className, children, ...props }) => {
            const childrenStr = String(children)
            const match = /language-(\w+)/.exec(className || '')
            // Treat as a code BLOCK if:
            //   - It has a language- class (typical ```lang fenced block), OR
            //   - Its content contains a newline (fenced block with no lang).
            // Otherwise render as inline code.
            const isBlock = match !== null || childrenStr.includes('\n')
            if (isBlock) {
              const language = match ? match[1] : ''
              return <MarkdownCodeBlock language={language}>{childrenStr.replace(/\n$/, '')}</MarkdownCodeBlock>
            }
            return (
              <code className="px-1.5 py-0.5 rounded bg-bg-code text-accent-cyan text-[0.9em] font-mono break-words" {...props}>
                {children}
              </code>
            )
          },
          pre: ({ children }) => {
            // Check if this <pre> contains a <code> with a language class — if so,
            // the <code> handler above already renders it as MarkdownCodeBlock,
            // so we just pass through to avoid double-wrapping
            return <>{children}</>
          },
          table: ({ children, ...props }) => (
            <div className="my-6 rounded-xl border border-border-primary overflow-hidden not-prose">
              <div className="overflow-x-auto">
                <table className="w-full text-sm" {...props}>
                  {children}
                </table>
              </div>
            </div>
          ),
          thead: ({ children, ...props }) => (
            <thead className="bg-bg-surface border-b border-border-primary" {...props}>
              {children}
            </thead>
          ),
          tbody: ({ children, ...props }) => (
            <tbody className="divide-y divide-border-subtle" {...props}>
              {children}
            </tbody>
          ),
          tr: ({ children, ...props }) => (
            <tr className="hover:bg-bg-surface-hover transition-colors" {...props}>
              {children}
            </tr>
          ),
          th: ({ children, ...props }) => (
            <th className="px-4 py-3 text-left text-xs font-semibold uppercase tracking-wider text-text-tertiary" {...props}>
              {children}
            </th>
          ),
          td: ({ children, ...props }) => (
            <td className="px-4 py-3 text-sm text-text-secondary align-top" {...props}>
              {children}
            </td>
          ),
          img: ({ src, alt, ...props }) => (
            <img src={src} alt={alt} className="my-6 rounded-lg border border-border-primary max-w-full" {...props} />
          ),
        }}
      >
        {content}
      </ReactMarkdown>
    </div>
  )
}
