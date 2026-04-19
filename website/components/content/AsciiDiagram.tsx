'use client'

import { useMemo } from 'react'
import { parseAsciiDiagram, Diagram, Rectangle } from '@/lib/ascii-diagram'

interface AsciiDiagramProps {
  /** Raw ASCII diagram text (contents of a ``` fenced block). */
  code: string
  /** Language tag from the fence (for fallback language badge). */
  language?: string
}

/**
 * Render an ASCII box-and-line diagram as beautiful SVG.
 *
 * Strategy:
 *   1. Parse the ASCII into rectangles, lines, arrows, and standalone text.
 *   2. If parsing succeeds (at least one rectangle found), render as SVG.
 *   3. Rectangles become styled rounded boxes with text inside.
 *   4. External lines become stroke paths.
 *   5. Arrows become styled triangle markers.
 *   6. Standalone text becomes SVG text labels.
 *
 * Falls back to plain <pre> with monospace styling if the diagram can't be parsed.
 */
export function AsciiDiagram({ code, language }: AsciiDiagramProps) {
  const diagram = useMemo(() => parseAsciiDiagram(code), [code])

  if (!diagram) {
    // Couldn't parse — fall back to styled ASCII
    return <AsciiFallback code={code} language={language} />
  }

  return <DiagramSVG diagram={diagram} />
}

// ============================================================
// SVG Renderer
// ============================================================

const CELL_W = 9.2  // width of one monospace column in px
const CELL_H = 20   // height of one row in px
const PADDING = 12

function DiagramSVG({ diagram }: { diagram: Diagram }) {
  const width = diagram.cols * CELL_W + PADDING * 2
  const height = diagram.rows * CELL_H + PADDING * 2

  return (
    <div className="my-6 rounded-xl border border-border-primary bg-bg-code overflow-hidden not-prose">
      <div className="flex items-center justify-between px-4 py-2 border-b border-border-primary bg-bg-surface/50">
        <span className="text-[10px] font-semibold px-2 py-0.5 rounded-md uppercase tracking-wider bg-cyan-500/20 text-cyan-400">
          Diagram
        </span>
        <span className="text-[10px] text-text-tertiary uppercase tracking-wider">
          {diagram.rectangles.length} {diagram.rectangles.length === 1 ? 'node' : 'nodes'}
        </span>
      </div>

      <div className="overflow-x-auto p-4">
        <svg
          viewBox={`0 0 ${width} ${height}`}
          width={width}
          height={height}
          className="block"
          style={{ maxWidth: '100%', height: 'auto' }}
        >
          <defs>
            {/* Arrow markers */}
            <marker id="arrow-end" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
              <path d="M 0 0 L 10 5 L 0 10 z" fill="var(--accent-cyan)" />
            </marker>
          </defs>

          {/* Render external lines first (behind rectangles) */}
          {diagram.lines.map((line, i) => {
            const x1 = line.c1 * CELL_W + CELL_W / 2 + PADDING
            const y1 = line.r1 * CELL_H + CELL_H / 2 + PADDING
            const x2 = line.c2 * CELL_W + CELL_W / 2 + PADDING
            const y2 = line.r2 * CELL_H + CELL_H / 2 + PADDING
            return (
              <line
                key={`L${i}`}
                x1={x1}
                y1={y1}
                x2={x2}
                y2={y2}
                stroke="var(--accent-cyan)"
                strokeWidth={1.5}
                opacity={0.7}
              />
            )
          })}

          {/* Render arrows */}
          {diagram.arrows.map((arr, i) => (
            <ArrowGlyph key={`A${i}`} arrow={arr} />
          ))}

          {/* Render rectangles with text */}
          {diagram.rectangles.map((rect, i) => (
            <RectNode key={`R${i}`} rect={rect} />
          ))}

          {/* Render standalone text */}
          {diagram.textRuns.map((run, i) => {
            const x = run.c * CELL_W + PADDING
            const y = run.r * CELL_H + CELL_H * 0.7 + PADDING
            return (
              <text
                key={`T${i}`}
                x={x}
                y={y}
                fontSize={12.5}
                fontFamily="var(--font-mono)"
                fill="var(--text-secondary)"
                style={{ whiteSpace: 'pre' }}
              >
                {run.text}
              </text>
            )
          })}
        </svg>
      </div>
    </div>
  )
}

function RectNode({ rect }: { rect: Rectangle }) {
  const x = rect.c1 * CELL_W + PADDING
  const y = rect.r1 * CELL_H + PADDING
  const w = (rect.c2 - rect.c1 + 1) * CELL_W
  const h = (rect.r2 - rect.r1 + 1) * CELL_H

  // Split into sections using divider rows. Each section gets its own text block.
  // sectionStarts[i] and sectionEnds[i] define the interior rows of section i.
  const sections: { startRow: number; endRow: number; lines: string[] }[] = []
  const interiorStart = rect.r1 + 1
  const interiorEnd = rect.r2 - 1
  let currentStart = interiorStart
  const sortedDividers = [...rect.dividerRows].sort((a, b) => a - b)
  for (const divRow of sortedDividers) {
    if (divRow > currentStart) {
      const sectionLines = rect.lines
        .slice(currentStart - interiorStart, divRow - interiorStart)
        .map((l) => l.trim())
        .filter((l) => l.length > 0)
      if (sectionLines.length > 0) {
        sections.push({ startRow: currentStart, endRow: divRow - 1, lines: sectionLines })
      }
    }
    currentStart = divRow + 1
  }
  // Last section after the final divider
  if (currentStart <= interiorEnd) {
    const sectionLines = rect.lines
      .slice(currentStart - interiorStart)
      .map((l) => l.trim())
      .filter((l) => l.length > 0)
    if (sectionLines.length > 0) {
      sections.push({ startRow: currentStart, endRow: interiorEnd, lines: sectionLines })
    }
  }

  // If no dividers, treat the whole interior as one section
  if (sections.length === 0) {
    const allLines = rect.lines.map((l) => l.trim()).filter((l) => l.length > 0)
    if (allLines.length > 0) {
      sections.push({ startRow: interiorStart, endRow: interiorEnd, lines: allLines })
    }
  }

  const lineHeight = 18

  return (
    <g>
      {/* Box background */}
      <rect
        x={x}
        y={y}
        width={w}
        height={h}
        rx={7}
        ry={7}
        fill="var(--bg-surface)"
        stroke="var(--accent-blue)"
        strokeWidth={1.5}
        opacity={0.95}
      />
      {/* Subtle inner highlight */}
      <rect
        x={x + 1}
        y={y + 1}
        width={w - 2}
        height={h - 2}
        rx={6}
        ry={6}
        fill="none"
        stroke="var(--accent-blue)"
        strokeWidth={0.5}
        opacity={0.2}
      />

      {/* Label in top edge (if present): render as pill that "breaks" the border */}
      {rect.label && (
        <g>
          <rect
            x={x + 12}
            y={y - 8}
            width={rect.label.length * 6.5 + 16}
            height={16}
            rx={4}
            fill="var(--bg-code)"
            stroke="var(--accent-blue)"
            strokeWidth={1}
          />
          <text
            x={x + 20}
            y={y + 4}
            fontSize={11}
            fontWeight={600}
            fontFamily="var(--font-sans)"
            fill="var(--accent-blue)"
            style={{ letterSpacing: '0.01em' }}
          >
            {rect.label}
          </text>
        </g>
      )}

      {/* Divider lines */}
      {rect.dividerRows.map((dr, i) => {
        const dy = dr * CELL_H + CELL_H / 2 + PADDING
        return (
          <line
            key={`div${i}`}
            x1={x + 2}
            y1={dy}
            x2={x + w - 2}
            y2={dy}
            stroke="var(--accent-blue)"
            strokeWidth={1}
            opacity={0.35}
          />
        )
      })}

      {/* Text in each section, vertically centered within the section */}
      {sections.map((section, i) => {
        const sectionY = section.startRow * CELL_H + PADDING
        const sectionH = (section.endRow - section.startRow + 1) * CELL_H
        const totalTextHeight = section.lines.length * lineHeight
        const textStartY = sectionY + (sectionH - totalTextHeight) / 2 + lineHeight * 0.75
        return section.lines.map((line, j) => (
          <text
            key={`S${i}L${j}`}
            x={x + w / 2}
            y={textStartY + j * lineHeight}
            textAnchor="middle"
            fontSize={13}
            fontWeight={500}
            fontFamily="var(--font-sans)"
            fill="var(--text-primary)"
            style={{ letterSpacing: '-0.01em' }}
          >
            {line}
          </text>
        ))
      })}
    </g>
  )
}

function ArrowGlyph({ arrow }: { arrow: { r: number; c: number; dir: string } }) {
  const cx = arrow.c * CELL_W + CELL_W / 2 + PADDING
  const cy = arrow.r * CELL_H + CELL_H / 2 + PADDING
  const size = 6

  let points: string
  switch (arrow.dir) {
    case 'up':
      points = `${cx},${cy - size} ${cx - size},${cy + size / 2} ${cx + size},${cy + size / 2}`
      break
    case 'down':
      points = `${cx},${cy + size} ${cx - size},${cy - size / 2} ${cx + size},${cy - size / 2}`
      break
    case 'left':
      points = `${cx - size},${cy} ${cx + size / 2},${cy - size} ${cx + size / 2},${cy + size}`
      break
    case 'right':
      points = `${cx + size},${cy} ${cx - size / 2},${cy - size} ${cx - size / 2},${cy + size}`
      break
    default:
      return null
  }
  return <polygon points={points} fill="var(--accent-cyan)" />
}

// ============================================================
// Fallback: styled ASCII pre (unchanged from the existing code-block rendering)
// ============================================================

function AsciiFallback({ code, language }: { code: string; language?: string }) {
  return (
    <div className="my-6 rounded-xl border border-border-primary bg-bg-code overflow-hidden not-prose">
      <div className="flex items-center justify-between px-4 py-2 border-b border-border-primary bg-bg-surface/50">
        <span className="text-[10px] font-semibold px-2 py-0.5 rounded-md uppercase tracking-wider bg-cyan-500/20 text-cyan-400">
          {language || 'Diagram'}
        </span>
      </div>
      <div className="overflow-x-auto">
        <pre
          className="p-4 text-sm font-mono text-text-primary"
          style={{ whiteSpace: 'pre', lineHeight: 1.25, letterSpacing: '-0.02em' }}
        >
          <code>{code.replace(/\n$/, '')}</code>
        </pre>
      </div>
    </div>
  )
}
