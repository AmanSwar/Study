/**
 * ASCII diagram parser.
 *
 * Parses Unicode box-drawing characters into a structured Diagram. The renderer
 * (components/content/AsciiDiagram.tsx) then draws this as beautiful SVG.
 *
 * Handles:
 * - Rectangles: ┌─┐│└─┘
 * - T-junctions: ┬ ┴ ├ ┤ ┼ (used for divided boxes and line connections)
 * - Arrows: ▲ ▼ ◀ ▶ → ← ↑ ↓
 * - Interior dividers that split a rectangle into sections
 * - Text labels (inside rectangles and standalone)
 *
 * If parsing finds at least one well-formed rectangle, returns a Diagram.
 * Otherwise returns null so the caller can fall back to styled ASCII.
 */

// Character classification
const CORNER_TL = '┌'
const CORNER_TR = '┐'
const CORNER_BL = '└'
const CORNER_BR = '┘'
const HORIZ = '─'
const VERT = '│'
const T_DOWN = '┬'  // horizontal edge with branch going down
const T_UP = '┴'    // horizontal edge with branch going up
const T_RIGHT = '├' // vertical edge with branch going right
const T_LEFT = '┤'  // vertical edge with branch going left
const CROSS = '┼'

const HORIZ_CHARS = new Set([HORIZ, T_DOWN, T_UP, CROSS])
const VERT_CHARS = new Set([VERT, T_RIGHT, T_LEFT, CROSS])
const CORNER_CHARS = new Set([CORNER_TL, CORNER_TR, CORNER_BL, CORNER_BR])
const LINE_CHARS = new Set([HORIZ, VERT, T_DOWN, T_UP, T_RIGHT, T_LEFT, CROSS, ...CORNER_CHARS])
const ARROW_CHARS = new Set(['▲', '▼', '◀', '▶', '→', '←', '↑', '↓'])

// Characters that may appear ON a rectangle's edge in place of a normal
// line char — e.g. the diagram may have "┌──▼──┐" where ▼ sits on the top
// edge as an indicator. These don't disqualify the rectangle.
const HORIZ_EDGE_COMPATIBLE = new Set([...HORIZ_CHARS, '▲', '▼'])
const VERT_EDGE_COMPATIBLE = new Set([...VERT_CHARS, '◀', '▶'])

export interface Rectangle {
  r1: number // top row (inclusive)
  c1: number // left col (inclusive)
  r2: number // bottom row (inclusive)
  c2: number // right col (inclusive)
  /** Multi-line text extracted from interior. Each line is already trimmed. */
  lines: string[]
  /** Optional label text embedded in the top edge, e.g. "┌─ SOURCE CODE ──┐". */
  label?: string
  /** Rows (absolute grid indices) of interior horizontal dividers (├──┤). */
  dividerRows: number[]
  /** Junctions on each edge where external lines connect. */
  exits: {
    top: number[]    // columns where ┴ appears on top edge
    bottom: number[] // columns where ┬ appears on bottom edge
    left: number[]   // rows where ┤ appears on left edge
    right: number[]  // rows where ├ appears on right edge
  }
}

export interface ArrowMark {
  r: number
  c: number
  dir: 'up' | 'down' | 'left' | 'right'
}

export interface LineSegment {
  r1: number
  c1: number
  r2: number
  c2: number
  direction: 'horizontal' | 'vertical'
}

export interface TextRun {
  r: number
  c: number
  text: string
}

export interface Diagram {
  rows: number
  cols: number
  rectangles: Rectangle[]
  arrows: ArrowMark[]
  lines: LineSegment[]
  textRuns: TextRun[] // standalone text (not inside any rectangle)
  raw: string
}

/**
 * Parse an ASCII diagram text into a structured Diagram.
 * Returns null if no rectangles were found (caller should use fallback).
 */
export function parseAsciiDiagram(text: string): Diagram | null {
  const rawLines = text.split('\n')
  // Drop trailing blank lines to avoid wasted height
  while (rawLines.length && rawLines[rawLines.length - 1].trim() === '') rawLines.pop()
  if (rawLines.length === 0) return null

  const cols = Math.max(...rawLines.map((l) => [...l].length))
  const rows = rawLines.length

  // Build grid (fill short lines with spaces)
  const grid: string[][] = rawLines.map((line) => {
    const chars = [...line]
    while (chars.length < cols) chars.push(' ')
    return chars
  })

  // --- Find all rectangles ---
  const rectangles: Rectangle[] = []
  const rectCellOwnership = new Map<string, number>() // "r:c" -> rect index

  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      if (grid[r][c] === CORNER_TL) {
        const rect = tryTraceRect(grid, r, c, rows, cols)
        if (rect) {
          const idx = rectangles.length
          rectangles.push(rect)
          // Mark border cells as owned by this rect
          for (let rr = rect.r1; rr <= rect.r2; rr++) {
            for (let cc = rect.c1; cc <= rect.c2; cc++) {
              const key = `${rr}:${cc}`
              if (!rectCellOwnership.has(key)) {
                rectCellOwnership.set(key, idx)
              }
            }
          }
        }
      }
    }
  }

  if (rectangles.length === 0) return null

  // Mark border cells (just the perimeter) as "used" so they don't become standalone lines/text
  const borderCells = new Set<string>()
  for (const rect of rectangles) {
    // Top and bottom edges
    for (let c = rect.c1; c <= rect.c2; c++) {
      borderCells.add(`${rect.r1}:${c}`)
      borderCells.add(`${rect.r2}:${c}`)
    }
    // Left and right edges
    for (let r = rect.r1; r <= rect.r2; r++) {
      borderCells.add(`${r}:${rect.c1}`)
      borderCells.add(`${r}:${rect.c2}`)
    }
  }

  // Mark interior cells so we don't extract text twice
  const interiorCells = new Set<string>()
  for (const rect of rectangles) {
    for (let r = rect.r1 + 1; r < rect.r2; r++) {
      for (let c = rect.c1 + 1; c < rect.c2; c++) {
        interiorCells.add(`${r}:${c}`)
      }
    }
  }

  // --- Find arrows (standalone; those not on rect borders are external arrows) ---
  const arrows: ArrowMark[] = []
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const ch = grid[r][c]
      if (!ARROW_CHARS.has(ch)) continue
      if (borderCells.has(`${r}:${c}`)) continue // skip ones on rect borders
      const dir = arrowDirection(ch)
      if (dir) arrows.push({ r, c, dir })
    }
  }

  // --- Find external line segments (lines between rectangles) ---
  // A line character is "external" if it's NOT inside any rectangle's border.
  // We collect connected runs as horizontal or vertical segments.
  const visited = new Set<string>()
  const lines: LineSegment[] = []

  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const key = `${r}:${c}`
      if (visited.has(key)) continue
      const ch = grid[r][c]
      if (!LINE_CHARS.has(ch)) continue
      if (borderCells.has(key)) continue
      if (interiorCells.has(key)) continue

      // Trace horizontal run
      if (HORIZ_CHARS.has(ch)) {
        let c2 = c
        while (
          c2 + 1 < cols &&
          HORIZ_CHARS.has(grid[r][c2 + 1]) &&
          !borderCells.has(`${r}:${c2 + 1}`)
        ) {
          c2++
        }
        if (c2 > c) {
          for (let cc = c; cc <= c2; cc++) visited.add(`${r}:${cc}`)
          lines.push({ r1: r, c1: c, r2: r, c2, direction: 'horizontal' })
          continue
        }
      }

      // Trace vertical run
      if (VERT_CHARS.has(ch)) {
        let r2 = r
        while (
          r2 + 1 < rows &&
          VERT_CHARS.has(grid[r2 + 1][c]) &&
          !borderCells.has(`${r2 + 1}:${c}`)
        ) {
          r2++
        }
        if (r2 > r) {
          for (let rr = r; rr <= r2; rr++) visited.add(`${rr}:${c}`)
          lines.push({ r1: r, c1: c, r2, c2: c, direction: 'vertical' })
          continue
        }
      }

      visited.add(key) // isolated single line char (probably junction)
    }
  }

  // --- Find standalone text (outside rectangles and not box-drawing) ---
  const textRuns: TextRun[] = []
  for (let r = 0; r < rows; r++) {
    let c = 0
    while (c < cols) {
      const key = `${r}:${c}`
      const ch = grid[r][c]
      // Skip non-text chars and cells that are part of rects/lines
      if (
        ch === ' ' ||
        LINE_CHARS.has(ch) ||
        ARROW_CHARS.has(ch) ||
        interiorCells.has(key)
      ) {
        c++
        continue
      }
      // Start of a text run — collect until we hit space / special char / rect border
      const startCol = c
      let runText = ''
      while (
        c < cols &&
        grid[r][c] !== ' ' &&
        !LINE_CHARS.has(grid[r][c]) &&
        !ARROW_CHARS.has(grid[r][c]) &&
        !interiorCells.has(`${r}:${c}`) &&
        !borderCells.has(`${r}:${c}`)
      ) {
        runText += grid[r][c]
        c++
      }
      // Allow single spaces within a run (extend if next non-space is also text)
      while (c < cols && grid[r][c] === ' ' && c + 1 < cols) {
        const next = grid[r][c + 1]
        if (
          next === ' ' ||
          LINE_CHARS.has(next) ||
          ARROW_CHARS.has(next) ||
          interiorCells.has(`${r}:${c + 1}`) ||
          borderCells.has(`${r}:${c + 1}`)
        ) {
          break
        }
        runText += ' '
        c++
      }
      if (runText.trim()) {
        textRuns.push({ r, c: startCol, text: runText.trimEnd() })
      }
    }
  }

  return {
    rows,
    cols,
    rectangles,
    arrows,
    lines,
    textRuns,
    raw: text,
  }
}

/**
 * Try to trace a rectangle starting from a top-left corner.
 * Returns null if the shape isn't a well-formed rectangle.
 */
function tryTraceRect(
  grid: string[][],
  r1: number,
  c1: number,
  rows: number,
  cols: number
): Rectangle | null {
  // Find right boundary: scan right along row r1 until we hit ┐.
  // ALLOW arbitrary text in the top edge (many diagrams put a label inline,
  // like "┌─ SOURCE CODE ──┐"). The edge just needs to START with ─ and END
  // with ─ before the corner; what's in between can be label text.
  const nextR1 = grid[r1][c1 + 1]
  if (nextR1 !== HORIZ && !HORIZ_EDGE_COMPATIBLE.has(nextR1)) return null
  let c2 = -1
  let topEdgeBoxChars = 0
  for (let c = c1 + 1; c < cols; c++) {
    const ch = grid[r1][c]
    if (ch === CORNER_TR) { c2 = c; break }
    if (ch === CORNER_TL || ch === CORNER_BL || ch === CORNER_BR) return null
    if (HORIZ_EDGE_COMPATIBLE.has(ch)) topEdgeBoxChars++
  }
  if (c2 === -1 || c2 - c1 < 3) return null
  // Require at least 2 box-drawing characters on the top edge (we don't want to
  // match arbitrary text runs that happen to start/end with corner glyphs).
  if (topEdgeBoxChars < 2) return null
  // Top edge must end with a horizontal char (right before ┐)
  const lastTop = grid[r1][c2 - 1]
  if (!HORIZ_EDGE_COMPATIBLE.has(lastTop)) return null

  // Find bottom boundary: scan down along col c1 until we hit └
  let r2 = -1
  for (let r = r1 + 1; r < rows; r++) {
    const ch = grid[r][c1]
    if (ch === CORNER_BL) {
      r2 = r
      break
    }
    if (!VERT_EDGE_COMPATIBLE.has(ch)) return null
  }
  if (r2 === -1 || r2 - r1 < 2) return null

  // The bottom-right corner should be near c2. Source files occasionally have
  // off-by-one misalignments between the border rows and interior rows. Look
  // for ┘ in a small window around c2 and adjust c2 if needed.
  let c2Bottom = -1
  for (const offset of [0, 1, -1, 2, -2]) {
    const tryC = c2 + offset
    if (tryC >= 0 && tryC < cols && grid[r2][tryC] === CORNER_BR) {
      c2Bottom = tryC
      break
    }
  }
  if (c2Bottom === -1) return null

  // Use the narrower of top/bottom widths as the canonical rectangle width
  const rectC2 = Math.min(c2, c2Bottom)
  c2 = rectC2

  // Verify bottom edge — same leniency as top (allow label text)
  let bottomBoxChars = 0
  for (let c = c1 + 1; c < c2Bottom; c++) {
    const ch = grid[r2][c]
    if (ch === CORNER_TL || ch === CORNER_TR || ch === CORNER_BL) return null
    if (HORIZ_EDGE_COMPATIBLE.has(ch)) bottomBoxChars++
  }
  if (bottomBoxChars < 2) return null

  // Verify right edge: each interior row must have a vertical char within
  // ±2 columns of c2 (handles off-by-one source misalignments).
  for (let r = r1 + 1; r < r2; r++) {
    let found = false
    for (const offset of [0, 1, -1, 2, -2]) {
      const tryC = c2 + offset
      if (tryC >= 0 && tryC < cols && VERT_EDGE_COMPATIBLE.has(grid[r][tryC])) {
        found = true
        break
      }
    }
    if (!found) return null
  }

  // Collect exit points
  const exits = {
    top: [] as number[],
    bottom: [] as number[],
    left: [] as number[],
    right: [] as number[],
  }
  for (let c = c1 + 1; c < c2; c++) {
    if (grid[r1][c] === T_UP) exits.top.push(c)
    if (grid[r2][c] === T_DOWN) exits.bottom.push(c)
  }
  for (let r = r1 + 1; r < r2; r++) {
    if (grid[r][c1] === T_LEFT) exits.left.push(r)
    if (grid[r][c2] === T_RIGHT) exits.right.push(r)
  }

  // Extract label from top edge if present (characters between ─'s that
  // aren't box-drawing). E.g. "┌─ SOURCE CODE ──┐" → "SOURCE CODE"
  let label: string | undefined
  {
    let labelChars = ''
    for (let c = c1 + 1; c < c2; c++) {
      const ch = grid[r1][c]
      if (!HORIZ_EDGE_COMPATIBLE.has(ch)) labelChars += ch
    }
    const trimmed = labelChars.trim()
    if (trimmed.length > 0) label = trimmed
  }

  // Extract interior text per row. Detect divider rows where the left edge
  // is ├ and right edge is ┤ (these are section separators, not content).
  const lines: string[] = []
  const dividerRows: number[] = []
  for (let r = r1 + 1; r <= r2 - 1; r++) {
    const leftChar = grid[r][c1]
    const rightChar = grid[r][c2]
    if (leftChar === T_RIGHT && rightChar === T_LEFT) {
      // This row is a divider — check interior is all horizontal chars
      let allDivider = true
      for (let c = c1 + 1; c < c2; c++) {
        if (!HORIZ_EDGE_COMPATIBLE.has(grid[r][c])) {
          allDivider = false
          break
        }
      }
      if (allDivider) {
        dividerRows.push(r)
        lines.push('') // placeholder so indexing matches row offset
        continue
      }
    }

    let line = ''
    for (let c = c1 + 1; c <= c2 - 1; c++) {
      line += grid[r][c]
    }
    lines.push(line.replace(/\s+$/g, ''))
  }

  return {
    r1,
    c1,
    r2,
    c2,
    lines,
    label,
    dividerRows,
    exits,
  }
}

function arrowDirection(ch: string): ArrowMark['dir'] | null {
  switch (ch) {
    case '▲':
    case '↑':
      return 'up'
    case '▼':
    case '↓':
      return 'down'
    case '◀':
    case '←':
      return 'left'
    case '▶':
    case '→':
      return 'right'
    default:
      return null
  }
}
