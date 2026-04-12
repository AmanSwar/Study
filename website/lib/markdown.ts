import { promises as fs } from 'fs'
import path from 'path'

/**
 * Root directory of the study material repository.
 * The website lives at <repo>/website/, so source files are one level up.
 */
const REPO_ROOT = path.resolve(process.cwd(), '..')

/**
 * Source directories mapped by prefix.
 * - Paths starting with "MLsys/", "intel/", "qualcomm/" are under "computer science/"
 * - Paths starting with "From_Zero_to_Quant/" are directly under the repo root
 */
function resolveSourcePath(relativePath: string): string {
  if (relativePath.startsWith('From_Zero_to_Quant/')) {
    return path.join(REPO_ROOT, relativePath)
  }
  // Default: under "computer science/"
  return path.join(REPO_ROOT, 'computer science', relativePath)
}

/**
 * Read a source markdown file and return its content as a string.
 *
 * Examples:
 *   loadMarkdown('MLsys/part_01_.../module_01_....md')          → computer science/MLsys/...
 *   loadMarkdown('From_Zero_to_Quant/Chapter_01_....md')        → From_Zero_to_Quant/...
 */
export async function loadMarkdown(relativePath: string): Promise<string> {
  const fullPath = resolveSourcePath(relativePath)
  const content = await fs.readFile(fullPath, 'utf-8')
  return content
}

/**
 * Extract a list of headings (## level) from markdown for the table of contents.
 * Returns an array of { id, title, level }.
 */
export function extractHeadings(content: string): Array<{ id: string; title: string; level: number }> {
  const headings: Array<{ id: string; title: string; level: number }> = []
  const lines = content.split('\n')
  let inCodeBlock = false

  for (const line of lines) {
    // Skip code fences
    if (line.startsWith('```')) {
      inCodeBlock = !inCodeBlock
      continue
    }
    if (inCodeBlock) continue

    // Match ## heading or ### heading
    const match = line.match(/^(#{2,3})\s+(.+?)\s*$/)
    if (match) {
      const level = match[1].length
      const title = match[2].replace(/^\d+\.\s*/, '').trim() // strip leading "1. "
      const fullTitle = match[2].trim()
      const id = slugify(fullTitle)
      headings.push({ id, title: fullTitle, level })
    }
  }

  return headings
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

/**
 * Strip the first H1 (# ...) from a markdown file. The page renders the title
 * via ModuleHeader, so we don't want it duplicated.
 */
export function stripFirstH1(content: string): string {
  const lines = content.split('\n')
  let firstH1Index = -1

  for (let i = 0; i < lines.length; i++) {
    if (lines[i].match(/^#\s+/)) {
      firstH1Index = i
      break
    }
    // If we hit non-empty content before any H1, stop
    if (lines[i].trim() && !lines[i].startsWith('#')) break
  }

  if (firstH1Index === -1) return content
  return lines.slice(firstH1Index + 1).join('\n').trimStart()
}
