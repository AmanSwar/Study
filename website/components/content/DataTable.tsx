'use client'

import { useState } from 'react'
import { ArrowUpDown } from 'lucide-react'

interface DataTableProps {
  headers: string[]
  rows: string[][]
  caption?: string
  highlightColumn?: number
  sortable?: boolean
}

export function DataTable({ headers, rows, caption, highlightColumn, sortable = false }: DataTableProps) {
  const [sortCol, setSortCol] = useState<number | null>(null)
  const [sortDir, setSortDir] = useState<'asc' | 'desc'>('asc')

  const handleSort = (colIdx: number) => {
    if (!sortable) return
    if (sortCol === colIdx) {
      setSortDir(sortDir === 'asc' ? 'desc' : 'asc')
    } else {
      setSortCol(colIdx)
      setSortDir('asc')
    }
  }

  const sortedRows = sortCol !== null
    ? [...rows].sort((a, b) => {
        const valA = a[sortCol] || ''
        const valB = b[sortCol] || ''
        const numA = parseFloat(valA.replace(/[^0-9.-]/g, ''))
        const numB = parseFloat(valB.replace(/[^0-9.-]/g, ''))
        if (!isNaN(numA) && !isNaN(numB)) {
          return sortDir === 'asc' ? numA - numB : numB - numA
        }
        return sortDir === 'asc'
          ? valA.localeCompare(valB)
          : valB.localeCompare(valA)
      })
    : rows

  return (
    <div className="my-6 rounded-xl border border-border-primary overflow-hidden">
      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="bg-bg-surface border-b border-border-primary">
              {headers.map((header, i) => (
                <th
                  key={i}
                  onClick={() => handleSort(i)}
                  className={`px-4 py-3 text-left text-xs font-semibold uppercase tracking-wider
                    ${highlightColumn === i ? 'text-accent-blue' : 'text-text-tertiary'}
                    ${sortable ? 'cursor-pointer hover:text-text-secondary select-none' : ''}`}
                >
                  <span className="flex items-center gap-1">
                    {header}
                    {sortable && (
                      <ArrowUpDown className={`w-3 h-3 ${sortCol === i ? 'text-accent-blue' : 'opacity-30'}`} />
                    )}
                  </span>
                </th>
              ))}
            </tr>
          </thead>
          <tbody className="divide-y divide-border-subtle">
            {sortedRows.map((row, i) => (
              <tr
                key={i}
                className="hover:bg-bg-surface-hover transition-colors"
              >
                {row.map((cell, j) => (
                  <td
                    key={j}
                    className={`px-4 py-3 text-sm whitespace-nowrap
                      ${j === 0 ? 'font-medium text-text-primary' : 'text-text-secondary'}
                      ${highlightColumn === j ? 'text-accent-blue font-medium' : ''}`}
                  >
                    {cell}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {caption && (
        <div className="px-4 py-2 border-t border-border-primary bg-bg-surface/30">
          <p className="text-xs text-text-tertiary italic">{caption}</p>
        </div>
      )}
    </div>
  )
}
