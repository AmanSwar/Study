'use client'

import { createContext, useContext, ReactNode } from 'react'
import { useTheme } from 'next-themes'

export interface DiagramColors {
  bg: string
  blockPrimary: string
  blockSecondary: string
  blockAccent: string
  blockMuted: string
  arrow: string
  arrowActive: string
  text: string
  textLight: string
  highlight: string
  tooltipBg: string
  tooltipBorder: string
  grid: string
  // Semantic colors
  compute: string
  memory: string
  success: string
  warning: string
  danger: string
}

const darkColors: DiagramColors = {
  bg: '#0f172a',
  blockPrimary: '#1e40af',
  blockSecondary: '#164e63',
  blockAccent: '#ea580c',
  blockMuted: '#334155',
  arrow: '#06b6d4',
  arrowActive: '#f97316',
  text: '#e2e8f0',
  textLight: '#94a3b8',
  highlight: '#fbbf24',
  tooltipBg: '#1e293b',
  tooltipBorder: '#334155',
  grid: '#1e293b',
  compute: '#3b82f6',
  memory: '#f97316',
  success: '#22c55e',
  warning: '#f59e0b',
  danger: '#ef4444',
}

const lightColors: DiagramColors = {
  bg: '#f8fafc',
  blockPrimary: '#2563eb',
  blockSecondary: '#0891b2',
  blockAccent: '#ea580c',
  blockMuted: '#e2e8f0',
  arrow: '#0891b2',
  arrowActive: '#f97316',
  text: '#1e293b',
  textLight: '#64748b',
  highlight: '#f59e0b',
  tooltipBg: '#ffffff',
  tooltipBorder: '#e2e8f0',
  grid: '#f1f5f9',
  compute: '#2563eb',
  memory: '#ea580c',
  success: '#16a34a',
  warning: '#d97706',
  danger: '#dc2626',
}

const DiagramThemeContext = createContext<DiagramColors>(darkColors)

export function DiagramThemeProvider({ children }: { children: ReactNode }) {
  const { resolvedTheme } = useTheme()
  const colors = resolvedTheme === 'light' ? lightColors : darkColors

  return (
    <DiagramThemeContext.Provider value={colors}>
      {children}
    </DiagramThemeContext.Provider>
  )
}

export function useDiagramTheme(): DiagramColors {
  return useContext(DiagramThemeContext)
}
