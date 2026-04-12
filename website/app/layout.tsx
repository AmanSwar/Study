import type { Metadata } from 'next'
import './globals.css'
import { ThemeProvider } from '@/components/layout/ThemeProvider'

export const metadata: Metadata = {
  title: {
    default: 'CS Study — PhD-Level Computer Science Curriculum',
    template: '%s | CS Study',
  },
  description:
    'Interactive study platform covering ML Systems, CPU Architecture, and Qualcomm Hexagon NPU. 76 modules with interactive diagrams and animated visualizations.',
}

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode
}>) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body className="min-h-screen flex flex-col bg-bg-primary text-text-primary antialiased">
        <ThemeProvider>
          {children}
        </ThemeProvider>
      </body>
    </html>
  )
}
