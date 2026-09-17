import type { Metadata } from 'next'
import { Inter, JetBrains_Mono } from 'next/font/google'
import './globals.css'
import { ThemeProvider } from '@/components/layout/ThemeProvider'
import { SearchProvider } from '@/components/search/SearchProvider'
import { KeyboardShortcutsHelp } from '@/components/layout/KeyboardShortcutsHelp'

// Self-hosted fonts via next/font — zero external requests, no layout shift,
// automatic subsetting and weight optimization.
const inter = Inter({
  subsets: ['latin'],
  display: 'swap',
  variable: '--font-inter',
  axes: ['opsz'],
})

const jetbrainsMono = JetBrains_Mono({
  subsets: ['latin'],
  display: 'swap',
  variable: '--font-mono',
})

export const metadata: Metadata = {
  title: {
    default: 'aman.study — PhD-level study material',
    template: '%s · aman.study',
  },
  description:
    'A personal knowledge base of expert-level courses and deep dives — ML systems, hardware architecture, software engineering, quantitative finance — researched from primary sources.',
  metadataBase: new URL('https://aman-study.vercel.app'),
  openGraph: {
    title: 'aman.study',
    description: 'Expert-level courses and deep dives, researched from primary sources.',
    type: 'website',
  },
}

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode
}>) {
  return (
    <html
      lang="en"
      suppressHydrationWarning
      className={`${inter.variable} ${jetbrainsMono.variable}`}
    >
      <body className="min-h-screen flex flex-col bg-bg-primary text-text-primary antialiased font-sans">
        <ThemeProvider>
          <SearchProvider>
            {children}
            <KeyboardShortcutsHelp />
          </SearchProvider>
        </ThemeProvider>
      </body>
    </html>
  )
}
