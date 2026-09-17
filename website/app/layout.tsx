import type { Metadata } from 'next'
import { Inter, JetBrains_Mono, Source_Serif_4 } from 'next/font/google'
import './globals.css'
import { ThemeProvider } from '@/components/layout/ThemeProvider'
import { SearchProvider } from '@/components/search/SearchProvider'
import { KeyboardShortcutsHelp } from '@/components/layout/KeyboardShortcutsHelp'

// Self-hosted via next/font: no external requests, no layout shift.
// Serif carries the reading stream, Inter the chrome and labels, mono the data.
const sourceSerif = Source_Serif_4({
  subsets: ['latin'],
  display: 'swap',
  variable: '--font-serif',
  axes: ['opsz'],
  style: ['normal', 'italic'],
})

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

// Reader preferences (typeface, size, measure) go onto <html> before first paint so
// the page never flashes the defaults. Mirrors READER_KEY in ReaderSettings.
const readerBootstrap = `try{var p=JSON.parse(localStorage.getItem('aman.study:reader')||'{}'),h=document.documentElement;if(p.font)h.dataset.font=p.font;if(p.size)h.dataset.size=p.size;if(p.width)h.dataset.width=p.width}catch(e){}`

export const metadata: Metadata = {
  title: {
    default: 'aman.study',
    template: '%s · aman.study',
  },
  description:
    'A personal library of expert-level courses and deep dives — ML systems, hardware architecture, software engineering, quantitative finance — researched from primary sources and written to be read.',
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
      className={`${sourceSerif.variable} ${inter.variable} ${jetbrainsMono.variable}`}
    >
      <head>
        <script dangerouslySetInnerHTML={{ __html: readerBootstrap }} />
        <link rel="stylesheet" href="/study/study.css" />
      </head>
      <body className="min-h-screen flex flex-col">
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
