'use client'

import { useEffect, useRef } from 'react'
import { usePathname } from 'next/navigation'

declare global {
  interface Window {
    Study?: { init(root: HTMLElement): void | (() => void); destroy?(root: HTMLElement): void }
  }
}

let studyJs: Promise<void> | null = null
function loadStudyJs() {
  return (studyJs ??= new Promise<void>((resolve, reject) => {
    if (window.Study) return resolve()
    const s = document.createElement('script')
    s.src = '/study/study.js'
    s.async = true
    s.onload = () => resolve()
    s.onerror = () => { studyJs = null; reject(new Error('study.js failed to load')) }
    document.head.appendChild(s)
  }))
}

// dangerouslySetInnerHTML never executes <script>; clone each one so the browser runs it.
function runInlineScripts(root: HTMLElement) {
  root.querySelectorAll('script').forEach((old) => {
    const s = document.createElement('script')
    for (const { name, value } of Array.from(old.attributes)) s.setAttribute(name, value)
    s.text = old.text
    old.replaceWith(s)
  })
}

/**
 * Mounts a self-contained HTML study module inside the site: injects the
 * module's inner HTML, loads /study/study.js once, re-executes the module's
 * inline scripts, and runs Study.init(root). A fresh root per pathname means
 * no listeners survive navigation.
 */
export function StudyRuntime({ html }: { html: string }) {
  const ref = useRef<HTMLDivElement>(null)
  const pathname = usePathname()

  useEffect(() => {
    const root = ref.current
    if (!root) return
    let cancelled = false
    let cleanup: void | (() => void)
    loadStudyJs()
      .then(() => {
        if (cancelled || !ref.current) return
        runInlineScripts(root)
        cleanup = window.Study?.init(root)
      })
      .catch(console.error)
    return () => {
      cancelled = true
      cleanup?.()
      window.Study?.destroy?.(root)
    }
  }, [pathname, html])

  return <div key={pathname} ref={ref} className="study" dangerouslySetInnerHTML={{ __html: html }} />
}
