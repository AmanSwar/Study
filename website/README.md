# aman.study — website

Next.js 16 (App Router) site that renders the study material in the parent repository.

## How content gets in

There are no per-module pages. `lib/registry.ts` walks the repository root (`..`) at build time for every `manifest.json` (depth ≤ 3, skipping `website/`, `node_modules/`, `.git/`, `none/`), validates it, and resolves URLs:

```
/<track>                          track index (parts grid or module cards + appendices)
/<track>/<part>                   part index
/<track>/<part>/<module>          module (hierarchical tracks)
/<track>/<module>                 module (flat tracks: deep dives, quant, qualcomm)
/<track>/appendices[/<id>]        appendices
/<track>/sources                  the research source map (generated tracks)
/search-index.json                prerendered search index, fetched lazily by ⌘K
```

`app/[track]/layout.tsx` + `app/[track]/page.tsx` + `app/[track]/[...slug]/page.tsx` handle all of it with `generateStaticParams` (`dynamicParams = false`). The manifest schema is documented in `../.claude/study/references/manifest-schema.md`. Tracks/modules whose `status` is not `published` are hidden in production and visible in `next dev`.

Two module formats:
- `format: "md"` — legacy markdown, rendered by `components/content/MarkdownModulePage.tsx` (react-markdown + KaTeX + ASCII→SVG diagrams).
- `format: "html"` — self-contained HTML modules written by the study skills. `lib/html-module.ts` extracts the inner HTML of `<main class="study">`, `components/content/HtmlModulePage.tsx` injects it, and `StudyRuntime.tsx` loads `/study/study.js` and runs `Study.init(root)` after every navigation. The module's own `<head>` is ignored; the site supplies `/study/study.css`.

## Design system for HTML modules — `public/study/`

`study.css` (tokens shared with `app/globals.css`; components scoped under `.study`), `study.js` (code headers/copy, highlight.js, KaTeX auto-render, tables, figure zoom, tabs, steppers, calculators, citation popovers; standalone TOC/theme/progress), `demo.html` (gallery of every component), `vendor/` (KaTeX 0.16, highlight.js 11 — copied from npm, see `../.claude/study/references/components.md`).

## Commands

```bash
npm run dev      # dev server; unpublished tracks visible
npm run build    # full prerender
npm run lint
npx tsc --noEmit
```

`scripts/migrate-manifests.ts` is the historical one-off that produced the four legacy manifests from the old `content/*.ts` + per-module pages (its inputs no longer exist; kept for the record, excluded from tsconfig/eslint).
