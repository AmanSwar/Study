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

## Reading layout

The site is built for reading 5–12k-word modules for hours, so the chrome stays out of the way:

- **One prose column** (`--measure`, 40 rem ≈ 72 characters) in Source Serif 4 at 18 px; Inter for labels, navigation and captions; JetBrains Mono for data. Figures, tables, steppers, KPI rows and comparison grids are *plates* that break out to `--wide` (56 rem) — the rule is the last section of `study.css`, measured with container queries against `.reading-article`. Code keeps the prose's left edge and grows rightward only as far as its longest line needs.
- **Chrome**: a 48 px bar (breadcrumb, Contents, search, `Aa`, theme) that hides on scroll-down; the course contents as a drawer (`c`); an "on this page" rail at ≥ 80 rem with the current section marked and past sections dimmed; a 2 px progress hairline.
- **Reader settings** (`Aa`): serif/sans, 15–21 px, narrow/normal/wide, theme — stored under `aman.study:reader` and applied to `<html data-font data-size data-width>` before first paint by a script in `app/layout.tsx`.
- **Reading position** (`lib/reading-state.ts`, `aman.study:positions` in localStorage): each module remembers scroll % and current section; reopening offers "Resume § … · 43 %"; ≥ 92 % marks it read. The home page lists *Continue reading*; the syllabus and drawer show read state.
- Markdown tracks render through `MarkdownRenderer` onto the same `study.css` components (`.code-block`, `.tbl`, `.callout`) as HTML modules, so both formats look identical; ASCII diagrams and visualisations sit in `.md-wide` plates.

## Design system for HTML modules — `public/study/`

`study.css` (tokens shared with `app/globals.css` — change both together; components scoped under `.study`; the wide-plate rules are the last section on purpose), `study.js` (code headers/copy, highlight.js, KaTeX auto-render, tables, figure zoom, tabs, steppers, calculators, citation popovers; standalone TOC/theme/progress and web-font loading), `demo.html` (gallery of every component), `vendor/` (KaTeX 0.16, highlight.js 11 — copied from npm, see `../.claude/study/references/components.md`). `study.css` is loaded globally from `app/layout.tsx`.

## Commands

```bash
npm run dev      # dev server; unpublished tracks visible
npm run build    # full prerender
npm run lint
npx tsc --noEmit
```

`scripts/migrate-manifests.ts` is the historical one-off that produced the four legacy manifests from the old `content/*.ts` + per-module pages (its inputs no longer exist; kept for the record, excluded from tsconfig/eslint).
