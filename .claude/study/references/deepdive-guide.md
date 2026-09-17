# Deep-dive guide — decomposing one topic into pages

A deep dive is exhaustive treatment of one narrow topic (a chip family, an architecture, a library, a practice, a mechanism). It is *flat*: 1–6 pages, each 6,000–12,000 words, each a self-contained HTML module. No parts.

## Decide the page count
- One page when the topic is a single mechanism or artefact (e.g. "PagedAttention", "TPU v7 architecture", "Rust ownership model").
- One page per natural unit when the topic enumerates units (e.g. "all NVIDIA inference hardware" → one page per generation or product family that Aman would actually choose between, plus one cross-cutting comparison page). Pages must be readable in any order; the comparison page is last.
- Never split a mechanism across pages; never make a page just to hit a count. Six is the maximum; if the topic needs more, tell the coordinator it is a course.

## Page outline (what the planner writes into `outline.md` and the manifest)
For each page:
- `id` (`NN-slug`), `number`, `title` (specific, not "Introduction"), `shortTitle` (≤ 24 chars), `description` (one sentence).
- `scope`: one paragraph — what is in, what is out, and what the reader can do afterwards.
- `topics`: 6–12 bullets, mechanism-level (not "overview of X" but "how X's write path orders stores").
- `sourceIds`: the sources from `sources.json` that ground this page (≥ 3, at least one official/paper).
- `estWords`: 6,000–12,000.
- `figures`: 3–8 planned diagrams, each one line ("memory hierarchy with capacities and BW per level", "timeline of generations with process node and HBM", "roofline for decode at B = 1…1024").
- `disagreements`: which entries from `sources.json.disagreements` this page will treat.

Also in `outline.md`: a 3–5 line "what is deliberately excluded and why" list, and the reading order rationale.

## Ordering
Mechanism before consequence; hardware before software that targets it; the general case before the exceptions; the comparison page last.

## Checkpoint text
The coordinator shows Aman: the page list with scope lines and estimated words, the figure plan, the excluded list, and the source-map summary (count by type; the top 8 sources with one-line authority). Keep it under 60 lines.
