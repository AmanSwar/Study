# Research guide — building the source map

You (the `study-researcher` agent) produce two files in the track directory:
- `sources.json` — structured, consumed by the planner and every writer.
- `sources.md` — the same information for humans; rendered on the site at `/<track-id>/sources`.

Everything downstream is grounded in what you fetch here. Writers may fetch more, but the plan is only allowed to schedule what you have shown can be grounded.

## Targets
| Kind | Sources | Fetch budget | Time |
|---|---|---|---|
| deep-dive | 15–25 | ≤ 60 WebFetch/curl calls | 20–40 min |
| course | 30–60 | ≤ 120 calls | 40–90 min |

Quality beats count. Ten official documents outrank forty blog posts.

## Procedure

1. **Frame the topic** in one paragraph: the subtopics a complete treatment needs (write them as a numbered list — the planner reuses them as `coverage` keys). Use the learner profile to decide what "complete" means for him.
2. **Search in tiers**, in this order, with `WebSearch` (2–4 queries per tier; refine with product names, version numbers, `site:` filters):
   1. Official docs / whitepapers / specs / repos (`site:docs.nvidia.com`, `site:developer.apple.com`, `site:github.com <org>`, `site:arxiv.org`, `filetype:pdf whitepaper`).
   2. Primary papers (arXiv id, venue). Prefer the paper over the blog about the paper.
   3. Textbooks (identify edition and chapter; you usually cannot fetch the text — cite the ToC/preview page and mark `fetched: false`).
   4. Builder-org engineering blogs and talks (GTC/Hot Chips slides are gold for numbers).
   5. Reputable third-party analyses (label `type: analysis`).
   Reject: Medium/Substack unknowns, SEO farms, Wikipedia (mine its references instead), forums, model-generated pages.
3. **Fetch and extract.** For each kept source:
   - HTML pages: `WebFetch` with a prompt that demands **verbatim** figures — e.g. *"Extract verbatim, with the section/table where each appears: every numeric spec (bandwidth, capacity, FLOPS, latency, sizes, prices, dates, version numbers), every named instruction/API/feature, and any explicit comparison or claim about trade-offs. Quote, do not paraphrase. List what the page does NOT cover."* Then a second prompt if the page is long: *"List the section headings and the exact numbers in tables 1–N."*
   - PDFs (papers, whitepapers, slide decks): `curl -L -o .cache/<id>.pdf "<url>"` into the track's `.cache/` directory (gitignored), then `Read` the PDF with `pages` in ranges of ≤ 20. Read the sections that carry numbers and claims; skip related-work boilerplate. arXiv: use `https://arxiv.org/pdf/<id>` (add `v<n>` for a fixed version and record it).
   - GitHub: fetch the README and the specific source/doc files that matter (`https://raw.githubusercontent.com/<org>/<repo>/<branch>/<path>`); record the commit or tag.
   - Record `accessed` as today's date, and the `version`/`date` the document states about itself.
4. **Write facts, not summaries.** For each source, 5–30 `facts`: each a verbatim or near-verbatim claim/figure with its location (`§3.2`, `Table 2`, `p. 14`, `slide 22`, `README#quantization`). These become the writers' citations. Prefer numbers, definitions, mechanisms, and explicit trade-off statements.
5. **Log conflicts as you notice them**: two sources giving different numbers or opposite recommendations. Record both sides with source ids — the writer's "Where experts disagree" section is built from these.
6. **Coverage map**: for every subtopic from step 1: `well` (≥ 2 primary sources with facts), `thin` (1 source or only analysis-tier), `none`. The planner must not schedule a module on `none`, and must scope `thin` ones honestly.
7. **Write `sources.json` and `sources.md`.** Validate the JSON parses (`node -e "JSON.parse(require('fs').readFileSync('sources.json','utf8'))"`).

## `sources.json` schema

```jsonc
{
  "topic": "…", "kind": "deep-dive|course", "createdAt": "ISO", "researcher": "study-researcher",
  "subtopics": [ { "id": "hbm", "title": "HBM generations and bandwidth" } ],
  "sources": [
    {
      "id": "S1",                                   // stable; writers cite by this id then renumber per module
      "type": "official|standard|paper|textbook|repo|talk|blog|analysis",
      "title": "…", "authors": "…", "org": "…", "year": "2024", "version": "v1.1 / commit / arXiv v2",
      "url": "https://…", "accessed": "2026-09-17", "fetched": true,
      "authority": "why this is a primary/authoritative source (one line)",
      "covers": ["hbm", "nvlink"],
      "facts": [
        { "claim": "B200 HBM3e capacity 192 GB, 8 TB/s", "where": "Table 1", "quote": "…verbatim…" }
      ],
      "conflicts": ["S4 gives 7.7 TB/s (measured) vs 8 TB/s (peak) here"]
    }
  ],
  "disagreements": [
    { "topic": "Are hand-written kernels worth it?", "positions": [ { "claim": "…", "sources": ["S3"] }, { "claim": "…", "sources": ["S1","S7"] } ] }
  ],
  "coverage": { "well": ["hbm"], "thin": ["nvfp4-accuracy"], "none": ["gb300-pricing"] },
  "notes": "anything the planner/writers should know (paywalls, missing docs, version ambiguity)"
}
```

## `sources.md` layout

```
# Sources — <topic>
As of YYYY-MM-DD · N sources · kind
## Coverage
well / thin / none lists
## Disagreements found
- topic — position A [S1, S3] vs position B [S4]
## Sources
### S1 · official · Org — Title (year/version)
URL · accessed · authority line
Facts:
- claim (where)
Conflicts: …
```

## Rules
- Never invent a URL. If you cannot fetch it, mark `fetched: false` and say what you know about it and from where.
- Prefer the newest official version, but keep the older one when the topic is generational (record both).
- When a vendor number and a measured number differ, keep both and label which is which.
- Do not summarise the topic in prose — that is the writer's job. Your product is the map and the facts.
