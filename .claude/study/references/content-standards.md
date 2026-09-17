# Content standards — what every module is graded against

These rules are enforced by `scripts/lint-module.mjs` where they can be, and by you where they cannot.

## 1. Grounding

**Source hierarchy** (use in this order; cite the highest tier that supports the claim):
1. Official: vendor documentation, ISA/API references, architecture whitepapers, technical briefs, specifications, standards (IEEE/ISO/RFC), the project's own repository and docs.
2. Peer-reviewed papers and arXiv preprints from the originating group or a top venue (NeurIPS/ICML/ICLR, ISCA/MICRO/HPCA/ASPLOS, OSDI/SOSP/NSDI, SIGMOD/VLDB, JFE/JF for finance).
3. Standard textbooks (Hennessy & Patterson, Kilts, Goodfellow, Sutton & Barto, Hull, Grinold & Kahn, etc.) with edition and chapter.
4. Engineering blogs of the organisation that built the thing (NVIDIA Developer, Google Research/Cloud, Meta Engineering, OpenAI, Anthropic, DeepMind, Microsoft Research, Apple ML, Qualcomm Developer, Jane Street, Two Sigma, Netflix Tech, Cloudflare, Uber Eng, Stripe Eng).
5. Talks and slide decks from those organisations (GTC, Hot Chips, ISCA keynotes, Strange Loop, QCon) — cite the talk, year, and timestamp/slide.
6. Reputable third-party analyses (SemiAnalysis, Chips and Cheese, Anandtech archive, Epoch AI, arXiv surveys) — labelled as analysis, never as the sole source for a spec.

**Never** as a source: Medium/Substack posts by unknown authors, SEO content farms, Wikipedia (use its references instead), forum threads, Reddit, unverified tweets, model-generated pages. Forums may be *pointers* to a primary source, not citations.

**Every** number, spec, version, date, benchmark, price, API/instruction name, and quoted claim carries a citation. A paragraph of general mechanism needs a citation at least once. If nothing supports a claim you believe is true: cut it, or keep it as `<mark class="unverified" title="why">…</mark>` (budget: ≤ 8 per module; each one is reported to Aman).

Currency: state the as-of date in the header chips. Flag anything likely to change within a year (prices, roadmap items, "latest" versions).

## 2. Citations (research-paper style)

- Inline: `<a class="cite" href="#ref-7">7</a>` placed immediately after the claim, before punctuation where natural. Multiple: `<a class="cite" href="#ref-2">2</a><a class="cite" href="#ref-5">5</a>`.
- References section (last in the module):
  ```html
  <section id="references">
  <h2 id="references-heading">References</h2>
  <ol class="refs">
    <li id="ref-1"><span class="ref-type">official</span>NVIDIA. <em>NVIDIA Blackwell Architecture Technical Brief</em>. v1.1, 2024. <a href="https://…">https://…</a>. Accessed 2026-09-17. <span class="ref-use">Used for: HBM3e capacity and bandwidth (Table 1), NVLink 5 figures.</span></li>
  </ol>
  </section>
  ```
  Format: `Authors/Org. <em>Title</em>. Venue/publisher, year or version. URL. Accessed YYYY-MM-DD. Used for: …`. `ref-type` ∈ official | standard | paper | textbook | repo | talk | blog | analysis.
- Numbering is per module, in order of first citation, 1…N, contiguous. Decide the reference list before writing (from `sources.json`), append new ones as you discover them.
- Every reference is cited at least once; every citation resolves. URLs must be real and were fetched by the researcher or by you — never reconstructed from memory.
- Cite the specific location when the source is long: `(§4.2)`, `(Table 3)`, `(p. 14)`, `(slide 22)`, `(commit abc123, file x.cu)` inside the `ref-use` note or in prose.

## 3. Structure of a module

```
<header class="study-header">  kicker · h1 · lede (2–3 sentences: what this is, what you can do after) · meta chips (reading time · N sources · as-of date · prerequisites)
Mental-model figure (optional but usual): one diagram that frames the whole module
1. … N.  sections, mechanism-first, each with its own figures/tables as the content demands
Where experts disagree (mandatory; ≥ 2 positions, each cited; pros/cons; your assessment)
What is changing / open questions (optional; short)
References
```
- Headings are numbered manually in the text: `1. `, `2. `, sub-sections `1.1 `. Ids kebab-case, unique. "Where experts disagree" and "References" are unnumbered.
- No exercises, quizzes, self-assessment, deliverables, projects, "check your understanding", "further reading" lists (references already serve that), or summaries that restate the section.
- Prerequisites, if any, go in a header chip and link to the module they depend on (`<a href="../<slug>/NN-x.html">` relative works in both contexts only when the site route matches; prefer naming the module in text).

## 4. Depth and size

- Deep-dive page: 6,000–12,000 words. Course module: 5,000–9,000 words. The topic decides within the band; never pad, never truncate a mechanism to fit.
- Per module: ≥ 3 figures (inline SVG), ≥ 1 comparison or spec table where the topic has comparable things, ≥ 1 worked numeric derivation where a quantity matters, code only when the code *is* the mechanism (an intrinsic, an API, an algorithm) — not as decoration.
- Cover the topic the way the best single expert would explain it to a peer: full mechanism, real numbers, failure modes, what the field has learned since the original paper, and what practitioners actually do.

## 5. Voice

- Direct, technical, candid. Second person is fine ("you cannot skip Hopper"). Contractions fine. American spelling.
- No emojis. No exclamation marks. No marketing adjectives (revolutionary, blazing, seamless).
- Banned phrases: "in this section we will", "let's dive in", "it is important to note", "in conclusion", "as we have seen", "a deep dive into", "unlock", "leverage" (as a verb), "game-changer".
- Numbers: units always; SI prefixes; thousands separators; keep the source's precision (do not round 3.35 TB/s to 3.4). Bytes vs bits explicit (GB vs Gb). Distinguish theoretical peak from measured.
- Math: inline `\( … \)`, display `$$ … $$` on its own lines. Never single `$` as a delimiter — prices appear in prose.
- Code blocks: `<pre class="code" data-lang="…" data-title="…"><code>…</code></pre>`; escape `<`, `>`, `&` inside code as `&lt; &gt; &amp;`.

## 6. Visuals (see diagram-guide.md)

- Draw the mechanism: memory layouts, dataflow, pipelines, timelines, topologies, tiling, state machines, plots of cited numbers.
- Every figure: `<figure class="fig">` with inline `<svg viewBox>` using the `d-*` palette classes (no hex colours) and `<figcaption><b>Figure N.</b> …</figcaption>` that states what to see and cites the data.
- Use a stepper for anything that happens in stages; a calculator when a formula has knobs worth turning; tabs to contrast implementations; a timeline for history.

## 7. Where experts disagree (mandatory)

A `<section class="disagree">` with 2–4 `<article class="position">` cards: position title, who holds it (with citations), the argument, pros, cons — then `<div class="verdict">` with your assessment: which position is right under which conditions, and what evidence would settle it. Disagreements come from the researcher's `sources.json` `disagreements` list plus what you find while writing. If the field genuinely agrees, the section contrasts *approaches* (e.g. static vs dynamic quantisation) with the same rigor — it is never skipped.

## 8. Self-check before you finish

1. `node .claude/study/scripts/lint-module.mjs <file> --json` → zero errors; read every warning.
2. Open the file mentally as Aman: is there a paragraph he could have written himself from general knowledge? Cut or deepen it.
3. Every figure earns its place (would a table say it better?). Every table has units in headers.
4. Every number: source? precision? unit? as-of?
5. Disagreement section present with real, cited positions.
