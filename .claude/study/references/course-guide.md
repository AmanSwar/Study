# Course guide — designing a syllabus

A course is a university-style treatment of a whole field: parts → modules, ordered by dependency, each module 5,000–9,000 words as a self-contained HTML module. The model is a graduate seminar at Stanford/MIT/Berkeley taught by the practitioner who built the systems — not a survey and not a bootcamp.

## Size
- Parts: 3–8. Modules: 10–40 total, 2–7 per part. Numbering is continuous across parts (Module 1…N).
- A module is one session's worth of mechanism (5–9k words, 3–8 figures). If a topic needs 15k words it is two modules; if it needs 2k it is a section of another module.

## Method
1. Read `sources.json`: `subtopics`, `coverage`, `disagreements`. Only `well`/`thin` subtopics may become modules; `thin` ones get a narrower scope and the writer is told to fetch more.
2. Draft the dependency graph of subtopics (what must be understood before what). Parts are the strongly-connected layers of that graph; modules are its nodes in topological order.
3. Apply the learner profile: skip what he already has (list it explicitly in "excluded"), start where a strong engineer's knowledge actually stops, and end at the frontier (what is unsettled as of today).
4. Every module gets:
   - `id` (`module-NN` or `NN-slug`), `number`, `title` (specific), `shortTitle` (≤ 24 chars), `description` (one sentence).
   - `scope`: one paragraph — in, out, what you can do afterwards.
   - `topics`: 6–12 mechanism-level bullets.
   - `sourceIds`: ≥ 3 grounding sources, at least one official/paper.
   - `estWords`: 5,000–9,000.
   - `figures`: 3–8 planned diagrams (one line each).
   - `disagreements`: which `sources.json.disagreements` entries it treats.
   - `prerequisites`: module numbers it depends on (as strings like "Module 3").
5. Each part gets `id` (`part-NN-slug`), `number`, `title`, `shortTitle`, `description` (one sentence on what the part unlocks).
6. `syllabus.md`: the full syllabus in reading order, with the part/module scope lines, the dependency rationale, the excluded list with reasons, and an honest note on where sources are thin.

## What a great syllabus has that a mediocre one lacks
- Modules named by mechanism ("Paged KV cache and block tables"), not by buzzword ("Advanced serving").
- A first module that establishes the quantitative frame (the arithmetic/roofline/cost model of the field) that every later module refers back to.
- Disagreements distributed where they arise, not collected into one "controversies" module.
- A final module on the frontier: open problems, what is changing, what to watch — with sources.
- No "capstone project", no "assessment", no "week-by-week schedule".

## Checkpoint text
The coordinator shows Aman: parts → modules with one-line scopes and estimated words, total words and modules, the excluded list, the thin-coverage note, and the source-map summary (count by type; top 10 sources). Keep it under 90 lines.
