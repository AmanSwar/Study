---
name: study-deepdive
description: 'Deep dive — exhaustive, citation-backed, visual-first HTML study material on ONE narrow topic (a chip, an architecture, a library, a mechanism, a practice) for aman.study, researched live from primary sources. Use when Aman wants to learn a specific thing in depth rather than a whole field; for a whole field use /study-course.'
when_to_use: 'deep dive on X; everything about X; I want to learn X in depth; make me study material on <specific topic>; explore all <product family>'
argument-hint: '<topic> [--category cs|finance|maths|business] [--go] [--pages N] [--parallel N]'
model: sonnet
effort: xhigh
allowed-tools: Agent, WebSearch, WebFetch, Read, Write, Edit, Glob, Grep, AskUserQuestion, Bash(node .claude/study/scripts/*), Bash(node ${CLAUDE_PROJECT_DIR}/.claude/study/scripts/*), Bash(mkdir *), Bash(ls *), Bash(cat *), Bash(curl *)
---

# /study-deepdive — coordinator

You coordinate; the agents do the heavy work. Keep your own output short. Every stage runs on Sonnet at xhigh effort (set in the agent definitions). The result is a flat track of 1–6 HTML pages under `<category dir>/<slug>/`, auto-registered in the site through `manifest.json`.

Request: `$ARGUMENTS`

## 0. Intake (you)
1. Parse: topic (everything that is not a flag), `--category` (default `cs`), `--go` (skip the checkpoint), `--pages N` (cap pages), `--parallel N` (writers per batch, default 4). Inline constraints in the request (scope limits, must-include sources, audience overrides) are passed **verbatim** to every agent.
2. Derive: `slug` = kebab-case of the topic (≤ 40 chars, no stop words); category dir = `computer science/` | `finance/` | `maths/` | `business/`; manifest `category` = `Computer Science` | `Finance` | `Mathematics` | `Business`; `<dir>` = `<category dir>/<slug>`.
   Pick `color`/`icon` by topic: hardware → cyan/Cpu, ML systems → blue/Brain, distributed/infra → blue/Server, software practice → orange/Layers, finance → green/TrendingUp, maths → purple/Sigma, networking → cyan/Network, data → amber/Database.
3. If `<dir>/manifest.json` exists: stop and tell Aman to use `/study-continue <slug>`.
4. Create the track:
   ```
   node .claude/study/scripts/manifest.mjs init "<dir>" --id <slug> --title "<Title>" --shortTitle "<≤18 chars>" --description "<one sentence>" --category "<Category>" --kind deep-dive --color <c> --icon <I> --skill study-deepdive --request "<verbatim request>"
   ```
5. Say in one line what you are doing: topic, slug, category, and that research is starting (20–40 min).

## 1. Research (agent `study-researcher`, foreground)
Call `Agent` with `subagent_type: "study-researcher"`, `run_in_background: false`, prompt:
> Topic: <topic>. Kind: deep-dive. Track directory: `<dir>` (repo-relative). Constraints from Aman: <verbatim or "none">. Read `.claude/study/references/research-guide.md`, `learner-profile.md`, and `content-standards.md` §1 first. Produce `<dir>/sources.json` and `<dir>/sources.md` per the guide (15–25 sources, ≤ 60 fetches), then report as the guide specifies.

Then `node .claude/study/scripts/manifest.mjs set-track "<dir>/manifest.json" --stage plan`. If `sources.json` is missing or has < 8 sources, tell Aman and stop.

## 2. Plan (agent `study-planner`, foreground)
Call `Agent` with `subagent_type: "study-planner"`, `run_in_background: false`, prompt:
> Kind: deep-dive (guide: `.claude/study/references/deepdive-guide.md`). Track directory: `<dir>`. Topic: <topic>. Constraints: <verbatim or "none">. <If --pages N: "Maximum N pages.">. Read the guide, `learner-profile.md`, `manifest-schema.md`, and `<dir>/sources.json`. Write `<dir>/outline.md` and `<dir>/plan.json`, apply it with `manifest.mjs apply-plan`, validate, and end with the checkpoint text.

## 3. Checkpoint (you) — unless `--go`
Show the planner's checkpoint text verbatim, then `AskUserQuestion` with options: **Approve** / **Edit** (free text: what to change) / **Cancel**.
- Approve → `manifest.mjs set-track "<dir>/manifest.json" --approved true --stage write --status published`.
- Edit → re-run step 2 with the edit text appended as "Edits requested by Aman: …", then ask again.
- Cancel → `set-track --stage plan`; stop.
With `--go`: approve automatically and print the checkpoint text for the record.

## 4. Write (agents `study-writer`, background, 4 at a time)
`node .claude/study/scripts/manifest.mjs next "<dir>/manifest.json"` lists pages not yet published. Take up to **4** (or `--parallel N`; use 2 when several study sessions run at once to stay under API rate limits); for each, one `Agent` call with `subagent_type: "study-writer"`, `run_in_background: true`, all in the same message. Prompt:
> Write module `<id>` of track `<slug>` (kind: deep-dive). Track directory: `<dir>`; output file: `<dir>/<file>`; `{{STUDY_BASE}}` = `../../website/public/study`. Read the references listed in your instructions, your manifest entry, the neighbouring pages' scopes (<prev id / next id, or "none">), and `<dir>/sources.json`. Word band 6000–12000 (`--min-words 6000 --max-words 12000`). Constraints from Aman: <verbatim or "none">. Follow your procedure exactly: plan file → write incrementally → lint to zero errors → `manifest.mjs set-module … --status published --from-lint …` → report.

Wait for all notifications of the batch, note each writer's report (especially `unverified` lists and `failed` status), then launch the next batch until `next` returns `[]`. A `failed` module gets one retry with the writer's "what remains" note appended; if it fails again, leave it `failed` and report.

## 5. Assemble (you)
```
node .claude/study/scripts/validate-manifest.mjs "<dir>/manifest.json" --strict
node .claude/study/scripts/manifest.mjs set-track "<dir>/manifest.json" --stage done
```
If any module is `failed`, keep the track `published` (finished pages are readable) and say which are missing.

## 6. Report to Aman (≤ 30 lines)
- Track: title, `<dir>`, pages with words / figures / tables / citations / reading time each.
- Sources used (count by type) and the top 5.
- Every `unverified` claim, verbatim, with its page.
- Anything cut for lack of sources; anything `failed`.
- How to view: `cd website && npm run dev` → `http://localhost:3000/<slug>`; or open `<dir>/<file>` directly. Nothing is committed — that is his call.

## Timing and failure notes
Research 20–40 min; each page 10–20 min; up to 4 pages in parallel. If a stage is interrupted, `/study-continue <slug>` resumes from the manifest. Never write module content yourself; never skip the linter; never fabricate a source to fill a gap.
