---
name: study-continue
description: 'Resume an unfinished deep dive or course from its manifest.json — picks up at the first incomplete stage (research, plan, checkpoint, remaining modules, assemble). Use when Aman says "continue <slug>", "finish the <x> course", "resume the deep dive", or when a previous /study-deepdive or /study-course run was interrupted.'
when_to_use: 'continue <slug>; resume <slug>; finish the <topic> course; pick up the deep dive where it stopped'
argument-hint: '<slug or topic words> [--go]'
model: sonnet
effort: xhigh
allowed-tools: Agent, WebSearch, WebFetch, Read, Write, Edit, Glob, Grep, AskUserQuestion, Bash(node .claude/study/scripts/*), Bash(node ${CLAUDE_PROJECT_DIR}/.claude/study/scripts/*), Bash(ls *), Bash(cat *), Bash(find *)
---

# /study-continue — resume from the manifest

Request: `$ARGUMENTS`

## 1. Locate
Find the track: `find . -maxdepth 3 -name manifest.json -not -path "./website/*" -not -path "*/node_modules/*"` and match the slug (or topic words) against `id`/`title`. If several match, ask which. If none, say so and suggest `/study-deepdive` or `/study-course`.

Print `node .claude/study/scripts/manifest.mjs status "<dir>/manifest.json"` and read `generator.kind`, `pipeline.stage`, `pipeline.approved`, and per-module status.

## 2. Resume at the first incomplete stage
Follow the matching skill's procedure from that stage on — read `.claude/skills/study-deepdive/SKILL.md` or `.claude/skills/study-course/SKILL.md` (by `kind`) for the exact agent prompts, batch size (4), word bands, and the report format. Constraints from the original request are in `generator.request`; pass them verbatim to agents.

| Manifest says | Do |
|---|---|
| no `sources.json` or stage `research` | Stage 1 (research) onward |
| `sources.json` present, no modules planned, or stage `plan` | Stage 2 (plan) onward |
| stage `checkpoint` / `approved: false` | Stage 3 (checkpoint) — show `outline.md`/`syllabus.md` and ask; `--go` approves |
| stage `write`, some modules not `published` | Stage 4 for exactly those modules (`manifest.mjs next`); `failed` modules get one retry with their `note` |
| every module `published`, stage ≠ `done` | Stage 5 (assemble) + report |
| stage `done` | Say it is complete; offer to re-run a specific module if he names one (set it to `planned` with `manifest.mjs set-module <id> --status planned`, then Stage 4) |

## 3. Report
Same report format as the originating skill, prefixed with what was resumed and what was already there.
