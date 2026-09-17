# manifest.json — the contract between the study skills and the site

One `manifest.json` per track directory (`<category dir>/<slug>/manifest.json`). The site
(`website/lib/registry.ts`) discovers every manifest under the repo root (depth ≤ 3) at build
time. Nothing else needs editing for a new track to appear.

```jsonc
{
  "version": 1,
  "id": "nvidia-inference-hardware",      // URL segment. /^[a-z0-9][a-z0-9-]*$/. Unique across the repo.
                                          // Reserved (rejected): study, search-index.json, favicon.ico, api
  "title": "NVIDIA Inference Hardware",   // full title (home card, track index)
  "shortTitle": "NVIDIA HW",              // ≤ 18 chars: sidebar, breadcrumbs, search badge
  "description": "One paragraph. What this covers and for whom.",
  "category": "Computer Science",         // home page groups by this. Use exactly one of:
                                          // "Computer Science" | "Finance" | "Mathematics" | "Business"
  "kind": "deep-dive",                    // "deep-dive" | "course"
  "color": "green",                       // blue | cyan | orange | green | amber | red | purple
  "icon": "Cpu",                          // lucide icon name (Cpu, CircuitBoard, Microchip, Smartphone,
                                          // TrendingUp, BookOpen, Server, Brain, Network, Database, Layers, Zap, Globe, Sigma)
  "unitLabel": "Module",                  // "Module" | "Chapter" | "Page"  (used in titles: "Module 3: …")
  "status": "published",                  // site shows the track only when "published".
                                          // skill lifecycle: researching → planned → writing → published
  "order": 10,                            // optional: sort key within the category on the home page (lower first; default 100)
  "sources": "sources.md",                // optional; rendered at /<id>/sources
  "generator": {                          // skill bookkeeping — the site ignores this object
    "skill": "study-deepdive",            // which command created it
    "request": "verbatim request text",
    "createdAt": "2026-09-17T10:00:00Z",
    "updatedAt": "2026-09-17T12:30:00Z",
    "profile": "learner-profile.md",
    "pipeline": { "stage": "write", "approved": true, "approvedAt": "…" }
                                          // stage: research | plan | checkpoint | write | done
  },

  // EXACTLY ONE of "modules" (flat) or "parts" (hierarchical). Deep dives are always flat.
  "modules": [
    {
      "id": "01-h100",                    // URL segment under the track. Deep dive/course flat: "NN-slug"
      "number": 1,
      "title": "H100: SXM vs PCIe, HBM3, and the Transformer Engine",
      "shortTitle": "H100",               // ≤ 24 chars, sidebar
      "description": "One sentence for cards and search.",
      "readingTime": "40 min",            // from lint-module.mjs (words / 220 wpm, rounded to 5)
      "file": "01-h100.html",             // relative to the manifest directory
      "format": "html",                   // "html" | "md"
      "prerequisites": ["Module 0"],      // optional, free text
      "status": "published",              // site shows the module only when "published".
                                          // skill lifecycle: planned → writing → published | failed
      "scope": "Planner's scope paragraph — what is in, what is out.",   // skill-only
      "topics": ["…", "…"],               // skill-only: key topics the writer must cover
      "sourceIds": ["S1", "S4"],          // skill-only: grounding sources from sources.json
      "estWords": 8000,                   // skill-only: planner estimate
      "words": 7800, "figures": 5, "tables": 3, "citations": 31, "unverified": 1   // from lint --json
    }
  ],
  "parts": [
    {
      "id": "part-01-foundations", "number": 1,
      "title": "Foundations", "shortTitle": "Foundations",
      "description": "One sentence.",
      "modules": [ /* same module object; ids like "module-01" or "01-slug" */ ]
    }
  ],
  "appendices": [                         // optional
    { "id": "bibliography", "letter": "A", "title": "Bibliography", "file": "bibliography.html", "format": "html" }
  ]
}
```

Derived by the site, never stored: `href`, `absPath`, prev/next (flattened reading order), counts.

Rules
- `id`s are stable once published: URLs depend on them.
- Module `file` must exist on disk before the module is set to `published`.
- Keep `estWords`/`scope`/`topics`/`sourceIds` — they are what `/study-continue` needs to resume a writer.
- Validate with `node .claude/study/scripts/validate-manifest.mjs <path>`; the site build fails on an invalid manifest.
