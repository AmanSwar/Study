# Diagram guide — inline SVG that shows the mechanism

Aman is a visual learner. A module's figures are not decoration; they are where the mechanism becomes obvious. Draw what prose cannot say: spatial structure, dataflow, sequence, and quantity.

## When a diagram earns its place
| Draw it | Because |
|---|---|
| Memory/cache/register hierarchy, die layout, chip topology, rack/interconnect topology | Spatial structure with capacities and bandwidths on the edges |
| Pipelines, dataflow, producer/consumer, request lifecycle, training loop | Flow with stages, buffers, and the arrows that carry the bytes |
| Anything that happens in stages (use a **stepper**) | The reader sees state change one step at a time |
| Timelines of generations/versions | History with dates and the one number that changed |
| Comparisons of measured numbers (bar/line/scatter) | A plot of *cited* data beats adjectives |
| Tiling, layouts, blocking, striding, swizzling, matrix partitioning | Index arithmetic is unreadable in prose |
| State machines, protocols, barriers/phases | Transitions and who waits on whom |
| Roofline / crossover / break-even curves | The shape of the trade-off |

Do **not** draw: lists of features (table), a single number (KPI), org charts, or generic "AI brain" art. If a table says it better, use the table.

## Mechanics (every figure)
```html
<figure class="fig">
  <svg viewBox="0 0 800 320" role="img" aria-labelledby="f2t"><title id="f2t">One-line description</title>
    <defs><marker id="f2-arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" class="d-marker"/></marker></defs>
    …
  </svg>
  <figcaption><b>Figure 2.</b> What to see. Data: source<a class="cite" href="#ref-1">1</a>.</figcaption>
</figure>
```
- Canvas: `viewBox="0 0 800 H"` (H 200–500). Width scales to the column; design at 800.
- **Every id inside an SVG is prefixed by the figure number** (`f2-arr`, `f2-grad`) — all figures share one document; duplicate marker ids silently break arrows.
- Text: ≥ 12px at the 800 canvas (`class="d-small"` = 11px is the floor for axis ticks). Labels ≤ 4 words; put explanation in the caption. `text-anchor="middle"` for centred labels. Use `class="d-mono"` for numbers/identifiers.
- Colours only via classes (theme-safe). Never `fill="#…"`/`stroke="#…"`/inline colour styles. `style="stroke-width:1.5"` is fine.
- Keep < ~200 elements; split otherwise. No external fonts/images. Use `<title>` for accessibility.
- Numbers on the diagram must match the text and carry a citation in the caption.

## Palette classes (from study.css)
- Blocks: `d-block` (primary blue), `d-block-2` (cyan), `d-block-accent` (orange), `d-block-muted` (grey), `d-hl` (amber highlight), `d-surface` (card), `d-panel` (recessed area), `d-outline` (no fill, stroked), add `d-dashed`.
- Text: default is themed; `d-text-light` (secondary), `d-text-inv` (white on coloured blocks), `d-bold`, `d-title` (14px), `d-small`, `d-mono`.
- Lines: `d-arrow` (+ `marker-end="url(#fN-arr)"`), `d-arrow-active` (orange, emphasised), `d-line` (plot lines; combine with `d-stroke-*`), `d-axis`, `d-grid`, `d-dashed`.
- Semantic colours: `d-fill-blue|cyan|orange|green|amber|red`, `d-fill-*-soft` (tinted backgrounds), `d-stroke-*`. `d-dim` for de-emphasis. `d-label-bg` for a legend box. `d-marker` / `d-marker-active` for arrowheads.

## Recipes (coordinates on the 800-wide canvas)

**Block diagram / pipeline (horizontal).** Columns at x = 20, 230, 440, 650; blocks 150–170 wide, 60 tall, `rx="8"`; arrows between at y = block-centre; bandwidth labels as `d-mono d-small` above arrows.
```html
<rect x="20" y="80" width="160" height="60" rx="8" class="d-block"/><text x="100" y="115" text-anchor="middle" class="d-text-inv d-bold">TMA load</text>
<path d="M180,110 L230,110" class="d-arrow" marker-end="url(#f2-arr)"/><text x="205" y="100" text-anchor="middle" class="d-mono d-small">128 B/clk</text>
```

**Hierarchy (vertical stack inside a panel).** Panel `d-panel` 200×240; rows 40 tall with 15px gaps; capacity in the label, bandwidth on the arrow to the next level; annotate "per SM"/"per die" with `d-text-light d-small`.

**Bar chart.** Axes: `<line class="d-axis">` at x=70 (y-axis) and y=220 (x-axis); gridlines `d-grid` every 50px; bars `rect` with `d-fill-blue` (highlight the one that matters with `d-fill-orange`); value labels above bars in `d-mono d-small`; axis titles in `d-text-light d-small`; state the unit in the y-axis title.

**Line plot.** `<polyline points="x1,y1 x2,y2 …" class="d-line d-stroke-blue"/>` per series; legend box `d-label-bg` top-right with a 24px sample line and label; log axes labelled as such (`batch size (log)`).

**Timeline (horizontal).** Baseline at y=120; ticks every 150px; year in `d-mono d-small` below, event above; a `d-fill-blue` circle r=5 at each tick; highlight the current generation with `d-hl`.

**Matrix / tiling.** Grid of `rect` cells 40×40 with `d-outline`; shade a tile with `d-fill-blue-soft` and stroke it `d-stroke-blue`; annotate dimensions with bracketed `d-mono` labels (`M = 128`); show thread/warp ownership with `d-fill-*-soft` bands.

**Topology.** Nodes as `rect rx="10"` in a ring/mesh; links as `d-arrow` without markers (bidirectional) with bandwidth labels; group with a dashed `d-outline d-dashed` rectangle labelled at its top-left.

**State machine.** States as `rect rx="20"`; transitions as curved paths `M… Q… …` with markers; guard labels in `d-mono d-small`.

**Stepper.** Wrap elements per stage in `<g data-step="n">`; keep static scaffolding outside any `data-step`. One `<li>` per step in `ol.steps`. 3–6 steps.

**Heat / roofline.** Roofline: log-log axes; memory-bound slope as a `d-line d-stroke-cyan` from origin, compute ceiling as `d-line d-stroke-blue` horizontal; mark the ridge with a `d-fill-orange` circle and label its arithmetic intensity; place the workload as a labelled point.

## Checklist per figure
1. Would a table be clearer? If yes, table.
2. Ids prefixed (`fN-…`)? viewBox present? No hex colours?
3. Text readable at 800 wide (≥ 12px, ≤ 4 words per label)?
4. Units on axes/arrows; numbers match text; caption cites the data source.
5. Figure numbered in order; caption says what to look at.
