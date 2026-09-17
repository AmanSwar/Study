# Component catalogue — exact markup

Everything here is rendered by `website/public/study/study.css` + `study.js`. Copy the markup exactly; the runtime (`Study.init`) adds behaviour (code headers, copy buttons, tab switching, steppers, calculators, citation popovers, zoom, math). A live gallery of every component: `website/public/study/demo.html` (open it in a browser, or `/study/demo.html` on the dev server).

Rules that apply everywhere
- Only these classes; no inline styles except `style="stroke-width:…"` on SVG paths when needed.
- No `<img>`. All visuals are inline `<svg viewBox="…">` (see diagram-guide.md).
- Escape `<`, `>`, `&` inside `<code>`.
- Math: `\( … \)` inline, `$$ … $$` display. Never a single `$` delimiter.
- Ids: kebab-case, unique in the file. Every `h2`/`h3` has one.

## Header
```html
<header class="study-header">
  <p class="kicker">Track title · Module 3</p>
  <h1>Title of the module</h1>
  <p class="lede">Two or three sentences: what this covers and what you can do afterwards.</p>
  <div class="meta">
    <span class="chip chip-blue">Course</span>
    <span class="chip"><b>45 min</b> read</span>
    <span class="chip"><b>14</b> sources</span>
    <span class="chip chip-amber">As of 2026-09-17</span>
    <span class="chip">Prerequisite: <b>Module 1</b></span>
  </div>
</header>
```

## Headings and prose
```html
<h2 id="roofline-model">1. The roofline model for decode</h2>
<h3 id="roofline-model-ridge">1.1 Where the ridge point sits</h3>
<p>Prose with <code>inline code</code>, <strong>emphasis</strong>, and a citation<a class="cite" href="#ref-2">2</a>.</p>
<p class="lede">Larger intro paragraph (header only).</p>
<p class="small muted">Fine print.</p>
<blockquote>Verbatim quote from a source<a class="cite" href="#ref-4">4</a>.</blockquote>
<div class="grid-2"><div>…left…</div><div>…right…</div></div>   <!-- also .grid-3 -->
<span class="badge badge-blue">primary source</span> <span class="badge badge-amber">estimate</span> <span class="badge badge-red">deprecated</span> <span class="badge badge-green">measured</span>
<span class="asof">as of 2026-09</span>
<mark class="unverified" title="No primary source found; figure from S9 (analysis)">≈ 7.7 TB/s measured</mark>
```

## Callouts (5 kinds)
```html
<aside class="callout callout-insight"><span class="callout-title">Key insight</span><p>The one thing to remember.</p></aside>
<aside class="callout callout-note"><span class="callout-title">Note</span><p>Definition or context.</p></aside>
<aside class="callout callout-tip"><span class="callout-title">In practice</span><p>What practitioners actually do.</p></aside>
<aside class="callout callout-warning"><span class="callout-title">Caveat</span><p>Where the simple story breaks.</p></aside>
<aside class="callout callout-critical"><span class="callout-title">Common error</span><p>The mistake that costs the most.</p></aside>
```

## Figure (inline SVG)
```html
<figure class="fig">
  <svg viewBox="0 0 800 300" role="img" aria-labelledby="f3t"><title id="f3t">Short description</title>
    …see diagram-guide.md…
  </svg>
  <figcaption><b>Figure 3.</b> What to look at, and where the numbers come from<a class="cite" href="#ref-1">1</a>.</figcaption>
</figure>
```
`class="fig fig-plain"` removes the card border (for small inline sketches). Figures are click-to-zoom automatically. Number figures manually and in order.

## Code
```html
<pre class="code" data-lang="cuda" data-title="tma_load.cu" data-hl="4-6"><code>…escaped code…</code></pre>
```
`data-lang`: python, cpp, c, cuda, metal, ptx, x86asm, armasm, llvm, bash, rust, go, typescript, javascript, json, yaml, sql, cmake, dockerfile, protobuf, verilog, glsl, julia, haskell, plaintext (and aliases py/ts/js/sh). `data-title` optional. `data-hl="3,7-9"` highlights lines. Add `data-nohl` to skip syntax colouring.

## Tables
```html
<table class="tbl tbl-compare">
  <caption><b>Table 2.</b> What is compared, units, and the source<a class="cite" href="#ref-2">2</a>.</caption>
  <thead><tr><th>Part</th><th class="num">HBM (GB)</th><th class="num">BW (TB/s)</th><th>Interconnect</th></tr></thead>
  <tbody>
    <tr><td>H100 SXM</td><td class="num">80</td><td class="num">3.35</td><td>NVLink 4</td></tr>
    <tr><td>B200 SXM</td><td class="num" data-best>192</td><td class="num" data-best>8.0</td><td>NVLink 5</td></tr>
  </tbody>
</table>
```
`th.num`/`td.num` right-align numerics (units in the header). `data-best` / `data-worst` colour a cell. `tbl-dense` for wide tables (≤ 9 columns; they scroll horizontally). `td.row` for a bold row label. Number tables manually.

## Spec grid and KPIs
```html
<dl class="spec-grid">
  <div class="spec"><dt>Process</dt><dd>TSMC 4NP</dd></div>
  <div class="spec"><dt>SMs</dt><dd>148 <small>of 160 physical</small></dd></div>
</dl>
<div class="kpis">
  <div class="kpi"><b>8 TB/s</b><span>HBM3e bandwidth<a class="cite" href="#ref-1">1</a></span></div>
  <div class="kpi kpi-cyan"><b>1.8 TB/s</b><span>NVLink 5 per GPU</span></div>   <!-- kpi-orange kpi-green kpi-amber kpi-red -->
</div>
```

## Tabs (contrast implementations / options)
```html
<div class="tabs">
  <nav class="tabs-nav"><button data-tab="a" class="is-active">vLLM</button><button data-tab="b">SGLang</button></nav>
  <section class="tabs-panel" data-panel="a"><p>…</p></section>
  <section class="tabs-panel" data-panel="b"><p>…</p></section>
</div>
```

## Stepper (a process in stages)
```html
<div class="stepper">
  <figure class="fig fig-plain">
    <svg viewBox="0 0 800 220" role="img" aria-label="…">
      <g data-step="1">…elements that appear at step 1…</g>
      <g data-step="2">…</g>
      <g data-step="2-4">…visible during steps 2–4…</g>
    </svg>
  </figure>
  <ol class="steps">
    <li><b>Step title.</b> One or two sentences.</li>
    <li>…one <li> per step, same count as the highest data-step…</li>
  </ol>
</div>
```
Prev/Next controls, keyboard ←/→, and click-on-step are added automatically. `data-step="n"` shows from step n on (dimmed once passed); `data-step="a-b"` shows only during a..b.

## Calculator (a formula with knobs)
```html
<div class="calc">
  <p class="calc-title">KV cache per token</p>
  <div class="calc-row"><label for="k1">Layers</label><input type="range" id="k1" name="L" min="16" max="128" step="1" value="61"></div>
  <div class="calc-row"><label for="k2">KV dtype</label><select id="k2" name="b"><option value="2">BF16 (2 B)</option><option value="1">FP8 (1 B)</option></select><span></span></div>
  <div class="calc-row"><label for="k3">Context</label><input type="range" id="k3" name="ctx" min="1024" max="131072" step="1024" value="32768" data-format="si"></div>
  <div class="calc-out">
    <div>Per token<output data-expr="2 * L * 8 * 128 * b" data-format="bytes"></output></div>
    <div>Per request<output data-expr="2 * L * 8 * 128 * b * ctx" data-format="bytes"></output></div>
  </div>
  <p class="calc-note">Formula: \( 2 L H_{kv} d\, b \). State assumptions and cite the constants.</p>
</div>
```
`name` attributes are the variables in `data-expr` (JavaScript expression; `Math.*` available unqualified: `floor`, `log2`, `pow`…). `data-format`: `int`, `fixed:2`, `si`, `bytes` (1024), `bytes10` (1000), `pct`, `raw`, default auto. `data-unit="ms"` appends a unit. Range inputs show their live value. Custom logic: `<script data-study>` inside the module may listen to `calc:update` on the `.calc` element.

## Timeline
```html
<ol class="timeline">
  <li><time>2022</time><b>Hopper (H100)</b><p>Transformer Engine, TMA, wgmma<a class="cite" href="#ref-3">3</a>.</p></li>
</ol>
```

## Where experts disagree
```html
<section class="disagree">
<h2 id="where-experts-disagree">Where experts disagree</h2>
<p>Framing paragraph.</p>
<div class="positions">
  <article class="position">
    <h4>Position title</h4>
    <p class="who">Who holds it<a class="cite" href="#ref-3">3</a></p>
    <p>The argument.</p>
    <ul class="pros"><li>…</li></ul>
    <ul class="cons"><li>…</li></ul>
  </article>
  <!-- 2–4 positions -->
</div>
<div class="verdict"><span class="verdict-title">Assessment</span><p>Which is right under which conditions; what would settle it.</p></div>
</section>
```

## Disclosure (derivations, long code, digressions)
```html
<details class="more"><summary>Derivation of the crossover batch size</summary>
<p>…</p>
</details>
```

## Citations and references
```html
…claim<a class="cite" href="#ref-7">7</a>.          <!-- renders as superscript [7]; hover shows the reference -->

<section id="references">
<h2 id="references-heading">References</h2>
<ol class="refs">
  <li id="ref-1"><span class="ref-type">official</span>NVIDIA. <em>CUDA C++ Programming Guide</em>. v12.6, 2024. <a href="https://docs.nvidia.com/cuda/cuda-c-programming-guide/">https://docs.nvidia.com/cuda/cuda-c-programming-guide/</a>. Accessed 2026-09-17. <span class="ref-use">Used for: TMA descriptor semantics (§7.29), cluster launch limits.</span></li>
  <li id="ref-2"><span class="ref-type">paper</span>Kwon, W. et al. <em>Efficient Memory Management for Large Language Model Serving with PagedAttention</em>. SOSP 2023. <a href="https://arxiv.org/abs/2309.06180">https://arxiv.org/abs/2309.06180</a>. Accessed 2026-09-17. <span class="ref-use">Used for: fragmentation measurements (§3, Fig. 2).</span></li>
</ol>
</section>
```
`ref-type` ∈ official · standard · paper · textbook · repo · talk · blog · analysis.

## Inline script (only when a calculator needs custom logic)
```html
<script data-study>
(function(){ var c=document.currentScript.previousElementSibling; c.addEventListener('calc:update',function(e){ /* e.detail = {name: value} */ }); })();
</script>
```
Must be idempotent and scoped to its own element — the site re-executes it on navigation.
