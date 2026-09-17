#!/usr/bin/env node
// build-bibliography.mjs — merge every module's <ol class="refs"> into bibliography.html and register it
// as an appendix in the manifest. Dedupes by URL. Idempotent.
// Usage: node build-bibliography.mjs <manifest.json>
import { readFileSync, writeFileSync, existsSync } from 'node:fs';
import { resolve, dirname, join, relative } from 'node:path';
import { withLock } from './lock.mjs';

const file = process.argv[2];
if (!file) { console.error('usage: build-bibliography.mjs <manifest.json>'); process.exit(2); }
const abs = resolve(file), dir = dirname(abs);
const m = JSON.parse(readFileSync(abs, 'utf8'));
const modules = (m.parts ? m.parts.flatMap(p => p.modules) : m.modules).filter(x => x.format === 'html' && existsSync(join(dir, x.file)));
const byUrl = new Map();
for (const mod of modules) {
  const html = readFileSync(join(dir, mod.file), 'utf8');
  for (const li of html.matchAll(/<li\b[^>]*\bid="ref-\d+"[^>]*>([\s\S]*?)<\/li>/gi)) {
    let body = li[1].replace(/<span class="ref-use">[\s\S]*?<\/span>/i, '').trim();
    const url = (body.match(/href="(https?:\/\/[^"]+)"/i) || [])[1] || body.replace(/<[^>]+>/g, '').slice(0, 80);
    const key = url.replace(/[#?].*$/, '').replace(/\/$/, '').toLowerCase();
    if (!byUrl.has(key)) byUrl.set(key, { body, used: [] });
    byUrl.get(key).used.push(`${m.unitLabel || 'Module'} ${mod.number}`);
  }
}
const typeRank = { official: 0, standard: 1, paper: 2, textbook: 3, repo: 4, talk: 5, blog: 6, analysis: 7 };
const entries = [...byUrl.values()].sort((a, b) => {
  const ta = (a.body.match(/ref-type">([a-z]+)</) || [])[1], tb = (b.body.match(/ref-type">([a-z]+)</) || [])[1];
  return (typeRank[ta] ?? 9) - (typeRank[tb] ?? 9) || a.body.replace(/<[^>]+>/g, '').localeCompare(b.body.replace(/<[^>]+>/g, ''));
});
// locate study.css relative to this directory (same base the modules use)
const first = modules[0] ? readFileSync(join(dir, modules[0].file), 'utf8') : '';
const cssRel = (first.match(/href="([^"]*study\.css)"/) || [])[1] || relative(dir, resolve(dir, '../../website/public/study/study.css'));
const jsRel = cssRel.replace(/study\.css$/, 'study.js');
const items = entries.map((e, i) => `  <li id="ref-${i + 1}">${e.body} <span class="ref-use">Cited in: ${[...new Set(e.used)].join(', ')}.</span></li>`).join('\n');
const out = `<!doctype html>
<html lang="en" class="study-standalone">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Bibliography · ${m.title}</title>
<link rel="stylesheet" href="${cssRel}">
<script defer src="${jsRel}"></script>
</head>
<body><div class="study-shell"><main class="study">
<header class="study-header">
  <p class="kicker">${m.title} · Appendix</p>
  <h1>Bibliography</h1>
  <p class="lede">Every source cited across the ${modules.length} modules, deduplicated (${entries.length} entries), ordered by source type: official documentation and standards first, then papers, textbooks, repositories, talks, blogs, third-party analyses.</p>
  <div class="meta"><span class="chip"><b>${entries.length}</b> sources</span><span class="chip"><b>${modules.length}</b> modules</span><span class="chip chip-amber">Generated ${new Date().toISOString().slice(0, 10)}</span></div>
</header>
<section id="references">
<h2 id="all-sources">All sources</h2>
<ol class="refs">
${items}
</ol>
</section>
</main></div></body></html>
`;
writeFileSync(join(dir, 'bibliography.html'), out);
const letter = withLock(abs, () => {
  const cur = JSON.parse(readFileSync(abs, 'utf8'));
  cur.appendices = (cur.appendices || []).filter(a => a.id !== 'bibliography');
  const l = String.fromCharCode(65 + cur.appendices.length);
  cur.appendices.push({ id: 'bibliography', letter: l, title: 'Bibliography', file: 'bibliography.html', format: 'html' });
  writeFileSync(abs, JSON.stringify(cur, null, 2) + '\n');
  return l;
});
console.log(`bibliography.html: ${entries.length} unique sources from ${modules.length} modules → appendix ${letter}`);
