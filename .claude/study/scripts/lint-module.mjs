#!/usr/bin/env node
// lint-module.mjs — deterministic checks for one HTML study module. No dependencies.
// Usage: node lint-module.mjs <file.html> [--json] [--min-words N] [--max-words N]
// Exit 1 on any ERROR. Warnings never fail. --json prints {words, figures, tables, citations, unverified, readingTime, errors, warnings}.
import { readFileSync, existsSync } from 'node:fs';
import { resolve, dirname, join } from 'node:path';

const args = process.argv.slice(2);
const file = args.find(a => !a.startsWith('--'));
if (!file) { console.error('usage: lint-module.mjs <file.html> [--json] [--min-words N] [--max-words N]'); process.exit(2); }
const opt = (name, def) => { const i = args.indexOf(name); return i >= 0 ? args[i + 1] : def; };
const MIN = +opt('--min-words', 3500), MAX = +opt('--max-words', 14000);
const json = args.includes('--json');
const abs = resolve(file);
const src = readFileSync(abs, 'utf8');
const errors = [], warnings = [];
const err = m => errors.push(m), warn = m => warnings.push(m);

// ---- structure ----
const mainOpen = src.match(/<main\b[^>]*\bclass="[^"]*\bstudy\b[^"]*"[^>]*>/i);
if (!mainOpen) err('missing <main class="study">');
if ((src.match(/<main\b/gi) || []).length > 1) err('more than one <main>');
const mainHtml = mainOpen ? src.slice(mainOpen.index + mainOpen[0].length, src.lastIndexOf('</main>')) : src;
const h1s = mainHtml.match(/<h1\b/gi) || [];
if (h1s.length !== 1) err(`expected exactly one <h1>, found ${h1s.length}`);
if (!/<header\b[^>]*class="[^"]*study-header/i.test(mainHtml)) err('missing <header class="study-header">');
if (!/<p class="kicker">/i.test(mainHtml)) warn('no <p class="kicker"> in header');
if (!/<p class="lede">/i.test(mainHtml)) warn('no <p class="lede"> in header');
if (!/<div class="meta">/i.test(mainHtml)) warn('no <div class="meta"> chips in header');

// headings & ids
const headings = [...mainHtml.matchAll(/<(h2|h3)\b([^>]*)>([\s\S]*?)<\/\1>/gi)];
const ids = new Map();
for (const [, tag, attrs, inner] of headings) {
  const id = (attrs.match(/\bid="([^"]+)"/) || [])[1];
  const text = inner.replace(/<[^>]+>/g, '').trim();
  if (!id) err(`<${tag}> without id: "${text.slice(0, 60)}"`);
  else if (ids.has(id)) err(`duplicate heading id "${id}"`);
  else ids.set(id, text);
  if (!/^[a-z0-9][a-z0-9-]*$/.test(id || 'x')) err(`heading id "${id}" must be kebab-case`);
  if (/\b(exercise|exercises|quiz|quizzes|self-assessment|deliverable|deliverables|homework|assignment)\b/i.test(text)) err(`forbidden section (no exercises/quizzes/deliverables): "${text}"`);
  if (/^(introduction|overview|conclusion|summary)$/i.test(text)) warn(`generic heading "${text}" — prefer a descriptive heading`);
}
const h2Texts = [...ids.values()];
if (!h2Texts.some(t => /where (the )?(experts|practitioners|sources) disagree|disagree|trade-?offs and disagreements/i.test(t))) err('missing "Where experts disagree" section');
if (!/<section\b[^>]*id="references"/i.test(mainHtml)) err('missing <section id="references">');
if (!/<ol class="refs">/i.test(mainHtml)) err('missing <ol class="refs"> inside references');
// all ids in main (for anchors)
const allIds = [...mainHtml.matchAll(/\bid="([^"]+)"/g)].map(m => m[1]);
const dup = allIds.filter((v, i, a) => a.indexOf(v) !== i);
for (const d of new Set(dup)) if (!ids.has(d)) err(`duplicate id "${d}"`);

// ---- citations ----
const refIds = new Set([...mainHtml.matchAll(/<li\b[^>]*\bid="(ref-\d+)"/g)].map(m => m[1]));
const cites = [...mainHtml.matchAll(/<a\b[^>]*class="cite"[^>]*href="#([^"]+)"[^>]*>([\s\S]*?)<\/a>/gi)];
const citedSet = new Set();
for (const [, target, label] of cites) {
  citedSet.add(target);
  if (!refIds.has(target)) err(`citation points to missing reference #${target}`);
  const n = target.replace('ref-', '');
  if (label.replace(/<[^>]+>/g, '').trim() !== n) warn(`citation label "${label.trim()}" != reference number ${n}`);
}
const refItems = [...mainHtml.matchAll(/<li\b[^>]*\bid="(ref-\d+)"[^>]*>([\s\S]*?)<\/li>/gi)];
let refN = 0;
for (const [, id, body] of refItems) {
  refN++;
  if (+id.replace('ref-', '') !== refN) warn(`reference ${id} is out of order (position ${refN})`);
  if (!/<a\b[^>]*href="https?:\/\/[^"]+"/i.test(body)) err(`reference ${id} has no http(s) link`);
  if (!/<em>/i.test(body)) warn(`reference ${id} has no <em>title</em>`);
  if (!/ref-use/.test(body)) warn(`reference ${id} has no "Used for:" note (<span class="ref-use">)`);
  if (!/accessed \d{4}-\d{2}-\d{2}/i.test(body)) warn(`reference ${id} has no "Accessed YYYY-MM-DD"`);
  if (!citedSet.has(id)) warn(`reference ${id} is never cited`);
}
if (cites.length === 0) err('no citations at all (<a class="cite" href="#ref-n">n</a>)');
const cited = citedSet.size;

// ---- media ----
for (const m of mainHtml.matchAll(/<img\b[^>]*>/gi)) {
  if (!/src="data:/i.test(m[0])) err(`<img> with external/relative src — visuals must be inline SVG: ${m[0].slice(0, 80)}`);
}
const svgs = [...mainHtml.matchAll(/<svg\b([^>]*)>/gi)];
for (const [, attrs] of svgs) if (!/\bviewBox=/.test(attrs)) err('<svg> without viewBox');
const figures = (mainHtml.match(/<figure class="fig/gi) || []).length;
// figures inside a .stepper are captioned by their <ol class="steps">; every other figure needs a figcaption
const noStepper = mainHtml.replace(/<div class="stepper">[\s\S]*?<ol class="steps">/gi, '');
const figNeedCap = (noStepper.match(/<figure class="fig/gi) || []).length;
const captions = (noStepper.match(/<figcaption>/gi) || []).length;
if (captions < figNeedCap) err(`${figNeedCap - captions} figure(s) without <figcaption>`);
for (const m of mainHtml.matchAll(/<figcaption>([\s\S]*?)<\/figcaption>/gi)) if (!/<b>Figure \d+\./.test(m[1])) warn('figcaption should start with <b>Figure N.</b>');
// marker id collisions across svgs
const markerIds = [...mainHtml.matchAll(/<marker\b[^>]*\bid="([^"]+)"/g)].map(m => m[1]);
for (const d of new Set(markerIds.filter((v, i, a) => a.indexOf(v) !== i))) err(`duplicate <marker id="${d}"> across figures — prefix marker ids per figure (e.g. f3-arrow)`);
if (/<svg[\s\S]*?(fill|stroke)="#[0-9a-fA-F]{3,8}"[\s\S]*?<\/svg>/.test(mainHtml) || /<svg[\s\S]*?style="[^"]*(fill|stroke):\s*#[0-9a-fA-F]{3,8}/.test(mainHtml)) warn('hard-coded hex colour inside <svg> — use d-* classes / CSS variables so dark mode works');
// right-anchored SVG labels that would run past the left edge of their viewBox (≈6.3px per char at 11–13px)
for (const svg of mainHtml.matchAll(/<svg\b([^>]*)>([\s\S]*?)<\/svg>/gi)) {
  const minX = parseFloat((svg[1].match(/viewBox="(-?[\d.]+)/) || [, '0'])[1]);
  for (const t of svg[2].matchAll(/<text\b([^>]*)text-anchor="end"([^>]*)>([^<]*)<\/text>/gi)) {
    const x = parseFloat(((t[1] + t[2]).match(/\bx="(-?[\d.]+)"/) || [, '0'])[1]);
    const w = t[3].replace(/&[a-z#0-9]+;/g, 'x').length * 6.3;
    if (x - w < minX - 2) { warn(`SVG label likely clipped at the left edge (x=${x}, ~${Math.round(w)}px wide): "${t[3].slice(0, 40)}" — widen the viewBox (e.g. viewBox="-60 0 860 H") or move the label`); break; }
  }
}
const tables = (mainHtml.match(/<table class="tbl/gi) || []).length;
for (const m of mainHtml.matchAll(/<table\b[^>]*>([\s\S]*?)<\/table>/gi)) {
  if (!/<caption>/.test(m[1])) warn('table without <caption><b>Table N.</b> …</caption>');
  const cols = (m[1].match(/<th\b/gi) || []).length;
  if (cols > 9) warn(`table with ${cols} header cells — will need horizontal scrolling`);
}
if (figures < 3) warn(`only ${figures} figure(s); modules should carry ≥ 3 diagrams`);

// ---- scripts / assets ----
for (const m of mainHtml.matchAll(/<script\b([^>]*)>/gi)) {
  const a = m[1];
  if (/\bsrc=/.test(a)) { if (!/study\/vendor\/|study\.js/.test(a)) err(`external <script src> not allowed inside module: ${a.trim()}`); }
  else if (!/data-study/.test(a)) err('inline <script> must carry data-study attribute');
}
if (/<link\b[^>]*rel="stylesheet"[^>]*>/i.test(mainHtml)) warn('<link rel="stylesheet"> inside <main> — belongs in <head>');
// head links (standalone viewing)
const head = src.slice(0, mainOpen ? mainOpen.index : 4000);
const cssHref = (head.match(/<link\b[^>]*href="([^"]*study\.css)"/i) || [])[1];
const jsSrc = (head.match(/<script\b[^>]*src="([^"]*study\.js)"/i) || [])[1];
if (!cssHref) err('<head> must link study.css (relative path) for standalone viewing');
if (!jsSrc) err('<head> must load study.js (relative path) for standalone viewing');
for (const rel of [cssHref, jsSrc]) if (rel && !/^https?:/.test(rel) && !existsSync(join(dirname(abs), rel))) err(`asset path does not resolve from this file: ${rel}`);
if (!/<html\b[^>]*class="[^"]*study-standalone/i.test(head)) err('<html> must have class="study-standalone"');
if (!/<title>/.test(head)) err('missing <title>');
if (!/<div class="study-shell">/.test(src)) warn('no <div class="study-shell"> wrapper (standalone TOC needs it; study.js will create one)');
// single-dollar math risk
const proseNoCode = mainHtml.replace(/<pre[\s\S]*?<\/pre>/gi, '').replace(/<code[\s\S]*?<\/code>/gi, '').replace(/\$\$[\s\S]*?\$\$/g, '');
const singleDollar = proseNoCode.match(/(?<!\$)\$(?!\$)[^\s$][^$]{0,80}?\$(?!\$)/g);
if (singleDollar) warn(`possible single-$ math (${singleDollar.length}×) — use \\( … \\) for inline math; prices are fine`);

// ---- text metrics ----
const textOnly = mainHtml
  .replace(/<section\b[^>]*id="references"[\s\S]*?<\/section>/i, '')
  .replace(/<svg[\s\S]*?<\/svg>/gi, ' ')
  .replace(/<pre[\s\S]*?<\/pre>/gi, ' ')
  .replace(/<script[\s\S]*?<\/script>/gi, ' ')
  .replace(/<[^>]+>/g, ' ')
  .replace(/&[a-z#0-9]+;/g, ' ');
const words = (textOnly.match(/[A-Za-z0-9][A-Za-z0-9'’\-\.\/%]*/g) || []).length;
if (words < MIN) err(`word count ${words} < minimum ${MIN}`);
if (words > MAX) err(`word count ${words} > maximum ${MAX}`);
const unverified = (mainHtml.match(/<mark class="unverified"/gi) || []).length;
if (unverified > 8) warn(`${unverified} unverified claims — consider cutting the weakest`);
const readingMin = Math.max(5, Math.round((words / 220 + figures * 0.5 + tables * 0.5) / 5) * 5);
const readingTime = `${readingMin} min`;
const forbidden = [/in this (section|module|chapter),? we (will|shall)/i, /let'?s (dive|take a look|explore)/i, /it is important to note that/i, /in conclusion/i, /as we (have )?seen/i];
for (const re of forbidden) { const m = textOnly.match(re); if (m) warn(`filler phrase: "${m[0]}"`); }

const out = { file: abs, words, figures, tables, citations: cited, references: refN, unverified, readingTime, errors, warnings };
if (json) console.log(JSON.stringify(out, null, 2));
else {
  console.log(`${abs}\n  words=${words} figures=${figures} tables=${tables} citations=${cited}/${refN} unverified=${unverified} readingTime="${readingTime}"`);
  for (const w of warnings) console.log(`  WARN  ${w}`);
  for (const e of errors) console.log(`  ERROR ${e}`);
  console.log(errors.length ? `  ✗ ${errors.length} error(s)` : '  ✓ ok');
}
process.exit(errors.length ? 1 : 0);
