#!/usr/bin/env node
// validate-manifest.mjs — schema + filesystem checks for a track manifest. No dependencies.
// Usage: node validate-manifest.mjs <manifest.json> [--strict]   (--strict: published modules must have metrics)
import { readFileSync, existsSync } from 'node:fs';
import { resolve, dirname, join } from 'node:path';

const args = process.argv.slice(2);
const file = args.find(a => !a.startsWith('--'));
if (!file) { console.error('usage: validate-manifest.mjs <manifest.json> [--strict]'); process.exit(2); }
const strict = args.includes('--strict');
const abs = resolve(file), dir = dirname(abs);
let m;
try { m = JSON.parse(readFileSync(abs, 'utf8')); } catch (e) { console.log(`ERROR cannot parse ${abs}: ${e.message}`); process.exit(1); }
const errors = [], warnings = [];
const err = s => errors.push(s), warn = s => warnings.push(s);
const ID = /^[a-z0-9][a-z0-9-]*$/;
const RESERVED = new Set(['study', 'search-index.json', 'favicon.ico', 'api', '_next']);
const COLORS = new Set(['blue', 'cyan', 'orange', 'green', 'amber', 'red', 'purple']);
const CATS = new Set(['Computer Science', 'Finance', 'Mathematics', 'Business']);
const TRACK_STATUS = new Set(['researching', 'planned', 'writing', 'published']);
const MOD_STATUS = new Set(['planned', 'writing', 'published', 'failed']);

if (m.version !== 1) err('version must be 1');
for (const k of ['id', 'title', 'shortTitle', 'description', 'category', 'kind', 'color', 'icon']) if (typeof m[k] !== 'string' || !m[k]) err(`missing string field "${k}"`);
if (m.id && !ID.test(m.id)) err(`id "${m.id}" must match ${ID}`);
if (RESERVED.has(m.id)) err(`id "${m.id}" is reserved`);
if (m.shortTitle && m.shortTitle.length > 18) warn(`shortTitle longer than 18 chars ("${m.shortTitle}")`);
if (!CATS.has(m.category)) err(`category must be one of ${[...CATS].join(' | ')}`);
if (!['deep-dive', 'course'].includes(m.kind)) err('kind must be "deep-dive" | "course"');
if (!COLORS.has(m.color)) err(`color must be one of ${[...COLORS].join(' | ')}`);
if (m.unitLabel && !['Module', 'Chapter', 'Page'].includes(m.unitLabel)) err('unitLabel must be Module | Chapter | Page');
if (m.status && !TRACK_STATUS.has(m.status)) err(`status "${m.status}" invalid`);
if (m.sources && !existsSync(join(dir, m.sources))) err(`sources file missing: ${m.sources}`);
const hasModules = Array.isArray(m.modules), hasParts = Array.isArray(m.parts);
if (hasModules === hasParts) err('exactly one of "modules" or "parts" must be present');
if (m.kind === 'deep-dive' && hasParts) err('deep dives must be flat ("modules"), not "parts"');

const seen = new Set();
function checkModule(mod, where) {
  for (const k of ['id', 'title', 'shortTitle', 'description', 'file', 'format']) if (typeof mod[k] !== 'string' || !mod[k]) err(`${where}: missing string field "${k}"`);
  if (typeof mod.number !== 'number') err(`${where}: number must be numeric`);
  if (mod.id && !ID.test(mod.id)) err(`${where}: id "${mod.id}" must match ${ID}`);
  if (seen.has(mod.id)) err(`${where}: duplicate module id "${mod.id}"`); seen.add(mod.id);
  if (!['html', 'md'].includes(mod.format)) err(`${where}: format must be html | md`);
  if (mod.status && !MOD_STATUS.has(mod.status)) err(`${where}: status "${mod.status}" invalid`);
  const published = (mod.status ?? 'published') === 'published';
  if (mod.file && published && !existsSync(join(dir, mod.file))) err(`${where}: file missing: ${mod.file}`);
  if (published && !mod.readingTime) warn(`${where}: no readingTime`);
  if (strict && published && mod.format === 'html' && typeof mod.words !== 'number') err(`${where}: published html module has no metrics (run lint-module.mjs --json)`);
  if (mod.shortTitle && mod.shortTitle.length > 24) warn(`${where}: shortTitle longer than 24 chars`);
}
if (hasModules) { m.modules.forEach((mod, i) => checkModule(mod, `modules[${i}]`)); const nums = m.modules.map(x => x.number); nums.forEach((n, i) => { if (n !== i + 1) warn(`modules[${i}].number is ${n}, expected ${i + 1}`); }); }
if (hasParts) {
  const pseen = new Set(); let n = 0;
  m.parts.forEach((p, i) => {
    for (const k of ['id', 'title', 'shortTitle']) if (typeof p[k] !== 'string' || !p[k]) err(`parts[${i}]: missing "${k}"`);
    if (pseen.has(p.id)) err(`duplicate part id "${p.id}"`); pseen.add(p.id);
    if (!Array.isArray(p.modules) || !p.modules.length) err(`parts[${i}] has no modules`);
    (p.modules || []).forEach((mod, j) => { checkModule(mod, `parts[${i}].modules[${j}]`); n++; if (mod.number !== n) warn(`parts[${i}].modules[${j}].number is ${mod.number}, expected ${n} (numbering is continuous across parts)`); });
  });
}
(m.appendices || []).forEach((a, i) => {
  for (const k of ['id', 'letter', 'title', 'file']) if (typeof a[k] !== 'string' || !a[k]) err(`appendices[${i}]: missing "${k}"`);
  if (a.file && !existsSync(join(dir, a.file))) err(`appendices[${i}]: file missing: ${a.file}`);
  if (a.format && !['html', 'md'].includes(a.format)) err(`appendices[${i}]: format must be html | md`);
});
const all = hasParts ? m.parts.flatMap(p => p.modules || []) : (m.modules || []);
const pub = all.filter(x => (x.status ?? 'published') === 'published').length;
console.log(`${abs}\n  id=${m.id} kind=${m.kind} status=${m.status ?? 'published'} modules=${all.length} published=${pub}`);
for (const w of warnings) console.log(`  WARN  ${w}`);
for (const e of errors) console.log(`  ERROR ${e}`);
console.log(errors.length ? `  ✗ ${errors.length} error(s)` : '  ✓ ok');
process.exit(errors.length ? 1 : 0);
