#!/usr/bin/env node
// manifest.mjs — safe edits to a track manifest so agents never hand-edit JSON.
//
//   init       <dir> --id X --title T --shortTitle S --description D --category C --kind K --color c --icon I --skill sk --request "…" [--unitLabel U]
//   apply-plan <manifest.json> <plan.json>          plan.json = { "modules": [...] } or { "parts": [...] } (+ optional "appendices")
//   set-module <manifest.json> <moduleId> [--status s] [--from-lint lint.json] [key=value ...]
//   set-track  <manifest.json> [--status s] [--stage st] [--approved true|false] [key=value ...]
//   status     <manifest.json>                        prints stage + per-module status table (used by /study-continue)
//   next       <manifest.json> [--limit N]            prints ids of modules with status != published (JSON array)
import { readFileSync, writeFileSync, mkdirSync, existsSync } from 'node:fs';
import { resolve, join } from 'node:path';

const [cmd, ...rest] = process.argv.slice(2);
const flags = {}, pos = [], kv = {};
for (let i = 0; i < rest.length; i++) {
  const a = rest[i];
  if (a.startsWith('--')) { const k = a.slice(2); const v = rest[i + 1]; if (v === undefined || v.startsWith('--')) flags[k] = true; else { flags[k] = v; i++; } }
  else if (/^[a-zA-Z]\w*=/.test(a)) { const [k, ...v] = a.split('='); kv[k] = coerce(v.join('=')); }
  else pos.push(a);
}
function coerce(v) { if (v === 'true') return true; if (v === 'false') return false; if (v !== '' && !isNaN(+v)) return +v; try { if (/^[\[{]/.test(v)) return JSON.parse(v); } catch {} return v; }
const now = () => new Date().toISOString();
const load = p => JSON.parse(readFileSync(resolve(p), 'utf8'));
const save = (p, m) => { m.generator = m.generator || {}; m.generator.updatedAt = now(); writeFileSync(resolve(p), JSON.stringify(m, null, 2) + '\n'); };
const allModules = m => (m.parts ? m.parts.flatMap(p => p.modules || []) : m.modules || []);
const die = s => { console.error(s); process.exit(1); };

if (cmd === 'init') {
  const dir = resolve(pos[0] || die('init needs <dir>'));
  for (const k of ['id', 'title', 'shortTitle', 'description', 'category', 'kind', 'color', 'icon', 'skill', 'request']) if (!flags[k]) die(`init: missing --${k}`);
  mkdirSync(dir, { recursive: true }); mkdirSync(join(dir, '.cache'), { recursive: true });
  const m = { version: 1, id: flags.id, title: flags.title, shortTitle: flags.shortTitle, description: flags.description, category: flags.category, kind: flags.kind,
    color: flags.color, icon: flags.icon, unitLabel: flags.unitLabel || (flags.kind === 'deep-dive' ? 'Page' : 'Module'), status: 'researching', sources: 'sources.md',
    generator: { skill: flags.skill, request: flags.request, createdAt: now(), updatedAt: now(), profile: 'learner-profile.md', pipeline: { stage: 'research', approved: false } },
    modules: [] };
  const p = join(dir, 'manifest.json');
  if (existsSync(p)) die(`init: ${p} already exists (use /study-continue)`);
  writeFileSync(p, JSON.stringify(m, null, 2) + '\n');
  console.log(p);
} else if (cmd === 'apply-plan') {
  const [mp, pp] = pos; if (!mp || !pp) die('apply-plan <manifest.json> <plan.json>');
  const m = load(mp), plan = load(pp);
  if (!!plan.modules === !!plan.parts) die('plan must have exactly one of modules | parts');
  if (m.kind === 'deep-dive' && plan.parts) die('deep dives are flat: use modules');
  const norm = mod => ({ status: 'planned', format: 'html', ...mod, file: mod.file || `${mod.id}.html` });
  if (plan.parts) { delete m.modules; m.parts = plan.parts.map(p => ({ ...p, modules: (p.modules || []).map(norm) })); }
  else { delete m.parts; m.modules = plan.modules.map(norm); }
  if (plan.appendices) m.appendices = plan.appendices;
  m.status = 'planned'; m.generator.pipeline = { ...(m.generator.pipeline || {}), stage: 'checkpoint', approved: false };
  save(mp, m);
  const mods = allModules(m);
  console.log(`applied: ${mods.length} modules${m.parts ? ` in ${m.parts.length} parts` : ''}; est ${mods.reduce((s, x) => s + (x.estWords || 0), 0)} words`);
} else if (cmd === 'set-module') {
  const [mp, id] = pos; if (!mp || !id) die('set-module <manifest.json> <moduleId> …');
  const m = load(mp); const mod = allModules(m).find(x => x.id === id) || die(`no module "${id}"`);
  if (flags['from-lint']) { const l = load(flags['from-lint']); Object.assign(mod, { words: l.words, figures: l.figures, tables: l.tables, citations: l.citations, unverified: l.unverified, readingTime: l.readingTime }); }
  if (flags.status) mod.status = flags.status;
  Object.assign(mod, kv);
  if (mod.status === 'published' && !existsSync(join(resolve(mp), '..', mod.file))) die(`cannot publish: file missing ${mod.file}`);
  save(mp, m); console.log(`${id}: status=${mod.status} words=${mod.words ?? '-'} readingTime=${mod.readingTime ?? '-'}`);
} else if (cmd === 'set-track') {
  const [mp] = pos; if (!mp) die('set-track <manifest.json> …');
  const m = load(mp);
  if (flags.status) m.status = flags.status;
  m.generator.pipeline = m.generator.pipeline || {};
  if (flags.stage) m.generator.pipeline.stage = flags.stage;
  if (flags.approved !== undefined) { m.generator.pipeline.approved = flags.approved === true || flags.approved === 'true'; if (m.generator.pipeline.approved) m.generator.pipeline.approvedAt = now(); }
  Object.assign(m, kv);
  save(mp, m); console.log(`track ${m.id}: status=${m.status} stage=${m.generator.pipeline.stage} approved=${m.generator.pipeline.approved}`);
} else if (cmd === 'status') {
  const m = load(pos[0] || die('status <manifest.json>'));
  const mods = allModules(m);
  console.log(`${m.id} · ${m.kind} · status=${m.status} · stage=${m.generator?.pipeline?.stage} · approved=${m.generator?.pipeline?.approved}`);
  console.log(`sources.json: ${existsSync(join(resolve(pos[0]), '..', 'sources.json')) ? 'present' : 'MISSING'} · modules: ${mods.length} · published: ${mods.filter(x => x.status === 'published').length}`);
  for (const x of mods) console.log(`  ${String(x.number).padStart(2)}  ${(x.status || 'planned').padEnd(9)} ${x.id.padEnd(28)} ${x.words ? x.words + 'w' : (x.estWords ? '~' + x.estWords + 'w' : '')}  ${x.title}`);
} else if (cmd === 'next') {
  const m = load(pos[0] || die('next <manifest.json>'));
  const lim = +(flags.limit || 1000);
  console.log(JSON.stringify(allModules(m).filter(x => x.status !== 'published').slice(0, lim).map(x => x.id)));
} else die('usage: manifest.mjs init|apply-plan|set-module|set-track|status|next …');
