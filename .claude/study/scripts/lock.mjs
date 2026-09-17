// lock.mjs — tiny cross-process mutex for read-modify-write on a JSON file.
// mkdir is atomic on POSIX, so a lock directory next to the file is the lock.
// Parallel study-writer agents all call manifest.mjs on the same manifest.json;
// without this, two writers finishing together could clobber each other's update.
import { mkdirSync, rmSync, statSync } from 'node:fs';

const STALE_MS = 120_000;
export function withLock(file, fn, { timeoutMs = 30_000 } = {}) {
  const dir = file + '.lock';
  const deadline = Date.now() + timeoutMs;
  for (;;) {
    try { mkdirSync(dir); break; } catch (e) {
      if (e.code !== 'EEXIST') throw e;
      try { if (Date.now() - statSync(dir).mtimeMs > STALE_MS) { rmSync(dir, { recursive: true, force: true }); continue; } } catch {}
      if (Date.now() > deadline) throw new Error(`lock timeout on ${dir} (remove it if no manifest.mjs is running)`);
      Atomics.wait(new Int32Array(new SharedArrayBuffer(4)), 0, 0, 50 + Math.floor(Math.random() * 100));
    }
  }
  const release = () => { try { rmSync(dir, { recursive: true, force: true }); } catch {} };
  process.once('exit', release);
  try { return fn(); } finally { release(); }
}
