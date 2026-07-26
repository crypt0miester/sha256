// Engine-portable wave-curve benchmark: simd128-4 vs 2x4 vs 4x4.
// Runs unmodified under Node/Bun and bare engine shells (SpiderMonkey, d8).
// Shells have no node:crypto, so this file only compares tape kernels
// against each other; check.mjs is the correctness gate, run under Node.
//
//   node matrix.cjs [path/to/wasm_lib.wasm]
//   bun  matrix.cjs [path]
//   ~/.jsvu/bin/spidermonkey matrix.cjs [path]
//
// Trust ratios and win-direction across repeated runs, not absolutes.
'use strict';

const IS_PROC = typeof process !== 'undefined' && !!process.versions;
const ENGINE = IS_PROC
  ? (process.versions.bun ? `bun ${process.versions.bun} (JavaScriptCore)`
    : `node ${process.versions.node} (V8 ${process.versions.v8})`)
  : (typeof version === 'function' ? `shell ${version()}` : 'unknown shell');

const args = IS_PROC ? process.argv.slice(2)
  : (typeof scriptArgs !== 'undefined' ? scriptArgs
    : (typeof arguments !== 'undefined' ? Array.from(arguments) : []));
const wasmPath = args[0] ?? 'wasm_lib.wasm';

function loadBytes(path) {
  if (IS_PROC) return require('fs').readFileSync(path);
  if (typeof readbuffer === 'function') return new Uint8Array(readbuffer(path)); // d8
  if (typeof os !== 'undefined' && os.file && os.file.readFile) {
    return os.file.readFile(path, 'binary'); // SpiderMonkey
  }
  if (typeof read === 'function') return read(path, 'binary'); // JSC shell
  throw new Error('no file API in this engine');
}

const now = (typeof performance !== 'undefined' && performance.now)
  ? () => performance.now()
  : () => Date.now();

const PREFIX_STR = 'SOLANA_MERKLE_SHREDS_LEAF';
const LEAVES = 64;

function report(name, best, worst, totalBytes) {
  const spread = (100 * (worst - best)) / best;
  const us = (best * 1e3).toFixed(2);
  const mbps = ((totalBytes / 1e6) / (best / 1e3)).toFixed(0);
  print(`${name.padEnd(24)}${us.padStart(9)} us/batch${mbps.padStart(8)} MB/s   (spread ${spread.toFixed(1)}%)`);
  return best;
}

const print = typeof console !== 'undefined' ? (s) => console.log(s) : globalThis.print;

async function main() {
  const prefix = new Uint8Array(1 + PREFIX_STR.length);
  for (let i = 0; i < PREFIX_STR.length; i++) prefix[1 + i] = PREFIX_STR.charCodeAt(i);

  const leaves = [];
  for (let i = 0; i < LEAVES; i++) {
    const len = i < 32 ? 1019 : 1044;
    const l = new Uint8Array(len);
    for (let j = 0; j < len; j++) l[j] = ((j & 0xff) * 31 + i) & 0xff;
    leaves.push(l);
  }
  const totalBytes = leaves.reduce((a, l) => a + l.length + prefix.length, 0);

  const { instance } = await WebAssembly.instantiate(loadBytes(wasmPath), {});
  const ex = instance.exports;
  const { memory, walloc } = ex;

  const bodyBytes = leaves.reduce((a, l) => a + l.length, 0);
  const prefixPtr = walloc(prefix.length);
  const dataPtr = walloc(bodyBytes);
  const lensPtr = walloc(LEAVES * 4);
  const outPtr = walloc(LEAVES * 32);
  {
    const mem = new Uint8Array(memory.buffer);
    mem.set(prefix, prefixPtr);
    const lens = new Uint32Array(memory.buffer, lensPtr, LEAVES);
    let off = dataPtr;
    leaves.forEach((l, i) => {
      lens[i] = l.length;
      mem.set(l, off);
      off += l.length;
    });
  }

  {
    const p = walloc(64);
    const n = ex.backend_name(p, 64);
    let name = '';
    const view = new Uint8Array(memory.buffer, p, n);
    for (let i = 0; i < n; i++) name += String.fromCharCode(view[i]);
    print(`engine: ${ENGINE}`);
    print(`dispatch backend: ${name}`);
    print('');
  }

  const rows = [
    ['simd128-4 (dispatch)', ex.hash_many_prefixed_raw],
    ['2x4', ex.hash_many_prefixed_raw_2x4],
    ['4x4', ex.hash_many_prefixed_raw_4x4],
  ].filter(([, f]) => f);

  // All kernels must produce identical digests before any timing counts.
  const digests = rows.map(([, f]) => {
    f(prefixPtr, prefix.length, dataPtr, lensPtr, LEAVES, outPtr);
    return Array.from(new Uint8Array(memory.buffer, outPtr, LEAVES * 32));
  });
  for (let i = 1; i < digests.length; i++) {
    if (digests[i].join() !== digests[0].join()) throw new Error(`${rows[i][0]} disagrees with ${rows[0][0]}`);
  }

  const ROUNDS = 15, ITERS = 50, WARMUP = 200;
  const results = [];
  for (const [name, f] of rows) {
    for (let i = 0; i < WARMUP; i++) f(prefixPtr, prefix.length, dataPtr, lensPtr, LEAVES, outPtr);
    let best = Infinity, worst = 0;
    for (let r = 0; r < ROUNDS; r++) {
      const t0 = now();
      for (let i = 0; i < ITERS; i++) f(prefixPtr, prefix.length, dataPtr, lensPtr, LEAVES, outPtr);
      const ms = (now() - t0) / ITERS;
      best = Math.min(best, ms);
      worst = Math.max(worst, ms);
    }
    results.push([name, report(name, best, worst, totalBytes)]);
  }

  const base = results[0][1];
  print('');
  for (const [name, t] of results) print(`${name.padEnd(24)}${(base / t).toFixed(3)}x vs dispatch`);
}

main().catch((e) => { print(String(e && e.stack || e)); if (IS_PROC) process.exit(1); });
