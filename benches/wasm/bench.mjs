// tape-sha256 compiled to wasm, called from JS, against the SHA-256
// implementations Solana's JS SDKs use. See README.md; workload and
// estimator mirror the native merkle_leaves bench (64 leaves, 67,680
// bytes/batch, best of 15 rounds x 50 iters).
//
// Methodology notes, learned the hard way:
//  - The tape rows pay the JS to wasm boundary (64 copies in, digests out);
//    a data-resident row isolates what the boundary costs.
//  - JS libraries get their fast paths: one-shot APIs, reused hashers, no
//    per-message object churn feeding the GC.
//  - WebCrypto gets Promise.all, its real batch shape; sequential await
//    would measure promise latency instead of hashing.
//  - 200 warmup iterations for V8 tier-up, gc() between rounds when exposed.
import { sha256 } from '@noble/hashes/sha2.js';
import { createSHA256 } from 'hash-wasm';
import { createHash, hash as oneShotHash } from 'node:crypto';
import { readFile } from 'node:fs/promises';
import { resolve } from 'node:path';

const PREFIX = new Uint8Array([0x00, ...new TextEncoder().encode('SOLANA_MERKLE_SHREDS_LEAF')]);
const LEAVES = 64;

const leaves = [];
for (let i = 0; i < LEAVES; i++) {
  const len = i < 32 ? 1019 : 1044;
  const l = new Uint8Array(len);
  for (let j = 0; j < len; j++) l[j] = ((j & 0xff) * 31 + i) & 0xff;
  leaves.push(l);
}
const totalBytes = leaves.reduce((a, l) => a + l.length + PREFIX.length, 0);

// Pre-joined prefix||leaf buffers for one-shot APIs, built outside the timed
// region -- the same treatment ../merkle_leaves.rs gives Firedancer's
// contiguous-buffer API.
const joined = leaves.map((l) => {
  const b = new Uint8Array(PREFIX.length + l.length);
  b.set(PREFIX, 0);
  b.set(l, PREFIX.length);
  return b;
});

const maybeGc = globalThis.gc ?? (() => {});

function report(name, best, worst) {
  const spread = (100 * (worst - best)) / best;
  const us = (best * 1e6).toFixed(2).padStart(9);
  const mbps = (totalBytes / best / 1e6).toFixed(0).padStart(11);
  console.log(`${name.padEnd(30)}${us} us/batch${mbps} MB/s   (spread ${spread.toFixed(1).padStart(4)}%)`);
  return best;
}

function bench(name, f) {
  const ROUNDS = 15, ITERS = 50, WARMUP = 200;
  for (let i = 0; i < WARMUP; i++) f();
  let best = Infinity, worst = 0;
  for (let r = 0; r < ROUNDS; r++) {
    maybeGc();
    const t0 = process.hrtime.bigint();
    for (let i = 0; i < ITERS; i++) f();
    const s = Number(process.hrtime.bigint() - t0) / 1e9 / ITERS;
    best = Math.min(best, s);
    worst = Math.max(worst, s);
  }
  return report(name, best, worst);
}

async function benchAsync(name, f) {
  const ROUNDS = 15, ITERS = 50, WARMUP = 200;
  for (let i = 0; i < WARMUP; i++) await f();
  let best = Infinity, worst = 0;
  for (let r = 0; r < ROUNDS; r++) {
    maybeGc();
    const t0 = process.hrtime.bigint();
    for (let i = 0; i < ITERS; i++) await f();
    const s = Number(process.hrtime.bigint() - t0) / 1e9 / ITERS;
    best = Math.min(best, s);
    worst = Math.max(worst, s);
  }
  return report(name, best, worst);
}

console.log(`${LEAVES} leaves, ${totalBytes} bytes/batch, node ${process.version}, gc ${globalThis.gc ? 'exposed' : 'NOT exposed (run with --expose-gc)'}`);
console.log('');

const out = new Array(LEAVES);

// ---- tape-sha256 as a wasm module ----
const wasmPath = process.argv[2]
  ?? resolve(import.meta.dirname, '../../target/wasm32-unknown-unknown/release/examples/wasm_lib.wasm');
const { instance } = await WebAssembly.instantiate(await readFile(wasmPath), {});
const { memory, walloc, hash_many_prefixed_raw, backend_name } = instance.exports;

{
  const p = walloc(64);
  const n = backend_name(p, 64);
  const name = new TextDecoder().decode(new Uint8Array(memory.buffer, p, n));
  console.log(`tape wasm dispatch backend: ${name}`);
  console.log('');
}

const bodyBytes = leaves.reduce((a, l) => a + l.length, 0);
const prefixPtr = walloc(PREFIX.length);
const dataPtr = walloc(bodyBytes);
const lensPtr = walloc(LEAVES * 4);
const outPtr = walloc(LEAVES * 32);

{
  const mem = new Uint8Array(memory.buffer);
  mem.set(PREFIX, prefixPtr);
  const lens = new Uint32Array(memory.buffer, lensPtr, LEAVES);
  leaves.forEach((l, i) => (lens[i] = l.length));
}

// The boundary paid in full: 64 copies in, one call, digests read back.
bench('tape wasm (copy in + out)', () => {
  const mem = new Uint8Array(memory.buffer);
  let off = dataPtr;
  for (let i = 0; i < LEAVES; i++) {
    mem.set(leaves[i], off);
    off += leaves[i].length;
  }
  hash_many_prefixed_raw(prefixPtr, PREFIX.length, dataPtr, lensPtr, LEAVES, outPtr);
  for (let i = 0; i < LEAVES; i++) {
    out[i] = mem.slice(outPtr + i * 32, outPtr + (i + 1) * 32);
  }
});

// Data already in linear memory, digests left there. The gap to the row
// above is the boundary tax.
bench('tape wasm (data resident)', () => {
  hash_many_prefixed_raw(prefixPtr, PREFIX.length, dataPtr, lensPtr, LEAVES, outPtr);
});

// Experimental two-wave kernel, present only in simd128 builds.
const raw2x4 = instance.exports.hash_many_prefixed_raw_2x4;
if (raw2x4) {
  bench('tape wasm 2x4 (resident)', () => {
    raw2x4(prefixPtr, PREFIX.length, dataPtr, lensPtr, LEAVES, outPtr);
  });
}

// ---- the libraries Solana JS SDKs use ----
bench('noble one-shot (web3.js)', () => {
  for (let i = 0; i < LEAVES; i++) out[i] = sha256(joined[i]);
});

const hw = await createSHA256();
bench('hash-wasm (reused hasher)', () => {
  for (let i = 0; i < LEAVES; i++) {
    hw.init();
    hw.update(joined[i]);
    out[i] = hw.digest('binary');
  }
});

await benchAsync('webcrypto Promise.all (kit)', async () => {
  const r = await Promise.all(joined.map((j) => crypto.subtle.digest('SHA-256', j)));
  for (let i = 0; i < LEAVES; i++) out[i] = r[i];
});

// ---- native references: Node only, a browser has neither ----
bench('node crypto.hash one-shot', () => {
  for (let i = 0; i < LEAVES; i++) out[i] = oneShotHash('sha256', joined[i], 'buffer');
});

bench('node createHash streaming', () => {
  for (let i = 0; i < LEAVES; i++) {
    out[i] = createHash('sha256').update(PREFIX).update(leaves[i]).digest();
  }
});

// ---- cross-checks ----
// The wasm module is checked differentially against node:crypto over every
// batch lane, plus ragged batches spanning the padding and block-boundary
// edge lengths, since only the wasm backend is new code here.
function tapeHashBatch(bodies) {
  const mem = new Uint8Array(memory.buffer);
  const lens = new Uint32Array(memory.buffer, lensPtr, bodies.length);
  let off = dataPtr;
  bodies.forEach((b, i) => {
    lens[i] = b.length;
    mem.set(b, off);
    off += b.length;
  });
  hash_many_prefixed_raw(prefixPtr, PREFIX.length, dataPtr, lensPtr, bodies.length, outPtr);
  return bodies.map((_, i) => Buffer.from(mem.slice(outPtr + i * 32, outPtr + (i + 1) * 32)));
}

function refHash(body) {
  return createHash('sha256').update(PREFIX).update(body).digest();
}

for (const [i, d] of tapeHashBatch(leaves).entries()) {
  if (!d.equals(refHash(leaves[i]))) throw new Error(`tape wasm diverges on batch lane ${i}`);
}

const edges = [0, 1, 3, 37, 53, 55, 56, 57, 63, 64, 65, 119, 127, 128, 129, 200, 955, 1019, 1044, 1070];
for (let start = 0; start < edges.length; start++) {
  const lens = edges.slice(start, start + 7);
  const bodies = lens.map((len, i) => {
    const b = new Uint8Array(len);
    for (let j = 0; j < len; j++) b[j] = ((j & 0xff) * 31 + i) & 0xff;
    return b;
  });
  const got = tapeHashBatch(bodies);
  for (let i = 0; i < bodies.length; i++) {
    if (!got[i].equals(refHash(bodies[i]))) throw new Error(`tape wasm diverges, lens ${lens} lane ${i}`);
  }
}

const want = oneShotHash('sha256', joined[0], 'buffer');
if (!Buffer.from(sha256(joined[0])).equals(want)) throw new Error('noble diverges from node:crypto');
hw.init();
hw.update(joined[0]);
if (!Buffer.from(hw.digest('binary')).equals(want)) throw new Error('hash-wasm diverges from node:crypto');
console.log('\ntape wasm matches node:crypto on all lanes, uniform and ragged');
