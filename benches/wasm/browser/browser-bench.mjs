// The wasm/JS benchmark in an actual browser tab: merkle batch, vanity
// grind, and PDA derivation. Bundled with `bun build` (see README); the
// curve check is @noble/curves ed25519 decompression, which is literally
// what web3.js's PublicKey.isOnCurve wraps.
import { sha256 } from '@noble/hashes/sha2.js';
import { createSHA256 } from 'hash-wasm';
import { ed25519 } from '@noble/curves/ed25519';

const out = [];
const el = document.getElementById('out');
function print(s) {
  out.push(s);
  el.textContent = out.join('\n');
  console.log(s);
}

const Point = ed25519.ExtendedPoint ?? ed25519.Point;
function isOnCurve(bytes) {
  try {
    Point.fromHex(bytes);
    return true;
  } catch {
    return false;
  }
}

function bench(f, { rounds = 12, iters = 20, warmup = 100 } = {}) {
  for (let i = 0; i < warmup; i++) f();
  let best = Infinity, worst = 0;
  for (let r = 0; r < rounds; r++) {
    const t0 = performance.now();
    for (let i = 0; i < iters; i++) f();
    const s = (performance.now() - t0) / 1e3 / iters;
    best = Math.min(best, s);
    worst = Math.max(worst, s);
  }
  return [best, worst];
}

async function benchAsync(f, { rounds = 12, iters = 20, warmup = 100 } = {}) {
  for (let i = 0; i < warmup; i++) await f();
  let best = Infinity, worst = 0;
  for (let r = 0; r < rounds; r++) {
    const t0 = performance.now();
    for (let i = 0; i < iters; i++) await f();
    const s = (performance.now() - t0) / 1e3 / iters;
    best = Math.min(best, s);
    worst = Math.max(worst, s);
  }
  return [best, worst];
}

function row(name, [best, worst], totalBytes) {
  const spread = (100 * (worst - best)) / best;
  const us = (best * 1e6).toFixed(2).padStart(9);
  const mbps = (totalBytes / best / 1e6).toFixed(0).padStart(8);
  print(`${name.padEnd(28)}${us} us/batch${mbps} MB/s   (spread ${spread.toFixed(1)}%)`);
  return best;
}

async function main() {
  print(`user agent: ${navigator.userAgent}`);

  const resp = await fetch('wasm_lib.wasm');
  const { instance } = await WebAssembly.instantiate(await resp.arrayBuffer(), {});
  const ex = instance.exports;
  const { memory, walloc } = ex;

  const MAX = 1024;
  const prefixPtr = walloc(64);
  const dataPtr = walloc(MAX * 128 + 70000);
  const lensPtr = walloc(MAX * 4);
  const outPtr = walloc(MAX * 32);

  {
    const p = walloc(64);
    const n = ex.backend_name(p, 64);
    print(`dispatch backend: ${new TextDecoder().decode(new Uint8Array(memory.buffer, p, n))}`);
    print('');
  }

  function tapeBatch(entry, msgs, prefix) {
    const mem = new Uint8Array(memory.buffer);
    let plen = 0;
    if (prefix) {
      mem.set(prefix, prefixPtr);
      plen = prefix.length;
    }
    const lens = new Uint32Array(memory.buffer, lensPtr, msgs.length);
    let off = dataPtr;
    msgs.forEach((m, i) => {
      lens[i] = m.length;
      mem.set(m, off);
      off += m.length;
    });
    entry(prefixPtr, plen, dataPtr, lensPtr, msgs.length, outPtr);
    return msgs.map((_, i) => mem.slice(outPtr + i * 32, outPtr + (i + 1) * 32));
  }

  // ------------------------------------------------ merkle leaves batch
  const PREFIX = new Uint8Array([0, ...new TextEncoder().encode('SOLANA_MERKLE_SHREDS_LEAF')]);
  const leaves = [];
  for (let i = 0; i < 64; i++) {
    const len = i < 32 ? 1019 : 1044;
    const l = new Uint8Array(len);
    for (let j = 0; j < len; j++) l[j] = ((j & 0xff) * 31 + i) & 0xff;
    leaves.push(l);
  }
  const totalBytes = leaves.reduce((a, l) => a + l.length + PREFIX.length, 0);
  const joined = leaves.map((l) => {
    const b = new Uint8Array(PREFIX.length + l.length);
    b.set(PREFIX, 0);
    b.set(l, PREFIX.length);
    return b;
  });

  // Correctness inside the browser first: tape vs noble on every lane.
  {
    const got = tapeBatch(ex.hash_many_prefixed_raw, leaves, PREFIX);
    for (let i = 0; i < 64; i++) {
      const want = sha256(joined[i]);
      for (let b = 0; b < 32; b++) {
        if (got[i][b] !== want[b]) throw new Error(`lane ${i} diverges from noble`);
      }
    }
    print('correctness: 64/64 lanes match noble in-browser');
    print('');
  }

  print('--- merkle batch, 64 leaves, 67680 bytes ---');
  const o = new Array(64);
  row('tape wasm (copy in + out)', bench(() => {
    tapeBatch(ex.hash_many_prefixed_raw, leaves, PREFIX);
  }), totalBytes);
  if (ex.hash_many_prefixed_raw_2x4) {
    row('tape wasm 2x4', bench(() => {
      tapeBatch(ex.hash_many_prefixed_raw_2x4, leaves, PREFIX);
    }), totalBytes);
  }
  if (ex.hash_many_prefixed_raw_4x4) {
    row('tape wasm 4x4', bench(() => {
      tapeBatch(ex.hash_many_prefixed_raw_4x4, leaves, PREFIX);
    }), totalBytes);
  }
  row('noble (web3.js path)', bench(() => {
    for (let i = 0; i < 64; i++) o[i] = sha256(joined[i]);
  }), totalBytes);
  const hw = await createSHA256();
  row('hash-wasm (reused hasher)', bench(() => {
    for (let i = 0; i < 64; i++) {
      hw.init();
      hw.update(joined[i]);
      o[i] = hw.digest('binary');
    }
  }), totalBytes);
  row('webcrypto Promise.all (kit)', await benchAsync(async () => {
    const r = await Promise.all(joined.map((j) => crypto.subtle.digest('SHA-256', j)));
    for (let i = 0; i < 64; i++) o[i] = r[i];
  }), totalBytes);

  // ------------------------------------------------ vanity grind
  print('');
  print('--- vanity grind, createWithSeed shape, 2048 candidates ---');
  const BASE = new Uint8Array(32).fill(7);
  const OWNER = new Uint8Array(32).fill(1);
  const GRIND = 2048;
  const grindMsgs = Array.from({ length: GRIND }, (_, n) => {
    const seedTxt = new TextEncoder().encode(`vanity-${n.toString(36).padStart(16, '0')}`);
    const m = new Uint8Array(32 + seedTxt.length + 32);
    m.set(BASE, 0);
    m.set(seedTxt, 32);
    m.set(OWNER, 32 + seedTxt.length);
    return m;
  });
  let sink = 0;
  const g1 = bench(() => {
    for (let off = 0; off < GRIND; off += 512) {
      const ds = tapeBatch(ex.hash_many_prefixed_raw, grindMsgs.slice(off, off + 512));
      for (const d of ds) if (d[0] === 0xab) sink++;
    }
  }, { iters: 4 })[0];
  const g2 = bench(() => {
    for (const m of grindMsgs) if (sha256(m)[0] === 0xab) sink++;
  }, { iters: 4 })[0];
  const g3 = bench(() => {
    for (const m of grindMsgs) {
      hw.init();
      hw.update(m);
      if (hw.digest('binary')[0] === 0xab) sink++;
    }
  }, { iters: 4 })[0];
  print(`tape wasm (batched 512)   ${(GRIND / g1 / 1e6).toFixed(2)} M cand/s`);
  print(`noble                     ${(GRIND / g2 / 1e6).toFixed(2)} M cand/s   tape ${(g2 / g1).toFixed(2)}x`);
  print(`hash-wasm                 ${(GRIND / g3 / 1e6).toFixed(2)} M cand/s   tape ${(g3 / g1).toFixed(2)}x`);

  // ------------------------------------------------ PDA derivation
  print('');
  print('--- PDA derivation, 64 PDAs (noble curve check, web3.js algorithm) ---');
  const MARKER = new TextEncoder().encode('ProgramDerivedAddress');
  const PROG = new Uint8Array(32).fill(3);
  const seeds = Array.from({ length: 64 }, (_, i) => {
    const s = new Uint8Array(8);
    new DataView(s.buffer).setUint32(0, (i * 2654435761) >>> 0);
    return s;
  });
  const cand = (seed, bump) => {
    const m = new Uint8Array(seed.length + 1 + 32 + MARKER.length);
    m.set(seed, 0);
    m[seed.length] = bump;
    m.set(PROG, seed.length + 1);
    m.set(MARKER, seed.length + 1 + 32);
    return m;
  };
  const serialFind = () => {
    for (const s of seeds) {
      for (let bump = 255; ; bump--) {
        if (!isOnCurve(sha256(cand(s, bump)))) break;
      }
    }
  };
  const batchFind = () => {
    let pending = seeds.map((seed) => ({ seed, bump: 255 }));
    while (pending.length > 0) {
      const ds = tapeBatch(ex.hash_many_prefixed_raw, pending.map((p) => cand(p.seed, p.bump)));
      pending = pending.filter((p, k) => isOnCurve(ds[k]) && (p.bump--, true));
    }
  };
  const p1 = bench(serialFind, { rounds: 8, iters: 3, warmup: 10 })[0];
  const p2 = bench(batchFind, { rounds: 8, iters: 3, warmup: 10 })[0];
  print(`serial (web3.js shape)    ${(p1 * 1e6 / 64).toFixed(2)} us/PDA`);
  print(`tape batched              ${(p2 * 1e6 / 64).toFixed(2)} us/PDA   ${(p1 / p2).toFixed(2)}x`);
  const [h] = bench(() => { for (let i = 0; i < 64; i++) sha256(cand(seeds[0], 255)); });
  const d0 = sha256(cand(seeds[0], 255));
  const [c] = bench(() => { for (let i = 0; i < 64; i++) isOnCurve(d0); });
  print(`decomposition: hash ${(h / 64 * 1e6).toFixed(2)} us, curve check ${(c / 64 * 1e6).toFixed(2)} us per candidate`);

  print('');
  print(`DONE (sink ${sink})`);
  document.title = 'BENCH-DONE';
}

main().catch((e) => {
  print(`ERROR: ${e && e.stack || e}`);
  document.title = 'BENCH-ERROR';
});
