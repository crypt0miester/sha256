// Two Solana address workloads, measured against the SDKs' own code paths:
// PDA derivation (hash bump candidates, keep the first off-curve; only the
// hashing differs between rows, the curve check is web3.js's own) and
// seed-address vanity grinding (createWithSeed shape, no curve check, the
// pure hash race vanity tools actually run).
//
// solana-keygen grind is absent on purpose: vanity keypairs are ed25519 and
// SHA-512 work, so this crate has nothing to say about it.
//
// Messages here are 2 blocks against 17 for a Merkle leaf, so per-batch
// overhead weighs heaviest here; that is what makes it worth measuring.
import { sha256 } from '@noble/hashes/sha2.js';
import { createSHA256 } from 'hash-wasm';
import { PublicKey } from '@solana/web3.js';
import { getProgramDerivedAddress, address } from '@solana/addresses';
import { createHash } from 'node:crypto';
import { readFile } from 'node:fs/promises';
import { resolve } from 'node:path';

const wasmPath = process.argv[2]
  ?? resolve(import.meta.dirname, '../../target/wasm32-unknown-unknown/release/examples/wasm_lib.wasm');
const { instance } = await WebAssembly.instantiate(await readFile(wasmPath), {});
const { memory, walloc, hash_many_prefixed_raw } = instance.exports;

const MAX = 1024;
const dataPtr = walloc(MAX * 128);
const lensPtr = walloc(MAX * 4);
const outPtr = walloc(MAX * 32);

/// One wasm call: hash `msgs` (each a Uint8Array), return 32-byte digests.
function tapeHashBatch(msgs) {
  const mem = new Uint8Array(memory.buffer);
  const lens = new Uint32Array(memory.buffer, lensPtr, msgs.length);
  let off = dataPtr;
  msgs.forEach((m, i) => {
    lens[i] = m.length;
    mem.set(m, off);
    off += m.length;
  });
  hash_many_prefixed_raw(0, 0, dataPtr, lensPtr, msgs.length, outPtr);
  return msgs.map((_, i) => mem.slice(outPtr + i * 32, outPtr + (i + 1) * 32));
}

function bench(name, f, { rounds = 10, iters = 10 } = {}) {
  for (let i = 0; i < 30; i++) f();
  let best = Infinity;
  for (let r = 0; r < rounds; r++) {
    const t0 = process.hrtime.bigint();
    for (let i = 0; i < iters; i++) f();
    best = Math.min(best, Number(process.hrtime.bigint() - t0) / 1e9 / iters);
  }
  return best;
}

async function benchAsync(name, f, { rounds = 10, iters = 10 } = {}) {
  for (let i = 0; i < 30; i++) await f();
  let best = Infinity;
  for (let r = 0; r < rounds; r++) {
    const t0 = process.hrtime.bigint();
    for (let i = 0; i < iters; i++) await f();
    best = Math.min(best, Number(process.hrtime.bigint() - t0) / 1e9 / iters);
  }
  return best;
}

// ---------------------------------------------------------------- PDA setup
const PROGRAM = new PublicKey('TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA');
const PROGRAM_KIT = address(PROGRAM.toBase58());
const PDA_MARKER = new TextEncoder().encode('ProgramDerivedAddress');
const N_PDAS = 64;

// 8-byte seeds, one per derivation.
const seeds = Array.from({ length: N_PDAS }, (_, i) => {
  const s = new Uint8Array(8);
  new DataView(s.buffer).setUint32(0, i * 2654435761 >>> 0);
  return s;
});

function pdaCandidate(seed, bump) {
  const m = new Uint8Array(seed.length + 1 + 32 + PDA_MARKER.length);
  m.set(seed, 0);
  m[seed.length] = bump;
  m.set(PROGRAM.toBytes(), seed.length + 1);
  m.set(PDA_MARKER, seed.length + 1 + 32);
  return m;
}

/// Batched PDA resolution: one wasm call hashes the current bump candidate
/// of every unresolved derivation; the curve check is web3.js's own.
function tapeFindPdas() {
  const out = new Array(N_PDAS);
  let pending = seeds.map((seed, i) => ({ seed, i, bump: 255 }));
  while (pending.length > 0) {
    const digests = tapeHashBatch(pending.map((p) => pdaCandidate(p.seed, p.bump)));
    const next = [];
    for (let k = 0; k < pending.length; k++) {
      if (PublicKey.isOnCurve(digests[k])) {
        pending[k].bump -= 1;
        next.push(pending[k]);
      } else {
        out[pending[k].i] = [digests[k], pending[k].bump];
      }
    }
    pending = next;
  }
  return out;
}

// Correctness: must equal web3.js exactly, address and bump.
{
  const ours = tapeFindPdas();
  for (let i = 0; i < N_PDAS; i++) {
    const [addr, bump] = PublicKey.findProgramAddressSync([seeds[i]], PROGRAM);
    if (!addr.toBuffer().equals(Buffer.from(ours[i][0])) || bump !== ours[i][1]) {
      throw new Error(`PDA mismatch at seed ${i}`);
    }
  }
  console.log(`correctness: ${N_PDAS} PDAs match web3.js exactly (address and bump)\n`);
}

console.log(`--- PDA derivation, ${N_PDAS} PDAs per iteration ---`);

const w3 = bench('web3', () => {
  for (const s of seeds) PublicKey.findProgramAddressSync([s], PROGRAM);
});
console.log(`web3.js findProgramAddressSync  ${(w3 * 1e6 / N_PDAS).toFixed(2).padStart(8)} us/PDA   ${(N_PDAS / w3).toFixed(0).padStart(8)} PDA/s`);

const kit = await benchAsync('kit', async () => {
  await Promise.all(seeds.map((s) => getProgramDerivedAddress({ programAddress: PROGRAM_KIT, seeds: [s] })));
});
console.log(`kit getProgramDerivedAddress    ${(kit * 1e6 / N_PDAS).toFixed(2).padStart(8)} us/PDA   ${(N_PDAS / kit).toFixed(0).padStart(8)} PDA/s`);

const tp = bench('tape', () => tapeFindPdas());
console.log(`tape batched + web3 isOnCurve   ${(tp * 1e6 / N_PDAS).toFixed(2).padStart(8)} us/PDA   ${(N_PDAS / tp).toFixed(0).padStart(8)} PDA/s`);

// Decomposition: what a PDA costs is hash + curve check; measure each alone.
const oneMsg = pdaCandidate(seeds[0], 255);
const oneDigest = sha256(oneMsg);
const hashAlone = bench('h', () => { for (let i = 0; i < 64; i++) sha256(oneMsg); }) / 64;
const curveAlone = bench('c', () => { for (let i = 0; i < 64; i++) PublicKey.isOnCurve(oneDigest); }) / 64;
console.log(`\ndecomposition (noble hash ${(hashAlone * 1e6).toFixed(2)} us, isOnCurve ${(curveAlone * 1e6).toFixed(2)} us per candidate)`);
console.log(`curve check share of a web3.js candidate: ${(100 * curveAlone / (hashAlone + curveAlone)).toFixed(0)}%`);

// ------------------------------------------------- seed-address vanity grind
// createWithSeed shape: sha256(base(32) || seed-ascii || owner(32)), no
// curve check. The grind varies the seed text and tests a byte prefix.
console.log(`\n--- vanity grind, createWithSeed shape (87-byte messages, no curve check) ---`);

const BASE = PROGRAM.toBytes();
const OWNER = new PublicKey('11111111111111111111111111111111').toBytes();
const GRIND = 2048;
const TARGET0 = 0xab;

function grindMsg(n) {
  const seedTxt = new TextEncoder().encode(`vanity-${n.toString(36).padStart(16, '0')}`);
  const m = new Uint8Array(32 + seedTxt.length + 32);
  m.set(BASE, 0);
  m.set(seedTxt, 32);
  m.set(OWNER, 32 + seedTxt.length);
  return m;
}
const grindMsgs = Array.from({ length: GRIND }, (_, n) => grindMsg(n));

let sink = 0;
const rows = [];

const tGrindTape = bench('tape', () => {
  for (let off = 0; off < GRIND; off += 512) {
    const digests = tapeHashBatch(grindMsgs.slice(off, off + 512));
    for (const d of digests) if (d[0] === TARGET0) sink++;
  }
}, { iters: 3 });
rows.push(['tape wasm (batched 512)', tGrindTape]);

const tGrindNoble = bench('noble', () => {
  for (const m of grindMsgs) if (sha256(m)[0] === TARGET0) sink++;
}, { iters: 3 });
rows.push(['noble (web3.js path)', tGrindNoble]);

const hw = await createSHA256();
const tGrindHw = bench('hashwasm', () => {
  for (const m of grindMsgs) {
    hw.init();
    hw.update(m);
    if (hw.digest('binary')[0] === TARGET0) sink++;
  }
}, { iters: 3 });
rows.push(['hash-wasm', tGrindHw]);

const tGrindNode = bench('node', () => {
  for (const m of grindMsgs) if (createHash('sha256').update(m).digest()[0] === TARGET0) sink++;
}, { iters: 3 });
rows.push(['node:crypto (native ref)', tGrindNode]);

for (const [name, t] of rows) {
  console.log(`${name.padEnd(26)}${(GRIND / t / 1e6).toFixed(2).padStart(7)} M candidates/s`);
}
console.log(`\n(prefix-hit sink: ${sink})`);

// -------------------------------------------- PDA vanity grind, MEASURED
// Fixed bump 255, the flow real PDA grinders use: hash the candidate,
// test the address prefix (cheap byte compare), and only prefix hits pay
// the curve check. Whether that amortizes the 54 us check depends on the
// prefix rarity, so both a common and a rare prefix are measured, curve
// checks included, not assumed away.
console.log('\n--- PDA vanity grind, fixed bump 255, curve check on prefix hits only ---');

const pdaGrindMsgs = Array.from({ length: GRIND }, (_, n) => {
  const s = new Uint8Array(8);
  new DataView(s.buffer).setUint32(0, n * 40503 >>> 0);
  new DataView(s.buffer).setUint32(4, n);
  return pdaCandidate(s, 255);
});

for (const [label, hitMask] of [['~1/58 prefix (1 base58 char)', 58], ['~1/3364 prefix (2 chars)', 3364]]) {
  let hits = 0, valid = 0;
  const isHit = (d, i) => (((d[0] << 8) | d[1]) % hitMask) === 0;

  const tTape = bench('t', () => {
    for (let off = 0; off < GRIND; off += 512) {
      const digests = tapeHashBatch(pdaGrindMsgs.slice(off, off + 512));
      for (const d of digests) {
        if (isHit(d)) { hits++; if (!PublicKey.isOnCurve(d)) valid++; }
      }
    }
  }, { iters: 3 });

  const tNoble = bench('n', () => {
    for (const m of pdaGrindMsgs) {
      const d = sha256(m);
      if (isHit(d)) { hits++; if (!PublicKey.isOnCurve(d)) valid++; }
    }
  }, { iters: 3 });

  console.log(`${label}:  tape ${(GRIND / tTape / 1e6).toFixed(2)} M cand/s  |  noble ${(GRIND / tNoble / 1e6).toFixed(2)} M cand/s  |  tape ${(tNoble / tTape).toFixed(2)}x  (hits/round ~${Math.round(hits / 8 / 2)}, valid ~${Math.round(valid / 8 / 2)})`);
}
