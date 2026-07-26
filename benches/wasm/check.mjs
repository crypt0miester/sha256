// Differential gate for the wasm backends, no dependencies: every exported
// kernel against node:crypto across uniform batches, ragged batches, and
// the padding and block-boundary edge lengths. CI runs this; the benches
// assume it passes.
import { createHash } from 'node:crypto';
import { readFile } from 'node:fs/promises';
import { resolve } from 'node:path';

const PREFIX = new Uint8Array([0x00, ...new TextEncoder().encode('SOLANA_MERKLE_SHREDS_LEAF')]);

const wasmPath = process.argv[2]
  ?? resolve(import.meta.dirname, '../../target/wasm32-unknown-unknown/release/examples/wasm_lib.wasm');
const { instance } = await WebAssembly.instantiate(await readFile(wasmPath), {});
const { memory, walloc, backend_name } = instance.exports;

{
  const p = walloc(64);
  const n = backend_name(p, 64);
  console.log(`dispatch backend: ${new TextDecoder().decode(new Uint8Array(memory.buffer, p, n))}`);
}

const CAP = 128;
const prefixPtr = walloc(PREFIX.length);
const dataPtr = walloc(CAP * 1100);
const lensPtr = walloc(CAP * 4);
const outPtr = walloc(CAP * 32);
new Uint8Array(memory.buffer).set(PREFIX, prefixPtr);

function hashBatch(entry, bodies) {
  const mem = new Uint8Array(memory.buffer);
  const lens = new Uint32Array(memory.buffer, lensPtr, bodies.length);
  let off = dataPtr;
  bodies.forEach((b, i) => {
    lens[i] = b.length;
    mem.set(b, off);
    off += b.length;
  });
  entry(prefixPtr, PREFIX.length, dataPtr, lensPtr, bodies.length, outPtr);
  return bodies.map((_, i) => Buffer.from(mem.slice(outPtr + i * 32, outPtr + (i + 1) * 32)));
}

const refHash = (body) => createHash('sha256').update(PREFIX).update(body).digest();
const body = (len, seed) => {
  const b = new Uint8Array(len);
  for (let j = 0; j < len; j++) b[j] = ((j & 0xff) * 31 + seed) & 0xff;
  return b;
};

// Empty, tiny, both sides of each block boundary and of the 55/56 padding
// break, plus the real Merkle-leaf sizes.
const edges = [0, 1, 2, 3];
for (let block = 0; block < 3; block++) {
  for (const d of [53, 54, 55, 56, 57, 62, 63, 64, 65, 66]) edges.push(block * 64 + d);
}
edges.push(955, 1019, 1044, 1045, 1070);

const kernels = Object.entries({
  'dispatch': instance.exports.hash_many_prefixed_raw,
  '2x4': instance.exports.hash_many_prefixed_raw_2x4,
  '4x4': instance.exports.hash_many_prefixed_raw_4x4,
}).filter(([, f]) => f);

let batches = 0;
for (const [name, entry] of kernels) {
  // Uniform batches at every edge length, at every batch size through
  // several lane widths (partial groups and inactive lanes included).
  for (const len of edges) {
    for (const count of [1, 2, 3, 4, 5, 7, 8, 9, 16, 17]) {
      const bodies = Array.from({ length: count }, (_, i) => body(len, i));
      const got = hashBatch(entry, bodies);
      bodies.forEach((b, i) => {
        if (!got[i].equals(refHash(b))) {
          throw new Error(`${name}: len ${len} count ${count} lane ${i} diverges`);
        }
      });
      batches++;
    }
  }
  // Ragged batches: sliding windows over the edge lengths.
  for (let s = 0; s + 8 <= edges.length; s++) {
    const lens = edges.slice(s, s + 8);
    const bodies = lens.map((len, i) => body(len, i));
    const got = hashBatch(entry, bodies);
    bodies.forEach((b, i) => {
      if (!got[i].equals(refHash(b))) {
        throw new Error(`${name}: ragged lens ${lens} lane ${i} diverges`);
      }
    });
    batches++;
  }
  console.log(`${name}: ok`);
}
console.log(`${batches} batches match node:crypto`);
