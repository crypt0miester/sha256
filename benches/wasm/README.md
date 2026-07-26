# wasm / JS benchmark

Measures this crate compiled to WebAssembly, called from JavaScript, against
the SHA-256 implementations Solana's JS SDKs use: `@noble/hashes` (what
`@solana/web3.js` 1.x hashes with), WebCrypto `crypto.subtle.digest` (what
`@solana/kit`'s `@solana/addresses` uses, and why its PDA derivation is
async), plus `hash-wasm` and Node's native `crypto` as references.

Files: `bench.mjs` (Node, merkle batch), `pda.mjs` (PDA derivation and
vanity grinding against the real SDK code paths), `matrix.cjs`
(engine-portable wave comparison, runs in bare shells), `check.mjs`
(dependency-free correctness gate, CI runs it), `browser/` (the same
measurements as a page for running in a real browser).

The tape rows pay the real JS to wasm boundary: 64 separate copies into
linear memory per batch and a digest readback. Workload and estimator are
identical to the native merkle_leaves bench.

Build with simd128 (every mainstream engine has shipped it since 2021) so
dispatch selects the 4-lane `simd128-4` backend; without the flag wasm falls
back to `portable-8`, which roughly ties `hash-wasm`:

```sh
RUSTFLAGS="-C target-feature=+simd128" \
  cargo build --release --example wasm_lib --target wasm32-unknown-unknown
npm install
node --expose-gc bench.mjs
```

The browser page needs two more steps, since both the bundle and the module
it fetches are build output and neither is in git. Run these from this
directory, after the build above:

```sh
npx esbuild browser/browser-bench.mjs --bundle --format=esm \
  --outfile=browser/bundle.js
cp ../../target/wasm32-unknown-unknown/release/examples/wasm_lib.wasm browser/
python3 -m http.server -d browser 8000
```

Then open `http://localhost:8000/`. It has to be served over http: the page
fetches the module, which `file://` refuses. Results print into the page, and
`document.title` becomes `BENCH-DONE` or `BENCH-ERROR` when it settles, so a
headless driver can poll for that instead of scraping the text.

## Where wasm wins, and where it loses

Measured in Node and in a real Chrome tab (`browser/`), same workloads. The
short version, stated plainly because the long version below earns it:

- **Batched ~1 KB messages in a browser: WebCrypto wins. Use WebCrypto.**
  Chrome's `subtle.digest` reaches the hardware SHA unit with far lower
  per-call overhead than Node's (75 vs 231 us for the 64-leaf batch), and
  beats this crate's wasm (80-90 us) at that size. Do not ship wasm SHA-256
  for browser merkle-batch workloads.
- **Small messages in tight sync loops (address grinding, 60-90 bytes):
  wasm wins, in the same Chrome tab**: 4.55 M candidates/s, 3.7x the
  web3.js stack, 1.6x hash-wasm. Per-call overhead swamps WebCrypto at this
  size, and a grinder wants a synchronous loop anyway.
- **PDA derivation: nobody's SHA-256 matters.** The ed25519 off-curve check
  is 94-99% of a candidate's cost (Chrome: 0.94 us hash vs 16.1 us check;
  Node: 0.72 vs 53.8). Batching the hashes wins 4%. The lever there is a
  faster curve check, which is not this crate's business.
- In Node, the right vehicle for this crate is a napi binding, not wasm:
  the native `neon-sha2-x4` backend does the merkle batch in ~16 us, 2.7x
  faster than `node:crypto`, because OpenSSL runs one hardware stream
  latency-bound and this crate interleaves four. Wasm structurally cannot
  reach the SHA units; `v128` integer SIMD is its ceiling.

## Node numbers (M4 Max, Node v24 / V8, unpinned, read as ratios)

```
tape wasm (copy in + out)        100.61 us/batch        673 MB/s
tape wasm (data resident)         99.07 us/batch        683 MB/s
tape wasm 2x4 (resident)          87.04 us/batch        778 MB/s
noble one-shot (web3.js)         337.97 us/batch        200 MB/s
hash-wasm (reused hasher)        156.24 us/batch        433 MB/s
webcrypto Promise.all (kit)      230.90 us/batch        293 MB/s
node crypto.hash one-shot         42.89 us/batch       1578 MB/s
node createHash streaming         48.43 us/batch       1397 MB/s
```

The boundary tax is small: copying 67 KB in and digests out costs a few
microseconds against a ~100 us batch.

## Chrome numbers (Chrome 150, same machine)

```
tape wasm (copy in + out)       90.00 us/batch     752 MB/s
tape wasm 2x4                   85.00 us/batch     796 MB/s
tape wasm 4x4                   80.00 us/batch     846 MB/s
noble (web3.js path)           285.00 us/batch     237 MB/s
hash-wasm (reused hasher)      135.00 us/batch     501 MB/s
webcrypto Promise.all (kit)     75.00 us/batch     902 MB/s

vanity grind (2048 x 87-byte candidates):
tape wasm (batched 512)   4.55 M cand/s
noble                     1.22 M cand/s   tape 3.72x
hash-wasm                 2.82 M cand/s   tape 1.61x

PDA derivation: serial 39.1 us/PDA, tape batched 37.5 (1.04x);
per candidate: hash 0.94 us, curve check 16.09 us
```

Note Chrome clamps `performance.now` to 100 us granularity, so the batch
rows are quantized to 5 us steps; directions are solid, last digits are
not. The important inversion vs Node: Chrome's WebCrypto per-call overhead
is ~3x smaller, which flips the ~1 KB batch verdict in its favour. Node's
WebCrypto numbers do not transfer to browsers; measure in the browser.

## Address grinding, bump-255 PDAs

`pda.mjs` measures the real grinder flow at fixed bump 255: hash the
candidate, test the address prefix, and only prefix hits pay the curve
check (a hit is a valid PDA only if off-curve, ~50% are). The check drags
exactly as its rarity predicts; measured with checks included:

| prefix rarity | tape | noble | edge |
|---|---|---|---|
| ~1/58 (1 base58 char) | 0.76 M cand/s | 0.56 M cand/s | 1.34x |
| ~1/3364 (2 chars) | 3.18 M cand/s | 1.35 M cand/s | 2.36x |
| rarer (check amortized away) | ~4.5 M cand/s | ~1.2 M cand/s | ~3.7x |

Expected time to find a vanity 255-bump PDA in one browser thread
(expected candidates = 2 x 58^k for k base58 chars; halve per extra core
with workers):

| prefix | candidates | tape wasm | noble (web3.js stack) |
|---|---|---|---|
| 3 chars | ~390 K | ~90 ms | ~320 ms |
| 4 chars | ~22.6 M | ~5 s | ~19 s |
| 5 chars | ~1.31 G | ~5 min | ~18 min |
| 6 chars | ~76 G | ~4.6 h | ~17 h |

A native grinder (this crate's own x86/ARM backends) is ~7x the tab's
rate; the browser numbers are a web-app convenience, not a mining rig.
`solana-keygen grind` is deliberately absent everywhere here: vanity
keypairs are ed25519/SHA-512 work, SHA-256 never appears in them.

## Wave count

The `_2x4` and `_4x4` entry points interleave 2 or 4 independent 4-lane
waves per group, giving the engine latency-hiding work a single wave
cannot. Whether that pays is the engine's call, and the engines disagree
(`matrix.cjs`, ratios stable across repeated runs on this host):

| engine | 2x4 | 3x4 | 4x4 |
|---|---|---|---|
| V8 (node) | +3-7% | -3-9% | +4-9% |
| JavaScriptCore (bun) | +13-21% | ~0 | +19-23% |
| SpiderMonkey (jsvu shell) | 0.64x | 0.57x | 0.62x |

Dispatch therefore stays on `simd128-4`: its worst case forgoes ~20%
where a multi-wave default's worst case loses ~40%. The module cannot
calibrate itself (`wasm32-unknown-unknown` has no clock), but JS glue
can: time each exported kernel once at startup with `performance.now`
and keep the winner. That captures the V8/JSC upside without the
SpiderMonkey cliff.

Do not "optimise" the transpose shuffles: folding the byte swap into the
interleave stage (48 shuffles/block down to 32) measured 3-5% slower on
V8, three paired runs of three, because engines pattern-match the
canonical shuffles to single instructions (`rev32`, `zip1`) and the fused
non-canonical one lowers to `tbl` with a materialized mask. The wasm-level
op count is not the machine-level op count.

## Module size

This bench module is a lab, not a shippable artifact: the three extra wave
kernels are ~163 KB of its 210 KB because each is a full flat-round body.
A production module keeping only the dispatched kernel measures:

```
dispatch-only build              47.1 KB
  + wasm-opt -O3                 38.7 KB   (-Oz identical; strips names too)
  gzipped                        14.2 KB
```

Verified identical output (`check.mjs`) and identical speed (94.7 us/batch)
after `wasm-opt`. `panic=abort` saves nothing further. Two notes for whoever
packages this: `wasm-metadce` cannot strip the unused kernels from the full
module, because every kernel sits in the indirect-call table via `GroupFn`
pointers -- build without the extra exports instead; and the size is
dominated by the deliberate flat-round unroll, which is the shape that wins
every engine we measured, so do not trade it away for bytes.
