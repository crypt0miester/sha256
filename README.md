# tape-sha256

Pure-Rust, SIMD-accelerated **multi-buffer SHA-256**: hash many independent
messages at once, one message per SIMD lane.

SHA-256 is inherently serial *within* one message, since each of the 64 rounds
depends on the last, so a single hash cannot be vectorised. But `N`
independent messages can run in lockstep, one per lane, turning every round
into an elementwise vector operation. The canonical workload is a Merkle
tree: every leaf is independent, so the whole bottom level hashes in one
pass.

```rust
use tape_sha256::hash_many_prefixed;

// tape's Merkle leaves: sha256("LEAF" || slice), one call per slice group
let slices: Vec<&[u8]> = vec![&[1u8; 4096], &[2u8; 4096]];
let mut out = vec![[0u8; 32]; slices.len()];
hash_many_prefixed(b"LEAF", &slices, &mut out);
```

`hash_many_prefixed` hashes `prefix || body` without ever materialising the
concatenation: the kernel reads straight out of both slices. For
domain-separated Merkle leaves, tape's and agave's alike, that skips a full
copy of every message per batch, an advantage baked into the API shape
rather than an optimisation flag.

## In tape

tape commits every erasure-coded blob to a Merkle root over its slice
group: up to 20 slices of up to 4 KiB, each leaf hashed as
`sha256("LEAF" || slice)` and pairs joined as
`sha256("LEFT" || left || "RIGHT" || right)`. The slicer builds that tree
on every write, the gateway and archive nodes rebuild it to verify reads,
and tape-sdk exposes the same hashing natively, over FFI to wasmtime hosts,
and to browsers through wasm-bindgen.

Measured on tape's own shape (`cargo bench --bench blob_group`, Apple
M-series, min of 15 rounds):

```
leaves sha2 (serial, current)    128.92 us/group       636 MB/s
leaves tape hash_many_prefixed    18.72 us/group      4381 MB/s
joins sha2 (serial, current)       4.19 us/group       366 MB/s
joins tape hash_many               0.72 us/group      2122 MB/s

whole commitment: 6.85x  (133.1 -> 19.4 us/group)
```

One core commits ~51,000 slice groups per second, about 4.2 GB/s of blob
payload. On x86 (AMD Zen 4, serial SHA-NI baseline) the same shape measures
1.74x end to end, and 1.12x faster than Firedancer's batch kernels: a
20-slice group leaves a tail the 16-lane kernel would run mostly empty, so
dispatch routes small remainders to the SHA-NI streams, an option
Firedancer does not have. Full numbers in [BENCHMARKS.md](BENCHMARKS.md);
for the browser SDK, the wasm findings in `benches/wasm/README.md` apply.

## Backends

The best kernel for the running CPU is chosen at load time and falls back
gracefully:

| backend | selected when | lanes |
|---|---|---|
| `avx512-2x16` | AMD Zen 5 (two 16-lane waves interlaced) | 32 |
| `avx512-16` | other x86-64 with AVX-512F+BW | 16 |
| `shani-x4` | x86-64 with SHA-NI but no AVX-512, and AMD Zen 4/3 | 4 streams |
| `avx2-8` | x86-64 with AVX2 only | 8 |
| `neon-sha2-x4` | AArch64 with the SHA-256 crypto extension | 4 streams |
| `neon-4` | other AArch64 | 4 |
| `simd128-4` | wasm32 built with `-C target-feature=+simd128` | 4 |
| `portable-8` | everything else | 8 |

All SIMD backends share one compression function written against a lane
trait; a backend only supplies the word arithmetic (e.g. `vpternlogd` for the
three-input booleans and `vprold` rotates on AVX-512, interlaced `sha256h`
streams on ARM). The `scalar`, `avx2`, `avx512`, and `neon` cargo features
pin one kernel at build time instead of detecting; a pinned kernel the CPU
lacks will fault at runtime, so pin only what your whole fleet supports.
(Note: pinning `neon` selects the integer NEON kernel, not the faster crypto
extension path.)

## Benchmarks

Full numbers, per-hardware analysis, and methodology live in
[BENCHMARKS.md](BENCHMARKS.md); the wasm/JS story is in
`benches/wasm/README.md`. The workload throughout is one Solana erasure
batch of Merkle leaves (64 leaves, ~1 KB each, 67,680 bytes), measured
against agave's current `sha2` path and Firedancer's batch kernels built
from source on the same machine.

![Batch SHA-256 throughput per core, MB/s: tape leads Firedancer on Zen 5 (5336 vs 4643), Zen 4 (3242 vs 3006) and Zen 3 (3264 vs 1476), sits level on Intel Granite Rapids (3923 vs 4006), and is the only batch kernel on Apple silicon at 4166; the serial sha2 baseline runs 610 to 1858](charts/speedup.svg)

Summarised:

| hardware | best backend | vs `sha2` | vs Firedancer |
|---|---|---|---|
| AMD Zen 5 | `avx512-2x16` | 2.9x | **~15% faster** |
| AMD Zen 4 | `shani-x4` | 2.1x | **~8% faster** |
| AMD Zen 3 | `shani-x4` | 2.1x | **~2.2x faster** |
| Intel EMR / GNR | `avx512-16` | 2.0-2.1x | ~1% slower (cycles) |
| AArch64 | `neon-sha2-x4` | 6.8x | no comparison |

The per-family AMD dispatch is measurement, not theory: the 2x16
interlace wins 15% on Zen 5's native 512-bit datapath, measures exactly
neutral on double-pumped Zen 4 (where the interlaced SHA-NI streams beat
every 16-lane kernel instead), and is deliberately not enabled anywhere
it has not been benched.

## What adopting it measures out to

All measured, per core, vs the serial `sha2` path both codebases use today.

In agave:

- shred recovery: **14-21% faster end-to-end** (measured integrated), and
  1.55x over its 7-thread rayon pool on one thread
- shred verify: **1.40x** the whole receive path (Zen 5); the Merkle
  rebuild in it is 54% of the cost and batches 2.13x
- leaf hashing: 2.0-2.9x on x86, 6.8x on ARM; interior joins 2.13x

In tape:

- blob commitment (slicer write, gateway/archive verified read):
  **6.85x on Apple silicon** (~51,000 groups/sec/core), **1.74x on
  Zen 4**, 1.12x over Firedancer on the same shape

Whole-process gains depend on each path's SHA-256 share; no number is
claimed beyond the paths above.

## Correctness

Output is bit-identical to any conforming SHA-256. The test suite gates
every backend differentially against the independent `sha2` crate across
message lengths spanning all block-boundary and padding edge cases, checks
every backend against every other, and validates the error-prone shuffle
machinery (the 8x8 and 16x16 transpose ladders and the `vpternlogd` control
bytes) against scalar models of the Intel intrinsic semantics. See `tests/`.

A kernel only *executes* where its instructions exist, so a green run on one
architecture compiles the others but cannot exercise them. The suite has been
run to completion on AMD Zen 3, AMD Zen 4, AMD Zen 5, Intel Emerald Rapids,
Intel Granite Rapids, and Apple M-series, so every backend has been validated
on real silicon of the vendor it targets. Run `cargo test --release` on a
representative machine before trusting a new build there;
`tests/differential.rs` prints which kernels a run actually exercised.

## License

Apache-2.0
