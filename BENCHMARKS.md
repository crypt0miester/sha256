# Benchmarks

Workload: one Solana erasure batch of Merkle leaves, 64 leaves
(32 data + 32 coding shreds), each a 26-byte domain-separation prefix plus
~1 KB of payload, 67,680 bytes per batch. Reported time is the best of
15 rounds of 50 iterations (`cargo bench --bench merkle_leaves`).

Every figure below is single-threaded: one thread, pinned with `taskset`
to one physical core of an otherwise idle machine, with that core's SMT
sibling also idle. The only exception is the SMT pairing section, which is
explicitly two threads on the two siblings of one core. Firedancer was
built from source on the same machine with its matching profile, so each
comparison is same-session; absolute numbers drift a few percent between
sessions on cloud hardware, so only same-session pairs are meaningful.

**AMD Zen 5 (EPYC 9B45, GCP c4d):**

```
sha2 (serial, current)          36.43 us/batch       1858 MB/s
tape shani (x4 stream)          19.82 us/batch       3414 MB/s
tape avx512 (16 lane)           14.36 us/batch       4638 MB/s
tape avx512 (2x16 interlace)    12.68 us/batch       5336 MB/s
firedancer avx (8 lane)         28.77 us/batch       2352 MB/s
firedancer avx512 (16 lane)     14.55 us/batch       4643 MB/s
```

Single-wave is kernel-for-kernel parity with Firedancer (14.36 vs 14.55,
spreads ~1%). The dispatched kernel is the `avx512-2x16` interlace: two
16-lane compressions with their rounds interlaced in one thread, filling
the dependency stalls a single wave leaves idle — the in-thread form of
the SMT observation below, and it beats the two-thread SMT pair (14.41)
on one thread. **2.87x over the serial baseline and ~15% faster than
Firedancer.** Enabled on AMD family 0x1a only: the two waves keep ~48
vectors live against 32 registers, and whether the hidden latency outpays
the spill traffic is a per-microarchitecture verdict — measured neutral
on Zen 4, and on Intel it wins the cycles but loses the clock to the
AVX-512 licence. Both keep the single wave; see those sections.

That is 5.05 M hashes/sec for ~1 KB messages, or **86 M SHA-256 block
compressions/sec** on one core. The block figure is the one worth
quoting, since hashes/sec depends entirely on message length.

**AMD Zen 4 (EPYC 9B14, GCP c3d):**

```
sha2 (serial, current)          43.71 us/batch       1548 MB/s
tape shani (x4 stream)          20.88 us/batch       3242 MB/s
tape avx512 (16 lane)           23.79 us/batch       2845 MB/s
firedancer avx (8 lane)         27.93 us/batch       2423 MB/s
firedancer avx512 (16 lane)     22.51 us/batch       3006 MB/s
```

After the interlaced rounds were unrolled, the SHA-NI streams beat every
16-lane kernel
on Zen 4 — ours, Firedancer's, and the two-thread avx512 SMT pair (23.84).
Dispatch therefore prefers `shani-x4` on Zen 4, detected as AMD family
0x19 with AVX-512 (the family spans Zen 3 and Zen 4, and Zen 3 has no
AVX-512; Zen 5 is family 0x1a and keeps `avx512-16`). That makes tape
**2.09x over the serial baseline and ~8% faster than Firedancer's best
kernel here** — Firedancer ships no SHA-NI batch kernel to route to.
Kernel for kernel at 16 lanes its AVX-512 still leads ours by ~5% on this
part; the shani flip wins the machine anyway. Firedancer was built
natively on this box (`linux_gcc_zen4`, GCC 15.2, real Genoa).

The 2x16 interlace was probed here too and measured exactly neutral
(23.69 vs 23.72 single-wave, same session, spreads under 3%): Zen 4
double-pumps 512-bit ops through a 256-bit datapath, so a single wave
already saturates the pipes and leaves no stalls for the second wave to
fill — the same reason its SMT gain is only 1.17x. Dispatch keeps
`shani-x4` on Zen 4.

**AMD Zen 3 (EPYC 7B13, GCP c2d):**

```
sha2 (serial, current)          42.66 us/batch       1587 MB/s
tape shani (x4 stream)          20.73 us/batch       3264 MB/s
tape avx2 (8 lane)              46.81 us/batch       1446 MB/s
firedancer avx (8 lane)         45.87 us/batch       1476 MB/s
```

**2.06x over the serial baseline, and ~2.2x faster than Firedancer.** The
reason is structural rather than clever: Firedancer ships no SHA-NI batch
kernel, so without AVX-512 it falls back to 8-lane AVX, which loses even to
serial SHA-NI. Note our own AVX2 row also loses to the baseline: on a
machine with SHA-NI and no AVX-512, integer multi-buffer is the wrong tool
and dispatch picks `shani-x4`.

**Intel Emerald Rapids (Xeon Platinum 8581C, GCP c4):**

First, the original run, kept because the correction below is the point. It
came from an earlier session on a c4 instance clocking about 15% lower than the
one used for everything after it, so compare its ratios and not its absolutes:

```
sha2 (serial, current)          49.18 us/batch       1376 MB/s
tape shani (x4 stream)          41.04 us/batch       1649 MB/s
tape avx512 (16 lane)           24.49 us/batch       2763 MB/s
firedancer avx (8 lane)         29.20 us/batch       2318 MB/s
firedancer avx512 (16 lane)     21.82 us/batch       3102 MB/s
```


SHA-NI interlacing is also much weaker on Intel, 1.19x over the serial
baseline versus 1.84x on Zen 5, which matches Intel not executing
`sha256rnds2` concurrently while AMD does. The interlace unroll that
bought 1.30x on Zen 3 is worth ~3% here (same-session paired, 32.61 to
31.62 us) for the same reason: a serialized unit is bound by round
latency, not by how fast the loop feeds it.

The `sha2` row is what agave's leaf hashing does today (one hash per leaf;
`sha2` uses SHA-NI hardware on all these CPUs). The Firedancer rows call
`fd_sha256_private_batch_{avx,avx512}` directly, the strongest batch
implementation we know of.

One methodological note: Firedancer's batch API takes each message as a
single contiguous buffer, so the harness concatenates `prefix || leaf` for
it outside the timed region. That reflects production accurately rather than
charitably. Firedancer's shredder writes the 26-byte Merkle prefix into
dead space at the tail of each shred's signature field
(`fd_shredder.c`), so its real per-shred prefix cost is a 26-byte write.
`Message::prefixed` reaches the same zero-copy result without requiring
writable dead space in front of the payload; that is a generality
difference in the API, not a performance edge, and no number is claimed
for it.

**Intel Granite Rapids (Xeon 6985P-C, GCP c4-standard-8-lssd):**

```
sha2 (serial, current)          36.87 us/batch       1835 MB/s
tape serial (1 lane)           167.08 us/batch        405 MB/s
tape portable (8 lane)         308.31 us/batch        220 MB/s
tape shani (x4 stream)          29.20 us/batch       2318 MB/s
tape smt pair (2x avx512)       17.35 us/batch       3900 MB/s
tape avx2 (8 lane)              44.11 us/batch       1534 MB/s
tape avx512 (16 lane)           17.25 us/batch       3923 MB/s
tape avx512 (2x16 interlace)    17.52 us/batch       3864 MB/s
firedancer avx (8 lane)         22.56 us/batch       2999 MB/s
firedancer avx512 (16 lane)     16.89 us/batch       4006 MB/s
```

2.14x over the serial baseline, best of six invocations. Unlike the c4
instance behind the Emerald Rapids numbers, this box is quiet: most rows
repeat to under 1% and the wall clock is usable here. Cycles anyway, six
paired rounds:

```
                     cyc/block-step   instr/block-step    IPC     GHz
tape avx512-16                 1048               2234   2.13    4.12
tape avx512-2x16               1004               2483   2.47    3.68
firedancer avx512              1034               2247   2.17    4.12
```

**Firedancer is 1.3% ahead on cycles, winning all six rounds at 0.3-0.4%
spread, not the 11% previously recorded on this part.** Wall clock puts it at
2.2%. Granite Rapids therefore behaves like Emerald Rapids, and the gap is a
percent or two Intel-wide rather than double digits.

One honest loose end. The earlier 11% came from a run with Firedancer at
16.03 us, about 6% faster than any build produced here. Its kernel was given
gcc 12.2 and gcc 14.2, and `-march=icelake-server` and `-march=graniterapids`,
and lands at 1034 to 1042 cycles per block step in every combination, on this
part and on Emerald Rapids alike. So the build is at least self-consistent,
but the older figure is not reproduced and its provenance is unknown. Compiler
tuning is dead as an explanation for the gap on both parts.

The 2x16 interlace again wins the cycles, 1004 against Firedancer's 1034, and
again cannot spend them: its clock averages 3.68 GHz against the single wave's
4.12, and it ends up 1.6% behind in wall time. Treat that margin loosely. This
kernel is the volatile one everywhere it has been measured, spreading 7% on
cycles and 3.7% within a single bench run, and its clock figure is a mean over
those noisy rounds rather than a stable operating point. What is solid is the
direction, and it is the same on both Intel parts: ahead on cycles, behind on
the clock, so dispatch keeps the single wave.

**AArch64 (Apple M-series):**

```
sha2 (serial, current)         110.95 us/batch        610 MB/s
tape neon (4 lane)              53.44 us/batch       1267 MB/s
tape neon-sha2 (x4 stream)      16.25 us/batch       4166 MB/s
```

(`sha2` is on its software path here.) The `neon-sha2` backend keeps four
hardware SHA-256 streams in flight to cover the crypto instructions'
latency, reaching 6.8x the serial baseline. Firedancer has no AArch64 batch
kernel, so there is nothing to compare against.

**SMT sibling pairing**: the entry points are stateless, so two
caller-pinned threads on the SMT siblings of one physical core can each
hash half a batch. What that is worth is entirely microarchitecture, one
pair per row, each measured same-session against its own single-thread
number:

```
AMD Zen 5        9.83 vs 14.93 us    1.52x
AMD Zen 4       20.16 vs 23.66 us    1.17x
Intel EMR       20.77 vs 23.88 us    1.15x
```

Zen 5's single thread is dependency-bound on its native 512-bit path, so
the sibling fills real gaps; Zen 4's double-pumped datapath leaves it less
room, and an early run that pinned both threads to one logical CPU
measured no gain at all, so verify the sibling pinning before trusting any
SMT number. Both caveats from the crate docs apply: the
sibling must actually be idle (this measures spare capacity, which a
loaded validator does not have), and the threads are the caller's, since
the library never spawns or pins. The `smt pair` row in `merkle_leaves`
is the reference harness.

**WebAssembly**: a `simd128-4` backend exists and is differentially gated
like the rest. Its honest scope, measured in Node and in a real Chrome tab:
for batched ~1 KB messages in a browser, WebCrypto's hardware-backed
`subtle.digest` wins (75 vs 80-90 us for this workload in Chrome), so use
WebCrypto there. Where the wasm backend earns its keep is small messages in
tight synchronous loops, address grinding being the canonical case: 4.55 M
candidates/s in a Chrome tab, 3.7x the web3.js stack and 1.6x `hash-wasm`.
PDA derivation is bound by the ed25519 curve check (94-99% of the cost), so
no SHA-256 implementation moves it. Numbers, methodology, and the full
where-it-wins/where-it-loses map are in `benches/wasm/README.md`.

## tape blob commitments

tape's own workload (`cargo bench --bench blob_group`): one blob commitment
is a Merkle tree over an erasure group of up to 20 slices of up to 4 KiB,
leaves hashed as `sha256("LEAF" || slice)`, pairs as
`sha256("LEFT" || left || "RIGHT" || right)`. The slicer builds it per
write, the gateway and archive nodes rebuild it per verified read, and
tape-sdk runs the same hashing on the serial `sha2` crate today.

Apple M-series, min of 15 rounds, batch rows asserted equal to serial:

```
leaves sha2 (serial, current)    128.92 us/group       636 MB/s
leaves tape hash_many_prefixed    18.72 us/group      4381 MB/s
joins sha2 (serial, current)       4.19 us/group       366 MB/s
joins tape hash_many               0.72 us/group      2122 MB/s

leaf level 6.9x, join level 5.8x, whole commitment 6.85x
```

That is ~51,000 slice groups per second on one core, about 4.2 GB/s of
blob payload. The 20-slice group also happens to divide evenly into the
4-stream ARM backend, so no lane sits idle.

On x86 the group does not divide the 16-lane kernel, and that is where
dispatch's tail routing earns its keep: a remainder of 1 to 12 messages
goes to the SHA-NI streams instead of a mostly-empty wide group. Measured
on Zen 4 (EPYC 9B14, Firedancer built natively on the same box,
same-session):

```
leaves sha2 (serial, current)     50.36 us/group      1628 MB/s
leaves tape hash_many_prefixed    28.78 us/group      2850 MB/s
leaves fd avx512 (16 lane)        32.43 us/group      2528 MB/s
joins sha2 (serial, current)       1.82 us/group       844 MB/s
joins tape hash_many               1.28 us/group      1199 MB/s
joins fd avx (8 lane)              1.32 us/group      1158 MB/s

whole commitment: tape 30.05 us vs firedancer 33.75 = 1.12x faster
```

Without the routing the leaf level ran at 42.7 us (a 4-message tail
occupying a 16-lane group), so the routing alone is worth 1.48x on this
shape. Firedancer pays the full tail tax: its only option for a partial
batch is the same wide kernel with idle lanes, and it ships no SHA-NI
kernel to route to.

## Summary

| hardware | best backend | vs `sha2` | vs Firedancer |
|---|---|---|---|
| AMD Zen 5 | `avx512-2x16` | 2.9x | **~15% faster** |
| AMD Zen 4 | `shani-x4` | 2.1x | **~8% faster** |
| AMD Zen 3 | `shani-x4` | 2.1x | **~2.2x faster** |
| Intel EMR / GNR | `avx512-16` | 2.0-2.1x | ~1% slower (cycles) |
| AArch64 | `neon-sha2-x4` | 6.8x | no comparison |

In agave integration measurements, switching shred recovery's leaf hashing
to this crate made recovery 14-21% faster end-to-end, and the
single-threaded SIMD path outperformed the existing 7-thread rayon pool by
1.55x, returning six cores to the rest of the validator.

## Running the Firedancer comparison

To include the Firedancer reference rows in `merkle_leaves` or
`blob_group`, point `FD_LIB_DIR` at a Firedancer build's lib directory and
enable the bench-only feature:

```sh
FD_LIB_DIR=/path/to/firedancer/build/linux/gcc/zen5/lib \
  cargo bench --bench merkle_leaves --features firedancer-bench
```

`firedancer-bench` links the 8-lane kernel only. Add the AVX-512 row with
`--features firedancer-bench-avx512`, but only against a Firedancer build
whose machine profile enabled AVX-512. The symbol does not exist otherwise
and the bench will fail to link rather than fall back.

Nothing from Firedancer is vendored into or linked against the library
itself.
