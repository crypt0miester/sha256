//! Interlace: two 16-lane AVX-512 waves woven into one thread
//!
//! A single 16-lane compression is one dependency chain: at ~2 IPC the
//! vector ports sit a third idle waiting on it, which is exactly the slack
//! an SMT sibling harvests for 1.5x. This kernel claims that slack from
//! within the thread instead: two independent waves with their rounds
//! interlaced, so wave B issues while wave A's chain stalls.
//!
//! The 48 live vectors against 32 ZMM guarantee spills, and whether the
//! latency hidden outweighs the spill traffic is a per-microarchitecture
//! verdict, not a principle.

use crate::{
    avx512::Avx512,
    batch::{stage_prefix_block, write_digest, Message, Shape, BLOCK, MAX_LANES},
    core::{big_sigma0, big_sigma1, small_sigma0, small_sigma1, H0, K},
    lanes::Lanes,
};

const WAVE: usize = 16;
pub(crate) const WIDTH: usize = 32;

// The drivers stage WIDTH messages at a time; a wider kernel would silently
// overrun their staging arrays, so fail the build instead.
const _: () = assert!(WIDTH <= crate::batch::MAX_WIDTH);

/// Compresses one block of each wave, rounds alternating A/B
///
/// Same structure as `core::compress`, twice over: the round and schedule
/// macros are restated because they capture the local working variables,
/// but the sigma arithmetic is shared with the single-wave core. The A/B
/// alternation is textual so the scheduler sees both chains at once, and
/// `rounds16x2` expands flat for the same reason `core`'s `rounds16` does:
/// literal indices keep both schedule windows in registers.
#[inline(always)]
fn compress2<L: Lanes>(sa: &mut [L; 8], sb: &mut [L; 8], mut wa: [L; 16], mut wb: [L; 16]) {
    let [mut a0, mut b0, mut c0, mut d0, mut e0, mut f0, mut g0, mut h0] = *sa;
    let [mut a1, mut b1, mut c1, mut d1, mut e1, mut f1, mut g1, mut h1] = *sb;

    macro_rules! round {
        ($a:ident, $b:ident, $c:ident, $d:ident,
         $e:ident, $f:ident, $g:ident, $h:ident, $w:expr, $k:expr) => {{
            let t1 = $w
                .add(L::splat($k))
                .add($h.add(big_sigma1($e)).add($e.ch($f, $g)));
            let t2 = big_sigma0($a).add($a.maj($b, $c));
            $d = $d.add(t1);
            $h = t1.add(t2);
        }};
    }
    // `direct` reads a block word (rounds 0..16); `extend` advances the
    // named wave's window in place (rounds 16..64).
    macro_rules! direct {
        ($w:ident, $i:expr) => {
            $w[$i]
        };
    }
    macro_rules! extend {
        ($w:ident, $i:expr) => {{
            const I: usize = $i;
            let s1 = small_sigma1($w[(I + 14) & 15]);
            let s0 = small_sigma0($w[(I + 1) & 15]);
            $w[I] = s1.add($w[(I + 9) & 15]).add(s0).add($w[I]);
            $w[I]
        }};
    }
    // One round per line on purpose; rustfmt would fold the listing into a
    // vertical argument stack, which buries the A/B alternation the layout
    // exists to show.
    #[rustfmt::skip]
    macro_rules! rounds16x2 {
        ($base:expr, $sched:ident) => {
            round!(a0,b0,c0,d0,e0,f0,g0,h0, $sched!(wa, 0), K[$base + 0]);
            round!(a1,b1,c1,d1,e1,f1,g1,h1, $sched!(wb, 0), K[$base + 0]);
            round!(h0,a0,b0,c0,d0,e0,f0,g0, $sched!(wa, 1), K[$base + 1]);
            round!(h1,a1,b1,c1,d1,e1,f1,g1, $sched!(wb, 1), K[$base + 1]);
            round!(g0,h0,a0,b0,c0,d0,e0,f0, $sched!(wa, 2), K[$base + 2]);
            round!(g1,h1,a1,b1,c1,d1,e1,f1, $sched!(wb, 2), K[$base + 2]);
            round!(f0,g0,h0,a0,b0,c0,d0,e0, $sched!(wa, 3), K[$base + 3]);
            round!(f1,g1,h1,a1,b1,c1,d1,e1, $sched!(wb, 3), K[$base + 3]);
            round!(e0,f0,g0,h0,a0,b0,c0,d0, $sched!(wa, 4), K[$base + 4]);
            round!(e1,f1,g1,h1,a1,b1,c1,d1, $sched!(wb, 4), K[$base + 4]);
            round!(d0,e0,f0,g0,h0,a0,b0,c0, $sched!(wa, 5), K[$base + 5]);
            round!(d1,e1,f1,g1,h1,a1,b1,c1, $sched!(wb, 5), K[$base + 5]);
            round!(c0,d0,e0,f0,g0,h0,a0,b0, $sched!(wa, 6), K[$base + 6]);
            round!(c1,d1,e1,f1,g1,h1,a1,b1, $sched!(wb, 6), K[$base + 6]);
            round!(b0,c0,d0,e0,f0,g0,h0,a0, $sched!(wa, 7), K[$base + 7]);
            round!(b1,c1,d1,e1,f1,g1,h1,a1, $sched!(wb, 7), K[$base + 7]);
            round!(a0,b0,c0,d0,e0,f0,g0,h0, $sched!(wa, 8), K[$base + 8]);
            round!(a1,b1,c1,d1,e1,f1,g1,h1, $sched!(wb, 8), K[$base + 8]);
            round!(h0,a0,b0,c0,d0,e0,f0,g0, $sched!(wa, 9), K[$base + 9]);
            round!(h1,a1,b1,c1,d1,e1,f1,g1, $sched!(wb, 9), K[$base + 9]);
            round!(g0,h0,a0,b0,c0,d0,e0,f0, $sched!(wa, 10), K[$base + 10]);
            round!(g1,h1,a1,b1,c1,d1,e1,f1, $sched!(wb, 10), K[$base + 10]);
            round!(f0,g0,h0,a0,b0,c0,d0,e0, $sched!(wa, 11), K[$base + 11]);
            round!(f1,g1,h1,a1,b1,c1,d1,e1, $sched!(wb, 11), K[$base + 11]);
            round!(e0,f0,g0,h0,a0,b0,c0,d0, $sched!(wa, 12), K[$base + 12]);
            round!(e1,f1,g1,h1,a1,b1,c1,d1, $sched!(wb, 12), K[$base + 12]);
            round!(d0,e0,f0,g0,h0,a0,b0,c0, $sched!(wa, 13), K[$base + 13]);
            round!(d1,e1,f1,g1,h1,a1,b1,c1, $sched!(wb, 13), K[$base + 13]);
            round!(c0,d0,e0,f0,g0,h0,a0,b0, $sched!(wa, 14), K[$base + 14]);
            round!(c1,d1,e1,f1,g1,h1,a1,b1, $sched!(wb, 14), K[$base + 14]);
            round!(b0,c0,d0,e0,f0,g0,h0,a0, $sched!(wa, 15), K[$base + 15]);
            round!(b1,c1,d1,e1,f1,g1,h1,a1, $sched!(wb, 15), K[$base + 15]);
        };
    }

    rounds16x2!(0, direct);
    rounds16x2!(16, extend);
    rounds16x2!(32, extend);
    rounds16x2!(48, extend);

    sa[0] = sa[0].add(a0);
    sa[1] = sa[1].add(b0);
    sa[2] = sa[2].add(c0);
    sa[3] = sa[3].add(d0);
    sa[4] = sa[4].add(e0);
    sa[5] = sa[5].add(f0);
    sa[6] = sa[6].add(g0);
    sa[7] = sa[7].add(h0);
    sb[0] = sb[0].add(a1);
    sb[1] = sb[1].add(b1);
    sb[2] = sb[2].add(c1);
    sb[3] = sb[3].add(d1);
    sb[4] = sb[4].add(e1);
    sb[5] = sb[5].add(f1);
    sb[6] = sb[6].add(g1);
    sb[7] = sb[7].add(h1);
}

/// Expands one wave's block into all 64 `w + K` round inputs, in memory
///
/// The rolling window lives in registers only for the length of this pass,
/// where no state vector is live, so the peak is 16 vectors instead of 48.
///
/// Expanded flat with literal indices for the same reason `rounds16x2` is: a
/// `for` loop makes the window indices runtime values, and an array indexed
/// dynamically cannot live in registers at all. Rolled, this function spills
/// the whole window to the stack and reloads it per access.
/// Both waves' schedules, A/B alternating for the same reason the rounds do
///
/// Scheduling one wave and then the other leaves each pass with only its own
/// dependency chain to issue against, which gives back exactly the overlap
/// the interlace exists to buy. Alternating keeps two chains live here too.
/// No state vector is live yet, so the 32 windows have the register file to
/// themselves.
#[inline(always)]
#[rustfmt::skip]
fn schedule2(
    wa: &mut [Avx512; 16],
    wb: &mut [Avx512; 16],
    ka: &mut [[u32; WAVE]; 64],
    kb: &mut [[u32; WAVE]; 64],
) {
    macro_rules! emit {
        ($buf:ident, $t:expr, $v:expr) => {
            $v.add(Avx512::splat(K[$t])).store(&mut $buf[$t])
        };
    }
    // Same rolling recurrence as `core::compress`'s `extend`: on entry to
    // step `t`, w[i] holds w[t-16], w[(i+1)&15] w[t-15], w[(i+9)&15] w[t-7]
    // and w[(i+14)&15] w[t-2].
    macro_rules! ext {
        ($w:ident, $buf:ident, $t:expr, $i:expr) => {{
            let s1 = small_sigma1($w[($i + 14) & 15]);
            let s0 = small_sigma0($w[($i + 1) & 15]);
            $w[$i] = s1.add($w[($i + 9) & 15]).add(s0).add($w[$i]);
            emit!($buf, $t, $w[$i]);
        }};
    }
    macro_rules! sched16x2 {
        (direct, $base:expr) => {
            emit!(ka, $base + 0,  wa[0]);  emit!(kb, $base + 0,  wb[0]);
            emit!(ka, $base + 1,  wa[1]);  emit!(kb, $base + 1,  wb[1]);
            emit!(ka, $base + 2,  wa[2]);  emit!(kb, $base + 2,  wb[2]);
            emit!(ka, $base + 3,  wa[3]);  emit!(kb, $base + 3,  wb[3]);
            emit!(ka, $base + 4,  wa[4]);  emit!(kb, $base + 4,  wb[4]);
            emit!(ka, $base + 5,  wa[5]);  emit!(kb, $base + 5,  wb[5]);
            emit!(ka, $base + 6,  wa[6]);  emit!(kb, $base + 6,  wb[6]);
            emit!(ka, $base + 7,  wa[7]);  emit!(kb, $base + 7,  wb[7]);
            emit!(ka, $base + 8,  wa[8]);  emit!(kb, $base + 8,  wb[8]);
            emit!(ka, $base + 9,  wa[9]);  emit!(kb, $base + 9,  wb[9]);
            emit!(ka, $base + 10, wa[10]); emit!(kb, $base + 10, wb[10]);
            emit!(ka, $base + 11, wa[11]); emit!(kb, $base + 11, wb[11]);
            emit!(ka, $base + 12, wa[12]); emit!(kb, $base + 12, wb[12]);
            emit!(ka, $base + 13, wa[13]); emit!(kb, $base + 13, wb[13]);
            emit!(ka, $base + 14, wa[14]); emit!(kb, $base + 14, wb[14]);
            emit!(ka, $base + 15, wa[15]); emit!(kb, $base + 15, wb[15]);
        };
        (extend, $base:expr) => {
            ext!(wa, ka, $base + 0,  0);  ext!(wb, kb, $base + 0,  0);
            ext!(wa, ka, $base + 1,  1);  ext!(wb, kb, $base + 1,  1);
            ext!(wa, ka, $base + 2,  2);  ext!(wb, kb, $base + 2,  2);
            ext!(wa, ka, $base + 3,  3);  ext!(wb, kb, $base + 3,  3);
            ext!(wa, ka, $base + 4,  4);  ext!(wb, kb, $base + 4,  4);
            ext!(wa, ka, $base + 5,  5);  ext!(wb, kb, $base + 5,  5);
            ext!(wa, ka, $base + 6,  6);  ext!(wb, kb, $base + 6,  6);
            ext!(wa, ka, $base + 7,  7);  ext!(wb, kb, $base + 7,  7);
            ext!(wa, ka, $base + 8,  8);  ext!(wb, kb, $base + 8,  8);
            ext!(wa, ka, $base + 9,  9);  ext!(wb, kb, $base + 9,  9);
            ext!(wa, ka, $base + 10, 10); ext!(wb, kb, $base + 10, 10);
            ext!(wa, ka, $base + 11, 11); ext!(wb, kb, $base + 11, 11);
            ext!(wa, ka, $base + 12, 12); ext!(wb, kb, $base + 12, 12);
            ext!(wa, ka, $base + 13, 13); ext!(wb, kb, $base + 13, 13);
            ext!(wa, ka, $base + 14, 14); ext!(wb, kb, $base + 14, 14);
            ext!(wa, ka, $base + 15, 15); ext!(wb, kb, $base + 15, 15);
        };
    }

    sched16x2!(direct, 0);
    sched16x2!(extend, 16);
    sched16x2!(extend, 32);
    sched16x2!(extend, 48);
}

/// The interlace with both schedules staged through memory
///
/// `compress2` keeps 48 vectors live against 32 ZMM, so the allocator spills
/// whatever it must wherever it must — 722 slots in the emitted kernel. This
/// variant makes the spill deliberate instead: each wave's schedule is
/// expanded into a stack buffer up front, leaving only the sixteen state
/// vectors live through the rounds, and `w + K` returns as a folded memory
/// operand on the round's `vpaddd`, which costs no instruction at all. The
/// two schedules run one after the other rather than interlaced, since each
/// alone wants the full sixteen-vector window.
#[inline(always)]
#[rustfmt::skip]
fn compress2_sched(
    sa: &mut [Avx512; 8],
    sb: &mut [Avx512; 8],
    mut wa: [Avx512; 16],
    mut wb: [Avx512; 16],
    ka: &mut [[u32; WAVE]; 64],
    kb: &mut [[u32; WAVE]; 64],
) {
    schedule2(&mut wa, &mut wb, ka, kb);

    let [mut a0, mut b0, mut c0, mut d0, mut e0, mut f0, mut g0, mut h0] = *sa;
    let [mut a1, mut b1, mut c1, mut d1, mut e1, mut f1, mut g1, mut h1] = *sb;

    macro_rules! round {
        ($a:ident, $b:ident, $c:ident, $d:ident,
         $e:ident, $f:ident, $g:ident, $h:ident, $buf:ident, $t:expr) => {{
            let wk = Avx512::load(&$buf[$t]);
            let t1 = wk.add($h.add(big_sigma1($e)).add($e.ch($f, $g)));
            let t2 = big_sigma0($a).add($a.maj($b, $c));
            $d = $d.add(t1);
            $h = t1.add(t2);
        }};
    }
    macro_rules! rounds16x2 {
        ($base:expr) => {
            round!(a0,b0,c0,d0,e0,f0,g0,h0, ka, $base + 0);
            round!(a1,b1,c1,d1,e1,f1,g1,h1, kb, $base + 0);
            round!(h0,a0,b0,c0,d0,e0,f0,g0, ka, $base + 1);
            round!(h1,a1,b1,c1,d1,e1,f1,g1, kb, $base + 1);
            round!(g0,h0,a0,b0,c0,d0,e0,f0, ka, $base + 2);
            round!(g1,h1,a1,b1,c1,d1,e1,f1, kb, $base + 2);
            round!(f0,g0,h0,a0,b0,c0,d0,e0, ka, $base + 3);
            round!(f1,g1,h1,a1,b1,c1,d1,e1, kb, $base + 3);
            round!(e0,f0,g0,h0,a0,b0,c0,d0, ka, $base + 4);
            round!(e1,f1,g1,h1,a1,b1,c1,d1, kb, $base + 4);
            round!(d0,e0,f0,g0,h0,a0,b0,c0, ka, $base + 5);
            round!(d1,e1,f1,g1,h1,a1,b1,c1, kb, $base + 5);
            round!(c0,d0,e0,f0,g0,h0,a0,b0, ka, $base + 6);
            round!(c1,d1,e1,f1,g1,h1,a1,b1, kb, $base + 6);
            round!(b0,c0,d0,e0,f0,g0,h0,a0, ka, $base + 7);
            round!(b1,c1,d1,e1,f1,g1,h1,a1, kb, $base + 7);
            round!(a0,b0,c0,d0,e0,f0,g0,h0, ka, $base + 8);
            round!(a1,b1,c1,d1,e1,f1,g1,h1, kb, $base + 8);
            round!(h0,a0,b0,c0,d0,e0,f0,g0, ka, $base + 9);
            round!(h1,a1,b1,c1,d1,e1,f1,g1, kb, $base + 9);
            round!(g0,h0,a0,b0,c0,d0,e0,f0, ka, $base + 10);
            round!(g1,h1,a1,b1,c1,d1,e1,f1, kb, $base + 10);
            round!(f0,g0,h0,a0,b0,c0,d0,e0, ka, $base + 11);
            round!(f1,g1,h1,a1,b1,c1,d1,e1, kb, $base + 11);
            round!(e0,f0,g0,h0,a0,b0,c0,d0, ka, $base + 12);
            round!(e1,f1,g1,h1,a1,b1,c1,d1, kb, $base + 12);
            round!(d0,e0,f0,g0,h0,a0,b0,c0, ka, $base + 13);
            round!(d1,e1,f1,g1,h1,a1,b1,c1, kb, $base + 13);
            round!(c0,d0,e0,f0,g0,h0,a0,b0, ka, $base + 14);
            round!(c1,d1,e1,f1,g1,h1,a1,b1, kb, $base + 14);
            round!(b0,c0,d0,e0,f0,g0,h0,a0, ka, $base + 15);
            round!(b1,c1,d1,e1,f1,g1,h1,a1, kb, $base + 15);
        };
    }

    rounds16x2!(0);
    rounds16x2!(16);
    rounds16x2!(32);
    rounds16x2!(48);

    sa[0] = sa[0].add(a0); sa[1] = sa[1].add(b0);
    sa[2] = sa[2].add(c0); sa[3] = sa[3].add(d0);
    sa[4] = sa[4].add(e0); sa[5] = sa[5].add(f0);
    sa[6] = sa[6].add(g0); sa[7] = sa[7].add(h0);
    sb[0] = sb[0].add(a1); sb[1] = sb[1].add(b1);
    sb[2] = sb[2].add(c1); sb[3] = sb[3].add(d1);
    sb[4] = sb[4].add(e1); sb[5] = sb[5].add(f1);
    sb[6] = sb[6].add(g1); sb[7] = sb[7].add(h1);
}

#[inline(always)]
fn write_digests<L: Lanes>(state: &[L; 8], out: &mut [[u8; 32]]) {
    let mut unpacked = [[0u32; MAX_LANES]; 8];
    for (i, s) in state.iter().enumerate() {
        s.store(&mut unpacked[i][..L::N]);
    }
    for (lane, o) in out.iter_mut().enumerate() {
        write_digest(&unpacked, lane, o);
    }
}

/// Hashes exactly 32 equal-block-count messages, 16 per wave
///
/// `SCHED` picks how the two schedules are held: in registers, letting the
/// allocator spill what will not fit, or expanded into stack buffers up
/// front. Same digests either way; the choice is measured, not principled.
#[inline(always)]
unsafe fn fused<const SCHED: bool>(msgs: &[Message<'_>], out: &mut [[u8; 32]], blocks: usize) {
    debug_assert_eq!(msgs.len(), WIDTH);

    let mut sa = H0.map(Avx512::splat);
    let mut sb = H0.map(Avx512::splat);
    let mut ka = [[0u32; WAVE]; 64];
    let mut kb = [[0u32; WAVE]; 64];

    let shape = Shape::of(msgs);
    let mut bases = [std::ptr::null::<u8>(); WIDTH];
    for (b, m) in bases.iter_mut().zip(msgs) {
        *b = m.body.as_ptr();
    }

    let mut staging = [[0u8; BLOCK]; WIDTH];
    let mut srcs = [std::ptr::null::<u8>(); WIDTH];

    let staged0 = stage_prefix_block(msgs, &shape, &mut staging);
    for k in 0..blocks {
        if shape.same && k >= shape.k_lo && k < shape.k_hi {
            let off = k * BLOCK - shape.plen;
            for (s, b) in srcs.iter_mut().zip(bases.iter()) {
                // SAFETY: the interior bound documented on `Shape`.
                *s = b.add(off);
            }
        } else if k == 0 && staged0 {
            for (s, b) in srcs.iter_mut().zip(staging.iter()) {
                *s = b.as_ptr();
            }
        } else {
            for (lane, m) in msgs.iter().enumerate() {
                if m.block_is_interior(k) {
                    srcs[lane] = m.interior_block(k).as_ptr();
                } else {
                    m.fill_block(k, &mut staging[lane]);
                    srcs[lane] = staging[lane].as_ptr();
                }
            }
        }
        // SAFETY: every source points at a full 64-byte block, either
        // interior to a borrowed body or freshly staged above.
        let wa = Avx512::transpose(&srcs[..WAVE], WAVE);
        let wb = Avx512::transpose(&srcs[WAVE..], WAVE);
        if SCHED {
            compress2_sched(&mut sa, &mut sb, wa, wb, &mut ka, &mut kb);
        } else {
            compress2(&mut sa, &mut sb, wa, wb);
        }
    }

    write_digests(&sa, &mut out[..WAVE]);
    write_digests(&sb, &mut out[WAVE..]);
}

/// Hashes up to 32 messages: the full interlace when full and uniform,
/// else split across the single-wave kernel
///
/// # Safety
///
/// The running CPU must support AVX-512F and AVX-512BW.
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn group2(msgs: &[Message<'_>], out: &mut [[u8; 32]]) {
    debug_assert!(msgs.len() <= WIDTH);
    debug_assert_eq!(msgs.len(), out.len());
    let blocks = match msgs.first() {
        Some(m) => m.blocks(),
        None => return,
    };
    if msgs.len() == WIDTH && msgs.iter().all(|m| m.blocks() == blocks) {
        fused::<false>(msgs, out, blocks);
    } else {
        // Ragged or partial: 16-lane groups through the shared driver; only
        // the cross-wave overlap is lost.
        crate::batch::drive(WAVE, crate::avx512::group, msgs, out);
    }
}

/// `group2` with the schedules staged through memory
///
/// # Safety
///
/// The running CPU must support AVX-512F and AVX-512BW.
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn group2_sched(msgs: &[Message<'_>], out: &mut [[u8; 32]]) {
    debug_assert!(msgs.len() <= WIDTH);
    debug_assert_eq!(msgs.len(), out.len());
    let blocks = match msgs.first() {
        Some(m) => m.blocks(),
        None => return,
    };
    if msgs.len() == WIDTH && msgs.iter().all(|m| m.blocks() == blocks) {
        fused::<true>(msgs, out, blocks);
    } else {
        crate::batch::drive(WAVE, crate::avx512::group, msgs, out);
    }
}
