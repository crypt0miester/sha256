//! SHA-256 compression, written once against `Lanes`
//!
//! FIPS 180-4 section 6.2. Backends supply only the word arithmetic.

use crate::lanes::Lanes;

/// FIPS 180-4 section 4.2.2 round constants
#[rustfmt::skip]
pub(crate) const K: [u32; 64] = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
];

/// FIPS 180-4 section 5.3.3 initial hash value
pub(crate) const H0: [u32; 8] = [
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
];

// Routed through `xor3` so three-input boolean units collapse each to one op.
#[inline(always)]
pub(crate) fn big_sigma0<L: Lanes>(x: L) -> L {
    x.rotr::<2>().xor3(x.rotr::<13>(), x.rotr::<22>())
}

#[inline(always)]
pub(crate) fn big_sigma1<L: Lanes>(x: L) -> L {
    x.rotr::<6>().xor3(x.rotr::<11>(), x.rotr::<25>())
}

#[inline(always)]
pub(crate) fn small_sigma0<L: Lanes>(x: L) -> L {
    x.rotr::<7>().xor3(x.rotr::<18>(), x.shr::<3>())
}

#[inline(always)]
pub(crate) fn small_sigma1<L: Lanes>(x: L) -> L {
    x.rotr::<17>().xor3(x.rotr::<19>(), x.shr::<10>())
}

#[inline(always)]
fn ch<L: Lanes>(x: L, y: L, z: L) -> L {
    x.ch(y, z)
}

#[inline(always)]
fn maj<L: Lanes>(x: L, y: L, z: L) -> L {
    x.maj(y, z)
}

/// Compresses one 64-byte block per lane into `state`
///
/// `w[j]` is word `j` of every lane's block, as produced by `transpose`.
#[inline(always)]
pub(crate) fn compress<L: Lanes>(state: &mut [L; 8], mut w: [L; 16]) {
    // Rolling 16-word window.
    let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut h] = *state;

    macro_rules! round {
        ($a:ident, $b:ident, $c:ident, $d:ident,
         $e:ident, $f:ident, $g:ident, $h:ident, $w:expr, $k:expr) => {{
            let t1 = $w
                .add(L::splat($k))
                .add($h.add(big_sigma1($e)).add(ch($e, $f, $g)));
            let t2 = big_sigma0($a).add(maj($a, $b, $c));
            $d = $d.add(t1);
            $h = t1.add(t2);
        }};
    }

    // `direct` reads a block word (rounds 0..16); `extend` advances the window
    // in place (rounds 16..64), w[t] = s1(w[t-2]) + w[t-7] + s0(w[t-15]) + w[t-16].
    macro_rules! direct {
        ($i:expr) => {
            w[$i]
        };
    }
    macro_rules! extend {
        ($i:expr) => {{
            let i = $i;
            let s1 = small_sigma1(w[(i + 14) & 15]);
            let s0 = small_sigma0(w[(i + 1) & 15]);
            w[i] = s1.add(w[(i + 9) & 15]).add(s0).add(w[i]);
            w[i]
        }};
    }

    // Rotating the working variables by name avoids the eight register moves a
    // naive `h = g; g = f; ...` costs per round.
    //
    // Flat expansion is what keeps the window in registers.
    macro_rules! rounds16 {
        ($base:expr, $sched:ident) => {
            round!(a, b, c, d, e, f, g, h, $sched!(0), K[$base]);
            round!(h, a, b, c, d, e, f, g, $sched!(1), K[$base + 1]);
            round!(g, h, a, b, c, d, e, f, $sched!(2), K[$base + 2]);
            round!(f, g, h, a, b, c, d, e, $sched!(3), K[$base + 3]);
            round!(e, f, g, h, a, b, c, d, $sched!(4), K[$base + 4]);
            round!(d, e, f, g, h, a, b, c, $sched!(5), K[$base + 5]);
            round!(c, d, e, f, g, h, a, b, $sched!(6), K[$base + 6]);
            round!(b, c, d, e, f, g, h, a, $sched!(7), K[$base + 7]);
            round!(a, b, c, d, e, f, g, h, $sched!(8), K[$base + 8]);
            round!(h, a, b, c, d, e, f, g, $sched!(9), K[$base + 9]);
            round!(g, h, a, b, c, d, e, f, $sched!(10), K[$base + 10]);
            round!(f, g, h, a, b, c, d, e, $sched!(11), K[$base + 11]);
            round!(e, f, g, h, a, b, c, d, $sched!(12), K[$base + 12]);
            round!(d, e, f, g, h, a, b, c, $sched!(13), K[$base + 13]);
            round!(c, d, e, f, g, h, a, b, $sched!(14), K[$base + 14]);
            round!(b, c, d, e, f, g, h, a, $sched!(15), K[$base + 15]);
        };
    }

    macro_rules! rounds16_looped {
        ($base:expr, $sched:ident) => {
            for i in (0..16).step_by(8) {
                round!(a, b, c, d, e, f, g, h, $sched!(i), K[$base + i]);
                round!(h, a, b, c, d, e, f, g, $sched!(i + 1), K[$base + i + 1]);
                round!(g, h, a, b, c, d, e, f, $sched!(i + 2), K[$base + i + 2]);
                round!(f, g, h, a, b, c, d, e, $sched!(i + 3), K[$base + i + 3]);
                round!(e, f, g, h, a, b, c, d, $sched!(i + 4), K[$base + i + 4]);
                round!(d, e, f, g, h, a, b, c, $sched!(i + 5), K[$base + i + 5]);
                round!(c, d, e, f, g, h, a, b, $sched!(i + 6), K[$base + i + 6]);
                round!(b, c, d, e, f, g, h, a, $sched!(i + 7), K[$base + i + 7]);
            }
        };
    }

    if L::FLAT_ROUNDS {
        rounds16!(0, direct);
        rounds16!(16, extend);
        rounds16!(32, extend);
        rounds16!(48, extend);
    } else {
        rounds16_looped!(0, direct);
        rounds16_looped!(16, extend);
        rounds16_looped!(32, extend);
        rounds16_looped!(48, extend);
    }

    state[0] = state[0].add(a);
    state[1] = state[1].add(b);
    state[2] = state[2].add(c);
    state[3] = state[3].add(d);
    state[4] = state[4].add(e);
    state[5] = state[5].add(f);
    state[6] = state[6].add(g);
    state[7] = state[7].add(h);
}
