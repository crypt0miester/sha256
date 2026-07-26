//! Validates the transpose ladders against a scalar model of the intrinsics
//!
//! The AVX kernels cannot execute on an AArch64 host, and Rosetta has no AVX,
//! so they compile but never run during development. The ladders are the most
//! error-prone part: everything else maps an arithmetic operation onto its
//! instruction one-to-one, but a ladder is pure index choreography that is
//! easy to get subtly wrong and that yields a wrong Merkle root, not a crash.
//!
//! Modelling each intrinsic scalar-side from the Intel semantics lets the
//! ladders be checked on any host, ARM included. This proves the algorithm,
//! not the compiled kernel, so it complements rather than replaces running
//! tests/differential.rs on real x86.

type V = [u32; 8];

/// `_mm256_unpacklo_epi32`: interleaves the low two u32 per 128-bit lane
fn unpacklo_epi32(a: V, b: V) -> V {
    let mut o = [0u32; 8];
    for l in 0..2 {
        let base = l * 4;
        o[base] = a[base];
        o[base + 1] = b[base];
        o[base + 2] = a[base + 1];
        o[base + 3] = b[base + 1];
    }
    o
}

/// `_mm256_unpackhi_epi32`: interleaves the high two u32 per 128-bit lane
fn unpackhi_epi32(a: V, b: V) -> V {
    let mut o = [0u32; 8];
    for l in 0..2 {
        let base = l * 4;
        o[base] = a[base + 2];
        o[base + 1] = b[base + 2];
        o[base + 2] = a[base + 3];
        o[base + 3] = b[base + 3];
    }
    o
}

/// `_mm256_unpacklo_epi64`: same but on 64-bit elements, two u32 at a time
fn unpacklo_epi64(a: V, b: V) -> V {
    let mut o = [0u32; 8];
    for l in 0..2 {
        let base = l * 4;
        o[base] = a[base];
        o[base + 1] = a[base + 1];
        o[base + 2] = b[base];
        o[base + 3] = b[base + 1];
    }
    o
}

/// `_mm256_unpackhi_epi64`: the same for the high 64-bit element
fn unpackhi_epi64(a: V, b: V) -> V {
    let mut o = [0u32; 8];
    for l in 0..2 {
        let base = l * 4;
        o[base] = a[base + 2];
        o[base + 1] = a[base + 3];
        o[base + 2] = b[base + 2];
        o[base + 3] = b[base + 3];
    }
    o
}

/// `_mm256_permute2x128_si256`, for the two control bytes the ladder uses
///
/// 0x20 selects (a low, b low), 0x31 selects (a high, b high).
fn permute2x128<const CTRL: u8>(a: V, b: V) -> V {
    let pick = |sel: u8| -> [u32; 4] {
        let src = if sel & 0x02 == 0 { a } else { b };
        let half = (sel & 0x01) as usize;
        [
            src[half * 4],
            src[half * 4 + 1],
            src[half * 4 + 2],
            src[half * 4 + 3],
        ]
    };
    let lo = pick(CTRL & 0x0f);
    let hi = pick((CTRL >> 4) & 0x0f);
    let mut o = [0u32; 8];
    o[..4].copy_from_slice(&lo);
    o[4..].copy_from_slice(&hi);
    o
}

/// The exact ladder from `transpose8` in src/avx2.rs, over the model above
fn transpose8(r: [V; 8]) -> [V; 8] {
    let t0 = unpacklo_epi32(r[0], r[1]);
    let t1 = unpackhi_epi32(r[0], r[1]);
    let t2 = unpacklo_epi32(r[2], r[3]);
    let t3 = unpackhi_epi32(r[2], r[3]);
    let t4 = unpacklo_epi32(r[4], r[5]);
    let t5 = unpackhi_epi32(r[4], r[5]);
    let t6 = unpacklo_epi32(r[6], r[7]);
    let t7 = unpackhi_epi32(r[6], r[7]);

    let s0 = unpacklo_epi64(t0, t2);
    let s1 = unpackhi_epi64(t0, t2);
    let s2 = unpacklo_epi64(t1, t3);
    let s3 = unpackhi_epi64(t1, t3);
    let s4 = unpacklo_epi64(t4, t6);
    let s5 = unpackhi_epi64(t4, t6);
    let s6 = unpacklo_epi64(t5, t7);
    let s7 = unpackhi_epi64(t5, t7);

    [
        permute2x128::<0x20>(s0, s4),
        permute2x128::<0x20>(s1, s5),
        permute2x128::<0x20>(s2, s6),
        permute2x128::<0x20>(s3, s7),
        permute2x128::<0x31>(s0, s4),
        permute2x128::<0x31>(s1, s5),
        permute2x128::<0x31>(s2, s6),
        permute2x128::<0x31>(s3, s7),
    ]
}

#[test]
fn avx2_ladder_is_an_exact_transpose() {
    // Each element encodes its own coordinates, so a misplaced one names
    // exactly where it came from.
    let mut input = [[0u32; 8]; 8];
    for (i, row) in input.iter_mut().enumerate() {
        for (j, v) in row.iter_mut().enumerate() {
            *v = (i as u32) * 8 + j as u32;
        }
    }

    let out = transpose8(input);

    for i in 0..8 {
        for j in 0..8 {
            assert_eq!(
                out[j][i], input[i][j],
                "element ({i},{j}) landed wrong: got {} at out[{j}][{i}]",
                out[j][i]
            );
        }
    }
}

/// A transpose is its own inverse, which catches whole-vector permutation
/// errors that a single pass can mask
#[test]
fn avx2_ladder_is_an_involution() {
    let mut input = [[0u32; 8]; 8];
    for (i, row) in input.iter_mut().enumerate() {
        for (j, v) in row.iter_mut().enumerate() {
            *v = ((i as u32) << 16) | (j as u32) | 0xa5000000;
        }
    }
    assert_eq!(transpose8(transpose8(input)), input);
}

// ---------------------------------------------------------------------------
// AVX-512 16x16 ladder
// ---------------------------------------------------------------------------

type W = [u32; 16];

/// A 512-bit register is four 128-bit lanes of four `u32`, and every intrinsic
/// below stays within them -- which is why the ladder needs two cross-lane
/// shuffle stages on top of the two unpack stages
fn z_unpacklo32(a: W, b: W) -> W {
    let mut o = [0u32; 16];
    for l in 0..4 {
        let p = l * 4;
        o[p] = a[p];
        o[p + 1] = b[p];
        o[p + 2] = a[p + 1];
        o[p + 3] = b[p + 1];
    }
    o
}

fn z_unpackhi32(a: W, b: W) -> W {
    let mut o = [0u32; 16];
    for l in 0..4 {
        let p = l * 4;
        o[p] = a[p + 2];
        o[p + 1] = b[p + 2];
        o[p + 2] = a[p + 3];
        o[p + 3] = b[p + 3];
    }
    o
}

fn z_unpacklo64(a: W, b: W) -> W {
    let mut o = [0u32; 16];
    for l in 0..4 {
        let p = l * 4;
        o[p] = a[p];
        o[p + 1] = a[p + 1];
        o[p + 2] = b[p];
        o[p + 3] = b[p + 1];
    }
    o
}

fn z_unpackhi64(a: W, b: W) -> W {
    let mut o = [0u32; 16];
    for l in 0..4 {
        let p = l * 4;
        o[p] = a[p + 2];
        o[p + 1] = a[p + 3];
        o[p + 2] = b[p + 2];
        o[p + 3] = b[p + 3];
    }
    o
}

/// `_mm512_shuffle_i32x4`: picks 128-bit lanes by two-bit fields of the
/// immediate, the low two from `a` and the high two from `b`
fn z_shuffle_i32x4<const IMM: u8>(a: W, b: W) -> W {
    let sel = [IMM & 3, (IMM >> 2) & 3, (IMM >> 4) & 3, (IMM >> 6) & 3];
    let mut o = [0u32; 16];
    for l in 0..4 {
        let src = if l < 2 { &a } else { &b };
        let k = sel[l] as usize;
        o[l * 4..l * 4 + 4].copy_from_slice(&src[k * 4..k * 4 + 4]);
    }
    o
}

/// The full 16x16 ladder from src/avx512.rs: stage 1, the 32-bit interleave
/// inlined into `Lanes::transpose`, then stages 2-4 from `transpose_rest`
fn transpose16(r: [W; 16]) -> [W; 16] {
    let mut t = [[0u32; 16]; 16];
    for i in 0..8 {
        t[2 * i] = z_unpacklo32(r[2 * i], r[2 * i + 1]);
        t[2 * i + 1] = z_unpackhi32(r[2 * i], r[2 * i + 1]);
    }
    let mut s = [[0u32; 16]; 16];
    for g in 0..4 {
        let b = g * 4;
        s[b] = z_unpacklo64(t[b], t[b + 2]);
        s[b + 1] = z_unpackhi64(t[b], t[b + 2]);
        s[b + 2] = z_unpacklo64(t[b + 1], t[b + 3]);
        s[b + 3] = z_unpackhi64(t[b + 1], t[b + 3]);
    }
    let mut u = [[0u32; 16]; 16];
    for g in 0..2 {
        let b = g * 8;
        for i in 0..4 {
            u[b + i] = z_shuffle_i32x4::<0x88>(s[b + i], s[b + i + 4]);
            u[b + i + 4] = z_shuffle_i32x4::<0xdd>(s[b + i], s[b + i + 4]);
        }
    }
    let mut o = [[0u32; 16]; 16];
    for i in 0..8 {
        o[i] = z_shuffle_i32x4::<0x88>(u[i], u[i + 8]);
        o[i + 8] = z_shuffle_i32x4::<0xdd>(u[i], u[i + 8]);
    }
    o
}

#[test]
fn avx512_ladder_is_an_exact_transpose() {
    let mut input = [[0u32; 16]; 16];
    for (i, row) in input.iter_mut().enumerate() {
        for (j, v) in row.iter_mut().enumerate() {
            *v = (i as u32) * 16 + j as u32;
        }
    }
    let out = transpose16(input);
    for i in 0..16 {
        for j in 0..16 {
            assert_eq!(
                out[j][i], input[i][j],
                "element ({i},{j}) landed wrong: got {} at out[{j}][{i}]",
                out[j][i]
            );
        }
    }
}

#[test]
fn avx512_ladder_is_an_involution() {
    let mut input = [[0u32; 16]; 16];
    for (i, row) in input.iter_mut().enumerate() {
        for (j, v) in row.iter_mut().enumerate() {
            *v = ((i as u32) << 16) | (j as u32) | 0x5a00_0000;
        }
    }
    assert_eq!(transpose16(transpose16(input)), input);
}
