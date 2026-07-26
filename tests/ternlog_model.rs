//! Validates the AVX-512 `vpternlogd` control bytes by exhaustive simulation
//!
//! Same reasoning as tests/transpose_model.rs: the AVX-512 kernel cannot run
//! on an AArch64 host, and modelling the intrinsic scalar-side checks the
//! truth-table bytes anywhere. A wrong byte is the ideal silent bug -- it does
//! not crash or fail to compile, and `0xCA` versus `0xAC` looks equally
//! plausible on inspection, but it means wrong roots and wrong block IDs.
//!
//! `vpternlogd` computes, per bit, `imm8 >> (a*4 + b*2 + c) & 1` for src1,
//! src2, src3. Only eight input combinations exist, so the bytes can be
//! checked exhaustively rather than argued about. This still complements
//! rather than replaces running tests/differential.rs on real x86.

/// Scalar model of `_mm512_ternarylogic_epi32::<IMM>(a, b, c)` for one bit
fn ternlog_bit(imm: u8, a: u32, b: u32, c: u32) -> u32 {
    let index = (a << 2) | (b << 1) | c;
    ((imm >> index) & 1) as u32
}

/// Applies the model across all 32 bit positions of a word
fn ternlog(imm: u8, a: u32, b: u32, c: u32) -> u32 {
    let mut out = 0u32;
    for bit in 0..32 {
        let (x, y, z) = ((a >> bit) & 1, (b >> bit) & 1, (c >> bit) & 1);
        out |= ternlog_bit(imm, x, y, z) << bit;
    }
    out
}

const XOR3: u8 = 0x96;
const CH: u8 = 0xCA;
const MAJ: u8 = 0xE8;

#[test]
fn xor3_control_byte() {
    for a in 0..2u32 {
        for b in 0..2u32 {
            for c in 0..2u32 {
                assert_eq!(
                    ternlog_bit(XOR3, a, b, c),
                    a ^ b ^ c,
                    "0x96 wrong at ({a},{b},{c})"
                );
            }
        }
    }
}

#[test]
fn ch_control_byte() {
    // Ch(x, y, z) = (x & y) ^ (!x & z), with a=x, b=y, c=z.
    for x in 0..2u32 {
        for y in 0..2u32 {
            for z in 0..2u32 {
                let want = (x & y) ^ ((x ^ 1) & z);
                assert_eq!(
                    ternlog_bit(CH, x, y, z),
                    want,
                    "0xCA wrong at ({x},{y},{z})"
                );
            }
        }
    }
}

#[test]
fn maj_control_byte() {
    // Maj(x, y, z) = (x & y) ^ (x & z) ^ (y & z).
    for x in 0..2u32 {
        for y in 0..2u32 {
            for z in 0..2u32 {
                let want = (x & y) ^ (x & z) ^ (y & z);
                assert_eq!(
                    ternlog_bit(MAJ, x, y, z),
                    want,
                    "0xE8 wrong at ({x},{y},{z})"
                );
            }
        }
    }
}

/// Word-level check against the exact expressions the portable `Lanes`
/// defaults use, so the ternlog override and the default cannot disagree
#[test]
fn ternlog_matches_portable_expressions() {
    let mut state = 0x1234_5678u32;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        state
    };
    for _ in 0..2000 {
        let (a, b, c) = (next(), next(), next());

        assert_eq!(ternlog(XOR3, a, b, c), a ^ b ^ c);

        // Ch as written in the trait default: (x & y) ^ (!x & z)
        assert_eq!(ternlog(CH, a, b, c), (a & b) ^ (!a & c));

        // Maj as written in the trait default: y ^ ((x ^ y) & (y ^ z))
        assert_eq!(ternlog(MAJ, a, b, c), b ^ ((a ^ b) & (b ^ c)));
        // and in its literal form
        assert_eq!(ternlog(MAJ, a, b, c), (a & b) ^ (a & c) ^ (b & c));
    }
}
