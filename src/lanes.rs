//! The word abstraction the SHA-256 core is written against
//!
//! Multi-buffer SHA-256 hashes `N` messages in lockstep,
//! backends cannot drift from the reference.

/// A vector of `N` lanes of `u32`, supporting the operations SHA-256 needs
///
/// # Correctness contract
///
/// Implementations must be elementwise and exactly 32-bit wrapping: `add` wraps
/// rather than saturates, and `rotr`/`shr` treat each lane as its own `u32`.
pub trait Lanes: Copy {
    /// Number of independent messages hashed in parallel
    const N: usize;

    /// Whether to expand the 64 compression rounds flat
    const FLAT_ROUNDS: bool = true;

    fn splat(v: u32) -> Self;

    fn load(v: &[u32]) -> Self;

    fn store(self, out: &mut [u32]);

    fn add(self, o: Self) -> Self;

    fn xor(self, o: Self) -> Self;

    fn and(self, o: Self) -> Self;

    fn not_and(self, o: Self) -> Self;

    fn shr<const B: u32>(self) -> Self;

    fn rotr<const B: u32>(self) -> Self;

    /// `a ^ b ^ c`, elementwise
    #[inline(always)]
    fn xor3(self, b: Self, c: Self) -> Self {
        self.xor(b).xor(c)
    }

    /// `Ch(x, y, z) = (x & y) ^ (!x & z)`
    #[inline(always)]
    fn ch(self, y: Self, z: Self) -> Self {
        self.and(y).xor(self.not_and(z))
    }

    /// `Maj(x, y, z) = (x & y) ^ (x & z) ^ (y & z)`
    #[inline(always)]
    fn maj(self, y: Self, z: Self) -> Self {
        y.xor(self.xor(y).and(y.xor(z)))
    }

    /// Gathers word `j` of each lane's block into `out[j]`, from big-endian.
    ///
    /// # Safety
    ///
    /// For every `i < n`, `ptrs[i]` must be valid for reads of 64 bytes, and
    /// `n` must be at least 1 and at most `Self::N`.
    #[inline(always)]
    unsafe fn transpose(ptrs: &[*const u8], n: usize) -> [Self; 16] {
        let mut scratch = [0u32; 16];
        let mut w = [Self::splat(0); 16];
        for (j, wj) in w.iter_mut().enumerate() {
            for (lane, s) in scratch[..n].iter_mut().enumerate() {
                let b = ptrs[lane].add(4 * j);
                *s = u32::from_be_bytes([*b, *b.add(1), *b.add(2), *b.add(3)]);
            }
            let fill = scratch[0];
            scratch[n..Self::N].fill(fill);
            *wj = Self::load(&scratch[..Self::N]);
        }
        w
    }
}

/// Portable backend: `N` lanes held as a plain array
///
/// The reference the SIMD backends are differentially tested against, and the
/// fallback where no vector unit we support exists. `N = 1` is plain SHA-256.
#[derive(Clone, Copy)]
pub struct Scalar<const N: usize>(pub [u32; N]);

impl<const N: usize> Lanes for Scalar<N> {
    const N: usize = N;

    // Flat rounds spill catastrophically here, see the trait doc.
    const FLAT_ROUNDS: bool = false;

    #[inline(always)]
    fn splat(v: u32) -> Self {
        Scalar([v; N])
    }

    #[inline(always)]
    fn load(v: &[u32]) -> Self {
        debug_assert_eq!(v.len(), N);
        let mut a = [0u32; N];
        a.copy_from_slice(v);
        Scalar(a)
    }

    #[inline(always)]
    fn store(self, out: &mut [u32]) {
        debug_assert_eq!(out.len(), N);
        out.copy_from_slice(&self.0);
    }

    #[inline(always)]
    fn add(self, o: Self) -> Self {
        let mut a = self.0;
        for (x, y) in a.iter_mut().zip(o.0) {
            *x = x.wrapping_add(y);
        }
        Scalar(a)
    }

    #[inline(always)]
    fn xor(self, o: Self) -> Self {
        let mut a = self.0;
        for (x, y) in a.iter_mut().zip(o.0) {
            *x ^= y;
        }
        Scalar(a)
    }

    #[inline(always)]
    fn and(self, o: Self) -> Self {
        let mut a = self.0;
        for (x, y) in a.iter_mut().zip(o.0) {
            *x &= y;
        }
        Scalar(a)
    }

    #[inline(always)]
    fn not_and(self, o: Self) -> Self {
        let mut a = self.0;
        for (x, y) in a.iter_mut().zip(o.0) {
            *x = !*x & y;
        }
        Scalar(a)
    }

    #[inline(always)]
    fn shr<const B: u32>(self) -> Self {
        let mut a = self.0;
        for x in a.iter_mut() {
            *x >>= B;
        }
        Scalar(a)
    }

    #[inline(always)]
    fn rotr<const B: u32>(self) -> Self {
        let mut a = self.0;
        for x in a.iter_mut() {
            *x = x.rotate_right(B);
        }
        Scalar(a)
    }
}
