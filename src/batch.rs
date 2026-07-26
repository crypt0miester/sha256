//! Message framing, transposition, and the multi-buffer driver loop

use crate::{
    core::{compress, H0},
    lanes::Lanes,
};

/// Largest lane count any backend uses
///
/// Sizes the stack scratch buffers so the driver never allocates.
pub(crate) const MAX_LANES: usize = 16;

/// Largest chunk any driver stages: the 2x16 interlace takes 32 messages
///
/// Separate from MAX_LANES so hash_lanes' per-group scratch does not double
/// for backends that never need it.
pub(crate) const MAX_WIDTH: usize = 32;

pub(crate) const BLOCK: usize = 64;

/// One message to hash: an optional shared prefix plus a body
///
/// Keeping the prefix separate lets the Merkle-leaf case skip materialising
/// `prefix || body` per message; the driver reads from both slices directly.
#[derive(Clone, Copy)]
pub struct Message<'a> {
    pub prefix: &'a [u8],
    pub body: &'a [u8],
    /// Optional third segment, hashed after `body`
    pub tail: &'a [u8],
}

impl<'a> Message<'a> {
    #[inline]
    pub fn new(body: &'a [u8]) -> Self {
        Message {
            prefix: &[],
            body,
            tail: &[],
        }
    }

    #[inline]
    pub fn prefixed(prefix: &'a [u8], body: &'a [u8]) -> Self {
        Message {
            prefix,
            body,
            tail: &[],
        }
    }

    /// A message of `prefix || left || right`, for Merkle interior nodes
    #[inline]
    pub fn pair(prefix: &'a [u8], left: &'a [u8], right: &'a [u8]) -> Self {
        Message {
            prefix,
            body: left,
            tail: right,
        }
    }

    #[inline]
    pub(crate) fn len(&self) -> usize {
        self.prefix.len() + self.body.len() + self.tail.len()
    }

    /// Blocks after padding: message, 0x80 terminator, 64-bit big-endian length
    #[inline]
    pub(crate) fn blocks(&self) -> usize {
        (self.len() + 1 + 8).div_ceil(BLOCK)
    }

    /// True when block `k` lies wholly inside `body`, so no staging copy is needed
    ///
    /// For a 1KB Merkle leaf that is 15 of 17 blocks, which is nearly all the
    /// per-batch memcpy.
    #[inline]
    pub(crate) fn block_is_interior(&self, k: usize) -> bool {
        let start = k * BLOCK;
        // Must end before `tail` begins, not merely before the message does.
        start >= self.prefix.len() && start + BLOCK <= self.prefix.len() + self.body.len()
    }

    /// Borrows block `k` straight out of `body`
    ///
    /// Only valid when block_is_interior says so.
    #[inline]
    pub(crate) fn interior_block(&self, k: usize) -> &'a [u8] {
        let off = k * BLOCK - self.prefix.len();
        &self.body[off..off + BLOCK]
    }

    /// Writes block `k` of the padded message into `out`
    #[inline]
    pub(crate) fn fill_block(&self, k: usize, out: &mut [u8; BLOCK]) {
        let start = k * BLOCK;
        let len = self.len();
        let plen = self.prefix.len();
        out.fill(0);

        if start < plen {
            let n = (plen - start).min(BLOCK);
            out[..n].copy_from_slice(&self.prefix[start..start + n]);
        }
        // Body, then tail if there is one.
        let bend = plen + self.body.len();
        let from = start.max(plen);
        let to = (start + BLOCK).min(bend);
        if from < to {
            let off = from - start;
            let n = to - from;
            let sfrom = from - plen;
            out[off..off + n].copy_from_slice(&self.body[sfrom..sfrom + n]);
        }
        if !self.tail.is_empty() {
            let tend = bend + self.tail.len();
            let from = start.max(bend);
            let to = (start + BLOCK).min(tend);
            if from < to {
                let off = from - start;
                let n = to - from;
                let sfrom = from - bend;
                out[off..off + n].copy_from_slice(&self.tail[sfrom..sfrom + n]);
            }
        }
        // Terminator lands in whichever block the message ends in.
        if start <= len && len < start + BLOCK {
            out[len - start] = 0x80;
        }
        // Length field only exists in the last block.
        if start + BLOCK == self.blocks() * BLOCK {
            let bits = (len as u64).wrapping_mul(8);
            out[BLOCK - 8..].copy_from_slice(&bits.to_be_bytes());
        }
    }
}

/// Shared length split of a batch, for the same-shape fast path
///
/// Merkle batches share one prefix length, body length, and total length,
/// so block interiority is a property of the block index alone: decide it
/// once per block instead of per lane, and every interior source pointer
/// is then pure arithmetic. Body length is part of the shape, not just the
/// total, because two messages can split one total differently between
/// body and tail, and the interior path indexes `body` alone.
///
/// The bound every same-shape fast path leans on: when `same` holds and
/// `k` is in `k_lo..k_hi`, then `k * BLOCK >= plen` and
/// `k * BLOCK + BLOCK <= plen + body.len()`, so the 64 bytes at
/// `body_ptr.add(k * BLOCK - plen)` lie wholly inside every lane's `body`.
pub(crate) struct Shape {
    pub(crate) same: bool,
    pub(crate) plen: usize,
    /// First and one-past-last block wholly inside `body`
    pub(crate) k_lo: usize,
    pub(crate) k_hi: usize,
}

impl Shape {
    /// Computes the shared shape; `msgs` must be non-empty
    #[inline(always)]
    pub(crate) fn of(msgs: &[Message<'_>]) -> Shape {
        let plen = msgs[0].prefix.len();
        let blen = msgs[0].body.len();
        let len = msgs[0].len();
        let same = msgs
            .iter()
            .all(|m| m.prefix.len() == plen && m.body.len() == blen && m.len() == len);
        Shape {
            same,
            plen,
            k_lo: plen.div_ceil(BLOCK),
            k_hi: (plen + blen) / BLOCK,
        }
    }
}

#[inline]
pub(crate) fn write_digest(state: &[[u32; MAX_LANES]; 8], lane: usize, out: &mut [u8; 32]) {
    for (i, chunk) in out.chunks_exact_mut(4).enumerate() {
        chunk.copy_from_slice(&state[i][lane].to_be_bytes());
    }
}

/// Hashes up to `L::N` messages in lockstep, one per lane
///
/// Lengths may differ; a lane that finishes early has its digest taken and
/// then idles to the end of the longest lane.
#[inline(always)]
pub(crate) fn hash_lanes<L: Lanes>(msgs: &[Message<'_>], out: &mut [[u8; 32]]) {
    // A wider backend would silently overrun the staging arrays.
    const { assert!(L::N <= MAX_LANES) };
    assert!(msgs.len() <= L::N);
    assert_eq!(msgs.len(), out.len());
    let n = msgs.len();
    if n == 0 {
        return;
    }

    let mut state = H0.map(L::splat);

    // Needed per lane per iteration, so derive them once up front.
    let mut nblocks = [0usize; MAX_LANES];
    let mut max_blocks = 0usize;
    for (lane, m) in msgs.iter().enumerate() {
        let b = m.blocks();
        nblocks[lane] = b;
        max_blocks = max_blocks.max(b);
    }
    let uniform = nblocks[..n].iter().all(|&b| b == max_blocks);

    let mut blocks = [[0u8; BLOCK]; MAX_LANES];
    let mut unpacked = [[0u32; MAX_LANES]; 8];

    let shape = Shape::of(msgs);
    let mut bases: [*const u8; MAX_LANES] = [std::ptr::null(); MAX_LANES];
    for (b, m) in bases.iter_mut().zip(msgs) {
        *b = m.body.as_ptr();
    }
    let mut staged = [usize::MAX; MAX_LANES];
    let mut kk = [0usize; MAX_LANES];
    let mut interior = [false; MAX_LANES];

    let mut srcs: [*const u8; MAX_LANES] = [std::ptr::null(); MAX_LANES];
    for k in 0..max_blocks {
        // Pre-filled then overwritten.
        if shape.same {
            if k >= shape.k_lo && k < shape.k_hi {
                let off = k * BLOCK - shape.plen;
                for (s, base) in srcs.iter_mut().zip(bases.iter()).take(n) {
                    // SAFETY: the interior bound documented on `Shape`.
                    *s = unsafe { base.add(off) };
                }
            } else {
                for (lane, m) in msgs.iter().enumerate() {
                    m.fill_block(k, &mut blocks[lane]);
                }
                for (lane, b) in blocks.iter().enumerate().take(n) {
                    srcs[lane] = b.as_ptr();
                }
            }
        } else {
            // Stage only blocks straddling the prefix, terminator, or length.
            for (lane, m) in msgs.iter().enumerate() {
                let idx = k.min(nblocks[lane] - 1);
                kk[lane] = idx;
                let is_interior = m.block_is_interior(idx);
                interior[lane] = is_interior;
                if !is_interior && staged[lane] != idx {
                    m.fill_block(idx, &mut blocks[lane]);
                    staged[lane] = idx;
                }
            }
            // Separate pass so the staging array is borrowed immutably here.
            for (lane, m) in msgs.iter().enumerate() {
                srcs[lane] = if interior[lane] {
                    m.interior_block(kk[lane]).as_ptr()
                } else {
                    blocks[lane].as_ptr()
                };
            }
        }
        // SAFETY: every lane below `n` was just set to a pointer valid for a
        // full 64-byte block, either into a borrowed body or into `blocks`.
        let w = unsafe { L::transpose(&srcs, n) };
        compress::<L>(&mut state, w);

        // Uniform lanes all finish together.
        if !uniform {
            let mut any = false;
            for lane in 0..n {
                if nblocks[lane] != k + 1 {
                    continue;
                }
                if !any {
                    for (i, s) in state.iter().enumerate() {
                        s.store(&mut unpacked[i][..L::N]);
                    }
                    any = true;
                }
                write_digest(&unpacked, lane, &mut out[lane]);
            }
        }
    }

    if uniform {
        for (i, s) in state.iter().enumerate() {
            s.store(&mut unpacked[i][..L::N]);
        }
        for (lane, o) in out.iter_mut().enumerate().take(n) {
            write_digest(&unpacked, lane, o);
        }
    }
}

/// Hashes `prefix || left || right` per pair, without materialising the join
///
/// # Safety
///
/// As for `drive`.
pub(crate) unsafe fn drive_pairs(
    width: usize,
    group: GroupFn,
    prefix: &[u8],
    left: &[&[u8]],
    right: &[&[u8]],
    out: &mut [[u8; 32]],
) {
    // Staging is sized to the caller's width class so narrow backends do
    // not pay MAX_WIDTH's init: 32 slots is 1.5KB of stores per call, and
    // only the interlace reads past 16.
    if width <= MAX_LANES {
        drive_pairs_staged::<MAX_LANES>(width, group, prefix, left, right, out)
    } else {
        drive_pairs_staged::<MAX_WIDTH>(width, group, prefix, left, right, out)
    }
}

unsafe fn drive_pairs_staged<const W: usize>(
    width: usize,
    group: GroupFn,
    prefix: &[u8],
    left: &[&[u8]],
    right: &[&[u8]],
    out: &mut [[u8; 32]],
) {
    debug_assert!(width <= W);
    assert_eq!(left.len(), right.len(), "left and right must pair up");
    assert_eq!(
        left.len(),
        out.len(),
        "output slice must have one digest per pair"
    );
    let mut staging = [Message::new(&[]); W];
    for ((l, r), o) in left
        .chunks(width)
        .zip(right.chunks(width))
        .zip(out.chunks_mut(width))
    {
        for ((slot, a), b) in staging.iter_mut().zip(l).zip(r) {
            *slot = Message::pair(prefix, a, b);
        }
        group(&staging[..l.len()], o);
    }
}

/// One lane-group's worth of hashing, the unit the drivers dispatch through.
pub(crate) type GroupFn = unsafe fn(&[Message<'_>], &mut [[u8; 32]]);

/// Hashes every message, `width` at a time, through `group`
///
/// # Safety
///
/// `group`'s CPU-feature contract must hold on this machine, and `width` must
/// be the lane count `group` was monomorphized for.
pub(crate) unsafe fn drive(
    width: usize,
    group: GroupFn,
    msgs: &[Message<'_>],
    out: &mut [[u8; 32]],
) {
    assert_eq!(
        msgs.len(),
        out.len(),
        "output slice must have one digest per message"
    );
    for (m, o) in msgs.chunks(width).zip(out.chunks_mut(width)) {
        group(m, o);
    }
}

/// Hashes bare byte slices without materialising a `Vec<Message>`
///
/// The public wrappers take `&[&[u8]]`, so attaching a prefix would otherwise
/// cost an allocation per call. Chunks are built on the stack instead; the
/// intended caller is a hot validator path.
///
/// # Safety
///
/// As for drive.
pub(crate) unsafe fn drive_slices(
    width: usize,
    group: GroupFn,
    prefix: &[u8],
    bodies: &[&[u8]],
    out: &mut [[u8; 32]],
) {
    // See drive_pairs for why staging is width-classed.
    if width <= MAX_LANES {
        drive_slices_staged::<MAX_LANES>(width, group, prefix, bodies, out)
    } else {
        drive_slices_staged::<MAX_WIDTH>(width, group, prefix, bodies, out)
    }
}

unsafe fn drive_slices_staged<const W: usize>(
    width: usize,
    group: GroupFn,
    prefix: &[u8],
    bodies: &[&[u8]],
    out: &mut [[u8; 32]],
) {
    debug_assert!(width <= W);
    assert_eq!(
        bodies.len(),
        out.len(),
        "output slice must have one digest per message"
    );
    let mut staging = [Message::new(&[]); W];
    for (chunk, o) in bodies.chunks(width).zip(out.chunks_mut(width)) {
        for (slot, body) in staging.iter_mut().zip(chunk) {
            *slot = Message::prefixed(prefix, body);
        }
        group(&staging[..chunk.len()], o);
    }
}
