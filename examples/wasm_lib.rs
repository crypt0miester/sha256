//! tape-sha256 as a JS-callable wasm module
//!
//! The shape an npm package would ship: the caller copies message bytes into
//! linear memory and reads digests back out of it. Exists for the wasm
//! benches, which measure this crate from JavaScript against the libraries
//! Solana's JS SDKs actually use.
//!
//! Build with:
//!
//! ```sh
//! cargo build --release --example wasm_lib --target wasm32-unknown-unknown
//! ```

use std::alloc::{alloc, Layout};

/// Hands out a buffer the JS side can write into
///
/// Never freed; the bench allocates once at startup and reuses.
#[no_mangle]
pub extern "C" fn walloc(size: usize) -> *mut u8 {
    unsafe { alloc(Layout::from_size_align(size, 8).unwrap()) }
}

/// Writes the active dispatch backend's name into `out` and returns its length
///
/// # Safety
///
/// `out` must be valid for writes of `cap` bytes.
#[no_mangle]
pub unsafe extern "C" fn backend_name(out: *mut u8, cap: usize) -> usize {
    let name = tape_sha256::backend().as_bytes();
    let n = name.len().min(cap);
    std::ptr::copy_nonoverlapping(name.as_ptr(), out, n);
    n
}

/// Hashes `count` messages laid back-to-back in `data` through dispatch
///
/// Each message is `lens[i]` bytes and gets `prefix` prepended; digests land
/// in `out`, 32 bytes per message.
///
/// # Safety
///
/// All pointers must be `walloc`-obtained regions of the sizes implied.
#[no_mangle]
pub unsafe extern "C" fn hash_many_prefixed_raw(
    prefix: *const u8,
    prefix_len: usize,
    data: *const u8,
    lens: *const u32,
    count: usize,
    out: *mut u8,
) {
    let prefix = std::slice::from_raw_parts(prefix, prefix_len);
    let lens = std::slice::from_raw_parts(lens, count);

    let mut bodies: Vec<&[u8]> = Vec::with_capacity(count);
    let mut off = 0usize;
    for &l in lens {
        bodies.push(std::slice::from_raw_parts(data.add(off), l as usize));
        off += l as usize;
    }

    let out = std::slice::from_raw_parts_mut(out.cast::<[u8; 32]>(), count);
    tape_sha256::hash_many_prefixed(prefix, &bodies, out);
}

/// Pins one wave-count kernel per export so the benchmark can walk the curve
///
/// Same contract and safety requirements as `hash_many_prefixed_raw`.
#[cfg(target_feature = "simd128")]
macro_rules! wave_entry {
    ($name:ident, $backend:ident) => {
        /// # Safety
        ///
        /// As for `hash_many_prefixed_raw`.
        #[no_mangle]
        pub unsafe extern "C" fn $name(
            prefix: *const u8,
            prefix_len: usize,
            data: *const u8,
            lens: *const u32,
            count: usize,
            out: *mut u8,
        ) {
            let prefix = std::slice::from_raw_parts(prefix, prefix_len);
            let lens = std::slice::from_raw_parts(lens, count);

            let mut bodies: Vec<&[u8]> = Vec::with_capacity(count);
            let mut off = 0usize;
            for &l in lens {
                bodies.push(std::slice::from_raw_parts(data.add(off), l as usize));
                off += l as usize;
            }

            let out = std::slice::from_raw_parts_mut(out.cast::<[u8; 32]>(), count);
            tape_sha256::backends::$backend(prefix, &bodies, out);
        }
    };
}

#[cfg(target_feature = "simd128")]
wave_entry!(hash_many_prefixed_raw_2x4, simd128_8_slices);
#[cfg(target_feature = "simd128")]
wave_entry!(hash_many_prefixed_raw_4x4, simd128_16_slices);
