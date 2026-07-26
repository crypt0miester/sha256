//! Correctness gate: every backend must match an independent SHA-256.

use {
    rand::{rngs::StdRng, Rng, SeedableRng},
    sha2::{Digest, Sha256},
    tape_sha256::{hash_many, hash_many_prefixed, hash_messages, Message},
};

fn reference(msg: &[u8]) -> [u8; 32] {
    let mut h = Sha256::new();
    h.update(msg);
    h.finalize().into()
}

/// Lengths bracketing every padding and block-boundary transition
///
/// Empty, tiny, either side of each 64-byte block, and either side of 55/56
/// where the 9 padding bytes force an extra block.
fn edge_lengths() -> Vec<usize> {
    let mut v = vec![0usize, 1, 2, 3];
    for block in 0..6usize {
        let b = block * 64;
        for d in [53, 54, 55, 56, 57, 62, 63, 64, 65, 66] {
            v.push(b + d);
        }
    }
    // Around the real Merkle-leaf sizes this crate was built for.
    v.extend([955, 980, 1019, 1044, 1045, 1070]);
    v.sort_unstable();
    v.dedup();
    v
}

fn pattern(len: usize, seed: u8) -> Vec<u8> {
    (0..len)
        .map(|i| (i as u8).wrapping_mul(31).wrapping_add(seed))
        .collect()
}

#[test]
fn single_message_every_edge_length() {
    for len in edge_lengths() {
        let msg = pattern(len, 7);
        let mut got = [[0u8; 32]; 1];
        hash_many(&[&msg], &mut got);
        assert_eq!(got[0], reference(&msg), "length {len}");
    }
}

/// A full batch of equal lengths, driving the uniform fast path
#[test]
fn uniform_batch_every_edge_length() {
    for len in edge_lengths() {
        let bodies: Vec<Vec<u8>> = (0..8).map(|i| pattern(len, i as u8)).collect();
        let refs: Vec<&[u8]> = bodies.iter().map(|b| b.as_slice()).collect();
        let mut got = vec![[0u8; 32]; bodies.len()];
        hash_many(&refs, &mut got);
        for (i, b) in bodies.iter().enumerate() {
            assert_eq!(got[i], reference(b), "length {len} lane {i}");
        }
    }
}

/// Lanes of differing lengths in one batch
///
/// Early finishers must have their digest snapped off while other lanes keep
/// going, so this is the likeliest place to capture a lane at the wrong round.
#[test]
fn ragged_batch_lengths() {
    let lens = edge_lengths();
    for window in lens.windows(8) {
        let bodies: Vec<Vec<u8>> = window
            .iter()
            .enumerate()
            .map(|(i, &l)| pattern(l, i as u8))
            .collect();
        let refs: Vec<&[u8]> = bodies.iter().map(|b| b.as_slice()).collect();
        let mut got = vec![[0u8; 32]; bodies.len()];
        hash_many(&refs, &mut got);
        for (i, b) in bodies.iter().enumerate() {
            assert_eq!(got[i], reference(b), "lens {window:?} lane {i}");
        }
    }
}

/// Every batch size up to several lane widths, covering partial final batches
/// and inactive lanes
#[test]
fn every_batch_size() {
    for count in 0..40usize {
        let bodies: Vec<Vec<u8>> = (0..count).map(|i| pattern(100 + i * 13, i as u8)).collect();
        let refs: Vec<&[u8]> = bodies.iter().map(|b| b.as_slice()).collect();
        let mut got = vec![[0u8; 32]; count];
        hash_many(&refs, &mut got);
        for (i, b) in bodies.iter().enumerate() {
            assert_eq!(got[i], reference(b), "count {count} lane {i}");
        }
    }
}

/// The prefixed form must equal hashing the concatenation, including when the
/// prefix alone spans a block and when it is empty
#[test]
fn prefixed_matches_concatenation() {
    let prefixes: Vec<Vec<u8>> = vec![
        vec![],
        b"\x00SOLANA_MERKLE_SHREDS_LEAF".to_vec(),
        vec![0xab; 63],
        vec![0xcd; 64],
        vec![0xef; 65],
    ];
    for prefix in prefixes {
        for len in edge_lengths() {
            let bodies: Vec<Vec<u8>> = (0..8).map(|i| pattern(len, i as u8)).collect();
            let refs: Vec<&[u8]> = bodies.iter().map(|b| b.as_slice()).collect();
            let mut got = vec![[0u8; 32]; bodies.len()];
            hash_many_prefixed(&prefix, &refs, &mut got);
            for (i, b) in bodies.iter().enumerate() {
                let joined = [prefix.as_slice(), b.as_slice()].concat();
                assert_eq!(
                    got[i],
                    reference(&joined),
                    "prefix {} body {len} lane {i}",
                    prefix.len()
                );
            }
        }
    }
}

/// Per-message prefixes, mixing prefixed and bare messages in one batch
#[test]
fn mixed_per_message_prefixes() {
    let pa = b"alpha".to_vec();
    let pb = vec![0x5au8; 70];
    let bodies: Vec<Vec<u8>> = (0..8).map(|i| pattern(200 + i * 29, i as u8)).collect();
    let msgs: Vec<Message<'_>> = bodies
        .iter()
        .enumerate()
        .map(|(i, b)| match i % 3 {
            0 => Message::new(b),
            1 => Message::prefixed(&pa, b),
            _ => Message::prefixed(&pb, b),
        })
        .collect();
    let mut got = vec![[0u8; 32]; bodies.len()];
    hash_messages(&msgs, &mut got);
    for (i, b) in bodies.iter().enumerate() {
        let joined = match i % 3 {
            0 => b.clone(),
            1 => [pa.as_slice(), b].concat(),
            _ => [pb.as_slice(), b].concat(),
        };
        assert_eq!(got[i], reference(&joined), "lane {i}");
    }
}

/// Broad randomised sweep on top of the targeted cases above
#[test]
fn randomised_sweep() {
    let mut rng = StdRng::seed_from_u64(0xD1FF_5EED);
    for _ in 0..300 {
        let count = rng.gen_range(0..24usize);
        let bodies: Vec<Vec<u8>> = (0..count)
            .map(|_| {
                let len = rng.gen_range(0..1200usize);
                (0..len).map(|_| rng.r#gen::<u8>()).collect()
            })
            .collect();
        let refs: Vec<&[u8]> = bodies.iter().map(|b| b.as_slice()).collect();
        let mut got = vec![[0u8; 32]; count];
        hash_many(&refs, &mut got);
        for (i, b) in bodies.iter().enumerate() {
            assert_eq!(got[i], reference(b), "random lane {i} len {}", b.len());
        }
    }
}

/// Every backend must agree with every other, not merely with the reference
///
/// A backend that dispatch never selects on this host still gets exercised
/// here, provided the host can actually run it.
#[test]
fn backends_agree() {
    use tape_sha256::backends;

    let lens = edge_lengths();
    for window in lens.windows(9) {
        let bodies: Vec<Vec<u8>> = window
            .iter()
            .enumerate()
            .map(|(i, &l)| pattern(l, i as u8))
            .collect();
        let prefix = b"\x00SOLANA_MERKLE_SHREDS_LEAF";
        let msgs: Vec<Message<'_>> = bodies
            .iter()
            .map(|b| Message::prefixed(prefix, b))
            .collect();

        let mut want = vec![[0u8; 32]; bodies.len()];
        backends::serial(&msgs, &mut want);
        for (i, b) in bodies.iter().enumerate() {
            let joined = [prefix.as_slice(), b.as_slice()].concat();
            assert_eq!(want[i], reference(&joined), "serial lane {i}");
        }

        let mut got = vec![[0u8; 32]; bodies.len()];
        backends::portable8(&msgs, &mut got);
        assert_eq!(got, want, "portable8 vs serial, lens {window:?}");

        #[cfg(all(target_arch = "aarch64", not(feature = "scalar")))]
        {
            let mut got = vec![[0u8; 32]; bodies.len()];
            backends::neon4(&msgs, &mut got);
            assert_eq!(got, want, "neon4 vs serial, lens {window:?}");

            if std::arch::is_aarch64_feature_detected!("sha2") {
                let mut got = vec![[0u8; 32]; bodies.len()];
                unsafe { backends::neon_sha2x4(&msgs, &mut got) };
                assert_eq!(got, want, "neon_sha2x4 vs serial, lens {window:?}");
            }
        }

        // Only ever run on real x86. The AVX kernels are developed on an
        // AArch64 host where they compile but never execute, so this is the
        // gate that must pass before they are trusted anywhere.
        #[cfg(all(target_arch = "x86_64", not(feature = "scalar")))]
        {
            if is_x86_feature_detected!("sha") {
                let mut got = vec![[0u8; 32]; bodies.len()];
                unsafe { backends::shani_x4(&msgs, &mut got) };
                assert_eq!(got, want, "shani_x4 vs serial, lens {window:?}");
            }
            if is_x86_feature_detected!("avx2") {
                let mut got = vec![[0u8; 32]; bodies.len()];
                unsafe { backends::avx2_8(&msgs, &mut got) };
                assert_eq!(got, want, "avx2_8 vs serial, lens {window:?}");
            }
            if is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512bw") {
                let mut got = vec![[0u8; 32]; bodies.len()];
                unsafe { backends::avx512_16(&msgs, &mut got) };
                assert_eq!(got, want, "avx512_16 vs serial, lens {window:?}");

                let mut got = vec![[0u8; 32]; bodies.len()];
                unsafe { backends::avx512_16x2(&msgs, &mut got) };
                assert_eq!(got, want, "avx512_16x2 vs serial, lens {window:?}");
            }
        }
    }
}

/// The `_slices` forms must agree with the `Message`-taking ones
///
/// These are the paths the public wrappers actually dispatch to, and without
/// this only the host's active kernel would ever exercise them. Uniform
/// batches drive the same-shape fast path, ragged windows the general one.
#[test]
fn slices_forms_agree() {
    use tape_sha256::backends;

    let prefix = b"\x00SOLANA_MERKLE_SHREDS_LEAF";
    let lens = edge_lengths();
    // 33 = one full chunk of the widest kernel (the 32-message interlace)
    // plus a straggler, so the fused path and its remainder are exercised
    // through the `_slices` drivers, not only the dedicated test below.
    let mut cases: Vec<Vec<usize>> = lens.iter().map(|&l| vec![l; 33]).collect();
    cases.extend(lens.windows(9).map(|w| w.to_vec()));

    for case in &cases {
        let bodies: Vec<Vec<u8>> = case
            .iter()
            .enumerate()
            .map(|(i, &l)| pattern(l, i as u8))
            .collect();
        let refs: Vec<&[u8]> = bodies.iter().map(|b| b.as_slice()).collect();
        let msgs: Vec<Message<'_>> = bodies
            .iter()
            .map(|b| Message::prefixed(prefix, b))
            .collect();

        let mut want = vec![[0u8; 32]; bodies.len()];
        backends::serial(&msgs, &mut want);

        let mut got = vec![[0u8; 32]; bodies.len()];
        backends::portable8_slices(prefix, &refs, &mut got);
        assert_eq!(got, want, "portable8_slices, lens {case:?}");

        #[cfg(all(target_arch = "aarch64", not(feature = "scalar")))]
        {
            let mut got = vec![[0u8; 32]; bodies.len()];
            backends::neon4_slices(prefix, &refs, &mut got);
            assert_eq!(got, want, "neon4_slices, lens {case:?}");

            if std::arch::is_aarch64_feature_detected!("sha2") {
                let mut got = vec![[0u8; 32]; bodies.len()];
                unsafe { backends::neon_sha2x4_slices(prefix, &refs, &mut got) };
                assert_eq!(got, want, "neon_sha2x4_slices, lens {case:?}");
            }
        }
        #[cfg(all(target_arch = "x86_64", not(feature = "scalar")))]
        {
            if is_x86_feature_detected!("sha") {
                let mut got = vec![[0u8; 32]; bodies.len()];
                unsafe { backends::shani_x4_slices(prefix, &refs, &mut got) };
                assert_eq!(got, want, "shani_x4_slices, lens {case:?}");
            }
            if is_x86_feature_detected!("avx2") {
                let mut got = vec![[0u8; 32]; bodies.len()];
                unsafe { backends::avx2_8_slices(prefix, &refs, &mut got) };
                assert_eq!(got, want, "avx2_8_slices, lens {case:?}");
            }
            if is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512bw") {
                let mut got = vec![[0u8; 32]; bodies.len()];
                unsafe { backends::avx512_16_slices(prefix, &refs, &mut got) };
                assert_eq!(got, want, "avx512_16_slices, lens {case:?}");

                let mut got = vec![[0u8; 32]; bodies.len()];
                unsafe { backends::avx512_16x2_slices(prefix, &refs, &mut got) };
                assert_eq!(got, want, "avx512_16x2_slices, lens {case:?}");
            }
        }
    }
}

/// The two-wave kernel's fused path needs a full uniform 32; the windows in
/// backends_agree only reach its split path. Covers fused (32), fused plus
/// the driver's next chunk (33), and ragged (24, split), at every edge
/// length.
#[cfg(all(target_arch = "x86_64", not(feature = "scalar")))]
#[test]
fn avx512_two_wave_full_group() {
    use tape_sha256::backends;

    if !(is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512bw")) {
        return;
    }
    let prefix = b"\x00SOLANA_MERKLE_SHREDS_LEAF";
    for len in edge_lengths() {
        for count in [24usize, 32, 33] {
            let bodies: Vec<Vec<u8>> = (0..count).map(|i| pattern(len, i as u8)).collect();
            let msgs: Vec<Message<'_>> = bodies
                .iter()
                .map(|b| Message::prefixed(prefix, b))
                .collect();
            let mut want = vec![[0u8; 32]; count];
            backends::serial(&msgs, &mut want);
            let mut got = vec![[0u8; 32]; count];
            unsafe { backends::avx512_16x2(&msgs, &mut got) };
            assert_eq!(got, want, "two-wave len {len} count {count}");
        }
    }
}

/// Reports which kernels this run actually exercised
///
/// A green run on a host that silently skipped the AVX kernels is not a
/// validated one, and only this output tells the two apart.
#[test]
fn report_exercised_backends() {
    println!("active dispatch backend: {}", tape_sha256::backend());
    #[cfg(target_arch = "x86_64")]
    {
        println!("avx2 available:   {}", is_x86_feature_detected!("avx2"));
        println!(
            "avx512 available: {}",
            is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512bw")
        );
    }
    #[cfg(target_arch = "aarch64")]
    println!(
        "aarch64 sha2 ext: {}",
        std::arch::is_aarch64_feature_detected!("sha2")
    );
    #[cfg(not(target_arch = "x86_64"))]
    println!("not x86_64: AVX kernels compiled but NOT executed by this run");
}

/// `hash_pairs` must equal hashing the materialised `prefix || left || right`
///
/// This is the Merkle interior-node shape: agave's `join_nodes` hashes a
/// 26-byte domain prefix and two 20-byte proof entries from separate buffers
/// (ledger/src/shred/merkle_tree.rs). The three-segment path must not drift
/// from simple concatenation, at any length or batch size.
#[test]
fn hash_pairs_matches_concatenation() {
    use tape_sha256::hash_pairs;

    let prefixes: Vec<Vec<u8>> = vec![
        vec![],
        b"\x01SOLANA_MERKLE_SHREDS_NODE".to_vec(),
        vec![0xab; 63],
        vec![0xcd; 64],
        vec![0xef; 65],
    ];
    // 20 is the real proof-entry size; the rest bracket block boundaries.
    for half in [0usize, 1, 20, 31, 32, 63, 64, 65, 500] {
        for prefix in &prefixes {
            for count in [1usize, 3, 8, 9, 16, 17, 20] {
                let lefts: Vec<Vec<u8>> = (0..count).map(|i| pattern(half, i as u8)).collect();
                let rights: Vec<Vec<u8>> =
                    (0..count).map(|i| pattern(half, (i + 99) as u8)).collect();
                let lr: Vec<&[u8]> = lefts.iter().map(|b| b.as_slice()).collect();
                let rr: Vec<&[u8]> = rights.iter().map(|b| b.as_slice()).collect();

                let mut got = vec![[0u8; 32]; count];
                hash_pairs(prefix, &lr, &rr, &mut got);

                for i in 0..count {
                    let joined = [prefix.as_slice(), &lefts[i], &rights[i]].concat();
                    assert_eq!(
                        got[i],
                        reference(&joined),
                        "prefix {} half {half} count {count} pair {i}",
                        prefix.len()
                    );
                }
            }
        }
    }
}
