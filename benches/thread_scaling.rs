//! What caller threads are worth, and what chunking costs
//!
//! Every other bench here is one thread. This one answers the two questions a
//! caller with a thread pool actually has: how far does aggregate throughput
//! scale as threads are added, and how much does it cost to hand each thread a
//! chunk that does not fill the dispatched kernel.
//!
//! The library never spawns or pins, so the threads are the harness's, one
//! erasure batch each, no sharing and no synchronisation inside a round. Run
//! it on an otherwise idle machine; on x86 pin with `taskset` to physical
//! cores if you want the per-core numbers to mean anything, since two threads
//! landing on one core's SMT siblings measure spare capacity rather than a
//! second core.

use {
    std::time::Instant,
    tape_sha256::{hash_messages, lane_width, Message},
};

/// A kernel to sweep: the dispatched one, or a named backend to compare
/// against it at full occupancy.
type Kernel = fn(&[Message<'_>], &mut [[u8; 32]]);

const PREFIX: &[u8] = b"\x00SOLANA_MERKLE_SHREDS_LEAF";
/// Leaf sizes for a 32:32 batch with 6 proof entries, as in `merkle_leaves`
const DATA_LEAF: usize = 1019;
const CODE_LEAF: usize = 1044;
const DATA_SHREDS: usize = 32;
const CODE_SHREDS: usize = 32;
const LEAVES: usize = DATA_SHREDS + CODE_SHREDS;

/// One erasure batch of leaves, distinct per worker so no two threads share a
/// cache line of input
fn batch(seed: u8) -> Vec<Vec<u8>> {
    (0..LEAVES)
        .map(|i| {
            let len = if i < DATA_SHREDS { DATA_LEAF } else { CODE_LEAF };
            (0..len)
                .map(|j| (j as u8) ^ (i as u8).wrapping_mul(31) ^ seed)
                .collect()
        })
        .collect()
}

/// Hashes `msgs` in chunks of `chunk`, the shape a rayon caller produces when
/// it splits a batch across tasks
fn run_chunked(msgs: &[Message<'_>], out: &mut [[u8; 32]], chunk: usize) {
    run_chunked_with(hash_messages, msgs, out, chunk)
}

fn run_chunked_with(k: Kernel, msgs: &[Message<'_>], out: &mut [[u8; 32]], chunk: usize) {
    for (m, o) in msgs.chunks(chunk).zip(out.chunks_mut(chunk)) {
        k(m, o);
    }
}

/// Best of several rounds, as in the other benches: interference only ever
/// makes a round slower, so the minimum is the most stable estimate.
fn timed<F: FnMut()>(rounds: usize, iters: usize, mut f: F) -> (f64, f64) {
    for _ in 0..5 {
        f();
    }
    let (mut best, mut worst) = (f64::INFINITY, 0.0f64);
    for _ in 0..rounds {
        let t0 = Instant::now();
        for _ in 0..iters {
            f();
        }
        let per = t0.elapsed().as_secs_f64() / iters as f64;
        best = best.min(per);
        worst = worst.max(per);
    }
    (best, 100.0 * (worst - best) / best)
}

/// Aggregate throughput with `n` workers, each hashing its own batch
///
/// Measures throughput rather than latency: every worker runs its share
/// `ITERS` times with no per-iteration synchronisation, so the only sync is at
/// the round boundary and a straggler cannot distort the inner loop.
fn scale(n: usize, data: &[Vec<Vec<u8>>], bytes_per_batch: usize, chunk: usize) -> f64 {
    scale_with(hash_messages, n, data, bytes_per_batch, chunk)
}

fn scale_with(
    k: Kernel,
    n: usize,
    data: &[Vec<Vec<u8>>],
    bytes_per_batch: usize,
    chunk: usize,
) -> f64 {
    const ROUNDS: usize = 9;
    const ITERS: usize = 200;

    let per_thread: Vec<Vec<Message<'_>>> = (0..n)
        .map(|t| {
            data[t]
                .iter()
                .map(|leaf| Message::prefixed(PREFIX, leaf))
                .collect()
        })
        .collect();

    let mut best = f64::INFINITY;
    for _ in 0..ROUNDS {
        let t0 = Instant::now();
        std::thread::scope(|sc| {
            for msgs in per_thread.iter() {
                sc.spawn(move || {
                    let mut out = vec![[0u8; 32]; msgs.len()];
                    for _ in 0..ITERS {
                        run_chunked_with(k, msgs, &mut out, chunk);
                        std::hint::black_box(&out);
                    }
                });
            }
        });
        // Wall time for the whole cohort: n batches x ITERS hashed in it.
        let per_batch_aggregate = t0.elapsed().as_secs_f64() / (ITERS * n) as f64;
        best = best.min(per_batch_aggregate);
    }
    bytes_per_batch as f64 / best / 1e6
}

fn main() {
    let width = lane_width();
    let cores = std::thread::available_parallelism().map_or(4, |n| n.get());
    let max_threads = cores.min(16);

    let data: Vec<Vec<Vec<u8>>> = (0..max_threads).map(|t| batch(t as u8)).collect();
    let bytes_per_batch: usize = data[0].iter().map(|l| l.len() + PREFIX.len()).sum();

    println!(
        "backend {}, lane width {width}, {LEAVES} leaves/batch, \
         {bytes_per_batch} bytes/batch, {cores} logical cores",
        tape_sha256::backend()
    );

    // 1. Chunk granularity on one thread. A chunk that is not a multiple of
    //    the dispatched width drops that call to a narrower kernel, which no
    //    aggregate-throughput number would reveal on its own.
    println!("\nchunk granularity, single thread:");
    let msgs: Vec<Message<'_>> = data[0]
        .iter()
        .map(|l| Message::prefixed(PREFIX, l))
        .collect();
    let mut out = vec![[0u8; 32]; LEAVES];
    let mut baseline = 0.0;
    for chunk in [LEAVES, width * 2, width, width / 2, width.saturating_sub(1)] {
        if chunk == 0 || chunk > LEAVES {
            continue;
        }
        let (t, spread) = timed(9, 300, || {
            run_chunked(&msgs, &mut out, chunk);
            std::hint::black_box(&out);
        });
        let mbps = bytes_per_batch as f64 / t / 1e6;
        if baseline == 0.0 {
            baseline = mbps;
        }
        println!(
            "  chunk {chunk:>3}   {:>8.2} us/batch{:>10.0} MB/s   {:>6.1}% of best   (spread {spread:>4.1}%)",
            t * 1e6,
            mbps,
            100.0 * mbps / baseline
        );
    }

    // 2. Thread scaling at the good chunk size.
    println!("\nthread scaling, chunk {width} (one batch per worker):");
    let mut one = 0.0;
    for n in 1..=max_threads {
        let mbps = scale(n, &data, bytes_per_batch, width);
        if n == 1 {
            one = mbps;
        }
        println!(
            "  {n:>2} threads  {mbps:>9.0} MB/s aggregate   {:>5.2}x over one   {:>5.0}% per-thread efficiency",
            mbps / one,
            100.0 * (mbps / one) / n as f64
        );
    }

    // 3. x86 only: the wide AVX-512 kernel runs the 512-bit datapath flat out
    //    while the SHA-NI streams sit in a much lower power envelope, so their
    //    idle-core ratio need not survive all-core load. Every dispatch
    //    decision in the crate was taken from single-thread numbers; this is
    //    the check that the choice still holds when every core is busy.
    #[cfg(all(target_arch = "x86_64", not(feature = "scalar")))]
    {
        use tape_sha256::backends;
        if is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512bw") {
            let kernels: [(&str, Kernel, usize); 2] = [
                ("avx512-2x16", |m, o| unsafe { backends::avx512_16x2(m, o) }, 32),
                ("shani-x4", |m, o| unsafe { backends::shani_x4(m, o) }, 4),
            ];
            println!("\nkernel choice under load (aggregate MB/s):");
            print!("  {:<14}", "threads");
            for n in 1..=max_threads {
                print!("{n:>10}");
            }
            println!();
            let mut rows = Vec::new();
            for (name, k, chunk) in kernels {
                let mut row = Vec::new();
                print!("  {name:<14}");
                for n in 1..=max_threads {
                    let mbps = scale_with(k, n, &data, bytes_per_batch, chunk);
                    print!("{mbps:>10.0}");
                    row.push(mbps);
                }
                println!();
                rows.push(row);
            }
            print!("  {:<14}", "ratio");
            for i in 0..max_threads {
                print!("{:>10.2}", rows[0][i] / rows[1][i]);
            }
            println!("\n  (ratio > 1 means avx512-2x16 still wins at that occupancy)");
        }
    }
}
