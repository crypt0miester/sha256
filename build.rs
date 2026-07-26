//! Links Firedancer for the comparison benchmark, when asked.

fn main() {
    println!("cargo:rerun-if-env-changed=FD_LIB_DIR");

    if std::env::var_os("CARGO_FEATURE_FIREDANCER_BENCH").is_none() {
        return;
    }

    let Some(dir) = std::env::var_os("FD_LIB_DIR") else {
        panic!(
            "feature `firedancer-bench` is enabled but FD_LIB_DIR is unset.\n\
             Point it at a Firedancer build's lib directory, e.g.\n  \
             FD_LIB_DIR=/path/to/firedancer/build/native/gcc/lib"
        );
    };

    println!("cargo:rustc-link-search=native={}", dir.to_string_lossy());
    // fd_ballet holds the hash implementations; fd_util holds the shared
    // runtime symbols they depend on.
    println!("cargo:rustc-link-lib=static=fd_ballet");
    println!("cargo:rustc-link-lib=static=fd_util");
}
