# Security policy

## Supported versions

Pre-1.0. Fixes land on the latest 0.1.x release; earlier patch versions are
not backported.

## What counts as a vulnerability here

This crate computes SHA-256 over data the caller already holds. It takes no
keys, does no I/O, and keeps no state between calls, so most of the usual
questions do not arise. SHA-256's control flow and memory access are fixed by
the algorithm rather than by the message, on every backend here, so there is
no message-dependent timing to leak.

Two things are worth reporting.

**A wrong digest.** Output that differs from conforming SHA-256, on any
backend, message length, or batch size. This is the one that matters. Callers
use these digests as Merkle commitments, so a single wrong digest on a single
backend is a divergence between machines that happened to select different
kernels. Report it even if you cannot tell which kernel produced it.

**Memory unsafety.** Every SIMD backend is unsafe code, and the batch driver
reads message bodies through raw pointers. A crash, a read outside a buffer
the caller supplied, or a finding from miri or a sanitizer is in scope.

Out of scope: the cryptographic properties of SHA-256 itself, and performance
regressions.

## Reporting

Use private vulnerability reporting on this repository, under the Security
tab. That opens a thread visible only to the maintainers. Please do not open a
public issue for a wrong-digest or memory-safety report before there is a fix.

Include the CPU model and the backend name. `tape_sha256::backend()` returns
the kernel the machine actually selected, and it is the first thing needed to
reproduce a report: a bug in one kernel is invisible on hardware that
dispatches somewhere else.

## What is already gated

Every backend is checked differentially against the independent `sha2` crate
across message lengths covering all block and padding edge cases, cross-checked
against every other backend, and the error-prone shuffle machinery is validated
against scalar models of the intrinsic semantics. The suite has been run to
completion on AMD Zen 3, Zen 4 and Zen 5, Intel Emerald Rapids and Granite
Rapids, and Apple M-series.

A kernel only executes where its instructions exist, so a green run on one
machine compiles the other backends without exercising them. A report from
hardware that is not on that list is genuinely useful.
