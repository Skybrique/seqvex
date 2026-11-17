//! Frozen fixture linear congruential generators shared by tests and benches.
//!
//! These are deterministic test/benchmark fixtures, not production randomness.
//! The `u64` generator preserves the existing centered-`f32` advance/draw
//! contract and adds the matching centered-`f64` and unit-interval extractions.
//! The `u32` generator preserves the advance-before-draw unit contract.
//!
//! This file deliberately imports nothing: it has no allocator, benchmark
//! harness, randomness-crate or production dependency. Consumers include it
//! directly with `#[path]`; `benches/common/mod.rs` re-exports only `Lcg64` as
//! `Lcg` for the existing bench facade.

#![allow(dead_code)] // shared by test/bench consumers that use different widths.

/// Fixed-seed `u64` LCG shared by the matching fixture consumers.
///
/// State is exposed only within the including crate so the exact-state checks
/// can compare independent old and new implementations.
pub struct Lcg64 {
    pub(crate) state: u64,
}

impl Lcg64 {
    pub fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    pub fn next_f32(&mut self) -> f32 {
        self.advance();
        ((self.state >> 40) as f32 / (1_u64 << 24) as f32) - 0.5
    }

    pub fn next_f64(&mut self) -> f64 {
        self.advance();
        ((self.state >> 40) as f64 / (1_u64 << 24) as f64) - 0.5
    }

    pub fn next_unit(&mut self) -> f64 {
        self.advance();
        ((self.state >> 11) as f64) / ((1_u64 << 53) as f64)
    }

    fn advance(&mut self) {
        self.state = self
            .state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
    }
}

/// Fixed-seed `u32` LCG shared by the matching advance-before-draw consumers.
pub struct Lcg32 {
    pub(crate) state: u32,
}

impl Lcg32 {
    pub fn new(seed: u32) -> Self {
        Self { state: seed }
    }

    pub fn next_unit(&mut self) -> f64 {
        self.state = self
            .state
            .wrapping_mul(1_664_525)
            .wrapping_add(1_013_904_223);
        f64::from(self.state) / 4_294_967_296.0
    }
}
