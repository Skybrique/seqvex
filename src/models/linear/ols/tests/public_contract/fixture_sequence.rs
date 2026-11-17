//! Exact old-versus-new sequence checks for the shared fixture LCGs.
//!
//! `OldLcg64`/`OldLcg32` are independent copies of the pre-T13 algorithms; they
//! are the oracle. The literal vectors add a second, implementation-independent
//! record for a small set of draws. The shared generator is never compared to
//! itself.

use super::support::fixture_rng::{Lcg32, Lcg64};

struct OldLcg64 {
    state: u64,
}

impl OldLcg64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn advance(&mut self) {
        self.state = self
            .state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
    }

    fn next_f32(&mut self) -> f32 {
        self.advance();
        ((self.state >> 40) as f32 / (1_u64 << 24) as f32) - 0.5
    }

    fn next_f64(&mut self) -> f64 {
        self.advance();
        ((self.state >> 40) as f64 / (1_u64 << 24) as f64) - 0.5
    }

    fn next_unit(&mut self) -> f64 {
        self.advance();
        ((self.state >> 11) as f64) / ((1_u64 << 53) as f64)
    }
}

struct OldLcg32 {
    state: u32,
}

impl OldLcg32 {
    fn new(seed: u32) -> Self {
        Self { state: seed }
    }

    fn next_unit(&mut self) -> f64 {
        self.state = self
            .state
            .wrapping_mul(1_664_525)
            .wrapping_add(1_013_904_223);
        f64::from(self.state) / 4_294_967_296.0
    }
}

const LCG64_SEEDS: &[u64] = &[
    0x5eed_1234_abcd_0001,
    0x5eed_0a11_ce00_0001,
    0x0051_0ee1_0000_0001,
    0x5eed_55d0_0000_0001,
    0x5eed_2500_0000_0001,
    0x34ba_0000,
    0x34ba_0080,
    0x34ba_0800,
    0x34ba_8000,
    0x5eed_5eed_5eed_5eed,
    0x5eed_0001,
    0xa0c0_0001,
    0xfeed_0001,
    0xfeed_0002,
    7,
    0xC0FFEE ^ 2,
    0xC0FFEE ^ 4,
    0xC0FFEE ^ 8,
    0xC0FFEE ^ 16,
    0xC0FFEE ^ 32,
    0xC0FFEE ^ 64,
    0xA11CE ^ 4 ^ 200,
    0xA11CE ^ 4 ^ 1000,
    0xA11CE ^ 4 ^ 5000,
    0xA11CE ^ 4 ^ 10_000,
    0xA11CE ^ 16 ^ 200,
    0xA11CE ^ 16 ^ 1000,
    0xA11CE ^ 16 ^ 5000,
    0xA11CE ^ 16 ^ 10_000,
    0xF0F0,
    0xD3D3 ^ 4,
    0xD3D3 ^ 16,
    0xD4D4 ^ 4,
    0xD4D4 ^ 16,
    0xD5D5 ^ 4,
    0xD5D5 ^ 16,
    0x5CA1E ^ 2,
    0x5CA1E ^ 8,
    0x5CA1E ^ 16,
    0x9999 ^ 2,
    0x9999 ^ 4,
];

const LCG64_UNIT_SEEDS: &[u64] = &[
    0x1234_5678_9abc_def0,
    0x0bad_c0de_0bad_c0de,
    0xdead_beef_dead_beef,
    0xfeed_face_feed_face,
    0x00c0_ffee_00c0_ffee,
    0x5151_5151_5151_5151,
    0x7f7f_7f7f_7f7f_7f7f,
    0xabcd_abcd_abcd_abcd,
];

const LCG32_SEEDS: &[u32] = &[20_261_016, 20_261_017, 20_261_018, 20_261_019];
const DRAWS: usize = 257;

fn assert_lcg64_f32(seed: u64) {
    let mut old = OldLcg64::new(seed);
    let mut new = Lcg64::new(seed);
    for draw in 0..DRAWS {
        assert_eq!(
            new.next_f32().to_bits(),
            old.next_f32().to_bits(),
            "centered f32 mismatch: seed=0x{seed:016x} draw={draw}"
        );
    }
    assert_eq!(
        new.state, old.state,
        "centered f32 state mismatch: seed=0x{seed:016x}"
    );
}

fn assert_lcg64_f64(seed: u64) {
    let mut old = OldLcg64::new(seed);
    let mut new = Lcg64::new(seed);
    for draw in 0..DRAWS {
        assert_eq!(
            new.next_f64().to_bits(),
            old.next_f64().to_bits(),
            "centered f64 mismatch: seed=0x{seed:016x} draw={draw}"
        );
    }
    assert_eq!(
        new.state, old.state,
        "centered f64 state mismatch: seed=0x{seed:016x}"
    );
}

fn assert_lcg64_unit(seed: u64) {
    let mut old = OldLcg64::new(seed);
    let mut new = Lcg64::new(seed);
    for draw in 0..DRAWS {
        assert_eq!(
            new.next_unit().to_bits(),
            old.next_unit().to_bits(),
            "unit f64 mismatch: seed=0x{seed:016x} draw={draw}"
        );
    }
    assert_eq!(
        new.state, old.state,
        "unit f64 state mismatch: seed=0x{seed:016x}"
    );
}

fn assert_lcg32(seed: u32) {
    let mut old = OldLcg32::new(seed);
    let mut new = Lcg32::new(seed);
    for draw in 0..DRAWS {
        assert_eq!(
            new.next_unit().to_bits(),
            old.next_unit().to_bits(),
            "u32 unit mismatch: seed={seed} draw={draw}"
        );
    }
    assert_eq!(new.state, old.state, "u32 state mismatch: seed={seed}");
}

#[test]
fn fixture_sequence_centered_f32_matches_old_definition() {
    for &seed in LCG64_SEEDS {
        assert_lcg64_f32(seed);
    }
    for index in 0..24u64 {
        assert_lcg64_f32(0x5eed_0000 + index);
    }
}

#[test]
fn fixture_sequence_centered_f64_matches_old_definition() {
    for &seed in LCG64_SEEDS {
        assert_lcg64_f64(seed);
    }
    for index in 0..24u64 {
        assert_lcg64_f64(0x5eed_0000 + index);
    }
}

#[test]
fn fixture_sequence_unit_matches_old_definition() {
    for &seed in LCG64_UNIT_SEEDS {
        assert_lcg64_unit(seed);
    }
}

#[test]
fn fixture_sequence_u32_state_and_unit_match_old_definition() {
    for &seed in LCG32_SEEDS {
        assert_lcg32(seed);
    }
}

#[test]
fn fixture_sequence_literal_vectors_match_both_definitions() {
    const F32_BITS: [u32; 4] = [0x3e3578e4, 0xbdf24428, 0xbd878430, 0xbdc7a738];
    const F64_BITS: [u64; 4] = [
        0x3fc6af1c80000000,
        0xbfbe488500000000,
        0xbfb0f08600000000,
        0xbfb8f4e700000000,
    ];
    const UNIT_BITS: [u64; 4] = [
        0x3fe5abc72a00a538,
        0x3fd86ddecd11d3be,
        0x3fdbc3de9af82d18,
        0x3fd9c2c647da3872,
    ];
    const U32_UNIT_BITS: [u64; 4] = [
        0x3fdc49a4c5c00000,
        0x3fe6e7d2b1400000,
        0x3fb4ab8761000000,
        0x3feb31cf29800000,
    ];

    let mut old = OldLcg64::new(0x5eed_5eed_5eed_5eed);
    let mut new = Lcg64::new(0x5eed_5eed_5eed_5eed);
    for (index, expected) in F32_BITS.iter().enumerate() {
        assert_eq!(old.next_f32().to_bits(), *expected);
        assert_eq!(new.next_f32().to_bits(), *expected, "draw {index}");
    }

    let mut old = OldLcg64::new(0x5eed_5eed_5eed_5eed);
    let mut new = Lcg64::new(0x5eed_5eed_5eed_5eed);
    for (index, expected) in F64_BITS.iter().enumerate() {
        assert_eq!(old.next_f64().to_bits(), *expected);
        assert_eq!(new.next_f64().to_bits(), *expected, "draw {index}");
    }

    let mut old = OldLcg64::new(0x5eed_5eed_5eed_5eed);
    let mut new = Lcg64::new(0x5eed_5eed_5eed_5eed);
    for (index, expected) in UNIT_BITS.iter().enumerate() {
        assert_eq!(old.next_unit().to_bits(), *expected);
        assert_eq!(new.next_unit().to_bits(), *expected, "draw {index}");
    }

    let mut old = OldLcg32::new(20_261_016);
    let mut new = Lcg32::new(20_261_016);
    for (index, expected) in U32_UNIT_BITS.iter().enumerate() {
        assert_eq!(old.next_unit().to_bits(), *expected);
        assert_eq!(new.next_unit().to_bits(), *expected, "draw {index}");
    }
}
