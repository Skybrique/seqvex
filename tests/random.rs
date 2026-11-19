//! Contract tests for the parameter-value generator facility.
//!
//! These cover seeded reproducibility, bounds, and range validation. The KAT
//! values are regression guards for a fixed dependency version; they do not
//! prove PRNG statistical quality (`docs/RANDOMNESS.md`). The entropy-seeding
//! seam is tested inside `src/foundation/numerical/random.rs`; automated tests
//! never call the real operating-system entropy path.

use seqvex::foundation::numerical::{RandomError, RandomGenerator};

const KAT_SEED: u64 = 0x5eed_5eed;
const KAT_BITS: [u32; 8] = [
    3211150720, 1045364496, 3195397184, 3210521216, 3198186664, 3200070664, 3210733200, 3207483648,
];

#[test]
fn seeded_draws_match_frozen_vector() {
    let mut generator = RandomGenerator::from_seed(KAT_SEED);
    let range = generator.uniform_range(-1.0, 1.0).unwrap();
    for (index, expected) in KAT_BITS.into_iter().enumerate() {
        let value = generator.draw_f32(&range);
        assert_eq!(value.to_bits(), expected, "index {index}");
    }
}

#[test]
fn same_seed_repeats_and_distinct_seeds_differ() {
    let mut left = RandomGenerator::from_seed(7);
    let mut right = RandomGenerator::from_seed(7);
    let mut other = RandomGenerator::from_seed(8);
    let range = left.uniform_range(-1.0, 1.0).unwrap();
    let other_range = other.uniform_range(-1.0, 1.0).unwrap();
    let left_values: Vec<u32> = (0..16).map(|_| left.draw_f32(&range).to_bits()).collect();
    let right_values: Vec<u32> = (0..16).map(|_| right.draw_f32(&range).to_bits()).collect();
    let other_values: Vec<u32> = (0..16)
        .map(|_| other.draw_f32(&other_range).to_bits())
        .collect();
    assert_eq!(left_values, right_values);
    assert_ne!(left_values, other_values);
}

#[test]
fn draws_stay_within_the_requested_bounds() {
    let mut generator = RandomGenerator::from_seed(99);
    let low = -0.25_f32;
    let high = 0.75_f32;
    let range = generator.uniform_range(low, high).unwrap();
    for _ in 0..10_000 {
        let value = generator.draw_f32(&range);
        assert!(value.is_finite());
        assert!(
            (low..=high).contains(&value),
            "value {value} outside [{low}, {high}]"
        );
    }
}

#[test]
fn invalid_ranges_are_rejected() {
    let generator = RandomGenerator::from_seed(3);
    assert!(matches!(
        generator.uniform_range(0.5, 0.5),
        Err(RandomError::InvalidRange)
    ));
    assert!(matches!(
        generator.uniform_range(1.0, -1.0),
        Err(RandomError::InvalidRange)
    ));
    assert!(matches!(
        generator.uniform_range(f32::INFINITY, 1.0),
        Err(RandomError::InvalidRange)
    ));
}

const KAT_U64_BOUNDS: [u64; 5] = [1, 2, 3, 1 << 53, (1 << 63) + 1];
const KAT_U64_VALUES: [u64; 15] = [
    0,
    0,
    0,
    0,
    1,
    0,
    0,
    1,
    2,
    3_983_747_634_605_040,
    8_145_224_125_137_271,
    8_303_390_446_523_346,
    1_148_991_139_499_910_279,
    1_877_847_795_317_891_071,
    4_426_926_430_244_945_454,
];

#[test]
fn draw_u64_below_matches_frozen_vector() {
    let mut generator = RandomGenerator::from_seed(KAT_SEED);
    let mut index = 0;
    for bound in KAT_U64_BOUNDS {
        for _ in 0..3 {
            let value = generator.draw_u64_below(std::num::NonZeroU64::new(bound).unwrap());
            assert!(value < bound, "index {index}: {value} outside [0, {bound})");
            assert_eq!(value, KAT_U64_VALUES[index], "index {index}");
            index += 1;
        }
    }
}
