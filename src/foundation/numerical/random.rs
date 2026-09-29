//! Seeded and entropy-seeded uniform parameter-value draws.
//!
//! This facility generates scalar values for model parameters. It does not
//! sample, shuffle, split, resample, or reorder observations; Seqvex's
//! temporal, sequential, non-IID data order is preserved
//! (`docs/RANDOMNESS.md`).
//!
//! The Rand dependency is confined to this module: callers see only
//! [`RandomGenerator`], [`UniformRange`], and [`RandomError`].

use std::error::Error;
use std::fmt;

use rand::SeedableRng;
use rand::TryRng;
use rand::distr::Distribution;
use rand::distr::Uniform;
use rand::rngs::ChaCha12Rng;
use rand::rngs::SysRng;

/// Opaque source retained for an operating-system entropy failure.
type EntropySource = Box<dyn Error + Send + Sync + 'static>;

/// A caller-owned generator for uniform parameter-value draws.
///
/// The engine is an implementation detail; it is never global, thread-local,
/// or model-owned, and advancement is explicit through `&mut self`.
#[derive(Debug)]
pub struct RandomGenerator {
    rng: ChaCha12Rng,
}

/// A validated uniform range for repeated parameter-value draws.
///
/// The nominal interval is half-open, `[low, high)`; Rand draws `f32` values
/// with approximate uniformity, and rounding may yield the nominal upper
/// bound.
#[derive(Debug)]
pub struct UniformRange {
    distribution: Uniform<f32>,
}

/// Failure classes for parameter-value generation.
#[derive(Debug)]
pub enum RandomError {
    /// The requested range is empty or not finite.
    InvalidRange,
    /// The operating system entropy source could not provide a seed.
    EntropyUnavailable {
        /// The underlying cause, kept opaque so no foreign type is exposed.
        source: EntropySource,
    },
}

impl fmt::Display for RandomError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidRange => formatter.write_str("uniform range must be finite and non-empty"),
            Self::EntropyUnavailable { .. } => {
                formatter.write_str("operating system entropy is unavailable")
            }
        }
    }
}

impl Error for RandomError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::EntropyUnavailable { source } => Some(source.as_ref() as &(dyn Error + 'static)),
            Self::InvalidRange => None,
        }
    }
}

impl RandomGenerator {
    /// Builds a generator from an explicit 64-bit seed.
    ///
    /// This is reproducible for a fixed dependency version but is not
    /// cryptographically secure.
    pub fn from_seed(seed: u64) -> Self {
        Self {
            rng: ChaCha12Rng::seed_from_u64(seed),
        }
    }

    /// Builds a generator seeded from operating-system entropy.
    ///
    /// Fails rather than panicking or falling back to a fixed seed when the
    /// entropy source is unavailable.
    pub fn from_entropy() -> Result<Self, RandomError> {
        Self::from_entropy_source(|seed| {
            SysRng
                .try_fill_bytes(seed)
                .map_err(|error| Box::new(error) as EntropySource)
        })
    }

    /// Validates `[low, high)` once for repeated parameter-value draws.
    pub fn uniform_range(&self, low: f32, high: f32) -> Result<UniformRange, RandomError> {
        Uniform::new(low, high)
            .map(|distribution| UniformRange { distribution })
            .map_err(|_| RandomError::InvalidRange)
    }

    /// Draws one value from a previously validated range.
    pub fn draw_f32(&mut self, range: &UniformRange) -> f32 {
        range.distribution.sample(&mut self.rng)
    }

    fn from_entropy_source(
        mut fill: impl FnMut(&mut [u8; 32]) -> Result<(), EntropySource>,
    ) -> Result<Self, RandomError> {
        let mut seed = [0_u8; 32];
        fill(&mut seed).map_err(|source| RandomError::EntropyUnavailable { source })?;
        Ok(Self {
            rng: ChaCha12Rng::from_seed(seed),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::error::Error;

    #[derive(Debug)]
    struct SeamError(&'static str);

    impl fmt::Display for SeamError {
        fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
            formatter.write_str(self.0)
        }
    }

    impl Error for SeamError {}

    #[test]
    fn entropy_seam_success_seeds_a_working_generator() {
        let mut generator = RandomGenerator::from_entropy_source(|seed| {
            seed.fill(7);
            Ok(())
        })
        .unwrap();
        let range = generator.uniform_range(0.0, 1.0).unwrap();
        let value = generator.draw_f32(&range);
        assert!((0.0..=1.0).contains(&value));
    }

    #[test]
    fn entropy_seam_failure_is_propagated_with_its_source() {
        let error =
            RandomGenerator::from_entropy_source(|_| Err(Box::new(SeamError("no entropy"))))
                .unwrap_err();
        match &error {
            RandomError::EntropyUnavailable { source } => {
                assert_eq!(source.to_string(), "no entropy");
                assert!(error.source().is_some());
            }
            other => panic!("expected entropy failure, got {other:?}"),
        }
    }

    #[test]
    fn invalid_ranges_are_rejected() {
        let generator = RandomGenerator::from_seed(1);
        assert!(matches!(
            generator.uniform_range(1.0, 1.0),
            Err(RandomError::InvalidRange)
        ));
        assert!(matches!(
            generator.uniform_range(1.0, 0.0),
            Err(RandomError::InvalidRange)
        ));
        assert!(matches!(
            generator.uniform_range(f32::NAN, 1.0),
            Err(RandomError::InvalidRange)
        ));
    }
}
