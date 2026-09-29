# Seqvex Randomness and Parameter Initialization

> **Status:** Scoped facility for issue #20. Not a project-wide randomness standard.

## 1. Purpose and scope invariant

This facility generates scalar values for **model parameters** (parameter-value
draws). It is not a data-selection mechanism.

> Seqvex is temporal, sequential, and non-IID: the order of observations is
> semantically meaningful. This facility does not sample, shuffle, permute,
> split, resample, or reorder observations.

Concretely, it does not provide shuffled mini-batches, random temporal splits,
data loaders, observation resampling, or sequence-window reordering. No part of
it changes GRU execution order, streaming, state advancement, resets,
`StateModel`, `GruExecutor`, hidden `State`, or `Workspace`.

## 2. Use cases

| Use case | Reproducibility | Distribution |
|---|---|---|
| GRU parameter initialization (untrained) | caller chooses `RandomGenerator::from_entropy()` (fallible OS entropy) or `RandomGenerator::from_seed(seed)` (reproducible); identical seed + dims + call order ⇒ identical values within a release | bounded uniform |
| Explicitly seeded parameter experiments | same seed + dims + call order ⇒ identical values within a release | bounded uniform |
| Tests / reference / benchmarks | deterministic fixture (`GruParameters::deterministic`), no entropy | fixed LCG fixture |

Future non-uniform parameter generation is out of scope until a concrete
consumer exists. Data sampling/ordering is out of scope and requires a separate
architecture decision that preserves temporal dependencies and prevents leakage.

## 3. Dependency and reversibility

The implementation uses the `rand` 0.10 facade with
`default-features = false, features = ["std", "chacha", "sys_rng"]`:

- `chacha` enables the named `rand::rngs::ChaCha12Rng` (Rand 0.10 implements its
  ChaCha RNG via `chacha20`, so this uses the `chacha20` engine through Rand).
- `sys_rng` enables `rand::rngs::SysRng` for fallible operating-system entropy.
- `std_rng`, `thread_rng`, `log`, `serde`, `simd_support`, and `unbiased` are not
  enabled.

All `rand`/`getrandom` types and errors are confined to
`src/foundation/numerical/random.rs`. Public Seqvex APIs expose only
`RandomGenerator`, `UniformRange`, and `RandomError`. A future engine or
distribution swap is localized to that module, the dependency configuration, and
that module's tests. This is a scoped choice, not a permanent project-wide RNG
standard.

## 4. Generator contract

- `RandomGenerator::from_seed(seed: u64)` — explicit seed; reproducible for a
  fixed dependency version; not cryptographically secure.
- `RandomGenerator::from_entropy() -> Result<RandomGenerator, RandomError>` —
  seeds from OS entropy; fallible; never panics and never falls back to a fixed
  seed or fixture.
- `RandomGenerator::uniform_range(low, high) -> Result<UniformRange, RandomError>`
  — validates a finite, non-empty range once.
- `RandomGenerator::draw_f32(&UniformRange) -> f32` — infallible single draw.

The generator is caller-owned and explicitly advanced through `&mut self`. It is
never global, thread-local, or model-owned.

## 5. Errors

- `RandomError::InvalidRange` — empty or non-finite range.
- `RandomError::EntropyUnavailable { source }` — the OS entropy source failed;
  the cause is retained opaquely so no foreign error type appears in the public
  API.

No new `GruError` variants are introduced. The GRU initializer returns
`GruError::ZeroDimension` for zero dimensions; entropy is acquired by the caller
before initialization, so no error conversion crosses the boundary. The
project-wide error architecture remains intentionally open.

## 6. GRU initializer

`GruParameters::init(input_dim, hidden_dim, &mut RandomGenerator)` fills:

- input matrices `W_z, W_r, W_h` (`hidden_dim × input_dim`) with Glorot/Xavier
  uniform `U(-a, a)`, `a = sqrt(6 / (input_dim + hidden_dim))`;
- recurrent matrices `U_z, U_r, U_h` (`hidden_dim × hidden_dim`) with Glorot
  uniform `U(-a, a)`, `a = sqrt(6 / (2·hidden_dim)) = sqrt(3 / hidden_dim)`;
- biases `b_z, b_r, b_h` with `0.0` (they consume no draws).

For a uniform `[-a, a]` draw, variance is `a²/3`: `2/(input_dim + hidden_dim)`
for input matrices and `1/hidden_dim` for recurrent matrices. Recurrent Glorot
variance `1/hidden_dim` is three times PyTorch's documented `U(-1/√H, 1/√H)`
recurrent variance `1/(3H)`; these are different choices, not equivalent
formulations. Glorot's variance-scaling rationale does not by itself prove
recurrent-gradient stability, long-sequence behavior, or fitness for any
particular dataset.

The initializer is an **untrained starting point**. It is not trained,
predictive, or competition-ready, and it does not replace training or loaded
weights.

## 7. Reproducibility

- Same seed + dimensions + call order ⇒ identical generated values within a
  release, for a fixed dependency version.
- Rand's named portable generators reproduce across platforms and patch
  releases; minor releases may change values. Cross-release reproduction is a
  documented **non-guarantee**; the resolved versions are pinned in `Cargo.lock`.
- Rand's `Uniform<f32>` is approximately uniform over a nominal `[low, high)`;
  rounding may yield the nominal upper bound. Exact mathematical uniformity and
  strict upper-bound exclusion are not claimed.
- The deterministic fixture `GruParameters::deterministic` is unchanged and
  entropy-free; its values must not change.
- A fixed-weight inference path needs no runtime randomness.

## 8. Verification

Seeded reproducibility and the initializer's shape/finiteness/bounds are covered
by tests; the entropy-seeding seam is tested with controlled success and failure
sources and automated tests never call the real OS entropy path. Statistical
sanity checks (mean/variance scale) are sanity only; they do not prove PRNG
quality. Rust 1.85 support is declared (`rust-version = "1.85"`) and was
verified for this revision by `cargo +1.85.1 test --all-targets --all-features`.
