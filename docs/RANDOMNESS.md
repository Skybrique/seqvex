# Seqvex Randomness Guide

> **Initialization is not fitting:** this guide covers starting values, selection draws, and fixtures. It does not supply learned parameters or a validated user model. Every ML model still requires actual data ingestion, training/learning and validated inference under the [mandatory training-to-inference contract](DEVELOPMENT.md#model-training-and-valid-inference--mandatory). Entropy failure must remain explicit; no fallback is inferred.

> **Status:** Project-wide guide to randomness in the Rust library. The shared foundation supplies draws; consumers own initialization, selection, ordering, and replay policy. Sampling delivery status is recorded in §8.

## 1. Vocabulary and boundary

| Term | Meaning |
|---|---|
| Draw | One value or state advancement produced by a generator. |
| Generator | Caller-owned state that produces draws. |
| Policy | What a consumer does with draws: initialization, selection, ordering, replay. |
| Fixture | A fixed, non-entropy value source used by a test or benchmark. |

The shared implementation is `RandomGenerator`, `UniformRange`, and `RandomError` in `src/foundation/numerical/random.rs`. The `rand` dependency is confined to that module; callers do not see `rand` types.

The foundation supplies draws. It does not decide which observations are eligible, how a sample is ordered, or what a seed means to a workflow. Those are consumer contracts.

This boundary is about ownership, not capability. The foundation exposes both parameter-value draws (`draw_f32`) and a data-agnostic integer draw (`draw_u64_below`) used by `preprocessing::sampling`. The sampling policy itself remains in `preprocessing::sampling`; the foundation does not select rows.

## 2. How randomness is chosen

There are three implemented sources of random values. They answer different questions and are not interchangeable evidence.

### 2.1 Fixed deterministic fixtures

A deterministic fixture is a fixed algorithm or table that produces values without entropy or a caller seed. It exists for repeatable tests and benchmarks.

- `GruParameters::deterministic` (`src/models/recurrent/gru.rs`) is the GRU fixture: a fixed 64-bit LCG, no entropy, same values for every run. It must not change; it is a named fixture, not the public generator.
- Benchmarks use the fixture so the measured workload is identical between runs.
- Fixture-stability claims are scoped to the toolchains and targets named in the owning test evidence. The verification recorded for this revision was Rust 1.97.1 (stable) and Rust 1.85.1 on x86_64. Untested targets and releases are not covered.
- Fixture equivalence and public randomness are separate claims (§3). A fixture check does not establish the behavior, distribution, or failure handling of the public generator.

### 2.2 Caller-seeded pseudorandom generation

`RandomGenerator::from_seed(seed: u64)` accepts a caller-provided `u64` seed for replay. It is infallible and reproducible for a fixed dependency version. On this path the seed is a reproducibility control, not a secret; the facility makes no cryptographic-security claim for caller-chosen seeds.

The generator is caller-owned and advanced explicitly through `&mut self`. It is never global, thread-local, or model-owned.

Implemented draws:

- `draw_f32(&UniformRange) -> f32` — one value from a range validated by `uniform_range(low, high)`. The nominal interval is half-open `[low, high)`; `rand` draws `f32` values with approximate uniformity, and rounding may yield the nominal upper bound.
- `draw_u64_below(upper: NonZeroU64) -> u64` — one uniformly distributed index in `[0, upper)`. The `NonZeroU64` type makes a zero bound unconstructible, and the draw uses the dependency's constructed integer sampler rather than a one-off range shortcut.

### 2.3 OS-entropy seeding

`RandomGenerator::from_entropy() -> Result<RandomGenerator, RandomError>` fills a 32-byte seed from the operating-system entropy source (`SysRng`) and initializes the algorithm from that seed. It is the path for a run that requests an OS-derived starting state rather than replay.

- An entropy-seeded generator may differ between runs because its starting seed differs. Once initialized, it follows its algorithm deterministically from that state.
- Entropy failure is explicit: `RandomError::EntropyUnavailable { source }`. The cause is retained opaquely and exposed through `source()`; no foreign error type crosses the public API.
- The constructor never panics and never falls back to a fixed seed. The controlled RNG tests exercise the entropy seam with controlled success and failure sources and never call the real OS entropy path.

### 2.4 What each implemented API exposes

| API | Entropy choice | Seed choice | Failure |
|---|---|---|---|
| `RandomGenerator::from_seed` | none | caller | infallible |
| `RandomGenerator::from_entropy` | OS entropy | drawn by the OS | `RandomError::EntropyUnavailable` |
| `RandomGenerator::uniform_range` | none | none | `RandomError::InvalidRange` |
| `RandomGenerator::draw_f32` | none | none | infallible |
| `RandomGenerator::draw_u64_below` | none | none | infallible for a non-zero bound |
| `GruParameters::deterministic` | none | fixed fixture | infallible |

The foundation never chooses entropy for a caller. A caller that wants entropy asks for it; a caller that wants replay passes a seed.

## 3. What a caller can replay

Replay is an operational property of a whole call path, not of the generator alone. To reproduce a seeded result, hold fixed:

1. the inputs (values, shapes, row sets, ranges, sequence ids, outcome supports, and any transformations);
2. the seed;
3. the exact draw order, including generator construction count, ranges constructed, and the interleaving of draws and other work;
4. the release, the pinned dependency versions, the toolchain, and the target.

Within that envelope, each draw API replays its own stream under the stated conditions:

- `draw_f32` replays its stream for the same seed, range construction, and call order (`tests/random.rs`, frozen KAT).
- `draw_u64_below` replays its stream for the same seed, bounds, and call order (`tests/random.rs`, frozen KAT).
- GRU initialization is bitwise reproducible for the same seed, dimensions, and draw order (`tests/gru.rs`, `initializer_is_reproducible_for_a_fixed_seed` and `initializer_matches_frozen_parameter_vector`).
- Training-fold sampling is reproducible for the same units, `SamplingSpec`, and seed (`tests/data_sampler.rs`, `seeded_replay_and_grouped_multiplicity` and `frozen_cutoff_24_sample_matches_seed_31`).

Cross-release, cross-toolchain, and cross-platform bitwise equality is **not** promised. Rand minor releases may change streams; `Cargo.lock` pins the resolved versions (currently `rand` 0.10.3, `rand_core` 0.10.1, `chacha20` 0.10.2). Floating-point weights in sequential bootstrap are same-executable, not cross-toolchain bitwise.

### 3.1 What known-answer tests establish

A KAT pins a seed, an order of operations, and the exact output bits or values. It detects:

- a changed dependency stream;
- a changed draw order or range construction;
- a changed conversion or output path.

It does **not** establish that the generator has a particular period, that its output is statistically suitable for a workload, or that a seed is secret. Keep three properties distinct:

| Property | What it means | Evidence |
|---|---|---|
| Period | The algorithm's cycle length | Dependency documentation; not tested here |
| Statistical quality | Distributional suitability | Dependency documentation plus scoped consumer-scale sanity checks; not proven by KATs or fixtures |
| Reproducibility | Same state and call order produce the same result | Seeded replay tests and KATs |

A different seed is expected to produce a different stream (`same_seed_repeats_and_distinct_seeds_differ`). An entropy-seeded generator is expected to differ between runs; that is the point of entropy, not a defect.

## 4. Who owns each decision

| Decision | Owner | Implemented rule |
|---|---|---|
| Generator algorithm and uniform-draw semantics | Foundation | `src/foundation/numerical/random.rs`; `rand` confined there |
| Seed source: fixed seed or OS entropy | Consumer | The foundation offers both constructors; it never decides for the caller |
| Generator lifetime and advancement | Consumer | Caller-owned `&mut RandomGenerator`; no global, thread-local, or model-owned generator |
| Draw order and mapping draws to a domain | Consumer | Stated in the consumer's contract; tests freeze it per consumer |
| Data eligibility, cutoffs, holdout membership | Splitter / caller | The sampler never receives a fold or a test range |
| Sampling method, multiplicity, presentation order, shuffle | Consumer (`preprocessing::sampling`) | `SamplingSpec` and `sample_training` |
| Fixture identity | Test or benchmark owner | Named deterministic fixtures, kept separate from public randomness |

### 4.1 Implemented workflows

- **GRU initialization** (`GruParameters::init`): the caller supplies `&mut RandomGenerator`, so the caller chooses `from_seed` or `from_entropy`. The initializer constructs the input and recurrent ranges, then draws the weight matrices in the order written in `init`; biases are zero and consume no draws.
- **Training-fold sampling** (`sample_training`): one generator is constructed per call from the seed passed to `SamplingSpec::new(...)`. Sampling draws happen first, then optional shuffle draws, then row expansion. The sampler does not derive seeds, keep state across calls, or consult an external generator.

### 4.2 Decisions still needing approval

These are not implemented; do not infer them from the existence of a seed or a generator:

- **Per-member or per-fold seed derivation.** The sampler takes one explicit seed. A caller policy such as `mix(evaluation_seed, fold_id, member_id)` is a recommendation, not an implemented mixer. Derivation policy and mixer identity require approval.
- **Parallel draws.** There is no partitioning contract for drawing concurrently or for assigning independent streams to threads. Do not share one `RandomGenerator` across threads without a designed ownership and ordering rule.
- **Checkpoint and replay across calls.** `RandomGenerator` exposes no state serialization or restore API, and the sampler is per-call. Checkpoint/restore and long-lived sampling state are deferred.
- **Entropy policy for training workflows.** Whether a workflow seeds from entropy, a recorded seed, or a fixture is a consumer decision that must be documented by that consumer.

## 5. Randomness and time-ordered data

Seqvex data is temporal, sequential, and non-IID. Randomness may not silently destroy eligibility, order, or holdout isolation.

### 5.1 Eligibility, cutoffs, and holdout

- The splitter owns fold geometry, eligibility, cutoffs, diagnostics, and test ranges.
- The caller establishes cutoff eligibility **before** any sampling draw: check that the fold is usable (`TimeFold::outcome()`), then build sampling units from `TimeFold::train_ranges()` only.
- Holdout membership is preserved by construction. The sampler never receives the fold, the cutoff, or the test range; it returns indices into caller-owned data and cannot itself prove isolation. That provenance remains the caller's obligation.
- `SampleUnit::outcomes` is caller-declared realized-outcome support, not label availability. Supports are built from eligible training observations only, from facts available at the fold cutoff; a span that needs post-cutoff information is a hindsight error.
- Sampling does not change the splitter, its cutoffs, or its held-out membership.

### 5.2 Sampling API and defaults

Resampling is opt-in. A caller enables it by selecting a non-`Disabled` `SamplingMode`; `SamplingSpec::disabled()` builds the identity configuration. The following API is on open PR #104 and is not yet on `main` (§8).

- `SampleUnit { rows: Vec<Range<usize>>, outcomes: Vec<OutcomeInterval>, sequence: u32 }`
- `OutcomeInterval { start: i64, end: i64 }` (half-open)
- `SamplingMode::{Disabled, Default, Method(SampleMethod)}`
- `SampleMethod::{SequentialBootstrap, WithoutReplacement, WithReplacement}`
- `SamplingSpec::new(mode, sample_size, seed)`, `SamplingSpec::disabled()`, `SamplingSpec::with_shuffle(bool)`
- `sample_training(&[SampleUnit], SamplingSpec) -> Result<TrainingSample, SamplingError>`
- `TrainingSample::{row_indices, unit_counts, len, is_empty}`

A public constructor example:

```rust
use seqvex::preprocessing::sampling::{SampleMethod, SamplingMode, SamplingSpec};

// This API is on open PR #104 and is not yet on `main`.
// Explicit standard sampling with replacement; `shuffle` starts disabled.
let replayable = SamplingSpec::new(
    SamplingMode::Method(SampleMethod::WithReplacement),
    64,
    0x5eed_5eed,
);

// The same sampling draws, then whole-occurrence shuffling within sequences.
let shuffled = replayable.with_shuffle(true);
```

Semantics:

| Mode | Behavior |
|---|---|
| `Disabled` | Every supplied unit once. `sample_size` is ignored. |
| `Default` | Selects sequential bootstrap once resampling is enabled; requires suitable outcome-support metadata, otherwise `MissingOutcomeSupport`. |
| `Method(SequentialBootstrap)` | As-if-appended average uniqueness with replacement. |
| `Method(WithoutReplacement)` | `sample_size` distinct units; requires `sample_size <= unit count`. |
| `Method(WithReplacement)` | `sample_size` uniform unit draws; duplicates retained. |

- **Defaults.** `SamplingMode::Default` selects sequential bootstrap once resampling is enabled. `shuffle` is independent and defaults to `false`; `SamplingSpec::new(...)` starts with it off.
- **Whole-unit selection and multiplicity.** A unit is drawn whole or not at all. `unit_counts()[i]` is the draw count for unit `i`; `row_indices()` expands every draw to all of the unit's rows, in declared range order. Repeated selections are retained.
- **Fragmented units.** A unit may own ordered, disjoint, non-adjacent ranges, so eligible–ineligible–eligible rows at one decision tick stay one draw.
- **Sequential bootstrap (default).** Gives greater weight to eligible units whose declared support overlaps less with units already drawn. For each draw it recomputes every candidate's average uniqueness as if that candidate were appended now, selects by that weight, then updates the concurrency of the selected support. Repeats are allowed.
  For a fragmented unit the support is the declared disjoint union of spans: overlap is measured on the union, a hole contributes no ticks, and the group counts once per draw rather than once per member row. The rule is length-weighted:
  `u_i = (1/|T_i|) Σ_{t∈T_i} 1/(1 + c(t))`, `w_i = u_i`.
  Selection uses unnormalized weights, `target = u·Σw`, the first strict `>` win, and a last-positive fallback.
- **Presentation order.** Draw units, order occurrences into source order within each sequence, optionally permute whole occurrences within each sequence, then expand rows.
- **Shuffle.** With shuffle off, source order is preserved within each sequence and repeat multiplicity is retained. With shuffle on, Fisher–Yates runs within each sequence after the sampling draws and moves whole sampled occurrences; a fragmented unit moves as one occurrence, its internal row order is preserved, and sequences are never interleaved.
- **Errors before draws.** Validation, applicability, capacity, and work checks complete before the generator is constructed. Every typed preflight failure consumes zero draws, verified by a counting seam; this includes the reachable sequential work-overflow fixture.

Sequential bootstrap requires suitable overlap metadata for every unit. Missing required overlap metadata produces `MissingOutcomeSupport`. A mode that cannot apply returns an explicit error; there is no silent fallback to another method. The sampler validates shape and order, not the truthfulness of the caller's eligibility or support provenance.

**Replay scope for sampling.** Same units, same `SamplingSpec` (including `shuffle`), same seed, same release, toolchain, and target ⇒ identical rows and counts. The sampler does not promise cross-release or cross-platform bitwise equality.

### 5.3 Deferred methods and separate work

**Block bootstrap is designed but deferred; it is not in the current API.** No block variant exists in `SampleMethod`. All three methods draw units, never raw rows, and never cross a `sequence` boundary. They are recorded here so the future design starts from agreed semantics; each requires its own design and approval.

- **Moving block bootstrap.** Draws fixed-length consecutive blocks with replacement from starts where a full block fits in the source sequence.
- **Circular block bootstrap.** Draws fixed-length consecutive blocks with replacement. A block may wrap end-to-start within the same source sequence.
- **Stationary block bootstrap.** Draws consecutive blocks with geometrically distributed lengths and a configured mean, with wrap confined to the source sequence. The configured mean fixes the geometric parameter (`p = 1 / configured mean`); a length cap changes the realized mean, so the future design must state both and must not claim exact stationarity for a capped or wrapped variant.

For all three, blocks are runs of consecutive sampled units and are appended in draw order; the order within each block is preserved, including wrap order. Consecutive eligible units may span an eligibility gap, and a block may contain a fragmented unit's holes, so a block's row extent can be non-contiguous; consumers must not treat it as contiguous. The future output must provide, at minimum, each block's output span, its source sequence, and the exact wrap offset (or an explicit no-seam marker); a boolean flag is insufficient. Eligibility-gap handling, block boundaries, and the exact wrap position remain for that design.

**Stratification is deferred.** It is a separate future operation (`sampling/stratified.rs` when activated), not wired into the sampler or the splitter, and requires its own design gate.

**Evaluator-side resampling is separate.** Applying block bootstrap to already-scored out-of-sample results to estimate uncertainty is an evaluation operation, not training-fold sampling.

**Other work is out of this slice.** Purging, embargo, and blocked K-fold belong to their own designs.

### 5.4 Constraints on consumers

- Shuffle destroys source order. Stateful learners (for example RLS, GRU, or ordered SGD trajectories) must not consume shuffled order without an approved learning-rule change.
- Block joins and wrap seams are artificial presentation boundaries; sequence models must not read them as temporal continuity.
- Sampling is a selection heuristic. It makes no statistical, non-IID, or model-quality guarantee.

## 6. Worked example: GRU parameter initialization

This facility began as the GRU parameter initializer for #20. The GRU path remains one verified consumer and preserves the behavior that motivated the original document. It is no longer the definition of the guide's scope.

`GruParameters::init(input_dim, hidden_dim, &mut RandomGenerator)` produces an **untrained starting point**:

- input matrices `W_z, W_r, W_h` (`hidden_dim × input_dim`) use Glorot/Xavier uniform `U(-a, a)` with `a = sqrt(6 / (input_dim + hidden_dim))`;
- recurrent matrices `U_z, U_r, U_h` (`hidden_dim × hidden_dim`) use `U(-a, a)` with `a = sqrt(6 / (2·hidden_dim)) = sqrt(3 / hidden_dim)`;
- biases `b_z, b_r, b_h` are `0.0` and consume no draws.

For a uniform `[-a, a]` draw the variance is `a²/3`: `2/(input_dim + hidden_dim)` for input matrices and `1/hidden_dim` for recurrent matrices. Recurrent Glorot variance `1/hidden_dim` is three times PyTorch's documented `U(-1/√H, 1/√H)` recurrent variance `1/(3H)`; these are different choices, not equivalent formulations. Glorot's variance rationale does not by itself prove recurrent-gradient stability, long-sequence behavior, or fitness for a dataset.

Verified behavior:

- `initializer_is_reproducible_for_a_fixed_seed` — same seed, dimensions, and call order give identical parameters.
- `initializer_matches_frozen_parameter_vector` — a frozen seed (`0x20_2026`) reproduces exact stored bits for the six weight matrices.
- `initializer_produces_shaped_finite_parameters_and_zero_biases` — shapes, finite values, zero biases, successful construction.
- `initializer_scale_matches_glorot_variance_and_shrinks_with_dimension` — bound checks, scale shrinkage, and empirical mean/variance sanity against the declared uniform variance.
- `tests/random.rs` — an eight-value `draw_f32` KAT guards the stream for seed `0x5eed_5eed` and range `[-1, 1)`; the later `draw_u64_below` addition did not change it.

`GruParameters::deterministic` is a separate fixed-fixture path used by tests and benchmarks. It fills every field from a fixed LCG, including biases, and must remain bitwise stable. It is not a public randomness path, and it is not evidence about the seeded generator's quality.

## 7. Failure and verification

- Entropy failure: `RandomError::EntropyUnavailable { source }`, with `source()` retained. No panic, no fixed-seed fallback.
- Invalid range: `RandomError::InvalidRange` from `uniform_range`; no partially constructed range is returned.
- Sampling preflight: typed `SamplingError` variants cover configuration, units, sequences, outcome shape/order, applicability, draw space, span, element/byte capacity, and work. `MissingOutcomeSupport` is the explicit error for sequential bootstrap without required metadata. The error enum is `#[non_exhaustive]`, implements `Display` and `std::error::Error`, and has no underlying `source()`.
- Allocation failure after a passing capacity check aborts; it is not a typed error.
- The controlled RNG tests in `src/foundation/numerical/random.rs` never call the real OS entropy path; they exercise a seam with controlled success and failure sources.

Verification evidence is bound to the revision that produced it. Passing seeds, KATs, or fixtures do not prove statistical quality, and a fixture check does not renew evidence for the public generator.

## 8. Where this guide applies

| Workload | Randomness choice | Seed owner | Replay identity | Status |
|---|---|---|---|---|
| Parameter initialization (GRU) | `from_seed` or `from_entropy`; `draw_f32` | Caller | Seed + dimensions + call order | On `main`; covered by GRU and RNG tests |
| Training-fold sampling | `from_seed`; `draw_u64_below` | Caller via the seed passed to `SamplingSpec::new(...)` | Units + spec + seed | On open PR #104; **not on `main`** |
| Test and benchmark fixtures | Deterministic fixture or fixed seed | Fixture owner | Fixture identity and version | On `main`; bounded by verified toolchains and targets |

Delivery status: the GRU initializer and deterministic fixture paths are on `main`. The sampler, `draw_u64_below`, and the sampling API example are on open PR #104 and are not yet on `main`; descriptions of that API are contingent on the PR merging.

## 9. Randomness policy rules

These rules apply to new and changed workflows.

1. **P1 — Foundation supplies draws; consumers supply policy.** The foundation owns the generator, uniform-draw semantics, and entropy failure. Consumers own eligibility, selection, ordering, and what a seed means.
2. **P2 — Every consumer documents three facts.** Its seed source, its generator lifetime, and its draw order. A consumer without those facts has an incomplete randomness contract.
3. **P3 — Replay is scoped.** Claim replay only for a fixed seed, fixed inputs, fixed draw order, pinned dependency, and the same release, toolchain, and target.
4. **P4 — Entropy is fallible and explicit.** Never substitute a fixed seed, fixture, or default when OS entropy fails. Report the failure to the caller.
5. **P5 — Fixtures are not public-randomness evidence.** Keep deterministic fixtures, seeded public generation, and independent references separate in tests, benchmarks, and reports. Bound fixture claims to verified toolchains and targets.
6. **P6 — Keep temporal policy with its owner.** The splitter owns eligibility, cutoffs, and holdout; the sampler owns unit selection and presentation; neither silently absorbs the other's policy.
7. **P7 — KATs guard drift, not quality.** Known-answer tests prove stream and order conformance; they do not prove period, statistical suitability, or secrecy.
8. **P8 — Shuffle is opt-in and subordinate to the learning contract.** Enabling shuffle is a consumer decision that must be compatible with the learner's ordering requirements.
9. **P9 — Deferred randomness decisions require approval.** Per-member seed derivation, parallel draw partitioning, checkpoint/restore, block bootstrap, stratification, and evaluator-side resampling are not authorized by this guide.

## 10. Related documents and separate reconciliation

- [DEVELOPMENT.md](DEVELOPMENT.md) — ownership, naming and errors, testing, and the mandatory training and failure contracts.
- [CORRECTNESS.md](CORRECTNESS.md) — evidence boundaries and the sampling/randomness validation questions.
- [STREAMING_ML_DESIGN.md](STREAMING_ML_DESIGN.md) — streaming and ensemble context.
- `.kilo/plans/1795086208396-training-fold-sampling-technical-design.md` — DS-TD-0.10, the sampling design.
- `.kilo/plans/1795086208396-sampling-first-slice-implementation-plan.md` — IS-IP-1.1, the implementation plan.
- `src/preprocessing/sampling/README.md` — the sampling contract.
- `src/foundation/numerical/random.rs` — the shared draw foundation.

The following wording conflicts with this guide and is on the separate reconciliation list; this edit does not change it:

- `docs/CORRECTNESS.md:155` states that RANDOMNESS.md covers parameter-value generation only and excludes sampling, resampling, shuffling, and data reordering.
- `docs/STREAMING_ML_DESIGN.md:55` describes RANDOMNESS.md as authoritative only within a narrower declared scope.
- `docs/STREAMING_ML_DESIGN.md:163` says the facility generates parameter values only and excludes sampling, resampling, and ordering.
- `docs/STREAMING_ML_DESIGN.md:1021` records the parameter-generation scope and a separate ensemble-sampling decision.
- `src/foundation/numerical/random.rs:1–6` describes the module as parameter-value draws only.
- `src/foundation/numerical/mod.rs:7` describes `random` as parameter-value draws only.
- `src/models/recurrent/README.md:528` says the generator performs parameter-value draws only.
