# Training-fold sampling

`seqvex::preprocessing::sampling` draws training rows from caller-built
eligible units. The splitter decides whether a fold is usable and supplies the
eligible rows; this module never inspects the splitter, owns observations, fits
preprocessing, or scores labels.

```text
expanding_time / rolling_time            (unchanged)
        │  TimeFold::outcome() must be Usable
        │  TimeFold::train_ranges() -> &[Range<usize>]
        ▼
caller groups eligible rows into SampleUnit values
        ▼
sample_training(&units, spec) -> TrainingSample
        ▼
row_indices() feed the fit; held-out membership is unchanged
```

## Units and multiplicity

- A unit owns ordered, disjoint, non-empty `rows` ranges. Ranges may be
  non-adjacent, so eligible rows at one decision timestamp separated by an
  ineligible row stay one draw.
- Units are globally ascending and disjoint; `sequence` is non-decreasing.
- `unit_counts()[i]` is the draw count for unit `i`. Every draw emits all of the
  unit's rows, in declared range order.
- The sampler validates unit shape and order. It cannot verify that the units
  are the fold's eligible rows, that held-out rows are absent, or that holes
  mean ineligibility.

## Outcome supports

Sequential bootstrap needs a declared realized-outcome support per unit: an
ordered, disjoint set of half-open intervals over the caller's tick axis.

- Build supports **only from eligible training observations**.
- Use only facts available at the fold cutoff; every span ends at or before the
  cutoff. Never use test-row outcomes, post-cutoff values, or held-out labels.
- A unit's support is the disjoint union of its members' spans; overlapping
  spans are merged and holes are not filled.
- Supports are not label-availability ticks, and availability does not imply a
  support.

This rule is caller metadata only: it needs no purge/embargo type and no Stage 3
interval machinery, so it is verifiable independently of #100.

## Modes

| Mode | Behavior |
|---|---|
| `Disabled` | Every eligible unit once. `sample_size` is ignored. |
| `Default` | Sequential bootstrap when every unit has outcome support; otherwise `MissingOutcomeSupport`. |
| `Method(SequentialBootstrap)` | As-if-appended average uniqueness with replacement. |
| `Method(WithoutReplacement)` | `sample_size` distinct units; `sample_size <= unit count`. |
| `Method(WithReplacement)` | `sample_size` uniform draws; duplicates retained. |

Sequential weights: `u_i = (1/|T_i|) Σ_{t∈T_i} 1/(1+c(t))` over prior draws;
`p_i = u_i/Σu`; draw, update `c`, repeat. Weighted selection uses unnormalized
weights, `target = u·Σw`, the first strict `>` win, and a last-positive fallback.

Resampling is opt-in. Shuffle is independent and defaults to `false`.

## Strict-`f32` sequential entry point

`sample_training(&units, spec)` keeps the legacy identity, integer and `f64`
sequential paths. The strict path is a separate additive entry point:

```rust
pub fn sample_training_sequential_f32(
    units: &[SampleUnit],
    spec: SamplingSpec,
    limits: &SequentialLimits,
) -> Result<TrainingSample, SamplingError>
```

It accepts `Default` and `Method(SequentialBootstrap)` identically and rejects
`Disabled`, `WithoutReplacement`, `WithReplacement` and any future unadmitted
mode with `UnsupportedMode`. It creates one fresh generator from `spec.seed`
only after preflight, and never calls the legacy `f64` sampler.

`SequentialLimits` is per call, not an aggregate bag limit:

| Field | Default | Inclusive ceiling |
|---|---:|---:|
| `max_units` | 65,536 | 1,048,576 |
| `max_support_spans` | 1,000,000 | 16,777,216 |
| `max_endpoint_scratch_bytes` | 16,777,216 (16 MiB) | 268,435,456 (256 MiB) |
| `max_expanded_rows` | 1,000,000 | 16,777,216 |
| `max_work_cells` | 1,000,000,000 | 1,000,000,000,000 |

Each field is admitted in `1..=ceiling`; zero and above-ceiling values return
`InvalidLimits` with the exact field spelling and widened values.

The strict preflight order is: limits, admitted mode, sample size, unit
presence, row/support shape and order, missing support, inherited `2^53`
outcome span, strict count `2^24`, strict per-unit support union `2^24`, then
`max_units`, `max_support_spans`, endpoint scratch, expanded rows, work, and the
hard element/byte/work capacities. The raw endpoint buffer is charged as
`P = 2*S` (not the deduplicated count) and retains that capacity. The scratch
charge is `64*S + 64*U + 256` bytes, expanded rows are
`n * max_i sum(range length)`, and work is `n*U*P` cells. Only covered segments
are subject to the strict `2^24` union bound; unused gaps and the total across
units are not. Every caller-reachable typed rejection consumes zero selection
and shuffle draws.

Weights use only `f32`: for each unit, sum
`length_f32 / (1.0_f32 + concurrency_f32)` over spans and ascending covered
segments, then divide by the unit's declared support-union length converted to
`f32`. Tickets are exact binary32-derived integers with `Q = 2^32`, `M` the
per-draw maximum stored weight, round-half-up, and a positive floor of one.
One `draw_u64_below` request is made per selection; the first cumulative ticket
end strictly greater than the draw wins. A zero stored weight or a failed
checked ticket/total is a defensive `F32WeightUnderflow` or
`F32WeightQuantumOverflow` before the failing draw; admitted input cannot reach
either after a draw. Presentation and shuffle follow the existing whole-unit
rules.

Strict replay is the same units, spec, seed, release, locked dependencies,
toolchain and target; cross-release, cross-toolchain and cross-precision
bitwise equality is not promised. Tree Profile C sequential integration and
bagger adapter/preflight work belong to #106, not this module.

## Presentation order

Every mode uses one order:

1. draw units;
2. order the drawn occurrences into source order within each sequence and, when
   shuffle is enabled, permute **whole occurrences** within each sequence;
3. expand occurrences into row indices and accumulate `unit_counts`.

With shuffle off, rows expand directly from `unit_counts` in source order and no
occurrence buffer is allocated. With shuffle on, Fisher–Yates runs inside each
sequence (`j = m_s−1 … 1`, one draw per step); a fragmented group moves as one
occurrence and keeps its internal row order. Sequences are never interleaved.

## Errors and bounds

Preflight is staged and every failure happens before the generator exists:

1. input validation, applicability, draw space, outcome-span `2^53` limit, and
   input-derived element/byte/work bounds;
2. for sequential bootstrap only: allocate the checked raw endpoint buffer
   (`2S` entries), sort and deduplicate it in place, run the defensive
   segment-count checks (`E−1`), then the work check `n × K × E`;
3. allocate the segment array, then every other buffer, then draw.

- The raw endpoint buffer keeps its `2S` capacity for the whole call.
- Because `E ≤ 2S` and `size_of::<usize>() ≤ size_of::<i64>()` on supported
  targets, the segment-count checks are bounded by the raw-buffer check; they
  remain defensive arithmetic guards.
- The work check precedes the segment-count allocation and every output-sized
  allocation, so a work overflow returns before any output buffer is allocated.
- Byte bounds use `size_of::<T>()` and `isize::MAX`; element counts use
  `usize::MAX`. Checked `u128` overflow maps to a typed error.
- Allocation failure after a passing check aborts; it is not a typed error.
- Every typed preflight failure consumes zero draws.

Errors are `#[non_exhaustive]`: `ZeroSampleSize`, `NoEligibleUnits`,
`EmptyUnitRows`, `UnitRowsOutOfOrder`, `SequenceOutOfOrder`,
`InvalidOutcomeInterval`, `OutcomeOutOfOrder`, `MissingOutcomeSupport`,
`SampleSizeExceedsUnits`, `OutcomeSpanOverflow`, `SampleLengthOverflow`,
`AllocationTooLarge`, and `SamplingWorkOverflow`. The strict entry point adds
`InvalidLimits`, `UnsupportedMode`, `F32CountUnsupported`,
`F32SpanUnsupported`, `UnitLimitExceeded`, `SupportSpanLimitExceeded`,
`EndpointScratchExceeded`, `ExpandedRowsExceeded`, `WorkLimitExceeded`,
`ArithmeticOverflow`, `F32WeightUnderflow`, and `F32WeightQuantumOverflow`.

## Replay

Same units, spec, seed, release, locked dependencies, toolchain and target
produce identical rows and counts. Cross-release, cross-toolchain and
cross-precision bitwise equality is not promised: `rand` streams may change
across releases, legacy sequential weights use `f64`, and the strict path uses
`f32`. Distinct folds or ensemble members need distinct caller-chosen seeds;
the sampler never derives seeds.

## Not in this slice

Moving block, circular block, and stationary bootstrap are deferred. The later
stationary design must reconcile the geometric length law, the configured mean,
and wrap/cap behavior before claiming stationarity. Stratification remains a
separate operation with its own design gate. Block-bootstrapping already-scored
results is a separate evaluation operation, not this module.

`examples/data_sampler.rs` shows the splitter → sampler → OLS path, the
cutoff-24 excluded-support mutation control, strict-`f32` replay, a fragmented
group, and Stage 1 composition.
