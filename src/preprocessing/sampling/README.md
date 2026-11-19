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
`AllocationTooLarge`, and `SamplingWorkOverflow`.

## Replay

Same units, spec, seed, release, and toolchain produce identical rows and
counts. Cross-release and cross-toolchain bitwise equality is not promised:
`rand` streams may change across releases, and sequential weights use `f64`.
Distinct folds or ensemble members need distinct caller-chosen seeds; the
sampler never derives seeds.

## Not in this slice

Moving block, circular block, and stationary bootstrap are deferred. The later
stationary design must reconcile the geometric length law, the configured mean,
and wrap/cap behavior before claiming stationarity. Stratification remains a
separate operation with its own design gate. Block-bootstrapping already-scored
results is a separate evaluation operation, not this module.

`examples/data_sampler.rs` shows the splitter → sampler → OLS path, the
cutoff-24 excluded-support mutation control, a fragmented group, and Stage 1
composition.
