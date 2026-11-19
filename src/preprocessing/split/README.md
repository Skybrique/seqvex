# Data Splitters

`seqvex::preprocessing::split` provides Stage 1 row-position splits and Stage 2
elapsed-time splits. Both return lazy, half-open row ranges. The caller owns the
ordered data, features, targets, metadata, model lifecycle, preprocessing,
scoring, and persisted results.

## Stage 1: row positions

```rust
use seqvex::preprocessing::split::{
    expanding, rolling, RowWindowSpec, TailPolicy,
};

let spec = RowWindowSpec::new(
    0,                  // first_train
    3,                  // train_rows
    4,                  // test_rows
    1,                  // step_rows
    0,                  // row_gap
    TailPolicy::Drop,
);

let folds = expanding(10, spec)?;
for fold in folds {
    let train = fold.train_range();
    let test = fold.test_range();
    // Slice aligned caller-owned views with these ranges.
}
```

`RowFolds` is lazy, fused, and constant-size. Each `RowFold` exposes
`train_range() -> Range<usize>` and `test_range() -> Range<usize>`.

### Row geometry

For row count `N`, anchor `A`, training width `W`, gap `G`, test width `H`, and
step `S`:

- `p = A + W + G + k*S`
- `q = p - G`
- expanding training: `A..q`
- rolling training: `(q-W)..q`
- full test: `p..(p+H)`
- short test: `p..N`

A full test ending exactly at `N` terminates the schedule. A short candidate is
emitted once by `KeepShort` or omitted by `Drop`. A step beyond the endpoint
creates no catch-up fold. `RowFolds` returns `None` permanently after the last
fold.

### Row errors

Constructor validation is ordered:

1. `ZeroTrainRows`
2. `ZeroTestRows`
3. `ZeroStepRows`
4. `AnchorOutOfBounds { first_train, row_count }`
5. `FirstTestStartOverflow`
6. `NoFolds`

`RowSplitError` is non-exhaustive, implements `Debug`, `Display`, and
`std::error::Error`, and has no underlying source. `RowWindowSpec::new` stores
configuration only; `expanding` and `rolling` validate against the row count.

### Stage 1 caller checks

The splitter does not inspect feature or target values. In particular, a NaN
feature does not change membership; the consuming model may reject it. Check
feature/target alignment before slicing, using
`seqvex::foundation::error::DimensionMismatch` with feature rows as `expected`
and target rows as `actual`.

`row_gap` is a row-position gap, not a time gap. Stage 1 does not establish
point-in-time feature availability, training-label maturity, purging, or
embargo. Fit preprocessing on the training range only and freeze it for the test
fold.

### Decision Tree evaluation handoff

A Decision Tree evaluation must independently verify:

- features were available at decision time;
- training labels were available at the fit cutoff;
- preprocessing was fitted on training rows only;
- repeated rows across overlapping test folds are identified by `(fold, row)`,
  with declared weighting and no claim of independence.

Forward-horizon labels or irregular grouped data require an independently
audited caller-side protocol or a suitable alternate dataset; otherwise the
scope decision returns to the maintainer. Stage 2 implements timestamps and
availability; Stage 3 purging and embargo remain deferred.

## Stage 2: elapsed time

Stage 2 selects rows on a single caller-normalized signed integer tick axis.
Durations, step, and gap are non-negative `u64` tick counts. Coverage, the gap,
training, and test intervals are half-open. The splitter does not fit
preprocessing, score labels, purge outcomes, or embargo.

### Supported API

```rust
use seqvex::preprocessing::split::{
    expanding_time, rolling_time, TimeCoverage, TimeFoldOutcome, TimeRow, TimeSplitError,
    TimeWindowSpec,
};

pub fn example() -> Result<(), TimeSplitError> {
    let rows = vec![
        TimeRow {
            decision_at: 1,
            features_available_at: 0,
            label_available_at: Some(3),
        },
        // One record per aligned caller row.
    ];
    let coverage = TimeCoverage::new(0, 30, 0); // start, end, anchor
    let spec = TimeWindowSpec::new(10, 5, 6, 2, 3); // train, test, step, gap, cap

    let folds = expanding_time(rows.len(), &rows, coverage, spec)?;
    for fold in folds {
        let _id = fold.scheduled_id();
        let _cutoff = fold.cutoff();
        let _test = fold.test_range();
        let _train = fold.train_ranges();
        match fold.outcome() {
            TimeFoldOutcome::Usable => {
                // Fit and score using the diagnostic ranges.
            }
            TimeFoldOutcome::Unusable(_) => {
                // Report the scheduled window without evaluating it.
            }
            _ => {
                // Future outcome variants remain non-evaluable here.
            }
        }
    }

    let _rolling = rolling_time(rows.len(), &rows, coverage, spec)?;
    Ok(())
}
```

`TimeFolds` is lazy, fused, and exposes `scheduled_fold_count()`. Each
`TimeFold` exposes the scheduled ID, cutoff, contiguous test candidate range,
ordered coalesced training ranges, disjoint exclusion counts, and outcome.
`scheduled_id` is the nominal index and is never renumbered. Membership and
counts remain available for unusable windows as diagnostics, but fitting or
scoring is permitted only when the outcome is `Usable`.

### Time geometry

For anchor `A`, training duration `W`, gap `G`, test duration `H`, step `S`, and
coverage `[C0, C1)`:

- `p = A + W + G + k*S` is the fit cutoff and test start.
- `q = p - G` is the exclusive training end.
- expanding training: `[A, q)`
- rolling training: `[q-W, q)`
- test: `[p, p+H)`

Only full windows with `p+H <= C1` are scheduled. The count is fixed before
enumeration: `1 + (C1 - (p0+H)) / S`. A test ending exactly at `C1` is emitted
once, then the schedule stops. There is no short time tail, and a window with no
rows is still scheduled and counted.

Training ends exclusively at `q`; the gap is `[q, p)`. With `G > 0`, a decision
at `q` is in the gap, not training. With `G == 0`, `q == p`, the gap is empty,
and the shared tick belongs to the test. A decision at `p+H` is outside that
test.

### Time eligibility

Training candidates are selected by decision tick and then filtered:

- `features_available_at < decision_at` (strict), and
- a known `label_available_at < cutoff` (strict).

A training row failing both predicates is counted as feature-unavailable. An
unknown training label is label-unavailable. A known label earlier than its
decision is a construction error; a label exactly at the decision is allowed
structurally.

Test candidates require `features_available_at < decision_at`. Any failing test
feature makes the whole scheduled window unusable and reports the first failing
row. Test labels do not affect prediction membership; the caller's scoring
cutoff decides when a test label is mature.

### Time outcomes and errors

The unusable precedence is:

1. `EmptyTest`
2. `UnavailableTestFeature { row, decision_at, features_available_at }`
3. `EmptyEligibleTraining`

`ExclusionCounts` is disjoint:
`train_candidates = train_feature_unavailable + train_label_unavailable +
train_eligible`, and `test_candidates = test_feature_unavailable +
test_eligible`.

Construction validation is ordered:

1. `ZeroTrainDuration`
2. `ZeroTestDuration`
3. `ZeroStepDuration`
4. `ZeroScheduleCap`
5. `CoverageOrder { start, end }`
6. `AnchorOutsideCoverage { anchor, start, end }`
7. `MetadataLengthMismatch { shape }`
8. `DecisionOutOfOrder { row, previous, actual }`
9. `DecisionOutsideCoverage { row, decision_at, start, end }`
10. `LabelBeforeDecision { row, decision_at, label_available_at }`
11. `FirstTestStartOverflow`
12. `FirstTestEndOverflow`
13. `NoFullWindows { first_test_start, test_end, coverage_end }`
14. `ScheduleCountOverflow`
15. `ScheduleCapExceeded { scheduled, cap }`

Metadata classes are resolved across the whole snapshot before the first
offending row is reported. `MetadataLengthMismatch` exposes the shared
`DimensionMismatch` through `source()`; other variants have no source.

### Resources and caller checks

The caller supplies a positive schedule cap. The cap bounds scheduled windows,
not metadata validation, row visits, or retained storage. Metadata validation is
one `O(N)` preflight across ordered-decision, coverage-containment, and
label-time checks; row visits are `O(N)` plus per-window scans; dense rolling
windows can revisit many rows. Training ranges can fragment to one range per
eligible row. The lazy iterator holds one fold at a time; callers that retain
all folds accumulate range storage.

Use one tick scale and record its unit, source, timezone treatment, and
resolution. Decisions must be nondecreasing; equal decisions remain in caller
order. Verify feature/target/metadata alignment before slicing. The splitter
does not prove that availability ticks are truthful, that features are the exact
version used, that labels are mature at the scoring cutoff, or that coverage is
complete.

## Stage 3

Blocked splits, purging, embargo, and CPCV remain deferred under their own
scope decisions. Stage 2 does not implement them.

## Verification

The Stage 1 test uses an independent stride-congruence oracle. The Stage 2 test
uses a row-wise oracle that does not call production scheduling, the count
formula, endpoint helpers, or boundary search. Run:

```bash
cargo test --locked --test data_splitter --all-features
cargo test --locked --release --test data_splitter --all-features
cargo test --locked --test data_splitter_time --all-features
cargo test --locked --release --test data_splitter_time --all-features
cargo run --locked --example data_splitter
cargo run --locked --example data_splitter_time
```
