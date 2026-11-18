# Stage 1 Row Splitters

`seqvex::preprocessing::split` selects ordered, half-open row positions for
expanding and fixed-width rolling walk-forward evaluation. The caller owns the
ordered rows, features, targets, metadata, model lifecycle, preprocessing,
scoring, and any persisted results.

## Supported API

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

## Geometry

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

## Errors

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

## Caller checks

The splitter does not inspect feature or target values. In particular, a NaN
feature does not change membership; the consuming model may reject it. Check
feature/target alignment before slicing, using
`seqvex::foundation::error::DimensionMismatch` with feature rows as `expected`
and target rows as `actual`.

`row_gap` is a row-position gap, not a time gap. Stage 1 does not establish
point-in-time feature availability, training-label maturity, purging, or
embargo. Fit preprocessing on the training range only and freeze it for the test
fold.

## Tree evaluation handoff

A Decision Tree evaluation must independently verify:

- features were available at decision time;
- training labels were available at the fit cutoff;
- preprocessing was fitted on training rows only;
- repeated rows across overlapping test folds are identified by `(fold, row)`,
  with declared weighting and no claim of independence.

Forward-horizon labels or irregular grouped data require an independently
audited caller-side protocol or a suitable alternate dataset; otherwise the
scope decision returns to the maintainer. Stage 2 handles timestamps and
availability; Stage 3 handles purging and embargo.

## Verification

The external integration test uses an independent stride-congruence oracle that
does not call production scheduling or endpoint helpers. It covers both methods,
both tail policies, the pinned `6..10` terminal fixture, error precedence,
overflow, huge steps and widths, the public `FusedIterator` bound, and repeated
post-exhaustion `next()` calls. Run:

```bash
cargo test --locked --test data_splitter --all-features
cargo test --locked --release --test data_splitter --all-features
cargo run --locked --example data_splitter
```
