# Regression bagging

`models::ensemble::bagging` provides one homogeneous Decision Tree bag and one
separate homogeneous OLS bag. Both use the same bagger-local sequential fit
orchestration and checked mean.

## Profiles

| Profile | Fit | Serving |
|---|---|---|
| A | f64 | f64 |
| B | f64, checked conversion | f32 |
| C | strict f32 | f32 |

Profile B constructors require an explicit underflow policy. `Reject` is the
recommended deployment policy; `AllowLoss` is explicit.

## Sampling

- Tree A/B: identity, without-replacement, with-replacement, and sequential
  bootstrap through the shared f64 sampler.
- Tree C: identity and the integer methods through the shared f64 sampler;
  `Default` and explicit sequential bootstrap route to
  `sample_training_sequential_f32` with `SequentialLimits::default()`.
- OLS: identity and explicit without-replacement only.
- Resampling is opt-in; shuffle is off by default.

Sequential strict-`f32` sampling passes both layers. The bagger fails fast on
its aggregate `BaggingLimits` (units, spans, scratch, work), then the sampler
applies its own per-call limits and hard capacities for every member. Raising a
bag limit cannot raise a sampler default, and one admitted call does not relax
the aggregate cap. The bagger runs its preflight before fitting any member and
returns its own `BaggingError` variants on rejection. Each member's sampler
preflight runs before that member is fitted. Sampler errors are wrapped in
`BaggingError::Sampling(..)`. A failed candidate is never published.
Expanded-row arithmetic is checked: equality at
`max_expanded_rows_per_member` passes, a representable over-limit result is
`ExpandedRowsExceeded`, and a checked-arithmetic overflow is
`ExpandedRowsOverflow`.

The caller owns eligibility, cutoff, outcome provenance, preprocessing, and
held-out membership. `support_cutoff: Option<i64>` validates declared support
ends when supplied; an end equal to the cutoff passes. A source range whose end
exceeds the input reports the first invalid row, `max(start, len)`, in
`SourceRowOutOfRange`.

## Publication and prediction

A candidate is returned only after every member fits and every required
conversion succeeds. A failure reports the member and source error and leaves
the caller's active bag untouched. `retained_bytes()` returns the exact
validated logical serving-payload total established during construction.

Prediction is all-or-nothing. On success, outputs are returned in input order.
On failure, the first failing row stops evaluation, no predictions are
published, and the typed error carries the zero-based row index, member index
where applicable, and `evaluated_rows`. A combined-mean failure reports the
aggregation stage and no member index.

## Mean

The reviewed mean uses the two-component residual sequence, outward-rounded
residual intervals, single fused `div_up_pos`, exact-zero witness precedence,
and signed-zero rules. Accuracy targets are `2^-40` relative for f64, `2^-20`
for f32, and `2η` absolute for subnormal/zero results. Rust 1.85-compatible
adjacent-float helpers are local; the standard `next_up`/`next_down` methods are
not required.

## P9 seeds

`member_seed = BAGGING_MEMBER_DOMAIN ^ base_seed ^ ((fold_id << 32) | member_id)`
with `BAGGING_MEMBER_DOMAIN = 0x1060_0000_0000_0001`. Fold IDs use
`TimeFold::scheduled_id`; row folds use the full enumeration index. Replay is
same-executable; cross-release, cross-toolchain, and cross-precision equality
are not promised.

## Member order and sampler use

Members are fitted in ascending member index order. Each member starts a fresh
shared-sampler generator from its derived P9 seed, selects units before any
rows are expanded, and uses shuffle disabled. Repeats selected by the sampler
are passed to the fitter intact.

## Example

`cargo run --locked --example bagging_regression` fits a Profile B candidate on
an ordered training split with explicit `TreeUnderflowPolicy::Reject`, publishes
it only after checked conversion, and predicts the fixed holdout:

```text
members=4 retained_bytes=416 holdout_rows=6 test_mae=0.000000
```

## Current checkpoint

The pre-#103 slice delivers S0–S5 and the independent S7/S8/S9a/S10a work. Tree C
sequential bootstrap now routes to the landed strict-`f32` sampler; full E0-27
release evidence and full #106 acceptance remain open under S10b.
