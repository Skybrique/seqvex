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
- Tree C: identity and integer-draw modes now. Sequential bootstrap requires
  the #103 strict-f32 sampler and is not integrated by the pre-#103 slice.
- OLS: identity and explicit without-replacement only.
- Resampling is opt-in; shuffle is off by default.

The caller owns eligibility, cutoff, outcome provenance, preprocessing, and
held-out membership. `support_cutoff: Option<i64>` validates declared support
ends when supplied; an end equal to the cutoff passes.

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

The pre-#103 slice delivers S0–S5 and the independent S7/S8/S9a/S10a work.
Tree C sequential integration, full E0-27 evidence, and full #106 acceptance
wait for #103 delivery and independent validation.
