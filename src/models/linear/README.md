# Linear models

> **Completeness:** the implemented OLS baseline is accepted: fitting, learned-result prediction, rank/conversion checks, diagnostics, artifact v1, checked serving and deliberate refit handover. Complete linear models must still ingest data, fit/learn, validate and predict using the learned result under the [mandatory training-to-inference contract](../../../docs/DEVELOPMENT.md#model-training-and-valid-inference--mandatory). Acceptance covers the declared baseline, not every production workload.

## Purpose

Holds the linear-model family: ordinary least-squares (OLS) regression,
recursive least squares (RLS), and SGD-trained linear regression (SGD). OLS was
relocated here from `models::classic` and RLS from `models::online`, both as
behavior-preserving moves; the classic and online import paths remain as
compatibility re-exports. SGD is implemented in place as a separate learning
contract; it does not replace OLS batch fitting or RLS.

## Responsibility

- [`ols`](ols/README.md) — the canonical OLS algorithm document: requirements,
  issue owners, implemented behavior and deferred work; the original Architect
  design and Planner implementation plan remain separate review records.
- [`rls`](rls/README.md) — the canonical recursive least squares document: the
  ordered recurrence, failure/atomicity, checked prediction, statistical
  applicability, bounded micro-batch, and measured evidence.
- [`sgd`](sgd/README.md) — the canonical SGD-trained linear-regression document:
  ordered single-observation learning with a fixed learning rate, prediction from
  learned weights, validation/atomicity, examples and measured evidence.

## SGD boundary

SGD learns by ordered single-observation updates (`w' = w + η·e·x`) from
caller-supplied features and targets. It is not a batch solver, does not shuffle
implicitly, and provides no automatic convergence or stability guarantee. The
model is immutable; learned weights are per-execution state. The caller supplies
the intercept feature and any preprocessing. See the
[canonical SGD README](sgd/README.md) for APIs, examples and evidence.

## Execution modes and profiles

OLS is a historical dataset fit/refit plus fixed single/stream/group prediction. RLS and SGD are ordered single-step learners; RLS also exposes a public bounded ordered-learning call, while SGD's bounded group is the documented caller protocol. Fixed grouped prediction for RLS/SGD is a caller collector over one selected identity. B reuses A's f64 learner/fit and requires checked f32 conversion; it is not a separate learner. The caller owns event order, pending labels, admission caps, serving replacement, retained outputs and progress identities. A processed count is not learning, serving or publication progress. No generic micro-batch executor, scheduler or version registry is implied.

## Legacy OLS baseline

The original supplied-parameter prediction component was relocated unchanged
from `src/models/classic/linear_regression.rs` to `ols/predict.rs`. The accepted
baseline now includes:

- The immutable supplied-parameter predictor `ŷ = w · x + b`, with reference
  `predict`, bounded `predict_batch`, and streaming via the foundation
  `StateModel` contract (state = most recent prediction).

- Dense unweighted fitting with selected training indices, multi-feature scalar
  targets and an optional intercept: [`ols/fit.rs`](ols/fit.rs). Inputs are `f32`,
  the QR solve is `f64`, and learned parameters/serving remain `f32`.
- Metadata/content identity and binary artifact v1:
  [`ols/artifact.rs`](ols/artifact.rs); deliberate refit handover uses a new
  executor and preserves the active model on failed candidates.
- Opt-in checked single/group/stream serving, preserving bare prediction behavior.
- Independent oracle, fitter, artifact, checked-serving and temporal-selection
  tests; the registered [`fit_salary`](ols/examples/fit_salary.rs) CSV example
  and [`ols_fit`](ols/benches/fit.rs) resource benchmark.

Explicitly deferred under [#23](https://github.com/Skybrique/seqvex/issues/23):
caller-side preparation/partition helpers (T10a), ordering/shuffle facilities
(T10b), and formal cross-precision numerical certification. OLS does not shuffle
implicitly; the prohibition on shuffling ordered workflows is retained. See the
canonical [OLS README](ols/README.md) for APIs, evidence and operating limits.

Out of scope until separately approved: ridge, lasso, logistic classification,
weighted/sparse/multi-output fitting, and any generic linear-model trait.

## Status (OLS)

Bare prediction behavior is preserved bitwise: reduction order, validation
order and existing errors are unchanged. Checked serving adds the public
`NonFiniteOutput` variant; downstream exhaustive error matches need a new arm.
The baseline follows the accepted fitting-design direction with the explicit
scope deferrals above. The Salary example demonstrates integration using an
interleaved every-fifth-row holdout; it is not chronological evaluation or a
production-readiness claim.
