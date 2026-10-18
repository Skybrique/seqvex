# SGD-trained linear regression

> **Status and scope.** This is the canonical SGD algorithm document for
> [#50](https://github.com/Skybrique/seqvex/issues/50). SGD-trained linear
> regression is the ordered, incremental member of the linear family: it learns
> from one supervised observation at a time with a fixed learning rate and
> predicts from the learned weights. Validation evidence was recorded on
> 2026-10-17; documentation and integration gates remain pending. This is not a
> production-readiness claim and does not imply a published implementation
> revision.

## What it does and when it is useful

SGD-trained linear regression learns a dense linear predictor `ŷ = wᵀx` from
ordered `(x, y)` observations. Each update uses the squared-error gradient from
one observation:

```text
ŷ_t = w_{t-1}ᵀ x_t
e_t = y_t − ŷ_t
w_t = w_{t-1} + η e_t x_t
```

The reference dataset objective is

```text
J(w) = (1 / (2N)) Σ_i (y_i − wᵀx_i)²
```

with per-observation loss `½(y − wᵀx)²`. In exact arithmetic the minimizer of
`J` is the unregularized OLS solution, but a finite SGD run at a fixed learning
rate need not reach that minimizer: progress depends on the learning rate, the
number of ordered passes, conditioning and order.

With a fixed step, a full ordered pass is an affine map. When the pass iteration
converges, it approaches a **cyclic fixed point** that need not equal the
least-squares optimum. More passes do not generally remove that bias; its size
depends on the learning rate and the data. No convergence, stability or quality
guarantee is offered.

It is useful when observations arrive in order and a small, explicit,
one-observation-at-a-time learning step is preferable to a batch refit. It is
**not** a batch solver, does not replace OLS, and is not RLS: there is no
covariance/precision matrix and no explicit forgetting-factor objective.

In exact arithmetic, for the constant-only model (`x_t ≡ 1`) with `0 < η < 1`,
the scalar update is an exponential moving average with decay `1−η`. This does
**not** generalize to arbitrary features or to every accepted learning rate:
`η = 1` takes the current target, `1 < η < 2` alternates while converging for a
constant target, `η = 2` is undamped for a constant target, and `η > 2` is
generally unstable. This is not the RLS weighted-least-squares recurrence.

### Choosing among the linear members

| Member | Learning mechanism | Typical use |
|---|---|---|
| [`ols`](../ols/README.md) | Batch column-pivoted QR fit; frozen coefficients | Fit once from a chosen training set |
| [`rls`](../rls/README.md) | Recursive covariance update with explicit forgetting | Ordered adaptation with a covariance state |
| `sgd` (this document) | Ordered first-order updates at a fixed learning rate | Simple incremental learning, no covariance state |

## Public API

Canonical namespace: `seqvex::models::linear::sgd`.

```rust
use seqvex::foundation::numerical::Vector;
use seqvex::foundation::observation::Observation;
use seqvex::foundation::state::StateModel;
use seqvex::models::linear::sgd::{Sgd, SgdError, SgdSample, SgdState};
```

| Item | Signature / shape |
|---|---|
| `Sgd::new` | `new(dimension: usize, learning_rate: f32) -> Result<Sgd, SgdError>` |
| `Sgd::dimension` | `-> usize` |
| `Sgd::learning_rate` | `-> f32` |
| `Sgd::initial_state` | `-> SgdState` (`w = 0`) |
| `Sgd::predict` | `predict(&self, state: &SgdState, features: &Vector) -> Result<f32, SgdError>` |
| `StateModel::update` | `update(&self, state: &SgdState, observation: &Observation<SgdSample>) -> Result<SgdState, SgdError>` |
| `SgdState` | `{ pub w: Vector }` — per-execution learned weights |
| `SgdSample` | `{ pub features: Vector, pub target: f32 }` |
| `SgdError` | `ZeroDimension`, `DimensionMismatch { expected, actual }`, `InvalidLearningRate`, `NonFiniteInput`, `NonFiniteCandidate`, `NonFiniteOutput` |

There is **no** `fit` method on `Sgd`. Learning is `update`; historical
learning is an ordered sequence of `update` calls owned by the caller.

## Minimal data → learning → prediction example

The caller supplies the feature vector, including any intercept feature. This
example appends a constant `1.0` as the third feature and learns from three
finite rows:

```rust
use seqvex::foundation::numerical::Vector;
use seqvex::foundation::observation::Observation;
use seqvex::foundation::state::StateModel;
use seqvex::models::linear::sgd::{Sgd, SgdSample};

// Two features plus a caller-supplied intercept (appended 1.0).
let model = Sgd::new(3, 0.05).expect("valid configuration");
let mut state = model.initial_state();

let rows = [
    ([2.0f32, -1.0, 1.0], 8.0f32),
    ([0.0, 1.0, 1.0], 2.0),
    ([-1.0, 0.5, 1.0], 0.5),
];
for _ in 0..50 {
    for (features, target) in rows {
        state = model
            .update(
                &state,
                &Observation::new(SgdSample {
                    features: Vector::from_slice(&features),
                    target,
                }),
            )
            .expect("finite update");
    }
}

let prediction = model
    .predict(&state, &Vector::from_slice(&[1.0, 1.0, 1.0]))
    .expect("finite prediction");
println!("prediction = {prediction}");
```

Observed output from compiling and running this literal snippet against the
reviewed crate on default Rust 1.97.1 and MSRV Rust 1.85.1:

```text
prediction = 4.25284
```

This is an observed output for this data, learning rate and pass count; it is
not an accuracy, convergence or stability claim. `Sgd` performs no implicit
shuffle; repeated passes replay the caller's order.

## Intercept and preprocessing

- The model does **not** add an intercept and does **not** transform features.
  The caller supplies the feature vector; include a constant `1.0` feature when
  an intercept is wanted.
- No internal standardization, scaling or imputation is applied. Feature scale
  affects the fixed learning rate's behavior, so caller-side preprocessing is
  part of the workflow.
- Targets stay in their original units.
- `sgd_regression` demonstrates prefix-only population standardization; that is
  an example policy, not a property of `Sgd`.

## Ownership, continuation and reset

- `Sgd` holds only immutable configuration (`dimension`, `learning_rate`) and
  implements `StateModel` with `&self`. One model is shareable.
- The learned `w` lives in `SgdState`, owned by the execution session; there is
  no persistent model-owned workspace or scratch (each update creates transient
  delta/candidate vectors).
- Continue by passing the committed `SgdState` to the next `update`. Reset with
  `model.initial_state()`.
- `StreamingExecutor` (`use seqvex::execution::streaming::StreamingExecutor;`)
  owns one state per execution and borrows the model immutably:
  `StreamingExecutor::new(&model, model.initial_state())`, `process_one`, and
  `reset(model.initial_state())`.
- The same ordered rows, learning rate and initial state produce a bitwise
  reproducible trajectory on a platform/toolchain; cross-platform/hardware
  bitwise identity is not promised. Different orders generally produce different
  weights.

## Validation, overflow and atomicity

Construction rejects `dimension == 0` with `ZeroDimension` and a non-finite or
non-positive learning rate with `InvalidLearningRate`. Every finite positive
learning rate is accepted; stability for arbitrary data is **not** promised.

`predict` validates, in order: feature dimension, state `w` dimension, feature
finiteness (`NonFiniteInput`), then output finiteness (`NonFiniteOutput`).
`update` validates, in order: feature dimension, `w` dimension, feature
finiteness, target finiteness, then candidate finiteness
(`NonFiniteCandidate`).

The candidate `w'` is a local value returned only on success. A failed update
leaves the committed state bitwise unchanged; no partial candidate is exposed.
Finite inputs can still produce non-finite intermediates through `f32` overflow;
those are rejected rather than substituted.

### Caller-side historical replay

The following is a complete function, not a fragment:

```rust
use seqvex::foundation::observation::Observation;
use seqvex::foundation::state::StateModel;
use seqvex::models::linear::sgd::{Sgd, SgdError, SgdSample, SgdState};

fn replay(
    model: &Sgd,
    observations: &[Observation<SgdSample>],
) -> Result<SgdState, (usize, SgdState, SgdError)> {
    let mut state = model.initial_state();
    for (index, observation) in observations.iter().enumerate() {
        match model.update(&state, observation) {
            Ok(next) => state = next,
            Err(error) => return Err((index, state, error)),
        }
    }
    Ok(state)
}
```

Failure stops the replay with the zero-based index, the last committed state and
the error. A full restart from `initial_state()` over the same order reproduces
the run deterministically. Do not resume by re-applying already-committed rows
on top of the committed state: ordered updates are non-idempotent and that
duplicates them. Retry is an optional caller action, and unchanged invalid data
does not become recoverable by retrying. Do not use
`StreamingExecutor::process_stream` for learning; it continues past failures by
design.

## Examples

### `sgd_learn` — deterministic synthetic learning and prediction

A bespoke deterministic generator (seeded LCG; no external files or network)
produces two feature streams plus an appended intercept, a declared regime change
at row 500, and a separate seeded noise stream. It learns the first 400 rows with
20 ordered passes at `η = 0.05`, then scores the remaining 600 rows.

The example reports, separately:

- **frozen** evaluation — weights frozen after the prefix; predictions only;
- **adaptive** (prequential) evaluation — predict before each update;
- overall and pre-/post-regime-change MAE/RMSE;
- expanding-target-mean and previous-target baselines on the identical rows.

Run:

```sh
cargo run --locked --example sgd_learn
cargo +1.85.1 run --locked --example sgd_learn
```

The seeded LCG generates the synthetic data. **SGD itself needs no RNG** and does
not shuffle. The recorded run is reported in the Evidence section; it does not
promise improvement on other data.

### `sgd_regression` — real CSV learning and prediction

The example reads a caller-supplied CSV offline; dataset acquisition is a
separate manual step. It parses by header name (order-independent) and requires
exactly these seven columns, each once:

```text
ordinal_date, five_thirty_eight, gallup, ipsos, morning_consult, rasmussen, you_gov
```

Model inputs are the five poll features plus an appended `1.0`:
`[gallup, ipsos, morning_consult, rasmussen, you_gov, 1.0]` (dimension 6). The
target is `five_thirty_eight`; `ordinal_date` is metadata used for chronology and
labels, never a model feature.

Fixed protocol:

- first 400 rows are the learning prefix; the remaining rows are scored;
- population mean/std are fitted on the **first 400 rows only** and frozen;
- `η = 0.01`, 50 ordered prefix passes;
- `M_prefix = max_i ‖x_i‖²` is measured from the **actual transformed `f32`
  prefix vectors** (including the intercept), and the example requires
  `η·M_prefix < 1`; a failure is reported, never repaired by retuning;
- scoring is predict-before-update; frozen and adaptive evaluations are reported
  separately with the expanding-target-mean, previous-target and raw
  five-feature-mean baselines on identical rows.

**Example-only damping guard.** Before each adaptive update the example checks
`η·‖x‖² < 1` on that row's actual transformed `f32` vector. On failure it stops
with the row index and `ordinal_date`, preserves the last committed state, and
reports incomplete execution; it does not skip the row, retune `η`, or present
partial metrics as completed acceptance. This guard is an **example policy
only**: `Sgd` itself accepts every finite positive learning rate and does not
promise stability for arbitrary data.

Dataset preparation: download River's
[pinned compressed CSV](https://raw.githubusercontent.com/online-ml/river/6af83abba269dbdaf38e9c0eaf61d82c451ad698/river/datasets/trump_approval.csv.gz)
and decompress it as `trump_approval.csv`. The evaluated snapshot has 1001 data
rows and SHA-256
`aef188d04b606270e9590013282e2ede3f5f92edae1106d85d3defa483189cb0`; the example
prints the computed checksum and does not enforce it. The dataset is not bundled.

Run:

```sh
cargo run --locked --example sgd_regression --all-features -- \
  src/models/linear/rls/examples/data/trump_approval.csv
cargo +1.85.1 run --locked --example sgd_regression --all-features -- \
  src/models/linear/rls/examples/data/trump_approval.csv
```

The example takes the path as an argument and has no runtime dependency on the
RLS directory; the command simply supplies the supplied dataset's path.

### Output interpretation and error guidance

`sgd_learn` prints the settings, feature order, a metrics table for frozen,
adaptive and both baselines, and a small sample of adaptive predictions.
`sgd_regression` prints the dataset path/checksum, row and prefix/scored counts,
`M_prefix` and `η·M_prefix`, prefix-boundary weights, the metrics table for
frozen, adaptive and all baselines, and sample predictions.

Do not compare frozen and adaptive metrics as if they were the same protocol:
frozen measures a fixed learned model; adaptive tracks by updating after each
prediction. Both use the same scoring rows and baselines.

Parse, preprocessing, prediction and update failures stop with row/column or
row/date/call-kind context and a non-zero exit. There is no silent skipping,
default output, retry, reset or continued execution. Common causes are missing or
extra headers, malformed or non-finite numbers, non-increasing/duplicate
`ordinal_date`, a zero/non-finite feature scale, or too few rows for the
400-row prefix.

## Evidence

Validation evidence was recorded on 2026-10-17. Documentation and integration
gates remain pending; this is not a production-readiness claim.

- `sgd` test target: **16 passed** on default Rust 1.97.1 and on MSRV Rust 1.85.1.
  Coverage includes hand-derived and explicit-`f32` arithmetic controls, a
  rounding-sensitive reduction-order control, a multiplication-grouping control,
  an exact-rational trajectory, an analytic-gradient check, validation
  precedence, overflow, atomicity, isolation, continuation/reset and
  order/reproducibility. The two rounding-sensitive controls were added in a
  test-only follow-up; example sources are unchanged.
- Example tests: `sgd_learn` **2 passed**; `sgd_regression` **13 passed**.
- Both examples were executed and independently reproduced before the test-only
  follow-up. The follow-up changed only tests.
- Check B (frozen `D=4`, `N=64`, advance-before-draw supplied `f32` fixture,
  `η=0.01`, 200 ordered passes, exact-rational oracle). Both sub-fixtures pass the
  unchanged criterion `J_sgd − J* ≤ 1e-6·J_initial + 1e-3·J*`:

| Sub-fixture | `J*` | Objective gap | Allowed gap |
|---|---|---|---|
| noisy | 1.53865282598672883e-3 | 7.38227672922959696e-7 | 7.03698321273655127e-6 |
| noiseless | 3.37995574951686784e-15 | 9.00228264632184831e-12 | 5.45352837212015964e-6 |

  The noisy fixture's `κ(X)` is 1.916249. The noiseless `J*` is a small nonzero
  `f32` floor, not an exact-zero claim.

### Executed real-data result

On the supplied 1001-row snapshot (400 learned, 601 scored; `η=0.01`, 50 ordered
passes; `M_prefix=27.476190586`, `η·M_prefix=0.274761900`):

| Predictor | MAE | RMSE |
|---|---|---|
| SGD frozen | 0.626993 | 0.769404 |
| SGD adaptive | 0.424209 | 0.534328 |
| Expanding target mean | 1.496969 | 1.659610 |
| Previous target | 0.133948 | 0.208678 |
| Raw five-feature mean | 0.588404 | 0.731256 |

In this evaluation the **previous-target baseline performed best**. The target is
an aggregate of the same polling series, partly circular and forward-filled, so
this demonstrates **aggregate approval reconstruction/nowcasting from polling
features**, not political forecasting, not a River benchmark reproduction, and
not production readiness. Adaptive SGD beats the expanding-target-mean and
raw-feature-mean baselines but not the previous-target baseline; results are
reported as measured.

### Executed synthetic result

`sgd_learn` (1000 rows, 400 learned, 600 scored, regime change at row 500,
`η=0.05`, 20 ordered passes): adaptive overall MAE/RMSE 0.141583/0.407878;
frozen 1.474419/1.983568; expanding mean 1.001343/1.195431; previous target
0.837283/1.046966. Adaptive scoring recovers after the declared regime change;
pre/post-change and baseline rows are printed by the example. The recorded run
does not promise improvement on other data.

## Deferred capabilities and limitations

Deferred: L1/L2/elastic-net regularization; mini-batch or gradient-averaging
updates; learning-rate schedules/decay and update counters; momentum/Adam or
other optimizers; sampling/shuffling and IID mini-batches; generic
optimizer/model/`fit` traits; convergence detection or early stopping;
serialization/checkpointing; performance measurement/optimization; `f64`
accumulation; and cross-algorithm/pre-hardware audit work.

Limitations: ordered updates are order-dependent; a fixed learning rate does not
guarantee convergence or stability for arbitrary data; fixed-step passes can
approach a cyclic fixed point that differs from the least-squares optimum; the
examples' learning rates and passes are example-specific; no universal warm-up,
accuracy or production-readiness claim is made; and the real-data example is
nowcasting evidence only.

## Local code and Cargo targets

| Path | Purpose |
|---|---|
| `mod.rs` | `Sgd`, `SgdState`, `SgdSample`, `SgdError`, `StateModel` implementation |
| `tests/sgd.rs` | Integration target `sgd` |
| `examples/sgd_learn.rs` | Synthetic learning/prediction example `sgd_learn` |
| `examples/sgd_regression.rs` | Real-data CSV example `sgd_regression` |
| `../README.md` | Linear-family membership |
| `../../../../docs/ML_VERTICAL_SLICES.md` | Slice matrix and evidence entry |

The Architect technical design and Planner implementation plan are separate,
issue-linked review records and are not duplicated here.

## References

- [Issue #50](https://github.com/Skybrique/seqvex/issues/50)
- [Development contract](../../../../docs/DEVELOPMENT.md)
- [Feature development runbook](../../../../FEATURE_DEVELOPMENT.md)
- [Failure and recovery](../../../../docs/FAILURE_AND_RECOVERY.md)
- [Vertical-slice evidence](../../../../docs/ML_VERTICAL_SLICES.md)
- [State-transition contract](../../../foundation/state/README.md)
- [Execution layer](../../../execution/README.md)
- [SGD tests](tests/sgd.rs), [synthetic example](examples/sgd_learn.rs),
  [real-data example](examples/sgd_regression.rs)
