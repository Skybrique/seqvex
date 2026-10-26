# SGD-trained linear regression

> **Status and scope.** This is the canonical SGD algorithm document for
> [#50](https://github.com/Skybrique/seqvex/issues/50). SGD-trained linear
> regression is the ordered, incremental member of the linear family: it learns
> from one supervised observation at a time with a fixed learning rate and
> predicts from the learned weights. Validation evidence was recorded on
> 2026-10-17. The original slice was merged through
> [PR #51](https://github.com/Skybrique/seqvex/pull/51). The replacement
> performance/resource baseline was measured and reviewed from an uncommitted
> cyclic-control correction on base
> `67d421cc9afcacd3b3f11c401075e91ee411a11c`; current integration status is
> tracked in
> [#55](https://github.com/Skybrique/seqvex/issues/55). Parent/workstream
> closure, documentation/integration gates and production readiness remain
> separate. This does not imply that the correction belongs to that base or is
> merged.

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
use seqvex::foundation::numerical::{Vector, Vector64};
use seqvex::foundation::observation::Observation;
use seqvex::foundation::state::StateModel;
use seqvex::models::linear::sgd::{
    Sgd, Sgd64, SgdError, SgdError64, SgdSample, SgdSample64, SgdServing32,
    SgdServingConversionError, SgdState, SgdState64,
};
```

| Item | Signature / shape |
|---|---|
| `Sgd::new` | `new(dimension: usize, learning_rate: f32) -> Result<Sgd, SgdError>` |
| `Sgd::dimension` | `-> usize` |
| `Sgd::learning_rate` | `-> f32` |
| `Sgd::initial_state` | `-> SgdState` (`w = 0`) |
| `Sgd::predict` | `predict(&self, state: &SgdState, features: &Vector) -> Result<f32, SgdError>` |
| `StateModel::update` (C) | `update(&self, state: &SgdState, observation: &Observation<SgdSample>) -> Result<SgdState, SgdError>` |
| `SgdState` | `{ pub w: Vector }` — per-execution `f32` learned weights |
| `SgdSample` | `{ pub features: Vector, pub target: f32 }` |
| `SgdError` | `ZeroDimension`, `DimensionMismatch { expected, actual }`, `InvalidLearningRate`, `NonFiniteInput`, `NonFiniteCandidate`, `NonFiniteOutput` |
| `Sgd64::new` | `new(dimension: usize, learning_rate: f64) -> Result<Sgd64, SgdError64>` |
| `Sgd64::predict` | `predict(&self, state: &SgdState64, features: &Vector64) -> Result<f64, SgdError64>` |
| `Sgd64::to_serving_f32` | `to_serving_f32(&self, state: &SgdState64) -> Result<SgdServing32, SgdServingConversionError>` |
| `StateModel::update` (A) | `update(&self, state: &SgdState64, observation: &Observation<SgdSample64>) -> Result<SgdState64, SgdError64>` |
| `SgdState64` / `SgdSample64` | `{ pub w: Vector64 }` / `{ pub features: Vector64, pub target: f64 }` |
| `SgdServing32` | Immutable `f32` snapshot; `dimension`, `weights`, output-validating `predict(&Vector) -> Result<f32, SgdError>` |
| `SgdServingConversionError` | `StateDimensionMismatch { expected, actual }`, `SourceNonFinite { index }`, `Overflow { index }`, `Underflow { index }` |

**Precision profiles.** `Sgd` is the existing strict-`f32` C surface and is
unchanged. `Sgd64` is the native-`f64` A learner. B reuses A's learner and creates
an independently owned, immutable `SgdServing32` snapshot through checked
conversion; the snapshot has no update, resume, transform or back-conversion
path. A and C expose one output-validating `predict`; there is no bare/checked
pair. Conversion checks the state dimension first, then source finiteness, cast
overflow and nonzero-to-zero underflow, and accepts finite values that round to
`f32::MAX` without clamping. Serving replacement is caller-owned and explicit;
a failed conversion or acceptance does not install a snapshot.

There is **no** `fit` method on `Sgd` or `Sgd64`. Learning is `update`; historical
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
  an example policy, not a property of `Sgd`. Its standardizer computes in `f64`
  before narrowing to `f32`, so `sgd_regression` is mixed-precision legacy, not
  strict C.

## Ownership, continuation and reset

- `Sgd` holds only immutable configuration (`dimension`, `learning_rate`) and
  implements `StateModel` with `&self`. One model is shareable.
- The learned `w` lives in `SgdState`, owned by the execution session; there is
  no persistent model-owned workspace or scratch (each update creates transient
  delta/candidate vectors).
- Continue by passing the committed `SgdState` to the next `update`. Reset with
  `model.initial_state()`.
- `Sgd64` holds only immutable configuration and owns no state; `SgdState64` is
  the caller-owned native-`f64` learning state. B's `SgdServing32` owns only
  serving weights and is immutable. Continued A learning does not mutate an
  already-created snapshot; converting again creates a new snapshot.
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

The seeded LCG generates the synthetic data in `f64` and then supplies `f32` model
rows. **SGD itself needs no RNG** and does not shuffle. `sgd_learn` is therefore a
supplied-`f32` model demonstration, not strict-C generation evidence. The
recorded run is reported in the Evidence section; it does not promise improvement
on other data.

### `sgd_profiles` — deterministic A/B/C quality fixture

`SGD-QUALITY800` is an in-memory, no-CSV fixture. A and B use native-`f64`
features and targets; C generates in strict `f32`. The named configuration is
`D=3` (two features plus an appended `1.0` intercept), 400 training rows, 400
held-out scoring rows, `η=0.01`, and 200 ordered passes (80,000 updates for each
of A and C). B performs no training updates: it is one checked conversion of A's
post-training state, evaluated with explicitly narrowed `f32` features against
A's `f64` targets. Scoring is exactly 400 fixed predictions per profile; no
labels enter the predictor and no updates occur during or after scoring.

The fixture records MAE and R² for A, B and C and enforces the frozen limits:

| Profile | MAE limit | R² limit |
|---|---:|---:|
| A native `f64` | `1e-5` | `1 - 1e-9` |
| B checked `f32` serving | `1e-4` | `1 - 1e-6` |
| C strict `f32` | `1e-4` | `1 - 1e-6` |

The accepted design-time conditioning check is `κ≈1.44790 ≤ 10`; the example
prints that accepted value and does not recompute a new conditioning study.
`SGD-QUALITY800-C` is the strict-C generation evidence: its features, targets and
`η` are generated in `f32`, not narrowed from f64.

Run:

```sh
cargo run --locked --example sgd_profiles --all-features
cargo +1.85.1 run --locked --example sgd_profiles --all-features
```

These are deterministic-fixture results, not general accuracy, convergence,
performance or production-readiness claims.

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
`sgd_profiles` prints the accepted conditioning value and one line per profile
with updates, conversions, scores, MAE and R².
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

Validation evidence was updated on 2026-10-26 for the A/B/C profile package.
The earlier 2026-10-17 evidence remains historical; this is not a
production-readiness claim.

- **Historical 2026-10-17 record:** the `sgd` test target had **16 passed** on
  default Rust 1.97.1 and MSRV Rust 1.85.1. That record is preserved here and is
  not silently rewritten.
- **Current `sgd` test target:** **33 passed** on default Rust 1.97.1 and MSRV
  Rust 1.85.1. The original 17 tests remain, plus A/B/C controls for native-f64
  and strict-f32 every-step predictions/states, exact-rational trajectories,
  native-f64 reduction/grouping and discrimination, conversion boundaries,
  output/precedence failures, learned-state failures, continuation, execution
  equivalence, A stream isolation, order/reproducibility and serving replacement.
- **Example tests:** `sgd_learn` **2 passed**; `sgd_regression` **13 passed**;
  `sgd_profiles` **5 passed**.
- **All-target tests:** default and MSRV `cargo test --locked --all-targets
  --all-features` passed, including the existing harness-free benchmark mains as
  test evidence only. No performance claim is made from those runs.
- **Doctests:** default and MSRV `cargo test --locked --doc --all-features`
  passed with 0 tests.
- **Documentation build:** `cargo doc --locked --no-deps --all-features` produced
  only the known unrelated unresolved-`Matrix` warning at
  `src/models/classic/decision_tree.rs:14`; no new warning was introduced.
- **Clippy:** `cargo clippy --locked --all-targets --all-features -- -D warnings`
  passed.
- Check B (frozen `D=4`, `N=64`, advance-before-draw supplied `f32` fixture,
  `η=0.01`, 200 ordered passes, exact-rational oracle). Both sub-fixtures pass the
  unchanged criterion `J_sgd − J* ≤ 1e-6·J_initial + 1e-3·J*`:

| Sub-fixture | `J*` | Objective gap | Allowed gap |
|---|---|---|---|
| noisy | 1.53865282598672883e-3 | 7.38227672922959696e-7 | 7.03698321273655127e-6 |
| noiseless | 3.37995574951686784e-15 | 9.00228264632184831e-12 | 5.45352837212015964e-6 |

  The noisy fixture's `κ(X)` is 1.916249. The noiseless `J*` is a small nonzero
  `f32` floor, not an exact-zero claim.

### `SGD-QUALITY800` profile result

The deterministic in-memory fixture uses `D=3`, 400 training rows, 400 held-out
scoring rows, `η=0.01`, and 200 ordered passes. A and C each perform 80,000
updates; B performs zero updates and one checked conversion; every profile makes
400 fixed scoring predictions. The accepted design-time conditioning value is
`κ≈1.44790 ≤ 10`.

| Profile | Updates | Conversions | Scores | MAE | R² |
|---|---:|---:|---:|---:|---:|
| A native `f64` | 80,000 | 0 | 400 | `3.871902798380e-15` | `1.000000000000000` |
| B checked `f32` serving | 0 | 1 | 400 | `4.409977633041e-8` | `0.999999999999997` |
| C strict `f32` | 80,000 | 0 | 400 | `2.736076712608e-6` | `0.999999999991000` |

These values were identical on default Rust 1.97.1 and MSRV Rust 1.85.1. They are
deterministic-fixture results, not general accuracy, convergence, performance or
production-readiness claims.

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

### Performance and resource baseline (#55)

Benchmark target: `src/models/linear/sgd/benches/sgd.rs`, registered as
`[[bench]] name = "sgd"` with `harness = false` (`Cargo.toml`). Release
command:

```sh
cargo bench --locked --bench sgd
```

The debug smoke (`cargo test --locked --bench sgd --all-features`) runs a tiny
cross-path/prediction/initialization check and exits before the release matrix;
it is not decision-grade. The release run is one invocation; no optimization is
implied.

**Package identity and provenance.** The original SGD slice was merged through
[PR #51](https://github.com/Skybrique/seqvex/pull/51). The replacement baseline
was measured and reviewed from an **uncommitted** cyclic-control correction on
base `67d421cc9afcacd3b3f11c401075e91ee411a11c`; it is **not** part of that base
commit. Current integration status is tracked in
[#55](https://github.com/Skybrique/seqvex/issues/55). Reviewed source hashes:

- `Cargo.toml` — `caa3dd69c9b2d05793ba14870c2cda429fda8a90aa5e15a1760e7f8de2e029a7`
- `benches/sgd.rs` — `b754db59b381c18fa66437205fb1e4c6da7a60dd5bb425d644297b43f46bd3bd`

The replacement raw invocation log supports the profile/runtime checks and the
measured output and is not published; its SHA-256 is
`1eeda215fda060e521e004befad1dc78d1d9bb561244472cd2b7c0b34fa7f5fc`.
The [replacement execution handoff](https://github.com/Skybrique/seqvex/issues/55#issuecomment-6273196910)
records the exact command, reported exit status, toolchain and detailed
environment; the [replacement evidence](https://github.com/Skybrique/seqvex/issues/55#issuecomment-6273278258)
reconciles the measured output. The
[correction implementation/check handoff](https://github.com/Skybrique/seqvex/issues/55#issuecomment-6272984646)
covers the implementation and checks; the
[corrected source review](https://github.com/Skybrique/seqvex/issues/55#issuecomment-6273061283)
covers the package.

**Environment and profile.** One optimized `bench`-profile invocation
(Rust/Cargo 1.97.1; `debug_assertions=false`; `RUSTFLAGS` unset; no Cargo
profile overrides), 20 runs per cell. WSL2 (Ubuntu 24.04), Linux 6.18.x, x86_64;
12th Gen Intel Core i7-12700H; 20 logical CPUs. Numbers are device-local and are
not production guarantees.

**Frozen workload.** `D ∈ {2, 8, 32, 128, 512}`; steps/run
`{400k, 200k, 100k, 25k, 5k}`; harness warm-up
`H = max(steps/10, 1000)` = `{40k, 20k, 10k, 2.5k, 1k}`; `H + 20·steps` total
calls = `{8,040,000; 4,020,000; 2,010,000; 502,500; 101,000}`. That total
includes the **untimed** harness warm-up `H`; the timed portion is `20·steps`.
Fixture length 2048; seed `0x5eed55d000000001`; `η=0.05`; 512-update warm-start;
measured index `(512 + k) mod 2048`; features generated from the shared LCG with
`inv = (1.0f64 / sqrt(D as f64)) as f32` and `x = (2.0f32 * raw) * inv`;
targets via `fixed_weights`/`linear_targets`.

**Measurement boundaries.** Setup (fixture, model, warm state) is outside
timing. Timed `predict`/`update` calls use the public checked APIs; update
timing includes committed-state replacement and the drop of the previous state.
Prediction uses fixed warmed weights; update samples are continued-state run
averages. Both precheck passes compare every prediction to a per-index `to_bits`
reference; update prechecks compare direct, foundation `process_one` and
`StreamingExecutor` states bitwise after harness warm-up and every run boundary.
Allocation windows compare two `steps` blocks after `H` warm-up calls. Resource
probes read `live_after` after replacement; the peak counter retains the
transient overlap.

**Reconciled timing.** ns/op, median (p95; min–max; IQR), over 20 run-average
samples. These are **not** individual-operation tail latencies.

| D | predict | update direct | update `process_one` | update executor |
|---|---|---|---|---|
| 2 | 4.7 (5.1; 4.2–5.1; 0.2) | 61.2 (63.2; 58.3–64.3; 0.9) | 60.1 (62.2; 58.1–62.4; 1.7) | 56.5 (57.8; 54.8–58.8; 0.7) |
| 8 | 7.1 (7.2; 6.4–7.5; 0.2) | 65.6 (70.6; 63.9–70.7; 1.7) | 67.7 (75.3; 64.5–104.9; 2.7) | 62.8 (70.4; 60.9–70.8; 3.6) |
| 32 | 15.8 (16.2; 13.4–16.3; 0.5) | 82.9 (86.9; 77.5–90.4; 2.1) | 81.6 (87.7; 77.2–89.4; 4.0) | 79.9 (85.8; 75.3–87.2; 1.4) |
| 128 | 97.5 (99.5; 92.0–102.8; 4.2) | 202.5 (230.3; 193.6–234.1; 2.9) | 205.2 (267.9; 194.0–326.5; 8.5) | 204.5 (212.8; 198.1–216.3; 2.7) |
| 512 | 372.6 (401.5; 344.8–410.3; 37.9) | 687.6 (796.2; 629.1–829.6; 48.2) | 693.4 (747.3; 645.1–791.7; 45.7) | 690.0 (755.7; 629.7–776.0; 68.7) |

Median-derived throughput (M obs/s): predict `212.2 / 140.9 / 63.46 / 10.26 /
2.684`; direct `16.33 / 15.25 / 12.06 / 4.939 / 1.454`; `process_one`
`16.64 / 14.77 / 12.26 / 4.873 / 1.442`; executor
`17.71 / 15.93 / 12.51 / 4.890 / 1.449`.

**Initialization and resources.** Repeated warmed initialization/drop cost
(aggregate average; counter sampling included; not cold-start or a
distribution): `sgd-new` 4.8/3.8/3.9/4.8/2.9 ns/init, 0 allocs/0 B;
`initial-state` 50.9/29.8/26.9/38.5/39.4 ns/init, 1 alloc/4D bytes. Prediction
allocates 0 allocs/0 B; every update path allocates 2 allocs/op and 8D requested
bytes/op (16/64/256/1024/4096 B); both allocation blocks match. Retained state
is 4D; the **initial-state additional peak is 4D**. On the measured x86-64
target, runtime `size_of` is `Vector` 24 B, `SgdState` 24 B, `Sgd` 16 B,
`StreamingExecutor` 32 B; logical caller-owned storage is `16 + 24 + 4D` B
(48/72/168/552/2088 B), executor-owned storage is `16 + 32 + 4D` B
(56/80/176/560/2096 B), with embedded headers counted once. Update additional
peak is 8D (16/64/256/1024/4096 B) over the pre-update baseline; `live_after`
delta is 0, meaning equal retained heap size after replacement. Allocator
live/requested bytes are not OS RSS. The 12D total pre-commit overlap is
inferred from the 4D baseline, the 8D additional peak and source lifetimes; it
was not directly observed as a total.

**Controls and wrapper comparison.** The cyclic `dot`-only control uses fixed
warmed weights and prebuilt observations, starting at `WARM_START`; its index
continues across harness warm-up and timed runs and follows the same sequence as
prediction. Its medians are 2.5/4.0/9.3/41.0/202.1 ns/op; the two-allocation
control medians are 50.5/51.2/50.3/53.8/71.3 ns/op. These are descriptive; no
timing subtraction or allocator cost share is valid. `D=2` direct-vs-executor
satisfies the harness's `clearly measurable` rule; the other reported
comparisons are not materially different. That is a descriptive
within-invocation materiality rule, not statistical significance or a general
wrapper winner. Measured log-log slopes are ≈0.819 (predict) and ≈0.430
(update-direct) over `D=2..512`; subunit empirical slopes do not contradict the
O(D) source work (feature scan, dot, scale, add, candidate scan). A fixed-cost
contribution is a possible explanation, but dominance was not measured.

**Historical fixed-vector control.** The original #55 dot control repeated one
fixed observation vector. Its source hash was
`2c020b2e561158319d2f0b768c6d6ffc865fcdb986fe7638782d8c2de180d73a`, its raw log
SHA-256 was
`a6bcd435a037f9c6bca6f1d4ba80ed28fdb8547d9fe64ea28f1d858d947026c2`, and its dot
medians were 2.0/3.1/8.7/41.8/209.9 ns/op for D=2/8/32/128/512. That control is
historical, not a second maintained benchmark. Its complete tables are in the
[README at commit 67d421c](https://github.com/Skybrique/seqvex/blob/67d421cc9afcacd3b3f11c401075e91ee411a11c/src/models/linear/sgd/README.md)
and the [original reconciled evidence](https://github.com/Skybrique/seqvex/issues/55#issuecomment-6267271634).
The workloads differ; no cross-run speedup or isolated cache-cost claim is made.

**Limitations.** One replacement invocation; no repeat-invocation variance,
cross-machine comparison or individual-operation tail measurement. No
cold-start or RSS measurement; instrumentation-bias magnitude is unmeasured.
No optimization, forecasting-quality or production-readiness claim.
Optimization remains deferred.

## Deferred capabilities and limitations

Deferred: L1/L2/elastic-net regularization; mini-batch or gradient-averaging
updates; learning-rate schedules/decay and update counters; momentum/Adam or
other optimizers; sampling/shuffling and IID mini-batches; generic
optimizer/model/`fit` traits; convergence detection or early stopping;
serialization/checkpointing; performance optimization; hidden `f64`
accumulation in the strict-`f32` C path (use A for native f64); and
cross-algorithm/pre-hardware audit work.

Limitations: ordered updates are order-dependent; a fixed learning rate does not
guarantee convergence or stability for arbitrary data; fixed-step passes can
approach a cyclic fixed point that differs from the least-squares optimum; the
examples' learning rates and passes are example-specific; no universal warm-up,
accuracy or production-readiness claim is made; and the real-data example is
nowcasting evidence only.

## Local code and Cargo targets

| Path | Purpose |
|---|---|
| `mod.rs` | C `Sgd`/`SgdState`/`SgdSample`/`SgdError`; canonical A/B reexports |
| `profiles.rs` | A `Sgd64`/`SgdState64`/`SgdSample64`/`SgdError64`; B `SgdServing32` and checked conversion |
| `tests/sgd.rs` | Integration target `sgd`, including A/B/C profile controls |
| `examples/sgd_learn.rs` | Synthetic learning/prediction example `sgd_learn` |
| `examples/sgd_profiles.rs` | Deterministic A/B/C `SGD-QUALITY800` example |
| `examples/sgd_regression.rs` | Real-data CSV example `sgd_regression` |
| `benches/sgd.rs` | Benchmark target `sgd` (performance/resource baseline, #55) |
| `../README.md` | Linear-family membership |
| `../../../../docs/ML_VERTICAL_SLICES.md` | Slice matrix and evidence entry |

The Architect technical design and Planner implementation plan are separate,
issue-linked review records and are not duplicated here.

## References

- [Issue #50](https://github.com/Skybrique/seqvex/issues/50)
- [Issue #55](https://github.com/Skybrique/seqvex/issues/55)
- [SGD continuation #71](https://github.com/Skybrique/seqvex/issues/71),
  [implementation #72](https://github.com/Skybrique/seqvex/issues/72),
  [numerical acceptance #73](https://github.com/Skybrique/seqvex/issues/73),
  [sequential/quality acceptance #74](https://github.com/Skybrique/seqvex/issues/74)
- [Development contract](../../../../docs/DEVELOPMENT.md)
- [Feature development runbook](../../../../FEATURE_DEVELOPMENT.md)
- [Failure and recovery](../../../../docs/FAILURE_AND_RECOVERY.md)
- [Vertical-slice evidence](../../../../docs/ML_VERTICAL_SLICES.md)
- [State-transition contract](../../../foundation/state/README.md)
- [Execution layer](../../../execution/README.md)
- [SGD tests](tests/sgd.rs), [synthetic example](examples/sgd_learn.rs),
  [real-data example](examples/sgd_regression.rs), [benchmark](benches/sgd.rs)
- [Correction implementation/check handoff (#55)](https://github.com/Skybrique/seqvex/issues/55#issuecomment-6272984646)
- [Corrected source review (#55)](https://github.com/Skybrique/seqvex/issues/55#issuecomment-6273061283)
- [Replacement execution handoff (#55)](https://github.com/Skybrique/seqvex/issues/55#issuecomment-6273196910)
- [Replacement evidence (#55)](https://github.com/Skybrique/seqvex/issues/55#issuecomment-6273278258)
- [Historical fixed-vector reconciled evidence (#55)](https://github.com/Skybrique/seqvex/issues/55#issuecomment-6267271634)
