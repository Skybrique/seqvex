# SGD-trained linear regression

> **Status and scope.** This is the canonical SGD algorithm document for
> [#50](https://github.com/Skybrique/seqvex/issues/50). SGD-trained linear
> regression is the ordered, incremental member of the linear family: it learns
> from one supervised observation at a time with a fixed learning rate and
> predicts from the learned weights. Validation evidence was recorded on
> 2026-10-17. The original slice was merged through
> [PR #51](https://github.com/Skybrique/seqvex/pull/51). The #55 baseline was
> measured and reviewed from an uncommitted package on the recorded base;
> current integration status and the 2026-10-19 performance/resource baseline
> are tracked in
> [#55](https://github.com/Skybrique/seqvex/issues/55). Parent/workstream
> closure, documentation/integration gates and production readiness remain
> separate. This does not imply that the #55 package is published or integrated.

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
[PR #51](https://github.com/Skybrique/seqvex/pull/51). The #55 baseline was
measured and reviewed from an **uncommitted** package on base
`ad6dc59ff5267e007ca7a571ec247d1545e9690b`; it is **not** part of that base
commit. Current integration status is tracked in
[#55](https://github.com/Skybrique/seqvex/issues/55). Reviewed source hashes:

- `Cargo.toml` — `caa3dd69c9b2d05793ba14870c2cda429fda8a90aa5e15a1760e7f8de2e029a7`
- `benches/sgd.rs` — `2c020b2e561158319d2f0b768c6d6ffc865fcdb986fe7638782d8c2de180d73a`

The raw invocation log is the measurement artifact and is not published; its
SHA-256 is
`a6bcd435a037f9c6bca6f1d4ba80ed28fdb8547d9fe64ea28f1d858d947026c2`.
The [release measurement handoff](https://github.com/Skybrique/seqvex/issues/55#issuecomment-6265273066)
is the provenance record for the release command, reported exit status,
toolchain and environment. The
[implementation/check handoff](https://github.com/Skybrique/seqvex/issues/55#issuecomment-6264966803)
covers the implementation and checks; the
[independent source review](https://github.com/Skybrique/seqvex/issues/55#issuecomment-6265020203)
covers the package; the
[reconciled measurement review](https://github.com/Skybrique/seqvex/issues/55#issuecomment-6267271634)
covers the results.

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
| 2 | 4.5 (5.0; 3.9–5.8; 0.2) | 61.3 (64.5; 59.7–65.4; 1.2) | 62.0 (64.2; 59.3–66.7; 1.5) | 57.5 (61.1; 55.2–61.4; 2.4) |
| 8 | 7.3 (7.8; 6.4–9.5; 0.3) | 64.4 (76.1; 61.7–82.9; 2.0) | 64.5 (68.7; 62.9–74.5; 1.8) | 60.7 (62.6; 57.2–64.6; 2.0) |
| 32 | 15.5 (18.4; 14.1–21.0; 1.3) | 80.6 (88.8; 78.1–93.1; 2.7) | 81.3 (83.3; 78.2–85.8; 1.5) | 82.8 (90.8; 77.1–101.6; 3.8) |
| 128 | 97.5 (117.6; 95.9–123.3; 2.3) | 207.4 (216.3; 201.4–232.2; 4.2) | 209.6 (225.4; 200.6–253.0; 9.4) | 205.0 (234.4; 200.3–305.6; 16.2) |
| 512 | 380.9 (406.5; 368.4–426.9; 16.8) | 685.8 (819.9; 636.2–972.8; 54.8) | 670.5 (743.1; 628.4–917.5; 43.7) | 677.3 (762.6; 632.1–770.4; 51.2) |

Median-derived throughput (M obs/s): predict `222.2 / 137.3 / 64.4 / 10.25 /
2.63`; direct `16.31 / 15.53 / 12.41 / 4.82 / 1.458`; `process_one`
`16.13 / 15.50 / 12.30 / 4.77 / 1.491`; executor
`17.39 / 16.47 / 12.08 / 4.88 / 1.476`.

**Initialization and resources.** Repeated warmed initialization/drop cost
(aggregate average; counter sampling included; not cold-start or a
distribution): `sgd-new` 4.0/5.0/4.0/4.1/6.2 ns/init, 0 allocs/0 B;
`initial-state` 27.7/28.7/29.7/39.3/42.2 ns/init, 1 alloc/4D bytes. Prediction
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

**Controls and wrapper comparison.** `dot`-only control medians:
2.0/3.1/8.7/41.8/209.9 ns/op; two-allocation control medians:
52.0/55.9/55.8/55.3/74.3 ns/op. These are descriptive; no timing subtraction or
allocator cost share is valid. No wrapper-performance winner is claimed:
`D=2`/`8` direct-vs-`process_one` is not materially different and
direct-vs-executor is borderline/noisy; `D=32`/`128`/`512` are not materially
different for both comparisons. Measured log-log slopes are ≈0.827 (predict) and
≈0.433 (update-direct) over `D=2..512`; subunit empirical slopes do not
contradict the O(D) source work (feature scan, dot, scale, add, candidate scan).
A fixed-cost contribution is a possible explanation, but dominance was not
measured.

**Limitations.** One invocation; no repeat-invocation variance, cross-machine
comparison or individual-operation tail measurement. No cold-start or RSS
measurement; instrumentation-bias magnitude is unmeasured. No optimization,
forecasting-quality or production-readiness claim. Optimization remains
deferred.

## Deferred capabilities and limitations

Deferred: L1/L2/elastic-net regularization; mini-batch or gradient-averaging
updates; learning-rate schedules/decay and update counters; momentum/Adam or
other optimizers; sampling/shuffling and IID mini-batches; generic
optimizer/model/`fit` traits; convergence detection or early stopping;
serialization/checkpointing; performance optimization; `f64`
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
| `benches/sgd.rs` | Benchmark target `sgd` (performance/resource baseline, #55) |
| `../README.md` | Linear-family membership |
| `../../../../docs/ML_VERTICAL_SLICES.md` | Slice matrix and evidence entry |

The Architect technical design and Planner implementation plan are separate,
issue-linked review records and are not duplicated here.

## References

- [Issue #50](https://github.com/Skybrique/seqvex/issues/50)
- [Issue #55](https://github.com/Skybrique/seqvex/issues/55)
- [Development contract](../../../../docs/DEVELOPMENT.md)
- [Feature development runbook](../../../../FEATURE_DEVELOPMENT.md)
- [Failure and recovery](../../../../docs/FAILURE_AND_RECOVERY.md)
- [Vertical-slice evidence](../../../../docs/ML_VERTICAL_SLICES.md)
- [State-transition contract](../../../foundation/state/README.md)
- [Execution layer](../../../execution/README.md)
- [SGD tests](tests/sgd.rs), [synthetic example](examples/sgd_learn.rs),
  [real-data example](examples/sgd_regression.rs), [benchmark](benches/sgd.rs)
- [Implementation/check handoff (#55)](https://github.com/Skybrique/seqvex/issues/55#issuecomment-6264966803)
- [Independent source review (#55)](https://github.com/Skybrique/seqvex/issues/55#issuecomment-6265020203)
- [Release measurement handoff (#55)](https://github.com/Skybrique/seqvex/issues/55#issuecomment-6265273066)
- [Reconciled measurement review (#55)](https://github.com/Skybrique/seqvex/issues/55#issuecomment-6267271634)
