# Recursive Least Squares (RLS)

> **Learning and readiness:** RLS supervised updates ingest features/targets and adapt state. Complete model delivery must demonstrate data-driven learning before user-facing prediction, with approved readiness, numerical and quality criteria under the [mandatory training-to-inference contract](../../../../docs/DEVELOPMENT.md#model-training-and-valid-inference--mandatory). An initialized/reset prior is not evidence of a learned model. Streaming prediction/update examples must use an eligible learned history without future-label leakage. Do not invent a warm-up, retry or reset policy; raise unresolved cases to the maintainer.

## Purpose

Recursive least squares (RLS) with a forgetting factor — the ordered/stateful
member of the linear-model family. The canonical namespace is
`models::linear::rls`.

## Location and compatibility

- Canonical: `seqvex::models::linear::rls` (`src/models/linear/rls/`).
- Compatibility: `seqvex::models::online::rls` and `seqvex::models::online`
  re-export the same canonical `Rls`, `RlsState`, `RlsSample`, `RlsError`,
  `RlsBatchError`, and `RlsStateAdmissionError` types and add no behavior. See
  the [vertical-slice evidence](../../../../docs/ML_VERTICAL_SLICES.md).

## Responsibility

- Legacy strict-`f32` surface — `Rls`, `RlsState`, `RlsSample`, `RlsError`,
  `RlsBatchError`, prediction, ordered update, and bounded
  `process_batch_reference`.
- Native-`f64` profile — `Rls64`, `RlsState64`, `RlsSample64`, `RlsError64`, and
  `RlsBatchError64`, with the same ordered recurrence and coupled commit.
- Checked `f32` serving — `RlsServing32` plus
  `Rls64::to_serving_f32`/`RlsServingConversionError`.
- External-state admission — shared `RlsStateAdmissionError`, with
  `RlsState64::admit` and legacy `RlsState::admit`.

## Relationship to Seqvex

RLS is the first slice whose model **state changes with every observation**, so
it is the strongest evidence so far about where adaptive state belongs:

- `Rls` holds only immutable configuration (`D`, `λ`, `δ`) and implements
  `StateModel` with a `&self` contract. It is shareable.
- The adaptive `(w, P)` lives entirely in `StateModel::State`, owned by the
  execution session (`StreamingExecutor`), not by the model.
- No model-owned workspace, cache, or scratch is used. The value-returning
  contract computes a candidate `(w', P')` and returns it.

This confirms, on a stateful model, the same finding linear regression made on a
stateless one: streaming does not require `&mut M`, and adaptive state is a
per-stream property.

That former limitation is resolved: the generic `StreamingExecutor` now borrows
the model immutably (`&Model`), so one immutable `Rls` can drive two executors
with independent States. One shared `&Rls` with independent states is also
expressible through the foundation `process_one`/`process_stream` functions,
which take `&M`, and the tests exercise that.

## Precision profiles

| Profile | Learning | Retained learner state | Inference |
|---|---|---|---|
| **A: native `f64`** | `Rls64::update` in `f64` | private `RlsState64 { w64, P64 }` | `Rls64::{predict,predict_checked}` in `f64` |
| **B: `f64` learning, `f32` serving** | same A learner | authoritative `RlsState64` | `Rls64::to_serving_f32` creates `RlsServing32`; checked `f32` prediction |
| **C: strict `f32`** | existing `Rls::update` in `f32` | public legacy `RlsState { w32, P32 }` | existing `Rls::{predict,predict_checked}` in `f32` |

B converts only weights. The serving snapshot has no covariance, configuration,
learner state, update path, resume path, or back-conversion. Conversion checks
the state dimension before any coefficient, then scans weights in increasing
index order. A non-finite source reports `SourceNonFinite`; a finite source that
rounds beyond `f32` reports `Overflow`; a nonzero source that rounds to zero
reports `Underflow`. Representable subnormals, signed zero, and finite values
rounding to `f32::MAX` are preserved without clamping. There is no
`AllowLoss` policy.

The snapshot is a selected serving version, not a live mirror. Continued `f64`
learning remains authoritative. Conversion and checked acceptance do not install
anything: replacement is explicit caller assignment after a nonempty checked
selection succeeds. A failed conversion or acceptance leaves the source weights,
the active snapshot, and in-flight work unchanged.

### External-state admission

`RlsState64::admit(weights, covariance)` and legacy
`RlsState::admit(weights, covariance)` return `RlsStateAdmissionError` after
checking, in order: zero dimension; square covariance; weight length; finite
weights at the first index; finite covariance entries at the first row-major
entry; bitwise covariance symmetry at the first row-major pair. Admission
establishes only those structural, finite-value, and exact-symmetry facts. It
does not establish positive semidefiniteness, conditioning, update provenance,
learned readiness, accuracy, or future denominator positivity, and it performs
no repair.

Legacy `RlsState` fields remain public for compatibility. A caller can bypass
admission or mutate a state after admission, so C admission describes that call,
not a permanent invariant, and legacy update/prediction gain no new check.
A private `RlsState64` can be created through `admit` or internally by learning.

### Fixed-model inference

No batch, streaming, or executor adapter is added for fixed inference. A caller
loops in input order over one committed `RlsState64`, one `RlsServing32`, or one
legacy `RlsState`. A grouped checked caller may discard its local output vector
and return the first error, producing no partial group; an empty group returns
empty. That all-or-error behavior is a documented and tested caller protocol,
not a library batch API or a transaction over outputs already published one at a
time. A model-version change requires a new state or snapshot and a new call.

## Equations

For feature `x_t`, target `y_t`, parameters `w_t`, covariance `P_t`, and
forgetting factor `λ`:

```text
ŷ_t = w_{t-1}ᵀ x_t
e_t = y_t − ŷ_t
v_t = P_{t-1} x_t
d_t = λ + x_tᵀ v_t
k_t = v_t / d_t
w_t = w_{t-1} + k_t e_t
P_t = (P_{t-1} − v_t v_tᵀ / d_t) / λ
```

The covariance step uses the transparent rank-one form: since `P` is symmetric,
`k_t x_tᵀ P_{t-1} = v_t v_tᵀ / d_t`. Initialization is `w_0 = 0`,
`P_0 = δI` (symmetric positive-definite by construction).

## Failure and atomicity

Validation order is deterministic: dimensions (`x`, then `w`, then `P`),
observation finiteness, the denominator `d` (finite and `> 0`), and finally
candidate finiteness. Failures are reported as explicit `RlsError` variants; the
reference never panics on trust-boundary input.

Atomicity is structural. `w'` and `P'` are computed as locals and returned as
one `RlsState` only after both validate, so a failed update leaves the committed
`(w, P)` bitwise unchanged and there is no path that commits one component
without the other. The tests assert bitwise state preservation after every
failure class, directly and through the executor.

The incoming `P` is assumed symmetric positive semidefinite. States produced by
`initial_state` and successful updates are bitwise symmetric. External states in
checked workflows enter through admission, which verifies dimensions, finite
entries, and bitwise symmetry but does not establish positive semidefiniteness.
The denominator guard rejects a non-finite or nonpositive `λ + xᵀPx` for the
current observation; it does not certify general symmetry or positive
semidefiniteness and may pass an indefinite `P` in an unexcited direction. With
`λ=1`, `P=diag(-1,1)`, and `x=[0,1]`, the update succeeds and `P` remains
indefinite; with `x=[1,0]`, the denominator is zero and the update rejects. No
silent covariance repair is performed.

## Checked prediction

`Rls::predict_checked` is the explicit caller-side checked-prediction
operation. It delegates to `Rls::predict`, so input validation, its error
categories, and their precedence are unchanged. A finite result is returned
with identical `f32` bits; a non-finite result is rejected with
`RlsError::NonFiniteOutput`. Bare `Rls::predict` remains unchanged and does
not check output finiteness.

`NonFiniteOutput` is a new public error variant. Downstream exhaustive matches
over `RlsError` require another arm. Existing bare prediction behavior remains
unchanged.

The guard checks only the output value. It does not establish general state
finiteness, covariance symmetry or positive-definiteness, accuracy, or
statistical validity. The supplied state is unchanged.

The A and B forms follow the same bare/checked split. `Rls64::predict` checks
features, weights, covariance rows, then covariance columns before feature
finiteness and returns the ordered `f64` dot; `Rls64::predict_checked` rejects a
non-finite output. `RlsServing32::predict` checks feature length and finiteness,
then evaluates the ordered `f32` dot; its checked form rejects a non-finite
output. Neither path updates, fixes a model version implicitly, or substitutes a
default or stale prediction.

## Applicability and statistical contract

> Recorded by the R5 checkpoint from the completed D1–D4/R1–R4 evidence. This
> section states what RLS **assumes**, what the implementation **guarantees**,
> and what remains the caller's responsibility. It adds no API and changes no
> behaviour. Every quantitative statement is bounded by the
> [operating envelope](#operating-envelope); none is a universal guarantee.

RLS is an estimator defined by an optimisation objective, not a probabilistic
model. The contract keeps the identity that *defines* the algorithm separate
from the conditions under which its coefficients are *meaningful*. The general
failure semantics remain those of `docs/FAILURE_AND_RECOVERY.md`; this section
specialises them for RLS.

### Mathematical assumptions

The estimator is defined by the weighted least-squares objective

```text
J_t(w) = Σ_{j≤t} λ^{t−j} (y_j − wᵀ x_j)² + (λ^t / δ) ‖w‖²
```

whose minimiser solves `A_t w = b_t`, with

```text
A_t = Σ_{j≤t} λ^{t−j} x_j x_jᵀ + (λ^t / δ) I
b_t = Σ_{j≤t} λ^{t−j} y_j x_j .
```

`P_t` is the inverse-correlation matrix maintained by the recurrence: the
`A_t^{-1}`-equivalent whose residual `‖A_t P_t − I‖∞` is what the tests bound.
In exact arithmetic, `w_t` is the exact minimiser of `J_t`. This identity
requires no probabilistic assumption. The `f32` recurrence approximates that
result; tests compare it with the independently solved `f64` literal objective
within the stated tolerances. The recurrence is never used as its own oracle.

Enforced requirements: `D ≥ 1`; `0 < λ ≤ 1` finite; `δ > 0` finite; `x_t` and
`y_t` finite; consistent dimensions across `x`, `w`, and `P`.

### Statistical assumptions

These are assumptions about the **data-generating process**. They are required
to interpret `w_t` statistically, are **not** enforced or validated at runtime,
and must not be turned into API checks.

- **Exogeneity / conditional zero mean.** `E[y_t − x_tᵀ w* | x_t] = 0` (no
  omitted-variable bias, no feedback from `y` into `x`).
- **Persistent / sufficient excitation.** The directions whose coefficients are
  to be estimated are exercised often enough to dominate the prior and noise
  (see [Persistent excitation and identifiability](#persistent-excitation-and-identifiability)).
- **Identifiability.** No perfect collinearity; every estimated direction is
  supported by the data.
- **Stationary target for `λ = 1`.** If a single fixed `w*` is to be interpreted,
  the target must not change; otherwise the pooled objective is the wrong target.
- **Slowly varying target for `λ < 1`.** The drift rate must be compatible with
  the effective memory `≈ 1/(1−λ)`.
- **Bounded second moments** of the features and targets, where variance
  statements are made.
- **Noise independence** is *not* required for the numerical envelope (the
  covariance, denominator, and guards are noise-independent), but is required
  for the bias/variance interpretation above.

### Persistent excitation and identifiability

Three distinct notions must not be conflated:

1. **Mathematical identifiability.** A direction `u` is identifiable only if it
   is supported by the (windowed, `λ`-weighted) data Gram
   `G_t = Σ_{j≤t} λ^{t−j} x_j x_jᵀ`; if `uᵀ G_t u = 0`, the data contains no
   information about that direction.
2. **Finite-sample identifiability.** A weakly excited direction has small but
   non-zero `uᵀ G_t u`; the coefficient is determined but with high variance and
   slow convergence.
3. **Numerical conditioning.** A large condition number `κ(A_t)` means the
   *computation* is sensitive to rounding; it is not the same statement as
   statistical identifiability.

Consequences, stated without creating any threshold:

- An **exactly unexcited** direction cannot be reliably identified from data. Its
  weight is frozen (`P[·,e] == 0` gives zero gain), so `w_e` does not move; under
  `λ = 1` its `P_ee` stays at the ridge `δ`, and under `λ < 1` its `P_ee` inflates
  by `1/λ` per step and can eventually overflow. This is a data/regime property,
  and the implementation supplies no excitation diagnostic. The supported
  envelope and response to lost identifiability/overflow require explicit review;
  this observation does not authorize an assumed recovery or valid prediction.
- `λ = 1` keeps `A_t` formally invertible forever through `(λ^t/δ)I`, but a
  direction the data never excites is determined by the **ridge**, not by
  information — formal invertibility is not statistical identifiability.
- Large `κ` increases numerical sensitivity; conditioning is **not** dimension
  and **not** finite-sample identifiability.

Seqvex currently exposes **no** public condition-number or excitation
diagnostic, and R5 deliberately creates none (`condition_number()`,
`excitation_score()`, automatic warnings, normalisation, or automatic feature
scaling would all be new public surfaces requiring a future architecture
decision). Persistent excitation remains a user/data responsibility.

### Forgetting factor `λ` semantics

`λ=1`:

- the estimator is **pooled/ridge least squares** over all observations;
- its final objective is **permutation invariant** — the fitted `w_N` does not
  depend on observation order (the sequential trajectory still does);
- evaluating it sequentially remains meaningful, but interpreting a fixed `w*`
  requires a sufficiently stable target.

`0 < λ < 1`:

- the estimator is **exponentially recency-weighted**: observation `j` carries
  weight `λ^{t−j}` at time `t`;
- it is **order-sensitive** — order is semantically real;
- the effective memory and sample size scale as `≈ 1/(1−λ)`;
- it adapts to changing regimes at a rate governed by `λ`;
- excessive forgetting with missing excitation **amplifies covariance** in
  unexcited directions `≈ 1/λ` per step.

`λ` does **not** guarantee: consistency (for `λ<1` the objective never stops
changing), optimality, uncertainty quantification, an "optimal" value,
stationarity of the target, or that any particular `λ` is appropriate. RLS
exposes no auto-tuning and R5 recommends none.

### Initial covariance `δ` semantics

- `δ` sets the initial covariance `P_0 = δ I` and, equivalently, the ridge/prior
  weight `λ^t/δ` on `‖w‖²` in `J_t`.
- Larger `δ` ⇒ weaker prior ⇒ the first observations dominate sooner; smaller
  `δ` ⇒ stronger shrinkage of `w` toward `0`.
- `δ` interacts with feature scale. For a **uniform** feature rescaling
  `x → s·x`, the objective similarity is preserved by the matched prior
  `δ_s = δ / s²` (then `w(x̃; δ_s) = w(x; δ)/s` and predictions are invariant).
  This is a mathematical relation used to make scaling experiments exact; it is
  **not** an API transformation and the implementation applies no scaling.
- No specific `δ` is universally appropriate, and the implementation cannot infer
  or auto-select one. Choosing `δ` is a user hyperparameter decision.

### Numerical contract

A numerical guard is **not** a statistical guarantee.

- `RlsError::InvalidDenominator` (the denominator `d = λ + xᵀPx` is not finite
  and positive) is a **numerical guard**.
- `RlsError::NonFiniteCandidate` (the candidate `(w',P')` is not finite) is a
  **numerical guard**.
- A rejected update **preserves the previously committed state bitwise**.
- A guard **does not** prove the statistical model is invalid; it reports that
  the computed state lost positivity/finiteness.
- Absence of a guard **does not** prove numerical accuracy.
- A **statistical** failure (poor excitation, wrong model, drift too fast) is
  **not** automatically an implementation defect; it is a data/regime condition.

`P` is expected symmetric PSD on entry; symmetry is preserved exactly by the
update, and finiteness is checked on the candidate. The tested `f32` envelope is
summarised in [Operating envelope](#operating-envelope) and is deliberately not
converted into API limits.

### Sequential evaluation contract

RLS updates are ordered and state-dependent, so evaluation must be causal. The
required protocol for any honest sequential evaluation is:

1. compute `ŷ_t = w_{t−1}ᵀ x_t` from the **committed** state **before** consuming
   `y_t`;
2. form the prediction error `e_t = y_t − ŷ_t`;
3. update and commit `(w_t, P_t)` from `(x_t, y_t)`;
4. continue strictly in temporal order.

Conceptually **prohibited**: shuffling, sorting, resampling, using future
observations, building labels from future observations, and scoring an
observation after fitting it on itself. `λ=1` is order-invariant only for the
final pooled fit; the *trajectory* and the prequential error remain temporal.

Additional semantics:

- **Warm-up.** Early estimates reflect the prior `P_0 = δI`; they should be
  discarded or flagged until the data dominates.
- **Reset.** `reset` starts a new sequence (`w = 0`, `P = δI`) and discards
  history; it is the explicit recovery from a diverged or overflowed state.
- **Continued learning.** Preserving `(w, P)` across a boundary continues the
  same estimator; this is the difference from reset.
- **Regime changes.** Evaluate pre-shift steady state, peak transient,
  convergence, and post-shift steady state separately, plus the prequential
  series; do not build regime labels from future data.
- **Delayed labels.** If `y_t` arrives late, the prediction at `t` may use only
  information available at `t`; the update, once `y_t` arrives, must preserve the
  original observation order. Buffering and matching labels to observations is a
  caller responsibility.

This is a semantics description only. No evaluation framework is added.

### Failure taxonomy

The eleven categories below distinguish expected data/regime conditions, numerical
guards, and hard errors. Categories that have no `RlsError` representation are
**observational/statistical** and must not be promoted to new error variants.

| # | Category | Signal | Class | Intended action |
|---|---|---|---|---|
| 1 | Insufficient excitation | large `P`, slow/biased `w`, large prequential error | Expected data/regime | Monitor; adjust `λ`/design |
| 2 | Rank deficiency (exactly unexcited direction) | weight frozen; `λ<1` `P_ee` inflates | Expected data | Data responsibility; `λ<1` may eventually guard |
| 3 | Severe conditioning | large `κ(A_t)`; rising residual before prediction fails | Numerical (observation) | Rescale features; no auto-normalisation |
| 4 | Dynamic-range limitation | `λ=1`, large `N`, `P→0`; gradual error growth | Numerical (observation) | Bound horizon or use `λ<1` |
| 5 | Numerical cancellation | `P` loses relative accuracy at high disparity | Numerical (observation) | Feature-scale policy |
| 6 | Denominator failure | `RlsError::InvalidDenominator` | Numerical guard | Reject + preserve; investigate — not automatically a bug |
| 7 | Non-finite candidate / overflow | `RlsError::NonFiniteCandidate` | Numerical guard | Reject + preserve; reset if persistent |
| 8 | Invalid input | `RlsError::NonFiniteInput`, `RlsError::DimensionMismatch` (features) | Hard caller/API error | Reject input; caller fixes |
| 9 | Invalid parameter/state | `RlsError::ZeroDimension`, `InvalidForgettingFactor`, `InvalidInitialCovariance`, `DimensionMismatch` (state) | Hard config/API error | Reject; caller fixes |
| 10 | Genuine implementation defect | symmetry residual `≠ 0`; non-finite **committed** state; broken atomicity; wrong trajectory on validated input | Implementation defect | Escalate; code change required |
| 11 | Non-finite prediction output | `RlsError::NonFiniteOutput` (checked prediction only) | Numerical guard at the serving boundary | Reject output without substitution; supplied state is unchanged. Investigate operands, state and representability; the guard neither establishes state finiteness nor diagnoses the cause. |

`RlsError::DimensionMismatch` serves both input (#8) and state (#9); the
validation order decides the context. A genuine defect (#10) is tied only to
actual invariant violations, never to a statistical miss or a guard.

### Guarantee boundary

| Property | Type | Evidence | Guarantee? |
|---|---|---|---|
| `w_t` minimises `J_t` (up to `f32` error) | Mathematical | D1–D4 literal oracle | Yes (definition) |
| Atomic coupled `(w,P)` commit | Implementation | D1/D4 atomicity tests | Yes |
| Failed update preserves committed state bitwise | Implementation | D1/D4 | Yes |
| Ordered sequential/chunk semantics | Implementation | D4 | Yes |
| `λ=1` final-fit permutation invariance | Mathematical | D4 | Yes |
| `P` symmetry exactly `0.0` residual | Numerical | R1–R4 | Yes |
| Finite committed state or rejection | Numerical | R1–R4 | Yes |
| Numerical accuracy inside tested envelope | Numerical | R1–R4 | Within envelope only |
| Uniform scaling similarity with `δ_s = δ/s²` | Mathematical/Numerical | R3/R4 | Within tested envelope |
| Convergence under assumptions | Statistical | R2 | Only under [statistical assumptions](#statistical-assumptions) |
| Unbiasedness under exogeneity | Statistical | `DERIVED` | Conditional only |
| Variance decreasing with memory | Statistical | R2 direction | Directional only |
| Persistent excitation / identifiability | Statistical | R1/R2 | **Not guaranteed** — user/data |

### Operating envelope

Expressed in four categories; none is a universal statement.

**Demonstrated envelope** (passed the registered tolerances in the cited stage):

- D1–D4: `D ∈ {1,2,4}`, `N ≤ 100`, `λ ∈ {1,0.9,0.99}`, `δ ∈ {1,1e3}`.
- R1: `D ≤ 4`, fully excited, partial, weak, and correlated designs;
  `δ ∈ {1,1e3,1e6}`; runs to `N ≈ 10⁴`.
- R2: `D ∈ {2,4}`, `N ≈ 1400`, bounded noise `σ ≤ 0.2`, `λ ∈ {1,0.9,0.95,0.99}`.
- R3: `D ∈ {2,4}`, `N ≤ 1000`; uniform matched scaling to `s = 1e5`; disparity
  `≤ 10`; `ρ ≤ 0.99`.
- R4 committed: `D ≤ 16`, `N ≤ 1000` (Track A `N = 64·D`), realised `κ ≤ 1e3`,
  `λ ∈ {1,0.99,0.9}`, disparity `≤ 10`, dense/correlated designs.
- R4 release-only diagnostics: `D ≤ 64`, `N = 10⁴`, `λ = 0.999/0.9999`,
  `s = 1000` at `D = 16`.

**Known boundary / failure behaviour** (recorded, outside the well-conditioned
fully excited envelope):

- Feature-scale disparity around ratio `100` in tested `λ=1`, small-`D`
  configurations: the covariance residual exceeds the registered tolerance while
  predictions remain small, with **no guard**; ratio `1000` produced
  `InvalidDenominator` guards.
- Exactly unexcited direction under `λ < 1`: `P_ee` inflates `≈ 1/λ` per step and
  eventually overflows to `NonFiniteCandidate` at the step predicted from
  `δ` and `λ`.
- `λ = 1` over long `N`: `P → 0` and parameter error grows gradually (dynamic
  range), distinct from `λ < 1`, which plateaus after `≈ 1/(1−λ)`.
- Weak excitation: the peak `P` grows as the weak-axis scale falls
  (finite-horizon evidence only).

**Unresolved regions** (do not infer from the above):

- decision-grade performance behaviour of the persistent-excitation streaming
  workload;
- real non-stationary data, drift models, and delayed labels;
- longer horizons and larger `D` than the released diagnostics;
- the exact disparity boundary as a function of `D` at matched conditioning
  (the R4 Track-A κ-dial limitation);
- any representation decision.

**Theoretical/derived expectations** (not measured guarantees):

- `κ(A) ≈ (s_max/s_min)²` for scale disparity; direct-solve forward error
  `≈ κ·eps` (RLS is a recursion, not a direct solve);
- `λ<1` data weight saturates (`Σλ^j → 1/(1−λ)`), so error should plateau;
- `λ=1` is the pooled objective and permutation-invariant;
- `P`, the denominator, and the guard behaviour are **noise-independent**;
- `v_i v_j == v_j v_i` in IEEE `f32`, hence exact symmetry.

### `f32` versus `f64`

The historical R1–R4 evidence remains bounded legacy **C** evidence generated by
the strict-`f32` recurrence; its measured envelopes, tolerances, allocation
counts, and timings are not transferred to A or B. C remains native `f32`
through configuration, state, updates, admission, and inference. Test-side
`f64` oracles and metrics are outside that production path.

A is a separate native-`f64` learner, not a widened C trajectory and not a
precision selector. The native-f64 discriminator test uses
`(x=1,y=0)` then `(x=1+2⁻²⁴,y=2²⁴)` with `D=λ=δ=1`: A produces
`w≈5592405.444444`, while quantized C produces `5592405.5`; the gap is greater
than `0.05`. B may round to the same serving coefficient while its source `w/P`
remain distinct and authoritative.

No runtime precision switch, configurable scalar framework, automatic
normalisation, preconditioning, or universal accuracy claim is added. Persistence
and `AllowLoss` remain deferred. See
[Major deferred decisions](#major-deferred-decisions).

### User and data responsibilities

Seqvex RLS cannot guarantee and the caller/data pipeline must supply:

- persistent excitation of the directions to be estimated;
- exogeneity (or an explicit, acknowledged bias);
- stationarity for `λ=1` and a drift rate compatible with `λ<1`;
- a feature-scale policy, including any normalisation;
- a suitable `λ` and `δ`, including the bias/variance trade-off;
- temporal integrity (ordering, causal labels, no leakage);
- handling of invalid or missing observations, and any retry/reset policy;
- uncertainty quantification — RLS exposes none;
- noise/variance expectations — only monotone directions are established.

### Production readiness and the optimization gate

This contract does **not** declare RLS production-ready. Remaining gates:

- decision-grade performance profiling of a persistent-excitation streaming
  workload, tied to a defined workload/service level;
- architecture review of where an optimised `P` buffer would live;
- any future numerical-representation decision;
- final production API review;
- measured optimization only after the above.

Shared-model / multi-stream execution is no longer a gate: the executor now
borrows the model immutably (`&Model`).

**OPTIMIZATION REMAINS CLOSED.** `O(D²)` complexity, the existing allocations,
the existing benchmark, known `f32` boundaries, or a theoretical SIMD benefit are
**not** sufficient reasons to optimize. Optimization requires, in order:
(1) the correctness gate, (2) the statistical-applicability gate, (3) the
architecture checkpoint, (4) decision-grade profiling, (5) an identified
bottleneck, and (6) a measured benefit. No exception.

## Micro-batch

RLS updates are ordered and state-dependent, so a bounded micro-batch must be an
ordered sequential fold — independent or parallel updates are invalid.

`Rls::process_batch_reference` is the RLS-local bounded reference batching
contract. The caller supplies `max_batch`; an oversized batch is rejected
before any transition and the supplied `State` is returned unchanged and
unvalidated; an empty batch returns the supplied `State` unchanged;
observations are folded strictly in order with `Rls::update`, stopping at the
first failure and returning `RlsBatchError::Transition` with the failed index,
the underlying `RlsError`, and the last committed `(w, P)`. There is no
independent or parallel update and no whole-batch rollback.

The foundation `process_batch` is the **unbounded** reference ordered fold
(`src/foundation/state`); it is not a bounded micro-batch executor and is not
the RLS bounded path.

## Measured evidence (#26)

Two separate runs are recorded here and are not merged: the historical
fixed-observation foundation-fold baseline ([historical]) and the release
batch-size sweep of the RLS-local bounded reference API ([new run], revision
`b63da4f`). Historical values below are unchanged.

### Historical baseline ([historical])

`cargo bench --bench rls`, release, median of 20 runs. The benchmark uses `λ = 1`
because it replays a single fixed observation vector.

With `λ < 1`, older observations receive geometrically less weight (`λ^{t−i}`),
and the covariance can grow in directions the input never excites: the
information matrix `A_t = λ A_{t−1} + x_t x_tᵀ` decays by `λ` per step in any
unexcited direction, so `P_t = A_t^{−1}` grows without bound there. A repeated or
low-excitation input therefore produces covariance inflation. In finite precision
this ill-conditioning eventually yields non-finite values (overflow), at which
point the implementation's denominator/candidate finiteness guards reject the
update. Those guards are implementation-level numerical protections, not a
statement that `λ < 1` is invalid. The update path and its `O(D²)` work are
unchanged; the correctness tests cover `λ < 1` over varying, exciting inputs.

This historical table keeps the benchmark's original `bounded-fold` label for
the foundation **unbounded** ordered fold (`src/foundation/state::process_batch`,
consuming owned observations). It is **not** the RLS-local bounded reference API
`Rls::process_batch_reference`, which is measured separately below; historical
values are unchanged and are not attributed to the new API.

| features | path | ns/obs | allocs/obs | bytes/obs |
|---:|---|---:|---:|---:|
| 8 | reference | 231.9 | 5.000 | 384 |
| 8 | streaming | 227.4 | 5.000 | 384 |
| 8 | bounded-fold | 240.3 | 6.000 | 416 |
| 32 | reference | 2255.8 | 5.000 | 4608 |
| 32 | streaming | 2268.4 | 5.000 | 4608 |
| 32 | bounded-fold | 2269.1 | 6.000 | 4736 |
| 128 | reference | 34952.7 | 5.000 | 67584 |
| 128 | streaming | 34622.8 | 5.000 | 67584 |
| 128 | bounded-fold | 34836.9 | 6.000 | 68096 |
| 256 | reference | 139636.4 | 5.000 | 266240 |
| 256 | streaming | 139734.8 | 5.000 | 266240 |
| 256 | bounded-fold | 141066.1 | 6.000 | 267264 |

Findings, stated without overclaiming:

- The reference is **not** allocation-free, unlike linear regression. Every
  observation returns a new `D×D` candidate `P'`, so the value-returning
  `StateModel` contract costs a steady **5 allocations/observation** dominated by
  `P'`; bytes/observation scale with `D²` (`384` at `D = 8`, `266240` at
  `D = 256`). Committed state is `D + D²` scalars (`288` bytes at `D = 8`,
  `263168` bytes at `D = 256`).
- Streaming is **not faster** than the direct reference (differences are within
  run noise); it is the same generic `update` path driven through the executor.
  Lower latency was never claimed for the streaming wrapper.
- The foundation (unbounded) ordered fold is **not faster** either, and costs
  one extra allocation/observation because it consumes owned observations (the
  batch is cloned to feed it). These historical numbers describe the foundation
  fold; the bounded RLS micro-batch is `Rls::process_batch_reference`, retained
  as a semantically correct capability rather than a performance claim.
- Latency scales roughly with `D²` (`232 ns` at `D = 8` to `140 µs` at
  `D = 256`), consistent with the covariance update dominating.

The measured `O(D²)` candidate allocation is the RLS-specific evidence. A future
double-buffered `P` could remove it, but **where the second buffer lives**
(model vs stream vs external) is the unresolved ownership question; a workspace
is deliberately not introduced here. Any such optimization requires CRITICAL
ARCHITECTURE REVIEW.

The 10,000-observation test checks weight and covariance finiteness after every
update. The scalar reference advances at every iteration. After the loop, the
final state is compared bitwise with that reference, and the final covariance is
checked for exact symmetry.

### Release batch-size sweep ([new run], recorded run-time revision `b63da4f`)

Release run `cargo bench --bench rls -j 2` at the recorded run-time revision
`b63da4f056c24b3475daf1adaf8ea1655060e827` (branch `rust-development`, clean
working tree), `env os=linux arch=x86_64 profile=release parallelism=20`, 20
normalized measurement windows per cell, single machine (12th Gen Intel Core
i7-12700H, 20 logical CPUs), rustc/cargo `1.97.1`, Linux
`6.18.40.1-microsoft-standard-WSL2`. The run-time revision, toolchain, and
working-tree status were captured in the run metadata (from `git rev-parse HEAD`
and the same-session `git status`), not inferred. `SEQVEX_EVIDENCE_ONLY` was
unset, so the full benchmark executed.

Full-horizon numerical validation passed **before** any timing: `D = 8`
obs `209920` (102 wrap crossings) and `D = 32` obs `168960` (82 wrap crossings).
The release validator aborts on the first failed update or non-finite state;
there was no reset, skip, retry, or shortened horizon. Matrix: `λ = 0.999`,
`D ∈ {8, 32}`, `B ∈ {1, 8, 32, 128}`, `outer = 128`
(`work_per_call = OUTER = 128`). `grouped` is
`StreamingExecutor::process_one`; `bounded-ref` is `Rls::process_batch_reference`;
`foundation-fold` is the foundation `process_batch` cloning control, labelled
separately.

Fixture limitation: one deterministic persistent-excitation observation is
replayed cyclically (period `2048`), so this is a fixed-target cyclic-replay
grouping-cost experiment, not input-distribution performance and not a claim of
statistical applicability. `ns/obs`, `p95`, `IQR`, `obs/s`, `allocs/obs`, and
`bytes/obs` are the harness's measured values; `p95`/`IQR` describe variability
across the normalized measurement windows, **not** individual-call or event
latency; `bytes/obs` is allocator allocation/reallocation traffic, **not**
live/peak memory. `derived ns/batch = median × B` is **harness-derived**.
Committed payload is the **calculated** `(D + D²) × 4` bytes.

**D = 8** (`outer_invocations/run` = 32, `timed_obs` = 4096):

| path | B | ns/obs | p95 | IQR | obs/s | allocs/obs | bytes/obs | derived ns/batch |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| grouped | – | 226.2 | 233.1 | 4.4 | 4420739 | 5.000 | 384.0 | – |
| bounded-ref | 1 | 228.4 | 242.3 | 3.6 | 4379147 | 5.000 | 384.0 | 228.4 |
| bounded-ref | 8 | 230.3 | 236.0 | 4.2 | 4341853 | 5.000 | 384.0 | 1842.5 |
| bounded-ref | 32 | 226.9 | 233.5 | 8.0 | 4407087 | 5.000 | 384.0 | 7261.0 |
| bounded-ref | 128 | 227.3 | 234.1 | 1.4 | 4399396 | 5.000 | 384.0 | 29094.9 |
| foundation-fold | 1 | 227.6 | 246.8 | 8.1 | 4393525 | 6.000 | 416.0 | 227.6 |
| foundation-fold | 8 | 238.5 | 368.3 | 12.9 | 4193057 | 6.000 | 416.0 | 1907.9 |
| foundation-fold | 32 | 239.1 | 253.1 | 8.3 | 4182003 | 6.000 | 416.0 | 7651.8 |
| foundation-fold | 128 | 247.0 | 252.7 | 5.6 | 4048371 | 6.000 | 416.0 | 31617.7 |

**D = 32** (`outer_invocations/run` = 16, `timed_obs` = 2048):

| path | B | ns/obs | p95 | IQR | obs/s | allocs/obs | bytes/obs | derived ns/batch |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| grouped | – | 2330.0 | 2399.2 | 66.8 | 429181 | 5.000 | 4608.0 | – |
| bounded-ref | 1 | 2197.7 | 2273.1 | 89.2 | 455019 | 5.000 | 4608.0 | 2197.7 |
| bounded-ref | 8 | 2252.8 | 2279.0 | 25.4 | 443897 | 5.000 | 4608.0 | 18022.2 |
| bounded-ref | 32 | 2244.2 | 2285.0 | 39.0 | 445597 | 5.000 | 4608.0 | 71813.7 |
| bounded-ref | 128 | 2296.5 | 2404.9 | 77.0 | 435439 | 5.000 | 4608.0 | 293956.2 |
| foundation-fold | 1 | 2275.3 | 2882.4 | 212.3 | 439506 | 6.000 | 4736.0 | 2275.3 |
| foundation-fold | 8 | 2275.2 | 2298.4 | 19.8 | 439515 | 6.000 | 4736.0 | 18201.9 |
| foundation-fold | 32 | 2282.6 | 2366.2 | 49.3 | 438094 | 6.000 | 4736.0 | 73043.7 |
| foundation-fold | 128 | 2184.2 | 2281.6 | 101.4 | 457840 | 6.000 | 4736.0 | 279573.7 |

Applying the harness materiality rule (`difference > 2·max(IQR)` AND
`difference > 5%·max(median)` → clearly measurable; exactly one → borderline/noisy;
neither → not materially different):

- Bounded-reference per-observation cost does not differ materially across
  `B ∈ {1, 8, 32, 128}` at either dimension. Grouping the bounded reference does
  not change `allocs/obs` (`5.000`) or `bytes/obs` (`384.0` at `D = 8`, `4608.0`
  at `D = 32`).
- Grouped streaming and the bounded reference do not differ materially, except
  `D = 32, B = 1`, which is borderline/noisy (2330.0 vs 2197.7 ns/obs).
- The foundation-fold cloning control meets the materiality rule only at
  `D = 8, B = 128` (247.0 vs 227.3 ns/obs; clearly measurable). `D = 8, B = 32`
  is borderline/noisy (difference `12.2` ns/obs against `2·max(IQR)` `16.6` and
  `5%·max(median)` `11.96`). At `D = 32, B = 128` the difference is `112.3`
  ns/obs against `2·max(IQR)` `202.8` and `5%·max(median)` ≈ `114.8`: neither
  threshold is exceeded, so it is **not materially different** under the rule in
  this capture; this does not establish equivalence or absence of a useful
  improvement. The foundation-fold control records `6.000` allocations/obs
  versus `5.000` for the bounded reference, with additional allocator-counted
  traffic of `32` bytes/obs at `D = 8` and `128` bytes/obs at `D = 32`.
- No clearly measurable grouping benefit is observed in this capture.

Limitations: fixed-target cyclic replay (period `2048`); `λ = 0.999`; single
machine; one capture with 20 windows and no independent repeat; background
editor/service processes were present (load average ≈ `1` before the run, and
the benchmark itself drove load to ≈ `4`). The measured numerical envelope is
`D ∈ {8, 32}` under the `λ = 0.999` persistent-excitation design; the
full-horizon validator traversed the complete intended trajectory with no guard
failure, but larger `D` remain subject to the finite-precision `λ < 1`
covariance-inflation envelope described above (`D = 128` is deferred). These
comparative outcomes are single-run **directional** evidence, not
acceptance-grade comparative performance conclusions.

The same run re-executed the unchanged `λ = 1` control, the decision-grade A/B/C
workloads, and the Category B evidence decomposition; those are not the subject
of this record.

The reference path is retained; optimization and profiling are deferred. This
run does not establish a CPU profile, a bottleneck identification, a universal
speedup, or production readiness, and it does not alter the RLS
production-readiness gates.

### Precision-profile quality evidence (`RLS-QUALITY800`)

Candidate working-tree run based on `e027eb13454bf7d429184f37d1581f8c2da1cfbd`
(uncommitted implementation), verified with default Rust/Cargo `1.97.1` and
MSRV `1.85.1`, via `cargo run --locked --example rls_profiles -j 1`. The
fixture uses `D=2`, `λ=1`, `δ=1000`, exactly 400 ordered training updates,
one frozen A state, one
one-time B conversion, and exactly 400 checked scored rows per profile over
rows 400–799 with no refit or scoring update.

| Profile | Training updates | Scored calls | MAE | R² |
|---|---:|---:|---:|---:|
| A native `f64` | 400 | 400 | `4.518996794317e-6` | `0.999999999973` |
| B `f64` learn / `f32` serve | 400 (A) | 400 | `4.525293214072e-6` | `0.999999999973` |
| C strict `f32` | 400 | 400 | `5.498211830854e-6` | `0.999999999961` |

Declared limits were A `MAE≤1e-5`, `R²≥1−1e−9`; B/C `MAE≤1e-4`,
`R²≥1−1e−6`. These are bounded fixture results, not universal accuracy or
production-readiness claims. The prior `1/δ=1e-3` dominates the small A error;
B additionally includes feature and weight serving quantization, and C includes
native `f32` generation and recurrence.

## Major deferred decisions

- Shared-model / multi-stream execution (the former `&mut M` question) is now
  supported by the immutable-borrow executor; where an optimized `P`
  double-buffer would live, if optimization is ever justified, remains deferred
  and requires a workspace decision.
- Persistence/checkpoint identity and round-trip across profiles remain
  deferred; no serialized learner or serving format is introduced.
- `UnderflowPolicy`/`AllowLoss` for B conversion remains deferred. A nonzero
  coefficient that rounds to zero is rejected in this package.
- Full PSD certification, covariance repair, conditioning diagnostics,
  variable/adaptive forgetting, ridge extensions, and UD/square-root forms
  remain deferred. Admission is structural/finite/symmetric only.
- Numerical performance envelopes for A/B, optimized covariance buffers, and
  any production precision selection remain open. C's historical `f32` evidence
  is preserved as bounded legacy evidence.

## Examples

Learning uses `Rls::update` on one supervised observation at a time.
`Rls::process_batch_reference` groups ordered updates and returns the final
state. `StreamingExecutor::process_one` manages learning-state transitions;
checked prediction remains an explicit caller operation before the current
target is consumed.

- [`examples/rls_learn.rs`](examples/rls_learn.rs) — deterministic,
  self-contained synthetic regime-switch demonstration
  (`cargo run --example rls_learn`). It learns a prefix through public
  `update` calls and scores later rows with `predict_checked`; no external
  files or network.
- [`examples/rls_trump_approval.rs`](examples/rls_trump_approval.rs) —
  real-data validation on a caller-supplied CSV
  (`cargo run --example rls_trump_approval --all-features -- <csv>`). Its `f64`
  standardizer is mixed-precision legacy preprocessing, **not strict C**.
- [`examples/rls_profiles.rs`](examples/rls_profiles.rs) —
  deterministic in-memory `RLS-QUALITY800` for A/B/C
  (`cargo run --example rls_profiles`). It performs exactly 400 ordered
  training updates per learner, freezes the states, converts A once into B's
  independent snapshot, and makes exactly 400 checked predictions per profile
  over rows 400–799 with no refit or scoring update. It also embeds caller
  collect/accept failure tests.

### Real-data example (`rls_trump_approval`)

Reads a caller-supplied CSV offline; dataset acquisition is a separate manual
step. Contract:

- Requires exactly seven CSV columns, parsed by header name (order-independent):
  `ordinal_date`, `five_thirty_eight`, `gallup`, `ipsos`, `morning_consult`,
  `rasmussen`, `you_gov`. Missing, duplicate, and unexpected headers, missing
  fields, malformed values, and non-finite numeric values are rejected.
- Model inputs are the five polling features plus a generated intercept:
  `[gallup, ipsos, morning_consult, rasmussen, you_gov, 1.0]`. The target is
  `five_thirty_eight`. `ordinal_date` is metadata used for chronology and
  labels and is excluded from the model features.
- First 400 observations are the learning prefix; remaining rows are scored
  prequentially. Polling features are standardized with a **population**
  standard deviation computed from the first 400 rows only
  (`mean = Σx/N`, `variance = Σ(x−mean)²/N`, `scale = sqrt(variance)`).
  The transformation is frozen for the whole run. Scaling arithmetic is
  `f64`; model features are `f32`, with a finiteness re-check after conversion.
  Zero or non-finite scales and non-finite standardized values are rejected
  as preprocessing/conversion failures before submitting the affected feature
  vector to RLS.
- Configuration: `delta = 1000`, lambda cases `1.0` and `0.95`, each starting
  from its own `initial_state()`. For each scoring row the example calls
  `predict_checked` on the previously committed state, records the prediction,
  then commits the next state only after a successful `update`.
- Any parse, preprocessing, prediction, or update failure stops the run with
  row/date/call-kind context and a non-zero exit. There is no silent recovery,
  skipping, retry, or reset.
- Automated example tests use synthetic temporary CSVs only and never read the
  real dataset. Real-data verification requires explicitly running the example
  against the supplied CSV.

Execution is offline. Separately download River's
[pinned compressed CSV](https://raw.githubusercontent.com/online-ml/river/6af83abba269dbdaf38e9c0eaf61d82c451ad698/river/datasets/trump_approval.csv.gz)
and decompress it as `trump_approval.csv`.

The evaluated decompressed snapshot contains 1001 data rows and has SHA-256
`aef188d04b606270e9590013282e2ede3f5f92edae1106d85d3defa483189cb0`.
The example checks header names, finiteness, strict chronology, and minimum row
count; it does not require that checksum or an exact 1001-row input.

### Recorded run on the supplied snapshot

On the observed 1001-row snapshot (400 learning rows, 601 scoring rows), over the
identical scoring rows:

| Predictor | MAE | RMSE |
|---|---|---|
| RLS `λ = 1.0` | 0.553289 | 0.687858 |
| RLS `λ = 0.95` | 0.286406 | 0.378739 |
| Expanding target mean | 1.496969 | 1.659610 |
| Previous target | 0.133948 | 0.208678 |
| Raw polling mean | 0.588404 | 0.731256 |

These reported evaluation results are bounded to that snapshot and protocol,
not general guarantees. In this evaluation, previous-target prediction
performed best. This reconstructs an aggregate approval estimate from polling
features; it is not political forecasting and not a reproduction of a River
model benchmark. Existing production-readiness and optimization gates remain
unchanged.

## Current tests / specification

- [`tests/rls.rs`](tests/rls.rs): integration target `rls`. The
  `precision_profiles` module adds A/B/C API/visibility, admission precedence,
  independent same-order scalar controls, literal weighted-objective checks for
  every nonempty named prefix, `RLS-ORDER4`, both orders of `RLS-PERM30-A/C`,
  B conversion/serving bounds, failure/prefix/continuation, native-f64
  discrimination, conversion boundaries, and caller collector/handover tests.
- [`profiles.rs`](profiles.rs): module-local private-state tests cover
  `SourceNonFinite`, malformed shapes, dimension-before-NaN precedence, and
  unchanged source/active bytes. They do not expose an invalid-state constructor.
- [`examples/rls_profiles.rs`](examples/rls_profiles.rs): embedded
  `RLS-QUALITY800` and caller-protocol tests.
- [`examples/rls_trump_approval.rs`](examples/rls_trump_approval.rs): embedded
  example tests using synthetic temporary CSVs only.

Run the embedded tests explicitly with
`cargo test --locked --example rls_profiles --all-features` and
`cargo test --locked --example rls_trump_approval --all-features`.
Automated tests do not require the local real dataset.
