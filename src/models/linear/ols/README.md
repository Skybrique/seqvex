# Linear Regression — ordinary least squares

This is the canonical OLS algorithm document. It describes user behavior, mathematical requirements, supported boundaries, issue ownership and evidence. **Current status: the implemented OLS baseline is accepted.** Dense unweighted fitting, rank/conversion checks, diagnostics, artifact v1, checked serving and deliberate refit handover are implemented and independently reviewed. This acceptance concerns the declared baseline; it does not establish suitability for every production workload. The additive A/B/C precision profiles described in §16 are implemented on top of that baseline and are awaiting independent review; they do not change the accepted legacy behavior.

The initial scope is dense, unweighted ordinary least squares with one scalar numeric response per observation, any supported number of input features and an optional intercept. Serving uses an immutable predictor, with streaming first and caller-bounded micro-batching as the additional execution mode. Periodic refitting and deliberate model acceptance are distinct from incremental learning. Ridge, Lasso, logistic classification with declared regularization support and RLS retain their own contracts. **([#23](https://github.com/Skybrique/seqvex/issues/23))**

This README does not replace the Architect technical design or the Planner implementation plan. Those are separate, issue-linked deliverables reviewed before the Coder starts the affected implementation. References to issues identify requirement owners, not completion.

**Explicitly deferred:** caller-side preparation/partition helpers (`PreparedData`/`WorkflowKind`, T10a), ordering/shuffle facilities (T10b), and formal cross-precision numerical-envelope certification. These do not block the accepted baseline. OLS performs no implicit shuffle; the prohibition on shuffling ordered workflows is retained. Requirement and design-question lists below preserve their original planning context alongside current-status clarifications; they do not reopen the deferred work.

| Requirement owner | Responsibility |
|---|---|
| [#23](https://github.com/Skybrique/seqvex/issues/23) | Scope, accepted design decisions, architecture boundaries, overall acceptance and integration |
| [#43](https://github.com/Skybrique/seqvex/issues/43) | Fitting/lifecycle, API, diagnostics, artifacts/recovery, local code/tests, rustdoc and runnable teaching examples |
| [#44](https://github.com/Skybrique/seqvex/issues/44) | Independent mathematics, rank/conditioning, precision/conversion and numerical validation |
| [#45](https://github.com/Skybrique/seqvex/issues/45) | Availability/causality, sequential/non-IID validation and regime-shift evidence |
| [#46](https://github.com/Skybrique/seqvex/issues/46) | Latency, throughput, allocations/allocator bytes, memory/scaling/tails, profiling and before/after audits |

Documentation is a completion requirement within each workstream. A separate documentation child is not required.

## 1. What this algorithm provides

OLS learns a linear predictor from a declared training dataset. For a new feature vector, the fitted predictor produces one numeric estimate.

**Illustrative quant use:** fit a relationship between features known at an observation's origin and a specified future return, then stream predictions using fixed coefficients until an explicitly accepted refit replaces them. The application defines the features, target horizon and business acceptance; this example is not a profitability claim.

| Capability | Current capability | Requirement / owner |
|---|---|---|
| Existing supplied-parameter prediction primitive | Implemented and preserved; fitting now supplies learned parameters | Preserved alongside the additive fitter. ([#43](https://github.com/Skybrique/seqvex/issues/43)) |
| Streaming and caller-bounded grouped prediction | Implemented for fitted models; opt-in checked paths reject non-finite outputs | Demonstrated by checked-serving tests and the Salary example. ([#43](https://github.com/Skybrique/seqvex/issues/43)) |
| Fit weights/intercept from labelled rows | Implemented: `fit`, multi-feature scalar response, optional intercept; faer column-pivoted QR, `Par::Seq` | Implemented in `fit.rs`; `tests/fitting.rs`. ([#43](https://github.com/Skybrique/seqvex/issues/43)) |
| Rank/numerical diagnostics | Implemented: relative-pivot rank indicator and rounded-f64/f32 RSS diagnostics; formal certification deferred | Independent references, fitter tests and cross-solver comparison. ([#44](https://github.com/Skybrique/seqvex/issues/44)) |
| Causal periodic refitting | Explicit training indices supported; temporal tests construct eligible/purged windows; automatic eligibility and transforms remain caller-owned | Selection isolation and regime-window tests; no automatic leakage detection. ([#45](https://github.com/Skybrique/seqvex/issues/45)) |
| Artifact import/export and replacement boundary | Implemented: artifact v1, metadata/content identity, rejection and handover through a new executor | `artifact.rs`, artifact tests and Salary refit demonstration. ([#43](https://github.com/Skybrique/seqvex/issues/43)) |

“Complete OLS” here includes the necessary fitting, prediction, failure, validation and documentation components. It does not include a generic trainer, model registry, market-data service or trading engine. Sparse, weighted and multi-output fitting are excluded unless explicitly added to scope. **([#23](https://github.com/Skybrique/seqvex/issues/23))**

## 2. Fitting and prediction user flow

The existing constructor accepts parameters and does not learn them. The implemented `fit` API learns parameters from selected labelled rows. Preset-coefficient demonstrations alone do not satisfy the mandatory [training-to-inference contract](../../../../docs/DEVELOPMENT.md#model-training-and-valid-inference--mandatory).

An accepted OLS demonstration must ingest labelled training data, execute the reviewed fitting procedure, validate the learned coefficients/diagnostics, and calculate predictions from that fit. The registered `fit_salary` example now performs fit → diagnostics → checked single/stream/group prediction → held-out RMSE against a training-mean baseline → artifact round-trip → refit handover → failed-candidate preservation. **([#43](https://github.com/Skybrique/seqvex/issues/43), supporting #44–#45)**

The canonical entry point is `seqvex::models::linear::ols::fit`:

```rust
fit(features: &[Vector], targets: &[f32], train: &[usize], config: &FitConfig)
    -> Result<FitResult, FitError>
```

Full feature/target array lengths must agree; fitting reads values only at the explicit training indices. `FitResult::into_predictor(self) -> LinearRegression` moves learned parameters into the predictor and is infallible. `FitResult::into_trained(self, &TrainingMeta) -> Result<TrainedModel, ArtifactError>` adds caller-supplied training linkage and derived artifact metadata.

### Candidate acceptance and installation

`fit` success means the solver produced a candidate; it does not by itself admit or install that candidate. A caller may check an explicit nonempty selection before conversion:

```rust,ignore
candidate.check_acceptance(&features, &targets, &acceptance_indices)?;
```

`check_acceptance` borrows the candidate and rows, counts repeated indices in supplied order, inspects selected rows only, checks the stored `rss_f64_reference` and `rss_f32_predictor`, and reports the sequential f64 RSS of checked deployed-f32 predictions. `fit` has already enforced its rank policy; this method does not revalidate rank. A finite report is not a predictive-quality threshold or a guarantee about unselected/future rows.

After admission, the caller may convert/build the artifact and verify a candidate executor, then explicitly install the candidate. Admission does not mutate or publish an active model. The existing `fit_salary` refit example demonstrates candidate executor verification while retaining the active model; installation remains an application-owned action. Bare `predict` retains its existing non-finite-output behavior; output rejection remains opt-in through checked prediction paths.

Run the example with a caller-supplied CSV path (positional, not `--input`):

```sh
cargo +1.85.1 run --locked --example fit_salary -- examples/Salary_dataset.csv
```

**Salary evaluation boundary:** the example holds out zero-based CSV positions where `position % 5 == 4`; it does not shuffle or use an RNG. The 30-row sample yields 24 training and 6 held-out rows. This interleaved partition is an end-to-end demonstration, not chronological evaluation or production-readiness evidence. CSV row position and years of experience do not establish observation time. An empty held-out partition is rejected before fitting/indexing/RMSE. The synthetic benchmark's fixed-seed LCG is separate from this example and from the fitter.

`weights()` and `bias()` expose the supplied parameters; they do not establish a durable artifact format. Both historical imports, `seqvex::models::classic::{LinearRegression, RegressionError}` and `seqvex::models::classic::linear_regression::{LinearRegression, RegressionError}`, remain compatibility re-exports of the canonical types. User-facing examples and rustdoc belong to [#43](https://github.com/Skybrique/seqvex/issues/43).

The model requires nonempty, finite weights and a finite bias. `predict` checks feature dimension before feature finiteness. The bare prediction errors remain `ZeroDimension`, `NonFiniteParameter`, `DimensionMismatch { expected, actual }` and `NonFiniteInput`. Opt-in checked serving additionally reports `NonFiniteOutput { observation: Option<usize> }`; grouped checked serving reports the zero-based position of the first non-finite output. The additive public enum variant requires a new arm in downstream exhaustive matches. The scalar response does not restrict the number of input features. **([#43](https://github.com/Skybrique/seqvex/issues/43))**

- Define fitting inputs as \(N\) rows of \(d\) features and \(N\) numeric targets, with explicit ownership and shape validation. **([#43](https://github.com/Skybrique/seqvex/issues/43))**
- Preserve the implemented `fit`/`FitResult` path to an immutable predictor; further public API changes require the applicable design review. **([#43](https://github.com/Skybrique/seqvex/issues/43))**
- Supply a working example covering fit → inspect → validate → predict → export/import when needed; document inputs, outputs and errors beside the actual API. **([#43](https://github.com/Skybrique/seqvex/issues/43))**
- Keep existing imports compatible during any local reorganization, or obtain explicit approval for a compatibility change. **([#43](https://github.com/Skybrique/seqvex/issues/43))**

## 3. OLS mathematics and numerical envelope

**Implemented numerical policy:** finite `f32` training values are promoted exactly to `f64`; faer `=0.24.4` performs column-pivoted QR factorization/solve in `f64` with sequential execution. Promotion preserves the supplied values; it does not recover precision previously lost in their representation. Higher fitting precision reduces additional rounding during the solve. Coefficients/intercept are checked and stored as `f32`, and serving uses the existing ordered `f32` arithmetic.

Rank is assessed before solve using `tau = c_rank * max(N, p) * 2^-53 * abs(R[0,0])`, with default `c_rank = 1` and `p = d + 1` when an intercept is fitted. Insufficient rows or numerical rank are rejected; no minimum-norm or silently regularized solution is substituted. This relative-pivot policy is an indicator, not a condition-number estimate. Conversion rejects overflow and, by default, nonzero coefficients/bias that become zero; `UnderflowPolicy::AllowLoss` explicitly permits that loss. Representable subnormals are retained.

`FitDiagnostics` reports rank/threshold, selected-row counts and rounded-`f64` reference RSS versus actual `f32` predictor RSS. The exact-real cross-precision envelope and its acceptance checks are **deferred and not implemented**. Independent predictions/refits do not feed previous prediction errors back into OLS; rounding can still accumulate within a calculation. Neither checked serving nor the reviewed datasets establish a general finite-output accuracy guarantee.

For \(X\in\mathbb R^{N\times d}\), targets \(y\in\mathbb R^N\), weights \(w\in\mathbb R^d\) and an enabled intercept \(b\),

\[
\min_{w,b} \sum_{i=1}^{N}(y_i-b-x_i^\top w)^2,
\qquad
\hat y_i=b+x_i^\top w.
\]

With no intercept, \(b=0\). Equivalently, append a column of ones to obtain \(A=[\mathbf 1\ X]\) and solve \(\min_\theta\|y-A\theta\|_2^2\). An enabled intercept therefore adds a coefficient and can make an existing constant feature redundant.

- Fix the exact objective, intercept handling, precision and supported rank envelope in the technical design. Unweighted OLS must not silently acquire a ridge penalty, feature dropping or sample weighting. **([#23](https://github.com/Skybrique/seqvex/issues/23))**
- Compare a rank-aware QR approach and SVD-based least squares for stability, diagnostics, dependency/resource cost and reversal cost. Choose one justified baseline; explicitly forming and inverting \(X^\top X\) is not the default proposal. **([#23](https://github.com/Skybrique/seqvex/issues/23))**
- Define whether rank-deficient/underdetermined inputs receive a documented solution policy or an explicit error. Publish the numerical rank threshold and its scale/precision interpretation. Do not describe a non-unique solution as uniquely identified coefficients. **([#23](https://github.com/Skybrique/seqvex/issues/23))**
- Implement and report the accepted rank/outcome policy, residual diagnostics and relevant numerical failures. A condition estimate is included only if the chosen method supports a meaningful estimate. **([#43](https://github.com/Skybrique/seqvex/issues/43))**
- Verify the selected solver and numerical boundary independently, including nearly dependent features and training-to-prediction conversion. **([#44](https://github.com/Skybrique/seqvex/issues/44))**

**Existing predictor boundary:** weights/features are `f32`; the dot product reduces in weight-index order. Finite operands can produce infinity/NaN through overflow. Bare `predict`/`predict_batch` preserve their existing behavior and do not check output finiteness; `predict_checked`, `predict_batch_checked` and the borrowing `CheckedLinearRegression` wrapper reject non-finite outputs without changing finite result bits. Checked serving does not certify finite accuracy or detect every underflow.

- Preserve required bitwise equivalence across existing prediction paths. Fitting comparisons use separately justified, conditioning-aware tolerances. **([#44](https://github.com/Skybrique/seqvex/issues/44))**
- If fitting uses another precision, specify conversion and validate the resulting `f32` parameters/predictions. A stricter workflow-level finite-result policy must be explicit; it must not silently change the old predictor API. **([#43](https://github.com/Skybrique/seqvex/issues/43))**

## 4. Data assumptions and non-IID suitability

**Implemented boundary:** the fitter honors explicit training indices and does not infer timestamps, label availability, evaluation partitions or leakage. `tests/ols_temporal.rs` constructs chronological eligibility, interval-based purging, excluded-row isolation and distinct regime windows using synthetic data. These tests verify caller construction and selected-row behavior; they do not implement automatic preprocessing, splitting or production temporal evaluation.

Calculating an OLS solution does not require IID rows. It also does not establish stable predictive performance under drift or validate conventional uncertainty estimates. Confidence intervals and hypothesis-testing APIs are outside this delivery.

- Define feature schema/order, target meaning, label horizon and availability rules for each temporal example. The OLS solver need not itself own market timestamps; the workflow must nevertheless demonstrate eligibility correctly. **([#45](https://github.com/Skybrique/seqvex/issues/45))**
- Train only on rows whose features and labels are available at the declared cutoff. Fit preprocessing on eligible training rows and use the corresponding transformation at prediction time. **([#45](https://github.com/Skybrique/seqvex/issues/45))**
- Use chronological/walk-forward evaluation; account for overlapping label intervals when they create leakage. Purging/embargo parameters must follow the target construction, not an unexplained universal gap. **([#45](https://github.com/Skybrique/seqvex/issues/45))**
- Test correlated features, serially dependent observations, repeated rows, scale changes and abrupt/gradual regime changes. Separate numerical solvability from predictive usefulness. **([#45](https://github.com/Skybrique/seqvex/issues/45))**
- Keep algebraic row-permutation tests distinct from a production decision to shuffle observations. Mathematical permutation invariance does not authorize shuffled temporal evaluation or guarantee bitwise-identical fitted coefficients. **([#44](https://github.com/Skybrique/seqvex/issues/44))**

## 5. Fitting and refitting lifecycle

The OLS lifecycle is: validate an eligible training window → fit a candidate → inspect diagnostics → evaluate → explicitly accept/install. Serving predictions do not train the model.

- Validate empty inputs, zero features, mismatched row/target counts, inconsistent dimensions and non-finite features/targets before accepting a candidate. Declare validation order and errors. **([#43](https://github.com/Skybrique/seqvex/issues/43))**
- Define dataset retention, solver passes and fit-time capacity. Delivering rows in bounded chunks does not by itself bound total training memory or make this an incremental estimator. **([#43](https://github.com/Skybrique/seqvex/issues/43))**
- Treat periodic refitting as fitting a new candidate under a new declared cutoff. Do not mutate the active model while a candidate is being computed or checked. **([#43](https://github.com/Skybrique/seqvex/issues/43))**
- Demonstrate refitting with training-only transformations and an eligible window, keeping evaluation information out of fitting. **([#45](https://github.com/Skybrique/seqvex/issues/45))**
- Document what reset means: existing executor reset concerns last-prediction state; it does not erase/retrain coefficients. RLS-style observation-by-observation learning remains a separate algorithm. **([#43](https://github.com/Skybrique/seqvex/issues/43))**

Promotion schedules, durable storage and business acceptance are application-owned. Artifact v1 and failed-candidate preservation are implemented and tested. Handover constructs a new executor for the accepted candidate; no mutable replacement service is provided. **([#23](https://github.com/Skybrique/seqvex/issues/23))**

## 6. Streaming, micro-batching and composition

**Existing behavior:** `predict_batch` operates on the caller's supplied slice, returns outputs in input order, performs a sequential map and allocates an output vector. It has no internal queue, timeout, buffer-fill trigger or maximum-batch configuration. Empty input returns an empty vector; the first invalid observation returns an error without a partial output vector.

- Preserve one-observation prediction as the primitive and test fitted models through the streaming executor. State remains the last committed prediction, not learned coefficients. **([#43](https://github.com/Skybrique/seqvex/issues/43))**
- For a group of \(B\) observations, return \(B\) predictions in input order on success, with the existing single-prediction semantics. One grouped call still contains \(B\) observations. **([#43](https://github.com/Skybrique/seqvex/issues/43))**
- Verify repeated-single, streaming and grouped predictions against the reference, including empty groups, invalid observations and continuation after failure. **([#44](https://github.com/Skybrique/seqvex/issues/44))**
- Define feature transformation → predictor → evaluator composition and ensure the installed predictor uses the intended transformation/schema. Avoid a generalized pipeline framework until concrete requirements justify it. **([#43](https://github.com/Skybrique/seqvex/issues/43))**
- If training ingestion is chunked, document its retention/pass mechanics. It must not be presented as \(B\) online learning updates unless a different estimator is explicitly introduced. **([#43](https://github.com/Skybrique/seqvex/issues/43))**

The proposed user workflow prioritizes streaming and bounded micro-batching. Standing architecture also allows larger batches as a secondary capability; this draft does not remove that capability or change critical documentation. Any project-wide narrowing remains an explicit decision. **([#23](https://github.com/Skybrique/seqvex/issues/23))**

## 7. Memory ownership and algorithm locality

The existing predictor owns immutable weights/bias and needs no mutable prediction workspace. Solver input storage, fitting workspace and serving state are different ownership domains.

- Keep fitted parameters immutable during prediction; give mutable fitting resources a separate, declared owner. Do not introduce model-owned serving scratch merely to reuse fit-time memory. **([#43](https://github.com/Skybrique/seqvex/issues/43))**
- Declare input retention/copying, factorization workspace, output storage and temporary overlap between old/candidate models. Preallocate repeated inner-loop workspace where useful; do not claim zero allocations for all fitting or grouped prediction. **([#43](https://github.com/Skybrique/seqvex/issues/43))**
- Measure model/workspace footprint, fitting peak memory and serving/refit overlap separately from allocated-byte traffic. **([#46](https://github.com/Skybrique/seqvex/issues/46))**
- Perform necessary local moves with a before/after module/target map, compatible exports and discoverable tests/benchmarks. Family extraction requires concrete consumers and a demonstrated benefit. **([#43](https://github.com/Skybrique/seqvex/issues/43))**

**Locality responsibilities:** the OLS module, fitter, predictor, artifact implementation, numerical/temporal fixtures, fitting example and fitting/prediction benchmarks exist at the algorithm-local paths below.

| Level / path | Contents | Owning issue |
|---|---|---|
| `src/models/linear/ols/` | Algorithm files, `mod.rs`, README, examples and local unit/public-contract tests | [#43](https://github.com/Skybrique/seqvex/issues/43) |
| OLS-local test fixtures | Independent analytical/numerical oracles | [#44](https://github.com/Skybrique/seqvex/issues/44) |
| OLS-local workflow fixtures | Temporal eligibility and refitting tests | [#45](https://github.com/Skybrique/seqvex/issues/45) |
| OLS-local `benches/` | Component fitting/prediction/resource measurements | [#46](https://github.com/Skybrique/seqvex/issues/46) |
| Family module/tests/benches | Exports and justified cross-algorithm/shared behavior | [#23](https://github.com/Skybrique/seqvex/issues/23); separate ownership only if independently meaningful |
| Project `tests/`, `benches/` | Cross-component execution/recovery tests and integrated workflow measurements | Respective [#43](https://github.com/Skybrique/seqvex/issues/43), [#45](https://github.com/Skybrique/seqvex/issues/45) or [#46](https://github.com/Skybrique/seqvex/issues/46) owner |

Nested tests/benches need explicit module/Cargo registration or root wrappers. Verify discovery and execution; placement alone is not sufficient. Shared benchmark infrastructure remains shared, not a runtime dependency. **([#43](https://github.com/Skybrique/seqvex/issues/43))**

## 8. Failure, recovery and explicit assumptions

Use the current [`FAILURE_AND_RECOVERY.md`](../../../../docs/FAILURE_AND_RECOVERY.md) and [mandatory failure decision rule](../../../../docs/DEVELOPMENT.md#calculation-failures-and-error-handling--mandatory) as the standing references.

Bring any missing, ambiguous or contradictory failure policy to the maintainer before affected work proceeds. Report the operation, numerical/data boundary, committed progress and candidate responses; do not infer substitution, continuation, retry or reset. Existing prediction behavior is a source fact, not approval of the final model contract. **([#23](https://github.com/Skybrique/seqvex/issues/23), implementation #43)**

| Failure / boundary | Required OLS behavior | Owner |
|---|---|---|
| Invalid training shape/value | Explicit error; no usable candidate claimed | [#43](https://github.com/Skybrique/seqvex/issues/43) |
| Rank/conditioning outside support | Defined error or approved solution policy; no silent regularization | [#43](https://github.com/Skybrique/seqvex/issues/43) |
| Fit or candidate checks fail | Active predictor remains unchanged | [#43](https://github.com/Skybrique/seqvex/issues/43) |
| Import is malformed/incompatible | Reject before installation; preserve active model | [#43](https://github.com/Skybrique/seqvex/issues/43) |
| Model/transform pair mismatches | Reject or prevent incoherent publication under the approved schema | [#43](https://github.com/Skybrique/seqvex/issues/43) |
| Prediction/group fails | Preserve the existing committed-state/partial-output contract | [#43](https://github.com/Skybrique/seqvex/issues/43) |
| Finite operands yield non-finite output | Bare prediction preserves its existing behavior; opt-in checked serving returns `NonFiniteOutput`; no substitute prediction | [#43](https://github.com/Skybrique/seqvex/issues/43), supporting [#44](https://github.com/Skybrique/seqvex/issues/44) |
| Training row contains unavailable information | Exclude/reject it under the declared workflow policy and test that policy | [#45](https://github.com/Skybrique/seqvex/issues/45) |

- Define the minimum artifact: approved format/version, coefficients/intercept, dimensions, feature-order/schema information and necessary transformation/version linkage. Document encoding precision and validate before accepting it. No durable storage service is implied. **([#43](https://github.com/Skybrique/seqvex/issues/43))**
- Specify when a new model becomes active and which model/transform version serves in-flight observations. Do not invent a scheduler, lock-free publication mechanism or process-wide synchronization API to solve this before design review. **([#23](https://github.com/Skybrique/seqvex/issues/23))**
- Define retry/resume guarantees and committed progress. A retry must not accidentally install or process the same operation twice where that has observable effects. **([#43](https://github.com/Skybrique/seqvex/issues/43))**
- State allocation/capacity failure limits, including what can be returned as an error and what the runtime may terminate on. Do not claim that all OOM, panic or hardware failures are recoverable. **([#43](https://github.com/Skybrique/seqvex/issues/43))**
- Independently test the material numerical failure outcomes and report limitations of those checks. **([#44](https://github.com/Skybrique/seqvex/issues/44))**

**Artifact/resource limits:** artifact v1 preserves parameter bits and explicit intercept presence, validates content identity/checksum and import expectations, and rejects malformed/incompatible candidates before application installation. Publication is excluded from model identity. Import validates the implied payload against available bytes and `max_bytes` before dimension-driven coefficient reservation. `max_bytes` bounds artifact size, not total resident memory or OOM recovery; input/body buffers and the `Vector` copy still use infallible allocations. Export can leave a partial stream on writer failure and provides no filesystem atomicity/durability. The fitter's default 256 MiB requested-workspace budget is also not a physical-memory ceiling.

Assumptions: the application supplies trustworthy feature/label availability, owns storage/supervision and chooses promotion policy. Any contradiction with the standing failure contract is surfaced before affected implementation continues. **([#23](https://github.com/Skybrique/seqvex/issues/23))**

## 9. Independent mathematical and temporal verification

**Reviewed baseline evidence:** fitter (19 tests), independent oracle (24), artifact (21), checked serving (13), temporal selection (5), plus the existing prediction target (64 passed, one pre-existing ignored). The final integration report records passing Rust 1.85.1 debug/release checks, clean Clippy/formatting and the actual Salary run. These are scoped implementation checks, not a formal certification or universal production claim.

The separately reviewed external Linfa 0.7.1 comparison uses identical `f32`-quantized values, rows, feature order and intercept. Seqvex parameters and matched-order predictions equal the Linfa parameters rounded to `f32` **numerically**; the harness did not establish bit identity. Native Linfa `f64` predictions differ by at most about `2.9e-5` on the held-out comparison. Linfa's bundled diabetes loader returns 441 rows because `has_headers(true)` consumes the first numeric observation as a header; both models receive the same loader output. This regression dataset is separate from the unused binary-target `diabetes_prediction_dataset.csv`. The comparison is useful solver evidence on the reviewed cases, not a general error bound or temporal validation.

**Analytical reference examples:** the independent oracle target and fitter tests now exercise known-answer cases. Exact rational references remain separate from the production QR implementation.

- With intercept, \(x=(0,1,2)\), \(y=(1,3,5)\): \(w=2,\ b=1\), squared residual sum \(0\). **([#44](https://github.com/Skybrique/seqvex/issues/44))**
- With intercept, \(x=(0,1,2)\), \(y=(1,2,2)\): \(w=1/2,\ b=7/6\), squared residual sum \(1/6\). This tests a nonzero-residual fit rather than only exact synthetic lines. **([#44](https://github.com/Skybrique/seqvex/issues/44))**

For a nonconstant single feature, an independent scalar control can use
\[
w=\frac{\sum_i(x_i-\bar x)(y_i-\bar y)}
        {\sum_i(x_i-\bar x)^2},\qquad b=\bar y-w\bar x.
\]
The zero-denominator case requires the declared rank policy.

| Verification | OLS-specific evidence | Owner |
|---|---|---|
| General solution | Independent reference/oracle with matched objective, intercept, rank policy and precision; residual/optimality checks supplement it | [#44](https://github.com/Skybrique/seqvex/issues/44) |
| Rank/numerics | Duplicate columns, constant feature plus intercept, \(N\) below coefficient count, nearly dependent columns, scale extremes and conversion boundaries | [#44](https://github.com/Skybrique/seqvex/issues/44) |
| Prediction regression | Existing reduction order, public imports, dimension/finiteness checks and single/stream/group equivalence | [#44](https://github.com/Skybrique/seqvex/issues/44) |
| Temporal eligibility | Labels unavailable at cutoff excluded; transformations fitted on training only; overlap handled where applicable | [#45](https://github.com/Skybrique/seqvex/issues/45) |
| Regime behavior | Autocorrelated rows, abrupt/gradual coefficient changes, repeated rows and correlated features; comparisons/limitations reported | [#45](https://github.com/Skybrique/seqvex/issues/45) |
| Code/runtime behavior | API errors, artifact validation, failed candidate preservation and defined replacement/retry outcomes | [#43](https://github.com/Skybrique/seqvex/issues/43) |

- Report actual tested revisions, commands, results and tolerance rationale. Existing predictor tests and historical [#27](https://github.com/Skybrique/seqvex/issues/27) evidence do not validate the new fitter. **([#44](https://github.com/Skybrique/seqvex/issues/44))**
- Keep mathematical correctness, leakage prevention and predictive usefulness distinct; a passing oracle comparison cannot substitute for temporal workflow evidence. **([#45](https://github.com/Skybrique/seqvex/issues/45))**

## 10. OLS performance and optimization evidence

**Current measured baseline:** the registered `ols_fit` benchmark records fitting/resource evidence at the six `(N, d)` combinations below; checked serving uses `d = 8, 128` and groups `B = 1, 8, 32, 128`. Corrected allocation windows exclude sample-storage bookkeeping: fitting uses 11 allocations per operation, checked single/stream serving uses zero, and grouped serving allocates one `B * 4`-byte output vector. For the measured 128-feature artifact, export uses one allocation and import three. Construction timing is labelled fit plus `TrainedModel` construction; zero additional construction allocations are source-confirmed, not separately timed.

Requested workspace, allocator traffic, retained bytes and peak bytes are distinct. These are instrumented local synthetic measurements, not production tail-latency guarantees or a speed comparison with Linfa. Numerical rank and scaling are observed outcomes. The earlier workload proposals below are preserved; they do not imply every proposed dimension, conditioning case or distribution was measured.

**Proposed initial workload selection, subject to approved resource budgets:**

| Operation | Proposed scaling variables | Evidence needed | Owner |
|---|---|---|---|
| Fit | \(N=\{1{,}024,16{,}384\}\), \(d=\{8,32,128\}\); full-rank baseline plus representative conditioning cases | Fit latency, allocation/bytes, retained/peak workspace and scaling | [#46](https://github.com/Skybrique/seqvex/issues/46) |
| Streaming prediction | \(d=\{8,32,128\}\), fixed model | Warm latency distribution, throughput and allocation/bytes | [#46](https://github.com/Skybrique/seqvex/issues/46) |
| Grouped prediction | Same dimensions; \(B=\{1,8,32,128\}\) | Per-observation and per-call costs, output allocation, batch-size effect | [#46](https://github.com/Skybrique/seqvex/issues/46) |
| Import/refit handoff | Representative model/schema sizes | Cold-path cost and old/candidate overlap, separate from serving compute | [#46](https://github.com/Skybrique/seqvex/issues/46) |
| Integrated workflow | Transform → predict → evaluate; refit scenario where supported | Whole-path latency/resource cost and relevant completion behavior | [#46](https://github.com/Skybrique/seqvex/issues/46) |

These values are candidates, not a required blind Cartesian sweep or newly collected evidence.

- Agree representative workload, capacity and latency/resource acceptance before claiming production readiness. **([#23](https://github.com/Skybrique/seqvex/issues/23))**
- Record release distributions, throughput, allocations and allocated bytes, model/fit/serving memory, scaling, raw provenance and limitations. Separate cold fitting/import from steady-state serving. **([#46](https://github.com/Skybrique/seqvex/issues/46))**
- Normalize observation/group work explicitly; allocator traffic is not live/peak memory. Report request tails only from a suitable request-level sampling method, not from normalized-window p95/p99. **([#46](https://github.com/Skybrique/seqvex/issues/46))**
- Use correctness guards outside timing and retain revision/toolchain/hardware/resource facts and justified repeat methodology. Historical #23 prediction evidence is not a new fitting or workflow benchmark. **([#46](https://github.com/Skybrique/seqvex/issues/46))**
- Audit avoidable copying/allocation, fitting workspace reuse, prediction hot loops and independent multicore opportunities before proposing optimization. Check numerical order, ownership, capacities and future layout/device compatibility. **([#46](https://github.com/Skybrique/seqvex/issues/46))**
- Keep reference semantics intact; any SIMD/parallel reduction, new precision, custom allocator, build profile or hardware backend requires its own justified design/evidence. CPU/software opportunities precede layout/device work; GPU remains later. No optimization is authorized by this measurement child. **([#23](https://github.com/Skybrique/seqvex/issues/23))**

### Ownership of later optimization — keep four workstreams

| Concern | Primary owner | Supporting verification |
|---|---|---|
| Approved Rust/process/CPU optimization code for OLS, local workspace reuse or independent execution | [#43](https://github.com/Skybrique/seqvex/issues/43) | [#46](https://github.com/Skybrique/seqvex/issues/46) establishes bottleneck and before/after evidence |
| Floating-point order, precision, rank or coefficient changes | [#44](https://github.com/Skybrique/seqvex/issues/44) | Implementation supplies code; #23 approves any contract change |
| Ordering, availability or refit/recovery effects | [#45](https://github.com/Skybrique/seqvex/issues/45) | Implementation supplies the affected workflow |
| New public/shared device, synchronization, allocator or storage architecture | [#23](https://github.com/Skybrique/seqvex/issues/23) for the decision | Propose separate ownership only for an independently meaningful new objective/architecture |

A deployment contract describes accepted behavior and limits; it is not the default container for every optimization issue. Retain tasks in these four workstreams where objective and category match. Correctness, measured bottleneck, applicable critical architecture review, explicit implementation authorization and after-change evidence remain necessary (§33.3).

The opportunity audit must examine allocations/copies, bounded capacities, exclusive mutable resources, independent work, CPU utilization, numerical reductions and layout/device assumptions. “Zero-cost abstraction” is a design/property claim to verify in the compiled hot path, not a promised total cost. Send/Sync express thread-safety contracts; they do not introduce blocking. Async scheduling is not itself parallel computation. Independent predictions may be candidates for CPU parallelism, while within-stream dependent updates remain ordered. No Rayon, SIMD, allocator, layout or profile implementation is part of folder preparation.

The current crate forbids unsafe code, while standing policy permits justified low-level unsafe under review. No lint, allocator or panic-policy change is proposed. **([#23](https://github.com/Skybrique/seqvex/issues/23))**

## 11. Practical temporal example and edge cases

**Illustrative eligibility example:** target = two-observation-ahead return; refit cutoff = after the close of observation 100. This table demonstrates availability only, not a sufficient training dataset.

| Feature origin | Features available | Target becomes available | Eligible at cutoff 100? | Validation owner |
|---|---:|---:|---|---|
| 98 | 98 | 100 | Yes | [#45](https://github.com/Skybrique/seqvex/issues/45) |
| 99 | 99 | 101 | No | [#45](https://github.com/Skybrique/seqvex/issues/45) |
| 100 | 100 | 102 | No | [#45](https://github.com/Skybrique/seqvex/issues/45) |

- Turn this timeline into a positive eligibility test and a negative leakage control; use explicit event/availability semantics rather than assuming sorted rows are enough. **([#45](https://github.com/Skybrique/seqvex/issues/45))**
- Provide chronological evaluation with an explicit baseline and suitable prediction-error metric. If \(R^2\) is offered, define constant-target behavior; do not silently hide undefined metrics. A local example need not create a generic metrics framework. **([#45](https://github.com/Skybrique/seqvex/issues/45))**
- Document what users should do for invalid shape, non-finite input, constant/duplicate features, inadequate rows, incompatible schema and failed refits, using the actual supported errors/outcomes. **([#43](https://github.com/Skybrique/seqvex/issues/43))**
- Explain that periodic refitting can adapt a candidate to a newer window but does not guarantee improved future performance or automatic regime robustness. **([#45](https://github.com/Skybrique/seqvex/issues/45))**

## 12. README maintenance and completion links

- Keep scope/API/fitting/execution/failure/examples accurate with the implementation delivery and link its tested revision. Rustdoc and working examples are part of the same workstream. **([#43](https://github.com/Skybrique/seqvex/issues/43))**
- Maintain mathematical formulas, tolerances, rank policy, oracle links and numerical limits beside the independent evidence. **([#44](https://github.com/Skybrique/seqvex/issues/44))**
- Maintain temporal assumptions, eligibility examples, regimes and evaluation limitations beside their evidence. **([#45](https://github.com/Skybrique/seqvex/issues/45))**
- Maintain performance tables, workload/provenance links and memory distinctions beside the measurement delivery. **([#46](https://github.com/Skybrique/seqvex/issues/46))**
- Maintain the parent acceptance/dependency/milestone map and final integration review. A linked issue alone is not proof that its requirement is implemented or validated. **([#23](https://github.com/Skybrique/seqvex/issues/23))**

The sections address the runbook's algorithm/problem, mathematics, assumptions/regimes, learning, execution/state, streaming/grouping, validation, performance/memory, API/examples, limitations and edge-case requirements.

## 13. Architect design and Planner plan

The governing lifecycle is defined by [FEATURE_DEVELOPMENT.md](../../../../FEATURE_DEVELOPMENT.md), particularly its roles, lifecycle and documentation sections, and [DEVELOPMENT.md](../../../../docs/DEVELOPMENT.md). The README supplies requirements and declared user behavior; its tables and candidate options are inputs to design, not approval to implement.

| Deliverable | Responsible role | Required contents | Review boundary |
|---|---|---|---|
| Technical design | Architect | Exact mathematics and supported envelope; public API; solver alternatives; precision/rank policy; ownership/lifetimes/capacities; artifact and failure/recovery contracts; architecture impact; risks, evidence and deferred decisions | Reviewed before Planner implementation sequencing; accepted decisions recorded or linked in #23 |
| Implementation plan | Planner | Approved design revision; issue-owned tasks; concrete source/module/Cargo target map; dependencies and order; meaningful validation/oracles; acceptance evidence; example/documentation work; independent-review handoff and stop conditions | Reviewed before Coder implementation; cannot redesign or silently settle unresolved choices |
| Implementation and evidence | Coder | Only the approved design and plan; code, tests, user examples and accurate documentation | Independent Code Reviewer assesses applicable correctness, architecture, mathematics, temporal and operational evidence |

The full technical design and implementation plan each have an identifiable revision and a durable issue-linked location. [#23](https://github.com/Skybrique/seqvex/issues/23) is the decision/traceability entry point; [#43](https://github.com/Skybrique/seqvex/issues/43) links the affected implementation tasks. A requirements checklist, folder map, generated example or passing test run does not substitute for either deliverable. For a small change, both layers may be concise, but their responsibilities and reviews remain explicit. **([#23](https://github.com/Skybrique/seqvex/issues/23))**

**Historical pre-implementation question table:** the implemented baseline is accepted following the design and scoped reviews. Preparation/partition helpers, ordering/shuffle facilities and formal numerical certification are explicitly deferred. The original questions below retain their planning context; they are not a claim that all broader lifecycle or production requirements have been implemented.

| Design question | Required decision/evidence | Owner |
|---|---|---|
| Objective, input and intercept | Row/target ownership and shapes, feature count, optional intercept, supported data sizes and capacity | #23; implementation #43 |
| Solver and rank | Compare rank-aware QR and SVD approaches, dependency/resource cost and reversal cost; define threshold/scale interpretation and rank-deficient/underdetermined outcomes; no default inversion of normal equations | #23; validation #44 |
| Fitting precision and conversion | Evaluate numerical need/cost of a separate fitting precision, including a possible f64 workspace; preserve the current ordered f32 predictor and validate conversion | #23; implementation #43; validation #44; measurements #46 |
| Resource ownership | Immutable active parameters; isolated mutable fitting resources; retention, passes, lifetimes, bounds, release points and old/candidate overlap | #23; implementation #43; measurements #46 |
| Public API and failures | Concrete local fit/result/diagnostic types, error precedence, capacity/allocation limits and unsupported inputs | #23; implementation #43 |
| Artifact and acceptance | Format/version/precision/limits; feature order and transform linkage; validation before installation; failed-candidate preservation; in-flight model policy and retry/restart boundaries | #23; implementation #43 |
| Temporal workflow | Task/horizon, availability, cutoff, training-only transforms, overlapping evaluation and declared statistical assumptions | #23; validation #45 |
| Performance acceptance | Representative fitting/serving workloads and budgets, memory distinctions, sampling/provenance and justified repeats | #23; evidence #46 |

Run the applicable critical architecture review before choices affecting ownership, workspace, synchronization, memory/device placement or other governed architectural boundaries are implemented. A scoped OLS artifact/recovery contract must not silently standardize storage, supervision or publication for the whole project. **([#23](https://github.com/Skybrique/seqvex/issues/23))**

## 14. Local code, tests and examples

| Current repository path | Responsibility | Owner |
|---|---|---|
| `src/models/linear/mod.rs` and family README | Family namespace and orientation | #43 |
| `src/models/linear/ols/mod.rs` | Canonical public exports | #43 |
| `src/models/linear/ols/predict.rs` | Immutable prediction, preserved bare behavior and additive checked serving | #43 |
| `src/models/linear/ols/fit.rs` | Selected-row fitting, rank/conversion checks and diagnostics | #43; verification #44 |
| `src/models/linear/ols/artifact.rs` | Metadata, identity and binary artifact v1 | #43 |
| `src/models/linear/ols/tests/oracle.rs` | Independent exact references; Cargo target `ols_oracle` | #44 |
| `src/models/linear/ols/tests/fitting.rs` | Fitter tests; Cargo target `ols_fit` | #43; #44 |
| `src/models/linear/ols/tests/ols_artifact.rs` | Artifact tests; Cargo target `ols_artifact` | #43 |
| `src/models/linear/ols/tests/checked_serving.rs` | Checked serving/executor tests; Cargo target `ols_checked` | #43; #44 |
| `src/models/linear/ols/tests/ols_temporal.rs` | Caller-constructed temporal selection tests; Cargo target `ols_temporal` | #45 |
| `src/models/linear/ols/tests/profiles.rs` | Native-f64, checked serving and strict-f32 evidence; Cargo target `ols_profiles` | #43; numerical support #44 |
| `src/models/linear/ols/examples/fit_salary.rs` | CSV fit/evaluation/artifact/refit demonstration; Cargo example `fit_salary` | #43 |
| `src/models/linear/ols/examples/fit_profiles.rs` | Typed in-memory A/B/C training-only fit and held-out MAE/R² demonstration; Cargo example `fit_profiles` | #43; numerical support #44 |
| `src/models/linear/ols/benches/fit.rs` | Fitting/checked serving/artifact/refit resource evidence; Cargo bench `ols_fit`, `harness = false` | #46 |
| `src/models/linear/ols/tests/public_contract.rs` | Public-contract integration tests; Cargo target `linear_regression` | #43; numerical support #44 |
| `src/models/linear/ols/benches/execution.rs` | Prediction benchmark; Cargo target `linear_regression`, `harness = false` | #43 for wiring; #46 for methodology/evidence |
| `src/models/classic/mod.rs` | Historical compatibility re-exports of the same canonical types | #43 |
| `benches/common/mod.rs` | Shared measurement support | #46 |

Algorithm-local unit/component tests and examples stay with their algorithm where practical. Cross-component integration belongs at the appropriate family/functional/project level. Nested integration tests, benchmarks and examples require actual Cargo/module wiring; a directory does not establish discovery. Introduce shared family helpers only for demonstrated concrete consumers. **([#43](https://github.com/Skybrique/seqvex/issues/43))**

User examples must ingest training data, execute the actual accepted fitting API, validate the learned result and demonstrate single/stream/group prediction from that result. Show canonical imports, feature order/shapes, expected outputs, errors and operating limits. Preset-coefficient demonstrations do not qualify. Register runnable examples explicitly and verify actual results; add useful compiled rustdoc where appropriate. Fitting/refit/artifact examples must follow approved executable APIs. Never replace a failed calculation with a fabricated/default prediction. **([#43](https://github.com/Skybrique/seqvex/issues/43))**

## 15. Evidence and requirement reconciliation

Evidence names the capability, checked revision, fixture/workload, command/method, result and practical limits. Historical prediction evidence does not establish correctness or performance of an absent fitter. Required mathematical, temporal and resource acceptance stay with their respective owners.

When a mismatch is found: identify the exact contract/evidence gap; revise the affected requirement for review; reconcile its existing issue; obtain the affected Architect design and Planner plan updates; implement and independently verify the accepted correction; update the README's actual behavior and evidence links. Do not create a child for one fixture, rerun or documentation correction. **([#23](https://github.com/Skybrique/seqvex/issues/23), supporting #43–#46)**

## 16. Precision profiles A/B/C

The additive typed profiles preserve the accepted legacy `fit`/`FitResult`/`LinearRegression` path, artifact v1, and `check_acceptance` behavior. They are implemented from the LF-3-r1 design and its approved implementation plan; the package is awaiting independent review and is not a new artifact or persistence format.

| Profile | Fit entry point | Learned result | Serving |
|---|---|---|---|
| A: native f64 | `fit_f64(&[Vector64], &[f64], &[usize], &FitConfig64)` | `FitResult64` with `FitDiagnostics64` and f64 weights/intercept | `LinearRegression64` or borrowing `CheckedLinearRegression64` |
| B: checked f32 serving | Same `fit_f64`/`FitResult64`; no separate learner | `FitResult64::to_serving_f32(&self, UnderflowPolicy)` | Owned existing `LinearRegression`; f64 learner remains available |
| C: strict f32 | `fit_f32(&[Vector], &[f32], &[usize], &FitConfig32)` | `FitResult32` with `FitDiagnostics32` and f32 weights/intercept | Existing `LinearRegression`/`CheckedLinearRegression` |

A and B share one native-f64 CPQR implementation; C uses a separate native-f32 CPQR path with f32 rank, solver, parameters, prediction and RSS arithmetic. All profiles inspect selected training rows only, preserve the legacy validation precedence and rank policy (with profile-specific unit roundoff), and expose single, streaming and caller-bounded grouped inference. They remain dataset fit/refit APIs: grouping does not imply incremental OLS.

`to_serving_f32` is non-mutating and scans weights in index order, then the intercept when enabled. It rejects non-finite sources, finite-to-infinite overflow, and (by default) nonzero-to-zero underflow; `UnderflowPolicy::AllowLoss` explicitly accepts the zero. Representable subnormals and signed zero are preserved. The returned f32 predictor is independently owned; on conversion or later checked-acceptance failure, the f64 learner and any active predictor remain unchanged. There is no automatic install or fallback.

Run the typed in-memory example with:

```sh
cargo run --locked --example fit_profiles
```

`PROFILE_QUALITY_FIXTURE` fits rows `[0,1,2,3,4]` only, predicts held-out rows `[5,6,7]`, and reports finite f64-evaluated MAE/R² for each profile. That evaluation is example-only and is not production C arithmetic; it is not an accuracy threshold, generalization claim or replacement for the independent numerical tests.

## References

- [Feature development runbook](../../../../FEATURE_DEVELOPMENT.md)
- [Development contract](../../../../docs/DEVELOPMENT.md)
- [Architecture](../../../../ARCHITECTURE.md)
- [Failure and recovery](../../../../docs/FAILURE_AND_RECOVERY.md)
- [Streaming ML programme draft](../../../../docs/STREAMING_ML_DESIGN.md)
- [Correctness methodology draft](../../../../docs/CORRECTNESS.md)
- [Fitter](fit.rs), [artifact implementation](artifact.rs), [predictor and checked serving](predict.rs)
- [Independent references](tests/oracle.rs), [fitter tests](tests/fitting.rs), [artifact tests](tests/ols_artifact.rs), [checked-serving tests](tests/checked_serving.rs) and [temporal-selection tests](tests/ols_temporal.rs)
- [CSV workflow example](examples/fit_salary.rs), [fitting/resource benchmark](benches/fit.rs), [public-contract tests](tests/public_contract.rs) and [prediction benchmark](benches/execution.rs)
- [Historical algorithm correctness audit](../../../../docs/ALGORITHM_CORRECTNESS_AUDIT.md) — historical evidence, not fitter acceptance
- [LAPACK least-squares guide](https://www.netlib.org/lapack/lug/node27.html) — solver-policy background, not a backend selection
