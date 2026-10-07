# Seqvex correctness and validation guide

## Mandatory learned-model evidence

Apply the [mandatory training-to-inference contract](DEVELOPMENT.md#model-training-and-valid-inference--mandatory). Every model must demonstrate actual data ingestion, its algorithm-specific training/learning, validation of the learned result and inference from it. Separate arithmetic correctness, numerical accuracy and measured prediction quality. Preset parameters, initialized state, printed expected outputs and successful kernel tests do not substitute for fitting evidence.

Raise unresolved failure cases and recovery/numerical assumptions to the maintainer before affected work proceeds; follow the [failure decision rule](DEVELOPMENT.md#calculation-failures-and-error-handling--mandatory). Record rejected operations explicitly without substituting successful predictions.

> **Status:** PROPOSED DRAFT v1, 2026-10-06, Asia/Singapore; for review.
> **Repository location:** docs/CORRECTNESS.md.
> **Authority:** Standing-process proposal subordinate to DEVELOPMENT.md, FEATURE_DEVELOPMENT.md, FAILURE_AND_RECOVERY.md and applicable algorithm contracts.
> **Source checkpoint:** rust-development at 61bca3156997b90671599fbeb48121c8284be899.
> **Boundary:** This file is a proposed methodology guide, subordinate to the governing development and algorithm contracts. It does not retroactively add acceptance conditions to existing issues.

## 1. Purpose and evidence limits

Establish how an algorithm implementation can be justified as correct within a declared scope. Use the Common Audit Core plus algorithm-specific analysis already described in DEVELOPMENT.md Sections 33.2-33.4.

This guide is different from ALGORITHM_CORRECTNESS_AUDIT.md. That report records findings for named audited revisions; this guide describes a repeatable evidence process. Preserve historical audit revisions, findings and limitations.

Passing tests does not prove absence of every defect, universal predictive accuracy or readiness for every deployment. A review verdict must name the implemented task, numerical/data envelope, failure contract, checked revision, actual evidence and limitations.

Distinguish:

- Mathematical correctness: implementation follows the specified algorithm.
- Numerical correctness: precision, stability and exceptional values satisfy the approved envelope.
- Temporal/statistical validity: data availability and evaluation respect the intended regime.
- Operational correctness: state, resources, concurrency, failure and recovery satisfy their contracts.
- Useful predictive quality: the declared evaluation demonstrates relevant results.

A mathematically correct model can be statistically inappropriate or predict poorly. Strong backtest results do not establish implementation correctness.

## 2. Start with an exact contract

Before selecting tests, specify equations/objective, task variants, shape and units, intercept/penalty normalization, parameterization, state transition and commit boundary. State initialization, reset, continuation, output meaning, unsupported inputs and capacity behavior.

Declare finite-value rules, precision, tolerances, reduction/FMA policy, convergence/status and determinism scope. Bitwise equivalence is required only where the owning contract requires it; elsewhere define justified error criteria.

Declare the learning lifecycle and what constitutes one observation or learning update. Do not assume prediction implies training or that every algorithm supports incremental learning.

**Example:** a GRU comparison must match gate definitions, matrix orientation, update convention and reset placement. A library implementing a different legitimate GRU variant is not automatically a matching oracle.

## 3. Independent oracles and comparisons

Choose evidence that would detect a shared implementation mistake.

| Evidence | Useful role | Important limitation |
|---|---|---|
| Hand-derived small cases | Check exact values, signs, indexing, transitions and boundaries | Cover only selected cases |
| Independently written scalar mathematics | Compare with optimized/vectorized paths without sharing the kernel | Independence requires examining shared assumptions/helpers |
| Higher-precision reference | Quantify rounding/stability under an aligned formulation | Higher precision alone does not establish semantic independence |
| Closed-form or alternate mathematical formulation | Check an iterative solver or recurrence where applicable | Conditioning, regularization and conventions must match |
| External implementation | Differential comparison under documented equivalent semantics | Versions, defaults, stopping and variants may differ |
| Properties/invariants | Check constraints across broad generated inputs | Self-consistency alone is insufficient |
| Realistic temporal workflow | Test causal execution and quality under intended use | Quality evidence is not a mathematical oracle |

Do not label two paths independent if they call the same arithmetic kernel or derive expected values from the implementation under test. Record oracle provenance and genuinely shared components.

Use fixed, understandable fixtures first. Broader generated cases should preserve the declared domain and retain seeds plus minimized failure inputs.

For fitted algorithms, validate the declared objective and appropriate optimality/convergence conditions. Gradient checks can compare analytic gradients against finite differences on small well-conditioned cases with justified step sizes; nonsmooth points and L1 subgradients require specialized tests rather than an unqualified finite-difference check.

External tools are reference material, not automatic dependencies or ground truth. Match data, objective scaling, sample weights, intercept, tie rules, seeds and stopping criteria before interpreting differences.

## 4. Algorithm-specific evidence examples

These are design prompts, not a mandatory identical test suite for every algorithm or new retroactive acceptance requirements.

| Algorithm/task | Useful independent evidence and boundaries |
|---|---|
| OLS/Ridge | Small analytic solutions; residual/optimality checks; rank/conditioning; correct penalty and intercept treatment |
| Lasso | Known sparse solutions; KKT/subgradient conditions; scaling, convergence and boundary behavior |
| Logistic classification | Stable loss/probabilities; gradient/optimality; class mapping and specified regularization |
| RLS | Independent scalar recurrence and aligned weighted least-squares comparison; forgetting, excitation, conditioning and long horizon |
| GRU predictor | Independent recurrence; gates/reset/order; task head; actual training objective, gradients and truncation policy required for complete model acceptance |
| Decision Tree | Hand-computed split objectives and predictions; thresholds/ties, stopping, leaf values and task-specific statistics |
| Hoeffding Tree | Statistics, split criterion and bound computation; selected variant assumptions; capacity, class growth and declared drift behavior |
| Bagging/Random Forest | Sampling/member identity, aggregation and base-learner evidence; whole-ensemble failure and RNG progress |
| Gradient-boosted trees | Loss, residual/gradient construction, sequential stage updates, weighting and stopping |
| KNN | Hand-computed distances/neighbors; weighting/ties; classification/regression outputs; retained-set lifecycle |
| K-means | Assignment and centroid updates; specified objective; initialization, empty clusters and selected incremental/replay method |

For dependent observations, distinguish evaluating behavior from claiming a theorem's assumptions hold. For example, correctness of a Hoeffding-bound calculation does not establish its usual probabilistic guarantee under arbitrary serial dependence.

## 5. Sequential data, causality and non-IID validation

Record event order, source of ordering, feature availability, label maturity, group/sequence boundaries and preprocessing update timing.

Design tests to detect:

- Future features/statistics used at an earlier decision.
- Labels used before maturity.
- Preprocessing fit across validation/evaluation boundaries.
- Recurrent states or learned state leaking between supposedly isolated streams.
- Unintended shuffle, sampling or reordering.
- Incorrect reset/continuation and horizon alignment.
- Unexpected sensitivity to future-only changes.

**Example:** changing data available only after time t should not change predictions already produced through t under a causal fixed-history workflow. Define availability precisely; a completed historical training fit is a different information boundary.

Test serial correlation, repeated inputs, shifts, changing excitation, abrupt regimes and long horizons where relevant. Separate correct behavior under the specified mathematics from evidence of predictive/adaptive effectiveness.

Use chronological/walk-forward evaluation with fit/update boundaries recorded. Where labels overlap across time, analyze the relevant leakage mechanism and appropriate isolation, purging/embargo or equivalent protocol. No random split or shuffle is automatically acceptable merely because a library supports it.

Permutation expectations are algorithm-specific. Reordering independent frozen predictions should preserve input/output association; it does not authorize reordering GRU transitions or forgetting-weighted RLS updates. Floating-point reduction order can also affect fitted results.

## 6. Streaming and bounded-group equivalence

Define one-observation behavior before testing groups. For an ordered-learning contract, compare repeated streaming updates with the same ordered observations in different legal chunk partitions.

Compare outputs, committed state and any learner/optimizer/RNG state required by the contract. Test bounds, empty groups, tail groups, resets, continuation and repeated calls.

**Example:** a group of three observations under the ordered-update policy performs three updates. It is not one observation, and it is not one gradient-accumulation update.

Check rejected bounds before state advancement where that is the API contract. Inject failure at each relevant position; verify reported committed progress, preserved valid state and documented next action. Do not assume whole-group rollback if the declared policy commits successful observations individually.

A prefix precheck does not establish full-horizon equivalence. State exactly what was checked before timing and which longer tests exist separately.

## 7. Failure, recovery and resource boundaries

Use FAILURE_AND_RECOVERY.md as the authority. Identify the last committed valid state and make failure observable. Select recovery mechanisms through the owning design, not through a universal rollback rule.

For relevant mutations, test failures before preparation, during candidate work, at commit and during persistence/publication. Include model, trainer/optimizer, preprocessing, RNG advancement and checkpoint metadata where they form a consistency boundary.

**Example:** if an ensemble partially updates members and then returns an error, retrying the observation can duplicate successful updates. The design must either prevent such a partial commit or explicitly identify progress and define a safe continuation/recovery path.

Test common invalid/unsupported inputs, checked size arithmetic, capacity exhaustion and shutdown/cancellation boundaries. Recoverable resource failures need explicit error behavior where supported; do not promise that every process-level allocation failure is catchable.

Persistence tests, when in scope, verify version/schema compatibility, complete restore/continuation, corruption handling and consistent component snapshots. A checkpoint test must cover the defined durability boundary, not merely an in-memory round trip.

panic=abort ends the process and cannot support in-process panic recovery. Validate a proposed abort deployment through subprocess behavior; do not rely on destructors or post-panic checkpointing. Returned Result errors are a separate mechanism.

Common edge cases and failure assumptions belong in the owning algorithm design. Findings that contradict standing failure principles must be communicated promptly before affected implementation proceeds.

## 8. Numerical stress and accuracy

Cover zero/small values, large dynamic range, cancellation, boundary thresholds, near-singular/ill-conditioned cases, long recurrence histories and domain-specific stability conditions.

Test NaN/infinity/overflow/underflow according to the approved input/output contract. Do not demand acceptance of values outside the supported envelope or silently widen that envelope.

Use justified absolute/relative error criteria, with care near zero; ULP-based or structural measures may be appropriate for particular kernels. Check algorithm invariants and meaningful downstream outputs, not only elementwise closeness.

For optimizations, inspect precision conversion, reassociation, FMA, reduction topology and approximation effects. A difference can be more accurate yet fail a bitwise contract; resolve policy before adoption.

Reduced precision/quantization needs both numerical and task-quality evidence under the intended temporal workload. It is not an equivalent implementation by default.

## 9. Concurrency, scheduling and randomness

Send/Sync bounds support safe ownership/sharing; they do not prove event order, determinism, deadlock freedom or failure atomicity. Async does not automatically make CPU work parallel or blocking operations nonblocking.

Test independent stream isolation, declared shared-learner ordering, model-version association, aggregation, task cancellation, shutdown, backpressure and bounded memory. Vary thread count and scheduling where those dimensions are supported.

Race-free code can still update the wrong model version or process observations twice. Stress tests are evidence, not proof of every interleaving. Use targeted concurrency-model tools or sanitizers when justified by a concrete risk and supported platform; do not impose an unrelated dependency on every algorithm.

CURRENT: RANDOMNESS.md covers caller-owned parameter-value generation only. It excludes sampling, resampling, shuffling and data reordering and forbids global/thread-local/model-owned generators within that facility. Future ensemble sampling requires a separate architecture decision; this guide does not expand the existing facility.

For an approved future sampling design, check logical member seed identity, draw order, retries, checkpoint restoration and determinism scope. Scheduling should not accidentally select different samples where reproducibility is promised. No cross-version or cross-hardware bitwise promise follows from a seed alone.

## 10. Optimization correctness and performance evidence

Before optimization: establish correctness, measure the bottleneck and review architecture constraints. Compare scalar/reference against compiler-vectorized, explicit SIMD, parallel or backend variants under the same approved semantics.

Check vector tails, bounds, target dispatch/fallback, layout conversions, reduction order, long horizons, thresholds/ties and failure behavior. AoS/SoA/AoSoA are representation choices; none alone establishes correct or efficient SIMD.

Run complete-workflow checks as well as kernel comparisons. Preprocessing, transfers, completion synchronization, output association and commit acknowledgement remain part of correctness.

Measure release latency and throughput, memory and allocations with a declared region and equivalent useful work. Timing-window variability is not event-level tail latency. Parallel work must finish within the measured completion boundary.

For learned models, preserve training policy and stopping/quality criteria. A faster result caused by fewer updates or lower quality is a different trade-off, not an equivalent speedup.

Use the interaction checklist in STREAMING_ML_DESIGN.md Section 16. Recheck changed ownership, synchronization, numerical or representation assumptions.

## 11. Validation levels and review

| Level | Required role when applicable |
|---|---|
| Contract/unit | Hand cases, independent arithmetic, boundaries and invariants |
| Algorithm integration | Full prediction/learning sequence, reset, failure and grouping |
| Composition integration | Transform/model/version/evaluation/recovery compatibility |
| Temporal evaluation | Causality, dependent regimes and relevant quality |
| Operational validation | Concurrency, capacity, persistence and deployment failure |
| Release benchmarking | Latency, throughput and resource behavior with correctness guards |
| Independent review | Implementation, architecture, mathematics, statistical regime and operations |

Use the runbook's validation matrix and required Rust checks. Compilation, fmt and Clippy are necessary engineering evidence; they do not establish mathematical correctness.

Mark evidence as EXECUTED, REUSED WITH APPLICABILITY VERIFIED, NOT RUN or NOT APPLICABLE with reasons. Distinguish reviewer-inspected evidence from execution facts recorded by others. Name exact revisions and source changes that limit reuse.

Choose meaningful tests that can fail for plausible defects. Avoid tests that simply restate the implementation. No single count of passing cases is a universal acceptance threshold.

## 12. Evidence record template

Embed this record in the owning issue-linked design/review; do not file a separate issue for each test.

| Field | Content to record |
|---|---|
| Owner/scope | Issue, task variant, design revision and declared deployment envelope |
| Implementation | Exact commit/dirty scope, source paths, backend and numerical policy |
| Mathematical oracle | Independent derivation/implementation, shared components and comparison criteria |
| Inputs/regime | Shapes, sequence/groups, availability, labels, seeds, conditioning and capacity |
| Expected behavior | Equation/property, ordering, state and failure boundary |
| Validation | Commands/tests, actual execution status, toolchain/platform and raw evidence |
| Observed behavior | Values/errors/states, comparison result and reproducible failures |
| Performance | Release region, equivalent work, latency/throughput/memory, variability and provenance |
| Limitations | Unchecked conditions, supported envelope and deferred choices |
| Findings | Severity, affected scope, minimal remedy, owner and verification |
| Verdict | Satisfied within scope, blocked, or explicit unresolved decision |
| Gates | Review status; separately authorized integration and closure |

A reused result requires source/contract applicability checks. A documentation-only change does not make a historical execution a new run.

## 13. Adoption and source references

This proposal adds a teaching/evidence guide, not a new mandatory audit framework, API or issue hierarchy. Canonical conflicts require notification and an explicit authorized decision. A guide edit cannot silently settle deferred architecture or change an existing issue's acceptance.

Relevant checkpoint sources:

- [DEVELOPMENT.md](https://github.com/Skybrique/seqvex/blob/61bca3156997b90671599fbeb48121c8284be899/docs/DEVELOPMENT.md), especially Sections 3, 17, 19, 24 and 33.
- [FEATURE_DEVELOPMENT.md](https://github.com/Skybrique/seqvex/blob/61bca3156997b90671599fbeb48121c8284be899/FEATURE_DEVELOPMENT.md), review dimensions, validation matrix and issue templates.
- [FAILURE_AND_RECOVERY.md](https://github.com/Skybrique/seqvex/blob/61bca3156997b90671599fbeb48121c8284be899/docs/FAILURE_AND_RECOVERY.md).
- [RANDOMNESS.md](https://github.com/Skybrique/seqvex/blob/61bca3156997b90671599fbeb48121c8284be899/docs/RANDOMNESS.md).
- [ALGORITHM_CORRECTNESS_AUDIT.md](https://github.com/Skybrique/seqvex/blob/61bca3156997b90671599fbeb48121c8284be899/docs/ALGORITHM_CORRECTNESS_AUDIT.md).
- [Companion design draft](STREAMING_ML_DESIGN.md), Sections 10, 13, 16 and 17.

Technical definitions: [Rust Sync](https://doc.rust-lang.org/std/marker/trait.Sync.html), [Rust concurrency and async](https://doc.rust-lang.org/book/ch17-06-futures-tasks-threads.html), [LLVM vectorizers](https://llvm.org/docs/Vectorizers.html).

**Review boundary:** this draft was reviewed for consistency with the inspected standing documents. No implementation tests, benchmarks or new independent algorithm audit were executed for it.
