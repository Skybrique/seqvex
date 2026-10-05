# Algorithm Correctness Audit — Issue #27

Status: OPEN — this report records evidence and remaining gaps; it does not close
#27.

Scope: mandatory initial algorithms Linear Regression (#23), GRU (#17–#19),
Recursive Least Squares (#26); Decision Tree (#24) and K-Nearest Neighbors (#25)
are recorded as supplementary coverage and as an unresolved scope clarification.

Evaluated revisions:

- `main` @ `cf95cd08b7f6b87b1d116acdbd947bc3ff92c10b` — inspected via git objects
  only; no tests were run on `main`.
- `rust-development` @ `276fcc5b36a14c78efd90c623970a6b6e98567a8` — inspected; the
  focused tests reported below were executed here.

Evidence labels: `FACT`, `MEASURED EVIDENCE`, `INFERENCE`, `ASSUMPTION`,
`PROVISIONAL DECISION`, `ARCHITECTURAL DECISION`, `OPEN QUESTION` (see
[`DEVELOPMENT.md`](DEVELOPMENT.md) §33.4), with provenance tags `[MAIN]` (present
on the evaluated `main`), `[DEV]` (present only on `rust-development`), `[HIST]`
(historical/reported), `[MISSING]`, and `[N/A — reason]`.

Delivery: the evaluated baseline lacked this report and the LR/GRU applicability
subsections; this delivery adds them. Independent review is complete (recorded
in #27 comment 8); integration remains pending.

Optimization: CLOSED — not authorized by this report.

---

## 1. Executive findings

- Mandatory initial scope is LR, GRU and RLS. Decision Tree and KNN coverage is
  supplementary; their missing full-audit coverage does not block #27's original
  scope (scope ruling recorded in §20).
- No defect identified within this review's scope (§16).
- Status by baseline: the durable structured report and the LR/GRU applicability
  subsections were missing at the evaluated baseline; this delivery adds them.
- Status pending: integration of this report and the two README subsections;
  development-only bounded GRU/RLS evidence; and the optimization-time
  architecture question. The Decision Tree/KNN scope ruling and the
  revision-of-record statement are recorded in §20.
- Optimization remains closed (§19).

## 2. Repository/code surfaces inspected

Inspected at `rust-development` @ `276fcc5`, compared against `main` @ `cf95cd0`
by git object inspection. No file was modified to produce this report beyond the
three documentation artifacts named in the delivery.

- Foundation and execution: `ARCHITECTURE.md` §6.3; `src/foundation/state/README.md`,
  `src/foundation/state/transition.rs`, `src/foundation/state/atomicity.rs`;
  `src/execution/README.md`.
- LR: `src/models/classic/README.md`, `src/models/classic/linear_regression.rs`,
  `tests/linear_regression.rs`.
- DT: `src/models/classic/decision_tree.rs`, `tests/decision_tree.rs`.
- KNN: `src/models/classic/knn.rs`, `tests/knn.rs`.
- GRU: `src/models/recurrent/README.md`, `src/models/recurrent/gru.rs`,
  `tests/gru.rs`.
- RLS: `src/models/online/README.md`, `src/models/online/rls.rs`,
  `tests/rls.rs`.
- Cross-cutting: `tests/streaming.rs`; `benches/gru.rs` history;
  `docs/ML_VERTICAL_SLICES.md`.
- Governance: `docs/DEVELOPMENT.md` §24, §27, §33; `FEATURE_DEVELOPMENT.md`.

### Revision provenance

- Merged on `main`: RLS validation PR #31 (merge `e8785ee`), GRU audit PR #32
  (merge `20645c9`), execution-ownership PR #39 (merge `38a4935`). [MAIN]
- The branches have diverged: `main` carries 2 commits absent from
  `rust-development` (CI work `0fe2394`, merge `cf95cd0`), and `rust-development`
  carries 7 commits absent from `main` (`e39478a`, `0c95401` #34 GRU bounded,
  `6be36f1` #8 numerical doc, `1735cd3` #18 GRU measurement doc, `c6302d9` #24 DT
  bench, `eff04ca`, `276fcc5` #26 RLS bounded/bench). [FACT]

## 3. Algorithm-by-algorithm mathematical specification

- LR [MAIN]: fixed-coefficient prediction `ŷ = w·x + b` over caller-supplied
  immutable weights (`src/models/classic/README.md`; `linear_regression.rs`). No
  fitting or adaptation.
- GRU [MAIN]: update gate `z = σ(W_z x + U_z h + b_z)`, reset gate
  `r = σ(W_r x + U_r h + b_r)`, candidate
  `c = tanh(W_h x + U_h (r ⊙ h) + b_h)`, next state `h' = (1 − z) ⊙ h + z ⊙ c`
  (`src/models/recurrent/README.md` §2; `gru.rs::compute`).
- RLS [MAIN]: ordered recurrence `v = P x`, `d = λ + xᵀ v`, `k = v / d`,
  `w' = w + k e`, `P' = (P − v vᵀ / d) / λ`. In exact arithmetic, under the
  documented objective and initialization assumptions, the resulting coefficient
  vector `w_t` minimises `J_t`; `P_t` is the inverse-correlation matrix. The `f32`
  implementation is subject to the documented numerical limits and oracle
  tolerances (`src/models/online/README.md`).

## 4. Independent oracle design

Each oracle is independent of the production recurrence, not a reproduction of it.

- LR [MAIN]: `scalar_predict_f32` (separate index loop, pins reduction order) and
  `scalar_predict_f64` (numerical diagnostic with a pre-registered tolerance) in
  `tests/linear_regression.rs`.
- GRU [MAIN]: independent scalar `ref_step`, an alternative formulation over
  randomized sequences, and an exact closed-form decay check in `tests/gru.rs`.
- RLS [MAIN]: an independent scalar recurrence asserted bitwise, a literal `f64`
  weighted-least-squares objective, and D=1 closed forms in `tests/rls.rs` and
  `src/models/online/README.md`.
- DT (supplementary) [MAIN]: an independent closed-form `region` function plus
  hand-derived boundary/equality/feature-selection cases in the "durable
  mathematical evidence" section of `tests/decision_tree.rs`. This is established
  evidence; no randomized/arbitrary-tree oracle is required by this audit.
- KNN (supplementary) [MAIN]: an independent `f64` oracle grid in `tests/knn.rs`.

## 5. Sequential ordering audit

- LR [MAIN]: order-independent; `reduction_order_is_weight_index_order` pins the
  documented reduction order; each prediction depends only on its own observation.
- GRU [MAIN]: order-sensitive by construction; `sequence_order_is_semantically_significant`
  confirms a different trajectory under reordering, matching the independent oracle.
- RLS [MAIN]: `sequence_matches_independent_scalar_reference_bitwise`;
  `permutation_is_invariant_for_lambda_one` and
  `permutation_changes_the_weighted_solution_for_lambda_less_than_one`.
- DT/KNN (supplementary) [MAIN]: inference is order-independent; KNN
  `reordered_queries_give_the_same_per_query_results`. DT has no explicit
  permutation test (supplementary scope).

## 6. Causality / leakage audit

Do not treat leakage as universally not-applicable because prediction is
stateless. Four responsibilities are distinguished:

- Inference dependency: LR `prediction_is_independent_of_previous_observations`
  and `prefix_and_suffix_do_not_affect_prediction`; GRU
  `prefix_state_depends_only_on_observations_through_t`; RLS causal prequential
  protocol documented in `src/models/online/README.md` §Sequential evaluation
  contract. [MAIN]
- Fitting/estimation: Within the mandatory initial scope, LR evaluates fixed
  coefficients and GRU executes recurrence without a parameter-training procedure.
  RLS performs online estimation: each accepted observation updates the coupled
  `(w, P)` state through the documented weighted least-squares recurrence.
  Fitting, preprocessing and temporal-validation responsibilities must be
  distinguished from inference execution. [MAIN]
- Preprocessing: no normalization or feature-scaling API is provided; any
  fit-time preprocessing and its split boundary remain the caller's
  responsibility. [MAIN]
- Dataset validation / temporal validation: ordering, causal labels, and
  no-future-splitting are the caller/data pipeline's responsibility; the RLS
  contract states this explicitly. [MAIN]

## 7. Permutation audit

The correct expectation is verified per algorithm rather than imposed generically.
LR/KNN inference is permutation-invariant per observation; GRU is order-dependent;
RLS is permutation-invariant only for the `λ = 1` final least-squares fit, while
the trajectory and prequential error remain temporal. Sources: `tests/linear_regression.rs`,
`tests/knn.rs`, `tests/gru.rs`, `tests/rls.rs`, `src/models/online/README.md`. [MAIN]

## 8. State isolation and reset audit

- LR [MAIN]: `reset_starts_a_new_streaming_sequence`;
  the carried state is only the most recent prediction
  (`streaming_state_is_the_latest_prediction`).
- GRU [MAIN]: `reset_starts_a_new_sequence`; per-context state under the approved
  ownership (PR #39). Development-only cross-context check
  `bounded_batch_is_deterministic_and_leaves_contexts_independent` [DEV].
- RLS [MAIN]: `one_shared_model_drives_independent_stream_states` and
  `reset_starts_a_new_sequence`.
- KNN (supplementary) [MAIN]: `clustered_queries_do_not_contaminate_each_other`.

Approved Option B ownership (immutable model borrow, execution-owned state,
algorithm-local GRU workspace) is preserved and not reopened.

## 9. Failure/atomicity audit

- LR [MAIN]: `failed_streaming_prediction_preserves_last_prediction`,
  `failure_does_not_corrupt_subsequent_prediction`,
  `batch_element_failure_preserves_model_and_recovers`.
- GRU [MAIN]: non-finite input, wrong-length input, and non-finite candidate are
  rejected while the committed hidden state is preserved;
  `stream_failure_preserves_state_and_continues_from_last_valid`.
- RLS [MAIN]: `failed_update_leaves_committed_state_bitwise_unchanged`; the
  coupled `(w, P)` is returned as one value only after validating, so no partial
  combination is observable (`src/models/online/README.md` §Failure and
  atomicity). The denominator and candidate guards are numerical guards, not
  statistical guarantees.

## 10. Micro-batch/chunking audit

The foundation `process_batch` is the unbounded reference ordered fold
(stop-on-first-failure, returns the last committed state), distinct from
per-algorithm bounded APIs (`ARCHITECTURE.md` §6.3;
`src/foundation/state/README.md`).

- LR [MAIN]: `arbitrary_chunking_preserves_each_prediction_bitwise`,
  `single_observation_batch_equals_predict`, `larger_batch_equivalence`.
- GRU [MAIN]: `chunked_execution_matches_continuous_stream`. Bounded reference
  micro-batch (`Gru::process_batch_reference`) and `bounded_batch_*` tests are
  [DEV] only.
- RLS [MAIN]: `chunking_preserves_the_ordered_trajectory`,
  `process_stream_matches_repeated_update_bitwise`. Bounded reference micro-batch
  (`Rls::process_batch_reference`) and `bounded_reference_*` tests are [DEV] only.
- DT/KNN (supplementary) [MAIN]: classic `micro_batch_*` equivalence tests.

## 11. Non-IID and regime-shift audit

- LR [MAIN]: `non_iid_*` (stationary/IID-like, strongly ordered, clustered
  regimes, abrupt shift, gradual drift, AR(1) autocorrelation, repeated
  observations, heterogeneous magnitudes).
- GRU [MAIN]: `regime_shift_sequence_matches_independent_reference`,
  `autocorrelated_sequence_matches_independent_reference`,
  `repeated_observation_does_not_drift_from_reference`.
- RLS [MAIN]: `regime_e7`–`regime_e11`, R2 noise behavior, and R1 excitation
  regimes (`tests/rls.rs`; `src/models/online/README.md`).
- DT/KNN: no regime suite [MISSING] (supplementary scope). Because inference is
  stateless, this is not evidence of defect.

The purpose is to confirm the algorithm behaves per its specification without
hidden IID assumptions, not to claim universal non-IID suitability.

## 12. Statistical assumptions and applicability

- RLS [MAIN]: the `src/models/online/README.md` "Applicability and statistical
  contract" section states mathematical assumptions, statistical assumptions
  (exogeneity, persistent excitation, identifiability, stationarity/drift),
  forgetting-factor and initial-covariance semantics, the numerical contract, the
  guarantee boundary, the operating envelope, user/data responsibilities, and a
  failure taxonomy.
- RLS exact-objective equivalence is a mathematical property under the documented
  assumptions: `w_t` is the exact minimiser of `J_t` by construction. The `f32`
  implementation, tolerance choices, and documented numerical limits are kept
  distinct from that mathematical identity (`tests/rls.rs`;
  `src/models/online/README.md`).
- LR and GRU: `[MISSING]` at the evaluated baseline; this delivery adds short
  applicability subsections in [`../src/models/classic/README.md`](../src/models/classic/README.md)
  and [`../src/models/recurrent/README.md`](../src/models/recurrent/README.md).
- DT/KNN: no applicability statement [MISSING] (supplementary scope).

### Conventional batch-workflow comparison

This comparison is illustrative of responsibility, not a claim about any specific
library's defaults. The issue is to make temporal assumptions and
responsibilities explicit.

| Aspect | Batch-workflow risk to temporal assumptions | Seqvex/algorithm position | Responsibility |
|---|---|---|---|
| Ordering | shuffled training/replay changes a stateful trajectory | LR/DT/KNN inference order-independent; GRU/RLS ordered | user selects order; Seqvex preserves it |
| Fitting | estimators may fit independently of stream order | implemented fitting-related capability differs per algorithm (below) | caller/upstream |
| Preprocessing | fit-time normalization/scaling can leak across splits | no normalization/scaling API is provided | caller/data pipeline |
| Temporal validation | random splits can invalidate temporal evaluation | causal/prequential protocol documented for RLS; prefix semantics for GRU | caller/data pipeline |

Implemented fitting-related capabilities (FACT, per inspected source):

- LR: fixed-coefficient prediction; no fitting.
- DT: inference over a caller-supplied tree; no tree training.
- KNN: prediction using caller-supplied stored examples.
- GRU: recurrent execution; no parameter-training procedure.
- RLS: sequential coefficient/state updates implementing its documented
  weighted-least-squares objective.

## 13. Long-run numerical audit

- LR [MAIN]: `long_run_*` (repeated, alternating/cyclic, pseudorandom with
  replay, streaming state is last-prediction only); numerical envelope documented
  in `src/models/classic/README.md`, including the `f32` finite-input to
  non-finite-output limitation; one extreme `[ignore]` release-only diagnostic.
- GRU [MAIN]: `long_run_matches_reference_and_stays_bounded` (10,000
  deterministic steps).
- RLS [MAIN]: `long_run_stays_finite_symmetric_and_matches_reference` (10,000
  steps, D=4, λ=0.999); R1–R4 operating envelope and known boundaries in
  `src/models/online/README.md`; eight `[ignore]` release-only diagnostics, not
  run.
- DT/KNN: no long-run numerical audit [MISSING] (supplementary; traversal is
  arithmetic-free for DT).

## 14. Findings by algorithm

- LR: prediction semantics complete (spec, oracle, ordering, causality,
  isolation/reset, failure, chunking, non-IID, long-run). The accepted `f32`
  finite-output limitation is preserved as documented, not a defect.
- GRU: recurrence and sequential/non-IID/long-run/reset/failure evidence complete
  on `main`; bounded micro-batch evidence is [DEV] only.
- RLS: the most complete, including excitation/conditioning boundaries, λ<1
  behavior, prequential evaluation, and a written applicability contract.
- DT/KNN (supplementary): DT has an independent closed-form oracle but no
  permutation/non-IID/long-run suite; KNN has an independent oracle, reordering
  and isolation checks. They are not mandatory #27 scope; full-audit coverage is
  future work (§20) and does not block #27's original scope.

## 15. Cross-algorithm recurring requirements

Recurring requirements across the mandatory algorithms: an explicit
specification; an oracle not derived from the production path; an order
expectation matched to the mathematics; causality; state isolation and reset;
failure atomicity that preserves committed state; chunk/grouping equivalence; a
controlled non-IID/regime check where meaningful; a documented applicability
statement; and an optimization gate not passed on self-consistency. See the
conventional batch-workflow comparison in §12 for the ordering, fitting,
preprocessing, and temporal-validation responsibilities.

## 16. Defects discovered

No defect identified within this review's scope. This is not an unrestricted
defect-free guarantee. The LR `f32` finite-output condition and the RLS
excitation/conditioning boundaries are documented limitations of the tested
envelope, not defects.

## 17. Tests added/recommended

- Added and merged on `main`: RLS validation evidence (PR #31), GRU audit
  evidence (PR #32), the LR audit block, the DT durable mathematical evidence,
  and the KNN independent oracle. [MAIN]
- Added only on `rust-development`: GRU bounded `bounded_batch_*` and RLS bounded
  `bounded_reference_*` tests, and the bounded reference APIs they exercise.
  [DEV]
- Recommended: none mandatory for LR/GRU/RLS. DT/KNN full-audit additions belong
  to future scope (§20); none are created here, and their absence does not block
  #27's original scope.

## 18. Architecture implications

- Approved Option B ownership is preserved: immutable model borrow
  (`&Model`), execution-owned state, and an algorithm-local GRU workspace; the
  former exclusive model borrow and model-owned workspace are not reopened.
  [ARCHITECTURAL DECISION] (PR #39)
- Corrected executor measurements are kept separate from the earlier
  direct-update experiment: benchmark source `2346179` (corrected comparison)
  versus `d2c8976` (earlier direct-update run), both [MAIN]; the consolidated
  measurement document (`1735cd3`, `docs/ML_VERTICAL_SLICES.md`) is [DEV] and is
  not integrated into `main`.
- No new architecture is implied by this report.

## 19. Optimization gate decision

OPTIMIZATION REMAINS CLOSED. No algorithm passes its gate on implementation
self-consistency. The remaining mandatory prerequisites are records
(this report and the applicability subsections), not new correctness evidence.
Order: correctness → statistical applicability → architecture checkpoint →
profiling → identified bottleneck → measured benefit
([`DEVELOPMENT.md`](DEVELOPMENT.md) §33.3).

## 20. Remaining unresolved questions

**Scope ruling (Architect decision record; recorded 2026-10-05).** #27's mandatory
audit scope is Linear Regression, GRU, and Recursive Least Squares, as enumerated
under Initial algorithms. Decision Tree and KNN coverage in this delivery is
supplementary. Their missing full-audit coverage does not block completion of
#27's original scope. Relevant correctness gates must still be satisfied before
future DT/KNN optimization.

**Review status.** Independent documentation review is complete (recorded in #27
comment 8). Integration of this report and the LR/GRU applicability subsections
remains pending.

**Revision of record.** Audit delivery commit
`b63136f5b5aa19f03a4c34ab7b4c4b863195ed40` (this report and the two applicability
subsections). Original audited checkpoints: `main` @
`cf95cd08b7f6b87b1d116acdbd947bc3ff92c10b` (inspected via git objects; no tests
run) and `rust-development` @
`276fcc5b36a14c78efd90c623970a6b6e98567a8` (inspected; focused tests executed).
Later evidence commits were not covered by the original audit and are recorded
separately: `b63da4f` (Decision Tree, #24), `6a38fec` (RLS, #26), `04e7d1e` (GRU
bounded, #34), and `4b2ae91` (GRU allocated-bytes, #18).

Remaining unresolved questions:

- Development-only bounded GRU/RLS APIs and tests are not on `main`.
- Optimization-time architecture (optimized `P` buffer placement, memory/device
  representation) remains a separate, deferred architecture question.
- Future #27 evidence should state whether it is evaluated against `main` or
  `rust-development`, because bounded GRU/RLS evidence is development-only.
