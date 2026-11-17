# Auditing an ML Algorithm

An audit is a deliberate check that an algorithm does what its users were promised. In Seqvex, it must answer two questions: can a user complete the supported task, and does the code calculate and manage state correctly?

This guide describes two complementary audits. Both are required for the supported algorithm handover. A successful example alone cannot prove the mathematics, and source inspection alone cannot prove that a user can run the example.

## 1. Start with the vocabulary

| Term | Meaning |
|---|---|
| Observation / row | One example supplied to a model, such as one person's years of experience |
| Feature | An input used to make a prediction, such as years of experience |
| Target / label | The answer available for learning or evaluation, such as salary |
| Parameter / weight | A number learned by the model that controls its predictions |
| Training / learning | Using labelled observations to obtain or update model parameters |
| Inference / serving | Using learned parameters to predict an answer |
| Preprocessing | Preparing input values, for example subtracting a training-set mean |
| State | Information retained between operations; RLS includes more learning state than weights alone |
| Contract | The supported behavior promised by the API and documentation |
| Oracle / reference | An independently justified expected answer used to check a calculation |
| Fixture | A fixed dataset or model setup used for a repeatable check |
| Tolerance | A justified numerical difference allowed when comparing calculated and reference answers |
| Regression | Previously supported behavior that a change breaks |
| f32 / f64 | Floating-point number formats using 32 or 64 bits; f64 usually retains more precision |
| API | The functions and types a user calls |
| Artifact / checkpoint | Saved model data / saved state for continuing work; the contents supported by each API must be stated |
| Candidate | The proposed code change or newly learned model being assessed; state which meaning applies |
| MSRV | Minimum supported Rust version |

### The three precision profiles

A profile describes the precision used for learning and prediction. B is not another training algorithm.

| Profile | Learning | Prediction |
|---|---|---|
| A | f64 inputs and f64 learning arithmetic | f64 parameters and arithmetic |
| B | The same f64 learning path as A | An explicitly converted, independently owned f32 serving model |
| C | f32 inputs and f32 learning arithmetic | f32 parameters and arithmetic |

For B, conversion means rounding the learned f64 parameters into f32 according to the API's checked conversion policy. The source f64 learning state must remain unchanged. The f32 snapshot is a selected version, not a live mirror of subsequent learning.

Learning and delivery are different choices. OLS learns by dataset batch fit/refit. It can still make predictions one row at a time, through a stream, or in bounded groups. RLS and SGD can update in observation order. Delivering a group must preserve the promised learning and prediction order.

## 2. Define the handover boundary

Use **algorithm handover accepted** for the ML-library component. Do not describe it as full operational deployment.

| Phase | What must be delivered |
|---|---|
| Algorithm | Verified mathematics, supported precision profiles, errors, state/ownership behavior and runnable examples |
| Algorithm handover | Both audits passed, accepted optimization outcome, accurate documentation, reproducible evidence and explicit obligations for the next phase |
| Application integration | Real data adapters, feature/preprocessing agreement, model selection, persistence where required and application recovery |
| Full operational deployment | Running service, monitoring, security, capacity, release/rollback process and continuing data/model evaluation |

Freezing weights means selecting parameters that prediction will not update. Deployment also needs the correct feature order, preprocessing, precision and model version. Saving weights alone does not necessarily save everything needed to resume learning.

The supported handover target determines whether save/reload or training continuation is required. Unsupported required behavior blocks that target. An explicitly narrower in-memory target must not be described as durable deployment.

## 3. Prepare either audit

1. Identify the owning issue and the exact user task.
2. Record the code revision, changed-file identities, compiler version, enabled features and dependency lockfile.
3. List supported profiles, learning/prediction modes and platforms.
4. Map each requirement to source, a check, documentation and evidence.
5. Choose normal, boundary and failure cases before evaluating the change.
6. Set expected outcomes and numerical tolerances before running checks.
7. Identify the independent reviewer and preserve previous evidence.

| Requirement | Source/API | Expected outcome | Check/reference | Evidence | Result |
|---|---|---|---|---|---|
| One row per requirement | Actual function or file | Specific behavior or bound | Named test/example/review | Exact candidate and log | Pass, fail, not run, or justified not applicable |

A missing required check is not a pass. Classify each apparent gap as required and missing, already satisfied by applicable evidence, an optional extension, or outside the supported scope. Link that decision to the accepted requirement. Choose additional checks for the affected behavior and its risks. A review of an earlier candidate does not automatically cover the final candidate.

## 4. Audit A: follow the user from CSV to prediction

### A worked task

Illustrative data, not reported Seqvex results:

| YearsExperience | Salary |
|---:|---:|
| 1 | 30 |
| 2 | 40 |
| 3 | 50 |
| 4 | 60 |

The task is to learn from eligible rows and predict salary for a new value of YearsExperience. The walkthrough must show which rows are used for learning, which are held back for evaluation, and what the model actually learns.

The existing Salary example uses these column names, but the actual dataset, units and file availability must be verified. A small educational dataset does not establish forecasting suitability.

### Procedure

| Step | What to do | What to record and check |
|---|---|---|
| A1. Open the data | Give the exact command, file source, column meanings and target units | File fingerprint, row count, column names, successful exit; distinguish supplied data from shipped data |
| A2. Read and validate | Parse the intended number type and reject malformed or unsupported input clearly | Accepted/rejected rows and useful field/row errors; no silent dropping or accidental index column |
| A3. Separate learning and evaluation | Declare selected rows and when features/labels become available | No evaluation leakage; changing future information must not change earlier predictions where time matters |
| A4. Prepare input transformations | Fit scaling or other transformations on eligible training data only | Retained transformation values, feature order and precision; reuse the same transformation for frozen prediction |
| A5. Select A, B or C | Show the actual API calls and resulting types | No implicit downgrade; parse f64 directly for native-f64 paths rather than first rounding to f32 |
| A6. Learn | Fit OLS or perform the declared ordered RLS/SGD updates | Actual learned parameters, settings, independent numerical checks and relevant diagnostics |
| A7. Prepare the checked serving model | Validate the result and use that same result for prediction | No check-then-discard followed by a fresh hidden fit |
| A8. Freeze the selected version | Stop updating that serving version; create B's separate f32 snapshot if selected | Parameter and prediction preservation; continued learning cannot silently modify the frozen snapshot |
| A9. Predict new inputs | Supply features without reading their target labels | Correct transformation/order, finite checked output, equivalent supported delivery paths and final execution state |
| A10. Evaluate | Compare predictions with held-out answers and suitable simple baselines | Predeclared metrics, target units and actual outcomes; no retuning after seeing the held-out answers |
| A11. Try failure cases | Bad shape/value, failed conversion and rejected replacement where applicable | Clear error, preserved active model and promised partial-progress behavior |
| A12. Save/reload where required | Restore exactly the state supported by that format | Compatibility/corruption checks and preserved predictions; distinguish serving restoration from learning continuation |
| A13. Reproduce | Run the walkthrough from its instructions | Exact command/result, understandable sample predictions and accurate documentation |

Before comparing execution paths, check the prepared inputs against the original selected rows. Verify count, feature/target mapping and ordered row identities. If both paths and their expected outputs share preparation code, state that dependency. Agreement between them does not independently verify that preparation retained the correct rows.

Record which checks ordinary CI executes and which require an explicit opt-in command or separately supplied data. For opt-in checks, retain the command, dataset identity, exit status and result. Downloading a file, matching its hash and passing a test establish different facts. Compare against a trusted expected hash before claiming a particular dataset snapshot.

### What to measure

- **MAE, mean absolute error:** average size of prediction errors, in the target's units.
- **RMSE, root mean squared error:** squares errors before averaging and taking the square root, so large errors receive more weight.
- **RSS, residual sum of squares:** sum of squared prediction errors.
- **R-squared:** comparison with predicting the mean; it can be negative and requires care with constant targets.
- **Rank:** number of independent directions in the fitting problem. Record the applied cutoff used to judge independence where the fitter exposes it.
- **State/prediction preservation:** compare the required parameters, identities and outputs before and after an operation.
- **Exit/error behavior:** whether the command succeeds or rejects the intended invalid case correctly.

Select metrics suitable for the task. Do not turn a descriptive number into a pass threshold after seeing the result.

## 5. Audit B: inspect implementation and promised behavior

| Step | What to inspect or execute | What to look for |
|---|---|---|
| B1. Match requirements to code | Read the relevant contract, source and complete change | Missing requirements, partial corrections, stale reports |
| B2. Trace public paths | Follow exports, callers, examples, tests, benchmarks and saved formats | Unexplained compatibility paths or duplicate responsibility; inventory callers before removal |
| B3. Verify mathematics independently | Use small hand calculations, independent references or suitable mathematical properties | Wrong objective/update equation, unfair reference comparison or self-consistency presented as proof |
| B4. Trace precision | Follow input, arithmetic, learning state, parameters, prediction and conversion | Accidental narrowing, non-finite values, overflow/underflow, changed addition order |
| B5. Check data timing | Test selected-row isolation, preprocessing timing and delayed-label/sequence behavior where applicable | Future information affecting earlier predictions |
| B6. Follow ownership | Track the actual checked result, converted snapshot, replacement and drop | Hidden refits, accidental copies and unintended source changes; identify which metadata and policy belong to the library and which belong to the caller; verify named public types and fields exist |
| B7. Trigger failures | Test validation, numerical, conversion and mid-sequence errors at the intended phase | Error order, unchanged state where promised, correct completed prefix and unattempted suffix; platform/privilege/process prerequisites declared; component-control and entry-point evidence distinguished |
| B8. Compare delivery | Check single, stream and groups where supported | Compare learned state at matching boundaries; compare frozen predictions from the same selected state; for adaptive predictions, also match when labels become available and updates are committed |
| B9. Review measurement code | Apply BENCHMARK.md to timing and memory probes | Wrong units, different objects, setup costs moved across boundaries or missing rows |
| B10. Check compatibility | Review signatures, errors, file formats, Rust versions, features and dependencies | Silent default changes or unsupported targets |
| B11. Check maintainability | Assess names, module responsibilities and reversal cost | Unnecessary abstractions, unclear lifetime/failure responsibilities |
| B12. Reconcile final delivery | Match final files, documentation and executed evidence | Old test results attributed to new behavior; source identity confused with execution |
| B13. Review a behavior-preserving refactor | Compare the declared contracts and the approved structural change map | Approved structural changes accounted for; item/export/attribute/comment/visibility/layout/qualified-name/filter mappings, intended selection checks and unresolved external compatibility recorded; comparator limits stated |

A refactor need not preserve identical visibility, module layout, qualified names or filter selections when those changes are approved. It must preserve the declared contracts and account for each approved change. A structural comparator supports the map; it does not establish semantic equivalence or approved scope. See [§28.1 of the development contract](DEVELOPMENT.md#281-behavior-preserving-refactor-evidence).

### Edge-case selection

Choose cases for distinct risks rather than a test-count quota:

- Empty/minimum/ordinary/large input and mismatched dimensions.
- Invalid selected indices and duplicate selection where the contract defines it.
- Zero, negative, constant, non-finite and extreme values.
- Rank deficiency, ill-conditioning and scaling extremes. Ill-conditioning means small input changes can produce large solution changes.
- Conversion values near representable limits, including small values rounded to zero.
- Empty/single/multiple groups and failures before and after progress.
- Candidate alive/dropped, repeated conversion, rejected replacement and learning after a serving snapshot is frozen.
- Workspace limits and recoverable allocation errors only where supported.
- Save/reload mismatches, corruption and continuation where supported.
- Required compiler versions, features and platforms.

Keep numerical comparisons faithful to the declared precision and addition order. Bitwise equality means identical stored bits; require it only where the contract promises that exact behavior.

### RLS deep-dive checklist

1. Derive the update and the effect of the forgetting factor, which controls how strongly older observations contribute.
2. Compare small ordered updates with an independent weighted-objective reference.
3. Check weights and the covariance-related learning state together; weights alone cannot resume RLS learning.
4. Test unexcited input directions, scaling extremes, numerical rejection and promised unchanged-state behavior.
5. Verify grouping does not reorder updates or expose predictions from the wrong state.
6. For B, verify conversion leaves the f64 learning state intact and that snapshot selection is explicit.
7. Run the declared frozen/adaptive CSV evaluation and resource baseline; adaptive evaluation updates after the permitted observation/label event.

### SGD deep-dive checklist

1. Derive the loss gradient and check individual updates and short sequences independently.
2. Verify learning rate, intercept, preprocessing, update order and training budget.
3. Check the declared numerical/objective comparisons and their justified tolerances.
4. Test zero/extreme inputs, overflow, rejected updates and promised state preservation.
5. Compare repeated single updates with ordered groups; averaging gradients at one starting state changes sequential SGD.
6. For B, verify rounding, source preservation and frozen versus current-serving-version behavior.
7. Run the declared frozen/adaptive CSV evaluation and resource baseline; keep their different protocols separate.

Use the applicable checklist when assessing the supported target. Reuse accepted revision-linked evidence where it still applies. Record omissions and their requirement-based disposition in the owning issue.

## 6. Verdict and handover

Use PASS, HOLD or FAIL with the candidate, scope and outstanding findings. No known unresolved in-scope errors or missing required checks may pass the handover gate. Correct rejection of an invalid input is a successful negative test.

### State exactly what was reviewed

Include a coverage record with the verdict:

| Field | Record |
|---|---|
| Source | Exact revision and any reviewed uncommitted files |
| Public behavior | APIs, precision profiles and learning/prediction modes covered |
| Inputs and risks | Data shapes, numerical ranges, normal cases and selected boundary/failure cases |
| Evidence | Independent references, executed checks, logs and reviewer findings; depth proportionate to the change and claim |
| Evidence provenance | Tested revision/tree and raw-log path; author execution, independent review, historical snapshot, scoped acceptance and delivered-source verification kept distinct |
| Counting and units | Unit of each count and per-record reconciliation; any converted total reconciled to the raw records |
| Remaining work | Untested requirements, unsupported capabilities and separate assessments, with their owners |

See [§11 of the correctness guide](CORRECTNESS.md#11-validation-levels-and-review) for the canonical evidence distinctions.

A PASS resolves the stated scope and its identified findings. Use wording such as “no known unresolved finding remains in the reviewed scope.” Do not turn completion of one example, test suite or benchmark package into a claim that every part of the algorithm was audited.

For rejection tests, keep other inputs valid so the intended check is reached. Assert the expected error and useful context. Test competing invalid conditions separately when error precedence matters; an earlier rejection does not exercise a later check.

Record functional acceptance separately from the optimization decision. State whether the supported learning, prediction and failure behavior passed, whether performance investigation remains open, and whether full handover is accepted. A functional PASS does not imply an accepted performance result. Full handover must still satisfy the governing requirements.

Before handover verify:

- Both audits passed and required edge cases have a reasoned coverage record.
- Supported A/B/C paths and any compatibility API have an explicit disposition.
- The runnable walkthrough, feature/transformation obligations and freezing behavior are accurate.
- The final optimization decision is accepted, including retaining the simpler implementation when justified.
- Final code, examples, dependencies, documentation and evidence match.
- Independent review, Maintainer acceptance and required integration checks are complete.

Review affected behavior again after a change. Repeat broader checks when a real dependency or unresolved risk justifies them.

## 7. Keep the guide readable and current

After each assessment, add only evidenced, reusable lessons. Keep algorithm findings, detailed run history and pending tasks in the owning issue or evidence record. Remove obsolete instructions when revising the guide. Retain actual supported boundaries, required input conditions and measurement conditions.

Use a short reporting pattern throughout the audit record. State what was checked, what passed, what remains and the next action.

A first-time reader must be able to identify the task, follow the steps, understand the terms and reproduce the supported outcome. Explain necessary terms inline or in a glossary. Label invented examples as illustrative; label real results with their source. Request a plain-language review alongside technical review.

## Related documents

- [Benchmarking](BENCHMARK.md)
- [Optimization](OPTIMIZATION.md)
- [Development principles](DEVELOPMENT.md)
- [Feature workflow](../FEATURE_DEVELOPMENT.md)
- [OLS](../src/models/linear/ols/README.md), [RLS](../src/models/linear/rls/README.md), [SGD](../src/models/linear/sgd/README.md)
