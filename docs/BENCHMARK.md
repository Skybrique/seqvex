# Measuring an ML Algorithm

A benchmark measures a defined piece of work. Its purpose is to answer a practical question, such as: how long does predicting 400 observations take, and how much memory is needed?

A number is useful only when the reader knows what was done, which model did it and what the number includes. This guide starts with a worked example, then gives the repeatable procedure.

## 1. A complete example before the rules

**Illustrative example: the numbers below explain the method; they are not Seqvex results.**

Suppose a trained regression model predicts salary from years of experience. Its weights are already learned and checked. We want to measure prediction, not CSV reading or training.

Prepare 400 input observations. Put them into 100 groups of four. Run the declared checked prediction operation once for each group. This may be a public group API or a caller-owned collector that invokes single-row prediction. Name the actual API or helper, and state how many model prediction calls each group performs.

| Quantity | Value | Meaning |
|---|---:|---|
| Observations | 400 | Individual inputs whose salaries are predicted |
| Group size | 4 | Inputs passed in one function call |
| Calls | 100 | Number of times the function runs |
| Elapsed time | 40 milliseconds | Total measured time for these calls |
| Time per call | 0.4 milliseconds | 40 divided by 100 |
| Time per observation | 0.1 milliseconds | 40 divided by 400 |

Those last two values describe the same run with different denominators. Comparing one model's time per call with another's time per observation would be wrong.

One repetition of this complete measured block is a **window**. Run some unrecorded warm-up windows first, then retain each measured window. Repeating the whole program creates a separate **process capture**.

If the average is 0.1 milliseconds per observation, that does not establish that every individual observation took 0.1 milliseconds. The window only timed the complete block.

A row label should identify the model and task, for example: "Profile B, checked group prediction, salary fixture, one feature, four observations per call". A date/time cannot replace this identity.

## 2. Terms used in this guide

| Term | Plain meaning |
|---|---|
| Feature / target | Input used for prediction / answer used for learning or evaluation |
| Observation | One input row |
| Fit / update / predict | Learn from a dataset / change learning state / calculate an answer |
| Precision profile | Choice of number formats for learning and prediction; see below |
| Fixture | Fixed data and settings chosen for repeatable checks |
| Baseline / candidate | Accepted starting implementation / proposed changed implementation |
| Operation / call | The exact piece of work being timed, such as one group-prediction call |
| Window | One timed block containing a declared number of operations |
| Warm-up | Repetitions before measurement that prepare code/data execution conditions |
| Capture | Recorded execution of the whole program, including output and exit status |
| Latency | Time taken for a defined request or operation |
| Throughput | Amount of completed work per unit of time |
| Raw result | Original recorded value before division or statistical summary |
| Median / minimum / maximum | Middle result after sorting / smallest / largest |
| Percentile | Value at a declared position in sorted results; p95 is near the 95% position |
| IQR | Interquartile range: difference between the 75th and 25th percentiles |
| Allocation | Request for dynamically managed memory |
| Heap | Memory owned through allocations, such as a vector's separate element buffer |
| Working set / input residency | Data actively accessed / inputs already held in memory |
| Setup boundary | Decision about which preparation is included in the measured operation |
| Instrumentation | Extra counting/timing code used to observe execution |
| Provenance | Evidence linking data, source, build settings, executable and recorded run |
| Hash / fingerprint | Identifier calculated from bytes; useful for detecting changed files |
| Exit status | Program completion code; zero normally means successful completion |
| ns / microsecond / ms | Nanosecond / one thousand ns / one million ns |

### A, B and C

| Profile | Learns using | Predicts using |
|---|---|---|
| A | f64, 64-bit floating-point values | f64 |
| B | The same f64 learning path as A | A separately created f32, 32-bit serving snapshot |
| C | f32 | f32 |

B conversion means checked rounding from the f64 learned parameters to an independently owned f32 model. Its conversion measurement must exclude fitting A when the row is specifically intended to measure conversion alone.

### Separate prediction quality from execution cost

A value measured in seconds can describe the predicted quantity or the computer's running time. Name the quantity as well as its unit.

| Quantity | What it describes | Typical unit |
|---|---|---|
| Mean absolute error (MAE) | Average absolute difference between predictions and targets | Same unit as the target |
| Root mean squared error (RMSE) | Prediction error with greater weight on large differences | Same unit as the target |
| Residual sum of squares | Sum of squared prediction errors | Target unit squared |
| Prediction latency | Time the computer takes for the declared prediction operation | Nanoseconds, microseconds or milliseconds |
| Process resident memory | Process memory currently present in physical RAM | Bytes, KiB or MiB |

**Illustrative example:** a target is 10.0 seconds and a model predicts 10.3 seconds. The absolute prediction error is 0.3 seconds. Suppose the computer took 20 microseconds to calculate that prediction. These are separate quantities; the prediction did not take 0.3 seconds to compute.

The abbreviation **RSS** has two common meanings: **residual sum of squares** in regression and **resident set size** in memory measurement. Spell out the quantity in report headings and output descriptions. Never compare the two as though they measured the same thing.

Keep predictive-quality metrics and execution-cost metrics in separate labelled columns or tables. Choose quality metrics suitable for the task; regression error measures do not apply to every algorithm.

## 3. Plan the measurement

Before running, write down:

1. **Question:** what user-relevant cost is being investigated?
2. **Work:** exact public call, profile, data size, feature count and delivery/learning order.
3. **Inputs:** data source/fingerprint, selected training rows, transformations and probe order. A probe is an input selected to exercise prediction.
4. **Included costs:** preparation, validation, conversion, output creation and destruction.
5. **Repetition:** warm-up windows, measured windows, calls/observations per window and separate program captures.
6. **Results:** metrics, units, summaries and expected records.
7. **Correctness:** guards, references and required preservation checks.
8. **Comparison:** baseline first, candidate second, matched conditions and a meaningful success criterion.
9. **Evidence:** complete records, original files and independent review.

Choose one practical decision per experiment and start with the smallest measurement that can answer it. Retain component and complete-workflow measurements when both are needed to assess the decision. A missing measurement elsewhere does not automatically expand the experiment; state its relevance to the chosen objective.

A guard is a check that stops an invalid experiment, such as detecting non-finite predictions. A successful guard is necessary but does not prove the experiment measures the intended user cost.

## 4. Check the exact model being measured

For serving measurements, learn the model, check it and use that same learned result for prediction. Do not check one fit, discard it, then silently fit another model for measurement.

For **profile B, f64 learning with an f32 serving snapshot**:

1. Record the source parameters and relevant learning state before conversion, including any separately stored intercept and required identity information.
2. Convert with the declared error policy.
3. Verify that source parameters and required learning state did not change. Check the intercept according to the actual representation; it may be a separate value or represented by an input feature.
4. Check that source and converted models contain the same number of weights.
5. Independently round reference parameters to f32 and compare the required stored values.
6. Construct an independent f32 prediction reference using the declared arithmetic order.
7. Compare the converted model's checked predictions on the same inputs.
8. Keep the actual checked converted model for serving measurement.

For A and C, move the checked fitted result into its serving predictor where supported. Moving transfers ownership without necessarily copying data.

Validate benchmark-only controls against production behavior on the intended inputs before using them to explain a cost. Check the full trajectory used by the measurement, including state validity and successful completion. A control that returns an old state or a default value after failure must not count that attempt as a successful update. Report or reject the failure. A small equivalence fixture alone does not establish validity over a longer measured trajectory. Label component-control measurements and entry-point measurements; a component control that omits production work supports only the component claim and is not complete-operation evidence. See [§13 of the failure and recovery guide](FAILURE_AND_RECOVERY.md#13-failure-testing) for the canonical evidence distinction.

Compare applicable single, stream and group predictions using the same ordered inputs. Check final execution state where the API retains one. Require exact stored-bit equality only where that profile/order contract promises it.

Fit timing creates repeated new results. If rank/error metadata is taken from a separate untimed fit using identical data/settings, label it as such. Do not imply it came from every timed result.

## 5. Define what the clock includes

| Task | Normally prepared beforehand when measuring that task alone | Costs that must be explicitly decided |
|---|---|---|
| Fit | Data selection and fixture construction | Public validation, fitting, result creation and destruction |
| B conversion | Checked f64 source model | Conversion checks, f32 model allocation and destruction |
| Prediction | Learned model and required transformation | Checked public call, result handling and output destruction |
| Stream | Inputs and, when declared, execution wrapper | Reset/setup costs, processing and final-state inspection |
| Group | Group construction when declared outside timing | Input borrowing/copying, output allocation and destruction |
| Save/reload | Model and destination setup where declared | Encoding, copying, file/device I/O and restoration checks |

There is no universal correct boundary. The boundary must answer the question and remain matched between baseline and candidate. Also measure preparation plus prediction when the user's task includes both.

Consume measured inputs/results in a way that prevents the compiler from simply removing unused work. Rust black_box is one tool for this purpose; it does not replace correctness checks or inspection of suspicious measurements.

Keep printing and statistical analysis outside timed windows. Do not remove required public validation to improve a number.

## 6. Keep grouping and memory residency comparable

**Illustrative example:** a group benchmark processes 4096 observations. It might:

- Reuse a cycle of 128 stored probes.
- Store 4096 different probes.
- Construct new inputs as it runs.

These are different memory workloads. They can process the same observation count but exercise caches differently. A cache is small, fast memory near a CPU core.

For 128 features stored as f32, 128 probes contain 64 KiB of feature values; 4096 contain 2 MiB. One KiB is 1024 bytes; one MiB is 1024 KiB. These payload figures exclude containers and other data.

Record distinct inputs held in memory, their representation, group ownership and construction/copying boundaries. Reuse is not wrong; claiming it represents fresh-data processing without evidence is wrong.

Ordered RLS/SGD learning must preserve the declared update sequence. A group is not permission to average gradients or reorder updates. Preserve the error behavior too: an unattempted suffix means observations after a failure have not been processed.

## 7. Retain original measurements and validate units

For every timing row retain:

- Unique identity: profile, operation, fixture, dimensions and relevant group size.
- Window index and integer elapsed nanoseconds.
- Calls and processed observations.
- Allocation count/bytes where instrumented.
- Excluded attempts and their reasons.

Check the expected identity set, number of rows and number of values per row. Count alone can hide an omitted row replaced by an unintended one.

Compute:

- Time per call = elapsed time divided by calls.
- Time per observation = elapsed time divided by observations.
- Throughput = completed observations divided by elapsed seconds.

Do not divide by zero for operations such as model conversion that process no observations.

If a report converts between units or denominators, state the conversion and reconcile the converted total to the raw records. Rounding or rescaling must not create, omit or hide a row. See [§11 of the correctness guide](CORRECTNESS.md#11-validation-levels-and-review) for the canonical counting distinction.

Report the summary method. Percentiles of window averages describe window averages, not individual-request tails. A request tail is the slow end of actual individual-request times. Three program runs do not automatically establish stable statistical distributions.

## 8. Explain memory before reporting it

**Illustrative example:** a model owns a 1 KiB weight buffer. Moving it into another owner may allocate zero new bytes while retaining the same 1 KiB. Zero allocation during the move is not zero model memory.

| Metric | Question answered | What it does not tell you |
|---|---|---|
| Allocation count/bytes | How many requests, and how many requested bytes, occurred during the operation? | Memory still owned afterwards |
| Live memory change | How much tracked allocated memory remains above the starting level? | Physical RAM used by the entire process |
| Peak live memory | What was the highest tracked live amount during the probe? | Device/native allocations the counter cannot see |
| Retained model heap | How much allocated memory does the retained model own? | Inline object size or unrelated process memory |
| Post-drop memory change | What remains after the measured result is destroyed? | A universal proof that no leak exists |
| Object size | How large is the inline Rust value? | A vector's separately allocated elements |
| Logical workspace | Buffer sizes calculated from the algorithm's needs | Actual OS memory residency or an enforced physical ceiling |
| Process resident memory (resident set size, RSS) | Process memory currently resident in RAM | Which specific model owns each byte |

A separate **memory probe** is an execution arranged to count memory while retaining the result long enough to inspect it. Keep its setup consistent with the stated operation. Declare whether native libraries, other threads or devices escape the counter.

**Illustrative example — measuring model replacement.** Start peak tracking before fitting the candidate, with the old model still live. Keep tracking through candidate construction and any required serving conversion. Record the starting live memory, peak during replacement, and retained memory after the old model is released. State which allocations the probe can observe.

| Observation | Illustrative result |
|---|---|
| Logical vector payload | 64 `f32` values occupy 256 bytes. |
| Reserved vector storage | Capacity for 256 `f32` values represents 1,024 bytes of element storage. |
| Peak during replacement | Capture tracked live memory while the old model, candidate, and temporary buffers may coexist. |

Counting memory can itself affect timing. Do not compare a counted candidate with an uncounted baseline without accounting for that difference.

## 9. Link each run to the right executable

Record source revision/changed files, lockfile, Rust/compiler version, target platform, enabled features, build profile and effective compiler flags. Record the exact command, full output/error logs, start/end, exit status and executable fingerprint.

A build profile is a named collection of compiler settings, such as bench. A target directory holds compiled files. Use separate build outputs for different candidates or otherwise verify exactly which executable ran.

Reusing the same verified candidate executable across captures is valid. Accidentally running the baseline executable and calling it the candidate is invalid, even if it exits successfully.

Record relevant machine/CPU, core/thread settings and known or unknown power, temperature and background-load conditions. Record any platform, filesystem, permission, privilege or process-isolation prerequisite that affects the measurement; if a required condition cannot be met, record the limitation rather than substituting a different run. Build time is separate from the timed model operation.

A hash identifies bytes; readers also need access to enough methods and results to assess the claim. A locally recorded input hash is a preservation receipt. Compare it with a trusted expected fingerprint and source record before claiming a particular snapshot. A valid different file can support supplied-data evidence when its observed hash, selection and counts are recorded.

## 10. Execute, review and decide

Before selecting commands, inspect Cargo targets and custom harnesses. An all-target test command can execute benchmark binaries registered with a custom main. Confirm which checks are ignored or opt-in, and obtain the applicable execution authorization. Compilation, correctness execution and release measurement remain separate evidence.

1. Freeze the question, protocol, guards and comparison criteria.
2. Run required correctness/build checks.
3. Capture the release baseline before candidate comparison.
4. Run the declared candidate protocol; retain full original records.
5. Retain invalid attempts with exclusion reasons; correct the cause before replacing evidence.
6. Independently check row identities, raw lengths, units, data fingerprints, semantic results and completion.
7. Recompute summaries from raw records.
8. Assess variation, run order and environmental differences.
9. Report gains, regressions and costs moved elsewhere.
10. Obtain independent review and update documentation to match actual execution.

Compiling without running is build evidence. A debug correctness smoke is a short functional check. Neither is a release performance measurement.

Stop the affected comparison when guards fail, the executable is wrong, boundaries/data differ unexpectedly, raw output is incomplete or promised numerical/failure behavior changes.

## 11. Write the evidence record

Use these fields:

- Question and owning issue.
- Baseline/candidate source and executable identities.
- Data, profile, preparation and operation boundaries.
- Warm-ups, windows, captures and counts.
- Raw records and summary calculation.
- Correctness checks and observed outcomes.
- Machine/build conditions and variation.
- Exclusions, comparison conclusion and independent review.

Keep public records understandable without private chat. Remove private machine paths from public copies, while preserving original evidence. Preserve original captures when documentation changes; explain the new documentation version rather than pretending it existed during the run.

## 12. Apply lessons and maintain readability

The OLS profile benchmark required fixes for ambiguous labels, normalized values labelled as raw time, changed input residency, measuring different fitted models, incomplete conversion checks, retained-memory accounting and stale execution statements.

These failures explain why the measurement procedure needs review as well as the algorithm. See [PR 79](https://github.com/Skybrique/seqvex/pull/79) and [issue 46](https://github.com/Skybrique/seqvex/issues/46) for the actual package.

After each assessment, retain reusable demonstrated lessons and remove obsolete instructions. Keep detailed run history, algorithm findings and pending tasks in their owning records. Preserve actual measurement boundaries and conditions. A first-time reader must be able to explain what was measured, reproduce the calculation and distinguish the result from a broader claim.

Use a short reporting pattern. State what was checked, what passed, what remains and the next action.

## Related documents

- [Audit procedures](AUDIT.md)
- [Optimization](OPTIMIZATION.md)
- [Development principles](DEVELOPMENT.md)
- [Feature workflow](../FEATURE_DEVELOPMENT.md)
- [OLS measurement contract](../src/models/linear/ols/README.md)
