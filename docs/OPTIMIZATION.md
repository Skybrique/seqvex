# Optimizing an ML Algorithm

Optimization means making a verified implementation meet its performance or resource objective while preserving its promised behavior.

Requirements, mathematics, implementation, the user workflow and audits establish the trustworthy starting point. Investigation and measurement can reveal questions that require an earlier design or correctness decision to be revisited. Production changes follow the applicable review and implementation gates. Final algorithm handover requires the accepted functional and optimization outcomes. Optimization can conclude that the simplest implementation should remain unchanged.

## 1. Understand the terms

| Term | Meaning |
|---|---|
| Baseline / candidate | Accepted starting implementation / proposed changed implementation |
| Bottleneck | Cost that limits the particular workload being investigated |
| Profiling | Measuring where execution time or resources are spent |
| CPU | General-purpose processor |
| Core / thread | A processing unit / a software execution sequence scheduled on processing units |
| Cache | Small, fast memory near CPU cores that keeps copies of recently or nearby accessed data |
| Cache line | Block of neighboring bytes transferred and tracked together in cache; size depends on the machine |
| Locality | Accessing nearby values or reusing recently used values |
| Working set | Data actively needed by an operation |
| Memory latency / bandwidth | Delay before data arrives / rate at which data can be transferred |
| Contiguous / stride | Adjacent in memory / distance between successive accesses |
| SIMD | Single instruction, multiple data: one instruction operates on several number lanes |
| Auto-vectorization | Compiler transformation that generates vector instructions from ordinary code |
| Rayon | Rust library for scheduling suitable work across worker threads |
| NUMA | Non-uniform memory access: memory access cost depends on the processor/memory location |
| GPU / accelerator | A processor designed for particular kinds of parallel or specialized work |
| Kernel | Small computation under study; on a GPU, a function executed on the device |
| Crate | A Rust library or program package |
| Compiler flag / build profile | Setting passed to the compiler / named collection of build settings |
| MSRV / ISA | Minimum supported Rust version / instruction set supported by a processor |
| Ownership | Responsibility for a value and its eventual destruction in Rust |
| Allocation / retained heap | Request for dynamic memory / allocated memory still owned by a result |
| Numerical contract | Promised precision, arithmetic order, accepted error and numerical failure behavior |
| Synchronization / contention | Coordination between threads / competing access that causes waiting |

The precision profiles used throughout Seqvex are:

| Profile | Learning | Serving predictions |
|---|---|---|
| A | f64, 64-bit floating-point values | f64 |
| B | The same f64 learning path as A | Explicitly converted, separately owned f32 snapshot |
| C | f32, 32-bit floating-point values | f32 |

B's snapshot is not complete resumable learning state. RLS, for example, needs its covariance-related information as well as weights to continue learning.

## 2. Enter optimization only with a clean supported target

Apply this gate to the supported target and behavior affected by a proposed production change. Record which requirements and existing evidence apply. Classify apparent gaps before adding work. Required unresolved defects or checks block the affected change; optional coverage needs a stated reason to become a prerequisite. A focused measurement can proceed when its own correctness controls and supported inputs are sufficient.

Before proposing a production change:

1. Define the user objective, supported profiles, learning/prediction modes and platforms.
2. Pass both audits in AUDIT.md: user workflow and implementation/contract.
3. Resolve known in-scope errors and missing required checks.
4. Cover reasonable normal, boundary and failure cases selected for distinct risks.
5. Pass the required compiler-version, feature and compatibility checks.
6. Validate the benchmark procedure and record a release baseline.
7. Identify the limiting cost or the specific measured resource objective.
8. Agree on scope, independent checks, success criteria and reversal cost.

Expected errors correctly rejected by the API are successful negative tests. An actual defect cannot be renamed a future enhancement to bypass this gate.

If a correctness error is found, correct it and revalidate before continuing the optimization comparison. A bug fix is not automatically evidence of faster execution.

A behavior-preserving refactor or file split is maintainability work, not an optimization, unless it has an explicit performance or resource objective with before/after measurement. Refactoring alone establishes no improvement and must not be recorded as a speedup. Preservation evidence for approved structural changes belongs to the audit; performance evidence remains subject to this gate. This applies the canonical rule in [§33.3 of the development contract](DEVELOPMENT.md#333-optimization-gate).

The gate is risk-based coverage, not a claim that every possible input has been tested. The accepted target and its required checks must be explicit.

## 3. Follow the optimization ladder

| Stage | Investigate first | Move further only when |
|---|---|---|
| 1. General engineering | Repeated work, copying, unnecessary intermediate data and avoidable data movement | A meaningful objective remains |
| 2. Rust, crates and compiler | Ownership, appropriate safe facilities, justified dependencies and generated code | Evidence identifies a remaining CPU/memory cost |
| 3. One-core CPU work | Cache/locality, layout, bandwidth, alignment and SIMD | The remaining work can legally and usefully run across threads |
| 4. Multiple cores/processors | Partitioning, Rayon where suitable, synchronization and NUMA | The complete workload justifies a different backend |
| 5. GPU and other accelerators | Data transfers, device execution, memory, synchronization and whole-workflow cost | A measured and reviewed result supports adoption |

Memory layout is software engineering that affects hardware behavior; investigate it early. Compiler auto-vectorization may already occur in stage 2. Explicit SIMD belongs before adding Rayon worker threads.

Single-core means controlling the operation to one execution thread/core and checking for hidden dependency threads. A single CPU socket can contain many cores. Multiple sockets and multiple machines are different scopes; distributed processing introduces separate communication and failure concerns.

Architecture review happens before ownership, representation or backend choices constrain future work. It does not wait until the last stage.

## 4. Engineering improvements before specialized hardware

Trace the user's actual path through parsing, input preparation, learning, serving preparation and prediction. Select the operation or complete workflow named by the objective, and check where repeated work occurs. Start with the smallest valid measurement that can support the decision. Expand to other phases when they can change that decision.

Examples to assess when measured:

- Repeating a transformation that can safely be prepared once.
- Copying a model merely to pass it to a read-only call.
- Constructing large temporary input collections that the user path does not need.
- Reallocating growing buffers where a known size and ownership contract permit a simpler construction.
- Fitting a model inside an operation intended to measure conversion or prediction.

Do not remove required validation or cache an outdated serving version. Buffer reuse can change ownership, failure recovery and simultaneous-use behavior; it needs a design rather than a blanket instruction to reuse everything.

Changing the mathematical method, objective or update rule is a separate behavior change. A faster result from doing different work is not a transparent optimization.

## 5. Rust, crates and compiler work

### Ownership and code clarity

Investigate measured cloning, temporary collections, conversions and repeated traversals. Borrow when read-only access is sufficient; move when ownership should transfer. Neither choice is universally correct.

An iterator is a convenient way to traverse values. An explicit loop states the traversal directly. Compare understandable implementations and generated code rather than treating either syntax as automatically faster.

A small-buffer crate stores some values inline and may move to heap storage when capacity is exceeded. Assess actual capacities, initialization, spill behavior, object/stack size, error behavior and measured benefit before selecting it.

### Crate assessment

Compare the current code, standard library and candidate crate against a concrete consumer. Keep a dependency decision record:

| Check | What to record |
|---|---|
| Purpose and baseline | The operation that needs improvement, current implementation and option of retaining it |
| Exact version | Published version or pinned source revision, enabled features, lockfile and benchmark identity |
| Maintenance | Relevant fixes, issue/review responses, active maintainers and release practices; repository age alone does not measure maintenance quality |
| Fix availability | Whether required fixes are present in the chosen release or only on an unreleased branch |
| Correctness and failure behavior | Relevant open defects and advisories, boundary behavior, ordering, conversion and compatibility checks |
| Supported environments | Minimum Rust version, platforms, standard-library requirements, dependency tree and license |
| Integration cost | Adapter code, ownership, initialization, allocation/threading behavior, build cost and future upgrade work |
| Measured trade-offs | Whole-operation time, allocation traffic, retained/peak memory, object/stack size and any costs moved elsewhere |
| Exit path | How to retain or restore the baseline, and the cost of removing the dependency |

A mature crate can need few changes because its responsibility is stable. Assess relevant maintenance evidence rather than treating age, stars or commit frequency as a quality score. Distinguish adopting an upstream dependency from maintaining a private fork; a fork creates additional responsibility for fixes and compatibility.

For inline-storage candidates, test below, at and above the chosen inline capacity. Inline storage holds elements inside the container; crossing that capacity may require a separate allocation or rejection, depending on the type. Include initialization, growth, copying and destruction in the appropriate operation boundary.

Do not introduce a container, hash function or solver merely because it is advertised as fast. First establish a real consumer and preserve ordering, numeric and saved-model identity requirements.

### Build settings and generated instructions

Rust's compiler translates source into machine instructions. Inspect its optimized output when necessary to determine whether the relevant loop became vector code. Do not infer this from source syntax alone.

Candidate settings include CPU target, permitted instruction features, optimization level, link-time optimization and code-generation units. Link-time optimization lets the compiler optimize across compiled units; code-generation units divide compilation work.

Use the exact compiler's help and official documentation:
- [rustc code-generation options](https://doc.rust-lang.org/rustc/codegen-options/)
- [Cargo build profiles](https://doc.rust-lang.org/cargo/reference/profiles.html)

In particular, target-cpu=native specializes to the build host. Do not make it a default for binaries distributed to different CPUs. Higher optimization settings do not guarantee faster execution.

For each configuration experiment, preserve the portable baseline, record effective flags, build into identifiable outputs, run required correctness checks on that configuration, and compare runtime/build/size trade-offs. Check option availability on the supported compiler versions.

Do not weaken floating-point addition order or error guarantees to enable vectorization. A different grouping of additions can change rounded results.

## 6. One-core CPU optimization: more than SIMD

Investigate what the processor waits for before selecting a technique.

| Area | Plain explanation | Coding/design candidate | Evidence to collect |
|---|---|---|---|
| Spatial locality | Values used together are stored near each other | Contiguous arrays and traversal matching their order | Time, cache misses and total data movement |
| Temporal locality | Recently used values are reused before leaving cache | Reuse a small tile of data before moving on | Small/large working-set comparisons and cache behavior |
| Cache blocking / tiling | Split a large operation into reusable smaller pieces | Process a matrix in blocks when mathematically permitted | Whole-operation time, extra buffers and unchanged results |
| AoS versus SoA | Group values by record or by field; example below | Select representation matching the access pattern | Packing cost, useful bytes accessed, kernel and end-to-end time |
| Row/column order and stride | Consecutive accesses may be adjacent or widely separated | Match traversal to stored rows/columns | Access pattern, cache behavior and conversion cost |
| Alignment and padding | Arrange addresses or add spacing where required/useful | Compare safe supported representations | Actual generated instructions, footprint and timing |
| Memory bandwidth | Data-transfer rate can limit calculation | Remove measured unnecessary reads/copies; reuse useful data | Bytes moved, achieved rate and whole-operation time |
| Pointer chasing | Follow addresses to separately stored objects | Compare an approved contiguous representation | Allocation/working-set changes and measured delay |
| Branch behavior | The CPU guesses which conditional path runs next | Simplify measured unpredictable hot decisions without removing checks | Branch-miss evidence and timing |
| Instruction work | Too many instructions or dependencies can limit a loop | Remove duplicated work; inspect generated code | Instructions/cycles and elapsed time where tools support them |
| Page translation | Hardware translates program addresses into physical locations | Investigate excessive large/fragmented working sets | Translation-cache misses and actual latency |
| Prefetching | Fetch data before it is needed | First use predictable access; consider explicit prefetch only for an evidenced need | Reduced waiting balanced against added memory traffic |
| SIMD | Compute several independent lanes per instruction | Inspect compiler vectorization before explicit vector code | Correctness including tails, actual instructions and total time |

A cache miss means needed data was not found in a particular cache. A hardware counter counts events such as misses or instructions; its meaning and availability depend on CPU/tool. Counters help explain a result but do not replace elapsed-time evidence.

The translation cache is commonly called the TLB, translation lookaside buffer. Large-page or placement experiments need their own platform and memory-cost assessment; they are not default recommendations.

### AoS and SoA with visible data

**Illustrative example:** three observations have two features:

| Observation | Experience | TrainingHours |
|---|---:|---:|
| 1 | 1 | 10 |
| 2 | 2 | 20 |
| 3 | 3 | 30 |

**AoS, array of structures:** store complete records together:

```text
records = [(1, 10), (2, 20), (3, 30)]
```

**SoA, structure of arrays:** store each field in its own array:

```text
experience     = [1, 2, 3]
training_hours = [10, 20, 30]
```

If a calculation needs only experience across many observations, SoA places those useful values together. If prediction needs both features for one observation at a time, record-oriented storage may be appropriate. Measure the actual operation, including converting between layouts.

In Rust, these choices may resemble Vec<Record> versus separate Vec fields. Vec<Vec<f32>> stores rows in separately allocated buffers, while a flat Vec<f32> can store adjacent rows with explicit indexing. These are illustrative representations, not selected Seqvex changes. Preserve dimension checks, input order, ownership and error behavior in any migration.

Memory packing means copying values into the arrangement a computation expects. Include its cost; a faster inner loop may lose its benefit if packing repeats for every call.

### SIMD procedure

1. Identify the measured loop and inspect existing auto-vectorization.
2. Keep the scalar reference: code processing one value at a time.
3. Declare instruction support and selection for the supported CPUs.
4. Verify independent lanes, arithmetic order and error behavior.
5. Test below/equal/above vector width and non-multiple lengths. The leftover values are the tail.
6. Compare scalar, auto-vectorized and explicit candidates on matched inputs/builds.
7. Measure the public operation and any packing/setup overhead.
8. Keep only worthwhile, understandable code consistent with the project's unsafe-code rules.

## 7. Multicore, Rayon and multiple sockets

Multiple threads add coordination, data sharing and scheduling costs. First prove that the work can be divided without changing the algorithm.

| Area | Meaning | Check |
|---|---|---|
| Partitioning | Dividing work between threads | Required ordering and state dependencies remain correct |
| Thread pool | Reused worker threads | Caller-owned/global pool behavior, limits and dependency interactions are explicit |
| False sharing | Threads modify different values in the same cache line and cause unnecessary cache coordination | Inspect the sharing pattern; assess changed layout/spacing against added memory use |
| Lock contention | Threads wait for the same protected resource | Measure waiting and consider reviewed ownership changes |
| Oversubscription | More active worker threads than useful CPU capacity | Include nested libraries and competing workloads |
| NUMA locality | Some memory is closer to particular processor groups | Measure placement and local/remote access; do not assume sockets map directly to nodes |
| Scaling | Benefit as threads/cores increase | Compare one thread, intermediate counts and the intended limit |

A thread pool used by Rayon does not establish which pool a solver dependency actually uses. Inspect that integration.

For ordered RLS/SGD learning, one update changes the state used by the next. Computing all observations from one starting state changes the learning rule. Independent models may be a different parallelization opportunity.

Preserve fail-stop behavior: if failure leaves later observations unattempted, do not speculatively process them and discard the results. Output reordering does not repair that contract.

Measure scheduling, synchronization, output assembly, extra memory and whole-operation latency/throughput. Keep the sequential reference. Multiple sockets require placement evidence; multiple machines require a separate distributed-system design.

## 8. GPU and other accelerators

A GPU can execute many suitable operations in parallel, but data and commands must reach it and results must return. Evaluate the whole workflow:

1. Which computation and data sizes justify the device?
2. Which data can remain on the device between calls?
3. What time is spent preparing, transferring, launching, synchronizing and returning results?
4. Which precision, shapes, errors and saved-state behavior are supported?
5. Does the result match the approved numerical reference?
6. What device memory, build, driver and maintenance costs are introduced?
7. Does the user-facing result improve over the accepted CPU implementation?

Synchronization here means waiting for required work to complete before using its result. Kernel-only time excludes other costs and cannot establish complete workflow acceleration.

Other specialized devices need the same analysis. No universal batch-size threshold or automatic backend-selection policy follows from this guide.

## 9. Plan, compare and keep or revert

Before implementing one candidate, record:

- Observed problem, workload and source/build/run identities.
- Proposed cause and alternative explanations.
- Smallest change and the option of keeping the baseline.
- Exact affected files, numerical/ownership/failure behavior and compatibility.
- Correctness checks and risk-based edge cases.
- Baseline-first measurement method and meaningful success criterion.
- Maintenance cost, architectural decisions, independent review and reversal plan.

Validate benchmark-only controls before using their decomposition to select a change. They must match the intended production calculation on the declared inputs, preserve the comparison's stated assumptions and expose failures. Verify that every measured transition succeeds; silent reuse of an old state invalidates a claimed successful-update comparison.

Change one interpretable factor at a time where practical. Follow BENCHMARK.md. Report separate outcomes for time, throughput, allocation traffic, retained/peak memory and numerical behavior. Do not hide smaller-workload regressions in one average.

Stop when a correctness discrepancy, invalid measurement or unreviewed architecture change appears. Fix the cause rather than widening tolerances. Inconclusive evidence remains inconclusive.

## 10. Keep an evidence ledger

| Field | What the reader needs |
|---|---|
| Observed problem | What was directly verified or measured |
| Proposed explanation | Why it might occur and how that explanation will be tested |
| Planned/implemented change | Intended scope and actual final change |
| Verification | Functional, numerical, failure and compatibility outcomes |
| Measured result | Original records, comparison, variation and conditions |
| Decision | Keep, revert, retain baseline or investigate further |
| Review and handover | Independent verdict, final documentation and next-phase obligations |

Keep detailed experiment records in the owning issue or evidence location. Add reusable lessons here only after assessment. The OLS measurement packages established measurement boundaries and a profile-labelled baseline; they did not establish a production speedup.

Apply the relevant deep-dive checks in AUDIT.md and link accepted evidence before recording an optimization outcome. Keep current algorithm findings and pending assessments in their owning records. Crates, CPU flags, layout, SIMD, Rayon and accelerators are investigation areas, not preselected changes.

## 11. Complete the stage and maintain the guides

Complete optimization when the supported target has no known unresolved in-scope errors, affected checks passed, the measured decision is accepted and final code/evidence/documentation reconcile.

Hand over the supported profiles, learning and frozen-serving behavior, feature/preprocessing obligations, platform/resource conditions and recovery limits. AUDIT.md defines this algorithm boundary separately from full operational deployment.

After each assessment, refine the relevant guide with demonstrated reusable lessons. Remove obsolete instructions and repetition while retaining actual supported conditions. Explain necessary terms and use worked examples where they reduce ambiguity.

Use a short reporting pattern. State what was checked, what passed, what remains and the next action.

Stopping before optional SIMD, Rayon or GPU work is a valid decision when the accepted objective is met. A required unmet performance budget remains a real gap.

## Further reading

- [Intel cache optimization guidance](https://www.intel.com/content/www/us/en/docs/ipp/developer-guide-reference/2021-11/cache-optimizations.html)
- [Linux kernel explanation of false sharing](https://www.kernel.org/doc/html/latest/kernel-hacking/false-sharing.html)
- [Linux kernel explanation of NUMA](https://docs.kernel.org/mm/numa.html)
- [Intel optimization reference manual](https://cdrdv2-public.intel.com/821612/248966-Optimization-Reference-Manual-V1-050.pdf)

## Related documents

- [Audits and handover](AUDIT.md)
- [Benchmark procedure](BENCHMARK.md)
- [Development principles](DEVELOPMENT.md)
- [Feature workflow](../FEATURE_DEVELOPMENT.md)
