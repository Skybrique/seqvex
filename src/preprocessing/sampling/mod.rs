//! Training-fold sampling for caller-owned, ordered data.
//!
//! The splitter decides whether a fold is usable and supplies the eligible
//! training rows. This module consumes caller-built [`SampleUnit`]s and returns
//! indices into the caller's data. It does not own observations, inspect the
//! splitter, fit preprocessing, or score labels.
//!
//! Resampling is opt-in. [`SamplingMode::Disabled`] is the identity over the
//! supplied units; the other modes select a method explicitly, and
//! [`SamplingMode::Default`] selects sequential bootstrap when its outcome
//! support is present. Shuffle is an independent presentation control that
//! defaults to `false`.
//!
//! Presentation order is fixed: draw units, order the drawn occurrences into
//! source order within each sequence, permute whole occurrences within each
//! sequence when shuffle is enabled, then expand occurrences into row indices.
//!
//! See `README.md` in this directory for the full contract.

use std::error::Error;
use std::fmt;
use std::num::NonZeroU64;
use std::ops::Range;

use crate::foundation::numerical::RandomGenerator;

mod sequential_f32;

pub use sequential_f32::sample_training_sequential_f32;

/// Exact-integer limit for outcome-span and weight arithmetic.
const OUTCOME_SPAN_LIMIT: u128 = 1_u128 << 53;

/// Bound for the uniform draw used by sequential selection (`[0, 2^53)`).
const SEQUENTIAL_DRAW_BOUND: u64 = 1_u64 << 53;

/// `2^53` as `f64`, exactly representable.
const SEQUENTIAL_DRAW_SCALE: f64 = 9_007_199_254_740_992.0;

/// One contiguous group of eligible caller rows drawn as one unit.
///
/// `rows` is ordered, disjoint, and non-empty. Ranges may be non-adjacent:
/// eligible rows at one decision timestamp separated by an ineligible row stay
/// in one unit and are drawn together.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SampleUnit {
    /// Ordered, disjoint, non-empty caller-row ranges.
    pub rows: Vec<Range<usize>>,
    /// Declared realized-outcome support: ordered, disjoint spans. Empty means
    /// "no outcome metadata".
    pub outcomes: Vec<OutcomeInterval>,
    /// Declared sequence index. Never decreases across units.
    pub sequence: u32,
}

/// Half-open caller-tick outcome interval `[start, end)`.
///
/// Realized-outcome overlap metadata. This is not a label-availability tick and
/// does not establish label maturity.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OutcomeInterval {
    /// Inclusive lower tick.
    pub start: i64,
    /// Exclusive upper tick.
    pub end: i64,
}

/// Which resampling rule to apply.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SamplingMode {
    /// Opt-out: every eligible unit once, in source order unless shuffled.
    Disabled,
    /// Caller omitted a method: sequential bootstrap, or an explicit error.
    Default,
    /// Explicitly selected method.
    Method(SampleMethod),
}

/// Supported resampling rules in the first delivery.
///
/// The without/with-replacement variants are distinct rules, not a boolean
/// flag.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SampleMethod {
    /// As-if-appended average uniqueness; requires outcome support per unit.
    SequentialBootstrap,
    /// Each unit appears at most once; `sample_size <= unit count`.
    WithoutReplacement,
    /// Standard uniform draw with replacement; duplicates retained.
    WithReplacement,
}

/// Data-only sampling configuration. Validated when sampling runs.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SamplingSpec {
    mode: SamplingMode,
    sample_size: usize,
    seed: u64,
    shuffle: bool,
}

impl SamplingSpec {
    /// Stores the configuration; `shuffle` starts disabled.
    ///
    /// `sample_size` is the number of drawn units. It is ignored when the mode
    /// is [`SamplingMode::Disabled`], where the sample is the full eligible
    /// membership once.
    pub fn new(mode: SamplingMode, sample_size: usize, seed: u64) -> Self {
        Self {
            mode,
            sample_size,
            seed,
            shuffle: false,
        }
    }

    /// Resampling disabled, no shuffle. The seed is unused here; use [`new`] with
    /// [`SamplingMode::Disabled`] when a shuffle needs a chosen seed.
    ///
    /// [`new`]: SamplingSpec::new
    #[must_use]
    pub fn disabled() -> Self {
        Self::new(SamplingMode::Disabled, 0, 0)
    }

    /// Independent presentation control; `false` is the default.
    #[must_use]
    pub fn with_shuffle(mut self, shuffle: bool) -> Self {
        self.shuffle = shuffle;
        self
    }
}

/// Per-call resource limits for the strict-`f32` sequential sampler.
///
/// Each field is admitted in `1..=ceiling`; [`Default`] supplies the approved
/// defaults. These are direct-sampler limits for one call, not aggregate bag
/// limits. Use [`sample_training_sequential_f32`] with these limits.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SequentialLimits {
    /// Maximum number of supplied units.
    pub max_units: usize,
    /// Maximum total number of declared outcome spans.
    pub max_support_spans: usize,
    /// Maximum charged raw-endpoint scratch in bytes.
    pub max_endpoint_scratch_bytes: usize,
    /// Maximum expanded rows across the returned sample.
    pub max_expanded_rows: usize,
    /// Maximum weight/ticket work cells.
    pub max_work_cells: u64,
}

impl Default for SequentialLimits {
    fn default() -> Self {
        Self {
            max_units: 65_536,
            max_support_spans: 1_000_000,
            max_endpoint_scratch_bytes: 16_777_216,
            max_expanded_rows: 1_000_000,
            max_work_cells: 1_000_000_000,
        }
    }
}

/// A drawn sample as caller-row indices in presentation order.
///
/// Slice 1 carries no block structure.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TrainingSample {
    row_indices: Vec<usize>,
    unit_counts: Vec<usize>,
}

impl TrainingSample {
    /// Caller-row indices in presentation order; duplicates retained.
    pub fn row_indices(&self) -> &[usize] {
        &self.row_indices
    }

    /// Draw multiplicity per input unit, indexed exactly like the supplied units.
    pub fn unit_counts(&self) -> &[usize] {
        &self.unit_counts
    }

    /// Number of caller-row indices.
    pub fn len(&self) -> usize {
        self.row_indices.len()
    }

    /// Whether the sample contains no rows.
    pub fn is_empty(&self) -> bool {
        self.row_indices.is_empty()
    }
}

/// Validation, capacity, or applicability failure. No draw is performed on error.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SamplingError {
    /// Resampling was enabled with a zero sample size.
    ZeroSampleSize,
    /// No eligible sampling units were supplied.
    NoEligibleUnits,
    /// A unit has no ranges, or one of its ranges is empty.
    EmptyUnitRows {
        /// Unit index.
        unit: usize,
        /// Span index, or `None` when the unit has no ranges.
        span: Option<usize>,
    },
    /// Unit rows are not globally ascending and disjoint.
    UnitRowsOutOfOrder {
        /// Unit containing the offending range.
        unit: usize,
        /// End of the preceding range in global order.
        previous_end: usize,
        /// Start of the offending range.
        start: usize,
    },
    /// Sequence ids decrease.
    SequenceOutOfOrder {
        /// Unit containing the offending id.
        unit: usize,
        /// Previous unit's sequence id.
        previous: u32,
        /// Offending sequence id.
        actual: u32,
    },
    /// An outcome interval is empty or inverted.
    InvalidOutcomeInterval {
        /// Unit index.
        unit: usize,
        /// Span index.
        span: usize,
        /// Supplied start.
        start: i64,
        /// Supplied end.
        end: i64,
    },
    /// Outcome intervals within a unit overlap or are out of order.
    OutcomeOutOfOrder {
        /// Unit index.
        unit: usize,
        /// Span index.
        span: usize,
        /// End of the preceding span.
        previous_end: i64,
        /// Start of the offending span.
        start: i64,
    },
    /// Sequential bootstrap was selected but a unit has no outcome support.
    MissingOutcomeSupport {
        /// Unit index.
        unit: usize,
    },
    /// Without-replacement sampling requested more units than exist.
    SampleSizeExceedsUnits {
        /// Requested sample size.
        requested: usize,
        /// Available units.
        available: usize,
    },
    /// Outcome spans exceed the exact-integer limit.
    OutcomeSpanOverflow {
        /// Computed tick span.
        computed: u128,
        /// Limit (`2^53`).
        limit: u128,
    },
    /// A planned element count exceeds `usize::MAX`.
    SampleLengthOverflow {
        /// Computed element count.
        computed: u128,
        /// Limit (`usize::MAX`).
        limit: u128,
    },
    /// A planned allocation exceeds `isize::MAX` bytes.
    AllocationTooLarge {
        /// Computed bytes.
        bytes: u128,
        /// Limit (`isize::MAX`).
        limit: u128,
    },
    /// The selected mode's work estimate exceeds `u64::MAX`.
    SamplingWorkOverflow {
        /// Computed work cells.
        computed: u128,
        /// Limit (`u64::MAX`).
        limit: u128,
    },
    /// A strict-sampler limit is zero or above its approved ceiling.
    InvalidLimits {
        /// Exact field spelling.
        field: &'static str,
        /// Supplied value.
        value: u128,
        /// Inclusive minimum (`1`).
        min: u128,
        /// Inclusive maximum.
        max: u128,
    },
    /// The supplied mode is not admitted by the strict sampler.
    UnsupportedMode {
        /// Original supplied mode.
        mode: SamplingMode,
    },
    /// The strict draw count exceeds `2^24`.
    F32CountUnsupported {
        /// Requested draw count.
        computed: u128,
        /// Limit (`2^24`).
        limit: u128,
    },
    /// A unit's full support length exceeds `2^24`.
    F32SpanUnsupported {
        /// Unit index.
        unit: usize,
        /// Unit support length.
        computed: u128,
        /// Limit (`2^24`).
        limit: u128,
    },
    /// The unit count exceeds the per-call limit.
    UnitLimitExceeded {
        /// Computed unit count.
        computed: u128,
        /// Configured limit.
        limit: u128,
    },
    /// The total support-span count exceeds the per-call limit.
    SupportSpanLimitExceeded {
        /// Computed span count.
        computed: u128,
        /// Configured limit.
        limit: u128,
    },
    /// The raw-endpoint scratch charge exceeds the per-call limit.
    EndpointScratchExceeded {
        /// Computed bytes.
        computed: u128,
        /// Configured limit.
        limit: u128,
    },
    /// The expanded-row bound exceeds the per-call limit.
    ExpandedRowsExceeded {
        /// Computed rows.
        computed: u128,
        /// Configured limit.
        limit: u128,
    },
    /// The sequential work bound exceeds the per-call limit.
    WorkLimitExceeded {
        /// Computed work cells.
        computed: u128,
        /// Configured limit.
        limit: u128,
    },
    /// A strict formula overflowed its checked widened arithmetic.
    ArithmeticOverflow {
        /// Formula name.
        bound: &'static str,
    },
    /// A stored strict weight was zero before ticket conversion.
    F32WeightUnderflow {
        /// Zero-based draw index.
        draw_index: usize,
        /// Unit index.
        unit: usize,
    },
    /// A strict ticket calculation or checked total overflowed.
    F32WeightQuantumOverflow {
        /// Zero-based draw index.
        draw_index: usize,
        /// Unit index.
        unit: usize,
    },
}

impl fmt::Display for SamplingError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroSampleSize => formatter.write_str("sample size must be positive"),
            Self::NoEligibleUnits => formatter.write_str("no eligible sampling units"),
            Self::EmptyUnitRows { unit, span } => match span {
                Some(span) => write!(formatter, "unit {unit} span {span} has empty row range"),
                None => write!(formatter, "unit {unit} has no row ranges"),
            },
            Self::UnitRowsOutOfOrder {
                unit,
                previous_end,
                start,
            } => write!(
                formatter,
                "unit {unit} row range starts at {start} before previous end {previous_end}"
            ),
            Self::SequenceOutOfOrder {
                unit,
                previous,
                actual,
            } => write!(
                formatter,
                "unit {unit} sequence {actual} follows {previous}"
            ),
            Self::InvalidOutcomeInterval {
                unit,
                span,
                start,
                end,
            } => write!(
                formatter,
                "unit {unit} span {span} outcome interval [{start}, {end}) is invalid"
            ),
            Self::OutcomeOutOfOrder {
                unit,
                span,
                previous_end,
                start,
            } => write!(
                formatter,
                "unit {unit} span {span} starts at {start} before previous end {previous_end}"
            ),
            Self::MissingOutcomeSupport { unit } => {
                write!(formatter, "unit {unit} has no outcome support")
            }
            Self::SampleSizeExceedsUnits {
                requested,
                available,
            } => write!(
                formatter,
                "sample size {requested} exceeds {available} available units"
            ),
            Self::OutcomeSpanOverflow { computed, limit } => write!(
                formatter,
                "outcome span {computed} ticks exceeds limit {limit}"
            ),
            Self::SampleLengthOverflow { computed, limit } => write!(
                formatter,
                "sample length {computed} elements exceeds limit {limit}"
            ),
            Self::AllocationTooLarge { bytes, limit } => {
                write!(
                    formatter,
                    "planned allocation {bytes} bytes exceeds limit {limit}"
                )
            }
            Self::SamplingWorkOverflow { computed, limit } => write!(
                formatter,
                "sampling work {computed} cells exceeds limit {limit}"
            ),
            Self::InvalidLimits {
                field,
                value,
                min,
                max,
            } => write!(
                formatter,
                "invalid {field} limit {value}; expected {min}..={max}"
            ),
            Self::UnsupportedMode { mode } => {
                write!(
                    formatter,
                    "sampling mode {mode:?} is not supported by the strict f32 sampler"
                )
            }
            Self::F32CountUnsupported { computed, limit } => write!(
                formatter,
                "strict f32 draw count {computed} exceeds limit {limit}"
            ),
            Self::F32SpanUnsupported {
                unit,
                computed,
                limit,
            } => write!(
                formatter,
                "unit {unit} strict f32 support span {computed} exceeds limit {limit}"
            ),
            Self::UnitLimitExceeded { computed, limit } => {
                write!(formatter, "unit count {computed} exceeds limit {limit}")
            }
            Self::SupportSpanLimitExceeded { computed, limit } => write!(
                formatter,
                "support span count {computed} exceeds limit {limit}"
            ),
            Self::EndpointScratchExceeded { computed, limit } => write!(
                formatter,
                "endpoint scratch {computed} bytes exceeds limit {limit}"
            ),
            Self::ExpandedRowsExceeded { computed, limit } => {
                write!(formatter, "expanded rows {computed} exceeds limit {limit}")
            }
            Self::WorkLimitExceeded { computed, limit } => {
                write!(formatter, "work {computed} cells exceeds limit {limit}")
            }
            Self::ArithmeticOverflow { bound } => {
                write!(formatter, "arithmetic overflow computing {bound}")
            }
            Self::F32WeightUnderflow { draw_index, unit } => write!(
                formatter,
                "strict f32 weight underflow at draw {draw_index} unit {unit}"
            ),
            Self::F32WeightQuantumOverflow { draw_index, unit } => write!(
                formatter,
                "strict f32 ticket quantum overflow at draw {draw_index} unit {unit}"
            ),
        }
    }
}

impl Error for SamplingError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        None
    }
}

/// Draws one training sample from caller-supplied eligible units.
///
/// Returns indices into caller-owned data. No data is copied, reordered, or
/// mutated; the splitter, fold cutoffs, diagnostics, and held-out membership are
/// untouched.
pub fn sample_training(
    units: &[SampleUnit],
    spec: SamplingSpec,
) -> Result<TrainingSample, SamplingError> {
    let mut plan = preflight(units, spec)?;
    let mut generator = RandomGenerator::from_seed(spec.seed);
    let mut next_below = |upper: u64| {
        generator.draw_u64_below(NonZeroU64::new(upper).expect("draw bounds are positive"))
    };
    Ok(draw(units, spec, &mut plan, &mut next_below))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Method {
    Disabled,
    WithoutReplacement,
    WithReplacement,
    Sequential,
}

fn resolve(mode: SamplingMode) -> Method {
    match mode {
        SamplingMode::Disabled => Method::Disabled,
        SamplingMode::Default => Method::Sequential,
        SamplingMode::Method(method) => match method {
            SampleMethod::SequentialBootstrap => Method::Sequential,
            SampleMethod::WithoutReplacement => Method::WithoutReplacement,
            SampleMethod::WithReplacement => Method::WithReplacement,
        },
    }
}

struct Plan {
    method: Method,
    sample_size: usize,
    output_bound: usize,
    unit_tick_len: Vec<u64>,
    unit_span_ranges: Vec<Range<usize>>,
    span_segments: Vec<(usize, usize)>,
    endpoints: Vec<i64>,
    segment_counts: Vec<usize>,
    weights: Vec<f64>,
}

fn preflight(units: &[SampleUnit], spec: SamplingSpec) -> Result<Plan, SamplingError> {
    let method = resolve(spec.mode);

    if method != Method::Disabled && spec.sample_size == 0 {
        return Err(SamplingError::ZeroSampleSize);
    }
    if units.is_empty() {
        return Err(SamplingError::NoEligibleUnits);
    }

    // Class: empty unit rows (whole input, lowest unit then span).
    for (unit_index, unit) in units.iter().enumerate() {
        if unit.rows.is_empty() {
            return Err(SamplingError::EmptyUnitRows {
                unit: unit_index,
                span: None,
            });
        }
        if let Some((span, _)) = unit
            .rows
            .iter()
            .enumerate()
            .find(|(_, range)| range.start >= range.end)
        {
            return Err(SamplingError::EmptyUnitRows {
                unit: unit_index,
                span: Some(span),
            });
        }
    }

    // Class: unit row ordering, within and across units.
    let mut previous_unit_end = None;
    for (unit_index, unit) in units.iter().enumerate() {
        if let Some(previous_end) = previous_unit_end {
            let first = &unit.rows[0];
            if first.start < previous_end {
                return Err(SamplingError::UnitRowsOutOfOrder {
                    unit: unit_index,
                    previous_end,
                    start: first.start,
                });
            }
        }
        for span in 1..unit.rows.len() {
            let previous = &unit.rows[span - 1];
            let current = &unit.rows[span];
            if current.start < previous.end {
                return Err(SamplingError::UnitRowsOutOfOrder {
                    unit: unit_index,
                    previous_end: previous.end,
                    start: current.start,
                });
            }
        }
        previous_unit_end = unit.rows.last().map(|range| range.end);
    }

    // Class: sequence order.
    for unit_index in 1..units.len() {
        let previous = units[unit_index - 1].sequence;
        let actual = units[unit_index].sequence;
        if actual < previous {
            return Err(SamplingError::SequenceOutOfOrder {
                unit: unit_index,
                previous,
                actual,
            });
        }
    }

    // Class: outcome shape, then outcome order.
    for (unit_index, unit) in units.iter().enumerate() {
        for (span, outcome) in unit.outcomes.iter().enumerate() {
            if outcome.start >= outcome.end {
                return Err(SamplingError::InvalidOutcomeInterval {
                    unit: unit_index,
                    span,
                    start: outcome.start,
                    end: outcome.end,
                });
            }
        }
    }
    for (unit_index, unit) in units.iter().enumerate() {
        for span in 1..unit.outcomes.len() {
            let previous = &unit.outcomes[span - 1];
            let current = &unit.outcomes[span];
            if current.start < previous.end {
                return Err(SamplingError::OutcomeOutOfOrder {
                    unit: unit_index,
                    span,
                    previous_end: previous.end,
                    start: current.start,
                });
            }
        }
    }

    // Class: applicability.
    if method == Method::Sequential {
        if let Some((unit, _)) = units
            .iter()
            .enumerate()
            .find(|(_, unit)| unit.outcomes.is_empty())
        {
            return Err(SamplingError::MissingOutcomeSupport { unit });
        }
    }

    // Class: draw space.
    if method == Method::WithoutReplacement && spec.sample_size > units.len() {
        return Err(SamplingError::SampleSizeExceedsUnits {
            requested: spec.sample_size,
            available: units.len(),
        });
    }

    // Class: outcome span bound.
    let mut total_span: u128 = 0;
    for unit in units {
        let mut unit_span: u128 = 0;
        for outcome in &unit.outcomes {
            unit_span += (outcome.end as i128 - outcome.start as i128) as u128;
        }
        if unit_span > OUTCOME_SPAN_LIMIT {
            return Err(SamplingError::OutcomeSpanOverflow {
                computed: unit_span,
                limit: OUTCOME_SPAN_LIMIT,
            });
        }
        total_span += unit_span;
    }
    if total_span > OUTCOME_SPAN_LIMIT {
        return Err(SamplingError::OutcomeSpanOverflow {
            computed: total_span,
            limit: OUTCOME_SPAN_LIMIT,
        });
    }

    // Class: input-derived capacity and work.
    let span_count: u128 = units.iter().map(|unit| unit.outcomes.len() as u128).sum();
    let max_unit_rows = units
        .iter()
        .map(|unit| {
            unit.rows
                .iter()
                .map(|range| (range.end - range.start) as u128)
                .sum::<u128>()
        })
        .max()
        .unwrap_or(0);
    let output_bound_u128 = if method == Method::Disabled {
        units
            .iter()
            .map(|unit| {
                unit.rows
                    .iter()
                    .map(|range| (range.end - range.start) as u128)
                    .sum::<u128>()
            })
            .sum::<u128>()
    } else {
        (spec.sample_size as u128) * max_unit_rows
    };

    let output_bound = ensure_length(output_bound_u128)?;
    ensure_bytes::<usize>(output_bound_u128)?;
    ensure_bytes::<usize>(units.len() as u128)?;
    if method == Method::Sequential {
        ensure_bytes::<f64>(units.len() as u128)?;
        ensure_bytes::<u64>(units.len() as u128)?;
        ensure_bytes::<Range<usize>>(units.len() as u128)?;
        ensure_bytes::<(usize, usize)>(span_count)?;
    }
    if method == Method::WithoutReplacement {
        ensure_bytes::<usize>(units.len() as u128)?;
    }
    if spec.shuffle {
        let occurrences = if method == Method::Disabled {
            units.len()
        } else {
            spec.sample_size
        };
        ensure_bytes::<usize>(occurrences as u128)?;
    }
    if method != Method::Sequential {
        let work = match method {
            Method::Disabled => units.len() as u128 + output_bound_u128,
            Method::WithoutReplacement => {
                units.len() as u128 + spec.sample_size as u128 + output_bound_u128
            }
            Method::WithReplacement => spec.sample_size as u128 + output_bound_u128,
            Method::Sequential => unreachable!("sequential work is checked after compression"),
        };
        if work > u64::MAX as u128 {
            return Err(SamplingError::SamplingWorkOverflow {
                computed: work,
                limit: u64::MAX as u128,
            });
        }
    }

    let mut plan = Plan {
        method,
        sample_size: spec.sample_size,
        output_bound,
        unit_tick_len: Vec::new(),
        unit_span_ranges: Vec::new(),
        span_segments: Vec::new(),
        endpoints: Vec::new(),
        segment_counts: Vec::new(),
        weights: Vec::new(),
    };
    if method == Method::Sequential {
        prepare_sequential(units, &mut plan)?;
    }
    Ok(plan)
}

fn prepare_sequential(units: &[SampleUnit], plan: &mut Plan) -> Result<(), SamplingError> {
    let span_count: usize = units.iter().map(|unit| unit.outcomes.len()).sum();
    let raw_len = checked_raw_endpoints(span_count as u128)?;

    let mut endpoints: Vec<i64> = Vec::with_capacity(raw_len);
    for unit in units {
        for outcome in &unit.outcomes {
            endpoints.push(outcome.start);
            endpoints.push(outcome.end);
        }
    }
    endpoints.sort_unstable();
    endpoints.dedup();

    let endpoint_count = endpoints.len();
    let segment_count = checked_segments(endpoint_count as u128)?;

    let work = (plan.sample_size as u128)
        .checked_mul(units.len() as u128)
        .and_then(|value| value.checked_mul(endpoint_count as u128))
        .ok_or(SamplingError::SamplingWorkOverflow {
            computed: u128::MAX,
            limit: u64::MAX as u128,
        })?;
    if work > u64::MAX as u128 {
        return Err(SamplingError::SamplingWorkOverflow {
            computed: work,
            limit: u64::MAX as u128,
        });
    }

    plan.segment_counts = vec![0_usize; segment_count];
    plan.weights = vec![0.0_f64; units.len()];
    plan.endpoints = endpoints;
    plan.unit_tick_len = Vec::with_capacity(units.len());
    plan.unit_span_ranges = Vec::with_capacity(units.len());
    plan.span_segments = Vec::with_capacity(span_count);

    for unit in units {
        let span_start = plan.span_segments.len();
        let mut ticks: u128 = 0;
        for outcome in &unit.outcomes {
            let start_index = plan
                .endpoints
                .binary_search(&outcome.start)
                .expect("declared start is an endpoint");
            let end_index = plan
                .endpoints
                .binary_search(&outcome.end)
                .expect("declared end is an endpoint");
            plan.span_segments.push((start_index, end_index));
            ticks += (outcome.end as i128 - outcome.start as i128) as u128;
        }
        plan.unit_span_ranges
            .push(span_start..plan.span_segments.len());
        plan.unit_tick_len
            .push(u64::try_from(ticks).expect("span bound fits u64"));
    }
    Ok(())
}

fn draw(
    units: &[SampleUnit],
    spec: SamplingSpec,
    plan: &mut Plan,
    next_below: &mut impl FnMut(u64) -> u64,
) -> TrainingSample {
    let mut counts = vec![0_usize; units.len()];
    match plan.method {
        Method::Disabled => {
            counts.fill(1);
        }
        Method::WithoutReplacement => {
            let mut pool: Vec<usize> = (0..units.len()).collect();
            for position in 0..plan.sample_size {
                let remaining = units.len() - position;
                let offset = next_below(remaining as u64) as usize;
                let swap = position + offset;
                pool.swap(position, swap);
                counts[pool[position]] += 1;
            }
        }
        Method::WithReplacement => {
            for _ in 0..plan.sample_size {
                let index = next_below(units.len() as u64) as usize;
                counts[index] += 1;
            }
        }
        Method::Sequential => {
            for _ in 0..plan.sample_size {
                let unit_value = (next_below(SEQUENTIAL_DRAW_BOUND) as f64) / SEQUENTIAL_DRAW_SCALE;
                let chosen = sequential_draw(plan, unit_value);
                counts[chosen] += 1;
            }
        }
    }

    let mut rows = Vec::with_capacity(plan.output_bound);
    if spec.shuffle {
        let occurrences_total: usize = counts.iter().sum();
        let mut occurrences = Vec::with_capacity(occurrences_total);
        let mut first_unit = 0;
        while first_unit < units.len() {
            let sequence = units[first_unit].sequence;
            let mut next_unit = first_unit + 1;
            while next_unit < units.len() && units[next_unit].sequence == sequence {
                next_unit += 1;
            }
            let occurrence_start = occurrences.len();
            for (offset, &count) in counts[first_unit..next_unit].iter().enumerate() {
                let unit_index = first_unit + offset;
                for _ in 0..count {
                    occurrences.push(unit_index);
                }
            }
            let occurrence_end = occurrences.len();
            shuffle(
                &mut occurrences[occurrence_start..occurrence_end],
                next_below,
            );
            first_unit = next_unit;
        }
        for unit_index in occurrences {
            push_unit_rows(&mut rows, &units[unit_index]);
        }
    } else {
        for (unit_index, &count) in counts.iter().enumerate() {
            for _ in 0..count {
                push_unit_rows(&mut rows, &units[unit_index]);
            }
        }
    }

    TrainingSample {
        row_indices: rows,
        unit_counts: counts,
    }
}

fn push_unit_rows(rows: &mut Vec<usize>, unit: &SampleUnit) {
    for range in &unit.rows {
        rows.extend(range.clone());
    }
}

fn shuffle(occurrences: &mut [usize], next_below: &mut impl FnMut(u64) -> u64) {
    for position in (1..occurrences.len()).rev() {
        let offset = next_below((position + 1) as u64) as usize;
        occurrences.swap(position, offset);
    }
}

fn sequential_draw(plan: &mut Plan, unit_value: f64) -> usize {
    let Plan {
        unit_tick_len,
        unit_span_ranges,
        span_segments,
        endpoints,
        segment_counts,
        weights,
        ..
    } = plan;

    for (unit_index, &ticks) in unit_tick_len.iter().enumerate() {
        let mut total = 0.0;
        for span in unit_span_ranges[unit_index].clone() {
            let (start_segment, end_segment) = span_segments[span];
            for segment in start_segment..end_segment {
                let length = (endpoints[segment + 1] as i128 - endpoints[segment] as i128) as f64;
                let concurrency = segment_counts[segment] as f64;
                total += length / (1.0 + concurrency);
            }
        }
        weights[unit_index] = total / ticks as f64;
    }

    let chosen = select_weighted(weights, unit_value);
    for span in unit_span_ranges[chosen].clone() {
        let (start_segment, end_segment) = span_segments[span];
        for count in &mut segment_counts[start_segment..end_segment] {
            *count = count.saturating_add(1);
        }
    }
    chosen
}

fn select_weighted(weights: &[f64], unit_value: f64) -> usize {
    let total: f64 = weights.iter().copied().sum();
    let target = unit_value * total;
    let mut cumulative = 0.0;
    let mut last_positive = 0;
    for (index, weight) in weights.iter().copied().enumerate() {
        if weight > 0.0 {
            last_positive = index;
        }
        cumulative += weight;
        if cumulative > target {
            return index;
        }
    }
    last_positive
}

fn ensure_length(computed: u128) -> Result<usize, SamplingError> {
    let limit = usize::MAX as u128;
    if computed > limit {
        return Err(SamplingError::SampleLengthOverflow { computed, limit });
    }
    Ok(computed as usize)
}

fn checked_raw_endpoints(span_count: u128) -> Result<usize, SamplingError> {
    let limit = usize::MAX as u128;
    let computed = span_count
        .checked_mul(2)
        .ok_or(SamplingError::SampleLengthOverflow {
            computed: u128::MAX,
            limit,
        })?;
    let len = ensure_length(computed)?;
    ensure_bytes::<i64>(computed)?;
    Ok(len)
}

fn checked_segments(endpoint_count: u128) -> Result<usize, SamplingError> {
    let computed = endpoint_count.saturating_sub(1);
    let len = ensure_length(computed)?;
    ensure_bytes::<usize>(computed)?;
    Ok(len)
}

fn ensure_bytes<T>(elements: u128) -> Result<(), SamplingError> {
    let size = size_of::<T>() as u128;
    let bytes = elements
        .checked_mul(size)
        .ok_or(SamplingError::AllocationTooLarge {
            bytes: u128::MAX,
            limit: isize::MAX as u128,
        })?;
    let limit = isize::MAX as u128;
    if bytes > limit {
        return Err(SamplingError::AllocationTooLarge { bytes, limit });
    }
    Ok(())
}

#[cfg(test)]
#[allow(clippy::single_range_in_vec_init)]
mod tests {
    use super::*;

    fn unit(rows: &[Range<usize>], outcomes: &[(i64, i64)], sequence: u32) -> SampleUnit {
        SampleUnit {
            rows: rows.to_vec(),
            outcomes: outcomes
                .iter()
                .map(|&(start, end)| OutcomeInterval { start, end })
                .collect(),
            sequence,
        }
    }

    fn single(rows: usize, outcomes: &[(i64, i64)], sequence: u32) -> SampleUnit {
        unit(&[rows..rows + 1], outcomes, sequence)
    }

    fn scripted(values: &[u64]) -> impl FnMut(u64) -> u64 + '_ {
        let mut values = values.iter().copied();
        move |upper| {
            let value = values.next().expect("scripted draw");
            assert!(value < upper, "scripted value {value} outside [0, {upper})");
            value
        }
    }

    fn gather(units: &[SampleUnit], spec: SamplingSpec, values: &[u64]) -> TrainingSample {
        sample_with_next(units, spec, &mut scripted(values)).expect("scripted sampling")
    }

    fn sample_with_next(
        units: &[SampleUnit],
        spec: SamplingSpec,
        next_below: &mut impl FnMut(u64) -> u64,
    ) -> Result<TrainingSample, SamplingError> {
        let mut plan = preflight(units, spec)?;
        Ok(draw(units, spec, &mut plan, next_below))
    }

    fn gather_error(units: &[SampleUnit], spec: SamplingSpec) -> SamplingError {
        sample_with_next(units, spec, &mut |_| panic!("no draws on error")).unwrap_err()
    }

    fn ticks(value: u64) -> u64 {
        value
    }

    #[test]
    fn public_types_carry_the_contract_derives() {
        fn assert_debug_clone_eq<T: fmt::Debug + Clone + PartialEq>() {}
        fn assert_copy<T: Copy>() {}
        assert_debug_clone_eq::<SampleUnit>();
        assert_debug_clone_eq::<OutcomeInterval>();
        assert_debug_clone_eq::<TrainingSample>();
        assert_debug_clone_eq::<SamplingMode>();
        assert_debug_clone_eq::<SampleMethod>();
        assert_debug_clone_eq::<SamplingSpec>();
        assert_debug_clone_eq::<SamplingError>();
        assert_copy::<OutcomeInterval>();
        assert_copy::<SamplingMode>();
        assert_copy::<SampleMethod>();
        assert_copy::<SamplingSpec>();
        assert_copy::<SamplingError>();
    }

    #[test]
    fn identity_keeps_source_order_without_draws() {
        let units = [
            single(0, &[(1, 3)], 0),
            single(1, &[(2, 4)], 0),
            single(2, &[(6, 8)], 0),
        ];
        let sample = gather(&units, SamplingSpec::disabled(), &[]);
        assert_eq!(sample.row_indices(), &[0, 1, 2]);
        assert_eq!(sample.unit_counts(), &[1, 1, 1]);
    }

    #[test]
    fn identity_shuffle_permutes_one_sequence_and_consumes_two_draws() {
        let units = [
            single(0, &[(1, 3)], 0),
            single(1, &[(2, 4)], 0),
            single(2, &[(6, 8)], 0),
        ];
        let mut calls = 0;
        let sample = sample_with_next(
            &units,
            SamplingSpec::disabled().with_shuffle(true),
            &mut |upper| {
                calls += 1;
                upper - 1
            },
        )
        .unwrap();
        assert_eq!(calls, 2);
        assert_eq!(sample.unit_counts(), &[1, 1, 1]);
        let mut rows = sample.row_indices().to_vec();
        rows.sort_unstable();
        assert_eq!(rows, vec![0, 1, 2]);
    }

    #[test]
    fn grouped_multiplicity_expands_every_row_of_each_draw() {
        let units = [single(0, &[], 0), unit(&[1..3], &[], 0)];
        let sample = gather(
            &units,
            SamplingSpec::new(SamplingMode::Method(SampleMethod::WithReplacement), 3, 1),
            &[0, 1, 1],
        );
        assert_eq!(sample.row_indices(), &[0, 1, 2, 1, 2]);
        assert_eq!(sample.unit_counts(), &[1, 2]);
    }

    #[test]
    fn without_replacement_full_and_scripted() {
        let units = [single(0, &[], 0), single(1, &[], 0), single(2, &[], 0)];
        let sample = gather(
            &units,
            SamplingSpec::new(SamplingMode::Method(SampleMethod::WithoutReplacement), 3, 1),
            &[0, 0, 0],
        );
        assert_eq!(sample.row_indices(), &[0, 1, 2]);
        assert_eq!(sample.unit_counts(), &[1, 1, 1]);

        let sample = gather(
            &units,
            SamplingSpec::new(SamplingMode::Method(SampleMethod::WithoutReplacement), 2, 1),
            &[2, 1],
        );
        assert_eq!(sample.row_indices(), &[0, 2]);
        assert_eq!(sample.unit_counts(), &[1, 0, 1]);
    }

    #[test]
    fn with_replacement_scripted_and_overflow() {
        let units = [single(0, &[], 0), single(1, &[], 0), single(2, &[], 0)];
        let sample = gather(
            &units,
            SamplingSpec::new(SamplingMode::Method(SampleMethod::WithReplacement), 3, 1),
            &[0, 1, 1],
        );
        assert_eq!(sample.row_indices(), &[0, 1, 1]);
        assert_eq!(sample.unit_counts(), &[1, 2, 0]);

        assert_eq!(
            gather_error(
                &units,
                SamplingSpec::new(SamplingMode::Method(SampleMethod::WithoutReplacement), 4, 1),
            ),
            SamplingError::SampleSizeExceedsUnits {
                requested: 4,
                available: 3,
            }
        );
    }

    #[test]
    fn validation_precedence_is_class_first() {
        assert_eq!(
            gather_error(&[], SamplingSpec::new(SamplingMode::Default, 0, 1)),
            SamplingError::ZeroSampleSize
        );
        assert_eq!(
            gather_error(&[], SamplingSpec::disabled()),
            SamplingError::NoEligibleUnits
        );
        assert_eq!(
            gather_error(&[unit(&[], &[], 0)], SamplingSpec::disabled()),
            SamplingError::EmptyUnitRows {
                unit: 0,
                span: None,
            }
        );
        assert_eq!(
            gather_error(&[unit(&[2..2], &[], 0)], SamplingSpec::disabled()),
            SamplingError::EmptyUnitRows {
                unit: 0,
                span: Some(0),
            }
        );
        assert_eq!(
            gather_error(
                &[unit(&[3..4], &[], 0), unit(&[2..3], &[], 0)],
                SamplingSpec::disabled()
            ),
            SamplingError::UnitRowsOutOfOrder {
                unit: 1,
                previous_end: 4,
                start: 2,
            }
        );
        assert_eq!(
            gather_error(
                &[unit(&[0..1], &[], 1), unit(&[1..2], &[], 0)],
                SamplingSpec::disabled()
            ),
            SamplingError::SequenceOutOfOrder {
                unit: 1,
                previous: 1,
                actual: 0,
            }
        );
        assert_eq!(
            gather_error(&[unit(&[0..1], &[(5, 5)], 0)], SamplingSpec::disabled()),
            SamplingError::InvalidOutcomeInterval {
                unit: 0,
                span: 0,
                start: 5,
                end: 5,
            }
        );
        assert_eq!(
            gather_error(
                &[unit(&[0..1], &[(1, 4), (3, 6)], 0)],
                SamplingSpec::disabled()
            ),
            SamplingError::OutcomeOutOfOrder {
                unit: 0,
                span: 1,
                previous_end: 4,
                start: 3,
            }
        );
        assert_eq!(
            gather_error(
                &[unit(&[0..1], &[], 0)],
                SamplingSpec::new(SamplingMode::Default, 1, 1)
            ),
            SamplingError::MissingOutcomeSupport { unit: 0 }
        );
        assert_eq!(
            gather_error(
                &[unit(&[3..4], &[], 0), unit(&[2..3], &[], 0)],
                SamplingSpec::new(SamplingMode::Default, 1, 1)
            ),
            SamplingError::UnitRowsOutOfOrder {
                unit: 1,
                previous_end: 4,
                start: 2,
            }
        );
    }

    #[test]
    fn outcome_span_bound_is_checked() {
        let units = [unit(&[0..1], &[(0, (1_i64 << 53) + 1)], 0)];
        assert_eq!(
            gather_error(&units, SamplingSpec::disabled()),
            SamplingError::OutcomeSpanOverflow {
                computed: (1_u128 << 53) + 1,
                limit: 1_u128 << 53,
            }
        );
    }

    #[test]
    fn sequential_partial_overlap_matches_expected_draws() {
        let units = [
            single(0, &[(1, 3)], 0),
            single(1, &[(2, 4)], 0),
            single(2, &[(6, 8)], 0),
        ];
        let spec = SamplingSpec::new(SamplingMode::Default, 4, 1);
        let sample = gather(
            &units,
            spec,
            &[0, ticks(1 << 51), ticks(1 << 50), ticks(5 << 49)],
        );
        assert_eq!(sample.row_indices(), &[0, 0, 1, 1]);
        assert_eq!(sample.unit_counts(), &[2, 2, 0]);
    }

    #[test]
    fn sequential_complete_overlap_and_touching() {
        let identical = [single(0, &[(1, 3)], 0), single(1, &[(1, 3)], 0)];
        let sample = gather(
            &identical,
            SamplingSpec::new(
                SamplingMode::Method(SampleMethod::SequentialBootstrap),
                3,
                1,
            ),
            &[0, ticks(7 << 50), ticks(3 << 50)],
        );
        assert_eq!(sample.unit_counts(), &[2, 1]);

        let touching = [single(0, &[(1, 3)], 0), single(1, &[(3, 5)], 0)];
        let sample = gather(
            &touching,
            SamplingSpec::new(SamplingMode::Default, 3, 1),
            &[0, ticks(1 << 52), ticks(7 << 50)],
        );
        assert_eq!(sample.unit_counts(), &[1, 2]);
    }

    #[test]
    fn sequential_fragmented_union_discriminates_bounding_interval() {
        let units = [
            unit(&[0..1, 2..3], &[(1, 3), (5, 7)], 0),
            single(3, &[(1, 7)], 0),
            single(4, &[(10, 12)], 0),
        ];
        let sample = gather(
            &units,
            SamplingSpec::new(SamplingMode::Default, 3, 1),
            &[0, ticks(5 << 49), ticks(101 << 44)],
        );
        assert_eq!(sample.unit_counts(), &[1, 2, 0]);
        assert_eq!(sample.row_indices(), &[0, 2, 3, 3]);
    }

    #[test]
    fn sequential_repeated_selection_updates_average_overlap() {
        let units = [
            single(0, &[(1, 3)], 0),
            single(1, &[(2, 4)], 0),
            single(2, &[(6, 8)], 0),
        ];
        let sample = gather(
            &units,
            SamplingSpec::new(SamplingMode::Default, 5, 1),
            &[
                0,
                ticks(1 << 51),
                ticks(1 << 50),
                ticks(5 << 49),
                ticks(1 << 51),
            ],
        );
        assert_eq!(sample.row_indices(), &[0, 0, 1, 1, 1]);
        assert_eq!(sample.unit_counts(), &[2, 3, 0]);
    }

    #[test]
    fn shuffle_moves_whole_occurrences_within_sequences() {
        let units = [
            unit(&[0..1], &[], 0),
            unit(&[1..3], &[], 0),
            unit(&[3..4], &[], 1),
        ];
        let spec = SamplingSpec::new(SamplingMode::Method(SampleMethod::WithReplacement), 3, 1)
            .with_shuffle(true);
        let script = [(3_u64, 0_u64), (3, 1), (3, 2), (2, 0)];
        let mut draws = script.into_iter();
        let mut bounds = Vec::new();
        let sample = sample_with_next(&units, spec, &mut |upper| {
            bounds.push(upper);
            let (expected, value) = draws.next().expect("scripted draw");
            assert_eq!(upper, expected, "draw bound");
            value
        })
        .unwrap();
        assert!(draws.next().is_none(), "unused scripted draws");
        assert_eq!(bounds, vec![3, 3, 3, 2]);
        assert_eq!(sample.row_indices(), &[1, 2, 0, 3]);
        assert_eq!(sample.unit_counts(), &[1, 1, 1]);
    }

    #[test]
    fn shuffle_single_occurrence_consumes_no_draw() {
        let units = [single(0, &[], 0)];
        let mut calls = 0;
        let sample = sample_with_next(
            &units,
            SamplingSpec::disabled().with_shuffle(true),
            &mut |upper| {
                calls += 1;
                upper - 1
            },
        )
        .unwrap();
        assert_eq!(calls, 0);
        assert_eq!(sample.row_indices(), &[0]);
    }

    #[test]
    fn weighted_selection_is_total_at_boundaries() {
        let u = ((1_u64 << 53) - 1) as f64 / (1_f64 * (1_u64 << 53) as f64);
        assert_eq!(select_weighted(&[0.1; 10], u), 9);
        assert_eq!(select_weighted(&[0.5, 0.0, 0.5], u), 2);
        assert_eq!(select_weighted(&[0.25, 0.75, 0.0, 0.0], u), 1);
        assert_eq!(select_weighted(&[0.5, 0.5], 1.0), 1);
    }

    #[cfg(target_pointer_width = "64")]
    #[test]
    fn sequential_work_overflow_draws_nothing() {
        let units = [
            single(0, &[(1, 3)], 0),
            single(1, &[(5, 7)], 0),
            single(2, &[(9, 11)], 0),
            single(3, &[(13, 15)], 0),
        ];
        let spec = SamplingSpec::new(SamplingMode::Default, 1_usize << 59, 1);
        let mut calls = 0;
        let error = sample_with_next(&units, spec, &mut |_| {
            calls += 1;
            0
        })
        .unwrap_err();
        assert_eq!(calls, 0);
        assert!(matches!(
            error,
            SamplingError::SamplingWorkOverflow { computed, limit }
                if computed == 1_u128 << 64 && limit == u64::MAX as u128
        ));
    }

    #[test]
    fn capacity_guards_cover_raw_and_segment_buffers() {
        assert!(matches!(
            checked_raw_endpoints((usize::MAX as u128) / 2 + 1),
            Err(SamplingError::SampleLengthOverflow { .. })
        ));
        assert!(matches!(
            checked_raw_endpoints((isize::MAX as u128) / 2),
            Err(SamplingError::AllocationTooLarge { .. })
        ));
        assert!(matches!(
            checked_segments(usize::MAX as u128 + 2),
            Err(SamplingError::SampleLengthOverflow { .. })
        ));
        let segment_boundary = (isize::MAX as u128) / (size_of::<usize>() as u128) + 2;
        assert!(matches!(
            checked_segments(segment_boundary),
            Err(SamplingError::AllocationTooLarge { .. })
        ));
    }

    #[test]
    fn every_typed_failure_consumes_zero_draws() {
        let cases: Vec<(Vec<SampleUnit>, SamplingSpec)> = vec![
            (vec![], SamplingSpec::new(SamplingMode::Default, 0, 1)),
            (vec![], SamplingSpec::disabled()),
            (vec![unit(&[], &[], 0)], SamplingSpec::disabled()),
            (
                vec![unit(&[3..4], &[], 0), unit(&[2..3], &[], 0)],
                SamplingSpec::disabled(),
            ),
            (
                vec![unit(&[0..1], &[], 1), unit(&[1..2], &[], 0)],
                SamplingSpec::disabled(),
            ),
            (vec![unit(&[0..1], &[(5, 5)], 0)], SamplingSpec::disabled()),
            (
                vec![unit(&[0..1], &[(1, 4), (3, 6)], 0)],
                SamplingSpec::disabled(),
            ),
            (
                vec![unit(&[0..1], &[], 0)],
                SamplingSpec::new(SamplingMode::Default, 1, 1),
            ),
            (
                vec![single(0, &[], 0)],
                SamplingSpec::new(SamplingMode::Method(SampleMethod::WithoutReplacement), 2, 1),
            ),
            (
                vec![unit(&[0..1], &[(0, (1_i64 << 53) + 1)], 0)],
                SamplingSpec::disabled(),
            ),
            (
                vec![unit(&[0..(usize::MAX / 2)], &[], 0)],
                SamplingSpec::disabled(),
            ),
            (
                vec![single(0, &[], 0), single(1, &[], 0)],
                SamplingSpec::new(
                    SamplingMode::Method(SampleMethod::WithReplacement),
                    usize::MAX,
                    1,
                ),
            ),
        ];
        for (units, spec) in cases {
            let mut calls = 0;
            let result = sample_with_next(&units, spec, &mut |_| {
                calls += 1;
                0
            });
            assert!(result.is_err(), "expected an error for {spec:?}");
            assert_eq!(calls, 0);
        }
    }
}
