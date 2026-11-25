//! Reusable regression bagging for Decision Tree and OLS.

use std::fmt;

use crate::foundation::numerical::{Vector, Vector64};
use crate::foundation::observation::Observation;
use crate::preprocessing::sampling::{
    SampleMethod, SampleUnit, SamplingError, SamplingMode, SamplingSpec, TrainingSample,
    sample_training,
};

mod decision_tree;
mod mean;
mod ols;

pub use decision_tree::{DecisionTreeBag, DecisionTreeBag64, DecisionTreeBagError};
pub use mean::MeanError;
pub use ols::{OlsBag, OlsBag64, OlsBagError};

/// Domain separator for bagging member seeds.
pub const BAGGING_MEMBER_DOMAIN: u64 = 0x1060_0000_0000_0001;

const DEFAULT_MAX_MEMBERS: usize = 64;
const MAX_MEMBERS_CEILING: usize = 1024;
const DEFAULT_MAX_EXPANDED_ROWS: usize = 1_000_000;
const MAX_EXPANDED_ROWS_CEILING: usize = 16_777_216;
const DEFAULT_MAX_UNITS: usize = 65_536;
const MAX_UNITS_CEILING: usize = 1_048_576;
const DEFAULT_MAX_SUPPORT_SPANS: usize = 1_000_000;
const MAX_SUPPORT_SPANS_CEILING: usize = 16_777_216;
const DEFAULT_MAX_ENDPOINT_SCRATCH: usize = 16 * 1024 * 1024;
const MAX_ENDPOINT_SCRATCH_CEILING: usize = 256 * 1024 * 1024;
const DEFAULT_MAX_SEQUENTIAL_WORK: u64 = 1_000_000_000;
const MAX_SEQUENTIAL_WORK_CEILING: u64 = 1_000_000_000_000;
const DEFAULT_MAX_PREDICTIONS: usize = 65_536;
const MAX_PREDICTIONS_CEILING: usize = 1_048_576;
const DEFAULT_MAX_RETAINED_BYTES: usize = 64 * 1024 * 1024;
const MAX_RETAINED_BYTES_CEILING: usize = 1024 * 1024 * 1024;

/// Bag-level resource limits.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BaggingLimits {
    pub max_members: usize,
    pub max_expanded_rows_per_member: usize,
    pub max_units: usize,
    pub max_support_spans: usize,
    pub max_endpoint_scratch_bytes: usize,
    pub max_total_sequential_work: u64,
    pub max_predictions: usize,
    pub max_retained_bytes: usize,
}

impl Default for BaggingLimits {
    fn default() -> Self {
        Self {
            max_members: DEFAULT_MAX_MEMBERS,
            max_expanded_rows_per_member: DEFAULT_MAX_EXPANDED_ROWS,
            max_units: DEFAULT_MAX_UNITS,
            max_support_spans: DEFAULT_MAX_SUPPORT_SPANS,
            max_endpoint_scratch_bytes: DEFAULT_MAX_ENDPOINT_SCRATCH,
            max_total_sequential_work: DEFAULT_MAX_SEQUENTIAL_WORK,
            max_predictions: DEFAULT_MAX_PREDICTIONS,
            max_retained_bytes: DEFAULT_MAX_RETAINED_BYTES,
        }
    }
}

impl BaggingLimits {
    /// Validates defaults/ceilings.
    pub fn validate(&self) -> Result<(), BaggingError> {
        self.check_nonzero("max_members", self.max_members, MAX_MEMBERS_CEILING)?;
        self.check_nonzero(
            "max_expanded_rows_per_member",
            self.max_expanded_rows_per_member,
            MAX_EXPANDED_ROWS_CEILING,
        )?;
        self.check_nonzero("max_units", self.max_units, MAX_UNITS_CEILING)?;
        self.check_nonzero(
            "max_support_spans",
            self.max_support_spans,
            MAX_SUPPORT_SPANS_CEILING,
        )?;
        self.check_nonzero(
            "max_endpoint_scratch_bytes",
            self.max_endpoint_scratch_bytes,
            MAX_ENDPOINT_SCRATCH_CEILING,
        )?;
        self.check_nonzero(
            "max_predictions",
            self.max_predictions,
            MAX_PREDICTIONS_CEILING,
        )?;
        self.check_nonzero(
            "max_retained_bytes",
            self.max_retained_bytes,
            MAX_RETAINED_BYTES_CEILING,
        )?;
        if self.max_total_sequential_work == 0
            || self.max_total_sequential_work > MAX_SEQUENTIAL_WORK_CEILING
        {
            return Err(BaggingError::InvalidLimits {
                field: "max_total_sequential_work",
                value: self.max_total_sequential_work as usize,
                limit: MAX_SEQUENTIAL_WORK_CEILING as usize,
            });
        }
        Ok(())
    }

    fn check_nonzero(
        &self,
        field: &'static str,
        value: usize,
        limit: usize,
    ) -> Result<(), BaggingError> {
        if value == 0 || value > limit {
            return Err(BaggingError::InvalidLimits {
                field,
                value,
                limit,
            });
        }
        Ok(())
    }
}

/// A bag fit request.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BaggingSpec {
    members: usize,
    mode: SamplingMode,
    sample_size: usize,
    base_seed: u64,
    fold_id: usize,
    support_cutoff: Option<i64>,
    limits: BaggingLimits,
}

impl BaggingSpec {
    /// Creates a spec; validation happens during fit.
    pub fn new(
        members: usize,
        mode: SamplingMode,
        sample_size: usize,
        base_seed: u64,
        fold_id: usize,
    ) -> Self {
        Self {
            members,
            mode,
            sample_size,
            base_seed,
            fold_id,
            support_cutoff: None,
            limits: BaggingLimits::default(),
        }
    }

    #[must_use]
    pub fn with_limits(mut self, limits: BaggingLimits) -> Self {
        self.limits = limits;
        self
    }

    #[must_use]
    pub fn with_support_cutoff(mut self, cutoff: Option<i64>) -> Self {
        self.support_cutoff = cutoff;
        self
    }

    pub fn members(&self) -> usize {
        self.members
    }

    pub fn mode(&self) -> SamplingMode {
        self.mode
    }

    pub fn sample_size(&self) -> usize {
        self.sample_size
    }

    pub fn limits(&self) -> &BaggingLimits {
        &self.limits
    }

    /// Derived member seed for replay/audit.
    pub fn member_seed(&self, member: usize) -> u64 {
        let packed = ((self.fold_id as u64) << 32) | member as u64;
        BAGGING_MEMBER_DOMAIN ^ self.base_seed ^ packed
    }
}

/// Shared bagging failure.
#[derive(Debug, Clone, PartialEq)]
pub enum BaggingError {
    InvalidLimits {
        field: &'static str,
        value: usize,
        limit: usize,
    },
    ZeroMembers,
    TooManyMembers {
        members: usize,
        limit: usize,
    },
    FoldIdTooLarge {
        fold_id: usize,
        limit: usize,
    },
    MemberIdTooLarge {
        member: usize,
        limit: usize,
    },
    UnsupportedMode {
        mode: SamplingMode,
    },
    MultipleSequences {
        first: u32,
        next: u32,
    },
    DataLengthMismatch {
        features: usize,
        targets: usize,
    },
    InvalidUnitRange {
        unit: usize,
        span: Option<usize>,
        start: usize,
        end: usize,
    },
    SourceRowOutOfRange {
        unit: usize,
        row: usize,
        len: usize,
    },
    SupportBeyondCutoff {
        unit: usize,
        span: usize,
        end: i64,
        cutoff: i64,
    },
    UnitLimitExceeded {
        units: usize,
        limit: usize,
    },
    SupportSpanLimitExceeded {
        spans: usize,
        limit: usize,
    },
    EndpointScratchExceeded {
        bytes: usize,
        limit: usize,
    },
    ExpandedRowsExceeded {
        rows: usize,
        limit: usize,
    },
    TotalSequentialWorkExceeded {
        computed: u128,
        limit: u64,
    },
    Sampling(SamplingError),
    MemberDimensionMismatch {
        member: usize,
        expected: usize,
        actual: usize,
    },
    RetainedBytesExceeded {
        requested: usize,
        limit: usize,
    },
    RetainedBytesOverflow,
    Mean(MeanError),
}

impl fmt::Display for BaggingError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidLimits {
                field,
                value,
                limit,
            } => {
                write!(
                    formatter,
                    "invalid limit {field}={value}; supported (0, {limit}]"
                )
            }
            Self::ZeroMembers => formatter.write_str("bag requires at least one member"),
            Self::TooManyMembers { members, limit } => {
                write!(formatter, "member count {members} exceeds limit {limit}")
            }
            Self::FoldIdTooLarge { fold_id, limit } => {
                write!(formatter, "fold id {fold_id} exceeds {limit}")
            }
            Self::MemberIdTooLarge { member, limit } => {
                write!(formatter, "member id {member} exceeds {limit}")
            }
            Self::UnsupportedMode { mode } => {
                write!(
                    formatter,
                    "sampling mode {mode:?} is unsupported for this bag"
                )
            }
            Self::MultipleSequences { first, next } => {
                write!(
                    formatter,
                    "sequential sampling requires one sequence; saw {first} then {next}"
                )
            }
            Self::DataLengthMismatch { features, targets } => {
                write!(
                    formatter,
                    "feature rows {features} do not match targets {targets}"
                )
            }
            Self::InvalidUnitRange {
                unit,
                span,
                start,
                end,
            } => {
                write!(
                    formatter,
                    "unit {unit} span {span:?} has invalid range {start}..{end}"
                )
            }
            Self::SourceRowOutOfRange { unit, row, len } => {
                write!(
                    formatter,
                    "unit {unit} row {row} is out of range for {len} source rows"
                )
            }
            Self::SupportBeyondCutoff {
                unit,
                span,
                end,
                cutoff,
            } => {
                write!(
                    formatter,
                    "unit {unit} span {span} ends at {end}, beyond cutoff {cutoff}"
                )
            }
            Self::UnitLimitExceeded { units, limit } => {
                write!(formatter, "unit count {units} exceeds limit {limit}")
            }
            Self::SupportSpanLimitExceeded { spans, limit } => {
                write!(
                    formatter,
                    "support span count {spans} exceeds limit {limit}"
                )
            }
            Self::EndpointScratchExceeded { bytes, limit } => {
                write!(formatter, "endpoint scratch {bytes} exceeds limit {limit}")
            }
            Self::ExpandedRowsExceeded { rows, limit } => {
                write!(formatter, "expanded rows {rows} exceed limit {limit}")
            }
            Self::TotalSequentialWorkExceeded { computed, limit } => {
                write!(
                    formatter,
                    "sequential work {computed} exceeds limit {limit}"
                )
            }
            Self::Sampling(error) => write!(formatter, "sampling failed: {error}"),
            Self::MemberDimensionMismatch {
                member,
                expected,
                actual,
            } => {
                write!(
                    formatter,
                    "member {member} dimension {actual} != {expected}"
                )
            }
            Self::RetainedBytesExceeded { requested, limit } => {
                write!(formatter, "retained bytes {requested} exceed limit {limit}")
            }
            Self::RetainedBytesOverflow => {
                formatter.write_str("retained-byte accounting overflowed")
            }
            Self::Mean(error) => write!(formatter, "bag mean failed: {error}"),
        }
    }
}

impl std::error::Error for BaggingError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Sampling(error) => Some(error),
            Self::Mean(error) => Some(error),
            _ => None,
        }
    }
}

impl From<SamplingError> for BaggingError {
    fn from(error: SamplingError) -> Self {
        Self::Sampling(error)
    }
}

impl From<MeanError> for BaggingError {
    fn from(error: MeanError) -> Self {
        Self::Mean(error)
    }
}

/// Prediction-side row validation cause.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RowError {
    DimensionMismatch,
    NonFiniteInput,
}

/// Whether a member returned an error or a non-finite `Ok` value.
#[derive(Debug, Clone, PartialEq)]
pub enum MemberPredictCause<PE> {
    Predictor(PE),
    NonFiniteOutput,
}

/// Aggregation stage for batch publication.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AggregationStage {
    Mean,
}

/// All-or-nothing prediction failure.
#[derive(Debug, Clone, PartialEq)]
pub enum PredictionError<PE> {
    BatchTooLarge {
        requested: usize,
        limit: usize,
        evaluated_rows: usize,
    },
    RowValidation {
        row_index: usize,
        cause: RowError,
        evaluated_rows: usize,
    },
    MemberPredict {
        row_index: usize,
        member_index: usize,
        cause: MemberPredictCause<PE>,
        evaluated_rows: usize,
    },
    Aggregation {
        row_index: usize,
        stage: AggregationStage,
        cause: MeanError,
        evaluated_rows: usize,
    },
}

impl<PE: fmt::Debug> fmt::Display for PredictionError<PE> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BatchTooLarge {
                requested, limit, ..
            } => {
                write!(formatter, "batch size {requested} exceeds limit {limit}")
            }
            Self::RowValidation { row_index, .. } => {
                write!(formatter, "row {row_index} validation failed")
            }
            Self::MemberPredict {
                row_index,
                member_index,
                ..
            } => {
                write!(formatter, "member {member_index} failed at row {row_index}")
            }
            Self::Aggregation { row_index, .. } => {
                write!(formatter, "mean aggregation failed at row {row_index}")
            }
        }
    }
}

impl<PE: fmt::Debug> std::error::Error for PredictionError<PE> {}

pub(crate) fn checked_logical_bytes(
    element_count: usize,
    element_size: usize,
    scalar_bytes: usize,
) -> Option<usize> {
    element_count
        .checked_mul(element_size)?
        .checked_add(scalar_bytes)
}

pub(crate) fn add_retained(total: usize, member: usize) -> Option<usize> {
    total.checked_add(member)
}

fn mode_is_sequential(mode: SamplingMode) -> bool {
    matches!(
        mode,
        SamplingMode::Default | SamplingMode::Method(SampleMethod::SequentialBootstrap)
    )
}

fn validate_units(
    units: &[SampleUnit],
    features: usize,
    targets: usize,
    spec: &BaggingSpec,
    tree_profile: bool,
    strict_f32: bool,
) -> Result<(), BaggingError> {
    spec.limits.validate()?;
    if spec.members == 0 {
        return Err(BaggingError::ZeroMembers);
    }
    if spec.members > spec.limits.max_members {
        return Err(BaggingError::TooManyMembers {
            members: spec.members,
            limit: spec.limits.max_members,
        });
    }
    if spec.fold_id > u32::MAX as usize {
        return Err(BaggingError::FoldIdTooLarge {
            fold_id: spec.fold_id,
            limit: u32::MAX as usize,
        });
    }
    if spec.members > u32::MAX as usize {
        return Err(BaggingError::MemberIdTooLarge {
            member: spec.members,
            limit: u32::MAX as usize,
        });
    }
    if features != targets {
        return Err(BaggingError::DataLengthMismatch { features, targets });
    }
    match spec.mode {
        SamplingMode::Disabled => {}
        SamplingMode::Default | SamplingMode::Method(SampleMethod::SequentialBootstrap) => {
            if !tree_profile {
                return Err(BaggingError::UnsupportedMode { mode: spec.mode });
            }
            if strict_f32 {
                // Tree C sequential requires the #103 strict-f32 path.
                return Err(BaggingError::UnsupportedMode { mode: spec.mode });
            }
        }
        SamplingMode::Method(SampleMethod::WithoutReplacement) => {}
        SamplingMode::Method(SampleMethod::WithReplacement) => {
            if !tree_profile {
                return Err(BaggingError::UnsupportedMode { mode: spec.mode });
            }
        }
    }

    let units_len = units.len();
    if units_len > spec.limits.max_units {
        return Err(BaggingError::UnitLimitExceeded {
            units: units_len,
            limit: spec.limits.max_units,
        });
    }
    let spans = units.iter().map(|unit| unit.outcomes.len()).sum::<usize>();
    if spans > spec.limits.max_support_spans {
        return Err(BaggingError::SupportSpanLimitExceeded {
            spans,
            limit: spec.limits.max_support_spans,
        });
    }
    let scratch = 64usize
        .checked_mul(spans)
        .and_then(|value| value.checked_add(64usize.checked_mul(units_len)?))
        .and_then(|value| value.checked_add(256))
        .ok_or(BaggingError::RetainedBytesOverflow)?;
    if scratch > spec.limits.max_endpoint_scratch_bytes {
        return Err(BaggingError::EndpointScratchExceeded {
            bytes: scratch,
            limit: spec.limits.max_endpoint_scratch_bytes,
        });
    }
    if mode_is_sequential(spec.mode) {
        let work = (spec.members as u128)
            .saturating_mul(spec.sample_size as u128)
            .saturating_mul(units_len as u128)
            .saturating_mul((2usize.saturating_mul(spans)) as u128);
        if work > spec.limits.max_total_sequential_work as u128 {
            return Err(BaggingError::TotalSequentialWorkExceeded {
                computed: work,
                limit: spec.limits.max_total_sequential_work,
            });
        }
    }

    for (unit_index, unit) in units.iter().enumerate() {
        if unit.rows.is_empty() {
            return Err(BaggingError::InvalidUnitRange {
                unit: unit_index,
                span: None,
                start: 0,
                end: 0,
            });
        }
        for (span_index, range) in unit.rows.iter().enumerate() {
            if range.start >= range.end {
                return Err(BaggingError::InvalidUnitRange {
                    unit: unit_index,
                    span: Some(span_index),
                    start: range.start,
                    end: range.end,
                });
            }
            if range.end > features {
                return Err(BaggingError::SourceRowOutOfRange {
                    unit: unit_index,
                    row: range.end,
                    len: features,
                });
            }
        }
        if let Some(cutoff) = spec.support_cutoff {
            for (span_index, outcome) in unit.outcomes.iter().enumerate() {
                if outcome.end > cutoff {
                    return Err(BaggingError::SupportBeyondCutoff {
                        unit: unit_index,
                        span: span_index,
                        end: outcome.end,
                        cutoff,
                    });
                }
            }
        }
        if mode_is_sequential(spec.mode) {
            if let Some(first) = units.first() {
                if unit.sequence != first.sequence {
                    return Err(BaggingError::MultipleSequences {
                        first: first.sequence,
                        next: unit.sequence,
                    });
                }
            }
        }
    }

    let expanded = match spec.mode {
        SamplingMode::Disabled => units
            .iter()
            .flat_map(|unit| unit.rows.iter())
            .map(|range| range.end - range.start)
            .sum::<usize>(),
        _ => {
            let max_unit_rows = units
                .iter()
                .map(|unit| {
                    unit.rows
                        .iter()
                        .map(|range| range.end - range.start)
                        .sum::<usize>()
                })
                .max()
                .unwrap_or(0);
            spec.sample_size.saturating_mul(max_unit_rows)
        }
    };
    if expanded > spec.limits.max_expanded_rows_per_member {
        return Err(BaggingError::ExpandedRowsExceeded {
            rows: expanded,
            limit: spec.limits.max_expanded_rows_per_member,
        });
    }
    Ok(())
}

fn sample_member(
    units: &[SampleUnit],
    spec: &BaggingSpec,
    member: usize,
) -> Result<TrainingSample, BaggingError> {
    let sampling = SamplingSpec::new(spec.mode, spec.sample_size, spec.member_seed(member));
    sample_training(units, sampling).map_err(BaggingError::Sampling)
}

fn row_is_finite_64(row: &Vector64) -> bool {
    row.as_slice().iter().all(|value| value.is_finite())
}

fn row_is_finite_32(row: &Vector) -> bool {
    row.as_slice().iter().all(|value| value.is_finite())
}

fn validate_row_64(row: &Vector64, expected: usize) -> Result<(), RowError> {
    if row.len() != expected {
        return Err(RowError::DimensionMismatch);
    }
    if !row_is_finite_64(row) {
        return Err(RowError::NonFiniteInput);
    }
    Ok(())
}

fn validate_row_32(row: &Vector, expected: usize) -> Result<(), RowError> {
    if row.len() != expected {
        return Err(RowError::DimensionMismatch);
    }
    if !row_is_finite_32(row) {
        return Err(RowError::NonFiniteInput);
    }
    Ok(())
}

fn retain_tree64(
    total: &mut usize,
    model: &crate::models::trees::decision_tree::DecisionTree64,
) -> Result<(), BaggingError> {
    let bytes = checked_logical_bytes(
        model.nodes().len(),
        std::mem::size_of::<crate::models::trees::decision_tree::DecisionTreeNode64>(),
        std::mem::size_of::<usize>(),
    )
    .ok_or(BaggingError::RetainedBytesOverflow)?;
    *total = add_retained(*total, bytes).ok_or(BaggingError::RetainedBytesOverflow)?;
    Ok(())
}

fn retain_tree32(
    total: &mut usize,
    model: &crate::models::trees::decision_tree::DecisionTree,
) -> Result<(), BaggingError> {
    let bytes = checked_logical_bytes(
        model.nodes().len(),
        std::mem::size_of::<crate::models::trees::decision_tree::DecisionTreeNode>(),
        std::mem::size_of::<usize>(),
    )
    .ok_or(BaggingError::RetainedBytesOverflow)?;
    *total = add_retained(*total, bytes).ok_or(BaggingError::RetainedBytesOverflow)?;
    Ok(())
}

fn retain_ols64(
    total: &mut usize,
    model: &crate::models::linear::ols::LinearRegression64,
) -> Result<(), BaggingError> {
    let bytes = checked_logical_bytes(
        model.weights().len(),
        std::mem::size_of::<f64>(),
        std::mem::size_of::<f64>(),
    )
    .ok_or(BaggingError::RetainedBytesOverflow)?;
    *total = add_retained(*total, bytes).ok_or(BaggingError::RetainedBytesOverflow)?;
    Ok(())
}

fn retain_ols32(
    total: &mut usize,
    model: &crate::models::linear::ols::LinearRegression,
) -> Result<(), BaggingError> {
    let bytes = checked_logical_bytes(
        model.weights().len(),
        std::mem::size_of::<f32>(),
        std::mem::size_of::<f32>(),
    )
    .ok_or(BaggingError::RetainedBytesOverflow)?;
    *total = add_retained(*total, bytes).ok_or(BaggingError::RetainedBytesOverflow)?;
    Ok(())
}

pub(crate) struct Candidate<M> {
    pub members: Vec<M>,
    pub feature_dimension: usize,
    pub retained_bytes: usize,
}

pub(crate) fn assemble_candidate<M, E>(
    units: &[SampleUnit],
    spec: &BaggingSpec,
    mut map_sampling: impl FnMut(BaggingError) -> E,
    mut fit_member: impl FnMut(usize, &[usize]) -> Result<(M, usize), E>,
    dimension: impl Fn(&M) -> usize,
    mismatch: impl Fn(usize, usize, usize) -> E,
    retained_exceeded: impl Fn(usize, usize) -> E,
) -> Result<Candidate<M>, E> {
    let mut members = Vec::with_capacity(spec.members);
    let mut retained_bytes = 0usize;
    let mut feature_dimension = None;
    for member in 0..spec.members {
        let sample = sample_member(units, spec, member).map_err(&mut map_sampling)?;
        let (model, bytes) = fit_member(member, sample.row_indices())?;
        let actual = dimension(&model);
        if let Some(expected) = feature_dimension {
            if actual != expected {
                return Err(mismatch(member, expected, actual));
            }
        } else {
            feature_dimension = Some(actual);
        }
        retained_bytes = add_retained(retained_bytes, bytes)
            .ok_or_else(|| map_sampling(BaggingError::RetainedBytesOverflow))?;
        members.push(model);
    }
    if retained_bytes > spec.limits.max_retained_bytes {
        return Err(retained_exceeded(
            retained_bytes,
            spec.limits.max_retained_bytes,
        ));
    }
    Ok(Candidate {
        members,
        feature_dimension: feature_dimension.unwrap_or(0),
        retained_bytes,
    })
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn publish_batch<R, O, PE>(
    batch: &[Observation<R>],
    max_predictions: usize,
    configured_max: usize,
    members: usize,
    mut validate: impl FnMut(usize, &R) -> Result<(), RowError>,
    mut predict: impl FnMut(usize, usize, &R) -> Result<O, PE>,
    is_finite: impl Fn(&O) -> bool,
    mut mean: impl FnMut(&[O]) -> Result<O, MeanError>,
) -> Result<Vec<O>, PredictionError<PE>> {
    let limit = max_predictions.min(configured_max);
    if batch.len() > limit {
        return Err(PredictionError::BatchTooLarge {
            requested: batch.len(),
            limit,
            evaluated_rows: 0,
        });
    }
    let mut outputs = Vec::with_capacity(batch.len());
    for (row_index, observation) in batch.iter().enumerate() {
        let row = observation.value();
        validate(row_index, row).map_err(|cause| PredictionError::RowValidation {
            row_index,
            cause,
            evaluated_rows: row_index,
        })?;
        let mut predictions = Vec::with_capacity(members);
        for member_index in 0..members {
            let value = predict(row_index, member_index, row).map_err(|source| {
                PredictionError::MemberPredict {
                    row_index,
                    member_index,
                    cause: MemberPredictCause::Predictor(source),
                    evaluated_rows: row_index,
                }
            })?;
            if !is_finite(&value) {
                return Err(PredictionError::MemberPredict {
                    row_index,
                    member_index,
                    cause: MemberPredictCause::NonFiniteOutput,
                    evaluated_rows: row_index,
                });
            }
            predictions.push(value);
        }
        let value = mean(&predictions).map_err(|cause| PredictionError::Aggregation {
            row_index,
            stage: AggregationStage::Mean,
            cause,
            evaluated_rows: row_index,
        })?;
        outputs.push(value);
    }
    Ok(outputs)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::preprocessing::sampling::{OutcomeInterval, SampleMethod, SampleUnit};

    #[allow(clippy::single_range_in_vec_init)]
    fn unit(index: usize) -> SampleUnit {
        SampleUnit {
            rows: vec![index..index + 1],
            outcomes: vec![OutcomeInterval {
                start: index as i64,
                end: index as i64 + 1,
            }],
            sequence: 0,
        }
    }

    fn units(count: usize) -> Vec<SampleUnit> {
        (0..count).map(unit).collect()
    }

    #[test]
    fn retained_size_arithmetic_is_checked() {
        assert_eq!(checked_logical_bytes(1, 8, 8), Some(16));
        assert_eq!(checked_logical_bytes(usize::MAX, 2, 0), None);
        assert_eq!(add_retained(usize::MAX, 1), None);
    }

    #[test]
    fn work_boundary_passes_at_500_and_rejects_at_501() {
        let units = units(1_000);
        let pass = BaggingSpec::new(
            1,
            SamplingMode::Method(SampleMethod::SequentialBootstrap),
            500,
            7,
            0,
        );
        assert!(validate_units(&units, 1_000, 1_000, &pass, true, false).is_ok());

        let reject = BaggingSpec::new(
            1,
            SamplingMode::Method(SampleMethod::SequentialBootstrap),
            501,
            7,
            0,
        );
        assert!(matches!(
            validate_units(&units, 1_000, 1_000, &reject, true, false),
            Err(BaggingError::TotalSequentialWorkExceeded { .. })
        ));
    }

    #[test]
    fn cutoff_equality_passes_and_later_end_rejects() {
        let units = units(1);
        let pass =
            BaggingSpec::new(1, SamplingMode::Disabled, 0, 1, 0).with_support_cutoff(Some(1));
        assert!(validate_units(&units, 1, 1, &pass, true, false).is_ok());

        let reject =
            BaggingSpec::new(1, SamplingMode::Disabled, 0, 1, 0).with_support_cutoff(Some(0));
        assert!(matches!(
            validate_units(&units, 1, 1, &reject, true, false),
            Err(BaggingError::SupportBeyondCutoff {
                end: 1,
                cutoff: 0,
                ..
            })
        ));
    }

    #[test]
    fn production_batch_scripted_nonfinite_and_mean_failures() {
        let batch: Vec<Observation<usize>> = (0..5).map(Observation::new).collect();

        let mut calls = Vec::new();
        let err = publish_batch::<usize, f64, ()>(
            &batch,
            8,
            8,
            2,
            |_, _| Ok(()),
            |row, member, _| {
                calls.push((row, member));
                if row == 3 && member == 1 {
                    Ok(f64::INFINITY)
                } else {
                    Ok(1.0)
                }
            },
            |value: &f64| value.is_finite(),
            mean::mean_f64,
        )
        .unwrap_err();
        match err {
            PredictionError::MemberPredict {
                row_index,
                member_index,
                cause: MemberPredictCause::NonFiniteOutput,
                evaluated_rows,
            } => {
                assert_eq!(row_index, 3);
                assert_eq!(member_index, 1);
                assert_eq!(evaluated_rows, 3);
            }
            other => panic!("unexpected production batch result: {other:?}"),
        }
        assert_eq!(
            calls,
            vec![
                (0, 0),
                (0, 1),
                (1, 0),
                (1, 1),
                (2, 0),
                (2, 1),
                (3, 0),
                (3, 1)
            ]
        );
        assert!(calls.iter().all(|(row, _)| *row < 4));

        let vector = [2f64.powi(60), 1.0, 2f64.powi(-60), -2f64.powi(60), -1.0];
        let mut calls = Vec::new();
        let err = publish_batch::<usize, f64, ()>(
            &batch,
            8,
            8,
            5,
            |_, _| Ok(()),
            |row, member, _| {
                calls.push((row, member));
                if row == 3 {
                    Ok(vector[member])
                } else {
                    Ok(1.0)
                }
            },
            |value: &f64| value.is_finite(),
            mean::mean_f64,
        )
        .unwrap_err();
        match err {
            PredictionError::Aggregation {
                row_index,
                stage: AggregationStage::Mean,
                cause: MeanError::MeanCancellation,
                evaluated_rows,
            } => {
                assert_eq!(row_index, 3);
                assert_eq!(evaluated_rows, 3);
            }
            other => panic!("unexpected production batch result: {other:?}"),
        }
        assert_eq!(calls.len(), 20);
        assert!(calls.iter().any(|&(row, member)| row == 3 && member == 4));
        assert!(calls.iter().all(|(row, _)| *row < 4));
    }
}
