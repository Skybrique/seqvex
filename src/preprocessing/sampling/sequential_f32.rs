//! Strict-`f32` sequential bootstrap for the direct training-fold sampler.
//!
//! This path is deliberately separate from `sample_training`: it validates the
//! strict per-call limits, uses only `f32` weight arithmetic, and converts the
//! weights to exact binary32-derived integer tickets before one shared-RNG draw
//! per selection. It never calls the legacy `f64` implementation.

use std::num::NonZeroU64;
use std::ops::Range;

use crate::foundation::numerical::RandomGenerator;

use super::{
    SampleMethod, SampleUnit, SamplingError, SamplingMode, SamplingSpec, SequentialLimits,
    TrainingSample,
};

/// Strict support/count domain: `2^24` covered ticks.
const F32_DOMAIN_LIMIT: u128 = 1 << 24;

/// Exact-binary ticket scale.
const TICKET_SCALE: u128 = 1 << 32;

const MAX_UNITS_CEILING: u128 = 1_048_576;
const MAX_SUPPORT_SPANS_CEILING: u128 = 16_777_216;
const MAX_ENDPOINT_SCRATCH_CEILING: u128 = 268_435_456;
const MAX_EXPANDED_ROWS_CEILING: u128 = 16_777_216;
const MAX_WORK_CELLS_CEILING: u128 = 1_000_000_000_000;

const ENDPOINT_SCRATCH_PER_SPAN: u128 = 64;
const ENDPOINT_SCRATCH_PER_UNIT: u128 = 64;
const ENDPOINT_SCRATCH_FIXED: u128 = 256;

/// Draws one strict-`f32` sequential-bootstrap sample.
///
/// `spec` must select [`SamplingMode::Default`] or
/// [`SamplingMode::Method`] with [`SampleMethod::SequentialBootstrap`].
/// Identity and both integer methods are rejected; use
/// [`super::sample_training`] for them.
/// The function accepts `Default` and explicit `SequentialBootstrap`
/// identically, creates one fresh generator from `spec.seed` after preflight,
/// and returns caller-row indices and per-unit draw counts.
///
/// The sampler recomputes every candidate's length-weighted average uniqueness
/// in `f32`, normalizes the stored weights by the per-draw maximum, converts
/// them to exact integer tickets with `Q = 2^32`, round-half-up and a positive
/// floor, requests one bounded draw per selection, and updates concurrency on
/// the selected unit's covered support. It validates shape, support, numerical
/// domain, per-call limits and hard capacities before constructing the
/// generator. Every caller-reachable rejection consumes zero selection and
/// shuffle draws.
///
/// # Errors
///
/// Returns [`SamplingError`] for invalid limits, an unadmitted mode, zero sample
/// size, no units, malformed shape/support, strict count/span domain failures,
/// per-call limit failures, and defensive capacity or arithmetic failures. A
/// zero stored weight or failed ticket calculation is a defensive
/// [`SamplingError::F32WeightUnderflow`] or
/// [`SamplingError::F32WeightQuantumOverflow`]; admitted input cannot reach
/// either after a draw.
///
/// # Example
///
/// ```
/// use seqvex::preprocessing::sampling::{
///     OutcomeInterval, SampleUnit, SamplingMode, SamplingSpec, SequentialLimits,
///     sample_training_sequential_f32,
/// };
///
/// let units = [SampleUnit {
///     rows: vec![0..1],
///     outcomes: vec![OutcomeInterval { start: 0, end: 1 }],
///     sequence: 0,
/// }];
/// let spec = SamplingSpec::new(SamplingMode::Default, 1, 7);
/// let sample = sample_training_sequential_f32(
///     &units,
///     spec,
///     &SequentialLimits::default(),
/// ).expect("strict sample");
/// assert_eq!(sample.row_indices(), &[0]);
/// ```
pub fn sample_training_sequential_f32(
    units: &[SampleUnit],
    spec: SamplingSpec,
    limits: &SequentialLimits,
) -> Result<TrainingSample, SamplingError> {
    let prepared = preflight(units, spec, limits)?;
    let mut plan = build_plan(units, prepared)?;
    let mut generator = RandomGenerator::from_seed(spec.seed);
    let mut selection_calls = 0;
    let mut shuffle_calls = 0;
    draw(
        units,
        spec,
        &mut plan,
        &mut |_, upper| {
            generator.draw_u64_below(NonZeroU64::new(upper).expect("draw bounds are positive"))
        },
        &mut selection_calls,
        &mut shuffle_calls,
        None,
    )
}

/// Which phase requested a bounded draw.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DrawKind {
    Selection,
    Shuffle,
}

#[derive(Debug)]
#[allow(dead_code)] // Populated only by the private test trace path.
struct StepTrace {
    draw_index: usize,
    weights: Vec<f32>,
    tickets: Vec<u64>,
    bound: u64,
    bounded_draw: u64,
    selected: usize,
}

struct Prepared {
    sample_size: usize,
    endpoints_capacity: usize,
    unit_count: usize,
    span_count: usize,
    output_bound: usize,
}

struct Plan {
    sample_size: usize,
    output_bound: usize,
    endpoints: Vec<i64>,
    segment_counts: Vec<usize>,
    unit_tick_len: Vec<u64>,
    unit_span_ranges: Vec<Range<usize>>,
    span_segments: Vec<(usize, usize)>,
    weights: Vec<f32>,
    tickets: Vec<u64>,
    counts: Vec<usize>,
}

fn preflight(
    units: &[SampleUnit],
    spec: SamplingSpec,
    limits: &SequentialLimits,
) -> Result<Prepared, SamplingError> {
    validate_limits(limits)?;
    admit_mode(spec.mode)?;
    if spec.sample_size == 0 {
        return Err(SamplingError::ZeroSampleSize);
    }
    if units.is_empty() {
        return Err(SamplingError::NoEligibleUnits);
    }

    validate_shape_and_support(units)?;

    // Numerical domain: inherited 2^53 first, then strict count, then span.
    let mut total_span: u128 = 0;
    for unit in units {
        let mut unit_span: u128 = 0;
        for outcome in &unit.outcomes {
            let length = (outcome.end as i128 - outcome.start as i128) as u128;
            unit_span = unit_span
                .checked_add(length)
                .ok_or(SamplingError::ArithmeticOverflow {
                    bound: "outcome_span",
                })?;
        }
        if unit_span > super::OUTCOME_SPAN_LIMIT {
            return Err(SamplingError::OutcomeSpanOverflow {
                computed: unit_span,
                limit: super::OUTCOME_SPAN_LIMIT,
            });
        }
        total_span =
            total_span
                .checked_add(unit_span)
                .ok_or(SamplingError::ArithmeticOverflow {
                    bound: "outcome_span",
                })?;
    }
    if total_span > super::OUTCOME_SPAN_LIMIT {
        return Err(SamplingError::OutcomeSpanOverflow {
            computed: total_span,
            limit: super::OUTCOME_SPAN_LIMIT,
        });
    }

    let sample_size = spec.sample_size as u128;
    if sample_size > F32_DOMAIN_LIMIT {
        return Err(SamplingError::F32CountUnsupported {
            computed: sample_size,
            limit: F32_DOMAIN_LIMIT,
        });
    }
    for (unit_index, unit) in units.iter().enumerate() {
        let mut unit_span: u128 = 0;
        for outcome in &unit.outcomes {
            unit_span = unit_span
                .checked_add((outcome.end as i128 - outcome.start as i128) as u128)
                .ok_or(SamplingError::ArithmeticOverflow {
                    bound: "outcome_span",
                })?;
        }
        if unit_span > F32_DOMAIN_LIMIT {
            return Err(SamplingError::F32SpanUnsupported {
                unit: unit_index,
                computed: unit_span,
                limit: F32_DOMAIN_LIMIT,
            });
        }
    }

    // Per-call operational caps, in approved order.
    let unit_count = units.len() as u128;
    if unit_count > limits.max_units as u128 {
        return Err(SamplingError::UnitLimitExceeded {
            computed: unit_count,
            limit: limits.max_units as u128,
        });
    }

    let span_count = checked_support_span_count(units)?;
    if span_count > limits.max_support_spans as u128 {
        return Err(SamplingError::SupportSpanLimitExceeded {
            computed: span_count,
            limit: limits.max_support_spans as u128,
        });
    }

    let scratch = checked_scratch_bytes(span_count, unit_count)?;
    if scratch > limits.max_endpoint_scratch_bytes as u128 {
        return Err(SamplingError::EndpointScratchExceeded {
            computed: scratch,
            limit: limits.max_endpoint_scratch_bytes as u128,
        });
    }

    let expanded_rows = checked_expanded_rows(units, sample_size)?;
    if expanded_rows > limits.max_expanded_rows as u128 {
        return Err(SamplingError::ExpandedRowsExceeded {
            computed: expanded_rows,
            limit: limits.max_expanded_rows as u128,
        });
    }

    let endpoints = checked_endpoint_count(span_count)?;
    let work = checked_work(sample_size, unit_count, endpoints)?;
    if work > limits.max_work_cells as u128 {
        return Err(SamplingError::WorkLimitExceeded {
            computed: work,
            limit: limits.max_work_cells as u128,
        });
    }

    // Hard capacities: element class, byte class, then work. Fixed descriptors
    // live on the stack; the optional shuffle buffer stays last and is checked
    // only when shuffle is enabled.
    let buffers = [
        (endpoints, size_of::<i64>()),
        (endpoints.saturating_sub(1), size_of::<usize>()),
        (unit_count, size_of::<u64>()),
        (unit_count, size_of::<Range<usize>>()),
        (span_count, size_of::<(usize, usize)>()),
        (unit_count, size_of::<f32>()),
        (unit_count, size_of::<u64>()),
        (unit_count, size_of::<usize>()),
        (expanded_rows, size_of::<usize>()),
        (sample_size, size_of::<usize>()),
    ];
    let buffer_count = if spec.shuffle {
        buffers.len()
    } else {
        buffers.len() - 1
    };
    check_hard_capacity(&buffers[..buffer_count], work)?;

    Ok(Prepared {
        sample_size: spec.sample_size,
        endpoints_capacity: checked_element_count(endpoints)?,
        unit_count: checked_element_count(unit_count)?,
        span_count: checked_element_count(span_count)?,
        output_bound: checked_element_count(expanded_rows)?,
    })
}

fn validate_limits(limits: &SequentialLimits) -> Result<(), SamplingError> {
    check_limit("max_units", limits.max_units as u128, MAX_UNITS_CEILING)?;
    check_limit(
        "max_support_spans",
        limits.max_support_spans as u128,
        MAX_SUPPORT_SPANS_CEILING,
    )?;
    check_limit(
        "max_endpoint_scratch_bytes",
        limits.max_endpoint_scratch_bytes as u128,
        MAX_ENDPOINT_SCRATCH_CEILING,
    )?;
    check_limit(
        "max_expanded_rows",
        limits.max_expanded_rows as u128,
        MAX_EXPANDED_ROWS_CEILING,
    )?;
    check_limit(
        "max_work_cells",
        limits.max_work_cells as u128,
        MAX_WORK_CELLS_CEILING,
    )
}

fn check_limit(field: &'static str, value: u128, max: u128) -> Result<(), SamplingError> {
    if value == 0 || value > max {
        return Err(SamplingError::InvalidLimits {
            field,
            value,
            min: 1,
            max,
        });
    }
    Ok(())
}

fn admit_mode(mode: SamplingMode) -> Result<(), SamplingError> {
    match mode {
        SamplingMode::Default | SamplingMode::Method(SampleMethod::SequentialBootstrap) => Ok(()),
        other => Err(SamplingError::UnsupportedMode { mode: other }),
    }
}

fn validate_shape_and_support(units: &[SampleUnit]) -> Result<(), SamplingError> {
    // Empty/reversed rows across the whole input.
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

    // Row order: cross-unit boundary before later ranges in the same unit.
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
        previous_unit_end = Some(unit.rows[unit.rows.len() - 1].end);
    }

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

    if let Some((unit, _)) = units
        .iter()
        .enumerate()
        .find(|(_, unit)| unit.outcomes.is_empty())
    {
        return Err(SamplingError::MissingOutcomeSupport { unit });
    }
    Ok(())
}

fn checked_support_span_count(units: &[SampleUnit]) -> Result<u128, SamplingError> {
    let mut total = 0_u128;
    for unit in units {
        total = total.checked_add(unit.outcomes.len() as u128).ok_or(
            SamplingError::ArithmeticOverflow {
                bound: "support_spans",
            },
        )?;
    }
    Ok(total)
}

fn checked_scratch_bytes(span_count: u128, unit_count: u128) -> Result<u128, SamplingError> {
    let spans = span_count.checked_mul(ENDPOINT_SCRATCH_PER_SPAN).ok_or(
        SamplingError::ArithmeticOverflow {
            bound: "endpoint_scratch_bytes",
        },
    )?;
    let units = unit_count.checked_mul(ENDPOINT_SCRATCH_PER_UNIT).ok_or(
        SamplingError::ArithmeticOverflow {
            bound: "endpoint_scratch_bytes",
        },
    )?;
    spans
        .checked_add(units)
        .and_then(|value| value.checked_add(ENDPOINT_SCRATCH_FIXED))
        .ok_or(SamplingError::ArithmeticOverflow {
            bound: "endpoint_scratch_bytes",
        })
}

fn checked_expanded_rows(units: &[SampleUnit], sample_size: u128) -> Result<u128, SamplingError> {
    let mut max_unit_rows = 0_u128;
    for unit in units {
        let mut rows = 0_u128;
        for range in &unit.rows {
            rows = rows.checked_add((range.end - range.start) as u128).ok_or(
                SamplingError::ArithmeticOverflow {
                    bound: "expanded_rows",
                },
            )?;
        }
        max_unit_rows = max_unit_rows.max(rows);
    }
    checked_expanded_product(sample_size, max_unit_rows)
}

fn checked_expanded_product(sample_size: u128, max_unit_rows: u128) -> Result<u128, SamplingError> {
    sample_size
        .checked_mul(max_unit_rows)
        .ok_or(SamplingError::ArithmeticOverflow {
            bound: "expanded_rows",
        })
}

/// Rows actually emitted for `counts`: `sum(count_i * row_count_i)`, skipping
/// units with no draws. Because `sum(count_i) = n` and
/// `row_count_i <= max_i row_count_i`, this is at most the preflight
/// `checked_expanded_rows` bound (`n * max_i row_count_i`), which is already
/// validated against `usize::MAX`; the result therefore fits `usize` on both
/// 32- and 64-bit targets.
fn realized_row_count(counts: &[usize], units: &[SampleUnit]) -> usize {
    let mut total = 0_u128;
    for (&count, unit) in counts.iter().zip(units) {
        if count == 0 {
            continue;
        }
        let unit_rows: u128 = unit
            .rows
            .iter()
            .map(|range| (range.end - range.start) as u128)
            .sum();
        total += count as u128 * unit_rows;
    }
    usize::try_from(total).expect("realized rows fit the checked preflight bound")
}

fn checked_endpoint_count(span_count: u128) -> Result<u128, SamplingError> {
    span_count
        .checked_mul(2)
        .ok_or(SamplingError::ArithmeticOverflow {
            bound: "endpoint_count",
        })
}

fn checked_work(
    sample_size: u128,
    unit_count: u128,
    endpoint_count: u128,
) -> Result<u128, SamplingError> {
    sample_size
        .checked_mul(unit_count)
        .and_then(|value| value.checked_mul(endpoint_count))
        .ok_or(SamplingError::ArithmeticOverflow {
            bound: "work_cells",
        })
}

fn checked_element_count(elements: u128) -> Result<usize, SamplingError> {
    if elements > usize::MAX as u128 {
        return Err(SamplingError::SampleLengthOverflow {
            computed: elements,
            limit: usize::MAX as u128,
        });
    }
    Ok(elements as usize)
}

fn checked_array_bytes(elements: u128, element_size: usize) -> Result<(), SamplingError> {
    let bytes =
        elements
            .checked_mul(element_size as u128)
            .ok_or(SamplingError::ArithmeticOverflow {
                bound: "allocation_bytes",
            })?;
    if bytes > isize::MAX as u128 {
        return Err(SamplingError::AllocationTooLarge {
            bytes,
            limit: isize::MAX as u128,
        });
    }
    Ok(())
}

fn check_hard_capacity(buffers: &[(u128, usize)], work: u128) -> Result<(), SamplingError> {
    for (elements, _) in buffers {
        checked_element_count(*elements)?;
    }
    for (elements, element_size) in buffers {
        checked_array_bytes(*elements, *element_size)?;
    }
    if work > u64::MAX as u128 {
        return Err(SamplingError::SamplingWorkOverflow {
            computed: work,
            limit: u64::MAX as u128,
        });
    }
    Ok(())
}

fn build_plan(units: &[SampleUnit], prepared: Prepared) -> Result<Plan, SamplingError> {
    let mut endpoints = Vec::with_capacity(prepared.endpoints_capacity);
    for unit in units {
        for outcome in &unit.outcomes {
            endpoints.push(outcome.start);
            endpoints.push(outcome.end);
        }
    }
    endpoints.sort_unstable();
    endpoints.dedup();

    let segment_elements = (endpoints.len() as u128).saturating_sub(1);
    checked_element_count(segment_elements)?;
    checked_array_bytes(segment_elements, size_of::<usize>())?;
    let segment_counts = vec![0_usize; segment_elements as usize];

    let mut span_segments = Vec::with_capacity(prepared.span_count);
    let mut unit_span_ranges = Vec::with_capacity(prepared.unit_count);
    let mut unit_tick_len = Vec::with_capacity(prepared.unit_count);
    for unit in units {
        let span_start = span_segments.len();
        let mut ticks = 0_u128;
        for outcome in &unit.outcomes {
            let start_index = endpoints
                .binary_search(&outcome.start)
                .expect("declared start is an endpoint");
            let end_index = endpoints
                .binary_search(&outcome.end)
                .expect("declared end is an endpoint");
            span_segments.push((start_index, end_index));
            ticks = ticks
                .checked_add((outcome.end as i128 - outcome.start as i128) as u128)
                .ok_or(SamplingError::ArithmeticOverflow {
                    bound: "outcome_span",
                })?;
        }
        unit_span_ranges.push(span_start..span_segments.len());
        unit_tick_len.push(u64::try_from(ticks).expect("strict unit span fits u64"));
    }

    Ok(Plan {
        sample_size: prepared.sample_size,
        output_bound: prepared.output_bound,
        endpoints,
        segment_counts,
        unit_tick_len,
        unit_span_ranges,
        span_segments,
        weights: vec![0.0_f32; prepared.unit_count],
        tickets: vec![0_u64; prepared.unit_count],
        counts: vec![0_usize; prepared.unit_count],
    })
}

fn draw(
    units: &[SampleUnit],
    spec: SamplingSpec,
    plan: &mut Plan,
    next: &mut impl FnMut(DrawKind, u64) -> u64,
    selection_calls: &mut usize,
    shuffle_calls: &mut usize,
    mut trace: Option<&mut Vec<StepTrace>>,
) -> Result<TrainingSample, SamplingError> {
    for draw_index in 0..plan.sample_size {
        compute_weights(units, plan);
        let total = compute_tickets(&plan.weights, draw_index, &mut plan.tickets)?;
        if total == 0 {
            return Err(SamplingError::F32WeightUnderflow {
                draw_index,
                unit: 0,
            });
        }

        *selection_calls += 1;
        let bounded_draw = next(DrawKind::Selection, total);
        if bounded_draw >= total {
            return Err(SamplingError::F32WeightQuantumOverflow {
                draw_index,
                unit: 0,
            });
        }
        let chosen = select_unit(&plan.tickets, bounded_draw).ok_or(
            SamplingError::F32WeightQuantumOverflow {
                draw_index,
                unit: 0,
            },
        )?;

        if let Some(trace) = trace.as_deref_mut() {
            trace.push(StepTrace {
                draw_index,
                weights: plan.weights.clone(),
                tickets: plan.tickets.clone(),
                bound: total,
                bounded_draw,
                selected: chosen,
            });
        }

        plan.counts[chosen] += 1;
        for span in plan.unit_span_ranges[chosen].clone() {
            let (start, end) = plan.span_segments[span];
            for count in &mut plan.segment_counts[start..end] {
                *count += 1;
            }
        }
    }

    let realized_rows = realized_row_count(&plan.counts, units);
    debug_assert!(realized_rows <= plan.output_bound);
    let mut rows = Vec::with_capacity(realized_rows);
    if spec.shuffle {
        let occurrences_total: usize = plan.counts.iter().sum();
        let mut occurrences = Vec::with_capacity(occurrences_total);
        let mut first_unit = 0;
        while first_unit < units.len() {
            let sequence = units[first_unit].sequence;
            let mut next_unit = first_unit + 1;
            while next_unit < units.len() && units[next_unit].sequence == sequence {
                next_unit += 1;
            }
            let occurrence_start = occurrences.len();
            for (offset, &count) in plan.counts[first_unit..next_unit].iter().enumerate() {
                let unit_index = first_unit + offset;
                for _ in 0..count {
                    occurrences.push(unit_index);
                }
            }
            let occurrence_end = occurrences.len();
            super::shuffle(
                &mut occurrences[occurrence_start..occurrence_end],
                &mut |upper| {
                    *shuffle_calls += 1;
                    next(DrawKind::Shuffle, upper)
                },
            );
            first_unit = next_unit;
        }
        for unit_index in occurrences {
            super::push_unit_rows(&mut rows, &units[unit_index]);
        }
    } else {
        for (unit_index, &count) in plan.counts.iter().enumerate() {
            for _ in 0..count {
                super::push_unit_rows(&mut rows, &units[unit_index]);
            }
        }
    }

    Ok(TrainingSample {
        row_indices: rows,
        unit_counts: std::mem::take(&mut plan.counts),
    })
}

fn compute_weights(units: &[SampleUnit], plan: &mut Plan) {
    for unit_index in 0..units.len() {
        let mut total = 0.0_f32;
        for span_index in plan.unit_span_ranges[unit_index].clone() {
            let (start, end) = plan.span_segments[span_index];
            for segment in start..end {
                let length =
                    (plan.endpoints[segment + 1] as i128 - plan.endpoints[segment] as i128) as f32;
                let concurrency = plan.segment_counts[segment] as f32;
                total += length / concurrency_denominator(concurrency);
            }
        }
        let ticks = plan.unit_tick_len[unit_index] as f32;
        plan.weights[unit_index] = total / ticks;
    }
}

fn concurrency_denominator(concurrency: f32) -> f32 {
    1.0_f32 + concurrency
}

fn compute_tickets(
    weights: &[f32],
    draw_index: usize,
    tickets: &mut [u64],
) -> Result<u64, SamplingError> {
    debug_assert_eq!(weights.len(), tickets.len());
    if let Some(unit) = weights.iter().position(|&weight| weight == 0.0) {
        return Err(SamplingError::F32WeightUnderflow { draw_index, unit });
    }
    let maximum = weights.iter().copied().fold(0.0_f32, f32::max);
    if maximum == 0.0 {
        return Err(SamplingError::F32WeightUnderflow {
            draw_index,
            unit: 0,
        });
    }

    let mut total = 0_u64;
    for (unit, (&weight, ticket)) in weights.iter().zip(tickets.iter_mut()).enumerate() {
        let value = ticket_for_weight(weight, maximum)
            .ok_or(SamplingError::F32WeightQuantumOverflow { draw_index, unit })?;
        *ticket = value;
        total = total
            .checked_add(value)
            .ok_or(SamplingError::F32WeightQuantumOverflow { draw_index, unit })?;
    }
    Ok(total)
}

fn ticket_for_weight(weight: f32, maximum: f32) -> Option<u64> {
    let (weight_mantissa, weight_exponent) = decode_positive_f32(weight)?;
    let (maximum_mantissa, maximum_exponent) = decode_positive_f32(maximum)?;
    let shift = 32_i32 + weight_exponent - maximum_exponent;
    if shift < -96 {
        return Some(1);
    }

    let (numerator, denominator) = if shift >= 0 {
        let factor = 1_u128.checked_shl(shift as u32)?;
        (
            (weight_mantissa as u128).checked_mul(factor)?,
            maximum_mantissa as u128,
        )
    } else {
        let factor = 1_u128.checked_shl((-shift) as u32)?;
        (
            weight_mantissa as u128,
            (maximum_mantissa as u128).checked_mul(factor)?,
        )
    };

    let twice_numerator = numerator.checked_mul(2)?;
    let rounded_numerator = twice_numerator.checked_add(denominator)?;
    let twice_denominator = denominator.checked_mul(2)?;
    let ticket = (rounded_numerator / twice_denominator).clamp(1, TICKET_SCALE);
    Some(ticket as u64)
}

fn decode_positive_f32(value: f32) -> Option<(u32, i32)> {
    if !value.is_finite() || value <= 0.0 {
        return None;
    }
    let bits = value.to_bits();
    let exponent = ((bits >> 23) & 0xFF) as i32;
    let fraction = bits & 0x7F_FFFF;
    if exponent == 0 {
        Some((fraction, -149))
    } else {
        Some(((1 << 23) | fraction, exponent - 127 - 23))
    }
}

fn select_unit(tickets: &[u64], bounded_draw: u64) -> Option<usize> {
    let mut cumulative = 0_u64;
    for (unit, &ticket) in tickets.iter().enumerate() {
        cumulative = cumulative.checked_add(ticket)?;
        if cumulative > bounded_draw {
            return Some(unit);
        }
    }
    None
}

#[cfg(test)]
#[allow(clippy::single_range_in_vec_init, clippy::reversed_empty_ranges)]
mod tests {
    use super::*;
    use crate::preprocessing::sampling::OutcomeInterval;

    const L: u128 = 1 << 24;
    const Q: u64 = 1 << 32;

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

    fn single(row: usize, outcomes: &[(i64, i64)], sequence: u32) -> SampleUnit {
        unit(&[row..row + 1], outcomes, sequence)
    }

    fn valid_units() -> Vec<SampleUnit> {
        vec![single(0, &[(0, 1)], 0)]
    }

    fn normalized_units() -> Vec<SampleUnit> {
        (0..4)
            .map(|index| single(index, &[(index as i64, index as i64 + 1)], 0))
            .collect()
    }

    fn spec(sample_size: usize, seed: u64) -> SamplingSpec {
        SamplingSpec::new(SamplingMode::Default, sample_size, seed)
    }

    struct Run {
        result: Result<TrainingSample, SamplingError>,
        selection_calls: usize,
        shuffle_calls: usize,
        trace: Vec<StepTrace>,
        final_weights: Vec<f32>,
        final_tickets: Vec<u64>,
    }

    fn run_with_next(
        units: &[SampleUnit],
        spec: SamplingSpec,
        limits: &SequentialLimits,
        mut next: impl FnMut(DrawKind, u64) -> u64,
    ) -> Run {
        let mut selection_calls = 0;
        let mut shuffle_calls = 0;
        let mut trace = Vec::new();
        let prepared = match preflight(units, spec, limits) {
            Ok(prepared) => prepared,
            Err(error) => {
                return Run {
                    result: Err(error),
                    selection_calls,
                    shuffle_calls,
                    trace,
                    final_weights: Vec::new(),
                    final_tickets: Vec::new(),
                };
            }
        };
        let mut plan = match build_plan(units, prepared) {
            Ok(plan) => plan,
            Err(error) => {
                return Run {
                    result: Err(error),
                    selection_calls,
                    shuffle_calls,
                    trace,
                    final_weights: Vec::new(),
                    final_tickets: Vec::new(),
                };
            }
        };
        let result = draw(
            units,
            spec,
            &mut plan,
            &mut next,
            &mut selection_calls,
            &mut shuffle_calls,
            Some(&mut trace),
        );
        let mut final_weights = Vec::new();
        let mut final_tickets = Vec::new();
        if result.is_ok() {
            compute_weights(units, &mut plan);
            final_weights = plan.weights.clone();
            let mut tickets = vec![0_u64; plan.tickets.len()];
            if compute_tickets(&plan.weights, plan.sample_size, &mut tickets).is_ok() {
                final_tickets = tickets;
            }
        }
        Run {
            result,
            selection_calls,
            shuffle_calls,
            trace,
            final_weights,
            final_tickets,
        }
    }

    fn expect_error(
        units: &[SampleUnit],
        spec: SamplingSpec,
        limits: &SequentialLimits,
    ) -> SamplingError {
        let run = run_with_next(units, spec, limits, |_, _| panic!("draw on rejection"));
        assert_eq!(run.selection_calls, 0, "selection draws on rejection");
        assert_eq!(run.shuffle_calls, 0, "shuffle draws on rejection");
        run.result.expect_err("expected a typed rejection")
    }

    fn checked_ticket_total(tickets: &[u64], draw_index: usize) -> Result<u64, SamplingError> {
        let mut total = 0_u64;
        for (unit, &ticket) in tickets.iter().enumerate() {
            total = total
                .checked_add(ticket)
                .ok_or(SamplingError::F32WeightQuantumOverflow { draw_index, unit })?;
        }
        Ok(total)
    }

    #[test]
    fn invalid_limits_and_mode_precedence() {
        let defaults = SequentialLimits::default();
        assert!(validate_limits(&defaults).is_ok());
        let cases: [(&'static str, u128, u128); 5] = [
            ("max_units", MAX_UNITS_CEILING, 1_048_576),
            ("max_support_spans", MAX_SUPPORT_SPANS_CEILING, 16_777_216),
            (
                "max_endpoint_scratch_bytes",
                MAX_ENDPOINT_SCRATCH_CEILING,
                268_435_456,
            ),
            ("max_expanded_rows", MAX_EXPANDED_ROWS_CEILING, 16_777_216),
            ("max_work_cells", MAX_WORK_CELLS_CEILING, 1_000_000_000_000),
        ];
        for (field, ceiling, _) in cases {
            let mut limits = defaults;
            match field {
                "max_units" => limits.max_units = 1,
                "max_support_spans" => limits.max_support_spans = 1,
                "max_endpoint_scratch_bytes" => limits.max_endpoint_scratch_bytes = 1,
                "max_expanded_rows" => limits.max_expanded_rows = 1,
                "max_work_cells" => limits.max_work_cells = 1,
                _ => unreachable!(),
            }
            assert!(validate_limits(&limits).is_ok(), "minimum for {field}");
            let mut limits = defaults;
            match field {
                "max_units" => limits.max_units = ceiling as usize,
                "max_support_spans" => limits.max_support_spans = ceiling as usize,
                "max_endpoint_scratch_bytes" => {
                    limits.max_endpoint_scratch_bytes = ceiling as usize
                }
                "max_expanded_rows" => limits.max_expanded_rows = ceiling as usize,
                "max_work_cells" => limits.max_work_cells = ceiling as u64,
                _ => unreachable!(),
            }
            assert!(validate_limits(&limits).is_ok(), "ceiling for {field}");

            let mut limits = defaults;
            match field {
                "max_units" => limits.max_units = 0,
                "max_support_spans" => limits.max_support_spans = 0,
                "max_endpoint_scratch_bytes" => limits.max_endpoint_scratch_bytes = 0,
                "max_expanded_rows" => limits.max_expanded_rows = 0,
                "max_work_cells" => limits.max_work_cells = 0,
                _ => unreachable!(),
            }
            assert_eq!(
                validate_limits(&limits),
                Err(SamplingError::InvalidLimits {
                    field,
                    value: 0,
                    min: 1,
                    max: ceiling,
                })
            );
            let mut limits = defaults;
            match field {
                "max_units" => limits.max_units = ceiling as usize + 1,
                "max_support_spans" => limits.max_support_spans = ceiling as usize + 1,
                "max_endpoint_scratch_bytes" => {
                    limits.max_endpoint_scratch_bytes = ceiling as usize + 1
                }
                "max_expanded_rows" => limits.max_expanded_rows = ceiling as usize + 1,
                "max_work_cells" => limits.max_work_cells = ceiling as u64 + 1,
                _ => unreachable!(),
            }
            assert_eq!(
                validate_limits(&limits),
                Err(SamplingError::InvalidLimits {
                    field,
                    value: ceiling + 1,
                    min: 1,
                    max: ceiling,
                })
            );
        }

        let mut limits = defaults;
        limits.max_units = 0;
        limits.max_support_spans = 0;
        assert_eq!(
            expect_error(
                &[],
                SamplingSpec::new(SamplingMode::Method(SampleMethod::WithReplacement), 0, 1),
                &limits,
            ),
            SamplingError::InvalidLimits {
                field: "max_units",
                value: 0,
                min: 1,
                max: 1_048_576,
            }
        );
        assert_eq!(
            expect_error(
                &[],
                SamplingSpec::new(SamplingMode::Method(SampleMethod::WithReplacement), 0, 1),
                &defaults,
            ),
            SamplingError::UnsupportedMode {
                mode: SamplingMode::Method(SampleMethod::WithReplacement),
            }
        );
        assert_eq!(
            expect_error(&[], spec(0, 1), &defaults),
            SamplingError::ZeroSampleSize
        );
        assert_eq!(
            expect_error(&[], spec(1, 1), &defaults),
            SamplingError::NoEligibleUnits
        );
        for mode in [
            SamplingMode::Disabled,
            SamplingMode::Method(SampleMethod::WithoutReplacement),
            SamplingMode::Method(SampleMethod::WithReplacement),
        ] {
            assert_eq!(
                expect_error(&valid_units(), SamplingSpec::new(mode, 1, 1), &defaults,),
                SamplingError::UnsupportedMode { mode }
            );
        }
    }

    #[test]
    fn shape_support_precedence() {
        let defaults = SequentialLimits::default();
        assert_eq!(
            expect_error(
                &[unit(&[0..2, 1..3], &[(0, 2)], 0), unit(&[], &[(0, 1)], 0)],
                spec(1, 1),
                &defaults,
            ),
            SamplingError::EmptyUnitRows {
                unit: 1,
                span: None,
            }
        );
        assert_eq!(
            expect_error(
                &[unit(&[0..2, 1..3], &[(0, 2)], 1), single(2, &[(2, 3)], 0),],
                spec(1, 1),
                &defaults,
            ),
            SamplingError::UnitRowsOutOfOrder {
                unit: 0,
                previous_end: 2,
                start: 1,
            }
        );
        assert_eq!(
            expect_error(
                &[
                    unit(&[0..2], &[(0, 1)], 0),
                    unit(&[1..3, 2..4], &[(1, 2)], 0),
                ],
                spec(1, 1),
                &defaults,
            ),
            SamplingError::UnitRowsOutOfOrder {
                unit: 1,
                previous_end: 2,
                start: 1,
            }
        );
        assert_eq!(
            expect_error(
                &[unit(&[0..1], &[(5, 5)], 1), single(1, &[(0, 1)], 0),],
                spec(1, 1),
                &defaults,
            ),
            SamplingError::SequenceOutOfOrder {
                unit: 1,
                previous: 1,
                actual: 0,
            }
        );
        assert_eq!(
            expect_error(
                &[unit(&[0..1], &[], 0), single(1, &[(5, 5)], 0),],
                spec(1, 1),
                &defaults,
            ),
            SamplingError::InvalidOutcomeInterval {
                unit: 1,
                span: 0,
                start: 5,
                end: 5,
            }
        );
        assert_eq!(
            expect_error(
                &[unit(&[0..1], &[(1, 4), (3, 6)], 0), single(1, &[], 0),],
                spec(1, 1),
                &defaults,
            ),
            SamplingError::OutcomeOutOfOrder {
                unit: 0,
                span: 1,
                previous_end: 4,
                start: 3,
            }
        );
        assert_eq!(
            expect_error(
                &[unit(&[0..1], &[(0, (L + 1) as i64)], 0), single(1, &[], 0),],
                spec(1, 1),
                &defaults,
            ),
            SamplingError::MissingOutcomeSupport { unit: 1 }
        );
        assert_eq!(
            expect_error(
                &[unit(&[0..1], &[(5, 5), (5, 6)], 0), single(1, &[(0, 1)], 0)],
                spec(1, 1),
                &defaults,
            ),
            SamplingError::InvalidOutcomeInterval {
                unit: 0,
                span: 0,
                start: 5,
                end: 5,
            }
        );
        let mut limits = defaults;
        limits.max_units = 1;
        assert_eq!(
            expect_error(
                &[unit(&[0..1, 7..3], &[(0, 1)], 0), single(1, &[(1, 2)], 0)],
                spec(1, 1),
                &limits,
            ),
            SamplingError::EmptyUnitRows {
                unit: 0,
                span: Some(1),
            }
        );
    }

    #[test]
    fn operational_caps_and_precedence() {
        let defaults = SequentialLimits::default();
        let two = [single(0, &[(0, 1)], 0), single(1, &[(1, 2)], 0)];

        let mut limits = defaults;
        limits.max_units = 1;
        assert_eq!(
            expect_error(&two, spec(1, 1), &limits),
            SamplingError::UnitLimitExceeded {
                computed: 2,
                limit: 1,
            }
        );

        let mut limits = defaults;
        limits.max_support_spans = 1;
        assert_eq!(
            expect_error(&[unit(&[0..1], &[(0, 1), (2, 3)], 0)], spec(1, 1), &limits,),
            SamplingError::SupportSpanLimitExceeded {
                computed: 2,
                limit: 1,
            }
        );

        let expanded = [unit(&[0..1, 2..3], &[(0, 1)], 0)];
        let mut limits = defaults;
        limits.max_expanded_rows = 4;
        let run = run_with_next(&expanded, spec(2, 1), &limits, |_, _| 0);
        assert_eq!(
            run.result.expect("expanded at limit").row_indices(),
            &[0, 2, 0, 2]
        );
        limits.max_expanded_rows = 3;
        assert_eq!(
            expect_error(&expanded, spec(2, 1), &limits),
            SamplingError::ExpandedRowsExceeded {
                computed: 4,
                limit: 3,
            }
        );

        assert_eq!(
            expect_error(
                &[unit(&[0..usize::MAX], &[(0, 1)], 0)],
                spec(2, 1),
                &defaults,
            ),
            SamplingError::ExpandedRowsExceeded {
                computed: 2 * (usize::MAX as u128),
                limit: 1_000_000,
            }
        );

        let identical = [single(0, &[(0, 1)], 0), single(1, &[(0, 1)], 0)];
        let mut limits = defaults;
        limits.max_endpoint_scratch_bytes = 512;
        assert!(
            run_with_next(&identical, spec(1, 1), &limits, |_, _| 0)
                .result
                .is_ok()
        );
        limits.max_endpoint_scratch_bytes = 511;
        assert_eq!(
            expect_error(&identical, spec(1, 1), &limits),
            SamplingError::EndpointScratchExceeded {
                computed: 512,
                limit: 511,
            }
        );

        let mut limits = defaults;
        limits.max_endpoint_scratch_bytes = 512;
        limits.max_work_cells = 4;
        assert_eq!(
            expect_error(&identical, spec(1, 1), &limits),
            SamplingError::WorkLimitExceeded {
                computed: 8,
                limit: 4,
            }
        );

        let mut limits = defaults;
        limits.max_units = 1;
        limits.max_support_spans = 1;
        assert_eq!(
            expect_error(&two, spec(2, 1), &limits),
            SamplingError::UnitLimitExceeded {
                computed: 2,
                limit: 1,
            }
        );
        let mut limits = defaults;
        limits.max_support_spans = 1;
        limits.max_endpoint_scratch_bytes = 511;
        assert_eq!(
            expect_error(&two, spec(2, 1), &limits),
            SamplingError::SupportSpanLimitExceeded {
                computed: 2,
                limit: 1,
            }
        );
        let mut limits = defaults;
        limits.max_endpoint_scratch_bytes = 511;
        limits.max_expanded_rows = 1;
        assert_eq!(
            expect_error(&two, spec(2, 1), &limits),
            SamplingError::EndpointScratchExceeded {
                computed: 512,
                limit: 511,
            }
        );
        let mut limits = defaults;
        limits.max_expanded_rows = 1;
        limits.max_work_cells = 4;
        assert_eq!(
            expect_error(&two, spec(2, 1), &limits),
            SamplingError::ExpandedRowsExceeded {
                computed: 2,
                limit: 1,
            }
        );

        let mut limits = defaults;
        limits.max_work_cells = 2;
        assert!(
            run_with_next(&[single(0, &[(0, 1)], 0)], spec(1, 1), &limits, |_, _| 0)
                .result
                .is_ok()
        );
        assert_eq!(
            expect_error(&[single(0, &[(0, 1)], 0)], spec(2, 1), &limits,),
            SamplingError::WorkLimitExceeded {
                computed: 4,
                limit: 2,
            }
        );

        let thousand: Vec<SampleUnit> = (0..1_000)
            .map(|index| single(index, &[((2 * index) as i64, (2 * index + 1) as i64)], 0))
            .collect();
        let mut limits = defaults;
        limits.max_work_cells = 1_000_000_000;
        assert!(preflight(&thousand, spec(500, 1), &limits).is_ok());
        assert_eq!(
            expect_error(&thousand, spec(501, 1), &limits),
            SamplingError::WorkLimitExceeded {
                computed: 1_002_000_000,
                limit: 1_000_000_000,
            }
        );
    }

    #[test]
    fn domain_boundaries() {
        let defaults = SequentialLimits::default();
        let width_l = [unit(&[0..1], &[(0, L as i64)], 0)];
        assert!(
            run_with_next(&width_l, spec(1, 1), &defaults, |_, _| 0)
                .result
                .is_ok()
        );
        assert_eq!(
            expect_error(
                &[unit(&[0..1], &[(0, (L + 1) as i64)], 0)],
                spec(1, 1),
                &defaults,
            ),
            SamplingError::F32SpanUnsupported {
                unit: 0,
                computed: L + 1,
                limit: L,
            }
        );
        let union_l = [unit(
            &[0..1],
            &[(0, (L / 2) as i64), ((L / 2 + 1) as i64, (L + 1) as i64)],
            0,
        )];
        assert!(
            run_with_next(&union_l, spec(1, 1), &defaults, |_, _| 0)
                .result
                .is_ok()
        );
        assert_eq!(
            expect_error(
                &[unit(
                    &[0..1],
                    &[(0, (L / 2) as i64), ((L / 2 + 1) as i64, (L + 2) as i64),],
                    0,
                )],
                spec(1, 1),
                &defaults,
            ),
            SamplingError::F32SpanUnsupported {
                unit: 0,
                computed: L + 1,
                limit: L,
            }
        );
        assert!(
            run_with_next(
                &[unit(&[0..1], &[(0, L as i64)], 0), single(1, &[(0, 1)], 0),],
                spec(1, 1),
                &defaults,
                |_, _| 0,
            )
            .result
            .is_ok()
        );

        let mut limits = defaults;
        limits.max_expanded_rows = L as usize;
        assert!(preflight(&valid_units(), spec(L as usize, 1), &limits).is_ok());
        assert_eq!(
            expect_error(&valid_units(), spec((L + 1) as usize, 1), &limits,),
            SamplingError::F32CountUnsupported {
                computed: L + 1,
                limit: L,
            }
        );
        assert_eq!(
            expect_error(
                &[unit(&[0..1], &[(0, (L + 1) as i64)], 0)],
                spec((L + 1) as usize, 1),
                &limits,
            ),
            SamplingError::F32CountUnsupported {
                computed: L + 1,
                limit: L,
            }
        );
        let mut scratch_limits = defaults;
        scratch_limits.max_endpoint_scratch_bytes = 1;
        assert_eq!(
            expect_error(
                &[unit(&[0..1], &[(0, (L + 1) as i64)], 0)],
                spec(1, 1),
                &scratch_limits,
            ),
            SamplingError::F32SpanUnsupported {
                unit: 0,
                computed: L + 1,
                limit: L,
            }
        );
        assert_eq!(
            expect_error(
                &[
                    unit(&[0..1], &[(0, (L + 1) as i64)], 0),
                    single(1, &[(0, (L + 2) as i64)], 0),
                ],
                spec(1, 1),
                &defaults,
            ),
            SamplingError::F32SpanUnsupported {
                unit: 0,
                computed: L + 1,
                limit: L,
            }
        );
        assert_eq!(
            expect_error(
                &[unit(&[0..1], &[(0, (1_i64 << 53) + 1)], 0)],
                spec((L + 1) as usize, 1),
                &limits,
            ),
            SamplingError::OutcomeSpanOverflow {
                computed: (1_u128 << 53) + 1,
                limit: 1_u128 << 53,
            }
        );
        assert_eq!(
            expect_error(
                &[
                    unit(&[0..1], &[(0, 1_i64 << 52)], 0),
                    single(1, &[(0, (1_i64 << 52) + 1)], 0),
                ],
                spec(1, 1),
                &defaults,
            ),
            SamplingError::OutcomeSpanOverflow {
                computed: (1_u128 << 53) + 1,
                limit: 1_u128 << 53,
            }
        );
        assert_eq!(
            concurrency_denominator((L - 1) as f32),
            L as f32,
            "c=L-1 is the last production denominator"
        );
        let negative = [unit(&[0..1], &[(-(L as i64) - 1, -(L as i64))], 0)];
        assert!(
            run_with_next(&negative, spec(1, 1), &defaults, |_, _| 0)
                .result
                .is_ok()
        );
        let near_max = [unit(&[0..1], &[(i64::MAX - 1, i64::MAX)], 0)];
        assert!(
            run_with_next(&near_max, spec(1, 1), &defaults, |_, _| 0)
                .result
                .is_ok()
        );
    }

    #[test]
    fn covered_segments_ignore_unused_gap() {
        let units = [
            single(0, &[(0, 1)], 0),
            single(1, &[((L + 2) as i64, (L + 3) as i64)], 0),
        ];
        let script = [0, Q / 2];
        let mut index = 0;
        let run = run_with_next(
            &units,
            spec(2, 1),
            &SequentialLimits::default(),
            |kind, upper| {
                assert_eq!(kind, DrawKind::Selection);
                let value = script[index];
                index += 1;
                assert!(value < upper);
                value
            },
        );
        let sample = run.result.expect("gap sample");
        assert_eq!(run.selection_calls, 2);
        assert_eq!(run.shuffle_calls, 0);
        assert_eq!(sample.row_indices(), &[0, 1]);
        assert_eq!(sample.unit_counts(), &[1, 1]);
        assert_eq!(run.trace[0].tickets, vec![Q, Q]);
        assert_eq!(run.trace[0].bound, 2 * Q);
        assert_eq!(run.trace[1].tickets, vec![Q / 2, Q]);
        assert_eq!(run.trace[1].bound, 3 * Q / 2);
    }

    #[test]
    fn realized_rows_reserve_only_selected_units() {
        // The largest unit (4 rows) is eligible, but the scripted draw selects
        // the shorter unit (1 row). Both start with weight 1 and ticket Q.
        let units = [unit(&[0..4], &[(0, 1)], 0), unit(&[4..5], &[(1, 2)], 0)];
        let run = run_with_next(
            &units,
            spec(1, 1),
            &SequentialLimits::default(),
            |kind, upper| {
                assert_eq!(kind, DrawKind::Selection);
                assert_eq!(upper, 2 * Q, "two equal tickets");
                Q
            },
        );
        let sample = run.result.expect("scripted short-unit selection");
        assert_eq!(sample.unit_counts(), &[0, 1], "short unit selected");
        assert_eq!(sample.row_indices(), &[4], "only the short unit's rows");
        // Realized reservation is 0*4 + 1*1 = 1, not the worst case 1*4 = 4.
        let realized = realized_row_count(sample.unit_counts(), &units);
        assert_eq!(realized, 1);
        assert_eq!(sample.row_indices().len(), realized);
        assert_eq!(checked_expanded_rows(&units, 1).expect("bound"), 4);
        assert!(realized < 4, "realized reservation beats the worst case");
        // Sizing skips zero-draw units and scales each unit by its final count.
        assert_eq!(realized_row_count(&[2, 0], &units), 2 * 4);
        assert_eq!(realized_row_count(&[2, 3], &units), 2 * 4 + 3);
    }

    #[test]
    fn cutoff_24_units_take_four_selection_draws_and_no_shuffle() {
        // Same literal geometry as
        // tests/data_sampler.rs::strict_f32_cutoff_24_matches_and_replays.
        let units = [
            unit(&[0..1], &[(1, 4)], 0),
            unit(&[1..3], &[(4, 7)], 0),
            unit(&[4..5], &[(12, 15)], 0),
            unit(&[6..7], &[(17, 20)], 0),
        ];
        let run = run_with_next(
            &units,
            spec(4, 21),
            &SequentialLimits::default(),
            |_, upper| {
                assert!(upper > 0, "draw bound must be positive");
                0
            },
        );
        let sample = run.result.expect("cutoff-24 strict sample");
        assert_eq!(run.selection_calls, 4, "selection draw requests");
        assert_eq!(run.shuffle_calls, 0, "shuffle draw requests");
        assert_eq!(sample.unit_counts().iter().sum::<usize>(), 4);
    }

    #[test]
    fn hard_bound_helpers() {
        assert!(matches!(
            checked_element_count(usize::MAX as u128),
            Ok(value) if value == usize::MAX
        ));
        assert_eq!(
            checked_element_count(usize::MAX as u128 + 1),
            Err(SamplingError::SampleLengthOverflow {
                computed: usize::MAX as u128 + 1,
                limit: usize::MAX as u128,
            })
        );
        let max_elements = (isize::MAX as u128) / (size_of::<i64>() as u128);
        assert!(checked_array_bytes(max_elements, size_of::<i64>()).is_ok());
        assert!(matches!(
            checked_array_bytes(max_elements + 1, size_of::<i64>()),
            Err(SamplingError::AllocationTooLarge { .. })
        ));
        assert_eq!(
            check_hard_capacity(
                &[(usize::MAX as u128, 1), (usize::MAX as u128 + 1, 1)],
                u64::MAX as u128,
            ),
            Err(SamplingError::SampleLengthOverflow {
                computed: usize::MAX as u128 + 1,
                limit: usize::MAX as u128,
            })
        );
        assert_eq!(
            check_hard_capacity(&[], u64::MAX as u128 + 1),
            Err(SamplingError::SamplingWorkOverflow {
                computed: u64::MAX as u128 + 1,
                limit: u64::MAX as u128,
            })
        );

        // Byte cap precedes the work cap when both fail in the same check.
        let byte_overflow_elements = (isize::MAX as u128) / (size_of::<i64>() as u128) + 1;
        let byte_overflow_bytes = byte_overflow_elements * size_of::<i64>() as u128;
        assert!(byte_overflow_bytes > isize::MAX as u128);
        assert_eq!(
            check_hard_capacity(
                &[(byte_overflow_elements, size_of::<i64>())],
                u64::MAX as u128 + 1,
            ),
            Err(SamplingError::AllocationTooLarge {
                bytes: byte_overflow_bytes,
                limit: isize::MAX as u128,
            })
        );

        // Two byte-class failures: the earlier buffer in the plan §3.5 order
        // wins with its own distinguishable payload.
        let later_elements = byte_overflow_elements + 1;
        let later_bytes = later_elements * size_of::<i64>() as u128;
        assert!(later_bytes > byte_overflow_bytes);
        assert_eq!(
            check_hard_capacity(
                &[
                    (byte_overflow_elements, size_of::<i64>()),
                    (later_elements, size_of::<i64>()),
                ],
                u64::MAX as u128,
            ),
            Err(SamplingError::AllocationTooLarge {
                bytes: byte_overflow_bytes,
                limit: isize::MAX as u128,
            })
        );

        assert_eq!(
            checked_endpoint_count(u128::MAX / 2 + 1),
            Err(SamplingError::ArithmeticOverflow {
                bound: "endpoint_count",
            })
        );
        assert_eq!(
            checked_expanded_product(2, u128::MAX),
            Err(SamplingError::ArithmeticOverflow {
                bound: "expanded_rows",
            })
        );
        assert_eq!(
            checked_work(u128::MAX, 1, 2),
            Err(SamplingError::ArithmeticOverflow {
                bound: "work_cells",
            })
        );
        assert_eq!(
            checked_scratch_bytes(u128::MAX, 1),
            Err(SamplingError::ArithmeticOverflow {
                bound: "endpoint_scratch_bytes",
            })
        );
    }

    #[test]
    fn defensive_weight_ticket_guards() {
        let mut tickets = [0_u64; 3];
        assert_eq!(
            compute_tickets(&[1.0, 0.0, 0.0], 2, &mut tickets),
            Err(SamplingError::F32WeightUnderflow {
                draw_index: 2,
                unit: 1,
            })
        );
        assert_eq!(
            checked_ticket_total(&[u64::MAX, 1], 2),
            Err(SamplingError::F32WeightQuantumOverflow {
                draw_index: 2,
                unit: 1,
            })
        );
        assert_eq!(ticket_for_weight(1.0, 1.0), Some(Q));
        assert_eq!(ticket_for_weight(0.5, 1.0), Some(Q / 2));
        assert_eq!(ticket_for_weight(5.0 * 2.0_f32.powi(-33), 1.0), Some(3));
        assert_eq!(ticket_for_weight(2.0_f32.powi(-34), 1.0), Some(1));
        assert_eq!(
            ticket_for_weight(f32::from_bits(0x3eaa_aaab), 0.5),
            Some(2_863_311_616)
        );
        assert_eq!(
            ticket_for_weight(f32::from_bits(0x3eaa_aaab), 1.0),
            Some(1_431_655_808)
        );
    }

    #[test]
    fn two_weight_and_selection_boundaries() {
        assert_eq!(
            ticket_for_weight(0.25, f32::from_bits(0x3eaa_aaab)),
            Some(3_221_225_376)
        );
        assert_eq!(select_unit(&[Q, Q], Q - 1), Some(0));
        assert_eq!(select_unit(&[Q, Q], Q), Some(1));
        assert_eq!(select_unit(&[Q, Q], 2 * Q - 1), Some(1));
        assert_eq!(select_unit(&[Q, 0, Q], 0), Some(0));
        assert_eq!(select_unit(&[Q, 0, Q], Q), Some(2));
        assert_eq!(select_unit(&[Q, 0, Q], 2 * Q - 1), Some(2));
        assert_eq!(ticket_for_weight(0.5, 0.75), Some(2_863_311_531));
    }

    #[test]
    fn overlap_union_controls() {
        let units = [unit(&[0..1], &[(0, 1), (2, 3)], 0), single(1, &[(1, 2)], 0)];
        let prepared = preflight(&units, spec(1, 1), &SequentialLimits::default()).unwrap();
        let mut plan = build_plan(&units, prepared).unwrap();
        for span in plan.unit_span_ranges[0].clone() {
            let (start, end) = plan.span_segments[span];
            for count in &mut plan.segment_counts[start..end] {
                *count += 1;
            }
        }
        compute_weights(&units, &mut plan);
        assert_eq!(plan.weights, vec![0.5, 1.0]);
        let mut tickets = vec![0_u64; 2];
        let total = compute_tickets(&plan.weights, 0, &mut tickets).unwrap();
        assert_eq!(tickets, vec![Q / 2, Q]);
        assert_eq!(total, 3 * Q / 2);

        let overlapping = [unit(&[0..1], &[(0, 2)], 0), single(1, &[(1, 3)], 0)];
        let prepared = preflight(&overlapping, spec(1, 1), &SequentialLimits::default()).unwrap();
        let mut plan = build_plan(&overlapping, prepared).unwrap();
        for span in plan.unit_span_ranges[0].clone() {
            let (start, end) = plan.span_segments[span];
            for count in &mut plan.segment_counts[start..end] {
                *count += 1;
            }
        }
        compute_weights(&overlapping, &mut plan);
        assert_eq!(plan.weights, vec![0.5, 0.75]);
        let mut tickets = vec![0_u64; 2];
        let total = compute_tickets(&plan.weights, 0, &mut tickets).unwrap();
        assert_eq!(tickets, vec![2_863_311_531, Q]);
        assert_eq!(total, 2_863_311_531 + Q);
        assert_eq!(select_unit(&tickets, 2_863_311_531), Some(1));
    }

    #[test]
    fn e0_31_scripted_ticket_trace() {
        let units = normalized_units();
        let script = [0, Q / 2, Q, 3 * Q / 2, 0];
        let mut index = 0;
        let run = run_with_next(
            &units,
            spec(5, 1),
            &SequentialLimits::default(),
            |kind, upper| {
                assert_eq!(kind, DrawKind::Selection);
                let value = script[index];
                index += 1;
                assert!(value < upper, "scripted value {value} outside [0,{upper})");
                value
            },
        );
        let sample = run.result.expect("E0-31 sample");
        let expected_weights = [
            vec![1.0, 1.0, 1.0, 1.0],
            vec![0.5, 1.0, 1.0, 1.0],
            vec![0.5, 0.5, 1.0, 1.0],
            vec![0.5, 0.5, 0.5, 1.0],
            vec![0.5, 0.5, 0.5, 0.5],
        ];
        let expected_tickets = [
            vec![Q, Q, Q, Q],
            vec![Q / 2, Q, Q, Q],
            vec![Q / 2, Q / 2, Q, Q],
            vec![Q / 2, Q / 2, Q / 2, Q],
            vec![Q, Q, Q, Q],
        ];
        let expected_bounds = [4 * Q, 7 * Q / 2, 3 * Q, 5 * Q / 2, 4 * Q];
        let expected_selected = [0, 1, 2, 3, 0];
        assert_eq!(run.trace.len(), 5);
        for index in 0..5 {
            let step = &run.trace[index];
            assert_eq!(step.draw_index, index);
            assert_eq!(step.weights, expected_weights[index]);
            assert_eq!(step.tickets, expected_tickets[index]);
            assert_eq!(step.bound, expected_bounds[index]);
            assert_eq!(
                step.bounded_draw, script[index],
                "scripted boundary is the bounded draw"
            );
            assert_eq!(step.selected, expected_selected[index]);
            println!(
                "E0-31 draw {} weights={:?} tickets={:?} bound={} draw={} selected={}",
                index + 1,
                step.weights,
                step.tickets,
                step.bound,
                step.bounded_draw,
                step.selected
            );
        }
        assert_eq!(run.selection_calls, 5);
        assert_eq!(run.shuffle_calls, 0);
        assert_eq!(sample.row_indices(), &[0, 0, 1, 2, 3]);
        assert_eq!(sample.unit_counts(), &[2, 1, 1, 1]);
        assert_eq!(run.final_tickets, vec![2_863_311_616, Q, Q, Q]);
        assert_eq!(run.final_weights, vec![1.0 / 3.0, 0.5, 0.5, 0.5]);
        println!(
            "E0-31 final tickets={:?} rows={:?} counts={:?}",
            run.final_tickets,
            sample.row_indices(),
            sample.unit_counts()
        );
    }

    #[test]
    fn e0_37_seeded_ticket_trace() {
        let units = normalized_units();
        let prepared = preflight(
            &units,
            spec(5, 0x1060_0000_0000_0001),
            &SequentialLimits::default(),
        )
        .unwrap();
        let mut plan = build_plan(&units, prepared).unwrap();
        let mut generator = RandomGenerator::from_seed(0x1060_0000_0000_0001);
        let mut selection_calls = 0;
        let mut shuffle_calls = 0;
        let mut trace = Vec::new();
        let sample = draw(
            &units,
            spec(5, 0x1060_0000_0000_0001),
            &mut plan,
            &mut |kind, upper| {
                assert_eq!(kind, DrawKind::Selection);
                generator.draw_u64_below(NonZeroU64::new(upper).unwrap())
            },
            &mut selection_calls,
            &mut shuffle_calls,
            Some(&mut trace),
        )
        .expect("E0-37 sample");
        let expected_weights = [
            vec![1.0, 1.0, 1.0, 1.0],
            vec![0.5, 1.0, 1.0, 1.0],
            vec![0.5, 1.0, 0.5, 1.0],
            vec![0.5, 1.0, 0.5, 0.5],
            vec![0.5, 1.0, 1.0 / 3.0, 0.5],
        ];
        let expected_tickets = [
            vec![Q, Q, Q, Q],
            vec![Q / 2, Q, Q, Q],
            vec![Q / 2, Q, Q / 2, Q],
            vec![Q / 2, Q, Q / 2, Q / 2],
            vec![Q / 2, Q, 1_431_655_808, Q / 2],
        ];
        let expected_bounds = [
            17_179_869_184,
            15_032_385_536,
            12_884_901_888,
            10_737_418_240,
            10_021_590_400,
        ];
        let expected_draws = [
            1_279_382_847,
            8_449_397_800,
            8_785_522_533,
            7_023_003_390,
            447_758_228,
        ];
        let expected_selected = [0, 2, 3, 2, 0];
        assert_eq!(trace.len(), 5);
        for index in 0..5 {
            let step = &trace[index];
            assert_eq!(step.draw_index, index);
            assert_eq!(step.weights, expected_weights[index]);
            assert_eq!(step.tickets, expected_tickets[index]);
            assert_eq!(step.bound, expected_bounds[index]);
            assert_eq!(step.bounded_draw, expected_draws[index]);
            assert_eq!(step.selected, expected_selected[index]);
            println!(
                "E0-37 draw {} weights={:?} tickets={:?} bound={} draw={} selected={}",
                index + 1,
                step.weights,
                step.tickets,
                step.bound,
                step.bounded_draw,
                step.selected
            );
        }
        assert_eq!(selection_calls, 5);
        assert_eq!(shuffle_calls, 0);
        assert_eq!(sample.row_indices(), &[0, 0, 2, 2, 3]);
        assert_eq!(sample.unit_counts(), &[2, 0, 2, 1]);
        compute_weights(&units, &mut plan);
        let mut final_tickets = vec![0_u64; plan.tickets.len()];
        compute_tickets(&plan.weights, plan.sample_size, &mut final_tickets).unwrap();
        assert_eq!(final_tickets, vec![1_431_655_808, Q, 1_431_655_808, Q / 2]);
        println!(
            "E0-37 final tickets={:?} rows={:?} counts={:?}",
            final_tickets,
            sample.row_indices(),
            sample.unit_counts()
        );
    }

    #[test]
    fn fragment_and_shuffle_presentation() {
        let units = [
            unit(&[0..1, 2..3], &[(0, 1), (2, 3)], 0),
            single(3, &[(3, 4)], 0),
            unit(&[4..5], &[(4, 5)], 1),
        ];
        let script = [0, Q / 2, 3 * Q / 2, 0];
        let mut index = 0;
        let run = run_with_next(
            &units,
            spec(4, 1),
            &SequentialLimits::default(),
            |kind, upper| {
                assert_eq!(kind, DrawKind::Selection);
                let value = script[index];
                index += 1;
                assert!(value < upper);
                value
            },
        );
        let sample = run.result.expect("fragment sample");
        assert_eq!(sample.row_indices(), &[0, 2, 0, 2, 3, 4]);
        assert_eq!(sample.unit_counts(), &[2, 1, 1]);
        assert_eq!(run.selection_calls, 4);
        assert_eq!(run.shuffle_calls, 0);

        let mut selection_index = 0;
        let mut shuffle_index = 0;
        let shuffle_script = [(3_u64, 0_u64), (2, 1)];
        let shuffled_run = run_with_next(
            &units,
            spec(4, 1).with_shuffle(true),
            &SequentialLimits::default(),
            |kind, upper| match kind {
                DrawKind::Selection => {
                    let value = script[selection_index];
                    selection_index += 1;
                    assert!(value < upper);
                    value
                }
                DrawKind::Shuffle => {
                    let (bound, value) = shuffle_script[shuffle_index];
                    shuffle_index += 1;
                    assert_eq!(upper, bound);
                    value
                }
            },
        );
        let shuffled = shuffled_run.result.expect("shuffled fragment");
        assert_eq!(shuffled_run.selection_calls, 4);
        assert_eq!(shuffled_run.shuffle_calls, 2);
        assert_eq!(shuffled.row_indices(), &[3, 0, 2, 0, 2, 4]);
        assert_eq!(shuffled.unit_counts(), &[2, 1, 1]);
    }

    #[test]
    fn zero_draws_on_caller_reachable_rejections() {
        let defaults = SequentialLimits::default();
        let mut limits = defaults;
        limits.max_units = 0;
        let cases: Vec<(
            Vec<SampleUnit>,
            SamplingSpec,
            SequentialLimits,
            SamplingError,
        )> = vec![
            (vec![], spec(1, 1), defaults, SamplingError::NoEligibleUnits),
            (
                vec![unit(&[], &[(0, 1)], 0)],
                spec(1, 1),
                defaults,
                SamplingError::EmptyUnitRows {
                    unit: 0,
                    span: None,
                },
            ),
            (
                vec![unit(&[0..1], &[(5, 5)], 0)],
                spec(1, 1),
                defaults,
                SamplingError::InvalidOutcomeInterval {
                    unit: 0,
                    span: 0,
                    start: 5,
                    end: 5,
                },
            ),
            (
                vec![single(0, &[], 0)],
                spec(1, 1),
                defaults,
                SamplingError::MissingOutcomeSupport { unit: 0 },
            ),
            (
                vec![unit(&[0..1], &[(0, (1_i64 << 53) + 1)], 0)],
                spec(1, 1),
                defaults,
                SamplingError::OutcomeSpanOverflow {
                    computed: (1_u128 << 53) + 1,
                    limit: 1_u128 << 53,
                },
            ),
            (
                vec![unit(&[0..1], &[(0, (L + 1) as i64)], 0)],
                spec(1, 1),
                defaults,
                SamplingError::F32SpanUnsupported {
                    unit: 0,
                    computed: L + 1,
                    limit: L,
                },
            ),
            (
                valid_units(),
                spec(1, 1).with_shuffle(true),
                limits,
                SamplingError::InvalidLimits {
                    field: "max_units",
                    value: 0,
                    min: 1,
                    max: 1_048_576,
                },
            ),
        ];
        for (units, spec, limits, expected) in cases {
            assert_eq!(expect_error(&units, spec, &limits), expected);
        }
    }

    #[test]
    fn touching_and_identical_supports_are_disjoint() {
        let touching = [unit(&[0..1], &[(0, 1)], 0), single(1, &[(1, 2)], 0)];
        let prepared = preflight(&touching, spec(1, 1), &SequentialLimits::default()).unwrap();
        let mut plan = build_plan(&touching, prepared).unwrap();
        for span in plan.unit_span_ranges[0].clone() {
            let (start, end) = plan.span_segments[span];
            for count in &mut plan.segment_counts[start..end] {
                *count += 1;
            }
        }
        compute_weights(&touching, &mut plan);
        assert_eq!(plan.weights, vec![0.5, 1.0]);
        let mut tickets = vec![0_u64; 2];
        compute_tickets(&plan.weights, 0, &mut tickets).unwrap();
        assert_eq!(tickets, vec![Q / 2, Q]);

        let identical = [unit(&[0..1], &[(0, 1)], 0), single(1, &[(0, 1)], 0)];
        let prepared = preflight(&identical, spec(1, 1), &SequentialLimits::default()).unwrap();
        let mut plan = build_plan(&identical, prepared).unwrap();
        for span in plan.unit_span_ranges[0].clone() {
            let (start, end) = plan.span_segments[span];
            for count in &mut plan.segment_counts[start..end] {
                *count += 1;
            }
        }
        compute_weights(&identical, &mut plan);
        assert_eq!(plan.weights, vec![0.5, 0.5]);
        let mut tickets = vec![0_u64; 2];
        compute_tickets(&plan.weights, 0, &mut tickets).unwrap();
        assert_eq!(tickets, vec![Q, Q]);
    }

    #[test]
    fn width_model_keeps_work_above_u32() {
        let work = 5_000_000_000_u128;
        assert!(work > u32::MAX as u128);
        assert_eq!(checked_work(2_500_000_000, 1, 2), Ok(work));
        let limits = SequentialLimits {
            max_work_cells: (u32::MAX as u64) + 1,
            ..SequentialLimits::default()
        };
        assert!(validate_limits(&limits).is_ok());
        assert_eq!(
            check_hard_capacity(&[], u64::MAX as u128 + 1),
            Err(SamplingError::SamplingWorkOverflow {
                computed: u64::MAX as u128 + 1,
                limit: u64::MAX as u128,
            })
        );
    }

    #[cfg(target_pointer_width = "64")]
    #[test]
    fn strict_rejects_legacy_large_draw_count_before_work() {
        assert_eq!(
            expect_error(
                &valid_units(),
                spec(1_usize << 59, 1),
                &SequentialLimits::default(),
            ),
            SamplingError::F32CountUnsupported {
                computed: 1_u128 << 59,
                limit: L,
            }
        );
    }

    #[test]
    fn nineteen_unit_reachability_keeps_positive_tickets() {
        let units: Vec<SampleUnit> = (0..19)
            .map(|index| {
                let support = if index == 0 || index == 17 {
                    (0, 1)
                } else {
                    (2, 3)
                };
                single(index, &[support], 0)
            })
            .collect();
        let sample_size = (1_usize << 20) + 1;
        let limits = SequentialLimits {
            max_expanded_rows: sample_size,
            ..SequentialLimits::default()
        };
        let spec = spec(sample_size, 1);

        assert_eq!(
            expect_error(&units, spec, &SequentialLimits::default()),
            SamplingError::ExpandedRowsExceeded {
                computed: sample_size as u128,
                limit: 1_000_000,
            }
        );

        let prepared = preflight(&units, spec, &limits).unwrap();
        let mut plan = build_plan(&units, prepared).unwrap();
        let mut calls = 0_usize;
        let mut selection_calls = 0_usize;
        let mut shuffle_calls = 0_usize;

        // 2^20 zero draws select unit 0 each time, leaving shared support A at
        // concurrency 2^20. The exact pre-draw history is weight 1/1048577 and
        // tickets 4096 for units 0 and 17, Q for every B unit.
        plan.sample_size = 1 << 20;
        let partial = draw(
            &units,
            spec,
            &mut plan,
            &mut |kind, upper| {
                assert_eq!(kind, DrawKind::Selection);
                calls += 1;
                assert!(0 < upper);
                0
            },
            &mut selection_calls,
            &mut shuffle_calls,
            None,
        )
        .expect("reachability prefix");
        assert_eq!(partial.unit_counts()[0], 1 << 20);
        assert!(partial.row_indices().iter().all(|&row| row == 0));
        compute_weights(&units, &mut plan);
        let mut pre_tickets = vec![0_u64; plan.tickets.len()];
        let pre_total = compute_tickets(&plan.weights, plan.sample_size, &mut pre_tickets).unwrap();
        assert_eq!(plan.weights[0], 1.0 / 1_048_577.0);
        assert_eq!(plan.weights[17], 1.0 / 1_048_577.0);
        assert_eq!(plan.weights[1], 1.0);
        assert_eq!(pre_tickets[0], 4096);
        assert_eq!(pre_tickets[17], 4096);
        assert_eq!(pre_tickets[1], Q);
        assert_eq!(pre_total, 17 * Q + 2 * 4096);

        // `draw` now moves the completed counts into the sample, so carry the
        // prefix counts forward to simulate the second half of one long run.
        plan.counts = partial.unit_counts().to_vec();

        // The exact start of unit 17's interval lies at 16Q+4096.
        plan.sample_size = 1;
        let final_draw = 16 * Q + 4096;
        let final_sample = draw(
            &units,
            spec,
            &mut plan,
            &mut |kind, upper| {
                assert_eq!(kind, DrawKind::Selection);
                calls += 1;
                assert!(final_draw < upper);
                final_draw
            },
            &mut selection_calls,
            &mut shuffle_calls,
            None,
        )
        .expect("reachability final draw");
        assert_eq!(selection_calls, sample_size);
        assert_eq!(shuffle_calls, 0);
        assert_eq!(calls, sample_size);
        let mut expected_counts = vec![0_usize; 19];
        expected_counts[0] = 1 << 20;
        expected_counts[17] = 1;
        assert_eq!(final_sample.unit_counts(), expected_counts.as_slice());
        assert_eq!(final_sample.len(), sample_size);
        assert!(
            final_sample.row_indices()[..(1 << 20)]
                .iter()
                .all(|&row| row == 0)
        );
        assert_eq!(*final_sample.row_indices().last().unwrap(), 17);

        compute_weights(&units, &mut plan);
        let mut tickets = vec![0_u64; plan.tickets.len()];
        let total = compute_tickets(&plan.weights, plan.sample_size, &mut tickets).unwrap();
        assert_eq!(plan.weights[0], 1.0 / 1_048_578.0);
        assert_eq!(plan.weights[17], 1.0 / 1_048_578.0);
        assert_eq!(tickets[0], 4096);
        assert_eq!(tickets[17], 4096);
        assert_eq!(tickets[1], Q);
        assert_eq!(total, 17 * Q + 2 * 4096);
    }
}
