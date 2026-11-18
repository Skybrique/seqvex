//! Strict-f32 recurrence and execution evidence.

use num_rational::BigRational;
use num_traits::Zero;

use seqvex::execution::streaming::StreamingExecutor;
use seqvex::foundation::error::FailureCause;
use seqvex::foundation::numerical::Vector;
use seqvex::foundation::state::{StateModel, process_batch, process_one};
use seqvex::models::linear::sgd::{Sgd, SgdError, SgdState};

use super::support::reference_update32_sum;
use crate::{TRAJECTORY_PATTERN, as_f64, bits, rat, sample};

fn run_local_bounded32<const N: usize>(
    model: &Sgd,
    mut state: SgdState,
    rows: &[([f32; N], f32)],
    max_batch: usize,
) -> Result<SgdState, (usize, SgdState, SgdError)> {
    for (index, (features, target)) in rows.iter().enumerate() {
        if index >= max_batch {
            return Ok(state);
        }
        match model.update(&state, &sample(features, *target)) {
            Ok(next) => state = next,
            Err(error) => return Err((index, state, error)),
        }
    }
    Ok(state)
}

struct OrderTrace32 {
    initial: Vec<u32>,
    steps: Vec<(u32, Vec<u32>)>,
}

fn order16_rows32() -> Vec<([f32; 3], f32)> {
    (0..16)
        .map(|step| TRAJECTORY_PATTERN[step % TRAJECTORY_PATTERN.len()])
        .collect()
}

fn order16_trace32() -> OrderTrace32 {
    let mut state = vec![0.0f32; 3];
    let initial = state.iter().map(|value| value.to_bits()).collect();
    let mut steps = Vec::with_capacity(16);
    for (x, y) in order16_rows32() {
        let prediction: f32 = state.iter().zip(x).map(|(w, x)| w * x).sum();
        state = reference_update32_sum(0.125, &state, &x, y);
        steps.push((
            prediction.to_bits(),
            state.iter().map(|value| value.to_bits()).collect(),
        ));
    }
    OrderTrace32 { initial, steps }
}

fn expected_boundary32(trace: &OrderTrace32, boundary: usize) -> &[u32] {
    if boundary == 0 {
        &trace.initial
    } else {
        &trace.steps[boundary - 1].1
    }
}

#[test]
fn c_order16_every_step_bits_and_rational() {
    let eta = 0.125f32;
    let model = Sgd::new(3, eta).expect("valid configuration");
    let mut state = model.initial_state();
    let mut reference = vec![0.0f32; 3];
    let mut exact = [
        BigRational::zero(),
        BigRational::zero(),
        BigRational::zero(),
    ];
    let eta_rat = rat(eta);

    for step in 0..16 {
        let (x, y) = TRAJECTORY_PATTERN[step % TRAJECTORY_PATTERN.len()];

        let predicted = model
            .predict(&state, &Vector::from_slice(&x))
            .expect("finite prediction");
        let reference_dot: f32 = reference.iter().zip(x).map(|(w, x)| w * x).sum();
        assert_eq!(
            predicted.to_bits(),
            reference_dot.to_bits(),
            "step {step}: pre-update prediction must be bitwise"
        );

        let mut dot = BigRational::zero();
        for j in 0..3 {
            dot = &dot + &(&exact[j] * &rat(x[j]));
        }
        let error = &rat(y) - &dot;
        let scale = &eta_rat * &error;
        for j in 0..3 {
            exact[j] = &exact[j] + &(&scale * &rat(x[j]));
        }

        reference = reference_update32_sum(eta, &reference, &x, y);
        state = model
            .update(&state, &sample(&x, y))
            .expect("ordered update");

        for j in 0..3 {
            assert_eq!(
                state.w.as_slice()[j].to_bits(),
                reference[j].to_bits(),
                "step {step} component {j}: scalar recurrence must match bitwise"
            );
            let actual = f64::from(state.w.as_slice()[j]);
            let expected = as_f64(&exact[j]);
            let tolerance = 1e-6 * expected.abs().max(1.0);
            assert!(
                (actual - expected).abs() <= tolerance,
                "step {step} component {j}: {actual} vs exact {expected} exceeds {tolerance}"
            );
        }

        let norm_sq: f64 = exact.iter().map(|value| as_f64(value).powi(2)).sum();
        assert!(norm_sq.sqrt() < 0.15, "step {step}: exact norm envelope");
    }
}

#[test]
fn c_execution_equivalence() {
    let model = Sgd::new(3, 0.125).expect("valid configuration");
    let rows = order16_rows32();
    let trace = order16_trace32();

    let initial = model.initial_state();
    assert_eq!(bits(&initial.w), trace.initial);

    let mut direct = model.initial_state();
    for (step, (x, y)) in rows.iter().enumerate() {
        let prediction = model
            .predict(&direct, &Vector::from_slice(x))
            .expect("direct prediction");
        assert_eq!(
            prediction.to_bits(),
            trace.steps[step].0,
            "direct step {step} incoming prediction"
        );
        direct = model
            .update(&direct, &sample(x, *y))
            .expect("direct update");
        assert_eq!(
            bits(&direct.w),
            trace.steps[step].1,
            "direct step {step} state"
        );
    }

    let mut single = model.initial_state();
    for (step, (x, y)) in rows.iter().enumerate() {
        let prediction = model
            .predict(&single, &Vector::from_slice(x))
            .expect("process_one prediction");
        assert_eq!(prediction.to_bits(), trace.steps[step].0);
        single = process_one(&model, &single, &sample(x, *y)).expect("process_one");
        assert_eq!(bits(&single.w), trace.steps[step].1);
    }

    let mut executor = StreamingExecutor::new(&model, model.initial_state());
    for (step, (x, y)) in rows.iter().enumerate() {
        let prediction = model
            .predict(executor.state(), &Vector::from_slice(x))
            .expect("executor prediction");
        assert_eq!(prediction.to_bits(), trace.steps[step].0);
        executor
            .process_one(&sample(x, *y))
            .expect("executor update");
        assert_eq!(bits(&executor.state().w), trace.steps[step].1);
    }

    let observations: Vec<_> = rows.iter().map(|(x, y)| sample(x, *y)).collect();
    let batch = process_batch(&model, model.initial_state(), observations).expect("batch");
    assert_eq!(bits(&batch.w), trace.steps[15].1);

    let boundaries = [0usize, 3, 3, 6, 9, 12, 15, 16];
    let mut chunked = model.initial_state();
    for pair in boundaries.windows(2) {
        let (start, end) = (pair[0], pair[1]);
        let observations: Vec<_> = rows[start..end]
            .iter()
            .map(|(x, y)| sample(x, *y))
            .collect();
        chunked = process_batch(&model, chunked, observations).expect("chunk boundary");
        assert_eq!(
            bits(&chunked.w),
            expected_boundary32(&trace, end),
            "chunk boundary {end}"
        );
    }

    let bounded = run_local_bounded32(&model, model.initial_state(), &rows, 7)
        .expect("local bounded success");
    assert_eq!(bits(&bounded.w), trace.steps[6].1);

    let mut failure_rows = rows.clone();
    failure_rows[5].1 = f32::NAN;
    let (index, prefix_state, error) =
        run_local_bounded32(&model, model.initial_state(), &failure_rows, 16)
            .expect_err("local bounded failure");
    assert_eq!(index, 5);
    assert_eq!(bits(&prefix_state.w), expected_boundary32(&trace, 5));
    assert_eq!(error, SgdError::NonFiniteInput);

    let failure_obs: Vec<_> = rows[..5]
        .iter()
        .map(|(x, y)| sample(x, *y))
        .chain(std::iter::once(sample(&rows[5].0, f32::NAN)))
        .collect();
    let prefix = process_batch(&model, model.initial_state(), failure_obs)
        .expect_err("batch prefix failure");
    assert_eq!(bits(&prefix.state.w), expected_boundary32(&trace, 5));
    assert_eq!(prefix.cause, FailureCause::Model(SgdError::NonFiniteInput));

    let mut errors = 0usize;
    let mut executor = StreamingExecutor::new(&model, model.initial_state());
    let progress = executor
        .process_stream_continuing(
            vec![
                sample(&rows[0].0, rows[0].1),
                sample(&[0.0, 0.0, 0.0], f32::NAN),
                sample(&rows[2].0, rows[2].1),
            ],
            |position, _| {
                assert_eq!(position, 1);
                errors += 1;
            },
        )
        .unwrap();
    assert_eq!(errors, 1);
    assert_eq!(progress.attempted, 3);
    assert_eq!(progress.committed, 2);
    assert_eq!(progress.failed, 1);
    let stream_state = executor.state().clone();
    let expected_stream = reference_update32_sum(
        0.125,
        &reference_update32_sum(0.125, &[0.0; 3], &rows[0].0, rows[0].1),
        &rows[2].0,
        rows[2].1,
    );
    assert_eq!(
        bits(&stream_state.w),
        expected_stream
            .iter()
            .map(|v| v.to_bits())
            .collect::<Vec<_>>()
    );
}
