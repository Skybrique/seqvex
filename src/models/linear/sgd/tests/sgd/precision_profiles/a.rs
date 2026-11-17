//! Native-f64 recurrence, ordering and execution evidence.

use num_rational::BigRational;
use num_traits::Zero;

use seqvex::execution::streaming::StreamingExecutor;
use seqvex::foundation::error::FailureCause;
use seqvex::foundation::numerical::Vector64;
use seqvex::foundation::state::{StateModel, process_batch, process_one};
use seqvex::models::linear::sgd::{Sgd, Sgd64, SgdError64, SgdState64};

use super::support::{bits64, reference_update64, sample64};
use crate::{as_f64, sample};

const TRAJECTORY_PATTERN64: [([f64; 3], f64); 4] = [
    ([0.5, 0.5, 1.0], 0.25),
    ([-0.5, 0.5, 1.0], 0.0),
    ([-0.5, -0.5, 1.0], -0.25),
    ([0.5, -0.5, 1.0], 0.0),
];

fn rat64(value: f64) -> BigRational {
    BigRational::from_float(value).expect("finite fixture")
}

fn run_local_bounded64<const N: usize>(
    model: &Sgd64,
    mut state: SgdState64,
    rows: &[([f64; N], f64)],
    max_batch: usize,
) -> Result<SgdState64, (usize, SgdState64, SgdError64)> {
    for (index, (features, target)) in rows.iter().enumerate() {
        if index >= max_batch {
            return Ok(state);
        }
        match model.update(&state, &sample64(features, *target)) {
            Ok(next) => state = next,
            Err(error) => return Err((index, state, error)),
        }
    }
    Ok(state)
}

struct OrderTrace64 {
    initial: Vec<u64>,
    steps: Vec<(u64, Vec<u64>)>,
}

fn order16_rows64() -> Vec<([f64; 3], f64)> {
    (0..16)
        .map(|step| TRAJECTORY_PATTERN64[step % TRAJECTORY_PATTERN64.len()])
        .collect()
}

fn order16_trace64() -> OrderTrace64 {
    let mut state = vec![0.0f64; 3];
    let initial = state.iter().map(|value| value.to_bits()).collect();
    let mut steps = Vec::with_capacity(16);
    for (x, y) in order16_rows64() {
        let prediction: f64 = state.iter().zip(x).map(|(w, x)| w * x).sum();
        state = reference_update64(0.125, &state, &x, y);
        steps.push((
            prediction.to_bits(),
            state.iter().map(|value| value.to_bits()).collect(),
        ));
    }
    OrderTrace64 { initial, steps }
}

fn expected_boundary64(trace: &OrderTrace64, boundary: usize) -> &[u64] {
    if boundary == 0 {
        &trace.initial
    } else {
        &trace.steps[boundary - 1].1
    }
}

#[test]
fn a_hand_derived_single_update() {
    let model = Sgd64::new(1, 0.1).expect("valid configuration");
    let next = model
        .update(&model.initial_state(), &sample64(&[2.0], 4.0))
        .expect("single update");
    assert_eq!(next.w.as_slice()[0].to_bits(), 0.8f64.to_bits());
}

#[test]
fn a_order16_every_step_bits_and_rational() {
    let eta = 0.125f64;
    let model = Sgd64::new(3, eta).expect("valid configuration");
    let mut state = model.initial_state();
    let mut reference = vec![0.0f64; 3];
    let mut exact = [
        BigRational::zero(),
        BigRational::zero(),
        BigRational::zero(),
    ];
    let eta_rat = rat64(eta);

    for step in 0..16 {
        let (x, y) = TRAJECTORY_PATTERN64[step % TRAJECTORY_PATTERN64.len()];

        let predicted = model
            .predict(&state, &Vector64::from_slice(&x))
            .expect("finite prediction");
        let reference_dot: f64 = reference.iter().zip(x).map(|(w, x)| w * x).sum();
        assert_eq!(
            predicted.to_bits(),
            reference_dot.to_bits(),
            "step {step}: pre-update prediction must be bitwise"
        );

        let mut dot = BigRational::zero();
        for j in 0..3 {
            dot = &dot + &(&exact[j] * &rat64(x[j]));
        }
        let error = &rat64(y) - &dot;
        let scale = &eta_rat * &error;
        for j in 0..3 {
            exact[j] = &exact[j] + &(&scale * &rat64(x[j]));
        }

        reference = reference_update64(eta, &reference, &x, y);
        state = model
            .update(&state, &sample64(&x, y))
            .expect("ordered update");

        for j in 0..3 {
            assert_eq!(
                state.w.as_slice()[j].to_bits(),
                reference[j].to_bits(),
                "step {step} component {j}: scalar recurrence must match bitwise"
            );
            let actual = state.w.as_slice()[j];
            let expected = as_f64(&exact[j]);
            let tolerance = 1e-12 * expected.abs().max(1.0);
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
fn a_reduction_order_and_grouping_controls() {
    let weights = [18_014_398_509_481_984.0f64, -18_014_398_509_481_984.0, 1.0];
    let features = [1.0f64, 1.0, 1.0];
    let model = Sgd64::new(3, 0.1).expect("valid configuration");
    let state = SgdState64 {
        w: Vector64::from_slice(&weights),
    };

    let mut sequential = 0.0f64;
    for (weight, feature) in weights.iter().zip(features) {
        sequential += weight * feature;
    }
    let mut reverse = 0.0f64;
    for (weight, feature) in weights.iter().zip(features).rev() {
        reverse += weight * feature;
    }
    assert_ne!(
        sequential.to_bits(),
        reverse.to_bits(),
        "fixture must distinguish f64 reduction order"
    );
    let prediction = model
        .predict(&state, &Vector64::from_slice(&features))
        .expect("finite prediction");
    assert_eq!(prediction.to_bits(), sequential.to_bits());
    assert_ne!(prediction.to_bits(), reverse.to_bits());

    let eta = 0.1f64;
    let target = 0.3f64;
    let feature = 0.7f64;
    let model = Sgd64::new(1, eta).expect("valid configuration");
    let state = model.initial_state();
    let approved = (eta * (target - 0.0)) * feature;
    let alternative = eta * ((target - 0.0) * feature);
    assert_ne!(
        approved.to_bits(),
        alternative.to_bits(),
        "fixture must distinguish f64 multiplication grouping"
    );
    let next = model
        .update(&state, &sample64(&[feature], target))
        .expect("update");
    assert_eq!(next.w.as_slice()[0].to_bits(), approved.to_bits());
}

#[test]
fn native_f64_discriminator_and_b_snapshot() {
    let model = Sgd64::new(1, 1.0).expect("valid configuration");
    let mut state = model.initial_state();
    state = model
        .update(&state, &sample64(&[1.0], 0.0))
        .expect("warm-up");
    state = model
        .update(&state, &sample64(&[1.0 + 2f64.powi(-24)], 2f64.powi(24)))
        .expect("native f64 update");
    assert_eq!(state.w.as_slice()[0].to_bits(), 16_777_217.0f64.to_bits());

    let f32_model = Sgd::new(1, 1.0).expect("valid configuration");
    let mut f32_state = f32_model.initial_state();
    f32_state = f32_model
        .update(&f32_state, &sample(&[1.0], 0.0))
        .expect("warm-up");
    f32_state = f32_model
        .update(&f32_state, &sample(&[1.0], 2f32.powi(24)))
        .expect("f32 update");
    assert_eq!(
        f32_state.w.as_slice()[0].to_bits(),
        16_777_216.0f32.to_bits()
    );
    assert!(
        (state.w.as_slice()[0] - f64::from(f32_state.w.as_slice()[0])).abs() > 0.5,
        "native f64 must differ from the f32-quantized trajectory"
    );

    let snapshot = model.to_serving_f32(&state).expect("serving snapshot");
    assert_eq!(
        snapshot.weights().as_slice()[0].to_bits(),
        16_777_216.0f32.to_bits()
    );
}

#[test]
fn a_execution_equivalence_and_prefix() {
    let model = Sgd64::new(3, 0.125).expect("valid configuration");
    let rows = order16_rows64();
    let trace = order16_trace64();

    let initial = model.initial_state();
    assert_eq!(bits64(&initial.w), trace.initial);

    let mut direct = model.initial_state();
    for (step, (x, y)) in rows.iter().enumerate() {
        let prediction = model
            .predict(&direct, &Vector64::from_slice(x))
            .expect("direct prediction");
        assert_eq!(
            prediction.to_bits(),
            trace.steps[step].0,
            "direct step {step} incoming prediction"
        );
        direct = model
            .update(&direct, &sample64(x, *y))
            .expect("direct update");
        assert_eq!(
            bits64(&direct.w),
            trace.steps[step].1,
            "direct step {step} state"
        );
    }

    let mut single = model.initial_state();
    for (step, (x, y)) in rows.iter().enumerate() {
        let prediction = model
            .predict(&single, &Vector64::from_slice(x))
            .expect("process_one prediction");
        assert_eq!(prediction.to_bits(), trace.steps[step].0);
        single = process_one(&model, &single, &sample64(x, *y)).expect("process_one");
        assert_eq!(bits64(&single.w), trace.steps[step].1);
    }

    let mut executor = StreamingExecutor::new(&model, model.initial_state());
    for (step, (x, y)) in rows.iter().enumerate() {
        let prediction = model
            .predict(executor.state(), &Vector64::from_slice(x))
            .expect("executor prediction");
        assert_eq!(prediction.to_bits(), trace.steps[step].0);
        executor
            .process_one(&sample64(x, *y))
            .expect("executor update");
        assert_eq!(bits64(&executor.state().w), trace.steps[step].1);
    }

    let observations: Vec<_> = rows.iter().map(|(x, y)| sample64(x, *y)).collect();
    let batch = process_batch(&model, model.initial_state(), observations).expect("batch");
    assert_eq!(bits64(&batch.w), trace.steps[15].1);

    let boundaries = [0usize, 3, 3, 6, 9, 12, 15, 16];
    let mut chunked = model.initial_state();
    for pair in boundaries.windows(2) {
        let (start, end) = (pair[0], pair[1]);
        let observations: Vec<_> = rows[start..end]
            .iter()
            .map(|(x, y)| sample64(x, *y))
            .collect();
        chunked = process_batch(&model, chunked, observations).expect("chunk boundary");
        assert_eq!(
            bits64(&chunked.w),
            expected_boundary64(&trace, end),
            "chunk boundary {end}"
        );
    }

    let bounded = run_local_bounded64(&model, model.initial_state(), &rows, 7)
        .expect("local bounded success");
    assert_eq!(bits64(&bounded.w), trace.steps[6].1);

    let mut failure_rows = rows.clone();
    failure_rows[5].1 = f64::NAN;
    let (index, prefix_state, error) =
        run_local_bounded64(&model, model.initial_state(), &failure_rows, 16)
            .expect_err("local bounded failure");
    assert_eq!(index, 5);
    assert_eq!(bits64(&prefix_state.w), expected_boundary64(&trace, 5));
    assert_eq!(error, SgdError64::NonFiniteInput);

    let failure_obs: Vec<_> = rows[..5]
        .iter()
        .map(|(x, y)| sample64(x, *y))
        .chain(std::iter::once(sample64(&rows[5].0, f64::NAN)))
        .collect();
    let prefix = process_batch(&model, model.initial_state(), failure_obs)
        .expect_err("batch prefix failure");
    assert_eq!(bits64(&prefix.state.w), expected_boundary64(&trace, 5));
    assert_eq!(
        prefix.cause,
        FailureCause::Model(SgdError64::NonFiniteInput)
    );

    let mut errors = 0usize;
    let mut executor = StreamingExecutor::new(&model, model.initial_state());
    let progress = executor
        .process_stream_continuing(
            vec![
                sample64(&rows[0].0, rows[0].1),
                sample64(&[0.0, 0.0, 0.0], f64::NAN),
                sample64(&rows[2].0, rows[2].1),
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
    let expected_stream = reference_update64(
        0.125,
        &reference_update64(0.125, &[0.0; 3], &rows[0].0, rows[0].1),
        &rows[2].0,
        rows[2].1,
    );
    assert_eq!(
        bits64(&stream_state.w),
        expected_stream
            .iter()
            .map(|v| v.to_bits())
            .collect::<Vec<_>>()
    );
}

#[test]
fn a_order_and_fixed_inference() {
    let model = Sgd64::new(2, 0.1).expect("valid configuration");
    let a = ([1.0f64, 1.0], 1.0f64);
    let b = ([2.0f64, -1.0], -1.0f64);

    let ab = model
        .update(
            &model
                .update(&model.initial_state(), &sample64(&a.0, a.1))
                .expect("a"),
            &sample64(&b.0, b.1),
        )
        .expect("b");
    let ba = model
        .update(
            &model
                .update(&model.initial_state(), &sample64(&b.0, b.1))
                .expect("b"),
            &sample64(&a.0, a.1),
        )
        .expect("a");
    assert_ne!(bits64(&ab.w), bits64(&ba.w));

    let repeat = model
        .update(
            &model
                .update(&model.initial_state(), &sample64(&a.0, a.1))
                .expect("a"),
            &sample64(&b.0, b.1),
        )
        .expect("b");
    assert_eq!(bits64(&ab.w), bits64(&repeat.w));

    let features = Vector64::from_slice(&[0.5, -0.5]);
    let state_bits = bits64(&ab.w);
    let first = model.predict(&ab, &features).expect("prediction");
    let second = model.predict(&ab, &features).expect("prediction");
    assert_eq!(first.to_bits(), second.to_bits());
    assert_eq!(bits64(&ab.w), state_bits);
}

#[test]
fn a_stream_isolation() {
    let model = Sgd64::new(2, 0.1).expect("valid configuration");
    let mut first = model.initial_state();
    let mut second = model.initial_state();

    let second_bits = bits64(&second.w);
    first = model
        .update(&first, &sample64(&[1.0, -1.0], 0.5))
        .expect("first stream update");
    assert_eq!(bits64(&second.w), second_bits);

    let first_bits = bits64(&first.w);
    second = model
        .update(&second, &sample64(&[1.0, 1.0], 1.0))
        .expect("second stream update");
    assert_eq!(bits64(&first.w), first_bits);
    assert_ne!(bits64(&first.w), bits64(&second.w));

    let expected_first = model
        .update(&model.initial_state(), &sample64(&[1.0, -1.0], 0.5))
        .expect("independent first expectation");
    let expected_second = model
        .update(&model.initial_state(), &sample64(&[1.0, 1.0], 1.0))
        .expect("independent second expectation");
    assert_eq!(bits64(&first.w), bits64(&expected_first.w));
    assert_eq!(bits64(&second.w), bits64(&expected_second.w));
}
