//! Independent evidence for the SGD-trained linear-regression slice.
//!
//! # Evidence categories
//!
//! * **Check A — recurrence correctness.** Hand-derived single step, an explicit
//!   `f32` grouping/reduction conformance control (bitwise), an exact-rational
//!   trajectory control, and an analytic-gradient verse central-difference check.
//! * **Check B — approximation accuracy.** The frozen A3 fixture, an exact-rational
//!   least-squares oracle built from the supplied `f32` values, and the
//!   absolute-plus-relative objective-gap criterion.
//! * **Contracts.** Validation precedence, overflow, atomicity, isolation,
//!   continuation/reset, order and reproducibility.
//!
//! Fixed-point quantities and `η`-sweeps are deliberately absent: they are
//! optional supplementary evidence, not gates.

use num_rational::BigRational;
use num_traits::{ToPrimitive, Zero};

use seqvex::foundation::numerical::Vector;
use seqvex::foundation::observation::Observation;
use seqvex::foundation::state::StateModel;
use seqvex::models::linear::sgd::{Sgd, SgdError, SgdSample, SgdState};

// --- helpers ---------------------------------------------------------------

fn sample(features: &[f32], target: f32) -> Observation<SgdSample> {
    Observation::new(SgdSample {
        features: Vector::from_slice(features),
        target,
    })
}

fn bits(vector: &Vector) -> Vec<u32> {
    vector
        .as_slice()
        .iter()
        .map(|value| value.to_bits())
        .collect()
}

fn rat(value: f32) -> BigRational {
    BigRational::from_float(f64::from(value)).expect("finite fixture")
}

fn as_f64(value: &BigRational) -> f64 {
    value.to_f64().expect("representable rational")
}

// --- Check A ---------------------------------------------------------------

#[test]
fn hand_derived_single_update() {
    // D=1, x=2, y=4, w0=0, eta=0.1: e=4, w1 = 0 + 0.1*4*2 = 0.8.
    let model = Sgd::new(1, 0.1).expect("valid configuration");
    let state = model.initial_state();
    let next = model
        .update(&state, &sample(&[2.0], 4.0))
        .expect("single update");
    assert_eq!(next.w.as_slice()[0].to_bits(), 0.8f32.to_bits());
}

/// Hand-written explicit-`f32` reference for one update. Every operation is an
/// `f32` operation in the declared order: dot products in feature-index order,
/// then `e = y - ŷ`, `s = η·e`, `Δ_j = s·x_j`, `w'_j = w_j + Δ_j`.
fn reference_update(eta: f32, w: &[f32], x: &[f32], y: f32) -> Vec<f32> {
    let mut dot = 0.0f32;
    for j in 0..x.len() {
        dot += w[j] * x[j];
    }
    let error = y - dot;
    let scale = eta * error;
    let mut out = vec![0.0f32; x.len()];
    for j in 0..x.len() {
        out[j] = w[j] + x[j] * scale;
    }
    out
}

#[test]
fn explicit_f32_grouping_and_reduction_control_is_bitwise() {
    let eta = 0.0625f32;
    let model = Sgd::new(4, eta).expect("valid configuration");
    let mut state = model
        .update(
            &model.initial_state(),
            &sample(&[0.5, -0.25, 0.125, 1.0], 0.75),
        )
        .expect("warm-up update");

    let x = [-0.375f32, 0.625, 0.25, 1.0];
    let y = -0.5f32;
    let expected = reference_update(eta, state.w.as_slice(), &x, y);
    state = model.update(&state, &sample(&x, y)).expect("second update");

    for (j, expected) in expected.iter().enumerate() {
        assert_eq!(
            state.w.as_slice()[j].to_bits(),
            expected.to_bits(),
            "component {j}: exact f32 grouping/reduction must match bitwise"
        );
    }
}

#[test]
fn reduction_order_is_feature_index_order() {
    // Illustrative design fixture: sequential f32 accumulation yields 1, while
    // reverse accumulation yields 0. The predictor must use index-order
    // reduction, and the fixture must actually distinguish the two.
    let weights = [1.0e8f32, -1.0e8, 1.0];
    let features = [1.0f32, 1.0, 1.0];
    let model = Sgd::new(3, 0.1).expect("valid configuration");
    let state = SgdState {
        w: Vector::from_slice(&weights),
    };

    let mut sequential = 0.0f32;
    for j in 0..features.len() {
        sequential += weights[j] * features[j];
    }
    let mut reverse = 0.0f32;
    for j in (0..features.len()).rev() {
        reverse += weights[j] * features[j];
    }
    assert_ne!(
        sequential.to_bits(),
        reverse.to_bits(),
        "fixture must distinguish reduction order"
    );

    let prediction = model
        .predict(&state, &Vector::from_slice(&features))
        .expect("finite prediction");
    assert_eq!(prediction.to_bits(), sequential.to_bits());
    assert_ne!(prediction.to_bits(), reverse.to_bits());
}

#[test]
fn multiplication_grouping_matches_approved_order() {
    // Illustrative design fixture: zero weight, eta = 0.1, target = 0.3,
    // feature = 0.7. The approved grouping is s = fl(eta*e), Δ = fl(s*x); the
    // alternative fl(eta*fl(e*x)) differs by one ULP, so the fixture
    // distinguishes them and the update must match the approved grouping.
    let eta = 0.1f32;
    let target = 0.3f32;
    let feature = 0.7f32;
    let model = Sgd::new(1, eta).expect("valid configuration");
    let state = SgdState {
        w: Vector::from_slice(&[0.0]),
    };

    let error = target - 0.0f32;
    let approved = (eta * error) * feature;
    let alternative = eta * (error * feature);
    assert_ne!(
        approved.to_bits(),
        alternative.to_bits(),
        "fixture must distinguish multiplication grouping"
    );

    let next = model
        .update(&state, &sample(&[feature], target))
        .expect("update");
    assert_eq!(next.w.as_slice()[0].to_bits(), approved.to_bits());
    assert_ne!(next.w.as_slice()[0].to_bits(), alternative.to_bits());
}

/// The frozen 16-step rational-trajectory fixture: a 4-row pattern repeated 4x.
const TRAJECTORY_PATTERN: [([f32; 3], f32); 4] = [
    ([0.5, 0.5, 1.0], 0.25),
    ([-0.5, 0.5, 1.0], 0.0),
    ([-0.5, -0.5, 1.0], -0.25),
    ([0.5, -0.5, 1.0], 0.0),
];

#[test]
#[allow(clippy::needless_range_loop)]
fn rational_trajectory_matches_within_declared_tolerance() {
    // eta = 2^-3, w0 = 0, 16 ordered steps. The exact-rational oracle is
    // independent of the implementation; the f32 comparison is tolerance-based.
    let eta = 0.125f32;
    let model = Sgd::new(3, eta).expect("valid configuration");
    let mut state = model.initial_state();
    let mut exact = [
        BigRational::zero(),
        BigRational::zero(),
        BigRational::zero(),
    ];
    let eta_rat = rat(eta);

    for step in 0..16 {
        let (x, y) = TRAJECTORY_PATTERN[step % TRAJECTORY_PATTERN.len()];

        // Exact step: dot, residual, scale, then per-component addition.
        let mut dot = BigRational::zero();
        for j in 0..3 {
            dot = &dot + &(&exact[j] * &rat(x[j]));
        }
        let error = &rat(y) - &dot;
        let scale = &eta_rat * &error;
        for j in 0..3 {
            exact[j] = &exact[j] + &(&scale * &rat(x[j]));
        }

        // Design envelope: exact-rational states satisfy ||w_t||_2 < 0.15.
        let norm_sq: f64 = exact.iter().map(|value| as_f64(value).powi(2)).sum();
        assert!(
            norm_sq.sqrt() < 0.15,
            "step {step}: exact state norm {} exceeds the 0.15 envelope",
            norm_sq.sqrt()
        );

        state = model.update(&state, &sample(&x, y)).expect("update");
    }

    for j in 0..3 {
        let actual = f64::from(state.w.as_slice()[j]);
        let expected = as_f64(&exact[j]);
        let tolerance = 1e-6 * expected.abs().max(1.0);
        assert!(
            (actual - expected).abs() <= tolerance,
            "component {j}: f32 {actual} vs exact {expected} exceeds {tolerance}"
        );
    }
}

/// The bounded gradient fixture: D=3, N=8, dyadic values.
const GRADIENT_SAMPLES: [([f32; 3], f32); 8] = [
    ([0.5, 0.25, 1.0], 0.5),
    ([-0.25, 0.5, 1.0], 0.0),
    ([0.25, -0.5, 1.0], -0.25),
    ([-0.5, -0.25, 1.0], 0.25),
    ([0.5, -0.5, 1.0], 0.0),
    ([-0.5, 0.5, 1.0], 0.5),
    ([0.25, 0.25, 1.0], -0.5),
    ([-0.25, -0.25, 1.0], 0.0),
];
const GRADIENT_WEIGHTS: [f64; 3] = [0.3, -0.7, 0.2];

#[allow(clippy::needless_range_loop)]
fn objective(samples: &[([f32; 3], f32)], w: &[f64; 3]) -> f64 {
    let n = samples.len() as f64;
    let mut sum = 0.0;
    for (x, y) in samples {
        let mut prediction = 0.0;
        for j in 0..3 {
            prediction += w[j] * f64::from(x[j]);
        }
        let error = f64::from(*y) - prediction;
        sum += error * error;
    }
    sum / (2.0 * n)
}

#[allow(clippy::needless_range_loop)]
fn analytic_gradient(samples: &[([f32; 3], f32)], w: &[f64; 3]) -> [f64; 3] {
    let n = samples.len() as f64;
    let mut gradient = [0.0f64; 3];
    for (x, y) in samples {
        let mut prediction = 0.0;
        for j in 0..3 {
            prediction += w[j] * f64::from(x[j]);
        }
        let error = f64::from(*y) - prediction;
        for j in 0..3 {
            gradient[j] -= error * f64::from(x[j]) / n;
        }
    }
    gradient
}

#[test]
fn analytic_gradient_matches_central_difference() {
    // Central differences of this quadratic have zero truncation error in exact
    // arithmetic; the residual is floating-point only (evaluation plus
    // cancellation). h = 1e-3 and 1e-9 are predeclared.
    let h = 1e-3;
    let analytic = analytic_gradient(&GRADIENT_SAMPLES, &GRADIENT_WEIGHTS);
    for j in 0..3 {
        let mut plus = GRADIENT_WEIGHTS;
        let mut minus = GRADIENT_WEIGHTS;
        plus[j] += h;
        minus[j] -= h;
        let finite_difference = (objective(&GRADIENT_SAMPLES, &plus)
            - objective(&GRADIENT_SAMPLES, &minus))
            / (2.0 * h);
        let tolerance = 1e-9 * analytic[j].abs().max(1.0);
        assert!(
            (finite_difference - analytic[j]).abs() <= tolerance,
            "component {j}: fd {finite_difference} vs analytic {} exceeds {tolerance}",
            analytic[j]
        );
    }
}

// --- Contracts -------------------------------------------------------------

#[test]
fn construction_rejects_zero_dimension_and_invalid_rates() {
    assert_eq!(Sgd::new(0, 0.1).unwrap_err(), SgdError::ZeroDimension);
    for rate in [0.0f32, -0.1, f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        assert_eq!(
            Sgd::new(3, rate).unwrap_err(),
            SgdError::InvalidLearningRate
        );
    }
    // Every finite positive rate is accepted, including very large and tiny.
    for rate in [f32::MIN_POSITIVE, 1.0e-30, 1.0, 1.0e30] {
        Sgd::new(3, rate).expect("finite positive rate must be accepted");
    }
}

#[test]
fn sgd_error_display_is_readable_and_source_free() {
    fn assert_error_traits<
        T: std::error::Error + Clone + Copy + PartialEq + Eq + std::fmt::Debug,
    >() {
    }
    assert_error_traits::<SgdError>();

    let mismatch = SgdError::DimensionMismatch {
        expected: 17,
        actual: 3,
    };
    assert_eq!(
        mismatch.to_string(),
        "dimension mismatch: expected 17, got 3"
    );

    for (error, designated) in [
        (SgdError::ZeroDimension, "dimension must be nonzero"),
        (
            SgdError::InvalidLearningRate,
            "learning rate must be finite and positive",
        ),
        (
            SgdError::NonFiniteInput,
            "features and target must be finite",
        ),
        (
            SgdError::NonFiniteCandidate,
            "candidate state is not finite",
        ),
        (SgdError::NonFiniteOutput, "non-finite output"),
    ] {
        let message = error.to_string();
        assert!(message.contains(designated), "{message:?}");
        assert!(std::error::Error::source(&error).is_none());
    }
    assert!(std::error::Error::source(&mismatch).is_none());
}

#[test]
fn validation_precedence_is_dimension_then_finiteness() {
    let model = Sgd::new(3, 0.1).expect("valid configuration");
    let state = model.initial_state();

    // Wrong feature dimension wins over a non-finite feature.
    assert_eq!(
        model
            .predict(&state, &Vector::from_slice(&[f32::NAN]))
            .unwrap_err(),
        SgdError::DimensionMismatch {
            expected: 3,
            actual: 1
        }
    );
    // Wrong state dimension wins over a non-finite feature.
    let short = SgdState {
        w: Vector::zeros(2),
    };
    assert_eq!(
        model
            .predict(&short, &Vector::from_slice(&[f32::NAN, 0.0, 0.0]))
            .unwrap_err(),
        SgdError::DimensionMismatch {
            expected: 3,
            actual: 2
        }
    );
    // Only then feature finiteness.
    assert_eq!(
        model
            .predict(&state, &Vector::from_slice(&[0.0, f32::NAN, 0.0]))
            .unwrap_err(),
        SgdError::NonFiniteInput
    );
}

#[test]
fn update_rejects_non_finite_feature_and_target_but_preserves_state() {
    let model = Sgd::new(3, 0.1).expect("valid configuration");
    let state = model.initial_state();
    let before = bits(&state.w);

    assert_eq!(
        model
            .update(&state, &sample(&[0.0, f32::NAN, 0.0], 1.0))
            .unwrap_err(),
        SgdError::NonFiniteInput
    );
    assert_eq!(
        model
            .update(&state, &sample(&[0.0, 0.0, 0.0], f32::INFINITY))
            .unwrap_err(),
        SgdError::NonFiniteInput
    );
    assert_eq!(bits(&state.w), before, "failed updates must not commit");
}

#[test]
fn prediction_overflow_is_non_finite_output() {
    let model = Sgd::new(1, 0.1).expect("valid configuration");
    let state = SgdState {
        w: Vector::from_slice(&[3.0e38]),
    };
    let features = Vector::from_slice(&[3.0e38]);
    assert_eq!(
        model.predict(&state, &features).unwrap_err(),
        SgdError::NonFiniteOutput
    );
    assert_eq!(
        state.w.as_slice()[0].to_bits(),
        3.0e38f32.to_bits(),
        "prediction must not mutate state"
    );
}

#[test]
fn update_overflow_is_non_finite_candidate() {
    let model = Sgd::new(1, 1.0e38).expect("valid configuration");
    let state = model.initial_state();
    let before = bits(&state.w);
    // e = 3e38, eta = 1e38, x = 3e38: eta*e overflows, so the candidate is
    // non-finite. Finite inputs, non-finite intermediate.
    assert_eq!(
        model
            .update(&state, &sample(&[3.0e38], 3.0e38))
            .unwrap_err(),
        SgdError::NonFiniteCandidate
    );
    assert_eq!(bits(&state.w), before);
}

#[test]
fn two_states_over_one_model_are_isolated() {
    let model = Sgd::new(2, 0.1).expect("valid configuration");
    let mut left = model.initial_state();
    let right = model.initial_state();
    let right_before = bits(&right.w);

    left = model
        .update(&left, &sample(&[1.0, 2.0], 1.0))
        .expect("update");
    assert_eq!(bits(&right.w), right_before);
    assert_ne!(bits(&left.w), bits(&right.w));
}

#[test]
fn continuation_and_reset_are_explicit() {
    let model = Sgd::new(2, 0.1).expect("valid configuration");
    let rows = [
        ([1.0f32, -1.0], 0.5f32),
        ([0.5, 2.0], -0.5),
        ([-1.0, 0.25], 0.25),
    ];

    let mut state = model.initial_state();
    for (x, y) in &rows {
        state = model.update(&state, &sample(x, *y)).expect("update");
    }

    // Split the same sequence at a boundary; passing the state on is bitwise
    // equivalent to the uninterrupted fold.
    let mut split = model.initial_state();
    split = model
        .update(&split, &sample(&rows[0].0, rows[0].1))
        .expect("update");
    let boundary = split.clone();
    split = model
        .update(&split, &sample(&rows[1].0, rows[1].1))
        .expect("update");
    split = model
        .update(&split, &sample(&rows[2].0, rows[2].1))
        .expect("update");
    assert_eq!(bits(&state.w), bits(&split.w));
    // Continuing from the boundary differs from continuing the prefix only.
    let continued = model
        .update(&boundary, &sample(&rows[2].0, rows[2].1))
        .expect("update");
    assert_ne!(bits(&continued.w), bits(&state.w));

    // Reset is a fresh zero state.
    assert!(model.initial_state().w.as_slice().iter().all(|v| *v == 0.0));
}

#[test]
fn order_dependence_and_reproducibility() {
    let model = Sgd::new(2, 0.1).expect("valid configuration");
    let a = ([1.0f32, 1.0], 1.0f32);
    let b = ([2.0f32, -1.0], -1.0f32);

    let ab = model
        .update(
            &model
                .update(&model.initial_state(), &sample(&a.0, a.1))
                .expect("update"),
            &sample(&b.0, b.1),
        )
        .expect("update");
    let ba = model
        .update(
            &model
                .update(&model.initial_state(), &sample(&b.0, b.1))
                .expect("update"),
            &sample(&a.0, a.1),
        )
        .expect("update");
    assert_ne!(
        bits(&ab.w),
        bits(&ba.w),
        "ordered updates are order-dependent"
    );

    let repeat = model
        .update(
            &model
                .update(&model.initial_state(), &sample(&a.0, a.1))
                .expect("update"),
            &sample(&b.0, b.1),
        )
        .expect("update");
    assert_eq!(
        bits(&ab.w),
        bits(&repeat.w),
        "same order must be reproducible"
    );
}

// --- Check B ---------------------------------------------------------------

/// Advance-before-draw LCG: advance the state, then read `u = s / 2^32`.
fn unit(state: &mut u32) -> f64 {
    *state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
    f64::from(*state) / 4_294_967_296.0
}

/// The frozen A3 coupled fixture (`D=4`, `N=64`). Values are exactly the
/// supplied `f32` values; nothing is reordered, dropped or imputed.
fn frozen_fixture(noisy: bool) -> (Vec<[f32; 4]>, Vec<f32>) {
    const W_TRUE: [f32; 4] = [1.5, -2.0, 0.5, 3.0];
    let mut feature_state = 20_261_016u32;
    let mut noise_state = 20_261_017u32;
    let mut features = Vec::with_capacity(64);
    let mut targets = Vec::with_capacity(64);
    for _ in 0..64 {
        let mut x = [0.0f32; 4];
        for value in x.iter_mut().take(3) {
            *value = (2.0 * unit(&mut feature_state) - 1.0) as f32;
        }
        x[3] = 1.0;
        let noise = if noisy {
            (0.1 * (2.0 * unit(&mut noise_state) - 1.0)) as f32
        } else {
            0.0
        };
        let mut linear = 0.0f64;
        for j in 0..4 {
            linear += f64::from(W_TRUE[j]) * f64::from(x[j]);
        }
        targets.push((linear + f64::from(noise)) as f32);
        features.push(x);
    }
    (features, targets)
}

#[allow(clippy::needless_range_loop)]
fn exact_solve(mut a: Vec<Vec<BigRational>>, mut b: Vec<BigRational>) -> Option<Vec<BigRational>> {
    let n = a.len();
    for col in 0..n {
        let pivot_row = (col..n).find(|&row| !a[row][col].is_zero())?;
        a.swap(col, pivot_row);
        b.swap(col, pivot_row);
        let pivot = a[col][col].clone();
        for row in (col + 1)..n {
            if !a[row][col].is_zero() {
                let factor = &a[row][col] / &pivot;
                for c in col..n {
                    let sub = &factor * &a[col][c];
                    a[row][c] = &a[row][c] - &sub;
                }
                let sub = &factor * &b[col];
                b[row] = &b[row] - &sub;
            }
        }
    }
    let mut x = vec![BigRational::zero(); n];
    for i in (0..n).rev() {
        let mut acc = b[i].clone();
        for j in (i + 1)..n {
            let sub = &a[i][j] * &x[j];
            acc = &acc - &sub;
        }
        x[i] = &acc / &a[i][i];
    }
    Some(x)
}

/// Exact-rational least-squares optimum over the actual supplied `f32` values.
fn exact_optimum(features: &[[f32; 4]], targets: &[f32]) -> Vec<BigRational> {
    let mut gram = vec![vec![BigRational::zero(); 4]; 4];
    let mut rhs = vec![BigRational::zero(); 4];
    for (x, y) in features.iter().zip(targets) {
        let target = rat(*y);
        for j in 0..4 {
            let xj = rat(x[j]);
            rhs[j] = &rhs[j] + &(&xj * &target);
            for k in 0..4 {
                let xk = rat(x[k]);
                gram[j][k] = &gram[j][k] + &(&xj * &xk);
            }
        }
    }
    exact_solve(gram, rhs).expect("frozen fixture must be full rank")
}

#[allow(clippy::needless_range_loop)]
fn objective4(features: &[[f32; 4]], targets: &[f32], w: &[f64; 4]) -> f64 {
    let n = features.len() as f64;
    let mut sum = 0.0;
    for (x, y) in features.iter().zip(targets) {
        let mut prediction = 0.0;
        for j in 0..4 {
            prediction += w[j] * f64::from(x[j]);
        }
        let error = f64::from(*y) - prediction;
        sum += error * error;
    }
    sum / (2.0 * n)
}

/// Jacobi eigenvalues of a symmetric 4x4 matrix (test-local; no production
/// eigensolver and no new primitive).
#[allow(clippy::needless_range_loop)]
fn jacobi_eigenvalues(mut a: [[f64; 4]; 4]) -> [f64; 4] {
    for _ in 0..100 {
        let mut p = 0usize;
        let mut q = 1usize;
        let mut largest = 0.0f64;
        for i in 0..4 {
            for j in (i + 1)..4 {
                if a[i][j].abs() > largest {
                    largest = a[i][j].abs();
                    p = i;
                    q = j;
                }
            }
        }
        if largest < 1e-15 {
            break;
        }
        let theta = (a[q][q] - a[p][p]) / (2.0 * a[p][q]);
        let t = if theta >= 0.0 {
            1.0 / (theta + (theta * theta + 1.0).sqrt())
        } else {
            -1.0 / (-theta + (theta * theta + 1.0).sqrt())
        };
        let c = 1.0 / (t * t + 1.0).sqrt();
        let s = t * c;
        let app = a[p][p];
        let aqq = a[q][q];
        let apq = a[p][q];
        for k in 0..4 {
            if k != p && k != q {
                let akp = a[k][p];
                let akq = a[k][q];
                a[k][p] = c * akp - s * akq;
                a[p][k] = a[k][p];
                a[k][q] = s * akp + c * akq;
                a[q][k] = a[k][q];
            }
        }
        a[p][p] = app - t * apq;
        a[q][q] = aqq + t * apq;
        a[p][q] = 0.0;
        a[q][p] = 0.0;
    }
    [a[0][0], a[1][1], a[2][2], a[3][3]]
}

#[allow(clippy::needless_range_loop)]
fn condition_number(features: &[[f32; 4]]) -> f64 {
    let mut gram = [[0.0f64; 4]; 4];
    for x in features {
        for j in 0..4 {
            for k in 0..4 {
                gram[j][k] += f64::from(x[j]) * f64::from(x[k]);
            }
        }
    }
    let eigenvalues = jacobi_eigenvalues(gram);
    let max = eigenvalues.iter().copied().fold(f64::MIN, f64::max);
    let min = eigenvalues.iter().copied().fold(f64::MAX, f64::min);
    (max / min).sqrt()
}

#[allow(clippy::needless_range_loop)]
fn train(features: &[[f32; 4]], targets: &[f32], eta: f32, passes: usize) -> SgdState {
    let model = Sgd::new(4, eta).expect("valid configuration");
    let mut state = model.initial_state();
    for _pass in 0..passes {
        for (x, y) in features.iter().zip(targets) {
            state = model
                .update(&state, &sample(x, *y))
                .expect("ordered update");
        }
    }
    state
}

fn check_coupled_fixture(noisy: bool) {
    let (features, targets) = frozen_fixture(noisy);

    // Conditioning is a design-time check on the frozen fixture. A failure must
    // be reported, never repaired by regenerating the fixture.
    let kappa = condition_number(&features);
    assert!(
        kappa <= 10.0,
        "frozen fixture conditioning kappa={kappa} exceeds 10; report for review"
    );

    let optimum = exact_optimum(&features, &targets);
    let optimum: [f64; 4] = std::array::from_fn(|j| as_f64(&optimum[j]));
    let j_star = objective4(&features, &targets, &optimum);
    let j_initial = objective4(&features, &targets, &[0.0; 4]);

    // w_true == w_opt may be asserted only when the normal-equation residual on
    // the supplied f32 values is exactly zero.
    const W_TRUE: [f32; 4] = [1.5, -2.0, 0.5, 3.0];
    let mut normal_residual = [0.0f64; 4];
    for (x, y) in features.iter().zip(&targets) {
        let mut prediction = 0.0;
        for j in 0..4 {
            prediction += f64::from(W_TRUE[j]) * f64::from(x[j]);
        }
        let error = f64::from(*y) - prediction;
        for (j, residual) in normal_residual.iter_mut().enumerate() {
            *residual += error * f64::from(x[j]);
        }
    }
    if normal_residual.iter().all(|value| *value == 0.0) {
        for j in 0..4 {
            assert!(
                (optimum[j] - f64::from(W_TRUE[j])).abs() <= 1e-6,
                "noiseless residual is zero, so w_opt must equal w_true"
            );
        }
    }

    let state = train(&features, &targets, 0.01, 200);
    let learned: [f64; 4] = std::array::from_fn(|j| f64::from(state.w.as_slice()[j]));
    let j_sgd = objective4(&features, &targets, &learned);

    let gap = j_sgd - j_star;
    let allowed = 1e-6 * j_initial + 1e-3 * j_star;
    println!(
        "fixture noisy={noisy}: kappa={kappa:.6} J*={j_star:.17e} \
         J_initial={j_initial:.17e} J_sgd={j_sgd:.17e} gap={gap:.17e} allowed={allowed:.17e}"
    );
    assert!(
        j_star.is_finite() && j_sgd.is_finite(),
        "non-finite objective"
    );
    assert!(
        gap <= allowed,
        "objective gap {gap:.17e} exceeds allowed {allowed:.17e} \
         (J*={j_star:.17e}, J_initial={j_initial:.17e}); report for review, do not retune"
    );
}

#[test]
fn coupled_fixture_noiseless_meets_objective_criterion() {
    check_coupled_fixture(false);
}

#[test]
fn coupled_fixture_noisy_meets_objective_criterion() {
    check_coupled_fixture(true);
}

mod precision_profiles {
    #![allow(clippy::needless_range_loop)]

    use num_rational::BigRational;
    use num_traits::Zero;

    use seqvex::execution::streaming::StreamingExecutor;
    use seqvex::foundation::error::FailureCause;
    use seqvex::foundation::numerical::{Vector, Vector64};
    use seqvex::foundation::observation::Observation;
    use seqvex::foundation::state::{StateModel, process_batch, process_one};
    use seqvex::models::linear::sgd::{
        Sgd, Sgd64, SgdError, SgdError64, SgdSample64, SgdServingConversionError, SgdState,
        SgdState64,
    };

    use super::{TRAJECTORY_PATTERN, as_f64, bits, rat, sample};

    const TRAJECTORY_PATTERN64: [([f64; 3], f64); 4] = [
        ([0.5, 0.5, 1.0], 0.25),
        ([-0.5, 0.5, 1.0], 0.0),
        ([-0.5, -0.5, 1.0], -0.25),
        ([0.5, -0.5, 1.0], 0.0),
    ];

    fn sample64(features: &[f64], target: f64) -> Observation<SgdSample64> {
        Observation::new(SgdSample64 {
            features: Vector64::from_slice(features),
            target,
        })
    }

    fn bits64(vector: &Vector64) -> Vec<u64> {
        vector
            .as_slice()
            .iter()
            .map(|value| value.to_bits())
            .collect()
    }

    fn rat64(value: f64) -> BigRational {
        BigRational::from_float(value).expect("finite fixture")
    }

    fn reference_update64(eta: f64, w: &[f64], x: &[f64], y: f64) -> Vec<f64> {
        let dot: f64 = w.iter().zip(x).map(|(w, x)| w * x).sum();
        let error = y - dot;
        let scale = eta * error;
        x.iter().zip(w).map(|(x, w)| w + x * scale).collect()
    }

    fn reference_update32_sum(eta: f32, w: &[f32], x: &[f32], y: f32) -> Vec<f32> {
        let dot: f32 = w.iter().zip(x).map(|(w, x)| w * x).sum();
        let error = y - dot;
        let scale = eta * error;
        x.iter().zip(w).map(|(x, w)| w + x * scale).collect()
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

    struct OrderTrace64 {
        initial: Vec<u64>,
        steps: Vec<(u64, Vec<u64>)>,
    }

    struct OrderTrace32 {
        initial: Vec<u32>,
        steps: Vec<(u32, Vec<u32>)>,
    }

    fn order16_rows64() -> Vec<([f64; 3], f64)> {
        (0..16)
            .map(|step| TRAJECTORY_PATTERN64[step % TRAJECTORY_PATTERN64.len()])
            .collect()
    }

    fn order16_rows32() -> Vec<([f32; 3], f32)> {
        (0..16)
            .map(|step| TRAJECTORY_PATTERN[step % TRAJECTORY_PATTERN.len()])
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

    fn expected_boundary64(trace: &OrderTrace64, boundary: usize) -> &[u64] {
        if boundary == 0 {
            &trace.initial
        } else {
            &trace.steps[boundary - 1].1
        }
    }

    fn expected_boundary32(trace: &OrderTrace32, boundary: usize) -> &[u32] {
        if boundary == 0 {
            &trace.initial
        } else {
            &trace.steps[boundary - 1].1
        }
    }

    fn assert_error_traits<T>()
    where
        T: std::error::Error + Clone + Copy + PartialEq + Eq + std::fmt::Debug,
    {
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
    fn a_construction_and_error_contract() {
        assert_eq!(Sgd64::new(0, 0.1).unwrap_err(), SgdError64::ZeroDimension);
        for rate in [0.0f64, -0.1, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert_eq!(
                Sgd64::new(3, rate).unwrap_err(),
                SgdError64::InvalidLearningRate
            );
        }
        for rate in [f64::MIN_POSITIVE, 1.0e-300, 1.0, 1.0e300] {
            let model = Sgd64::new(3, rate).expect("finite positive rate");
            assert_eq!(model.dimension(), 3);
            assert_eq!(model.learning_rate().to_bits(), rate.to_bits());
        }

        assert_error_traits::<SgdError64>();
        assert_error_traits::<SgdServingConversionError>();

        let mismatch = SgdError64::DimensionMismatch {
            expected: 4,
            actual: 1,
        };
        assert_eq!(
            mismatch.to_string(),
            "dimension mismatch: expected 4, got 1"
        );
        assert!(std::error::Error::source(&mismatch).is_none());

        let conversion = SgdServingConversionError::Underflow { index: 2 };
        assert!(conversion.to_string().contains("underflows"));
        assert!(std::error::Error::source(&conversion).is_none());

        let state = Sgd64::new(2, 0.1)
            .expect("valid configuration")
            .initial_state();
        assert!(state.w.as_slice().iter().all(|value| *value == 0.0));
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
    fn conversion_exact_bits_ties_and_special_values() {
        let model = Sgd64::new(4, 0.1).expect("valid configuration");
        let state = SgdState64 {
            w: Vector64::from_slice(&[1.0, -2.0, 1.0 + 2f64.powi(-24), 1.0 + 3.0 * 2f64.powi(-24)]),
        };
        let snapshot = model.to_serving_f32(&state).expect("serving snapshot");
        let expected = [1.0f32, -2.0, 1.0, 1.0 + 2f32.powi(-22)];
        for (actual, expected) in snapshot.weights().as_slice().iter().zip(expected) {
            assert_eq!(actual.to_bits(), expected.to_bits());
        }

        let model = Sgd64::new(4, 0.1).expect("valid configuration");
        let state = SgdState64 {
            w: Vector64::from_slice(&[f32::MAX as f64, f32::from_bits(1) as f64, -0.0, 0.0]),
        };
        let snapshot = model.to_serving_f32(&state).expect("special values");
        assert_eq!(
            snapshot.weights().as_slice()[0].to_bits(),
            f32::MAX.to_bits()
        );
        assert_eq!(snapshot.weights().as_slice()[1].to_bits(), 1);
        assert_eq!(
            snapshot.weights().as_slice()[2].to_bits(),
            (-0.0f32).to_bits()
        );
        assert_eq!(snapshot.weights().as_slice()[3].to_bits(), 0.0f32.to_bits());

        let model = Sgd64::new(1, 0.1).expect("valid configuration");
        let underflow = SgdState64 {
            w: Vector64::from_slice(&[2f64.powi(-200)]),
        };
        assert_eq!(
            model.to_serving_f32(&underflow).unwrap_err(),
            SgdServingConversionError::Underflow { index: 0 }
        );
    }

    #[test]
    fn conversion_dimension_overflow_nonfinite_and_preservation() {
        let model = Sgd64::new(2, 0.1).expect("valid configuration");
        let source = SgdState64 {
            w: Vector64::from_slice(&[f64::NAN]),
        };
        let source_bits = bits64(&source.w);
        assert_eq!(
            model.to_serving_f32(&source).unwrap_err(),
            SgdServingConversionError::StateDimensionMismatch {
                expected: 2,
                actual: 1
            }
        );
        assert_eq!(bits64(&source.w), source_bits);

        let model = Sgd64::new(2, 0.1).expect("valid configuration");
        let source = SgdState64 {
            w: Vector64::from_slice(&[f64::NAN, 1.0]),
        };
        assert_eq!(
            model.to_serving_f32(&source).unwrap_err(),
            SgdServingConversionError::SourceNonFinite { index: 0 }
        );
        let source = SgdState64 {
            w: Vector64::from_slice(&[1.0, f64::INFINITY]),
        };
        assert_eq!(
            model.to_serving_f32(&source).unwrap_err(),
            SgdServingConversionError::SourceNonFinite { index: 1 }
        );

        let model = Sgd64::new(2, 0.1).expect("valid configuration");
        let source = SgdState64 {
            w: Vector64::from_slice(&[f32::MAX as f64, f64::MAX]),
        };
        let source_bits = bits64(&source.w);
        assert_eq!(
            model.to_serving_f32(&source).unwrap_err(),
            SgdServingConversionError::Overflow { index: 1 }
        );
        assert_eq!(bits64(&source.w), source_bits);

        let model = Sgd64::new(1, 0.1).expect("valid configuration");
        let active = model
            .to_serving_f32(&SgdState64 {
                w: Vector64::from_slice(&[1.0]),
            })
            .expect("active snapshot");
        let active_bits = bits(active.weights());
        let failed = model.to_serving_f32(&SgdState64 {
            w: Vector64::from_slice(&[f64::MAX]),
        });
        assert_eq!(
            failed.unwrap_err(),
            SgdServingConversionError::Overflow { index: 0 }
        );
        assert_eq!(bits(active.weights()), active_bits);
    }

    #[test]
    fn continued_learning_and_serving_replacement() {
        let model = Sgd64::new(2, 0.1).expect("valid configuration");
        let mut s0 = model
            .update(&model.initial_state(), &sample64(&[1.0, -0.5], 0.5))
            .expect("learned state");
        let b0 = model.to_serving_f32(&s0).expect("snapshot B0");
        let b0_bits = bits(b0.weights());
        let b0_prediction = b0
            .predict(&Vector::from_slice(&[0.5, -0.5]))
            .expect("B0 prediction")
            .to_bits();

        let expected_s1 = reference_update64(0.1, s0.w.as_slice(), &[0.25, 1.0], -0.25);
        s0 = model
            .update(&s0, &sample64(&[0.25, 1.0], -0.25))
            .expect("continued learning");
        assert_eq!(bits(b0.weights()), b0_bits, "B0 must remain immutable");
        let s1 = s0;
        assert_eq!(
            bits64(&s1.w),
            expected_s1.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
        let b1 = model.to_serving_f32(&s1).expect("snapshot B1");
        assert_ne!(bits(b1.weights()), b0_bits);
        assert_eq!(
            b0.predict(&Vector::from_slice(&[0.5, -0.5]))
                .expect("B0 prediction")
                .to_bits(),
            b0_prediction
        );

        let other_model = Sgd64::new(3, 0.1).expect("valid configuration");
        let s1_bits = bits64(&s1.w);
        assert_eq!(
            other_model.to_serving_f32(&s1).unwrap_err(),
            SgdServingConversionError::StateDimensionMismatch {
                expected: 3,
                actual: 2
            }
        );
        assert_eq!(bits64(&s1.w), s1_bits);
        assert_eq!(bits(b0.weights()), b0_bits);

        let candidate_model = Sgd64::new(1, 0.1).expect("valid configuration");
        let candidate = candidate_model
            .to_serving_f32(&SgdState64 {
                w: Vector64::from_slice(&[f32::MAX as f64]),
            })
            .expect("candidate snapshot");
        assert_eq!(
            candidate.predict(&Vector::from_slice(&[2.0])).unwrap_err(),
            SgdError::NonFiniteOutput
        );
        assert_eq!(
            bits(b0.weights()),
            b0_bits,
            "failed replacement preserves active"
        );
    }

    #[test]
    fn learned_state_failures_preserve_and_continue() {
        let model = Sgd64::new(1, 1.0e308).expect("valid configuration");
        let learned = model
            .update(&model.initial_state(), &sample64(&[1.0], 1.0))
            .expect("learned finite state");
        let learned_bits = bits64(&learned.w);
        assert_eq!(
            model
                .update(&learned, &sample64(&[1.0e308], 1.0e308))
                .unwrap_err(),
            SgdError64::NonFiniteCandidate
        );
        assert_eq!(bits64(&learned.w), learned_bits);
        let next = model
            .update(&learned, &sample64(&[0.0], 1.0))
            .expect("valid continuation");
        let expected = reference_update64(1.0e308, learned.w.as_slice(), &[0.0], 1.0);
        assert_eq!(
            bits64(&next.w),
            expected.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );

        let model = Sgd::new(1, 1.0e38).expect("valid configuration");
        let learned = model
            .update(&model.initial_state(), &sample(&[1.0], 1.0))
            .expect("learned finite state");
        let learned_bits = bits(&learned.w);
        assert_eq!(
            model
                .update(&learned, &sample(&[3.0e38], 3.0e38))
                .unwrap_err(),
            SgdError::NonFiniteCandidate
        );
        assert_eq!(bits(&learned.w), learned_bits);
        let next = model
            .update(&learned, &sample(&[0.0], 1.0))
            .expect("valid continuation");
        let expected = reference_update32_sum(1.0e38, learned.w.as_slice(), &[0.0], 1.0);
        assert_eq!(
            bits(&next.w),
            expected.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );

        let model64 = Sgd64::new(2, 0.1).expect("valid configuration");
        let learned = model64
            .update(&model64.initial_state(), &sample64(&[1.0, -1.0], 0.5))
            .expect("learned state");
        let learned_bits = bits64(&learned.w);
        assert_eq!(
            model64
                .predict(&learned, &Vector64::from_slice(&[1.0]))
                .unwrap_err(),
            SgdError64::DimensionMismatch {
                expected: 2,
                actual: 1
            }
        );
        assert_eq!(
            model64
                .predict(&learned, &Vector64::from_slice(&[f64::NAN, 0.0]))
                .unwrap_err(),
            SgdError64::NonFiniteInput
        );
        let short = SgdState64 {
            w: Vector64::zeros(1),
        };
        assert_eq!(
            model64
                .update(&short, &sample64(&[1.0, 0.0], 0.0))
                .unwrap_err(),
            SgdError64::DimensionMismatch {
                expected: 2,
                actual: 1
            }
        );
        assert_eq!(
            model64
                .update(&learned, &sample64(&[1.0, 0.0], f64::INFINITY))
                .unwrap_err(),
            SgdError64::NonFiniteInput
        );
        assert_eq!(
            model64
                .update(&learned, &sample64(&[f64::NAN, 0.0], 0.5))
                .unwrap_err(),
            SgdError64::NonFiniteInput
        );
        assert_eq!(bits64(&learned.w), learned_bits);
        let continued = model64
            .update(&learned, &sample64(&[0.5, 1.0], 0.0))
            .expect("valid continuation after failures");
        let expected = reference_update64(0.1, learned.w.as_slice(), &[0.5, 1.0], 0.0);
        assert_eq!(
            bits64(&continued.w),
            expected.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );

        let model32 = Sgd::new(2, 0.1).expect("valid configuration");
        let learned32 = model32
            .update(&model32.initial_state(), &sample(&[1.0, -1.0], 0.5))
            .expect("learned state");
        let learned32_bits = bits(&learned32.w);
        assert_eq!(
            model32
                .predict(&learned32, &Vector::from_slice(&[1.0]))
                .unwrap_err(),
            SgdError::DimensionMismatch {
                expected: 2,
                actual: 1
            }
        );
        assert_eq!(
            model32
                .update(&learned32, &sample(&[1.0, 0.0], f32::INFINITY))
                .unwrap_err(),
            SgdError::NonFiniteInput
        );
        assert_eq!(
            model32
                .update(&learned32, &sample(&[f32::NAN, 0.0], 0.5))
                .unwrap_err(),
            SgdError::NonFiniteInput
        );
        // Deliberately constructed wrong-dimension state; the learned state above
        // is checked separately for feature/target failures and continuation.
        let short32 = SgdState {
            w: Vector::zeros(1),
        };
        assert_eq!(
            model32
                .update(&short32, &sample(&[1.0, 0.0], 0.0))
                .unwrap_err(),
            SgdError::DimensionMismatch {
                expected: 2,
                actual: 1
            }
        );
        assert_eq!(bits(&learned32.w), learned32_bits);
        let continued32 = model32
            .update(&learned32, &sample(&[0.5, 1.0], 0.0))
            .expect("valid continuation after failures");
        let expected32 = reference_update32_sum(0.1, learned32.w.as_slice(), &[0.5, 1.0], 0.0);
        assert_eq!(
            bits(&continued32.w),
            expected32.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
    }

    #[test]
    fn a_output_and_precedence_failures_preserve_state() {
        let model = Sgd64::new(1, 0.1).expect("valid configuration");
        // Deliberately constructed extreme state for the output-overflow boundary;
        // it is not a learned state.
        let extreme = SgdState64 {
            w: Vector64::from_slice(&[f64::MAX]),
        };
        let extreme_bits = bits64(&extreme.w);
        assert_eq!(
            model
                .predict(&extreme, &Vector64::from_slice(&[2.0]))
                .unwrap_err(),
            SgdError64::NonFiniteOutput
        );
        assert_eq!(bits64(&extreme.w), extreme_bits);

        let model = Sgd64::new(2, 0.1).expect("valid configuration");
        let state = model.initial_state();
        let state_bits = bits64(&state.w);
        assert_eq!(
            model
                .predict(&state, &Vector64::from_slice(&[f64::NAN]))
                .unwrap_err(),
            SgdError64::DimensionMismatch {
                expected: 2,
                actual: 1
            }
        );
        assert_eq!(bits64(&state.w), state_bits);

        let snapshot = model.to_serving_f32(&state).expect("snapshot");
        let snapshot_bits = bits(snapshot.weights());
        assert_eq!(
            snapshot
                .predict(&Vector::from_slice(&[f32::NAN]))
                .unwrap_err(),
            SgdError::DimensionMismatch {
                expected: 2,
                actual: 1
            }
        );
        assert_eq!(bits(snapshot.weights()), snapshot_bits);
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

    #[test]
    fn b_prediction_validation_and_overflow() {
        let model = Sgd64::new(2, 0.1).expect("valid configuration");
        let snapshot = model
            .to_serving_f32(&SgdState64 {
                w: Vector64::from_slice(&[1.0, -2.0]),
            })
            .expect("snapshot");
        assert_eq!(
            snapshot.predict(&Vector::from_slice(&[1.0])).unwrap_err(),
            SgdError::DimensionMismatch {
                expected: 2,
                actual: 1
            }
        );
        assert_eq!(
            snapshot
                .predict(&Vector::from_slice(&[f32::NAN, 0.0]))
                .unwrap_err(),
            SgdError::NonFiniteInput
        );
        let bits_before = bits(snapshot.weights());
        let _ = snapshot.predict(&Vector::from_slice(&[0.5, -0.5]));
        assert_eq!(bits(snapshot.weights()), bits_before);

        let model = Sgd64::new(1, 0.1).expect("valid configuration");
        let overflow = model
            .to_serving_f32(&SgdState64 {
                w: Vector64::from_slice(&[f32::MAX as f64]),
            })
            .expect("overflow-capable snapshot");
        assert_eq!(
            overflow.predict(&Vector::from_slice(&[2.0])).unwrap_err(),
            SgdError::NonFiniteOutput
        );
    }
}

mod public_batch_api {
    use std::error::Error;

    use seqvex::foundation::numerical::{Vector, Vector64};
    use seqvex::foundation::observation::Observation;
    use seqvex::foundation::state::StateModel;
    use seqvex::models::linear::sgd::{
        Sgd, Sgd64, SgdError, SgdError64, SgdPredictionError, SgdPredictionError64, SgdSample,
        SgdSample64, SgdServing32, SgdState, SgdState64, SgdUpdateBatchError,
        SgdUpdateBatchError64,
    };

    fn sample(features: &[f32], target: f32) -> Observation<SgdSample> {
        Observation::new(SgdSample {
            features: Vector::from_slice(features),
            target,
        })
    }

    fn sample64(features: &[f64], target: f64) -> Observation<SgdSample64> {
        Observation::new(SgdSample64 {
            features: Vector64::from_slice(features),
            target,
        })
    }

    fn features(values: &[f32]) -> Observation<Vector> {
        Observation::new(Vector::from_slice(values))
    }

    fn features64(values: &[f64]) -> Observation<Vector64> {
        Observation::new(Vector64::from_slice(values))
    }

    fn state32(values: &[f32]) -> SgdState {
        SgdState {
            w: Vector::from_slice(values),
        }
    }

    fn state64(values: &[f64]) -> SgdState64 {
        SgdState64 {
            w: Vector64::from_slice(values),
        }
    }

    fn bits(vector: &Vector) -> Vec<u32> {
        vector
            .as_slice()
            .iter()
            .map(|value| value.to_bits())
            .collect()
    }

    fn bits64(vector: &Vector64) -> Vec<u64> {
        vector
            .as_slice()
            .iter()
            .map(|value| value.to_bits())
            .collect()
    }

    fn bits32_vec(values: &[f32]) -> Vec<u32> {
        values.iter().map(|value| value.to_bits()).collect()
    }

    fn bits64_vec(values: &[f64]) -> Vec<u64> {
        values.iter().map(|value| value.to_bits()).collect()
    }

    fn assert_learning_transition_at(
        model: &Sgd,
        initial: &SgdState,
        rows: &[Observation<SgdSample>],
        expected_index: usize,
        expected: SgdError,
    ) {
        match model.update_batch(initial.clone(), rows, rows.len()) {
            Err(SgdUpdateBatchError::Transition {
                failed_index,
                committed,
                error,
                ..
            }) => {
                assert_eq!(failed_index, expected_index);
                assert_eq!(committed, expected_index);
                assert_eq!(error, expected);
            }
            other => panic!("expected learning Transition, got {other:?}"),
        }
    }

    fn assert_learning_transition_at64(
        model: &Sgd64,
        initial: &SgdState64,
        rows: &[Observation<SgdSample64>],
        expected_index: usize,
        expected: SgdError64,
    ) {
        match model.update_batch(initial.clone(), rows, rows.len()) {
            Err(SgdUpdateBatchError64::Transition {
                failed_index,
                committed,
                error,
                ..
            }) => {
                assert_eq!(failed_index, expected_index);
                assert_eq!(committed, expected_index);
                assert_eq!(error, expected);
            }
            other => panic!("expected f64 learning Transition, got {other:?}"),
        }
    }

    fn assert_prediction_at(
        model: &Sgd,
        state: &SgdState,
        rows: &[Observation<Vector>],
        expected_index: usize,
        expected: SgdError,
    ) {
        match model.predict_batch(state, rows, rows.len()) {
            Err(SgdPredictionError::At { index, error }) => {
                assert_eq!(index, expected_index);
                assert_eq!(error, expected);
            }
            other => panic!("expected prediction At, got {other:?}"),
        }
    }

    fn assert_prediction_at64(
        model: &Sgd64,
        state: &SgdState64,
        rows: &[Observation<Vector64>],
        expected_index: usize,
        expected: SgdError64,
    ) {
        match model.predict_batch(state, rows, rows.len()) {
            Err(SgdPredictionError64::At { index, error }) => {
                assert_eq!(index, expected_index);
                assert_eq!(error, expected);
            }
            other => panic!("expected f64 prediction At, got {other:?}"),
        }
    }

    #[test]
    fn bound_first_rejects_before_any_row_validation() {
        let model = Sgd::new(2, 0.1).expect("valid configuration");
        let initial = model.initial_state();
        let rows = [sample(&[0.5, 0.5], 1.0), sample(&[f32::NAN, 0.0], 1.0)];
        match model.update_batch(initial.clone(), &rows, 1).unwrap_err() {
            SgdUpdateBatchError::TooLarge { max, actual, state } => {
                assert_eq!((max, actual), (1, 2));
                assert_eq!(bits(&state.w), bits(&initial.w));
            }
            other => panic!("expected TooLarge, got {other:?}"),
        }
        assert!(matches!(
            model.update_batch(initial.clone(), &rows, 0),
            Err(SgdUpdateBatchError::TooLarge {
                max: 0,
                actual: 2,
                ..
            })
        ));

        let prediction_rows = [features(&[0.5, 0.5]), features(&[f32::NAN, 0.0])];
        assert!(matches!(
            model.predict_batch(&initial, &prediction_rows, 1),
            Err(SgdPredictionError::TooLarge { max: 1, actual: 2 })
        ));

        let model64 = Sgd64::new(2, 0.1).expect("valid configuration");
        let initial64 = model64.initial_state();
        let rows64 = [sample64(&[0.5, 0.5], 1.0), sample64(&[f64::NAN, 0.0], 1.0)];
        assert!(matches!(
            model64.update_batch(initial64.clone(), &rows64, 1),
            Err(SgdUpdateBatchError64::TooLarge {
                max: 1,
                actual: 2,
                ..
            })
        ));
        let prediction_rows64 = [features64(&[0.5, 0.5]), features64(&[f64::NAN, 0.0])];
        assert!(matches!(
            model64.predict_batch(&initial64, &prediction_rows64, 1),
            Err(SgdPredictionError64::TooLarge { max: 1, actual: 2 })
        ));
    }

    #[test]
    fn empty_and_zero_preserve_state_bits() {
        let model = Sgd::new(3, 0.1).expect("valid configuration");
        let state = state32(&[f32::NAN, -0.0, 0.0]);
        let state_bits = bits(&state.w);
        let empty: [Observation<SgdSample>; 0] = [];
        let returned = model
            .update_batch(state.clone(), &empty, 0)
            .expect("empty update");
        assert_eq!(bits(&returned.w), state_bits);
        let returned = model
            .update_batch(state.clone(), &empty, usize::MAX)
            .expect("empty update with max");
        assert_eq!(bits(&returned.w), state_bits);
        assert!(matches!(
            model.update_batch(state.clone(), &[sample(&[0.0, 0.0, 0.0], 0.0)], 0),
            Err(SgdUpdateBatchError::TooLarge {
                max: 0,
                actual: 1,
                ..
            })
        ));

        let empty_predictions: [Observation<Vector>; 0] = [];
        let predicted = model
            .predict_batch(&state, &empty_predictions, 0)
            .expect("empty prediction");
        assert!(predicted.is_empty());
        let predicted = model
            .predict_batch(&state, &empty_predictions, usize::MAX)
            .expect("empty prediction with max");
        assert!(predicted.is_empty());
        assert!(matches!(
            model.predict_batch(&state, &[features(&[0.0, 0.0, 0.0])], 0),
            Err(SgdPredictionError::TooLarge { max: 0, actual: 1 })
        ));

        let model64 = Sgd64::new(3, 0.1).expect("valid configuration");
        let state64 = state64(&[f64::NAN, -0.0, 0.0]);
        let state64_bits = bits64(&state64.w);
        let empty64: [Observation<SgdSample64>; 0] = [];
        let returned = model64
            .update_batch(state64.clone(), &empty64, 0)
            .expect("empty f64 update");
        assert_eq!(bits64(&returned.w), state64_bits);
        assert!(matches!(
            model64.update_batch(state64.clone(), &[sample64(&[0.0, 0.0, 0.0], 0.0)], 0),
            Err(SgdUpdateBatchError64::TooLarge {
                max: 0,
                actual: 1,
                ..
            })
        ));
    }

    #[test]
    fn exact_limit_and_usize_max_accept() {
        let model = Sgd::new(2, 0.1).expect("valid configuration");
        let initial = model.initial_state();
        let rows = [sample(&[0.5, 0.5], 1.0), sample(&[-0.5, 0.5], 0.0)];
        let expected = rows.iter().fold(initial.clone(), |state, row| {
            model.update(&state, row).expect("update")
        });
        let exact = model
            .update_batch(initial.clone(), &rows, rows.len())
            .expect("exact limit");
        assert_eq!(bits(&exact.w), bits(&expected.w));
        let unlimited = model
            .update_batch(initial.clone(), &rows, usize::MAX)
            .expect("usize::MAX");
        assert_eq!(bits(&unlimited.w), bits(&expected.w));

        let invalid = [sample(&[1.0, 1.0], 1.0), sample(&[1.0], 1.0)];
        assert!(matches!(
            model.update_batch(initial.clone(), &invalid, usize::MAX),
            Err(SgdUpdateBatchError::Transition {
                failed_index: 1,
                committed: 1,
                ..
            })
        ));

        let prediction_rows = [features(&[0.5, 0.5]), features(&[-0.5, 0.5])];
        let expected_predictions: Vec<f32> = prediction_rows
            .iter()
            .map(|row| model.predict(&initial, row.value()).expect("prediction"))
            .collect();
        let exact_predictions = model
            .predict_batch(&initial, &prediction_rows, prediction_rows.len())
            .expect("exact prediction limit");
        assert_eq!(
            bits32_vec(&exact_predictions),
            bits32_vec(&expected_predictions)
        );
        let unlimited_predictions = model
            .predict_batch(&initial, &prediction_rows, usize::MAX)
            .expect("usize::MAX prediction");
        assert_eq!(
            bits32_vec(&unlimited_predictions),
            bits32_vec(&expected_predictions)
        );

        let invalid_predictions = [features(&[0.5, 0.5]), features(&[f32::NAN, 0.0])];
        assert!(matches!(
            model.predict_batch(&initial, &invalid_predictions, usize::MAX),
            Err(SgdPredictionError::At { index: 1, .. })
        ));

        let model64 = Sgd64::new(2, 0.1).expect("valid configuration");
        let initial64 = model64.initial_state();
        let rows64 = [sample64(&[0.5, 0.5], 1.0), sample64(&[-0.5, 0.5], 0.0)];
        let expected64 = rows64.iter().fold(initial64.clone(), |state, row| {
            model64.update(&state, row).expect("update")
        });
        let unlimited64 = model64
            .update_batch(initial64.clone(), &rows64, usize::MAX)
            .expect("usize::MAX f64");
        assert_eq!(bits64(&unlimited64.w), bits64(&expected64.w));

        let invalid64 = [sample64(&[0.5, 0.5], 1.0), sample64(&[1.0], 1.0)];
        assert!(matches!(
            model64.update_batch(initial64.clone(), &invalid64, usize::MAX),
            Err(SgdUpdateBatchError64::Transition {
                failed_index: 1,
                committed: 1,
                ..
            })
        ));

        let prediction_rows64 = [features64(&[0.5, 0.5]), features64(&[-0.5, 0.5])];
        let expected_predictions64: Vec<f64> = prediction_rows64
            .iter()
            .map(|row| {
                model64
                    .predict(&initial64, row.value())
                    .expect("prediction")
            })
            .collect();
        let exact64 = model64
            .predict_batch(&initial64, &prediction_rows64, prediction_rows64.len())
            .expect("exact f64 prediction limit");
        assert_eq!(bits64_vec(&exact64), bits64_vec(&expected_predictions64));
        let invalid_predictions64 = [features64(&[0.5, 0.5]), features64(&[f64::NAN, 0.0])];
        assert!(matches!(
            model64.predict_batch(&initial64, &invalid_predictions64, usize::MAX),
            Err(SgdPredictionError64::At { index: 1, .. })
        ));
    }

    #[test]
    fn chunked_and_continuous_folds_match() {
        let model = Sgd::new(2, 0.1).expect("valid configuration");
        let initial = model.initial_state();
        let rows = [
            sample(&[0.5, 0.5], 1.0),
            sample(&[-0.5, 0.5], 0.0),
            sample(&[0.5, -0.5], -1.0),
            sample(&[-0.5, -0.5], 0.0),
            sample(&[0.25, 0.75], 0.5),
            sample(&[-0.25, -0.75], -0.5),
        ];
        let continuous = rows.iter().fold(initial.clone(), |state, row| {
            model.update(&state, row).expect("update")
        });
        let single = model
            .update_batch(initial.clone(), &rows, rows.len())
            .expect("single group");
        assert_eq!(bits(&single.w), bits(&continuous.w));
        let mut chunked = initial.clone();
        for chunk in [&rows[..1], &rows[1..3], &rows[3..]] {
            chunked = model
                .update_batch(chunked, chunk, chunk.len())
                .expect("chunk");
        }
        assert_eq!(bits(&chunked.w), bits(&continuous.w));

        let model64 = Sgd64::new(2, 0.1).expect("valid configuration");
        let initial64 = model64.initial_state();
        let rows64: Vec<Observation<SgdSample64>> = (0..6)
            .map(|index| {
                let value = index as f64 * 0.125;
                sample64(&[value, -value], value * 0.5)
            })
            .collect();
        let continuous64 = rows64.iter().fold(initial64.clone(), |state, row| {
            model64.update(&state, row).expect("update")
        });
        let mut chunked64 = initial64.clone();
        for chunk in [&rows64[..2], &rows64[2..5], &rows64[5..]] {
            chunked64 = model64
                .update_batch(chunked64, chunk, chunk.len())
                .expect("f64 chunk");
        }
        assert_eq!(bits64(&chunked64.w), bits64(&continuous64.w));
    }

    #[test]
    fn failure_positions_prefix_slice_retry_and_correction() {
        let model = Sgd::new(2, 0.1).expect("valid configuration");
        let initial = model.initial_state();
        let valid = [
            sample(&[0.5, 0.5], 1.0),
            sample(&[-0.5, 0.5], 0.0),
            sample(&[0.5, -0.5], -1.0),
            sample(&[-0.5, -0.5], 0.0),
        ];
        for failure_index in [0, 2, 4] {
            let mut rows = valid.to_vec();
            rows.insert(failure_index, sample(&[1.0], 1.0));
            let err = model
                .update_batch(initial.clone(), &rows, rows.len())
                .unwrap_err();
            let (failed_index, committed, prefix) = match &err {
                SgdUpdateBatchError::Transition {
                    failed_index,
                    committed,
                    state,
                    error,
                } => {
                    assert_eq!(*failed_index, failure_index);
                    assert_eq!(*committed, failure_index);
                    assert_eq!(
                        *error,
                        SgdError::DimensionMismatch {
                            expected: 2,
                            actual: 1
                        }
                    );
                    (*failed_index, *committed, state.clone())
                }
                other => panic!("expected Transition, got {other:?}"),
            };
            assert_eq!(failed_index, committed);
            let expected_prefix = rows[..failure_index]
                .iter()
                .fold(initial.clone(), |state, row| {
                    model.update(&state, row).expect("update")
                });
            assert_eq!(bits(&prefix.w), bits(&expected_prefix.w));

            let retry = model
                .update_batch(
                    prefix.clone(),
                    &rows[failure_index..],
                    rows.len() - failure_index,
                )
                .unwrap_err();
            match retry {
                SgdUpdateBatchError::Transition {
                    failed_index: retry_index,
                    committed: retry_committed,
                    state,
                    error,
                } => {
                    assert_eq!(retry_index, 0);
                    assert_eq!(retry_committed, 0);
                    assert_eq!(bits(&state.w), bits(&prefix.w));
                    assert_eq!(
                        error,
                        SgdError::DimensionMismatch {
                            expected: 2,
                            actual: 1
                        }
                    );
                }
                other => panic!("expected unchanged retry Transition, got {other:?}"),
            }

            let mut corrected = rows.clone();
            corrected[failure_index] = sample(&[0.25, -0.25], 0.5);
            let expected = model
                .update_batch(initial.clone(), &corrected, corrected.len())
                .expect("corrected full sequence");
            let retried = model
                .update_batch(
                    prefix,
                    &corrected[failure_index..],
                    corrected.len() - failure_index,
                )
                .expect("corrected retry");
            assert_eq!(bits(&retried.w), bits(&expected.w));
        }
    }

    #[test]
    fn f64_failure_positions_prefix_slice_retry_and_correction() {
        let model = Sgd64::new(2, 0.1).expect("valid configuration");
        let initial = model.initial_state();
        let valid = [
            sample64(&[0.5, 0.5], 1.0),
            sample64(&[-0.5, 0.5], 0.0),
            sample64(&[0.5, -0.5], -1.0),
            sample64(&[-0.5, -0.5], 0.0),
        ];
        for failure_index in [0, 2, 4] {
            let mut rows = valid.to_vec();
            rows.insert(failure_index, sample64(&[1.0], 1.0));
            let err = model
                .update_batch(initial.clone(), &rows, rows.len())
                .unwrap_err();
            let (prefix, failed_index) = match &err {
                SgdUpdateBatchError64::Transition {
                    failed_index,
                    committed,
                    state,
                    error,
                } => {
                    assert_eq!(*failed_index, failure_index);
                    assert_eq!(*committed, failure_index);
                    assert_eq!(
                        *error,
                        SgdError64::DimensionMismatch {
                            expected: 2,
                            actual: 1
                        }
                    );
                    (state.clone(), *failed_index)
                }
                other => panic!("expected f64 Transition, got {other:?}"),
            };
            let expected_prefix = rows[..failed_index]
                .iter()
                .fold(initial.clone(), |state, row| {
                    model.update(&state, row).expect("update")
                });
            assert_eq!(bits64(&prefix.w), bits64(&expected_prefix.w));

            let retry = model
                .update_batch(
                    prefix.clone(),
                    &rows[failed_index..],
                    rows.len() - failed_index,
                )
                .unwrap_err();
            match retry {
                SgdUpdateBatchError64::Transition {
                    failed_index: retry_index,
                    committed: retry_committed,
                    state,
                    error,
                } => {
                    assert_eq!(retry_index, 0);
                    assert_eq!(retry_committed, 0);
                    assert_eq!(bits64(&state.w), bits64(&prefix.w));
                    assert_eq!(
                        error,
                        SgdError64::DimensionMismatch {
                            expected: 2,
                            actual: 1
                        }
                    );
                }
                other => panic!("expected f64 unchanged retry Transition, got {other:?}"),
            }

            let mut corrected = rows.clone();
            corrected[failed_index] = sample64(&[0.25, -0.25], 0.5);
            let expected = model
                .update_batch(initial.clone(), &corrected, corrected.len())
                .expect("corrected f64 full sequence");
            let retried = model
                .update_batch(
                    prefix,
                    &corrected[failed_index..],
                    corrected.len() - failed_index,
                )
                .expect("corrected f64 retry");
            assert_eq!(bits64(&retried.w), bits64(&expected.w));
        }
    }

    #[test]
    fn prediction_failure_positions_and_no_partial_vector() {
        let model = Sgd::new(2, 0.1).expect("valid configuration");
        let state = state32(&[0.5, -0.5]);
        let state_bits = bits(&state.w);
        let valid = [
            features(&[0.5, 0.5]),
            features(&[-0.5, 0.5]),
            features(&[0.5, -0.5]),
        ];
        for failure_index in [0, 1, 2] {
            let mut rows = valid.to_vec();
            rows.insert(failure_index, features(&[f32::NAN, 0.0]));
            match model.predict_batch(&state, &rows, rows.len()) {
                Err(SgdPredictionError::At { index, error }) => {
                    assert_eq!(index, failure_index);
                    assert_eq!(error, SgdError::NonFiniteInput);
                }
                other => panic!("expected At, got {other:?}"),
            }
            assert_eq!(bits(&state.w), state_bits);
        }

        let model64 = Sgd64::new(2, 0.1).expect("valid configuration");
        let state64 = state64(&[0.5, -0.5]);
        let state64_bits = bits64(&state64.w);
        let rows64 = [features64(&[0.5, 0.5]), features64(&[f64::NAN, 0.0])];
        match model64.predict_batch(&state64, &rows64, rows64.len()) {
            Err(SgdPredictionError64::At { index, error }) => {
                assert_eq!(index, 1);
                assert_eq!(error, SgdError64::NonFiniteInput);
            }
            other => panic!("expected f64 At, got {other:?}"),
        }
        assert_eq!(bits64(&state64.w), state64_bits);
    }

    #[test]
    fn b_snapshot_prediction_failure_preserves_snapshot() {
        let model = Sgd64::new(2, 0.1).expect("valid configuration");
        let snapshot: SgdServing32 = model
            .to_serving_f32(&state64(&[1.0, -2.0]))
            .expect("snapshot");
        let snapshot_bits = bits(snapshot.weights());
        let rows = [features(&[0.5, 0.5]), features(&[f32::NAN, 0.0])];
        match snapshot.predict_batch(&rows, rows.len()) {
            Err(SgdPredictionError::At { index, error }) => {
                assert_eq!(index, 1);
                assert_eq!(error, SgdError::NonFiniteInput);
            }
            other => panic!("expected B At, got {other:?}"),
        }
        assert_eq!(bits(snapshot.weights()), snapshot_bits);
    }

    #[test]
    fn prediction_failure_positions_cover_a_b_and_c_at_three_rows() {
        let model_c = Sgd::new(2, 0.1).expect("valid C configuration");
        let state_c = state32(&[0.5, -0.5]);
        let c_bits = bits(&state_c.w);
        let valid_c = [
            features(&[0.5, 0.5]),
            features(&[-0.5, 0.5]),
            features(&[0.25, -0.75]),
        ];
        for failure_index in [0, 1, valid_c.len() - 1] {
            let mut rows = valid_c.to_vec();
            rows[failure_index] = features(&[f32::NAN, 0.0]);
            match model_c.predict_batch(&state_c, &rows, rows.len()) {
                Err(SgdPredictionError::At { index, error }) => {
                    assert_eq!(index, failure_index);
                    assert_eq!(error, SgdError::NonFiniteInput);
                }
                other => panic!("expected C At, got {other:?}"),
            }
            assert_eq!(bits(&state_c.w), c_bits);
        }

        let model_a = Sgd64::new(2, 0.1).expect("valid A configuration");
        let state_a = state64(&[0.5, -0.5]);
        let a_bits = bits64(&state_a.w);
        let valid_a = [
            features64(&[0.5, 0.5]),
            features64(&[-0.5, 0.5]),
            features64(&[0.25, -0.75]),
        ];
        for failure_index in [0, 1, valid_a.len() - 1] {
            let mut rows = valid_a.to_vec();
            rows[failure_index] = features64(&[f64::NAN, 0.0]);
            match model_a.predict_batch(&state_a, &rows, rows.len()) {
                Err(SgdPredictionError64::At { index, error }) => {
                    assert_eq!(index, failure_index);
                    assert_eq!(error, SgdError64::NonFiniteInput);
                }
                other => panic!("expected A At, got {other:?}"),
            }
            assert_eq!(bits64(&state_a.w), a_bits);
        }

        let snapshot = model_a.to_serving_f32(&state_a).expect("B snapshot");
        let b_bits = bits(snapshot.weights());
        let valid_b = [
            features(&[0.5, 0.5]),
            features(&[-0.5, 0.5]),
            features(&[0.25, -0.75]),
        ];
        for failure_index in [0, 1, valid_b.len() - 1] {
            let mut rows = valid_b.to_vec();
            rows[failure_index] = features(&[f32::NAN, 0.0]);
            match snapshot.predict_batch(&rows, rows.len()) {
                Err(SgdPredictionError::At { index, error }) => {
                    assert_eq!(index, failure_index);
                    assert_eq!(error, SgdError::NonFiniteInput);
                }
                other => panic!("expected B At, got {other:?}"),
            }
            assert_eq!(bits(snapshot.weights()), b_bits);
        }
    }

    #[test]
    fn b_zero_limit_and_oversize_reject_before_validation() {
        let model = Sgd64::new(2, 0.1).expect("valid configuration");
        let snapshot = model
            .to_serving_f32(&state64(&[1.0, -2.0]))
            .expect("snapshot");
        let snapshot_bits = bits(snapshot.weights());
        let rows = [
            features(&[f32::NAN, 0.0]),
            features(&[0.5, 0.5]),
            features(&[-0.5, 0.5]),
        ];
        assert!(matches!(
            snapshot.predict_batch(&rows, 0),
            Err(SgdPredictionError::TooLarge { max: 0, actual: 3 })
        ));
        assert!(matches!(
            snapshot.predict_batch(&rows, rows.len() - 1),
            Err(SgdPredictionError::TooLarge { max: 2, actual: 3 })
        ));
        assert!(matches!(
            snapshot.predict_batch(&rows[..1], 0),
            Err(SgdPredictionError::TooLarge { max: 0, actual: 1 })
        ));
        assert_eq!(bits(snapshot.weights()), snapshot_bits);
    }

    #[test]
    fn empty_prediction_with_zero_limit_succeeds_for_a_and_b() {
        let model = Sgd64::new(2, 0.1).expect("valid configuration");
        let state = state64(&[0.5, -0.5]);
        let empty_a: [Observation<Vector64>; 0] = [];
        assert!(
            model
                .predict_batch(&state, &empty_a, 0)
                .expect("empty A prediction")
                .is_empty()
        );

        let snapshot = model.to_serving_f32(&state).expect("snapshot");
        let empty_b: [Observation<Vector>; 0] = [];
        assert!(
            snapshot
                .predict_batch(&empty_b, 0)
                .expect("empty B prediction")
                .is_empty()
        );
    }

    #[test]
    fn learning_validation_precedence_32() {
        let model = Sgd::new(2, 0.1).expect("valid configuration");
        let initial = model.initial_state();
        let valid = sample(&[0.5, 0.5], 1.0);
        assert_learning_transition_at(
            &model,
            &initial,
            &[valid.clone(), sample(&[1.0], 1.0)],
            1,
            SgdError::DimensionMismatch {
                expected: 2,
                actual: 1,
            },
        );
        assert_learning_transition_at(
            &model,
            &state32(&[0.0, 0.0, 0.0]),
            &[valid.clone(), sample(&[1.0, 1.0], 1.0)],
            0,
            SgdError::DimensionMismatch {
                expected: 2,
                actual: 3,
            },
        );
        assert_learning_transition_at(
            &model,
            &initial,
            &[valid.clone(), sample(&[f32::NAN, 1.0], 1.0)],
            1,
            SgdError::NonFiniteInput,
        );
        assert_learning_transition_at(
            &model,
            &initial,
            &[valid, sample(&[1.0, 1.0], f32::NAN)],
            1,
            SgdError::NonFiniteInput,
        );
        let overflow_model = Sgd::new(1, 1.0e38).expect("valid configuration");
        assert_learning_transition_at(
            &overflow_model,
            &state32(&[1.0e38]),
            &[sample(&[0.0], 0.0), sample(&[1.0e38], 0.0)],
            1,
            SgdError::NonFiniteCandidate,
        );
    }

    #[test]
    fn learning_validation_precedence_64() {
        let model = Sgd64::new(2, 0.1).expect("valid configuration");
        let initial = model.initial_state();
        let valid = sample64(&[0.5, 0.5], 1.0);
        assert_learning_transition_at64(
            &model,
            &initial,
            &[valid.clone(), sample64(&[1.0], 1.0)],
            1,
            SgdError64::DimensionMismatch {
                expected: 2,
                actual: 1,
            },
        );
        assert_learning_transition_at64(
            &model,
            &state64(&[0.0, 0.0, 0.0]),
            &[valid.clone(), sample64(&[1.0, 1.0], 1.0)],
            0,
            SgdError64::DimensionMismatch {
                expected: 2,
                actual: 3,
            },
        );
        assert_learning_transition_at64(
            &model,
            &initial,
            &[valid.clone(), sample64(&[f64::NAN, 1.0], 1.0)],
            1,
            SgdError64::NonFiniteInput,
        );
        assert_learning_transition_at64(
            &model,
            &initial,
            &[valid, sample64(&[1.0, 1.0], f64::NAN)],
            1,
            SgdError64::NonFiniteInput,
        );
        let overflow_model = Sgd64::new(1, 1.0e308).expect("valid configuration");
        assert_learning_transition_at64(
            &overflow_model,
            &state64(&[1.0e308]),
            &[sample64(&[0.0], 0.0), sample64(&[1.0e308], 0.0)],
            1,
            SgdError64::NonFiniteCandidate,
        );
    }

    #[test]
    fn prediction_validation_precedence_32() {
        let model = Sgd::new(2, 0.1).expect("valid configuration");
        let initial = model.initial_state();
        let valid = features(&[0.5, 0.5]);
        assert_prediction_at(
            &model,
            &initial,
            &[valid.clone(), features(&[1.0])],
            1,
            SgdError::DimensionMismatch {
                expected: 2,
                actual: 1,
            },
        );
        assert_prediction_at(
            &model,
            &state32(&[0.0, 0.0, 0.0]),
            &[valid.clone(), features(&[1.0, 1.0])],
            0,
            SgdError::DimensionMismatch {
                expected: 2,
                actual: 3,
            },
        );
        assert_prediction_at(
            &model,
            &initial,
            &[valid.clone(), features(&[f32::NAN, 0.0])],
            1,
            SgdError::NonFiniteInput,
        );
        assert_prediction_at(
            &model,
            &state32(&[f32::MAX, 0.0]),
            &[features(&[0.0, 0.0]), features(&[f32::MAX, 0.0])],
            1,
            SgdError::NonFiniteOutput,
        );
    }

    #[test]
    fn prediction_validation_precedence_64() {
        let model = Sgd64::new(2, 0.1).expect("valid configuration");
        let initial = model.initial_state();
        let valid = features64(&[0.5, 0.5]);
        assert_prediction_at64(
            &model,
            &initial,
            &[valid.clone(), features64(&[1.0])],
            1,
            SgdError64::DimensionMismatch {
                expected: 2,
                actual: 1,
            },
        );
        assert_prediction_at64(
            &model,
            &state64(&[0.0, 0.0, 0.0]),
            &[valid.clone(), features64(&[1.0, 1.0])],
            0,
            SgdError64::DimensionMismatch {
                expected: 2,
                actual: 3,
            },
        );
        assert_prediction_at64(
            &model,
            &initial,
            &[valid.clone(), features64(&[f64::NAN, 0.0])],
            1,
            SgdError64::NonFiniteInput,
        );
        assert_prediction_at64(
            &model,
            &state64(&[f64::MAX, 0.0]),
            &[features64(&[0.0, 0.0]), features64(&[f64::MAX, 0.0])],
            1,
            SgdError64::NonFiniteOutput,
        );
    }

    #[test]
    fn prediction_validation_precedence_b() {
        let model = Sgd64::new(2, 0.1).expect("valid configuration");
        let snapshot = model
            .to_serving_f32(&state64(&[1.0, -2.0]))
            .expect("snapshot");
        let valid = features(&[0.5, 0.5]);
        match snapshot.predict_batch(&[valid.clone(), features(&[1.0])], 2) {
            Err(SgdPredictionError::At { index, error }) => {
                assert_eq!(index, 1);
                assert_eq!(
                    error,
                    SgdError::DimensionMismatch {
                        expected: 2,
                        actual: 1
                    }
                );
            }
            other => panic!("expected B dimension At, got {other:?}"),
        }
        match snapshot.predict_batch(&[valid, features(&[f32::NAN, 0.0])], 2) {
            Err(SgdPredictionError::At { index, error }) => {
                assert_eq!(index, 1);
                assert_eq!(error, SgdError::NonFiniteInput);
            }
            other => panic!("expected B input At, got {other:?}"),
        }
        let overflow_snapshot = model
            .to_serving_f32(&state64(&[f32::MAX as f64, 0.0]))
            .expect("overflow-capable snapshot");
        match overflow_snapshot
            .predict_batch(&[features(&[0.0, 0.0]), features(&[f32::MAX, 0.0])], 2)
        {
            Err(SgdPredictionError::At { index, error }) => {
                assert_eq!(index, 1);
                assert_eq!(error, SgdError::NonFiniteOutput);
            }
            other => panic!("expected B output At, got {other:?}"),
        }
    }

    #[test]
    fn error_display_and_source_32() {
        let model = Sgd::new(2, 0.1).expect("valid configuration");
        let initial = model.initial_state();
        let too_large = model
            .update_batch(
                initial.clone(),
                &[sample(&[0.0, 0.0], 0.0), sample(&[0.0, 0.0], 0.0)],
                1,
            )
            .unwrap_err();
        let text = too_large.to_string();
        assert!(text.contains("max 1"));
        assert!(text.contains("got 2"));
        assert!(text.contains("supplied state was returned unchanged"));
        assert!(too_large.source().is_none());

        let transition = model
            .update_batch(
                initial.clone(),
                &[sample(&[0.0, 0.0], 0.0), sample(&[1.0], 1.0)],
                2,
            )
            .unwrap_err();
        let text = transition.to_string();
        let inner = match &transition {
            SgdUpdateBatchError::Transition { error, .. } => error.to_string(),
            other => panic!("expected Transition, got {other:?}"),
        };
        assert!(text.contains("index 1"));
        assert!(text.contains("1 committed rows"));
        assert!(!text.contains(&inner));
        assert_eq!(
            transition.source().expect("inner source").to_string(),
            inner
        );

        let prediction_too_large = model
            .predict_batch(&initial, &[features(&[0.0, 0.0]), features(&[0.0, 0.0])], 1)
            .unwrap_err();
        let text = prediction_too_large.to_string();
        assert!(text.contains("max 1"));
        assert!(text.contains("got 2"));
        assert!(text.contains("no output was returned"));
        assert!(prediction_too_large.source().is_none());

        let prediction_at = model
            .predict_batch(
                &initial,
                &[features(&[0.0, 0.0]), features(&[f32::NAN, 0.0])],
                2,
            )
            .unwrap_err();
        let text = prediction_at.to_string();
        let inner = match &prediction_at {
            SgdPredictionError::At { error, .. } => error.to_string(),
            other => panic!("expected At, got {other:?}"),
        };
        assert!(text.contains("index 1"));
        assert!(text.contains("no partial prediction vector was returned"));
        assert!(!text.contains(&inner));
        assert_eq!(
            prediction_at.source().expect("inner source").to_string(),
            inner
        );
    }

    #[test]
    fn error_display_and_source_64() {
        let model = Sgd64::new(2, 0.1).expect("valid configuration");
        let initial = model.initial_state();
        let too_large = model
            .update_batch(
                initial.clone(),
                &[sample64(&[0.0, 0.0], 0.0), sample64(&[0.0, 0.0], 0.0)],
                1,
            )
            .unwrap_err();
        let text = too_large.to_string();
        assert!(text.contains("max 1"));
        assert!(text.contains("got 2"));
        assert!(too_large.source().is_none());

        let transition = model
            .update_batch(
                initial.clone(),
                &[sample64(&[0.0, 0.0], 0.0), sample64(&[1.0], 1.0)],
                2,
            )
            .unwrap_err();
        let text = transition.to_string();
        let inner = match &transition {
            SgdUpdateBatchError64::Transition { error, .. } => error.to_string(),
            other => panic!("expected f64 Transition, got {other:?}"),
        };
        assert!(text.contains("index 1"));
        assert!(!text.contains(&inner));
        assert_eq!(
            transition.source().expect("inner source").to_string(),
            inner
        );

        let prediction_too_large = model
            .predict_batch(
                &initial,
                &[features64(&[0.0, 0.0]), features64(&[0.0, 0.0])],
                1,
            )
            .unwrap_err();
        assert!(matches!(
            prediction_too_large,
            SgdPredictionError64::TooLarge { max: 1, actual: 2 }
        ));
        assert!(prediction_too_large.source().is_none());

        let prediction_at = model
            .predict_batch(
                &initial,
                &[features64(&[0.0, 0.0]), features64(&[f64::NAN, 0.0])],
                2,
            )
            .unwrap_err();
        let text = prediction_at.to_string();
        let inner = match &prediction_at {
            SgdPredictionError64::At { error, .. } => error.to_string(),
            other => panic!("expected f64 At, got {other:?}"),
        };
        assert!(text.contains("index 1"));
        assert!(!text.contains(&inner));
        assert_eq!(
            prediction_at.source().expect("inner source").to_string(),
            inner
        );
    }

    #[test]
    fn bitwise_retention_with_nan_and_signed_zero() {
        let model = Sgd::new(3, 0.1).expect("valid configuration");
        let state = state32(&[f32::NAN, -0.0, 0.0]);
        let expected = bits(&state.w);
        let empty: [Observation<SgdSample>; 0] = [];
        assert_eq!(
            bits(
                &model
                    .update_batch(state.clone(), &empty, 0)
                    .expect("empty update")
                    .w
            ),
            expected
        );
        assert_eq!(
            bits(
                &model
                    .update_batch(state.clone(), &empty, usize::MAX)
                    .expect("empty update with max")
                    .w
            ),
            expected
        );
        let oversize = [sample(&[0.0, 0.0, 0.0], 0.0), sample(&[0.0, 0.0, 0.0], 0.0)];
        match model.update_batch(state.clone(), &oversize, 1).unwrap_err() {
            SgdUpdateBatchError::TooLarge {
                state: returned, ..
            } => {
                assert_eq!(bits(&returned.w), expected);
            }
            other => panic!("expected TooLarge, got {other:?}"),
        }

        let model64 = Sgd64::new(3, 0.1).expect("valid configuration");
        let state64 = state64(&[f64::NAN, -0.0, 0.0]);
        let expected64 = bits64(&state64.w);
        let empty64: [Observation<SgdSample64>; 0] = [];
        assert_eq!(
            bits64(
                &model64
                    .update_batch(state64.clone(), &empty64, 0)
                    .expect("empty f64 update")
                    .w
            ),
            expected64
        );
        let oversize64 = [
            sample64(&[0.0, 0.0, 0.0], 0.0),
            sample64(&[0.0, 0.0, 0.0], 0.0),
        ];
        match model64
            .update_batch(state64.clone(), &oversize64, 1)
            .unwrap_err()
        {
            SgdUpdateBatchError64::TooLarge {
                state: returned, ..
            } => {
                assert_eq!(bits64(&returned.w), expected64);
            }
            other => panic!("expected f64 TooLarge, got {other:?}"),
        }
    }
}
