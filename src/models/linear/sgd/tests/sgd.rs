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
