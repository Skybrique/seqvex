//! Deterministic `SGD-QUALITY800` demonstration for A/B/C.
//!
//! A learns in native `f64`; B is a checked `f32` serving snapshot of A; C is
//! the existing strict-`f32` learner. The example performs actual ordered
//! learning, then freezes the models and scores held-out rows. It is a
//! deterministic in-memory fixture, not a general accuracy or performance claim.

use seqvex::foundation::numerical::{Vector, Vector64};
use seqvex::foundation::observation::Observation;
use seqvex::foundation::state::StateModel;
use seqvex::models::linear::sgd::{Sgd, Sgd64, SgdSample, SgdSample64, SgdState64};

const D: usize = 3;
const TRAIN_ROWS: usize = 400;
const SCORE_ROWS: usize = 400;
const TOTAL_ROWS: usize = TRAIN_ROWS + SCORE_ROWS;
const PASSES: usize = 200;
const ETA: f64 = 0.01;
const ETA_C: f32 = 0.01;
const A_MAE_LIMIT: f64 = 1.0e-5;
const A_R2_LIMIT: f64 = 1.0 - 1.0e-9;
const BC_MAE_LIMIT: f64 = 1.0e-4;
const BC_R2_LIMIT: f64 = 1.0 - 1.0e-6;
/// Accepted design-time conditioning value for the named fixture. This is not
/// recomputed here and does not create a new numerical study.
const ACCEPTED_CONDITIONING_KAPPA: f64 = 1.447_896_65;

struct ProfileMetrics {
    name: &'static str,
    updates: usize,
    conversions: usize,
    scores: usize,
    mae: f64,
    r2: f64,
}

struct QualityReport {
    profiles: [ProfileMetrics; 3],
    conditioning_kappa: f64,
}

fn synthetic64(t: usize) -> ([f64; 3], f64) {
    let x0 = (t as f64 * 0.13).sin();
    let x1 = (t as f64 * 0.07).cos();
    let y = (0.7 * x0) - (1.3 * x1) + 0.5;
    ([x0, x1, 1.0], y)
}

fn synthetic32(t: usize) -> ([f32; 3], f32) {
    let t = t as f32;
    let x0 = (t * 0.13f32).sin();
    let x1 = (t * 0.07f32).cos();
    let y = (0.7f32 * x0) - (1.3f32 * x1) + 0.5f32;
    ([x0, x1, 1.0f32], y)
}

fn narrow_features(features: &[f64; 3]) -> [f32; 3] {
    std::array::from_fn(|index| {
        assert!(
            features[index].is_finite(),
            "fixture feature must be finite"
        );
        let narrowed = features[index] as f32;
        assert!(narrowed.is_finite(), "fixture feature must fit f32");
        narrowed
    })
}

fn metrics(predictions: &[f64], targets: &[f64]) -> (f64, f64) {
    assert_eq!(predictions.len(), targets.len());
    assert!(!predictions.is_empty(), "scoring rows must be nonempty");
    let count = predictions.len() as f64;
    let mean = targets.iter().sum::<f64>() / count;
    let mae = predictions
        .iter()
        .zip(targets)
        .map(|(prediction, target)| (prediction - target).abs())
        .sum::<f64>()
        / count;
    let sse = predictions
        .iter()
        .zip(targets)
        .map(|(prediction, target)| (prediction - target).powi(2))
        .sum::<f64>();
    let sst = targets
        .iter()
        .map(|target| (target - mean).powi(2))
        .sum::<f64>();
    assert!(sst > 0.0, "target variance must be nonzero");
    assert!(mae.is_finite() && sst.is_finite());
    (mae, 1.0 - sse / sst)
}

fn train_a() -> (Sgd64, SgdState64) {
    let model = Sgd64::new(D, ETA).expect("valid A configuration");
    let mut state = model.initial_state();
    for _pass in 0..PASSES {
        for t in 0..TRAIN_ROWS {
            let (features, target) = synthetic64(t);
            state = model
                .update(
                    &state,
                    &Observation::new(SgdSample64 {
                        features: Vector64::from_slice(&features),
                        target,
                    }),
                )
                .expect("ordered A update");
        }
    }
    (model, state)
}

fn train_c() -> (Sgd, seqvex::models::linear::sgd::SgdState) {
    let model = Sgd::new(D, ETA_C).expect("valid C configuration");
    let mut state = model.initial_state();
    for _pass in 0..PASSES {
        for t in 0..TRAIN_ROWS {
            let (features, target) = synthetic32(t);
            state = model
                .update(
                    &state,
                    &Observation::new(SgdSample {
                        features: Vector::from_slice(&features),
                        target,
                    }),
                )
                .expect("ordered C update");
        }
    }
    (model, state)
}

fn run_suite() -> QualityReport {
    let (model_a, state_a) = train_a();
    let snapshot_b = model_a
        .to_serving_f32(&state_a)
        .expect("checked B conversion");
    let (model_c, state_c) = train_c();

    let mut predictions_a = Vec::with_capacity(SCORE_ROWS);
    let mut predictions_b = Vec::with_capacity(SCORE_ROWS);
    let mut predictions_c = Vec::with_capacity(SCORE_ROWS);
    let mut targets_ab = Vec::with_capacity(SCORE_ROWS);
    let mut targets_c = Vec::with_capacity(SCORE_ROWS);

    for t in TRAIN_ROWS..TOTAL_ROWS {
        let (features_64, target_64) = synthetic64(t);
        targets_ab.push(target_64);
        predictions_a.push(
            model_a
                .predict(&state_a, &Vector64::from_slice(&features_64))
                .expect("A prediction"),
        );
        let features_32 = narrow_features(&features_64);
        predictions_b.push(f64::from(
            snapshot_b
                .predict(&Vector::from_slice(&features_32))
                .expect("B prediction"),
        ));

        let (features_c, target_c) = synthetic32(t);
        targets_c.push(f64::from(target_c));
        predictions_c.push(f64::from(
            model_c
                .predict(&state_c, &Vector::from_slice(&features_c))
                .expect("C prediction"),
        ));
    }

    let (mae_a, r2_a) = metrics(&predictions_a, &targets_ab);
    let (mae_b, r2_b) = metrics(&predictions_b, &targets_ab);
    let (mae_c, r2_c) = metrics(&predictions_c, &targets_c);
    let updates = PASSES * TRAIN_ROWS;

    let report = QualityReport {
        profiles: [
            ProfileMetrics {
                name: "A",
                updates,
                conversions: 0,
                scores: SCORE_ROWS,
                mae: mae_a,
                r2: r2_a,
            },
            ProfileMetrics {
                name: "B",
                updates: 0,
                conversions: 1,
                scores: SCORE_ROWS,
                mae: mae_b,
                r2: r2_b,
            },
            ProfileMetrics {
                name: "C",
                updates,
                conversions: 0,
                scores: SCORE_ROWS,
                mae: mae_c,
                r2: r2_c,
            },
        ],
        conditioning_kappa: ACCEPTED_CONDITIONING_KAPPA,
    };
    check_thresholds(&report);
    report
}

fn check_thresholds(report: &QualityReport) {
    let a = &report.profiles[0];
    let b = &report.profiles[1];
    let c = &report.profiles[2];
    assert_eq!((a.name, b.name, c.name), ("A", "B", "C"));
    assert_eq!(a.updates, PASSES * TRAIN_ROWS);
    assert_eq!(a.conversions, 0);
    assert_eq!(a.scores, SCORE_ROWS);
    assert_eq!(b.updates, 0);
    assert_eq!(b.conversions, 1);
    assert_eq!(b.scores, SCORE_ROWS);
    assert_eq!(c.updates, PASSES * TRAIN_ROWS);
    assert_eq!(c.conversions, 0);
    assert_eq!(c.scores, SCORE_ROWS);
    assert!(report.conditioning_kappa <= 10.0);
    assert!(
        a.mae <= A_MAE_LIMIT,
        "A MAE {} exceeds {}",
        a.mae,
        A_MAE_LIMIT
    );
    assert!(a.r2 >= A_R2_LIMIT, "A R2 {} below {}", a.r2, A_R2_LIMIT);
    assert!(
        b.mae <= BC_MAE_LIMIT,
        "B MAE {} exceeds {}",
        b.mae,
        BC_MAE_LIMIT
    );
    assert!(b.r2 >= BC_R2_LIMIT, "B R2 {} below {}", b.r2, BC_R2_LIMIT);
    assert!(
        c.mae <= BC_MAE_LIMIT,
        "C MAE {} exceeds {}",
        c.mae,
        BC_MAE_LIMIT
    );
    assert!(c.r2 >= BC_R2_LIMIT, "C R2 {} below {}", c.r2, BC_R2_LIMIT);
}

fn main() {
    let report = run_suite();
    println!(
        "SGD-QUALITY800 conditioning_kappa={:.8} (accepted design-time; bound 10)",
        report.conditioning_kappa
    );
    for profile in &report.profiles {
        println!(
            "SGD-QUALITY800 profile={} updates={} conversions={} scores={} mae={:.12e} r2={:.15}",
            profile.name,
            profile.updates,
            profile.conversions,
            profile.scores,
            profile.mae,
            profile.r2
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use seqvex::models::linear::sgd::{SgdError, SgdServing32};

    fn bits(vector: &Vector) -> Vec<u32> {
        vector
            .as_slice()
            .iter()
            .map(|value| value.to_bits())
            .collect()
    }

    fn collect_checked(
        snapshot: &SgdServing32,
        rows: &[([f32; 1], f32)],
    ) -> Result<Vec<f32>, SgdError> {
        let mut output = Vec::with_capacity(rows.len());
        for (features, _target) in rows {
            output.push(snapshot.predict(&Vector::from_slice(features))?);
        }
        Ok(output)
    }

    fn accept_candidate(
        _active: &SgdServing32,
        candidate: &SgdServing32,
        rows: &[([f32; 1], f32)],
    ) -> bool {
        if rows.is_empty() {
            return false;
        }
        collect_checked(candidate, rows).is_ok()
    }

    #[test]
    fn quality_fixture_meets_frozen_limits() {
        let report = run_suite();
        check_thresholds(&report);
        assert!(report.conditioning_kappa <= 10.0);
    }

    #[test]
    fn strict_c_eta_is_direct_f32() {
        assert_eq!(ETA_C.to_bits(), 0x3c23_d70a);
    }

    #[test]
    fn fixed_inference_collector_is_ordered_and_failure_preserving() {
        let model = Sgd64::new(1, 0.1).expect("valid configuration");
        let active = model
            .to_serving_f32(&SgdState64 {
                w: Vector64::from_slice(&[2.0]),
            })
            .expect("active snapshot");
        let active_bits = bits(active.weights());

        let success_rows = [([1.0f32], 0.0f32), ([2.0], 0.0), ([3.0], 0.0)];
        let ordered = collect_checked(&active, &success_rows).expect("ordered success");
        assert_eq!(
            ordered
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            vec![2.0f32, 4.0, 6.0]
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>()
        );

        assert!(collect_checked(&active, &[]).expect("empty").is_empty());
        assert!(!accept_candidate(&active, &active, &[]));
        assert_eq!(bits(active.weights()), active_bits);

        let candidate = model
            .to_serving_f32(&SgdState64 {
                w: Vector64::from_slice(&[f32::MAX as f64]),
            })
            .expect("failure candidate");
        for rows in [
            [([2.0f32], 0.0), ([1.0], 0.0), ([1.0], 0.0)],
            [([1.0f32], 0.0), ([2.0], 0.0), ([1.0], 0.0)],
            [([1.0f32], 0.0), ([1.0], 0.0), ([2.0], 0.0)],
        ] {
            assert_eq!(
                collect_checked(&candidate, &rows).unwrap_err(),
                SgdError::NonFiniteOutput
            );
            assert!(!accept_candidate(&active, &candidate, &rows));
            assert_eq!(bits(active.weights()), active_bits);
        }
    }

    #[test]
    fn serving_replacement_succeeds_with_explicit_assignment() {
        let model = Sgd64::new(1, 0.1).expect("valid configuration");
        let active = model
            .to_serving_f32(&SgdState64 {
                w: Vector64::from_slice(&[1.0]),
            })
            .expect("active snapshot");
        let candidate = model
            .to_serving_f32(&SgdState64 {
                w: Vector64::from_slice(&[2.0]),
            })
            .expect("candidate snapshot");

        assert!(accept_candidate(&active, &candidate, &[([1.0], 1.0)]));
        let active = candidate;
        assert_eq!(
            active
                .predict(&Vector::from_slice(&[1.0]))
                .expect("new active prediction")
                .to_bits(),
            2.0f32.to_bits()
        );
    }

    #[test]
    fn serving_replacement_is_failure_preserving() {
        let model = Sgd64::new(1, 0.1).expect("valid configuration");
        let active = model
            .to_serving_f32(&SgdState64 {
                w: Vector64::from_slice(&[1.0]),
            })
            .expect("active snapshot");
        let active_bits = bits(active.weights());
        let candidate = model
            .to_serving_f32(&SgdState64 {
                w: Vector64::from_slice(&[f32::MAX as f64]),
            })
            .expect("candidate snapshot");

        assert!(!accept_candidate(&active, &candidate, &[([2.0], 1.0)]));
        assert_eq!(bits(active.weights()), active_bits);
        assert!(!accept_candidate(&active, &candidate, &[]));
        assert_eq!(bits(active.weights()), active_bits);
    }
}
