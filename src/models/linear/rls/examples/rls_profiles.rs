//! RLS precision-profile quality fixture.
//!
//! This example trains native-`f64` RLS (A), converts its frozen weights into an
//! independently owned `f32` serving snapshot (B), and trains the legacy strict
//! `f32` RLS (C) on the deterministic `RLS-QUALITY800` fixture. It then scores a
//! frozen held-out slice without further learning.

use seqvex::foundation::numerical::{Vector, Vector64};
use seqvex::foundation::observation::Observation;
use seqvex::foundation::state::StateModel;
#[cfg(test)]
use seqvex::models::linear::rls::RlsError;
use seqvex::models::linear::rls::{Rls, Rls64, RlsSample, RlsSample64};

const TRAIN_ROWS: usize = 400;
const SCORE_ROWS: usize = 400;
const TOTAL_ROWS: usize = TRAIN_ROWS + SCORE_ROWS;
const DIMENSION: usize = 2;
const DELTA: f64 = 1000.0;
const LAMBDA: f64 = 1.0;

#[derive(Debug, Clone, Copy)]
struct Metrics {
    mae: f64,
    r2: f64,
}

#[derive(Debug)]
struct QualityRun {
    a: Metrics,
    b: Metrics,
    c: Metrics,
    a_updates: usize,
    b_score_calls: usize,
    c_updates: usize,
    a_score_calls: usize,
    c_score_calls: usize,
}

fn generate_f64() -> (Vec<Vec<f64>>, Vec<f64>) {
    let mut xs = Vec::with_capacity(TOTAL_ROWS);
    let mut ys = Vec::with_capacity(TOTAL_ROWS);
    for t in 0..TOTAL_ROWS {
        let x0 = (t as f64 * 0.13_f64).sin();
        let x1 = (t as f64 * 0.07_f64).cos();
        let y = (0.7_f64 * x0) - (1.3_f64 * x1);
        xs.push(vec![x0, x1]);
        ys.push(y);
    }
    (xs, ys)
}

fn generate_f32() -> (Vec<Vec<f32>>, Vec<f32>) {
    let mut xs = Vec::with_capacity(TOTAL_ROWS);
    let mut ys = Vec::with_capacity(TOTAL_ROWS);
    for t in 0..TOTAL_ROWS {
        let x0 = (t as f32 * 0.13_f32).sin();
        let x1 = (t as f32 * 0.07_f32).cos();
        let y = (0.7_f32 * x0) - (1.3_f32 * x1);
        xs.push(vec![x0, x1]);
        ys.push(y);
    }
    (xs, ys)
}

fn sample64(x: &[f64], y: f64) -> Observation<RlsSample64> {
    Observation::new(RlsSample64 {
        features: Vector64::from_slice(x),
        target: y,
    })
}

fn sample32(x: &[f32], y: f32) -> Observation<RlsSample> {
    Observation::new(RlsSample {
        features: Vector::from_slice(x),
        target: y,
    })
}

fn metrics(predictions: &[f64], targets: &[f64]) -> Result<Metrics, String> {
    if predictions.is_empty() || predictions.len() != targets.len() {
        return Err("metric group must be nonempty with matching lengths".to_string());
    }
    if !predictions.iter().all(|value| value.is_finite())
        || !targets.iter().all(|value| value.is_finite())
    {
        return Err("metric inputs must be finite".to_string());
    }
    let count = predictions.len() as f64;
    let mae = predictions
        .iter()
        .zip(targets)
        .map(|(prediction, target)| (prediction - target).abs())
        .sum::<f64>()
        / count;
    let mean = targets.iter().sum::<f64>() / count;
    let sse = predictions
        .iter()
        .zip(targets)
        .map(|(prediction, target)| (prediction - target).powi(2))
        .sum::<f64>();
    let sst = targets
        .iter()
        .map(|target| (target - mean).powi(2))
        .sum::<f64>();
    if sst <= 0.0 || !sst.is_finite() {
        return Err("target sum of squares must be positive".to_string());
    }
    Ok(Metrics {
        mae,
        r2: 1.0 - sse / sst,
    })
}

fn run_quality() -> Result<QualityRun, String> {
    let (xs64, ys64) = generate_f64();
    let (xs32, ys32) = generate_f32();
    let model_a = Rls64::new(DIMENSION, LAMBDA, DELTA)
        .map_err(|error| format!("A configuration failed: {error}"))?;
    let model_c = Rls::new(DIMENSION, LAMBDA as f32, DELTA as f32)
        .map_err(|error| format!("C configuration failed: {error}"))?;

    let mut state_a = model_a.initial_state();
    for t in 0..TRAIN_ROWS {
        state_a = model_a
            .update(&state_a, &sample64(&xs64[t], ys64[t]))
            .map_err(|error| format!("A training failed at row {t}: {error}"))?;
    }

    let mut state_c = model_c.initial_state();
    for t in 0..TRAIN_ROWS {
        state_c = model_c
            .update(&state_c, &sample32(&xs32[t], ys32[t]))
            .map_err(|error| format!("C training failed at row {t}: {error}"))?;
    }

    let serving = model_a
        .to_serving_f32(&state_a)
        .map_err(|error| format!("B conversion failed after row {TRAIN_ROWS}: {error}"))?;

    let mut predictions_a = Vec::with_capacity(SCORE_ROWS);
    let mut predictions_b = Vec::with_capacity(SCORE_ROWS);
    let mut predictions_c = Vec::with_capacity(SCORE_ROWS);
    for t in TRAIN_ROWS..TOTAL_ROWS {
        let features_a = Vector64::from_slice(&xs64[t]);
        let prediction_a = model_a
            .predict_checked(&state_a, &features_a)
            .map_err(|error| format!("A prediction failed at row {t}: {error}"))?;
        predictions_a.push(prediction_a);

        let narrowed: Vec<f32> = xs64[t]
            .iter()
            .map(|&value| {
                if !value.is_finite() {
                    return Err(format!("B feature source is not finite at row {t}"));
                }
                let narrowed = value as f32;
                if !narrowed.is_finite() {
                    return Err(format!("B feature conversion is not finite at row {t}"));
                }
                Ok(narrowed)
            })
            .collect::<Result<Vec<_>, _>>()?;
        let features_b = Vector::from_slice(&narrowed);
        let prediction_b = serving
            .predict_checked(&features_b)
            .map_err(|error| format!("B prediction failed at row {t}: {error}"))?;
        predictions_b.push(f64::from(prediction_b));

        let features_c = Vector::from_slice(&xs32[t]);
        let prediction_c = model_c
            .predict_checked(&state_c, &features_c)
            .map_err(|error| format!("C prediction failed at row {t}: {error}"))?;
        predictions_c.push(f64::from(prediction_c));
    }

    let targets_c: Vec<f64> = ys32[TRAIN_ROWS..]
        .iter()
        .map(|&value| f64::from(value))
        .collect();
    let targets_a = &ys64[TRAIN_ROWS..];
    let a = metrics(&predictions_a, targets_a)?;
    let b = metrics(&predictions_b, targets_a)?;
    let c = metrics(&predictions_c, &targets_c)?;

    Ok(QualityRun {
        a,
        b,
        c,
        a_updates: TRAIN_ROWS,
        b_score_calls: SCORE_ROWS,
        c_updates: TRAIN_ROWS,
        a_score_calls: SCORE_ROWS,
        c_score_calls: SCORE_ROWS,
    })
}

fn assert_quality(run: &QualityRun) -> Result<(), String> {
    if run.a_updates != TRAIN_ROWS
        || run.c_updates != TRAIN_ROWS
        || run.a_score_calls != SCORE_ROWS
        || run.b_score_calls != SCORE_ROWS
        || run.c_score_calls != SCORE_ROWS
    {
        return Err("RLS-QUALITY800 count contract was not satisfied".to_string());
    }
    if run.a.mae > 1e-5 || run.a.r2 < 1.0 - 1e-9 {
        return Err(format!(
            "A quality failed: MAE={} R2={}",
            run.a.mae, run.a.r2
        ));
    }
    if run.b.mae > 1e-4 || run.b.r2 < 1.0 - 1e-6 {
        return Err(format!(
            "B quality failed: MAE={} R2={}",
            run.b.mae, run.b.r2
        ));
    }
    if run.c.mae > 1e-4 || run.c.r2 < 1.0 - 1e-6 {
        return Err(format!(
            "C quality failed: MAE={} R2={}",
            run.c.mae, run.c.r2
        ));
    }
    Ok(())
}

#[cfg(test)]
fn collect_checked(
    rows: &[Vector],
    mut predict: impl FnMut(&Vector) -> Result<f32, RlsError>,
) -> Result<Vec<f32>, RlsError> {
    let mut outputs = Vec::with_capacity(rows.len());
    for row in rows {
        outputs.push(predict(row)?);
    }
    Ok(outputs)
}

#[cfg(test)]
fn accept_candidate(
    active: &mut seqvex::models::linear::rls::RlsServing32,
    candidate: seqvex::models::linear::rls::RlsServing32,
    rows: &[Vector],
) -> Result<bool, RlsError> {
    if rows.is_empty() {
        return Ok(false);
    }
    let _ = collect_checked(rows, |row| candidate.predict_checked(row))?;
    *active = candidate;
    Ok(true)
}

fn main() {
    match run_quality() {
        Ok(run) => {
            println!(
                "RLS-QUALITY800 D={DIMENSION} lambda={LAMBDA} delta={DELTA} train={TRAIN_ROWS} score={SCORE_ROWS}"
            );
            println!(
                "A updates={} score_calls={} MAE={:.12e} R2={:.12}",
                run.a_updates, run.a_score_calls, run.a.mae, run.a.r2
            );
            println!(
                "B f64_updates={} score_calls={} MAE={:.12e} R2={:.12}",
                run.a_updates, run.b_score_calls, run.b.mae, run.b.r2
            );
            println!(
                "C updates={} score_calls={} MAE={:.12e} R2={:.12}",
                run.c_updates, run.c_score_calls, run.c.mae, run.c.r2
            );
            if let Err(error) = assert_quality(&run) {
                eprintln!("RLS-QUALITY800 failed: {error}");
                std::process::exit(1);
            }
            println!("RLS-QUALITY800 passed all declared limits");
        }
        Err(error) => {
            eprintln!("RLS-QUALITY800 failed: {error}");
            std::process::exit(1);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use seqvex::models::linear::rls::RlsState64;

    fn learned_serving() -> (Rls64, RlsState64, seqvex::models::linear::rls::RlsServing32) {
        let model = Rls64::new(2, 1.0, 1000.0).unwrap();
        let mut state = model.initial_state();
        for t in 0..40 {
            let x0 = (t as f64 * 0.13).sin();
            let x1 = (t as f64 * 0.07).cos();
            let y = 0.7 * x0 - 1.3 * x1;
            state = model.update(&state, &sample64(&[x0, x1], y)).unwrap();
        }
        let serving = model.to_serving_f32(&state).unwrap();
        (model, state, serving)
    }

    fn huge_state() -> RlsState64 {
        let weights = Vector64::from_slice(&[f32::MAX as f64, 0.0]);
        let covariance =
            seqvex::foundation::numerical::Matrix64::from_rows(&[&[1.0, 0.0], &[0.0, 1.0]])
                .unwrap();
        RlsState64::admit(weights, covariance).unwrap()
    }

    #[test]
    fn quality_fixture_passes_declared_counts_and_limits() {
        let run = run_quality().unwrap();
        assert_quality(&run).unwrap();
        assert_eq!(run.a_updates, 400);
        assert_eq!(run.c_updates, 400);
        assert_eq!(run.a_score_calls, 400);
        assert_eq!(run.b_score_calls, 400);
        assert_eq!(run.c_score_calls, 400);
        assert!(run.a.r2 >= 1.0 - 1e-9);
        assert!(run.b.r2 >= 1.0 - 1e-6);
        assert!(run.c.r2 >= 1.0 - 1e-6);
    }

    #[test]
    fn grouped_caller_protocol_is_all_or_error_and_installs_explicitly() {
        let (model, state, active) = learned_serving();
        let state_bits: Vec<u64> = state
            .weights()
            .as_slice()
            .iter()
            .map(|value| value.to_bits())
            .collect();
        let rows = [
            Vector::from_slice(&[0.5, -0.25]),
            Vector::from_slice(&[1.0, 0.25]),
        ];
        let expected = collect_checked(&rows, |row| active.predict_checked(row)).unwrap();
        assert_eq!(expected.len(), rows.len());
        assert!(
            collect_checked(&[], |row| active.predict_checked(row))
                .unwrap()
                .is_empty()
        );

        let huge = model.to_serving_f32(&huge_state()).unwrap();
        let failures = [
            Vector::from_slice(&[2.0, 0.0]),
            Vector::from_slice(&[1.0, 0.0]),
            Vector::from_slice(&[1.0, 0.0]),
        ];
        for start in 0..failures.len() {
            let mut rows = failures.clone();
            rows.rotate_left(start);
            assert_eq!(
                collect_checked(&rows, |row| huge.predict_checked(row)),
                Err(RlsError::NonFiniteOutput)
            );
        }

        let continued = model.update(&state, &sample64(&[0.75, -0.5], 1.5)).unwrap();
        let candidate = model.to_serving_f32(&continued).unwrap();
        let mut active_mut = active;
        let active_before = active_mut.weights().as_slice().to_vec();
        assert!(!accept_candidate(&mut active_mut, candidate.clone(), &[]).unwrap());
        assert_eq!(active_mut.weights().as_slice(), active_before.as_slice());
        assert!(accept_candidate(&mut active_mut, candidate.clone(), &rows).unwrap());
        assert_eq!(
            active_mut.weights().as_slice(),
            candidate.weights().as_slice()
        );

        let failing = model.to_serving_f32(&huge_state()).unwrap();
        let active_bits: Vec<u32> = active_mut
            .weights()
            .as_slice()
            .iter()
            .map(|value| value.to_bits())
            .collect();
        assert_eq!(
            accept_candidate(&mut active_mut, failing, &[Vector::from_slice(&[2.0, 0.0])],),
            Err(RlsError::NonFiniteOutput)
        );
        let after_bits: Vec<u32> = active_mut
            .weights()
            .as_slice()
            .iter()
            .map(|value| value.to_bits())
            .collect();
        assert_eq!(active_bits, after_bits);
        let state_after: Vec<u64> = state
            .weights()
            .as_slice()
            .iter()
            .map(|value| value.to_bits())
            .collect();
        assert_eq!(state_bits, state_after);
    }
}
