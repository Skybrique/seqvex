use super::support::{Lcg, assert_series_matches_oracle, model};

// --- E.7 non-IID prediction behavior ----------------------------------------

#[test]
fn non_iid_stationary_iid_like() {
    let model = model(&[0.5, -1.25, 2.0], -0.375);
    let mut lcg = Lcg::new(0x5eed_0001);
    let series: Vec<Vec<f32>> = (0..256)
        .map(|_| (0..3).map(|_| lcg.next_f32()).collect())
        .collect();
    assert_series_matches_oracle(&model, &series);
}

#[test]
fn non_iid_strongly_ordered() {
    let model = model(&[0.5, -1.25, 2.0], -0.375);
    let series: Vec<Vec<f32>> = (0..256)
        .map(|index| {
            let value = index as f32 * 0.25;
            vec![value, value * value * 0.01, -value]
        })
        .collect();
    assert_series_matches_oracle(&model, &series);
}

#[test]
fn non_iid_clustered_regimes() {
    let model = model(&[0.5, -1.25, 2.0], -0.375);
    let series: Vec<Vec<f32>> = (0..256)
        .map(|index| {
            if index % 2 == 0 {
                vec![0.5, 0.5, 0.5]
            } else {
                vec![-1.5, 2.5, -3.5]
            }
        })
        .collect();
    assert_series_matches_oracle(&model, &series);
}

#[test]
fn non_iid_abrupt_regime_shift() {
    let model = model(&[0.5, -1.25, 2.0], -0.375);
    let series: Vec<Vec<f32>> = (0..256)
        .map(|index| {
            if index < 128 {
                vec![0.1, -0.2, 0.3]
            } else {
                vec![10.0, -20.0, 30.0]
            }
        })
        .collect();
    assert_series_matches_oracle(&model, &series);
}

#[test]
fn non_iid_gradual_drift() {
    let model = model(&[0.5, -1.25, 2.0], -0.375);
    let series: Vec<Vec<f32>> = (0..256)
        .map(|index| {
            let time = index as f32 * 0.05;
            vec![time, 1.0 - 0.5 * time, -0.25 + 0.1 * time]
        })
        .collect();
    assert_series_matches_oracle(&model, &series);
}

#[test]
fn non_iid_autocorrelated_ar1() {
    let model = model(&[0.5, -1.25, 2.0], -0.375);
    let mut lcg = Lcg::new(0xa0c0_0001);
    let mut previous = [0.0_f32; 3];
    let mut series = Vec::with_capacity(256);
    for _ in 0..256 {
        let mut current = [0.0_f32; 3];
        for (slot, &value) in current.iter_mut().zip(&previous) {
            *slot = 0.8 * value + 0.1 * lcg.next_f32();
        }
        previous = current;
        series.push(current.to_vec());
    }
    assert_series_matches_oracle(&model, &series);
}

#[test]
fn non_iid_repeated_observations() {
    let model = model(&[0.5, -1.25, 2.0], -0.375);
    let series = vec![vec![1.0_f32, -2.0, 0.5]; 256];
    assert_series_matches_oracle(&model, &series);
}

#[test]
fn non_iid_heterogeneous_feature_magnitudes() {
    let model = model(&[0.5, -1.25, 2.0], -0.375);
    let series = vec![vec![1.0e-3_f32, 1.0, 1.0e3]; 64];
    assert_series_matches_oracle(&model, &series);
}
