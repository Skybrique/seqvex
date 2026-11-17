use super::support::{
    assert_a_rank_deficient, assert_a_weights, assert_c_rank_deficient, assert_c_weights,
    exact_rank32, exact_rank64, fit_a, fit_a_result, fit_c, fit_c_result, row32, row64, rows32,
    rows64,
};
use seqvex::models::linear::ols::{
    FitConfig32, FitConfig64, FitError32, FitError64, UnderflowPolicy, fit_f32, fit_f64,
};

#[test]
fn profiles_coexist_without_shared_mutation() {
    let a_features = [row64(&[0.0]), row64(&[1.0]), row64(&[2.0])];
    let a_targets = [1.0, 3.0, 5.0];
    let a = fit_a(&a_features, &a_targets, &[0, 1, 2], &FitConfig64::default());
    let a_model = a.predictor();
    let b_model = a
        .to_serving_f32(UnderflowPolicy::Reject)
        .expect("B conversion");

    let c_features = [row32(&[0.0]), row32(&[1.0]), row32(&[2.0])];
    let c = fit_c(
        &c_features,
        &[1.0, 3.0, 5.0],
        &[0, 1, 2],
        &FitConfig32::default(),
    );
    let c_model = c.into_predictor();

    let a_before = a_model.predict_checked(&a_features[2]).unwrap();
    let b_before = b_model.predict_checked(&c_features[2]).unwrap();
    let c_before = c_model.predict_checked(&c_features[2]).unwrap();

    let _other_a = fit_a(
        &[row64(&[0.0]), row64(&[1.0])],
        &[2.0, 4.0],
        &[0, 1],
        &FitConfig64::default(),
    );
    let _other_c = fit_c(
        &[row32(&[0.0]), row32(&[1.0])],
        &[4.0, 8.0],
        &[0, 1],
        &FitConfig32::default(),
    );

    assert_eq!(
        a_model.predict_checked(&a_features[2]).unwrap().to_bits(),
        a_before.to_bits()
    );
    assert_eq!(
        b_model.predict_checked(&c_features[2]).unwrap().to_bits(),
        b_before.to_bits()
    );
    assert_eq!(
        c_model.predict_checked(&c_features[2]).unwrap().to_bits(),
        c_before.to_bits()
    );
}

#[test]
fn a_and_c_ignore_unselected_nonfinite_values_and_use_one_feature() {
    let features64 = [row64(&[0.0]), row64(&[1.0]), row64(&[f64::NAN])];
    let targets64 = [1.0, 3.0, f64::NAN];
    let a = fit_a(&features64, &targets64, &[0, 1], &FitConfig64::default());
    let mut changed_features64 = features64.clone();
    changed_features64[2] = row64(&[f64::INFINITY]);
    let changed_targets64 = [1.0, 3.0, f64::NEG_INFINITY];
    let a_changed = fit_a(
        &changed_features64,
        &changed_targets64,
        &[0, 1],
        &FitConfig64::default(),
    );
    assert_eq!(a, a_changed);
    assert_eq!(a.weights().len(), 1);
    assert_eq!(a.diagnostics().features, 1);
    assert_eq!(a.diagnostics().rank, 2);
    assert_eq!(a.diagnostics().required_rank, 2);

    let selected_change = fit_a(
        &[row64(&[0.0]), row64(&[1.0]), row64(&[f64::NAN])],
        &[1.0, 4.0, f64::NAN],
        &[0, 1],
        &FitConfig64::default(),
    );
    assert_ne!(a.weights(), selected_change.weights());

    let features32 = [row32(&[0.0]), row32(&[1.0]), row32(&[f32::NAN])];
    let targets32 = [1.0, 3.0, f32::NAN];
    let c = fit_c(&features32, &targets32, &[0, 1], &FitConfig32::default());
    let changed_features32 = [row32(&[0.0]), row32(&[1.0]), row32(&[f32::INFINITY])];
    let changed_targets32 = [1.0, 3.0, f32::NEG_INFINITY];
    let c_changed = fit_c(
        &changed_features32,
        &changed_targets32,
        &[0, 1],
        &FitConfig32::default(),
    );
    assert_eq!(c, c_changed);
    assert_eq!(c.weights().len(), 1);
    assert_eq!(c.diagnostics().features, 1);
    assert_eq!(c.diagnostics().rank, 2);
    assert_eq!(c.diagnostics().required_rank, 2);

    let selected_change = fit_c(
        &[row32(&[0.0]), row32(&[1.0]), row32(&[f32::NAN])],
        &[1.0, 4.0, f32::NAN],
        &[0, 1],
        &FitConfig32::default(),
    );
    assert_ne!(c.weights(), selected_change.weights());
}

#[test]
fn a_and_c_enforce_requested_workspace_boundary() {
    let features64 = [row64(&[0.0]), row64(&[1.0]), row64(&[2.0])];
    let targets64 = [1.0, 3.0, 5.0];
    let baseline_a = fit_a(&features64, &targets64, &[0, 1, 2], &FitConfig64::default());
    let requested_a = baseline_a.diagnostics().requested_workspace_bytes;
    let exact_a = fit_f64(
        &features64,
        &targets64,
        &[0, 1, 2],
        &FitConfig64 {
            max_workspace_bytes: requested_a,
            ..FitConfig64::default()
        },
    )
    .expect("A fits at its requested workspace boundary");
    assert_eq!(exact_a.diagnostics().requested_workspace_bytes, requested_a);
    assert_eq!(
        fit_f64(
            &features64,
            &targets64,
            &[0, 1, 2],
            &FitConfig64 {
                max_workspace_bytes: requested_a - 1,
                ..FitConfig64::default()
            },
        ),
        Err(FitError64::Capacity {
            requested: requested_a,
            limit: requested_a - 1,
        })
    );

    let features32 = [row32(&[0.0]), row32(&[1.0]), row32(&[2.0])];
    let targets32 = [1.0, 3.0, 5.0];
    let baseline_c = fit_c(&features32, &targets32, &[0, 1, 2], &FitConfig32::default());
    let requested_c = baseline_c.diagnostics().requested_workspace_bytes;
    let exact_c = fit_f32(
        &features32,
        &targets32,
        &[0, 1, 2],
        &FitConfig32 {
            max_workspace_bytes: requested_c,
            ..FitConfig32::default()
        },
    )
    .expect("C fits at its requested workspace boundary");
    assert_eq!(exact_c.diagnostics().requested_workspace_bytes, requested_c);
    assert_eq!(
        fit_f32(
            &features32,
            &targets32,
            &[0, 1, 2],
            &FitConfig32 {
                max_workspace_bytes: requested_c - 1,
                ..FitConfig32::default()
            },
        ),
        Err(FitError32::Capacity {
            requested: requested_c,
            limit: requested_c - 1,
        })
    );
}

#[test]
fn native_profiles_preserve_index_before_rank_configuration_precedence() {
    let features64 = [row64(&[0.0]), row64(&[1.0])];
    let targets64 = [0.0, 1.0];
    let config64 = FitConfig64 {
        rank_threshold: f64::NAN,
        max_workspace_bytes: 0,
        ..FitConfig64::default()
    };
    assert_eq!(
        fit_f64(&features64, &targets64, &[2], &config64),
        Err(FitError64::IndexOutOfRange { index: 2, rows: 2 })
    );

    let features32 = [row32(&[0.0]), row32(&[1.0])];
    let targets32 = [0.0, 1.0];
    let config32 = FitConfig32 {
        rank_threshold: f32::NAN,
        max_workspace_bytes: 0,
        ..FitConfig32::default()
    };
    assert_eq!(
        fit_f32(&features32, &targets32, &[2], &config32),
        Err(FitError32::IndexOutOfRange { index: 2, rows: 2 })
    );
}

#[test]
fn faer_265_public_controls_use_explicit_policy_and_independent_rank() {
    let original = [
        [
            f64::from_bits(0x4341c37937e08000),
            f64::from_bits(0xc30c6bf526340000),
            0.0,
        ],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ];
    let original32 = original.map(|row| row.map(|value| value as f32));
    let zero3 = [0.0f64; 3];
    let zero3f = [0.0f32; 3];
    assert_eq!(exact_rank64(&original, false), 3);
    assert_eq!(exact_rank32(&original32, false), 3);
    assert_a_rank_deficient(&original, &zero3, false, 3);
    assert_c_rank_deficient(&original32, &zero3f, false, 3);
    assert_a_rank_deficient(&original, &zero3, true, 4);
    assert_c_rank_deficient(&original32, &zero3f, true, 4);

    let augmented = [original[0], original[1], original[2], [0.0, 0.0, 0.0]];
    let augmented32 = augmented.map(|row| row.map(|value| value as f32));
    let zero4 = [0.0f64; 4];
    let zero4f = [0.0f32; 4];
    assert_eq!(exact_rank64(&augmented, true), 4);
    assert_eq!(exact_rank32(&augmented32, true), 4);
    assert_a_rank_deficient(&augmented, &zero4, true, 4);
    assert_c_rank_deficient(&augmented32, &zero4f, true, 4);

    let rank_one = [
        [1.0e6, 1.0e6, 1.0e6],
        [1.0e3, 1.0e3, 1.0e3],
        [1.0, 1.0, 1.0],
    ];
    let rank_one32 = rank_one.map(|row| row.map(|value| value as f32));
    assert_eq!(exact_rank64(&rank_one, false), 1);
    assert_eq!(exact_rank32(&rank_one32, false), 1);
    assert_a_rank_deficient(&rank_one, &zero3, false, 3);
    assert_c_rank_deficient(&rank_one32, &zero3f, false, 3);
    assert_a_rank_deficient(&rank_one, &zero3, true, 4);
    assert_c_rank_deficient(&rank_one32, &zero3f, true, 4);
}

#[test]
fn moderate_controls_solve_and_restore_feature_order() {
    let x = [[2.0, -1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    let y = [0.0, 2.0, 3.0];
    assert_eq!(exact_rank64(&x, false), 3);
    assert_eq!(exact_rank32(&x.map(|row| row.map(|v| v as f32)), false), 3);
    let a = fit_a_result(&x, &y, false).expect("moderate A fit");
    assert_eq!(a.diagnostics().rank, 3);
    assert_a_weights(&a, &[1.0, 2.0, 3.0], 0.0);
    assert!(a.diagnostics().rss_f64.abs() <= 2.0 * 3.0 * 1.0e-22);
    let a_model = a.into_predictor();
    for (row, target) in rows64(&x).iter().zip(y) {
        assert!((a_model.predict_checked(row).unwrap() - target).abs() <= 1.0e-11);
    }

    let x32 = x.map(|row| row.map(|value| value as f32));
    let y32 = y.map(|value| value as f32);
    let c = fit_c_result(&x32, &y32, false).expect("moderate C fit");
    assert_eq!(c.diagnostics().rank, 3);
    assert_c_weights(&c, &[1.0, 2.0, 3.0], 0.0);
    assert!(c.diagnostics().rss_f32.abs() <= 2.0 * 3.0 * 1.0e-8);
    let c_model = c.into_predictor();
    for (row, target) in rows32(&x32).iter().zip(y32) {
        assert!((c_model.predict_checked(row).unwrap() - target).abs() <= 1.0e-4);
    }

    let x_aug = [x[0], x[1], x[2], [0.0, 0.0, 0.0]];
    let y_aug = [4.0, 6.0, 7.0, 4.0];
    assert_eq!(exact_rank64(&x_aug, true), 4);
    let a_aug = fit_a_result(&x_aug, &y_aug, true).expect("moderate intercept A fit");
    assert_eq!(a_aug.diagnostics().required_rank, 4);
    assert_a_weights(&a_aug, &[1.0, 2.0, 3.0], 4.0);
    assert!(a_aug.diagnostics().rss_f64.abs() <= 2.0 * 4.0 * 1.0e-22);

    let x_aug32 = x_aug.map(|row| row.map(|value| value as f32));
    let y_aug32 = y_aug.map(|value| value as f32);
    let c_aug = fit_c_result(&x_aug32, &y_aug32, true).expect("moderate intercept C fit");
    assert_eq!(c_aug.diagnostics().required_rank, 4);
    assert_c_weights(&c_aug, &[1.0, 2.0, 3.0], 4.0);
    assert!(c_aug.diagnostics().rss_f32.abs() <= 2.0 * 4.0 * 1.0e-8);

    let permuted = x.map(|row| [row[2], row[0], row[1]]);
    let permuted32 = x_aug32.map(|row| [row[2], row[0], row[1]]);
    assert_eq!(exact_rank64(&permuted, false), 3);
    let permuted_a = fit_a_result(&permuted, &y, false).expect("permuted A fit");
    assert_a_weights(&permuted_a, &[3.0, 1.0, 2.0], 0.0);
    let permuted_a_model = permuted_a.into_predictor();
    for (row, target) in rows64(&permuted).iter().zip(y) {
        assert!((permuted_a_model.predict_checked(row).unwrap() - target).abs() <= 1.0e-11);
    }
    let permuted_c = fit_c_result(&permuted32, &y_aug32, true).expect("permuted C fit");
    assert_c_weights(&permuted_c, &[3.0, 1.0, 2.0], 4.0);
}
