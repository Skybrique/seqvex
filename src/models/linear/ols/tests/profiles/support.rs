use num_bigint::BigInt;
use num_rational::BigRational;
use num_traits::Zero;
use seqvex::foundation::numerical::{Vector, Vector64};
use seqvex::foundation::observation::Observation;
use seqvex::models::linear::ols::{
    FitConfig32, FitConfig64, FitError32, FitError64, fit_f32, fit_f64,
};

pub(super) fn row64(values: &[f64]) -> Vector64 {
    Vector64::from_slice(values)
}

pub(super) fn row32(values: &[f32]) -> Vector {
    Vector::from_slice(values)
}

pub(super) fn obs64(values: &[f64]) -> Observation<Vector64> {
    Observation::new(row64(values))
}

pub(super) fn obs32(values: &[f32]) -> Observation<Vector> {
    Observation::new(row32(values))
}

pub(super) fn fit_a(
    features: &[Vector64],
    targets: &[f64],
    train: &[usize],
    config: &FitConfig64,
) -> seqvex::models::linear::ols::FitResult64 {
    fit_f64(features, targets, train, config).expect("A fit succeeds")
}

pub(super) fn fit_c(
    features: &[Vector],
    targets: &[f32],
    train: &[usize],
    config: &FitConfig32,
) -> seqvex::models::linear::ols::FitResult32 {
    fit_f32(features, targets, train, config).expect("C fit succeeds")
}

pub(super) fn one_weight(value: f64) -> seqvex::models::linear::ols::FitResult64 {
    fit_a(
        &[row64(&[1.0])],
        &[value],
        &[0],
        &FitConfig64 {
            intercept: false,
            ..FitConfig64::default()
        },
    )
}

pub(super) fn br64(value: f64) -> BigRational {
    BigRational::from_float(value).expect("finite fixture")
}

pub(super) fn br32(value: f32) -> BigRational {
    BigRational::from_float(f64::from(value)).expect("finite fixture")
}

pub(super) fn rat_int(value: i64) -> BigRational {
    BigRational::from_integer(BigInt::from(value))
}

#[allow(clippy::needless_range_loop)]
pub(super) fn exact_rank(matrix: &[Vec<BigRational>]) -> usize {
    let rows = matrix.len();
    if rows == 0 {
        return 0;
    }
    let cols = matrix[0].len();
    let mut a = matrix.to_vec();
    let mut rank = 0usize;
    for col in 0..cols {
        let Some(pivot_row) = (rank..rows).find(|&r| !a[r][col].is_zero()) else {
            continue;
        };
        a.swap(rank, pivot_row);
        let pivot = a[rank][col].clone();
        for r in 0..rows {
            if r != rank && !a[r][col].is_zero() {
                let factor = &a[r][col] / &pivot;
                for c in col..cols {
                    let sub = &factor * &a[rank][c];
                    a[r][c] = &a[r][c] - &sub;
                }
            }
        }
        rank += 1;
        if rank == rows {
            break;
        }
    }
    rank
}

pub(super) fn exact_rank64(xs: &[[f64; 3]], intercept: bool) -> usize {
    let matrix = xs
        .iter()
        .map(|row| {
            let mut columns = Vec::with_capacity(if intercept { 4 } else { 3 });
            if intercept {
                columns.push(rat_int(1));
            }
            columns.extend(row.iter().map(|&value| br64(value)));
            columns
        })
        .collect::<Vec<_>>();
    exact_rank(&matrix)
}

pub(super) fn exact_rank32(xs: &[[f32; 3]], intercept: bool) -> usize {
    let matrix = xs
        .iter()
        .map(|row| {
            let mut columns = Vec::with_capacity(if intercept { 4 } else { 3 });
            if intercept {
                columns.push(rat_int(1));
            }
            columns.extend(row.iter().map(|&value| br32(value)));
            columns
        })
        .collect::<Vec<_>>();
    exact_rank(&matrix)
}

pub(super) fn rows64(xs: &[[f64; 3]]) -> Vec<Vector64> {
    xs.iter().map(|row| row64(row)).collect()
}

pub(super) fn rows32(xs: &[[f32; 3]]) -> Vec<Vector> {
    xs.iter().map(|row| row32(row)).collect()
}

pub(super) fn fit_a_result(
    xs: &[[f64; 3]],
    ys: &[f64],
    intercept: bool,
) -> Result<seqvex::models::linear::ols::FitResult64, FitError64> {
    let indices: Vec<usize> = (0..xs.len()).collect();
    fit_f64(
        &rows64(xs),
        ys,
        &indices,
        &FitConfig64 {
            intercept,
            rank_threshold: 1.0,
            ..FitConfig64::default()
        },
    )
}

pub(super) fn fit_c_result(
    xs: &[[f32; 3]],
    ys: &[f32],
    intercept: bool,
) -> Result<seqvex::models::linear::ols::FitResult32, FitError32> {
    let indices: Vec<usize> = (0..xs.len()).collect();
    fit_f32(
        &rows32(xs),
        ys,
        &indices,
        &FitConfig32 {
            intercept,
            rank_threshold: 1.0,
            ..FitConfig32::default()
        },
    )
}

pub(super) fn assert_a_rank_deficient(
    xs: &[[f64; 3]],
    ys: &[f64],
    intercept: bool,
    required: usize,
) {
    match fit_a_result(xs, ys, intercept) {
        Err(FitError64::RankDeficient {
            rank,
            required: got,
        }) => {
            assert_eq!(got, required);
            assert!(rank < required);
        }
        other => panic!("A expected RankDeficient required={required}, got {other:?}"),
    }
}

pub(super) fn assert_c_rank_deficient(
    xs: &[[f32; 3]],
    ys: &[f32],
    intercept: bool,
    required: usize,
) {
    match fit_c_result(xs, ys, intercept) {
        Err(FitError32::RankDeficient {
            rank,
            required: got,
        }) => {
            assert_eq!(got, required);
            assert!(rank < required);
        }
        other => panic!("C expected RankDeficient required={required}, got {other:?}"),
    }
}

pub(super) fn assert_a_weights(
    result: &seqvex::models::linear::ols::FitResult64,
    expected: &[f64],
    bias: f64,
) {
    assert_eq!(result.weights().len(), expected.len());
    for (index, (&actual, &wanted)) in result.weights().as_slice().iter().zip(expected).enumerate()
    {
        assert!(
            (actual - wanted).abs() <= 1.0e-12,
            "A weight {index}: {actual} vs {wanted}"
        );
    }
    assert!((result.intercept() - bias).abs() <= 1.0e-12);
}

pub(super) fn assert_c_weights(
    result: &seqvex::models::linear::ols::FitResult32,
    expected: &[f32],
    bias: f32,
) {
    assert_eq!(result.weights().len(), expected.len());
    for (index, (&actual, &wanted)) in result.weights().as_slice().iter().zip(expected).enumerate()
    {
        assert!(
            (f64::from(actual) - f64::from(wanted)).abs() <= 1.0e-5,
            "C weight {index}: {actual} vs {wanted}"
        );
    }
    assert!((f64::from(result.intercept()) - f64::from(bias)).abs() <= 1.0e-5);
}
