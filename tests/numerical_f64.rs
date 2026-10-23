use seqvex::foundation::numerical::{self, linalg::Matrix64, vector::Vector64};

fn baseline_f64_dot(left: &[f64], right: &[f64]) -> f64 {
    left.iter().zip(right).map(|(&a, &b)| a * b).sum()
}

fn baseline_f64_matvec_row(weights: &[f64], input: &[f64]) -> f64 {
    weights
        .iter()
        .zip(input)
        .map(|(weight, value)| weight * value)
        .sum()
}

#[test]
fn vector64_construction_access_and_callbacks() {
    let zeros = Vector64::zeros(2);
    assert_eq!(zeros.as_slice(), &[0.0, 0.0]);
    assert_eq!(zeros.get(2), None);
    assert!(zeros.iter().all(|value| *value == 0.0));

    let mut index = 0;
    let mut vector = Vector64::from_fn(3, |_| {
        index += 1;
        index as f64
    });
    assert_eq!(vector.as_slice(), &[1.0, 2.0, 3.0]);
    vector.as_mut_slice()[1] = 4.0;
    assert_eq!(Vector64::from_slice(vector.as_slice()), vector);
    assert_eq!(vector.map(|value| value * 2.0).as_slice(), &[2.0, 8.0, 6.0]);
}

#[test]
fn vector64_arithmetic_matches_analytical_values() {
    let left = Vector64::from_slice(&[1.0, 2.0, 3.0]);
    let right = Vector64::from_slice(&[4.0, 5.0, 6.0]);
    assert_eq!(left.add(&right).unwrap().as_slice(), &[5.0, 7.0, 9.0]);
    assert_eq!(
        left.multiply(&right).unwrap().as_slice(),
        &[4.0, 10.0, 18.0]
    );
    assert_eq!(left.scale(2.0).as_slice(), &[2.0, 4.0, 6.0]);
    assert_eq!(left.complement().as_slice(), &[0.0, -1.0, -2.0]);
    assert_eq!(left.dot(&right), Ok(32.0));
    assert_eq!(
        Vector64::from_slice(&[1.0e8, 1.0e8 + 1.0]).dot(&Vector64::from_slice(&[1.0, 1.0])),
        Ok(200000001.0)
    );
    assert_eq!(
        Vector64::from_slice(&[1.0e8, 1.0e8 + 1.0]).dot(&Vector64::from_slice(&[1.0, -1.0])),
        Ok(-1.0)
    );
}

#[test]
fn vector64_errors_empty_and_ieee_behavior() {
    let empty = Vector64::zeros(0);
    assert!(empty.is_empty());
    assert_eq!(empty.dot(&empty), Ok(0.0));
    let one = Vector64::from_slice(&[1.0]);
    assert_eq!(
        one.add(&empty),
        Err(numerical::DimensionMismatch {
            expected: 1,
            actual: 0
        })
    );
    assert_eq!(
        one.multiply(&empty),
        Err(numerical::DimensionMismatch {
            expected: 1,
            actual: 0
        })
    );
    assert_eq!(
        one.dot(&empty),
        Err(numerical::DimensionMismatch {
            expected: 1,
            actual: 0
        })
    );
    assert_eq!(
        Vector64::from_slice(&[f64::NAN]).dot(&empty),
        Err(numerical::DimensionMismatch {
            expected: 1,
            actual: 0
        })
    );

    let product = Vector64::from_slice(&[-0.0])
        .multiply(&Vector64::from_slice(&[1.0]))
        .unwrap();
    assert!(product.get(0).unwrap().is_sign_negative());
    assert!(
        Vector64::from_slice(&[-0.0])
            .dot(&Vector64::from_slice(&[1.0]))
            .unwrap()
            .is_sign_negative()
    );
    assert_eq!(
        Vector64::from_slice(&[f64::INFINITY]).scale(1.0).get(0),
        Some(f64::INFINITY)
    );
    assert!(
        Vector64::from_slice(&[f64::NAN])
            .map(|value| value)
            .get(0)
            .unwrap()
            .is_nan()
    );

    let half_min = f64::MIN_POSITIVE / 2.0;
    let subnormal = Vector64::from_slice(&[f64::MIN_POSITIVE])
        .scale(0.5)
        .get(0)
        .unwrap();
    let bits = subnormal.to_bits();
    println!(
        "subnormal observation: arch={} os={} f64::MIN_POSITIVE/2 bits={bits:#018x}",
        std::env::consts::ARCH,
        std::env::consts::OS
    );
    assert!(bits == 0 || bits == half_min.to_bits());
}

#[test]
fn matrix64_shape_order_and_matvec_are_native_f64() {
    let matrix = Matrix64::from_rows(&[&[1.0, 2.0], &[3.0, 4.0]]).unwrap();
    assert_eq!((matrix.rows(), matrix.cols()), (2, 2));
    assert_eq!(matrix.as_slice(), &[1.0, 2.0, 3.0, 4.0]);
    assert_eq!(matrix.get(1, 0), Some(3.0));
    assert_eq!(matrix.get(2, 0), None);
    assert_eq!(
        matrix
            .mul_vector(&Vector64::from_slice(&[1.0, 0.0]))
            .unwrap()
            .as_slice(),
        &[1.0, 3.0]
    );

    let mut calls = Vec::new();
    let built = Matrix64::from_fn(2, 2, |row, col| {
        calls.push((row, col));
        (row * 2 + col) as f64
    });
    assert_eq!(calls, [(0, 0), (0, 1), (1, 0), (1, 1)]);
    assert_eq!(built.as_slice(), &[0.0, 1.0, 2.0, 3.0]);

    // Independent exact oracle: this first entry is not representable as f32.
    let discriminating = Matrix64::from_rows(&[&[16_777_217.0, -16_777_216.0]]).unwrap();
    let result = discriminating
        .mul_vector(&Vector64::from_slice(&[1.0, 1.0]))
        .unwrap();
    assert_eq!(result.get(0).unwrap().to_bits(), 1.0_f64.to_bits());
}

#[test]
fn matrix64_empty_ragged_and_mismatch_contracts() {
    assert_eq!(
        (
            Matrix64::from_rows(&[]).unwrap().rows(),
            Matrix64::from_rows(&[]).unwrap().cols()
        ),
        (0, 0)
    );
    let empty_rows = Matrix64::from_rows(&[&[], &[]]).unwrap();
    assert_eq!(
        empty_rows
            .mul_vector(&Vector64::zeros(0))
            .unwrap()
            .as_slice(),
        &[0.0, 0.0]
    );
    assert_eq!(
        Matrix64::from_rows(&[&[1.0, 2.0], &[3.0]]),
        Err(numerical::DimensionMismatch {
            expected: 2,
            actual: 1
        })
    );
    let matrix = Matrix64::from_rows(&[&[1.0, 2.0]]).unwrap();
    assert_eq!(
        matrix.mul_vector(&Vector64::from_slice(&[f64::NAN])),
        Err(numerical::DimensionMismatch {
            expected: 2,
            actual: 1
        })
    );
}

#[test]
fn f64_public_types_keep_debug_and_equality_contracts() {
    let vector = Vector64::from_slice(&[1.0]);
    assert_eq!(vector, vector.clone());
    assert_eq!(format!("{vector:?}"), "Vector64 { data: [1.0] }");

    let matrix = Matrix64::from_rows(&[&[1.0]]).unwrap();
    assert_eq!(matrix, matrix.clone());
    assert_eq!(
        format!("{matrix:?}"),
        "Matrix64 { rows: 1, cols: 1, data: [1.0] }"
    );
}

#[test]
fn f64_dot_and_matvec_preserve_baseline_signed_zero_bits() {
    let positive_zero = 0.0_f64.to_bits();
    let negative_zero = (-0.0_f64).to_bits();
    let cases: &[(&[f64], &[f64], u64)] = &[
        (&[], &[], negative_zero),
        (&[-0.0], &[1.0], negative_zero),
        (&[-0.0, -0.0], &[1.0, 1.0], negative_zero),
        (&[0.0, -0.0], &[1.0, 1.0], positive_zero),
        (&[-0.0, 0.0], &[1.0, 1.0], positive_zero),
        (&[1.0, -1.0], &[1.0, 1.0], positive_zero),
    ];

    for (left, right, expected_bits) in cases {
        let left_vector = Vector64::from_slice(left);
        let right_vector = Vector64::from_slice(right);
        let candidate = left_vector.dot(&right_vector).unwrap();
        let baseline = baseline_f64_dot(left, right);
        assert_eq!(candidate.to_bits(), baseline.to_bits());
        assert_eq!(candidate.to_bits(), *expected_bits);

        let matrix = Matrix64::from_rows(&[*left]).unwrap();
        let output = matrix.mul_vector(&right_vector).unwrap();
        let baseline_row = baseline_f64_matvec_row(matrix.as_slice(), right);
        assert_eq!(output.get(0).unwrap().to_bits(), baseline_row.to_bits());
        assert_eq!(output.get(0).unwrap().to_bits(), *expected_bits);
    }

    let empty_rows = Matrix64::from_rows(&[&[], &[]]).unwrap();
    let empty_output = empty_rows.mul_vector(&Vector64::zeros(0)).unwrap();
    assert_eq!(
        empty_output
            .as_slice()
            .iter()
            .map(|x| x.to_bits())
            .collect::<Vec<_>>(),
        [negative_zero; 2]
    );

    let zero_rows = Matrix64::from_rows(&[]).unwrap();
    assert!(
        zero_rows
            .mul_vector(&Vector64::zeros(0))
            .unwrap()
            .is_empty()
    );
}
