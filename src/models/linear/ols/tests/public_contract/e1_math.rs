use super::support::{assert_bits_eq, model, scalar_predict_f32};
use seqvex::foundation::numerical::Vector;

#[test]
fn scalar_oracle_matches_public_negative_zero_prediction_bits() {
    let weights = [-0.0_f32];
    let features = [1.0_f32];
    let bias = -0.0_f32;
    let expected = scalar_predict_f32(&weights, bias, &features);
    let actual = model(&weights, bias)
        .predict(&Vector::from_slice(&features))
        .unwrap();

    assert_eq!(expected.to_bits(), (-0.0_f32).to_bits());
    assert_bits_eq(actual, expected);
}

// --- E.1 mathematical correctness: hand-computable exact cases --------------

#[test]
fn oracle_zero_vector_is_bias_bitwise() {
    let zero_weights = model(&[0.0, 0.0], 0.0);
    let prediction = zero_weights
        .predict(&Vector::from_slice(&[0.0, 0.0]))
        .unwrap();
    assert_bits_eq(prediction, 0.0);

    let with_bias = model(&[1.5, -2.0, 0.25], 0.5);
    let prediction = with_bias
        .predict(&Vector::from_slice(&[0.0, 0.0, 0.0]))
        .unwrap();
    assert_bits_eq(prediction, 0.5);
}

#[test]
fn oracle_one_and_two_dimensional_exact() {
    let one_dimensional = model(&[3.0], 1.0);
    assert_bits_eq(
        one_dimensional
            .predict(&Vector::from_slice(&[2.0]))
            .unwrap(),
        7.0,
    );

    let two_dimensional = model(&[1.0, 2.0], 0.5);
    assert_bits_eq(
        two_dimensional
            .predict(&Vector::from_slice(&[3.0, 4.0]))
            .unwrap(),
        11.5,
    );
}

#[test]
fn oracle_sign_combinations_exact() {
    let negative_coefficients = model(&[-2.0, 3.0], -0.25);
    assert_bits_eq(
        negative_coefficients
            .predict(&Vector::from_slice(&[4.0, -2.0]))
            .unwrap(),
        -14.25,
    );

    let mixed_sign_features = model(&[1.5, -1.5], 2.0);
    assert_bits_eq(
        mixed_sign_features
            .predict(&Vector::from_slice(&[-4.0, 4.0]))
            .unwrap(),
        -10.0,
    );

    let zero_bias = model(&[1.0, 1.0], 0.0);
    assert_bits_eq(
        zero_bias
            .predict(&Vector::from_slice(&[0.5, 0.25]))
            .unwrap(),
        0.75,
    );
}

#[test]
fn oracle_integer_representable_f32_exact() {
    let cases: [(Vec<f32>, f32, Vec<f32>, f32); 5] = [
        (vec![0.25, 4.0], 0.5, vec![8.0, 2.0], 10.5),
        (vec![0.5, 0.5], 0.0, vec![2.0, 4.0], 3.0),
        (vec![8.0, 1024.0], 1.0, vec![0.125, 0.0625], 66.0),
        (vec![16_777_216.0], 0.0, vec![1.0], 16_777_216.0),
        (vec![-0.25, 0.5, 1024.0], 0.5, vec![4.0, -4.0, 0.5], 509.5),
    ];
    for (weights, bias, features, expected) in &cases {
        let model = model(weights, *bias);
        let prediction = model.predict(&Vector::from_slice(features)).unwrap();
        assert_bits_eq(prediction, *expected);
        assert_bits_eq(scalar_predict_f32(weights, *bias, features), *expected);
    }
}

#[test]
fn oracle_exact_cancellation_bitwise() {
    let model = model(&[1.0, 1.0], 1.0);
    let features = [16_777_216.0_f32, -16_777_216.0_f32];
    assert_bits_eq(model.predict(&Vector::from_slice(&features)).unwrap(), 1.0);
}

#[test]
fn reduction_order_is_weight_index_order() {
    // At |1e8| the f32 ulp is 8, so 1e8 + 1.0 rounds back to 1e8. A left fold in
    // weight order therefore gives 0.0, while a reordered sum (pairing the two
    // 1e8-magnitude terms first) would give 1.0. Pinning production to the
    // scalar index-order loop documents the reduction-order contract the
    // micro-batch policy relies on.
    let weights = [1.0e8_f32, 1.0, -1.0e8];
    let features = [1.0_f32, 1.0, 1.0];
    let model = model(&weights, 0.0);
    let produced = model.predict(&Vector::from_slice(&features)).unwrap();
    assert_bits_eq(produced, 0.0);
    assert_bits_eq(produced, scalar_predict_f32(&weights, 0.0, &features));

    let reordered = [1.0e8_f32, -1.0e8, 1.0];
    assert_bits_eq(scalar_predict_f32(&reordered, 0.0, &features), 1.0);
}
