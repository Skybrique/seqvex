//! Private scalar-generic loops shared by the concrete f32 and f64 containers.

use std::iter::Sum;
use std::ops::{Add, Mul};

pub(crate) fn map_slice<T: Copy>(data: &[T], f: impl Fn(T) -> T) -> Vec<T> {
    data.iter().map(|&value| f(value)).collect()
}

pub(crate) fn zip_with<T: Copy>(a: &[T], b: &[T], f: impl Fn(T, T) -> T) -> Vec<T> {
    a.iter()
        .zip(b)
        .map(|(&left, &right)| f(left, right))
        .collect()
}

pub(crate) fn dot<T: Copy + Add<Output = T> + Mul<Output = T> + Sum<T>>(a: &[T], b: &[T]) -> T {
    a.iter().zip(b).map(|(&left, &right)| left * right).sum()
}

pub(crate) fn matvec<T: Copy + Add<Output = T> + Mul<Output = T> + Sum<T>>(
    rows: usize,
    cols: usize,
    data: &[T],
    input: &[T],
) -> Vec<T> {
    let mut output = Vec::with_capacity(rows);
    for row in 0..rows {
        let start = row * cols;
        output.push(
            data[start..start + cols]
                .iter()
                .zip(input)
                .map(|(&weight, &value)| weight * value)
                .sum(),
        );
    }
    output
}
