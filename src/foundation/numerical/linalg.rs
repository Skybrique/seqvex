//! Dense row-major f32 and f64 matrix-vector computation: `y = W x`.
//!
//! This module provides the linear matrix-vector operation `y = W x`; an affine
//! transform additionally requires adding a bias. Storage is dense and row-major,
//! so the hot loop reads each weight row and the input sequentially. The GRU uses
//! the f32 operation. No transpose, blocking, packing, or SIMD until a benchmark
//! justifies it (`docs/KILO_CONNECTOME_SPRINT_REVISED.md` §11).

use super::vector::{DimensionMismatch, Vector, Vector64};

/// A dense, row-major `f32` matrix.
#[derive(Debug, Clone, PartialEq)]
pub struct Matrix {
    rows: usize,
    cols: usize,
    data: Vec<f32>,
}

impl Matrix {
    /// Builds a matrix from rows, requiring every row to have equal length.
    pub fn from_rows(rows: &[&[f32]]) -> Result<Self, DimensionMismatch> {
        let cols = rows.first().map_or(0, |row| row.len());
        let mut data = Vec::with_capacity(rows.len() * cols);
        for row in rows {
            if row.len() != cols {
                return Err(DimensionMismatch {
                    expected: cols,
                    actual: row.len(),
                });
            }
            data.extend_from_slice(row);
        }
        Ok(Self {
            rows: rows.len(),
            cols,
            data,
        })
    }

    /// Builds a matrix by evaluating `f(row, col)` in row-major order.
    pub fn from_fn(rows: usize, cols: usize, mut f: impl FnMut(usize, usize) -> f32) -> Self {
        let mut data = Vec::with_capacity(rows * cols);
        for row in 0..rows {
            for col in 0..cols {
                data.push(f(row, col));
            }
        }
        Self { rows, cols, data }
    }

    /// Number of rows (output dimension).
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Number of columns (input dimension).
    pub fn cols(&self) -> usize {
        self.cols
    }

    /// The elements as a contiguous row-major slice.
    pub fn as_slice(&self) -> &[f32] {
        &self.data
    }

    /// The element at `(row, col)`, if in bounds.
    pub fn get(&self, row: usize, col: usize) -> Option<f32> {
        if row < self.rows && col < self.cols {
            Some(self.data[row * self.cols + col])
        } else {
            None
        }
    }

    /// Computes `y = W x`, failing when the input length differs from `cols`.
    pub fn mul_vector(&self, input: &Vector) -> Result<Vector, DimensionMismatch> {
        if self.cols != input.len() {
            return Err(DimensionMismatch {
                expected: self.cols,
                actual: input.len(),
            });
        }
        let input = input.as_slice();
        Ok(Vector::from_vec(super::kernels::matvec(
            self.rows, self.cols, &self.data, input,
        )))
    }
}

/// A dense, row-major `f64` matrix with native `f64` arithmetic.
///
/// Matrix values are not checked for finiteness. Ragged rows and incompatible
/// input dimensions return [`DimensionMismatch`].
#[derive(Debug, Clone, PartialEq)]
pub struct Matrix64 {
    rows: usize,
    cols: usize,
    data: Vec<f64>,
}

impl Matrix64 {
    /// Builds a matrix from rows, requiring every row to have equal length.
    pub fn from_rows(rows: &[&[f64]]) -> Result<Self, DimensionMismatch> {
        let cols = rows.first().map_or(0, |row| row.len());
        let mut data = Vec::with_capacity(rows.len() * cols);
        for row in rows {
            if row.len() != cols {
                return Err(DimensionMismatch {
                    expected: cols,
                    actual: row.len(),
                });
            }
            data.extend_from_slice(row);
        }
        Ok(Self {
            rows: rows.len(),
            cols,
            data,
        })
    }

    /// Builds a matrix by evaluating `f(row, col)` in row-major order.
    pub fn from_fn(rows: usize, cols: usize, mut f: impl FnMut(usize, usize) -> f64) -> Self {
        let mut data = Vec::with_capacity(rows * cols);
        for row in 0..rows {
            for col in 0..cols {
                data.push(f(row, col));
            }
        }
        Self { rows, cols, data }
    }

    /// Number of rows (output dimension).
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Number of columns (input dimension).
    pub fn cols(&self) -> usize {
        self.cols
    }

    /// The elements as a contiguous row-major slice.
    pub fn as_slice(&self) -> &[f64] {
        &self.data
    }

    /// The element at `(row, col)`, if in bounds.
    pub fn get(&self, row: usize, col: usize) -> Option<f64> {
        if row < self.rows && col < self.cols {
            Some(self.data[row * self.cols + col])
        } else {
            None
        }
    }

    /// Computes `y = W x`, failing when the input length differs from `cols`.
    pub fn mul_vector(&self, input: &Vector64) -> Result<Vector64, DimensionMismatch> {
        if self.cols != input.len() {
            return Err(DimensionMismatch {
                expected: self.cols,
                actual: input.len(),
            });
        }
        Ok(Vector64::from_vec(super::kernels::matvec(
            self.rows,
            self.cols,
            &self.data,
            input.as_slice(),
        )))
    }
}
