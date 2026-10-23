//! Contiguous one-dimensional `f32` and `f64` data.
//!
//! The concrete types are deliberately not tensors: fixed rank, no broadcasting,
//! no device memory, and no autodiff (`docs/KILO_CONNECTOME_SPRINT_REVISED.md` §10).

/// Error returned when operands or parameters have incompatible dimensions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DimensionMismatch {
    /// The dimension required by the operation.
    pub expected: usize,
    /// The dimension actually supplied.
    pub actual: usize,
}

impl std::fmt::Display for DimensionMismatch {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "dimension mismatch: expected {}, got {}",
            self.expected, self.actual
        )
    }
}

impl std::error::Error for DimensionMismatch {}

/// A contiguous, fixed-length `f32` vector.
#[derive(Debug, Clone, PartialEq)]
pub struct Vector {
    data: Vec<f32>,
}

impl Vector {
    /// A vector of `len` zeros.
    pub fn zeros(len: usize) -> Self {
        Self {
            data: vec![0.0; len],
        }
    }

    /// Copies `values` into a new vector.
    pub fn from_slice(values: &[f32]) -> Self {
        Self {
            data: values.to_vec(),
        }
    }

    pub(crate) fn from_vec(data: Vec<f32>) -> Self {
        Self { data }
    }

    /// Builds a vector by evaluating `f` for each index in order.
    pub fn from_fn(len: usize, f: impl FnMut(usize) -> f32) -> Self {
        Self {
            data: (0..len).map(f).collect(),
        }
    }

    /// Number of elements.
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Whether the vector has no elements.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// The elements as a contiguous slice.
    pub fn as_slice(&self) -> &[f32] {
        &self.data
    }

    /// The elements as a mutable contiguous slice.
    pub fn as_mut_slice(&mut self) -> &mut [f32] {
        &mut self.data
    }

    /// The element at `index`, if present.
    pub fn get(&self, index: usize) -> Option<f32> {
        self.data.get(index).copied()
    }

    /// Iterates over the elements in order.
    pub fn iter(&self) -> impl Iterator<Item = &f32> {
        self.data.iter()
    }

    /// Applies `f` to each element, allocating one output vector.
    pub fn map(&self, f: impl Fn(f32) -> f32) -> Self {
        Self {
            data: super::kernels::map_slice(&self.data, f),
        }
    }

    /// Elementwise `self + other`; fails when lengths differ.
    pub fn add(&self, other: &Vector) -> Result<Self, DimensionMismatch> {
        self.zip_with(other, |a, b| a + b)
    }

    /// Elementwise `self * other`; fails when lengths differ.
    pub fn multiply(&self, other: &Vector) -> Result<Self, DimensionMismatch> {
        self.zip_with(other, |a, b| a * b)
    }

    /// Elementwise scalar multiplication.
    pub fn scale(&self, factor: f32) -> Self {
        self.map(|value| value * factor)
    }

    /// Dot product of two equal-length vectors; fails when lengths differ.
    pub fn dot(&self, other: &Vector) -> Result<f32, DimensionMismatch> {
        if self.len() != other.len() {
            return Err(DimensionMismatch {
                expected: self.len(),
                actual: other.len(),
            });
        }
        Ok(super::kernels::dot(&self.data, &other.data))
    }

    /// Elementwise `1 - self`.
    pub fn complement(&self) -> Self {
        self.map(|value| 1.0 - value)
    }

    fn zip_with(
        &self,
        other: &Vector,
        f: impl Fn(f32, f32) -> f32,
    ) -> Result<Self, DimensionMismatch> {
        if self.len() != other.len() {
            return Err(DimensionMismatch {
                expected: self.len(),
                actual: other.len(),
            });
        }
        Ok(Self {
            data: super::kernels::zip_with(&self.data, &other.data, f),
        })
    }
}

/// A contiguous, fixed-length `f64` vector with native `f64` storage and arithmetic.
///
/// Values are not checked for finiteness. Operations retain primitive IEEE
/// behavior, and shape mismatches return [`DimensionMismatch`].
#[derive(Debug, Clone, PartialEq)]
pub struct Vector64 {
    data: Vec<f64>,
}

impl Vector64 {
    /// A vector of `len` zeros.
    pub fn zeros(len: usize) -> Self {
        Self {
            data: vec![0.0; len],
        }
    }

    /// Copies `values` into a new vector.
    pub fn from_slice(values: &[f64]) -> Self {
        Self {
            data: values.to_vec(),
        }
    }

    /// Builds a vector by evaluating `f` for each index in order.
    pub fn from_fn(len: usize, f: impl FnMut(usize) -> f64) -> Self {
        Self {
            data: (0..len).map(f).collect(),
        }
    }

    pub(crate) fn from_vec(data: Vec<f64>) -> Self {
        Self { data }
    }

    /// Number of elements.
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Whether the vector has no elements.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// The elements as a contiguous slice.
    pub fn as_slice(&self) -> &[f64] {
        &self.data
    }

    /// The elements as a mutable contiguous slice.
    pub fn as_mut_slice(&mut self) -> &mut [f64] {
        &mut self.data
    }

    /// The element at `index`, if present.
    pub fn get(&self, index: usize) -> Option<f64> {
        self.data.get(index).copied()
    }

    /// Iterates over the elements in order.
    pub fn iter(&self) -> impl Iterator<Item = &f64> {
        self.data.iter()
    }

    /// Applies `f` to each element, allocating one output vector.
    pub fn map(&self, f: impl Fn(f64) -> f64) -> Self {
        Self {
            data: super::kernels::map_slice(&self.data, f),
        }
    }

    /// Elementwise `self + other`; fails when lengths differ.
    pub fn add(&self, other: &Vector64) -> Result<Self, DimensionMismatch> {
        self.zip_with(other, |a, b| a + b)
    }

    /// Elementwise `self * other`; fails when lengths differ.
    pub fn multiply(&self, other: &Vector64) -> Result<Self, DimensionMismatch> {
        self.zip_with(other, |a, b| a * b)
    }

    /// Elementwise scalar multiplication.
    pub fn scale(&self, factor: f64) -> Self {
        self.map(|value| value * factor)
    }

    /// Dot product of two equal-length vectors; fails when lengths differ.
    pub fn dot(&self, other: &Vector64) -> Result<f64, DimensionMismatch> {
        if self.len() != other.len() {
            return Err(DimensionMismatch {
                expected: self.len(),
                actual: other.len(),
            });
        }
        Ok(super::kernels::dot(&self.data, &other.data))
    }

    /// Elementwise `1 - self`.
    pub fn complement(&self) -> Self {
        self.map(|value| 1.0 - value)
    }

    fn zip_with(
        &self,
        other: &Vector64,
        f: impl Fn(f64, f64) -> f64,
    ) -> Result<Self, DimensionMismatch> {
        if self.len() != other.len() {
            return Err(DimensionMismatch {
                expected: self.len(),
                actual: other.len(),
            });
        }
        Ok(Self {
            data: super::kernels::zip_with(&self.data, &other.data, f),
        })
    }
}
