//! Independent OLS numerical oracle (T0) and Stage-C analysis controls (T2).
//!
//! This target is deliberately **self-contained**: it does not use a Seqvex
//! fitter, the future production fitting APIs, or `faer`. It exists to produce
//! trustworthy expected results *before* any fitter is written, and to hold the
//! executable controls for the numerical assumptions the fitter will rely on.
//!
//! # References
//!
//! * **(A) closed-form scalar OLS** for a single feature with intercept, in
//!   `f64` — a method-independent check for the `d = 1` fixtures.
//! * **(B) exact rational least squares** via normal equations solved with exact
//!   Gaussian elimination over [`BigRational`]. Exact arithmetic means the
//!   *result* is exact for full-rank rational inputs regardless of conditioning;
//!   this is the authority for the small integer and extreme binary fixtures.
//! * **(C) symbolic power-of-two controls** derived by hand for the underflow
//!   fixtures (documented beside each assertion).
//! * **(D) optional `f64` dense reference** (normal equations + Gaussian
//!   elimination with partial pivoting). Used **only on demonstrably
//!   well-conditioned fixtures** to corroborate (B); it is never the authority
//!   for ill-conditioned or extreme inputs.
//!
//! # Exact fixture construction
//!
//! Exact inputs are constructed directly as [`BigRational`] values, or as
//! powers of two. They are never routed through `f64` before the exact
//! computation: converting through `f64` can already lose the intended exact
//! value. Conversion controls evaluate the **actual** `f32` value produced from
//! the computed `f64` coefficient (`fl32(w64)`), not an assumed result.
//!
//! # Mathematical reference (three distinct quantities)
//!
//! * `P64_exact` — the **exact real** evaluation `Σ_i x_i·w64_i + b64`, built
//!   from exact rational representations of the binary input/parameter values
//!   ([`pred64_exact`]). Never routed through a floating-point dot product.
//! * `P64_float` — the **rounded sequential `f64`** evaluation ([`predict_f64`]).
//!   Retained only where its divergence from `P64_exact` is the point.
//! * `P32` — the **actual deployed `f32`** evaluation ([`deployed_predict_f32`]).
//!
//! The prediction envelope is compared against `|P32 − P64_exact|`.
//!
//! # Limits
//!
//! [`BigRational`] does not overflow and has no fixed size (it is bounded only
//! by memory). The [`MAX_FIXTURE_BITS`] cap applies to the **exact fixture
//! inputs** (every feature and every target constructed as `BigRational`); it is
//! **not** applied to intermediate rational arithmetic, which is unbounded but
//! small for these fixtures. A fixture input exceeding the cap is **rejected**
//! ([`enforce_dataset`] / [`enforce_bits`]), never truncated or wrapped.

use num_bigint::BigInt;
use num_rational::BigRational;
use num_traits::{One, Signed, ToPrimitive, Zero};

/// Maximum bit-length permitted for a fixture numerator or denominator.
///
/// A fixture exceeding this cap is rejected (`panic`) rather than silently
/// truncated; `2^200` and `2^298` are far inside the cap.
const MAX_FIXTURE_BITS: u64 = 16_384;

/// `f32` unit roundoff `u32 = 2^-24` (not `f32::EPSILON = 2^-23`).
const U32_ROUNDOFF: f64 = f64::from_bits((1023u64 - 24) << 52);
/// `f64` unit roundoff `u64 = 2^-53`.
const U64_ROUNDOFF: f64 = f64::from_bits((1023u64 - 53) << 52);
/// Absolute operation-underflow contribution `theta = 2^-150`.
const THETA: f64 = f64::from_bits((1023u64 - 150) << 52);

// ---------------------------------------------------------------------------
// Exact rational helpers (reference B)
// ---------------------------------------------------------------------------

fn rat(n: i64) -> BigRational {
    BigRational::from_integer(BigInt::from(n))
}

/// Exact `2^exp` as a rational (`exp` may be negative).
fn pow2(exp: i32) -> BigRational {
    if exp >= 0 {
        BigRational::from_integer(BigInt::one() << (exp as usize))
    } else {
        BigRational::new(BigInt::one(), BigInt::one() << ((-exp) as usize))
    }
}

/// Rejects a fixture whose numerator or denominator exceeds [`MAX_FIXTURE_BITS`].
fn enforce_bits(value: &BigRational) {
    let numer_bits = value.numer().bits();
    let denom_bits = value.denom().bits();
    assert!(
        numer_bits <= MAX_FIXTURE_BITS && denom_bits <= MAX_FIXTURE_BITS,
        "fixture value exceeds the declared {MAX_FIXTURE_BITS}-bit cap \
         (numer {numer_bits} bits, denom {denom_bits} bits); rejected, not truncated"
    );
}

fn assert_rat_eq(actual: &BigRational, expected: &BigRational) {
    assert_eq!(actual, expected, "actual = {actual}, expected = {expected}");
}

/// Exact rank of a rational matrix via Gaussian elimination.
#[allow(clippy::needless_range_loop)]
fn exact_rank(matrix: &[Vec<BigRational>]) -> usize {
    enforce_matrix(matrix);
    let rows = matrix.len();
    if rows == 0 {
        return 0;
    }
    let cols = matrix[0].len();
    let mut a = matrix.to_vec();
    let mut rank = 0usize;
    for col in 0..cols {
        let pivot_row = (rank..rows).find(|&r| !a[r][col].is_zero());
        let Some(p) = pivot_row else { continue };
        a.swap(rank, p);
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

/// Exact least squares through the normal equations, for full-rank rational
/// inputs. Returns `None` when the normal matrix is singular (rank-deficient).
#[allow(clippy::needless_range_loop)]
fn exact_normal_equations_solve(
    a: &[Vec<BigRational>],
    y: &[BigRational],
) -> Option<Vec<BigRational>> {
    enforce_dataset(a, y);
    let n = a.len();
    let p = a[0].len();
    let mut ata = vec![vec![BigRational::zero(); p]; p];
    let mut aty = vec![BigRational::zero(); p];
    for i in 0..n {
        for j in 0..p {
            let prod = &a[i][j] * &y[i];
            aty[j] = &aty[j] + &prod;
            for k in 0..p {
                let prod = &a[i][j] * &a[i][k];
                ata[j][k] = &ata[j][k] + &prod;
            }
        }
    }
    exact_solve(ata, aty)
}

/// Exact Gaussian elimination with partial pivoting; `None` if singular.
#[allow(clippy::needless_range_loop)]
fn exact_solve(mut a: Vec<Vec<BigRational>>, mut b: Vec<BigRational>) -> Option<Vec<BigRational>> {
    let n = a.len();
    for col in 0..n {
        let pivot_row = (col..n).find(|&r| !a[r][col].is_zero())?;
        a.swap(col, pivot_row);
        b.swap(col, pivot_row);
        let pivot = a[col][col].clone();
        for r in (col + 1)..n {
            if !a[r][col].is_zero() {
                let factor = &a[r][col] / &pivot;
                for c in col..n {
                    let sub = &factor * &a[col][c];
                    a[r][c] = &a[r][c] - &sub;
                }
                let sub = &factor * &b[col];
                b[r] = &b[r] - &sub;
            }
        }
    }
    let mut x = vec![BigRational::zero(); n];
    for i in (0..n).rev() {
        let mut acc = b[i].clone();
        for j in (i + 1)..n {
            let sub = &a[i][j] * &x[j];
            acc = &acc - &sub;
        }
        x[i] = &acc / &a[i][i];
    }
    Some(x)
}

/// Exact residual sum of squares `sum_i (a_i . w - y_i)^2`.
#[allow(clippy::needless_range_loop)]
fn exact_rss(a: &[Vec<BigRational>], y: &[BigRational], w: &[BigRational]) -> BigRational {
    let mut rss = BigRational::zero();
    for i in 0..a.len() {
        let mut pred = BigRational::zero();
        for j in 0..w.len() {
            let prod = &a[i][j] * &w[j];
            pred = &pred + &prod;
        }
        let residual = &pred - &y[i];
        let sq = &residual * &residual;
        rss = &rss + &sq;
    }
    rss
}

// ---------------------------------------------------------------------------
// Floating references (A and D) and deployed serving arithmetic
// ---------------------------------------------------------------------------

/// (A) closed-form scalar OLS with intercept, in `f64`.
fn closed_form_scalar(x: &[f64], y: &[f64]) -> (f64, f64) {
    let n = x.len() as f64;
    let xbar = x.iter().sum::<f64>() / n;
    let ybar = y.iter().sum::<f64>() / n;
    let num: f64 = x
        .iter()
        .zip(y)
        .map(|(xi, yi)| (xi - xbar) * (yi - ybar))
        .sum();
    let den: f64 = x.iter().map(|xi| (xi - xbar).powi(2)).sum();
    let w = num / den;
    let b = ybar - w * xbar;
    (w, b)
}

/// (D) `f64` dense least squares via normal equations + partial pivoting.
/// Only used to corroborate (B) on well-conditioned fixtures.
#[allow(clippy::needless_range_loop)]
fn f64_dense_lstsq(a: &[Vec<f64>], y: &[f64]) -> Option<Vec<f64>> {
    let n = a.len();
    let p = a[0].len();
    let mut ata = vec![vec![0.0f64; p]; p];
    let mut aty = vec![0.0f64; p];
    for i in 0..n {
        for j in 0..p {
            aty[j] += a[i][j] * y[i];
            for k in 0..p {
                ata[j][k] += a[i][j] * a[i][k];
            }
        }
    }
    f64_solve(ata, aty)
}

#[allow(clippy::needless_range_loop)]
fn f64_solve(mut a: Vec<Vec<f64>>, mut b: Vec<f64>) -> Option<Vec<f64>> {
    let n = a.len();
    for col in 0..n {
        let mut pivot_row = col;
        for r in (col + 1)..n {
            if a[r][col].abs() > a[pivot_row][col].abs() {
                pivot_row = r;
            }
        }
        if a[pivot_row][col] == 0.0 {
            return None;
        }
        a.swap(col, pivot_row);
        b.swap(col, pivot_row);
        let pivot = a[col][col];
        for r in (col + 1)..n {
            let factor = a[r][col] / pivot;
            for c in col..n {
                a[r][c] -= factor * a[col][c];
            }
            b[r] -= factor * b[col];
        }
    }
    let mut x = vec![0.0f64; n];
    for i in (0..n).rev() {
        let mut acc = b[i];
        for j in (i + 1)..n {
            acc -= a[i][j] * x[j];
        }
        x[i] = acc / a[i][i];
    }
    Some(x)
}

/// The deployed serving arithmetic, re-implemented independently: sequential
/// `f32` left fold of `w_i * x_i` from the baseline `-0.0`, then the bias added
/// in `f32`.
/// This mirrors `Vector::dot(features) + bias` without using the library.
fn deployed_predict_f32(x: &[f32], w: &[f32], bias: f32) -> f32 {
    let mut acc = -0.0f32;
    for (xi, wi) in x.iter().zip(w.iter()) {
        acc += *xi * *wi;
    }
    acc + bias
}

#[test]
fn t0_deployed_scalar_reference_preserves_negative_zero() {
    let result = deployed_predict_f32(&[1.0], &[-0.0], -0.0);
    assert_eq!(result.to_bits(), (-0.0f32).to_bits());
}

/// `f64` prediction used as the "exact-side" comparison for error envelopes.
fn predict_f64(x: &[f64], w: &[f64], bias: f64) -> f64 {
    x.iter().zip(w).map(|(a, b)| a * b).sum::<f64>() + bias
}

fn as_f64(value: &BigRational) -> f64 {
    value.to_f64().expect("fixture fits in f64")
}

/// Exact rational representation of a finite `f64` value.
fn rat_of_f64(value: f64) -> BigRational {
    BigRational::from_float(value).expect("finite fixture")
}

/// Exact rational representation of a finite `f32` value.
fn rat_of_f32(value: f32) -> BigRational {
    BigRational::from_float(value as f64).expect("finite fixture")
}

/// `P64_exact = Σ_i x_i·w64_i + b64` evaluated exactly over the rationals.
fn pred64_exact(x: &[BigRational], w64: &[BigRational], b64: &BigRational) -> BigRational {
    let mut acc = b64.clone();
    for (xi, wi) in x.iter().zip(w64) {
        let product = xi * wi;
        acc = &acc + &product;
    }
    acc
}

/// `P64_exact` as `f64`, for comparison against the deployed `f32` result.
fn pred64_exact_f64(x: &[f64], w64: &[f64], b64: f64) -> f64 {
    let xr: Vec<BigRational> = x.iter().map(|v| rat_of_f64(*v)).collect();
    let wr: Vec<BigRational> = w64.iter().map(|v| rat_of_f64(*v)).collect();
    as_f64(&pred64_exact(&xr, &wr, &rat_of_f64(b64)))
}

/// Rejects an oversized exact input matrix (features).
fn enforce_matrix(a: &[Vec<BigRational>]) {
    for row in a {
        for value in row {
            enforce_bits(value);
        }
    }
}

/// Rejects oversized exact fixture inputs: every feature and every target.
fn enforce_dataset(a: &[Vec<BigRational>], y: &[BigRational]) {
    enforce_matrix(a);
    for value in y {
        enforce_bits(value);
    }
}

/// Per-row envelope components `(C_j, M_j, E_j)` in `f64` (mirrors production,
/// which evaluates these in `f64`; see the f64-accounting note in the plan).
fn envelope_terms(x64: &[f64], w64: &[f64], w32: &[f32], b64: f64, b32: f32) -> (f64, f64, f64) {
    let mut c_j = 0.0f64;
    let mut m_j = 0.0f64;
    for i in 0..x64.len() {
        c_j += x64[i].abs() * (w64[i] - w32[i] as f64).abs();
        m_j += x64[i].abs() * (w32[i] as f64).abs();
    }
    c_j += (b64 - b32 as f64).abs();
    (c_j, m_j, envelope_e(c_j, m_j, b32 as f64, x64.len()))
}

/// Independently computed objective quantities, all exact:
/// `(r64, r32, Σ Δ_j², Σ e64_j·Δ_j)` with `Δ_j = P32_j − P64_exact_j` and
/// `e64_j = P64_exact_j − y_j`.
fn objective_terms(
    x_rows: &[Vec<f32>],
    y: &[f64],
    w64: &[f64],
    b64: f64,
    w32: &[f32],
    b32: f32,
) -> (BigRational, BigRational, BigRational, BigRational) {
    let mut r64 = BigRational::zero();
    let mut r32 = BigRational::zero();
    let mut sum_delta_sq = BigRational::zero();
    let mut cross = BigRational::zero();
    for (row, &target) in x_rows.iter().zip(y) {
        let xr: Vec<BigRational> = row.iter().map(|v| rat_of_f32(*v)).collect();
        let wr: Vec<BigRational> = w64.iter().map(|v| rat_of_f64(*v)).collect();
        let p64 = pred64_exact(&xr, &wr, &rat_of_f64(b64));
        let p32 = rat_of_f32(deployed_predict_f32(row, w32, b32));
        let target = rat_of_f64(target);
        let e64 = &p64 - &target;
        let delta = &p32 - &p64;
        let e32 = &p32 - &target;
        r64 = &r64 + &(&e64 * &e64);
        r32 = &r32 + &(&e32 * &e32);
        sum_delta_sq = &sum_delta_sq + &(&delta * &delta);
        cross = &cross + &(&e64 * &delta);
    }
    (r64, r32, sum_delta_sq, cross)
}

// ---------------------------------------------------------------------------
// T2 helpers — envelope, domain guard, rank indicator
// ---------------------------------------------------------------------------

/// `E_j = C_j + gamma_k*(M_j + |b32|) + (2n+1)*theta` with `k = 3n + 3`.
fn envelope_e(c_j: f64, m_j: f64, b32: f64, n: usize) -> f64 {
    let k = (3 * n + 3) as f64;
    let gamma_k = k * U32_ROUNDOFF / (1.0 - k * U32_ROUNDOFF);
    c_j + gamma_k * (m_j + b32.abs()) + (2 * n + 1) as f64 * THETA
}

/// `roundoff_bound = 2*sqrt(r64)*sqrt(sum E_j^2) + sum E_j^2`.
fn roundoff_bound(r64: f64, e: &[f64]) -> f64 {
    let sum_sq: f64 = e.iter().map(|v| v * v).sum();
    2.0 * r64.max(0.0).sqrt() * sum_sq.sqrt() + sum_sq
}

/// Operation-count domain guard: `k*u32 < 1` (`k = 3n + 3`).
fn envelope_supported(n: usize) -> bool {
    ((3 * n + 3) as f64) * U32_ROUNDOFF < 1.0
}

/// Relative-pivot rank indicator: counts pivoted diagonals exceeding
/// `tau = c_rank * max(N,p) * u64 * |first pivot|`. A rank *indicator*,
/// not a condition number.
fn relative_pivot_rank(diag: &[f64], c_rank: f64, max_dim: usize) -> usize {
    let first = diag.first().copied().unwrap_or(0.0).abs();
    let tau = c_rank * max_dim as f64 * U64_ROUNDOFF * first;
    diag.iter().filter(|d| d.abs() > tau).count()
}

// ===========================================================================
// T0 — analytical / symbolic / rank / underflow fixtures
// ===========================================================================

#[test]
fn t0_exact_line_with_intercept() {
    // x = [0,1,2], y = [1,3,5]; intercept. Exact optimum: w = 2, b = 1, rss = 0.
    let a = vec![
        vec![rat(1), rat(0)],
        vec![rat(1), rat(1)],
        vec![rat(1), rat(2)],
    ];
    let y = vec![rat(1), rat(3), rat(5)];
    for row in &a {
        for value in row {
            enforce_bits(value);
        }
    }
    let w = exact_normal_equations_solve(&a, &y).expect("full rank");
    assert_rat_eq(&w[0], &rat(1)); // b
    assert_rat_eq(&w[1], &rat(2)); // w
    assert_rat_eq(&exact_rss(&a, &y, &w), &rat(0));

    // (A) closed form agrees.
    let (cw, cb) = closed_form_scalar(&[0.0, 1.0, 2.0], &[1.0, 3.0, 5.0]);
    assert!((cw - 2.0).abs() < 1e-12);
    assert!((cb - 1.0).abs() < 1e-12);

    // (D) well-conditioned dense reference agrees.
    let af = vec![vec![1.0, 0.0], vec![1.0, 1.0], vec![1.0, 2.0]];
    let yf = vec![1.0, 3.0, 5.0];
    let wf = f64_dense_lstsq(&af, &yf).expect("full rank");
    assert!((wf[0] - 1.0).abs() < 1e-12);
    assert!((wf[1] - 2.0).abs() < 1e-12);
}

#[test]
fn t0_nonzero_residual_with_intercept() {
    // x = [0,1,2], y = [1,2,2]; intercept. Exact: w = 1/2, b = 7/6, rss = 1/6.
    let a = vec![
        vec![rat(1), rat(0)],
        vec![rat(1), rat(1)],
        vec![rat(1), rat(2)],
    ];
    let y = vec![rat(1), rat(2), rat(2)];
    let w = exact_normal_equations_solve(&a, &y).expect("full rank");
    assert_rat_eq(&w[0], &(rat(7) / rat(6))); // b
    assert_rat_eq(&w[1], &(rat(1) / rat(2))); // w
    assert_rat_eq(&exact_rss(&a, &y, &w), &(rat(1) / rat(6)));

    let (cw, cb) = closed_form_scalar(&[0.0, 1.0, 2.0], &[1.0, 2.0, 2.0]);
    assert!((cw - 0.5).abs() < 1e-12);
    assert!((cb - 7.0 / 6.0).abs() < 1e-12);
}

#[test]
fn t0_conversion_control_with_intercept() {
    // x = [0,1,2], y = [0,1,1]; intercept. Exact: w = 1/2, b = 1/6.
    let a = vec![
        vec![rat(1), rat(0)],
        vec![rat(1), rat(1)],
        vec![rat(1), rat(2)],
    ];
    let y = vec![rat(0), rat(1), rat(1)];
    let w = exact_normal_equations_solve(&a, &y).expect("full rank");
    assert_rat_eq(&w[0], &(rat(1) / rat(6))); // b
    assert_rat_eq(&w[1], &(rat(1) / rat(2))); // w
}

#[test]
fn t0_conversion_underflow_no_intercept() {
    // x = 2^100, y = 2^-100; no intercept. Exact w64 = 2^-200; fl32(w64) = 0.
    let a = vec![vec![pow2(100)]];
    let y = vec![pow2(-100)];
    enforce_bits(&a[0][0]);
    let w = exact_normal_equations_solve(&a, &y).expect("full rank");
    assert_rat_eq(&w[0], &pow2(-200));

    let w64 = as_f64(&w[0]);
    let w32 = w64 as f32;
    assert_eq!(w32, 0.0, "fl32(2^-200) must underflow to zero");

    // Actual deployed prediction uses the converted f32 coefficient.
    let x32 = as_f64(&a[0][0]) as f32;
    let pred32 = deployed_predict_f32(&[x32], &[w32], 0.0);
    assert_eq!(pred32, 0.0);
    // The exact-side prediction is 2^-100.
    assert!((predict_f64(&[as_f64(&a[0][0])], &[w64], 0.0) - as_f64(&y[0])).abs() < 1e-30);
}

#[test]
fn t0_operation_underflow_no_intercept() {
    // x = 2^-100, w = 2^-50, X = [[x, x], [2x, 0]], y = [2^-149, 2^-149]; no
    // intercept. Exact optimum is w = [2^-50, 2^-50] (representable, C_j = 0).
    // Symbolic control: X^T X = 2^-200 * [[5,1],[1,1]]; X^T y = 2^-249 * [3,1];
    // solving gives w = [2^-50, 2^-50].
    let a = vec![
        vec![pow2(-100), pow2(-100)],
        vec![pow2(-99), BigRational::zero()],
    ];
    let y = vec![pow2(-149), pow2(-149)];
    for row in &a {
        for value in row {
            enforce_bits(value);
        }
    }
    let w = exact_normal_equations_solve(&a, &y).expect("full rank");
    assert_rat_eq(&w[0], &pow2(-50));
    assert_rat_eq(&w[1], &pow2(-50));

    // Deployed f32 predictions are [0, 2^-149]: row 1 loses the product to
    // underflow; row 2 keeps a representable subnormal.
    let x32 = [
        as_f64(&a[0][0]) as f32,
        as_f64(&a[0][1]) as f32,
        as_f64(&a[1][0]) as f32,
        as_f64(&a[1][1]) as f32,
    ];
    let w32 = [as_f64(&w[0]) as f32, as_f64(&w[1]) as f32];
    let pred0 = deployed_predict_f32(&x32[0..2], &w32, 0.0);
    let pred1 = deployed_predict_f32(&x32[2..4], &w32, 0.0);
    assert_eq!(pred0, 0.0, "row 1 product 2^-150 underflows to zero");
    // 2^-149 is the smallest positive f32 subnormal: bits 0x0000_0001.
    assert_eq!(pred1, f32::from_bits(1), "row 2 keeps subnormal 2^-149");

    // r32 = 2^-298; S = 2^-297; r32/S = 1/2.
    let y0 = as_f64(&y[0]);
    let y1 = as_f64(&y[1]);
    let r32 = (pred0 as f64 - y0).powi(2) + (pred1 as f64 - y1).powi(2);
    let s = y0 * y0 + y1 * y1;
    assert_eq!(r32, 2.0f64.powi(-298));
    assert_eq!(s, 2.0f64.powi(-297));
    assert_eq!(r32 / s, 0.5);
}

#[test]
fn t0_multifeature_full_rank() {
    // Two features + intercept, N = 4 > p = 3. Exact (B) vs dense (D).
    let a = vec![
        vec![rat(1), rat(1), rat(2)],
        vec![rat(1), rat(2), rat(3)],
        vec![rat(1), rat(3), rat(5)],
        vec![rat(1), rat(4), rat(7)],
    ];
    let y = vec![rat(1), rat(2), rat(4), rat(5)];
    assert_eq!(exact_rank(&a), 3);
    let w = exact_normal_equations_solve(&a, &y).expect("full rank");

    let af: Vec<Vec<f64>> = a
        .iter()
        .map(|row| row.iter().map(as_f64).collect())
        .collect();
    let yf: Vec<f64> = y.iter().map(as_f64).collect();
    let wf = f64_dense_lstsq(&af, &yf).expect("full rank");
    for (exact, dense) in w.iter().zip(&wf) {
        assert!((as_f64(exact) - dense).abs() < 1e-9);
    }
}

#[test]
fn t0_square_and_overdetermined() {
    // Square (N = p = 2, intercept) and overdetermined (N = 4).
    let square = vec![vec![rat(1), rat(3)], vec![rat(1), rat(5)]];
    let y_square = vec![rat(4), rat(6)];
    let w_square = exact_normal_equations_solve(&square, &y_square).expect("full rank");
    assert_rat_eq(&w_square[0], &rat(1)); // b
    assert_rat_eq(&w_square[1], &rat(1)); // w

    let over = vec![
        vec![rat(1), rat(0)],
        vec![rat(1), rat(1)],
        vec![rat(1), rat(2)],
        vec![rat(1), rat(3)],
    ];
    let y_over = vec![rat(0), rat(2), rat(4), rat(6)];
    let w_over = exact_normal_equations_solve(&over, &y_over).expect("full rank");
    assert_rat_eq(&w_over[0], &rat(0));
    assert_rat_eq(&w_over[1], &rat(2));
}

#[test]
fn t0_column_permutation_invariance() {
    // Swapping feature columns swaps the coefficients; predictions are equal.
    let a = vec![
        vec![rat(1), rat(2), rat(3)],
        vec![rat(1), rat(5), rat(1)],
        vec![rat(1), rat(7), rat(4)],
    ];
    let a_swapped = vec![
        vec![rat(1), rat(3), rat(2)],
        vec![rat(1), rat(1), rat(5)],
        vec![rat(1), rat(4), rat(7)],
    ];
    let y = vec![rat(6), rat(5), rat(9)];
    let w = exact_normal_equations_solve(&a, &y).expect("full rank");
    let w_swapped = exact_normal_equations_solve(&a_swapped, &y).expect("full rank");
    // Intercept unchanged; feature coefficients swap.
    assert_rat_eq(&w[0], &w_swapped[0]);
    assert_rat_eq(&w[1], &w_swapped[2]);
    assert_rat_eq(&w[2], &w_swapped[1]);
}

#[test]
fn t0_duplicate_columns_rank_deficient() {
    // Column 2 is 2 * column 1: exact rank 1, required 2.
    let a = vec![
        vec![rat(1), rat(2)],
        vec![rat(2), rat(4)],
        vec![rat(3), rat(6)],
    ];
    assert_eq!(exact_rank(&a), 1);
    let y = vec![rat(1), rat(2), rat(3)];
    assert!(
        exact_normal_equations_solve(&a, &y).is_none(),
        "rank-deficient system has no unique exact solution"
    );
}

#[test]
fn t0_constant_column_with_intercept_rank_deficient() {
    // A constant feature duplicates the intercept column: rank 1, required 2.
    let a = vec![
        vec![rat(1), rat(1)],
        vec![rat(1), rat(1)],
        vec![rat(1), rat(1)],
    ];
    assert_eq!(exact_rank(&a), 1);
}

#[test]
fn t0_near_dependent_exact_full_rank() {
    // Columns are nearly dependent but exactly independent over the rationals:
    // classification is full rank and a unique exact solution exists.
    let a = vec![
        vec![rat(1), rat(1)],
        vec![rat(2), rat(2_000_001) / rat(1_000_000)],
    ];
    assert_eq!(exact_rank(&a), 2);
    let y = vec![rat(1), rat(2)];
    assert!(exact_normal_equations_solve(&a, &y).is_some());
}

#[test]
fn t0_scale_sensitive_exact_full_rank() {
    // Extreme column scales: exact full rank despite ill-conditioning.
    let a = vec![
        vec![rat(1), pow2(-100)],
        vec![rat(1), rat(1)],
        vec![rat(1), pow2(100)],
    ];
    assert_eq!(exact_rank(&a), 2);
}

#[test]
#[should_panic(expected = "exceeds the declared")]
fn t0_fixture_size_limit_rejects_oversized_values() {
    // 2^20000 as a feature exceeds the cap; must be rejected rather than truncated.
    enforce_dataset(&[vec![pow2(20_000)]], &[rat(1)]);
}

#[test]
#[should_panic(expected = "exceeds the declared")]
fn t0_fixture_size_limit_rejects_oversized_target() {
    // 2^20000 as a target exceeds the cap; targets are covered by the invariant.
    enforce_dataset(&[vec![rat(1)]], &[pow2(20_000)]);
}

#[test]
fn t0_fixture_size_invariant_accepts_in_range_inputs() {
    // Both feature and target at the supported extreme are accepted.
    enforce_dataset(&[vec![pow2(100)]], &[pow2(-100)]);
}

#[test]
fn t0_reference_exact_vs_rounded_f64() {
    // x = [1,1,1], w64 = [2^54, 1, -2^54], b64 = 0.
    // Exact real result: 2^54 + 1 - 2^54 = 1.
    // Sequential rounded f64: 0 + 2^54 = 2^54; 2^54 + 1 rounds to 2^54;
    // then + (-2^54) = 0. The exact and rounded references genuinely differ.
    let x = vec![rat(1), rat(1), rat(1)];
    let w = vec![pow2(54), rat(1), -pow2(54)];
    let b = rat(0);
    let p64_exact = pred64_exact(&x, &w, &b);
    assert_rat_eq(&p64_exact, &rat(1));

    let xf = [1.0f64, 1.0, 1.0];
    let wf = [(1u64 << 54) as f64, 1.0, -((1u64 << 54) as f64)];
    let p64_float = predict_f64(&xf, &wf, 0.0);
    assert_eq!(
        p64_float, 0.0,
        "sequential rounded f64 loses the +1 in 2^54 + 1"
    );
    assert_ne!(as_f64(&p64_exact), p64_float);
    // The exact-reference helper agrees with the hand result.
    assert_eq!(pred64_exact_f64(&xf, &wf, 0.0), 1.0);
}

// ===========================================================================
// T2 — executable controls for the numerical assumptions
// ===========================================================================

#[test]
fn t2_conversion_term_is_nonzero_under_conversion_underflow() {
    // x = 2^100, w64 = 2^-200, w32 = 0. C_j = |x| * |w64 - w32| = 2^-100 != 0.
    let x = 2.0f64.powi(100);
    let w64 = 2.0f64.powi(-200);
    let w32 = w64 as f32;
    let c_j = x.abs() * (w64 - w32 as f64).abs();
    assert!(c_j > 0.0);
    assert!((c_j - 2.0f64.powi(-100)).abs() < 1e-60);
    let m_j = x.abs() * (w32 as f64).abs();
    assert_eq!(m_j, 0.0);
    let e_j = envelope_e(c_j, m_j, 0.0, 1);
    // Prediction error |0 - 2^-100| = 2^-100 must be bounded by E_j.
    assert!(2.0f64.powi(-100) <= e_j);
}

#[test]
fn t2_check1_and_check2_on_underflow_controls() {
    // --- Conversion underflow -----------------------------------------------------------------
    // x = 2^100, y = 2^-100, no intercept; w64 = 2^-200, w32 = 0.
    let x = [2.0f64.powi(100)];
    let y = [2.0f64.powi(-100)];
    let w64 = [2.0f64.powi(-200)];
    let w32 = [w64[0] as f32];
    let c = x
        .iter()
        .zip(&w64)
        .zip(&w32)
        .map(|((xi, w6), w3)| xi.abs() * (w6 - *w3 as f64).abs())
        .sum::<f64>();
    let m = x
        .iter()
        .zip(&w32)
        .map(|(xi, w3)| xi.abs() * (*w3 as f64).abs())
        .sum::<f64>();
    let e_conv = [envelope_e(c, m, 0.0, 1)];
    // P64_exact (exact-real), not P64_float: the envelope is defined against the
    // exact real evaluation of the computed f64 coefficients.
    let pred64_conv = pred64_exact_f64(&x, &w64, 0.0);
    let pred32_conv = deployed_predict_f32(&[x[0] as f32], &w32, 0.0) as f64;
    assert!(pred64_conv == y[0]);
    let r64_conv = (pred64_conv - y[0]).powi(2);
    let r32_conv = (pred32_conv - y[0]).powi(2);
    let bound_conv = roundoff_bound(r64_conv, &e_conv);
    assert!(
        (r32_conv - r64_conv).abs() <= bound_conv,
        "Check 1 must hold with c_round = 1 on conversion underflow"
    );
    // Check 2: S = 2^-200, r32 - r64 = 2^-200, so it fails whenever g_abs < 1.
    let s_conv = y[0] * y[0];
    assert_eq!(r32_conv - r64_conv, 2.0f64.powi(-200));
    let g_abs = 1e-6;
    assert!(r32_conv - r64_conv > g_abs * s_conv);

    // --- Operation underflow ------------------------------------------------------------------
    // x = 2^-100, w = 2^-50, X = [[x,x],[2x,0]], y = [2^-149, 2^-149].
    let w64 = [2.0f64.powi(-50), 2.0f64.powi(-50)];
    let w32 = [w64[0] as f32, w64[1] as f32];
    assert_eq!(
        w64[0] as f32,
        2.0f32.powi(-50),
        "coefficient is representable"
    );
    let x0 = [2.0f64.powi(-100), 2.0f64.powi(-100)];
    let x1 = [2.0f64.powi(-99), 0.0];
    let y_op = [2.0f64.powi(-149), 2.0f64.powi(-149)];
    // C_j = 0 (coefficients representable exactly).
    let c_fun = |xj: &[f64]| -> f64 {
        xj.iter()
            .zip(&w64)
            .zip(&w32)
            .map(|((xi, w6), w3)| xi.abs() * (w6 - *w3 as f64).abs())
            .sum()
    };
    let m_fun = |xj: &[f64]| -> f64 {
        xj.iter()
            .zip(&w32)
            .map(|(xi, w3)| xi.abs() * (*w3 as f64).abs())
            .sum()
    };
    let e_op = [
        envelope_e(c_fun(&x0), m_fun(&x0), 0.0, 2),
        envelope_e(c_fun(&x1), m_fun(&x1), 0.0, 2),
    ];
    let pred64 = [
        pred64_exact_f64(&x0, &w64, 0.0),
        pred64_exact_f64(&x1, &w64, 0.0),
    ];
    let pred32 = [
        deployed_predict_f32(&[x0[0] as f32, x0[1] as f32], &w32, 0.0) as f64,
        deployed_predict_f32(&[x1[0] as f32, x1[1] as f32], &w32, 0.0) as f64,
    ];
    assert_eq!(pred32[0], 0.0);
    assert_eq!(pred32[1], 2.0f64.powi(-149));
    for ((p32, p64), e) in pred32.iter().zip(&pred64).zip(&e_op) {
        assert!((p32 - p64).abs() <= *e, "prediction error bounded by E_j");
    }
    let r64_op = (pred64[0] - y_op[0]).powi(2) + (pred64[1] - y_op[1]).powi(2);
    let r32_op = (pred32[0] - y_op[0]).powi(2) + (pred32[1] - y_op[1]).powi(2);
    assert_eq!(r64_op, 0.0);
    assert_eq!(r32_op, 2.0f64.powi(-298));
    let bound_op = roundoff_bound(r64_op, &e_op);
    assert!(
        (r32_op - r64_op).abs() <= bound_op,
        "Check 1 must hold with c_round = 1 on operation underflow"
    );
    // Check 2: r32/S = 1/2, rejected whenever g_abs < 1/2.
    let s_op = y_op[0] * y_op[0] + y_op[1] * y_op[1];
    assert_eq!(r32_op / s_op, 0.5);
    assert!(r32_op - r64_op > g_abs * s_op);
}

#[test]
fn t2_domain_guard_boundary() {
    // Supported iff (3n + 3) * 2^-24 < 1, i.e. n <= 5_592_404.
    assert!(envelope_supported(0));
    assert!(envelope_supported(5_592_404));
    assert!(!envelope_supported(5_592_405));
    // k * u32 at the first unsupported n is >= 1.
    let k = (3 * 5_592_405 + 3) as f64;
    assert!(k * U32_ROUNDOFF >= 1.0);
}

#[test]
fn t2_rank_threshold_is_a_relative_indicator() {
    // A well-scaled diagonal is full rank; a pivot that collapses below the
    // relative threshold is reported as a rank deficiency. This is an
    // indicator, not a condition number.
    let max_dim = 3;
    assert_eq!(relative_pivot_rank(&[4.0, 2.0, 1.0], 1.0, max_dim), 3);
    // tau = 1 * 3 * 2^-53 * 4 ~= 1.33e-15; 1e-16 is below it.
    assert_eq!(relative_pivot_rank(&[4.0, 2.0, 1e-16], 1.0, max_dim), 2);
    // A larger c_rank is strictly more conservative.
    assert_eq!(relative_pivot_rank(&[4.0, 1e-15, 1e-16], 1e3, max_dim), 1);
}

#[test]
fn t2_exact_rank_matches_control_expectations() {
    // The T2 rank classification uses the independent exact rank test (T0),
    // not any Seqvex adapter.
    let duplicate = vec![vec![rat(1), rat(2)], vec![rat(2), rat(4)]];
    let constant = vec![vec![rat(1), rat(1)], vec![rat(1), rat(1)]];
    let full = vec![vec![rat(1), rat(1)], vec![rat(1), rat(2)]];
    assert_eq!(exact_rank(&duplicate), 1);
    assert_eq!(exact_rank(&constant), 1);
    assert_eq!(exact_rank(&full), 2);
}

#[test]
fn t2_prediction_envelope_against_exact_real_reference() {
    // The envelope is compared against |P32 - P64_exact|, where P64_exact is
    // the exact-real evaluation of the computed f64 coefficients.

    // Conversion underflow.
    let x = [2.0f64.powi(100)];
    let w64 = [2.0f64.powi(-200)];
    let w32 = [0.0f32];
    let (_, _, e) = envelope_terms(&x, &w64, &w32, 0.0, 0.0);
    let p64 = pred64_exact_f64(&x, &w64, 0.0);
    let p32 = deployed_predict_f32(&[x[0] as f32], &w32, 0.0) as f64;
    assert_eq!(p64, 2.0f64.powi(-100));
    assert!((p32 - p64).abs() <= e);

    // Operation underflow (both rows).
    let w64 = [2.0f64.powi(-50), 2.0f64.powi(-50)];
    let w32 = [w64[0] as f32, w64[1] as f32];
    let rows = [
        [2.0f64.powi(-100), 2.0f64.powi(-100)],
        [2.0f64.powi(-99), 0.0],
    ];
    for row in &rows {
        let (_, _, e) = envelope_terms(row, &w64, &w32, 0.0, 0.0);
        let p64 = pred64_exact_f64(row, &w64, 0.0);
        let p32 = deployed_predict_f32(&[row[0] as f32, row[1] as f32], &w32, 0.0) as f64;
        assert!((p32 - p64).abs() <= e, "|P32 - P64_exact| <= E_j");
    }
}

/// Checks the per-row prediction bound and returns the per-row `E_j` values.
fn check_prediction_rows(
    x_rows: &[Vec<f32>],
    w64: &[f64],
    b64: f64,
    w32: &[f32],
    b32: f32,
) -> Vec<f64> {
    let mut e_vec = Vec::new();
    for row in x_rows {
        let xf: Vec<f64> = row.iter().map(|v| *v as f64).collect();
        let (_, _, e) = envelope_terms(&xf, w64, w32, b64, b32);
        let p64 = pred64_exact_f64(&xf, w64, b64);
        let p32 = deployed_predict_f32(row, w32, b32) as f64;
        assert!((p32 - p64).abs() <= e, "|P32 - P64_exact| <= E_j");
        e_vec.push(e);
    }
    e_vec
}

#[test]
fn t2_objective_bound_with_positive_residual() {
    // Nonzero-residual OLS control: x=[0,1,2], y=[1,2,2], computed coefficients
    // b = f64(7/6), w = 1/2. r64 = 1/6 > 0. At the optimum the residuals are
    // orthogonal to the linear coefficient-conversion change, so the cross term
    // is (near-)zero even though r64 > 0 -- the case the reviewer flagged.
    let x_rows = vec![vec![0.0f32], vec![1.0], vec![2.0]];
    let y = [1.0f64, 2.0, 2.0];
    let w64 = [0.5f64];
    let b64 = 7.0f64 / 6.0;
    let w32 = [0.5f32];
    let b32 = (7.0f64 / 6.0) as f32;

    let (r64, r32, sum_delta_sq, cross) = objective_terms(&x_rows, &y, &w64, b64, &w32, b32);
    assert!(r64 > BigRational::zero(), "positive residual");
    // Exact identity: r32 - r64 = ΣΔ² + 2 Σ e64·Δ.
    let lhs = &r32 - &r64;
    let rhs = &sum_delta_sq + &(&cross * &rat(2));
    assert_eq!(lhs, rhs);

    let e_vec = check_prediction_rows(&x_rows, &w64, b64, &w32, b32);
    let bound = roundoff_bound(as_f64(&r64), &e_vec);
    assert!(
        (as_f64(&r32) - as_f64(&r64)).abs() <= bound,
        "objective bound holds with r64 > 0"
    );
    println!(
        "[positive-residual] r64={} r32={} sum_delta_sq={} cross={}",
        as_f64(&r64),
        as_f64(&r32),
        as_f64(&sum_delta_sq),
        as_f64(&cross)
    );
}

#[test]
fn t2_objective_bound_nonzero_cross_term() {
    // Slightly approximate f64 coefficients (b = 1.0, w = 0.7, not the optimum).
    // In decimal-real arithmetic the residuals are 0, -0.3, 0.4, so r64 = 1/4.
    // The exact rational value below is for the stored binary-f64 coefficient;
    // it is close to, but not identical to, that decimal-real quantity. The
    // cross term remains nonzero because residuals are not orthogonal to the
    // coefficient-conversion change.
    let x_rows = vec![vec![0.0f32], vec![1.0], vec![2.0]];
    let y = [1.0f64, 2.0, 2.0];
    let w64 = [0.7f64];
    let b64 = 1.0f64;
    let w32 = [0.7f32];
    let b32 = 1.0f32;

    let (r64, r32, sum_delta_sq, cross) = objective_terms(&x_rows, &y, &w64, b64, &w32, b32);
    assert!(r64 > BigRational::zero(), "positive residual");
    assert!(!cross.is_zero(), "cross term Σ e64_j·Δ_j must be nonzero");
    let lhs = &r32 - &r64;
    let rhs = &sum_delta_sq + &(&cross * &rat(2));
    assert_eq!(lhs, rhs);

    let e_vec = check_prediction_rows(&x_rows, &w64, b64, &w32, b32);
    let bound = roundoff_bound(as_f64(&r64), &e_vec);
    assert!(
        (as_f64(&r32) - as_f64(&r64)).abs() <= bound,
        "objective bound holds with a nonzero cross term"
    );
    println!(
        "[nonzero-cross] r64={} r32={} sum_delta_sq={} cross={}",
        as_f64(&r64),
        as_f64(&r32),
        as_f64(&sum_delta_sq),
        as_f64(&cross)
    );
}
