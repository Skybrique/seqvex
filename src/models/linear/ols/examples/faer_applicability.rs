//! Untimed upstream high-level QR diagnostic for the reviewed applicability
//! assessment. This is evidence capture, not a public A/C regression test.
//!
//! Exit status:
//! 0: finite result exactly matching the independent zero expectation
//! 1: non-finite result (upstream #265-like reproduction)
//! 2: finite but non-zero result
//! 3: setup/API error

use std::process::ExitCode;

use faer::linalg::solvers::SolveLstsq;
use faer::{Mat, mat};

const LARGE: u64 = 0x4341_c379_37e0_8000;
const NEGATIVE_MID: u64 = 0xc30c_6bf5_2634_0000;

fn main() -> ExitCode {
    let a = mat![
        [f64::from_bits(LARGE), f64::from_bits(NEGATIVE_MID), 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ];
    let b: Mat<f64> = mat![[0.0], [0.0], [0.0]];
    let x = a.as_ref().qr().solve_lstsq(b.as_ref());
    let values = [x[(0, 0)], x[(1, 0)], x[(2, 0)]];

    println!("diagnostic: faer 0.24.4 high-level unpivoted qr().solve_lstsq");
    println!("input f64 bits: {LARGE:016x} {NEGATIVE_MID:016x}");
    println!("independent expectation: [0, 0, 0]");
    println!("actual bits: {:016x?}", values.map(f64::to_bits));
    println!("actual values: {values:?}");

    if values.iter().any(|value| !value.is_finite()) {
        println!("classification: upstream-path_nonfinite");
        ExitCode::from(1)
    } else if values.iter().any(|value| *value != 0.0) {
        println!("classification: finite_nonzero_mismatch");
        ExitCode::from(2)
    } else {
        println!("classification: finite_expected_zero");
        ExitCode::SUCCESS
    }
}
