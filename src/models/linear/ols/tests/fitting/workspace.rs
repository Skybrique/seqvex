use super::support::features;
use seqvex::models::linear::ols::{FitConfig, FitError, fit};

#[test]
fn workspace_budget_matches_independently_enumerated_total() {
    use faer::dyn_stack::StackReq;
    use faer::linalg::qr::col_pivoting::factor::{qr_in_place_scratch, recommended_block_size};
    use faer::linalg::qr::col_pivoting::solve::solve_lstsq_in_place_scratch;
    use faer::{Par, Spec};

    let x = features(&[&[0.0], &[1.0], &[2.0]]);
    let y = [1.0f32, 3.0, 5.0];
    let base = fit(&x, &y, &[0, 1, 2], &FitConfig::default()).expect("fit");
    let requested = base.diagnostics().requested_workspace_bytes;

    // Independently enumerate every owned backing buffer for the fixture's
    // actual dimensions (N=3, d=1, p=2, intercept on). The faer block size and
    // combined scratch requirement come from the pinned public APIs, not from
    // Seqvex's budget helper or `FitDiagnostics`.
    let rows = 3usize;
    let feature_count = 1usize;
    let p = feature_count + 1; // intercept column
    let size = rows.min(p);
    let block = recommended_block_size::<f64>(rows, p);
    let scratch = StackReq::any_of(&[
        qr_in_place_scratch::<usize, f64>(rows, p, block, Par::Seq, Spec::default()),
        solve_lstsq_in_place_scratch::<usize, f64>(rows, p, block, 1, Par::Seq),
    ])
    .unaligned_bytes_required();

    let f64_bytes = std::mem::size_of::<f64>();
    let f32_bytes = std::mem::size_of::<f32>();
    let usize_bytes = std::mem::size_of::<usize>();
    let a = rows * p * f64_bytes;
    let q = block * size * f64_bytes;
    let rhs = rows * f64_bytes;
    let perm = 2 * p * usize_bytes;
    let solution = p * f64_bytes;
    let converted = p * f32_bytes;
    let retained = p * f32_bytes;
    let diagnostics_bytes = p * f32_bytes;
    let uniq = x.len().div_ceil(8);

    let mut full = 0usize;
    for term in [
        a,
        q,
        rhs,
        perm,
        solution,
        converted,
        retained,
        diagnostics_bytes,
        uniq,
        scratch,
    ] {
        full = full.checked_add(term).expect("no overflow");
    }
    assert_eq!(
        requested, full,
        "requested {requested} != independently enumerated {full}"
    );

    // The pre-correction formula omitted the f64 solution plus the retained and
    // transient diagnostics coefficient vectors, keeping only a single 4p
    // converted buffer: exactly 16p bytes short. The equality above would fail
    // under that accounting, which would report `full - 16p`.
    let pre_correction = a + q + rhs + perm + converted + uniq + scratch;
    assert_eq!(full.checked_sub(pre_correction).expect("ordered"), 16 * p);

    // Exact budget accepts; a budget one byte below the complete total rejects.
    let exact = FitConfig {
        max_workspace_bytes: full,
        ..FitConfig::default()
    };
    assert!(fit(&x, &y, &[0, 1, 2], &exact).is_ok());
    let below = FitConfig {
        max_workspace_bytes: full - 1,
        ..FitConfig::default()
    };
    match fit(&x, &y, &[0, 1, 2], &below) {
        Err(FitError::Capacity {
            requested: got,
            limit,
        }) => {
            assert_eq!(got, full);
            assert_eq!(limit, full - 1);
        }
        other => panic!("expected Capacity, got {other:?}"),
    }
}
