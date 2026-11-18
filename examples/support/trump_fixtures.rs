//! Shared test-only Trump CSV fixture construction and serialization.
//!
//! Included only by the two real-data example test modules. Learner workflows,
//! assertions and temporary-file wrappers remain in those examples.

use crate::fixture_rng::Lcg64;

const HEADER: &str =
    "ordinal_date,five_thirty_eight,gallup,ipsos,morning_consult,rasmussen,you_gov";

/// Deterministic fixture row in original units.
pub(crate) struct Fixture {
    pub(crate) date: i64,
    pub(crate) features: [f64; 5],
    pub(crate) target: f64,
}

pub(crate) fn fixtures(count: usize, seed: u64) -> Vec<Fixture> {
    let mut rng = Lcg64::new(seed);
    (0..count)
        .map(|i| {
            let mut features = [0.0f64; 5];
            for value in &mut features {
                *value = 30.0 + 20.0 * rng.next_unit();
            }
            let target =
                1.0 + 2.0 * features[0] - 1.5 * features[1] + 0.5 * features[2] + features[3]
                    - features[4];
            Fixture {
                date: 736_389 + i as i64,
                features,
                target,
            }
        })
        .collect()
}

pub(crate) fn to_csv(rows: &[Fixture]) -> String {
    let mut content = String::from(HEADER);
    content.push('\n');
    for row in rows {
        content.push_str(&format!(
            "{},{},{},{},{},{},{}\n",
            row.date,
            row.target,
            row.features[0],
            row.features[1],
            row.features[2],
            row.features[3],
            row.features[4]
        ));
    }
    content
}
