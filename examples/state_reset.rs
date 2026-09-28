//! Makes the sequence boundary explicit: `x1, x2, x3`, `reset`, `y1`.
//!
//! Run with `cargo run --example state_reset`.

use seqvex::execution::streaming::StreamingExecutor;
use seqvex::foundation::numerical::Vector;
use seqvex::foundation::observation::Observation;
use seqvex::models::recurrent::gru::{Gru, GruParameters};

fn main() {
    let (input_dim, hidden_dim) = (2, 3);
    let model = Gru::new(
        input_dim,
        hidden_dim,
        GruParameters::deterministic(input_dim, hidden_dim),
    )
    .unwrap();

    let mut executor = StreamingExecutor::new(&model, Vector::zeros(hidden_dim));
    for input in [[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]] {
        let hidden = executor
            .process_one(&Observation::new(Vector::from_slice(&input)))
            .unwrap();
        println!("x={input:?} -> hidden={:?}", hidden.as_slice());
    }

    println!("-- reset --");
    executor.reset(Vector::zeros(hidden_dim));
    println!("after reset hidden={:?}", executor.state().as_slice());

    let y = [0.25, -0.75];
    let after_reset = executor
        .process_one(&Observation::new(Vector::from_slice(&y)))
        .unwrap()
        .clone();

    let fresh_model = Gru::new(
        input_dim,
        hidden_dim,
        GruParameters::deterministic(input_dim, hidden_dim),
    )
    .unwrap();
    let mut fresh = StreamingExecutor::new(&fresh_model, Vector::zeros(hidden_dim));
    let from_fresh = fresh
        .process_one(&Observation::new(Vector::from_slice(&y)))
        .unwrap()
        .clone();

    println!("y={y:?} -> hidden={:?}", after_reset.as_slice());
    println!("fresh y={y:?} -> hidden={:?}", from_fresh.as_slice());
    assert_eq!(after_reset, from_fresh);
    println!("reset made the next observation start a new sequence");
}
