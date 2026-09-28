//! Streams observations through a GRU and prints the persistent hidden state.
//!
//! Run with `cargo run --example streaming_gru`.

use seqvex::execution::streaming::StreamingExecutor;
use seqvex::foundation::numerical::Vector;
use seqvex::foundation::observation::{Observation, SequenceNumber};
use seqvex::models::recurrent::gru::{Gru, GruParameters};

fn main() {
    let input_dim = 3;
    let hidden_dim = 4;
    let model = Gru::new(
        input_dim,
        hidden_dim,
        GruParameters::deterministic(input_dim, hidden_dim),
    )
    .unwrap();
    let mut executor = StreamingExecutor::new(&model, Vector::zeros(hidden_dim));

    println!("initial hidden state: {:?}", executor.state().as_slice());

    let sequence = [
        [0.1, 0.2, 0.3],
        [0.4, -0.5, 0.6],
        [-0.7, 0.8, -0.9],
        [0.2, 0.2, 0.2],
    ];
    for (step, input) in sequence.into_iter().enumerate() {
        let observation = Observation::new(Vector::from_slice(&input))
            .with_sequence(SequenceNumber::new(step as u64));
        let hidden = executor.process_one(&observation).unwrap();
        println!(
            "step {step}: input={input:?} -> hidden={:?}",
            hidden.as_slice()
        );
    }
}
