//! Recurrent model namespace.

pub mod gru;

pub use gru::{Gru, GruBatchError, GruError, GruExecutor, GruParameters};
