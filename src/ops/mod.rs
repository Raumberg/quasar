//! Tensor operations

pub mod arithmetic;
pub mod linalg;
pub mod activation;
pub mod reduction;
pub mod creation;

// Re-export commonly used functions
pub use arithmetic::{add, sub, mul, div};
pub use linalg::{matmul, dot};
pub use activation::{relu, sigmoid, tanh};
pub use reduction::*;
pub use creation::*; 