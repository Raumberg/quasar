//! Core tensor system components

pub mod tensor;
pub mod dtype;
pub mod shape;

// Re-exports
pub use tensor::Tensor;
pub use dtype::{TensorElement, DType};
pub use shape::{Shape, Stride}; 