//! # Quasar - Blazingly Fast Autograd Engine 🌟
//! 
//! Quasar is a high-performance automatic differentiation library written in Rust
//! with hand-optimized assembly code for critical operations.
//! 
//! ## Features
//! - Zero-cost abstractions
//! - SIMD optimizations (AVX2/AVX512)
//! - Automatic differentiation
//! - Memory-efficient tensor operations
//! - Multi-threaded computation
//! 
//! ## Example
//! ```rust
//! use quasar::prelude::*;
//! 
//! # fn main() -> Result<()> {
//! let x = Tensor::new(vec![2.0, 3.0], Shape::from(&[2]))?.requires_grad(true);
//! // let y = &x * &x + 1.0;  // TODO: implement scalar ops
//! // y.backward();           // TODO: implement full backward
//! println!("Tensor: {:?}", x);
//! # Ok(())
//! # }
//! ```

// Core modules
pub mod core;
pub mod memory;
pub mod ops;
pub mod autograd;
pub mod backend;
pub mod utils;
pub mod error;

// Convenience module
pub mod prelude;

// Re-export основных типов
pub use core::Tensor;
pub use error::{QuasarError, Result};
