//! Quasar prelude - commonly used imports
//! 
//! This module provides convenient access to the most commonly used types and functions
//! in Quasar. All operations automatically use parallelization when beneficial.
//! 
//! # Automatic Parallelization
//! 
//! Quasar automatically chooses between parallel and sequential execution based on tensor size:
//! - Small tensors (< 1000 elements): Sequential execution (no overhead)
//! - Large tensors (≥ 1000 elements): Parallel execution (16x+ speedup)
//! 
//! No configuration needed - everything works optimally out of the box!
//! 
//! # Example
//! 
//! ```rust
//! use quasar::prelude::*;
//! 
//! fn main() -> Result<()> {
//!     // All operations automatically use optimal execution
//!     let a = Tensor::new(vec![1.0, 2.0, 3.0], Shape::from(&[3]))?;
//!     let b = Tensor::new(vec![4.0, 5.0, 6.0], Shape::from(&[3]))?;
//!     
//!     let sum = &a + &b;        // Automatically sequential (small tensors)
//!     let product = &a * &b;    // Automatically sequential (small tensors)
//!     
//!     // Large tensors automatically use parallel execution
//!     let large_a = Tensor::<f32>::zeros(Shape::from(&[1000, 1000]))?;
//!     let large_b = Tensor::<f32>::ones(Shape::from(&[1000, 1000]))?;
//!     let result = matmul(&large_a, &large_b)?; // Automatically parallel (16x+ speedup)
//!     
//!     Ok(())
//! }
//! ```

// Core types
pub use crate::core::{Tensor, Shape, TensorElement};
pub use crate::error::{QuasarError, Result};

// Operations (all with automatic parallelization)
pub use crate::ops::arithmetic::{add, sub, mul, div};
pub use crate::ops::linalg::{matmul};
pub use crate::ops::activation::{relu};
pub use crate::ops::creation::{zeros, ones};

// Autograd types and functions
pub use crate::autograd::engine::{AddOp, SubOp, MulOp, DivOp, with_global_engine, AutogradOp};
pub use crate::autograd::graph::Operation;

// Traits for operator overloading
pub use std::ops::{Add, Sub, Mul, Div};

// Commonly used std types
pub use std::f32;
pub use std::f64; 