//! Convenient re-exports for common usage

// Core types
pub use crate::core::{Tensor, TensorElement, DType, Shape};
pub use crate::error::{QuasarError, Result};

// Operations
pub use crate::ops::*;

// Autograd
pub use crate::autograd::{Variable, AutogradFunction, AutogradOp, AddOp, MulOp};
pub use crate::autograd::engine::with_global_engine;

// Memory
pub use crate::memory::AlignedVec;

// Backend
pub use crate::backend::CpuBackend;

// New operations
pub use crate::ops::activation::{ReLUOp, relu};
pub use crate::ops::linalg::{MatMulOp, matmul};

// Parallel operations
pub use crate::ops::parallel_ops::{par_add, par_mul, par_matmul, par_relu, par_sum, par_statistics, TensorStats};
pub use crate::autograd::parallel::{ParallelConfig, ParallelTensorOps, init_parallel_autograd, with_local_engine}; 