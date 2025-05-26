//! Convenient re-exports for common usage

// Core types
pub use crate::core::{Tensor, TensorElement, DType, Shape};
pub use crate::error::{QuasarError, Result};

// Operations
pub use crate::ops::*;

// Autograd
pub use crate::autograd::{Variable, AutogradFunction, AutogradOp, AddOp, MulOp};

// Memory
pub use crate::memory::AlignedVec;

// Backend
pub use crate::backend::CpuBackend; 