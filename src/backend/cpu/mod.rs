//! CPU computational backend

pub mod simd;
pub mod blas;
pub mod kernels;

use crate::core::TensorElement;
use crate::error::Result;

/// CPU backend for tensor operations
pub struct CpuBackend;

impl CpuBackend {
    /// Create new CPU backend
    pub fn new() -> Self {
        Self
    }
    
    /// Check if SIMD is available
    pub fn has_simd() -> bool {
        simd::has_avx2()
    }
}

impl Default for CpuBackend {
    fn default() -> Self {
        Self::new()
    }
} 