//! Error handling for Quasar autograd engine

use thiserror::Error;
use std::fmt;

/// Result type alias for Quasar operations
pub type Result<T> = std::result::Result<T, QuasarError>;

/// Errors that can occur in Quasar operations
#[derive(Debug, Clone, PartialEq)]
pub enum QuasarError {
    ShapeMismatch {
        expected: Vec<usize>,
        actual: Vec<usize>,
    },

    InvalidDimension { dim: usize, ndim: usize },

    IndexOutOfBounds { index: usize, size: usize },

    MemoryError { message: String },

    SimdNotSupported,

    GradientError { message: String },

    InvalidOperation { message: String },

    NumericalError { message: String },

    /// Division by zero
    DivisionByZero,
}

impl QuasarError {
    /// Create a shape mismatch error
    pub fn shape_mismatch(expected: &[usize], actual: &[usize]) -> Self {
        Self::ShapeMismatch {
            expected: expected.to_vec(),
            actual: actual.to_vec(),
        }
    }

    /// Create an invalid dimension error
    pub fn invalid_dimension(dim: usize, ndim: usize) -> Self {
        Self::InvalidDimension { dim, ndim }
    }

    /// Create an index out of bounds error
    pub fn index_out_of_bounds(index: usize, size: usize) -> Self {
        Self::IndexOutOfBounds { index, size }
    }

    /// Create a memory error
    pub fn memory_error(message: impl Into<String>) -> Self {
        Self::MemoryError {
            message: message.into(),
        }
    }

    /// Create a gradient error
    pub fn gradient_error(message: impl Into<String>) -> Self {
        Self::GradientError {
            message: message.into(),
        }
    }

    /// Create an invalid operation error
    pub fn invalid_operation(message: impl Into<String>) -> Self {
        Self::InvalidOperation {
            message: message.into(),
        }
    }

    /// Create a numerical error
    pub fn numerical_error(message: impl Into<String>) -> Self {
        Self::NumericalError {
            message: message.into(),
        }
    }

    /// Create division by zero error
    pub fn division_by_zero() -> Self {
        Self::DivisionByZero
    }
}

impl fmt::Display for QuasarError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ShapeMismatch { expected, actual } => {
                write!(f, "Shape mismatch: expected {:?}, got {:?}", expected, actual)
            }
            Self::InvalidDimension { dim, ndim } => {
                write!(f, "Invalid dimension {} for tensor with {} dimensions", dim, ndim)
            }
            Self::IndexOutOfBounds { index, size } => {
                write!(f, "Index {} out of bounds for size {}", index, size)
            }
            Self::MemoryError { message } => write!(f, "Memory error: {}", message),
            Self::InvalidOperation { message } => write!(f, "Invalid operation: {}", message),
            Self::SimdNotSupported => write!(f, "SIMD operation not supported on this architecture"),
            Self::GradientError { message } => write!(f, "Gradient computation failed: {}", message),
            Self::NumericalError { message } => write!(f, "Numerical error: {}", message),
            Self::DivisionByZero => write!(f, "Division by zero"),
        }
    }
}

impl std::error::Error for QuasarError {}

// Convenience functions for common errors
impl QuasarError {
    /// Create a generic error message
    pub fn generic(message: impl Into<String>) -> Self {
        Self::InvalidOperation {
            message: message.into(),
        }
    }

    /// Check if error is a shape mismatch
    pub fn is_shape_mismatch(&self) -> bool {
        matches!(self, Self::ShapeMismatch { .. })
    }

    /// Check if error is division by zero
    pub fn is_division_by_zero(&self) -> bool {
        matches!(self, Self::DivisionByZero)
    }
} 