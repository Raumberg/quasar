//! Data types for tensors

use num_traits::{Float, NumCast, Zero};
use std::fmt;

/// Data type enumeration for tensors
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DType {
    F32,
    F64,
}

impl DType {
    /// Get DType from type parameter
    pub fn from_type<T: TensorElement>() -> Self {
        if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>() {
            DType::F32
        } else if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f64>() {
            DType::F64
        } else {
            panic!("Unsupported tensor element type")
        }
    }
    
    /// Get size in bytes
    pub fn size_bytes(&self) -> usize {
        match self {
            DType::F32 => 4,
            DType::F64 => 8,
        }
    }
}

/// Trait for numeric types that can be used in tensors
pub trait TensorElement: 
    Float + NumCast + Zero + Copy + Send + Sync + fmt::Debug + fmt::Display + 'static 
{
    /// Type name for debugging
    fn type_name() -> &'static str;
    
    /// Machine epsilon for this type
    fn epsilon() -> Self;
}

impl TensorElement for f32 {
    fn type_name() -> &'static str {
        "f32"
    }
    
    fn epsilon() -> Self {
        f32::EPSILON
    }
}

impl TensorElement for f64 {
    fn type_name() -> &'static str {
        "f64"
    }
    
    fn epsilon() -> Self {
        f64::EPSILON
    }
} 