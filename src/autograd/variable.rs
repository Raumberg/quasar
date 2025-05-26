//! Variables with gradient tracking

use crate::core::{Tensor, TensorElement};
use crate::error::Result;

/// Variable wrapper for tensors with gradient tracking
pub struct Variable<T: TensorElement = f32> {
    /// Underlying tensor
    pub tensor: Tensor<T>,
    /// Whether this variable requires gradients
    pub requires_grad: bool,
}

impl<T: TensorElement> Variable<T> {
    /// Create new variable from tensor
    pub fn new(tensor: Tensor<T>, requires_grad: bool) -> Self {
        Self {
            tensor,
            requires_grad,
        }
    }
    
    /// Create variable that requires gradients
    pub fn with_grad(tensor: Tensor<T>) -> Self {
        Self::new(tensor, true)
    }
    
    /// Create variable that doesn't require gradients
    pub fn no_grad(tensor: Tensor<T>) -> Self {
        Self::new(tensor, false)
    }
    
    /// Get gradient if available
    pub fn grad(&self) -> Option<Tensor<T>> {
        self.tensor.grad().cloned()
    }
    
    /// Zero gradients
    pub fn zero_grad(&mut self) -> Result<()> {
        // TODO: Implement gradient zeroing
        todo!("Zero gradients")
    }
    
    /// Backward pass
    pub fn backward(&mut self) -> Result<()> {
        self.tensor.backward()
    }
    
    /// Detach from computation graph
    pub fn detach(&self) -> Variable<T> {
        // TODO: Implement detachment
        todo!("Detach from graph")
    }
}

impl<T: TensorElement> From<Tensor<T>> for Variable<T> {
    fn from(tensor: Tensor<T>) -> Self {
        Self::no_grad(tensor)
    }
} 