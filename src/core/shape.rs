//! Shape and stride utilities for tensors

use crate::error::{QuasarError, Result};

/// Tensor shape representation
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Shape {
    /// Dimensions of the tensor
    dims: Vec<usize>,
}

impl Shape {
    /// Create new shape from dimensions
    pub fn new(dims: Vec<usize>) -> Result<Self> {
        if dims.is_empty() {
            return Err(QuasarError::invalid_operation("Shape cannot be empty"));
        }
        
        for &dim in &dims {
            if dim == 0 {
                return Err(QuasarError::invalid_operation("Shape dimensions cannot be zero"));
            }
        }
        
        Ok(Self { dims })
    }

    /// Create scalar shape (0-dimensional)
    pub fn scalar() -> Self {
        Self { dims: vec![1] }
    }

    /// Get dimensions as slice
    pub fn dims(&self) -> &[usize] {
        &self.dims
    }

    /// Get number of dimensions
    pub fn ndim(&self) -> usize {
        self.dims.len()
    }

    /// Check if shape is scalar
    pub fn is_scalar(&self) -> bool {
        self.dims.len() == 1 && self.dims[0] == 1
    }

    /// Get total number of elements
    pub fn total_elements(&self) -> usize {
        self.dims.iter().product()
    }

    /// Convert multi-dimensional indices to flat index
    pub fn indices_to_flat(&self, indices: &[usize]) -> Result<usize> {
        if indices.len() != self.dims.len() {
            return Err(QuasarError::invalid_dimension(indices.len(), self.dims.len()));
        }

        let mut flat_index = 0;
        let mut stride = 1;

        // Calculate flat index using row-major order
        for i in (0..self.dims.len()).rev() {
            if indices[i] >= self.dims[i] {
                return Err(QuasarError::index_out_of_bounds(indices[i], self.dims[i]));
            }
            flat_index += indices[i] * stride;
            stride *= self.dims[i];
        }

        Ok(flat_index)
    }

    /// Convert flat index to multi-dimensional indices
    pub fn flat_to_indices(&self, flat_index: usize) -> Result<Vec<usize>> {
        let total = self.total_elements();
        if flat_index >= total {
            return Err(QuasarError::index_out_of_bounds(flat_index, total));
        }

        let mut indices = vec![0; self.dims.len()];
        let mut remaining = flat_index;

        for i in (0..self.dims.len()).rev() {
            indices[i] = remaining % self.dims[i];
            remaining /= self.dims[i];
        }

        Ok(indices)
    }

    /// Check if two shapes are compatible for broadcasting
    pub fn is_broadcastable_with(&self, other: &Shape) -> bool {
        let max_ndim = self.ndim().max(other.ndim());
        
        for i in 0..max_ndim {
            let dim_a = if i < self.ndim() { self.dims[self.ndim() - 1 - i] } else { 1 };
            let dim_b = if i < other.ndim() { other.dims[other.ndim() - 1 - i] } else { 1 };
            
            if dim_a != dim_b && dim_a != 1 && dim_b != 1 {
                return false;
            }
        }
        
        true
    }

    /// Reshape to new dimensions (must preserve total elements)
    pub fn reshape(&self, new_dims: Vec<usize>) -> Result<Shape> {
        let new_total = new_dims.iter().product::<usize>();
        if new_total != self.total_elements() {
            return Err(QuasarError::shape_mismatch(
                &self.dims, &new_dims
            ));
        }
        
        Shape::new(new_dims)
    }
}

impl std::fmt::Display for Shape {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Shape({:?})", self.dims)
    }
}

impl From<Vec<usize>> for Shape {
    fn from(dims: Vec<usize>) -> Self {
        Self::new(dims).expect("Invalid shape dimensions")
    }
}

impl From<&[usize]> for Shape {
    fn from(dims: &[usize]) -> Self {
        Self::new(dims.to_vec()).expect("Invalid shape dimensions")
    }
}

impl<const N: usize> From<&[usize; N]> for Shape {
    fn from(dims: &[usize; N]) -> Self {
        Self::new(dims.to_vec()).expect("Invalid shape dimensions")
    }
}

impl<const N: usize> From<[usize; N]> for Shape {
    fn from(dims: [usize; N]) -> Self {
        Self::new(dims.to_vec()).expect("Invalid shape dimensions")
    }
}

/// Stride representation for memory layout
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Stride {
    strides: Vec<usize>,
}

impl Stride {
    /// Compute row-major strides for given shape
    pub fn row_major(shape: &Shape) -> Self {
        let dims = shape.dims();
        let mut strides = vec![1; dims.len()];
        
        for i in (0..dims.len().saturating_sub(1)).rev() {
            strides[i] = strides[i + 1] * dims[i + 1];
        }
        
        Self { strides }
    }
    
    /// Get strides
    pub fn strides(&self) -> &[usize] {
        &self.strides
    }
    
    /// Compute flat index from multi-dimensional indices
    pub fn flat_index(&self, indices: &[usize], shape: &Shape) -> Result<usize> {
        if indices.len() != shape.ndim() {
            return Err(QuasarError::invalid_dimension(indices.len(), shape.ndim()));
        }
        
        let mut flat_index = 0;
        for (i, &idx) in indices.iter().enumerate() {
            if idx >= shape.dims()[i] {
                return Err(QuasarError::index_out_of_bounds(idx, shape.dims()[i]));
            }
            flat_index += idx * self.strides[i];
        }
        Ok(flat_index)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shape_creation() {
        let shape = Shape::new(vec![2, 3, 4]).unwrap();
        assert_eq!(shape.dims(), &[2, 3, 4]);
        assert_eq!(shape.ndim(), 3);
        assert_eq!(shape.total_elements(), 24);
    }

    #[test]
    fn test_scalar_shape() {
        let shape = Shape::scalar();
        assert!(shape.is_scalar());
        assert_eq!(shape.total_elements(), 1);
    }

    #[test]
    fn test_indices_conversion() {
        let shape = Shape::new(vec![2, 3]).unwrap();
        
        // Test indices_to_flat
        assert_eq!(shape.indices_to_flat(&[0, 0]).unwrap(), 0);
        assert_eq!(shape.indices_to_flat(&[0, 1]).unwrap(), 1);
        assert_eq!(shape.indices_to_flat(&[1, 0]).unwrap(), 3);
        assert_eq!(shape.indices_to_flat(&[1, 2]).unwrap(), 5);
        
        // Test flat_to_indices
        assert_eq!(shape.flat_to_indices(0).unwrap(), vec![0, 0]);
        assert_eq!(shape.flat_to_indices(1).unwrap(), vec![0, 1]);
        assert_eq!(shape.flat_to_indices(3).unwrap(), vec![1, 0]);
        assert_eq!(shape.flat_to_indices(5).unwrap(), vec![1, 2]);
    }

    #[test]
    fn test_broadcasting() {
        let shape1 = Shape::new(vec![2, 3]).unwrap();
        let shape2 = Shape::new(vec![1, 3]).unwrap();
        let shape3 = Shape::new(vec![2, 1]).unwrap();
        let shape4 = Shape::new(vec![2, 4]).unwrap();
        
        assert!(shape1.is_broadcastable_with(&shape2));
        assert!(shape1.is_broadcastable_with(&shape3));
        assert!(!shape1.is_broadcastable_with(&shape4));
    }
} 