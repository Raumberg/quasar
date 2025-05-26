//! Core tensor implementation

use crate::core::{DType, Shape, TensorElement};
use crate::memory::AlignedVec;
use crate::error::{QuasarError, Result};
use crate::autograd::engine::{with_global_engine};
use std::fmt;

/// Main tensor structure
pub struct Tensor<T: TensorElement> {
    /// Tensor data stored in SIMD-aligned memory
    data: AlignedVec<T>,
    /// Shape of the tensor
    shape: Shape,
    /// Data type
    dtype: DType,
    /// Whether this tensor requires gradients
    requires_grad: bool,
    /// Gradient tensor (computed during backward pass)
    grad: Option<Box<Tensor<T>>>,
    /// Node ID in computational graph
    node_id: Option<usize>,
}

impl<T: TensorElement> Tensor<T> {
    /// Create new tensor with given shape and data
    pub fn new(data: Vec<T>, shape: Shape) -> Result<Self> {
        let total_elements = shape.total_elements();
        if data.len() != total_elements {
            return Err(QuasarError::invalid_operation(
                format!("Data length {} doesn't match shape {:?} (expected {})", 
                       data.len(), shape, total_elements)
            ));
        }

        Ok(Self {
            data: AlignedVec::from_vec(data)?,
            shape,
            dtype: DType::from_type::<T>(),
            requires_grad: false,
            grad: None,
            node_id: None,
        })
    }

    /// Create tensor filled with zeros
    pub fn zeros(shape: Shape) -> Result<Self> {
        let total_elements = shape.total_elements();
        let data = vec![T::zero(); total_elements];
        Self::new(data, shape)
    }

    /// Create tensor filled with ones
    pub fn ones(shape: Shape) -> Result<Self> {
        let total_elements = shape.total_elements();
        let data = vec![T::one(); total_elements];
        Self::new(data, shape)
    }

    /// Create scalar tensor
    pub fn from_scalar(value: T) -> Self {
        Self {
            data: AlignedVec::from_vec(vec![value]).expect("Failed to create scalar tensor"),
            shape: Shape::scalar(),
            dtype: DType::from_type::<T>(),
            requires_grad: false,
            grad: None,
            node_id: None,
        }
    }

    /// Set whether this tensor requires gradients
    pub fn requires_grad(mut self, requires_grad: bool) -> Self {
        self.requires_grad = requires_grad;
        self
    }

    /// Check if tensor requires gradients
    pub fn requires_grad_flag(&self) -> bool {
        self.requires_grad
    }

    /// Get tensor shape
    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    /// Get tensor data type
    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// Get tensor data as slice
    pub fn data(&self) -> &[T] {
        self.data.as_slice()
    }

    /// Get mutable tensor data
    pub fn data_mut(&mut self) -> &mut [T] {
        self.data.as_mut_slice()
    }

    /// Reshape tensor
    pub fn reshape(&self, new_shape: Shape) -> Result<Self> {
        if new_shape.total_elements() != self.shape.total_elements() {
            return Err(QuasarError::shape_mismatch(
                &[self.shape.total_elements()], &[new_shape.total_elements()]
            ));
        }

        Ok(Self {
            data: self.data.clone(),
            shape: new_shape,
            dtype: self.dtype,
            requires_grad: self.requires_grad,
            grad: None,
            node_id: self.node_id,
        })
    }

    /// Get single item from tensor (for scalars or single-element tensors)
    pub fn item(&self) -> Result<T> {
        if self.data.len() != 1 {
            return Err(QuasarError::invalid_operation(
                "item() can only be called on tensors with exactly one element"
            ));
        }
        Ok(self.data[0])
    }

    /// Get item at specific indices
    pub fn item_at(&self, indices: &[usize]) -> Result<T> {
        let flat_index = self.shape.indices_to_flat(indices)?;
        if flat_index >= self.data.len() {
            return Err(QuasarError::index_out_of_bounds(flat_index, self.data.len()));
        }
        Ok(self.data[flat_index])
    }

    /// Get gradient tensor
    pub fn grad(&self) -> Option<&Tensor<T>> {
        self.grad.as_ref().map(|g| g.as_ref())
    }

    /// Set gradient tensor
    pub fn set_grad(&mut self, grad: Tensor<T>) {
        self.grad = Some(Box::new(grad));
    }

    /// Clear gradients
    pub fn zero_grad(&mut self) {
        self.grad = None;
        with_global_engine::<T, _, _>(|engine| {
            engine.zero_grad();
        });
    }

    /// Perform backward pass
    pub fn backward(&mut self) -> Result<()> {
        if !self.requires_grad {
            return Err(QuasarError::invalid_operation(
                "backward() can only be called on tensors with requires_grad=True"
            ));
        }

        if let Some(node_id) = self.node_id {
            // Create gradient tensor (ones with same shape as output)
            let grad_output = Tensor::ones(self.shape.clone())?;
            
            // Execute backward pass using global engine
            with_global_engine::<T, _, _>(|engine| {
                engine.backward(node_id, grad_output)
            })?;
            
            // Get computed gradient from global engine
            let grad = with_global_engine::<T, _, _>(|engine| {
                engine.get_gradient(node_id).cloned()
            });
            
            if let Some(grad) = grad {
                self.grad = Some(Box::new(grad));
            }
        } else {
            return Err(QuasarError::invalid_operation(
                "No node ID found for tensor"
            ));
        }

        Ok(())
    }

    /// Check if tensor is scalar
    pub fn is_scalar(&self) -> bool {
        self.shape.is_scalar()
    }

    /// Get number of dimensions
    pub fn ndim(&self) -> usize {
        self.shape.ndim()
    }

    /// Get total number of elements
    pub fn numel(&self) -> usize {
        self.shape.total_elements()
    }

    /// Get node ID in computational graph (internal use)
    pub fn node_id(&self) -> Option<usize> {
        self.node_id
    }

    /// Set node ID in computational graph (internal use)
    pub fn set_node_id(&mut self, node_id: usize) {
        self.node_id = Some(node_id);
    }
}

// Implement Clone
impl<T: TensorElement> Clone for Tensor<T> {
    fn clone(&self) -> Self {
        Self {
            data: self.data.clone(),
            shape: self.shape.clone(),
            dtype: self.dtype,
            requires_grad: self.requires_grad,
            grad: self.grad.clone(),
            node_id: self.node_id,
        }
    }
}

// Implement Debug
impl<T: TensorElement> fmt::Debug for Tensor<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Tensor")
            .field("shape", &self.shape)
            .field("dtype", &self.dtype)
            .field("requires_grad", &self.requires_grad)
            .field("data", &self.data.as_slice())
            .finish()
    }
}

// Implement Display
impl<T: TensorElement> fmt::Display for Tensor<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Tensor(shape={:?}, dtype={:?}, requires_grad={})", 
               self.shape, self.dtype, self.requires_grad)?;
        
        if self.data.len() <= 10 {
            write!(f, "\ndata: {:?}", self.data.as_slice())?;
        } else {
            write!(f, "\ndata: [{}, {}, ..., {}, {}] ({} elements)", 
                   self.data[0], self.data[1], 
                   self.data[self.data.len()-2], self.data[self.data.len()-1],
                   self.data.len())?;
        }
        
        if let Some(ref grad) = self.grad {
            write!(f, "\ngrad: Some({:?})", grad.shape)?;
        } else {
            write!(f, "\ngrad: None")?;
        }
        
        Ok(())
    }
}

// Arithmetic operator overloading
impl<T: TensorElement> std::ops::Add<&Tensor<T>> for &Tensor<T> {
    type Output = Result<Tensor<T>>;

    fn add(self, other: &Tensor<T>) -> Self::Output {
        crate::ops::arithmetic::add(self, other)
    }
}

impl<T: TensorElement> std::ops::Sub<&Tensor<T>> for &Tensor<T> {
    type Output = Result<Tensor<T>>;

    fn sub(self, other: &Tensor<T>) -> Self::Output {
        crate::ops::arithmetic::sub(self, other)
    }
}

impl<T: TensorElement> std::ops::Mul<&Tensor<T>> for &Tensor<T> {
    type Output = Result<Tensor<T>>;

    fn mul(self, other: &Tensor<T>) -> Self::Output {
        crate::ops::arithmetic::mul(self, other)
    }
}

impl<T: TensorElement> std::ops::Div<&Tensor<T>> for &Tensor<T> {
    type Output = Result<Tensor<T>>;

    fn div(self, other: &Tensor<T>) -> Self::Output {
        crate::ops::arithmetic::div(self, other)
    }
} 