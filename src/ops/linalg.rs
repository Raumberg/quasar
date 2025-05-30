//! Linear algebra operations with automatic parallelization

use crate::core::{Tensor, TensorElement, Shape};
use crate::error::{QuasarError, Result};
use crate::autograd::engine::{AutogradOp, with_global_engine, is_in_backward_pass};
use crate::autograd::graph::Operation;
use crate::autograd::parallel::{get_global_coordinator, parallel_matmul};

/// Matrix multiplication with automatic parallelization
pub fn matmul<T: TensorElement>(a: &Tensor<T>, b: &Tensor<T>) -> Result<Tensor<T>> {
    // Check if tensors are large enough to benefit from parallelization
    let tensor_size = a.data().len() * b.data().len();
    let coordinator = get_global_coordinator();
    
    // Automatically choose parallel or sequential execution
    let result = if coordinator.should_parallelize(tensor_size) {
        parallel_matmul(a, b)?
    } else {
        sequential_matmul_impl(a, b)?
    };

    // Handle autograd registration (skip if in backward pass)
    let requires_grad = (a.requires_grad_flag() || b.requires_grad_flag()) && !is_in_backward_pass();
    if requires_grad {
        let mut result = result.requires_grad(true);
        register_matmul_operation(a, b, &mut result)?;
        Ok(result)
    } else {
        Ok(result)
    }
}

/// Sequential matrix multiplication implementation
fn sequential_matmul_impl<T: TensorElement>(a: &Tensor<T>, b: &Tensor<T>) -> Result<Tensor<T>> {
    let a_shape = a.shape();
    let b_shape = b.shape();
    
    if a_shape.ndim() != 2 || b_shape.ndim() != 2 {
        return Err(QuasarError::invalid_operation("MatMul requires 2D tensors"));
    }
    
    let a_dims = a_shape.dims();
    let b_dims = b_shape.dims();
    
    if a_dims[1] != b_dims[0] {
        return Err(QuasarError::shape_mismatch(a_dims, b_dims));
    }
    
    let m = a_dims[0];
    let k = a_dims[1];
    let n = b_dims[1];
    
    // Perform matrix multiplication: C = A * B
    let mut result_data = vec![T::zero(); m * n];
    
    for i in 0..m {
        for j in 0..n {
            let mut sum = T::zero();
            for l in 0..k {
                let a_val = a.data()[i * k + l];
                let b_val = b.data()[l * n + j];
                sum = sum + a_val * b_val;
            }
            result_data[i * n + j] = sum;
        }
    }
    
    let result_shape = Shape::from(&[m, n]);
    Tensor::new(result_data, result_shape)
}

/// Dot product
pub fn dot<T: TensorElement>(_lhs: &Tensor<T>, _rhs: &Tensor<T>) -> Result<Tensor<T>> {
    // TODO: Implement with SIMD optimizations
    todo!("Dot product")
}

/// Matrix multiplication operation struct
pub struct MatMulOp;

impl<T: TensorElement> AutogradOp<T> for MatMulOp {
    fn forward(&self, inputs: &[&Tensor<T>]) -> Result<Tensor<T>> {
        if inputs.len() != 2 {
            return Err(QuasarError::invalid_operation("MatMul requires exactly 2 inputs"));
        }
        matmul(inputs[0], inputs[1])
    }

    fn backward(&self, grad_output: &Tensor<T>, inputs: &[&Tensor<T>]) -> Result<Vec<Tensor<T>>> {
        if inputs.len() != 2 {
            return Err(QuasarError::invalid_operation("MatMul requires exactly 2 inputs"));
        }
        
        let lhs = inputs[0];  // A
        let rhs = inputs[1];  // B
        
        // For C = A * B:
        // dA = grad_output * B^T
        // dB = A^T * grad_output
        
        let grad_lhs = matmul_transpose_rhs(grad_output, rhs)?;
        let grad_rhs = matmul_transpose_lhs(lhs, grad_output)?;
        
        Ok(vec![grad_lhs, grad_rhs])
    }
}

/// Helper: multiply A * B^T
pub fn matmul_transpose_rhs<T: TensorElement>(a: &Tensor<T>, b: &Tensor<T>) -> Result<Tensor<T>> {
    let a_dims = a.shape().dims();
    let b_dims = b.shape().dims();
    
    let m = a_dims[0];
    let n = a_dims[1];
    let k = b_dims[0]; // B is k x n, B^T is n x k
    
    if n != b_dims[1] {
        return Err(QuasarError::shape_mismatch(a_dims, b_dims));
    }
    
    let mut result_data = vec![T::zero(); m * k];
    
    for i in 0..m {
        for j in 0..k {
            let mut sum = T::zero();
            for l in 0..n {
                let a_val = a.data()[i * n + l];
                let b_val = b.data()[j * n + l]; // B^T[j][l] = B[j][l]
                sum = sum + a_val * b_val;
            }
            result_data[i * k + j] = sum;
        }
    }
    
    let result_shape = Shape::from(&[m, k]);
    Tensor::new(result_data, result_shape)
}

/// Helper: multiply A^T * B
pub fn matmul_transpose_lhs<T: TensorElement>(a: &Tensor<T>, b: &Tensor<T>) -> Result<Tensor<T>> {
    let a_dims = a.shape().dims();
    let b_dims = b.shape().dims();
    
    let k = a_dims[0]; // A is k x m, A^T is m x k
    let m = a_dims[1];
    let n = b_dims[1];
    
    if k != b_dims[0] {
        return Err(QuasarError::shape_mismatch(a_dims, b_dims));
    }
    
    let mut result_data = vec![T::zero(); m * n];
    
    for i in 0..m {
        for j in 0..n {
            let mut sum = T::zero();
            for l in 0..k {
                let a_val = a.data()[l * m + i]; // A^T[i][l] = A[l][i]
                let b_val = b.data()[l * n + j];
                sum = sum + a_val * b_val;
            }
            result_data[i * n + j] = sum;
        }
    }
    
    let result_shape = Shape::from(&[m, n]);
    Tensor::new(result_data, result_shape)
}

/// Helper function to register matmul operations in global computational graph
fn register_matmul_operation<T: TensorElement>(
    lhs: &Tensor<T>,
    rhs: &Tensor<T>,
    result: &mut Tensor<T>,
) -> Result<()> {
    with_global_engine::<T, _, _>(|engine| {
        // Get or register input nodes
        let lhs_node = if let Some(node_id) = lhs.node_id() {
            node_id
        } else if lhs.requires_grad_flag() {
            engine.get_or_register_leaf(lhs)
        } else {
            engine.register_leaf(lhs.clone())
        };
        
        let rhs_node = if let Some(node_id) = rhs.node_id() {
            node_id
        } else if rhs.requires_grad_flag() {
            engine.get_or_register_leaf(rhs)
        } else {
            engine.register_leaf(rhs.clone())
        };
        
        let input_nodes = vec![lhs_node, rhs_node];
        
        // Create gradient function with saved inputs
        let grad_fn = create_matmul_gradient_function(lhs.clone(), rhs.clone());
        
        // Register operation
        let output_node = engine.register_operation(
            Operation::MatMul,
            input_nodes,
            result.clone(),
            grad_fn,
        )?;
        
        // Set node ID for result tensor
        result.set_node_id(output_node);
        
        Ok(())
    })
}

/// Create gradient function for matmul with saved inputs
fn create_matmul_gradient_function<T: TensorElement>(
    lhs: Tensor<T>,
    rhs: Tensor<T>,
) -> Box<dyn Fn(&Tensor<T>, &[&Tensor<T>]) -> Result<Vec<Tensor<T>>> + Send + Sync> {
    Box::new(move |grad_output: &Tensor<T>, _inputs: &[&Tensor<T>]| -> Result<Vec<Tensor<T>>> {
        // Use saved inputs for gradient computation
        let grad_lhs = matmul_transpose_rhs(grad_output, &rhs)?;
        let grad_rhs = matmul_transpose_lhs(&lhs, grad_output)?;
        Ok(vec![grad_lhs, grad_rhs])
    })
} 