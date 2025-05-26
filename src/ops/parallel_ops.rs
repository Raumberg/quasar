//! Parallel tensor operations with automatic parallelization

use crate::core::{Tensor, TensorElement, Shape};
use crate::error::{QuasarError, Result};
use crate::autograd::parallel::{
    get_global_coordinator, parallel_matmul, parallel_elementwise, ParallelTensorOps
};
use crate::autograd::engine::{AutogradOp, with_global_engine};
use crate::autograd::graph::Operation;
use rayon::prelude::*;
use num_traits::FromPrimitive;

/// Parallel addition with automatic parallelization
pub fn par_add<T: TensorElement + Send + Sync>(a: &Tensor<T>, b: &Tensor<T>) -> Result<Tensor<T>> {
    // Check shape compatibility
    if a.shape() != b.shape() {
        return Err(QuasarError::shape_mismatch(
            a.shape().dims(), b.shape().dims()
        ));
    }

    let coordinator = get_global_coordinator();
    let config = coordinator.get_config();
    let size = a.numel();

    let result = if config.auto_parallel && size >= config.parallel_threshold {
        // Parallel execution
        parallel_elementwise(&[a, b], |values| values[0] + values[1])?
    } else {
        // Sequential execution
        crate::ops::arithmetic::add(a, b)?
    };

    // Handle autograd registration
    let requires_grad = a.requires_grad_flag() || b.requires_grad_flag();
    if requires_grad {
        let mut result = result.requires_grad(true);
        register_parallel_operation(Operation::Add, a, b, &mut result)?;
        Ok(result)
    } else {
        Ok(result)
    }
}

/// Parallel multiplication with automatic parallelization
pub fn par_mul<T: TensorElement + Send + Sync>(a: &Tensor<T>, b: &Tensor<T>) -> Result<Tensor<T>> {
    // Check shape compatibility
    if a.shape() != b.shape() {
        return Err(QuasarError::shape_mismatch(
            a.shape().dims(), b.shape().dims()
        ));
    }

    let coordinator = get_global_coordinator();
    let config = coordinator.get_config();
    let size = a.numel();

    let result = if config.auto_parallel && size >= config.parallel_threshold {
        // Parallel execution
        parallel_elementwise(&[a, b], |values| values[0] * values[1])?
    } else {
        // Sequential execution
        crate::ops::arithmetic::mul(a, b)?
    };

    // Handle autograd registration
    let requires_grad = a.requires_grad_flag() || b.requires_grad_flag();
    if requires_grad {
        let mut result = result.requires_grad(true);
        register_parallel_operation(Operation::Mul, a, b, &mut result)?;
        Ok(result)
    } else {
        Ok(result)
    }
}

/// Parallel matrix multiplication with automatic parallelization
pub fn par_matmul<T: TensorElement + Send + Sync>(a: &Tensor<T>, b: &Tensor<T>) -> Result<Tensor<T>> {
    let coordinator = get_global_coordinator();
    let config = coordinator.get_config();
    
    let a_dims = a.shape().dims();
    let b_dims = b.shape().dims();
    let operation_size = a_dims[0] * a_dims[1] * b_dims[1];

    let result = if config.auto_parallel && operation_size >= config.parallel_threshold {
        // Parallel matrix multiplication
        parallel_matmul(a, b)?
    } else {
        // Sequential matrix multiplication
        crate::ops::linalg::matmul(a, b)?
    };

    // Handle autograd registration
    let requires_grad = a.requires_grad_flag() || b.requires_grad_flag();
    if requires_grad {
        let mut result = result.requires_grad(true);
        register_parallel_operation(Operation::MatMul, a, b, &mut result)?;
        Ok(result)
    } else {
        Ok(result)
    }
}

/// Parallel ReLU activation
pub fn par_relu<T: TensorElement + Send + Sync>(input: &Tensor<T>) -> Result<Tensor<T>> {
    let coordinator = get_global_coordinator();
    let config = coordinator.get_config();
    let size = input.numel();

    let result = if config.auto_parallel && size >= config.parallel_threshold {
        // Parallel ReLU
        let result_data: Vec<T> = input.data().par_iter()
            .map(|&x| if x > T::zero() { x } else { T::zero() })
            .collect();
        Tensor::new(result_data, input.shape().clone())?
    } else {
        // Sequential ReLU
        crate::ops::activation::relu(input)?
    };

    // Handle autograd registration
    if input.requires_grad_flag() {
        let mut result = result.requires_grad(true);
        register_parallel_unary_operation(Operation::ReLU, input, &mut result)?;
        Ok(result)
    } else {
        Ok(result)
    }
}

/// Parallel reduction operations
pub fn par_sum<T: TensorElement + Send + Sync>(input: &Tensor<T>, dim: Option<usize>) -> Result<Tensor<T>> {
    let coordinator = get_global_coordinator();
    let config = coordinator.get_config();
    let size = input.numel();

    if config.auto_parallel && size >= config.parallel_threshold {
        match dim {
            None => {
                // Sum all elements in parallel
                let sum = input.data().par_iter().cloned().reduce(|| T::zero(), |a, b| a + b);
                Tensor::new(vec![sum], Shape::from(&[1]))
            }
            Some(axis) => {
                // Sum along specific axis (more complex implementation needed)
                // For now, fall back to sequential
                par_sum_sequential(input, dim)
            }
        }
    } else {
        par_sum_sequential(input, dim)
    }
}

/// Sequential sum implementation (fallback)
fn par_sum_sequential<T: TensorElement>(input: &Tensor<T>, dim: Option<usize>) -> Result<Tensor<T>> {
    match dim {
        None => {
            let sum = input.data().iter().cloned().fold(T::zero(), |a, b| a + b);
            Tensor::new(vec![sum], Shape::from(&[1]))
        }
        Some(_axis) => {
            // TODO: Implement axis-specific sum
            Err(QuasarError::invalid_operation("Axis-specific sum not implemented yet"))
        }
    }
}

/// Parallel convolution (placeholder for future implementation)
pub fn par_conv2d<T: TensorElement + Send + Sync>(
    input: &Tensor<T>,
    kernel: &Tensor<T>,
    stride: (usize, usize),
    padding: (usize, usize),
) -> Result<Tensor<T>> {
    // TODO: Implement parallel 2D convolution
    // This would involve:
    // 1. Parallel processing of different output channels
    // 2. Parallel processing of different spatial locations
    // 3. SIMD optimizations for inner loops
    
    Err(QuasarError::invalid_operation("Parallel conv2d not implemented yet"))
}

/// Batch operations with automatic parallelization
pub fn par_batch_matmul<T: TensorElement + Send + Sync>(
    batch_a: &[&Tensor<T>],
    batch_b: &[&Tensor<T>],
) -> Result<Vec<Tensor<T>>> {
    if batch_a.len() != batch_b.len() {
        return Err(QuasarError::invalid_operation("Batch sizes must match"));
    }

    let coordinator = get_global_coordinator();
    let config = coordinator.get_config();

    if config.auto_parallel && batch_a.len() > 1 {
        // Parallel batch processing
        let results: Result<Vec<_>> = batch_a.par_iter()
            .zip(batch_b.par_iter())
            .map(|(a, b)| par_matmul(a, b))
            .collect();
        results
    } else {
        // Sequential batch processing
        let mut results = Vec::new();
        for (a, b) in batch_a.iter().zip(batch_b.iter()) {
            results.push(par_matmul(a, b)?);
        }
        Ok(results)
    }
}

/// Register parallel operation in autograd graph
fn register_parallel_operation<T: TensorElement>(
    operation: Operation<T>,
    a: &Tensor<T>,
    b: &Tensor<T>,
    result: &mut Tensor<T>,
) -> Result<()> {
    // Use global engine for registration (same as sequential operations)
    with_global_engine::<T, _, _>(|engine| {
        let input_node_a = if let Some(node_id) = a.node_id() {
            node_id
        } else if a.requires_grad_flag() {
            engine.get_or_register_leaf(a)
        } else {
            engine.register_leaf(a.clone())
        };
        
        let input_node_b = if let Some(node_id) = b.node_id() {
            node_id
        } else if b.requires_grad_flag() {
            engine.get_or_register_leaf(b)
        } else {
            engine.register_leaf(b.clone())
        };
        
        let input_nodes = vec![input_node_a, input_node_b];
        
        // Create gradient function (same as sequential)
        let grad_fn = create_parallel_gradient_function(operation.clone(), a.clone(), b.clone());
        
        let output_node = engine.register_operation(
            operation,
            input_nodes,
            result.clone(),
            grad_fn,
        )?;
        
        result.set_node_id(output_node);
        Ok(())
    })
}

/// Register parallel unary operation in autograd graph
fn register_parallel_unary_operation<T: TensorElement>(
    operation: Operation<T>,
    input: &Tensor<T>,
    result: &mut Tensor<T>,
) -> Result<()> {
    with_global_engine::<T, _, _>(|engine| {
        let input_node = if let Some(node_id) = input.node_id() {
            node_id
        } else if input.requires_grad_flag() {
            engine.get_or_register_leaf(input)
        } else {
            engine.register_leaf(input.clone())
        };
        
        let input_nodes = vec![input_node];
        
        // Create gradient function for unary operation
        let grad_fn = create_parallel_unary_gradient_function(operation.clone(), input.clone());
        
        let output_node = engine.register_operation(
            operation,
            input_nodes,
            result.clone(),
            grad_fn,
        )?;
        
        result.set_node_id(output_node);
        Ok(())
    })
}

/// Create gradient function for parallel binary operations
fn create_parallel_gradient_function<T: TensorElement>(
    operation: Operation<T>,
    a: Tensor<T>,
    b: Tensor<T>,
) -> Box<dyn Fn(&Tensor<T>, &[&Tensor<T>]) -> Result<Vec<Tensor<T>>> + Send + Sync> {
    Box::new(move |grad_output: &Tensor<T>, _inputs: &[&Tensor<T>]| -> Result<Vec<Tensor<T>>> {
        match operation {
            Operation::Add => {
                // Gradient of addition: both inputs get the same gradient
                Ok(vec![grad_output.clone(), grad_output.clone()])
            }
            Operation::Mul => {
                // Gradient of multiplication: da = grad * b, db = grad * a
                let grad_a = par_mul(grad_output, &b)?;
                let grad_b = par_mul(grad_output, &a)?;
                Ok(vec![grad_a, grad_b])
            }
            Operation::MatMul => {
                // Gradient of matrix multiplication (using parallel operations)
                let grad_a = par_matmul_transpose_rhs(grad_output, &b)?;
                let grad_b = par_matmul_transpose_lhs(&a, grad_output)?;
                Ok(vec![grad_a, grad_b])
            }
            _ => Err(QuasarError::invalid_operation("Unsupported parallel operation"))
        }
    })
}

/// Create gradient function for parallel unary operations
fn create_parallel_unary_gradient_function<T: TensorElement>(
    operation: Operation<T>,
    input: Tensor<T>,
) -> Box<dyn Fn(&Tensor<T>, &[&Tensor<T>]) -> Result<Vec<Tensor<T>>> + Send + Sync> {
    Box::new(move |grad_output: &Tensor<T>, _inputs: &[&Tensor<T>]| -> Result<Vec<Tensor<T>>> {
        match operation {
            Operation::ReLU => {
                // Gradient of ReLU: 1 if x > 0, 0 otherwise (parallel)
                let coordinator = get_global_coordinator();
                let config = coordinator.get_config();
                
                if config.auto_parallel && input.numel() >= config.parallel_threshold {
                    let grad_data: Vec<T> = input.data().par_iter()
                        .zip(grad_output.data().par_iter())
                        .map(|(&x, &grad)| if x > T::zero() { grad } else { T::zero() })
                        .collect();
                    let grad_input = Tensor::new(grad_data, input.shape().clone())?;
                    Ok(vec![grad_input])
                } else {
                    // Fall back to sequential
                    let grad_data: Vec<T> = input.data().iter()
                        .zip(grad_output.data().iter())
                        .map(|(&x, &grad)| if x > T::zero() { grad } else { T::zero() })
                        .collect();
                    let grad_input = Tensor::new(grad_data, input.shape().clone())?;
                    Ok(vec![grad_input])
                }
            }
            _ => Err(QuasarError::invalid_operation("Unsupported parallel unary operation"))
        }
    })
}

/// Helper: parallel matrix multiplication with transposed RHS
fn par_matmul_transpose_rhs<T: TensorElement + Send + Sync>(
    a: &Tensor<T>, 
    b: &Tensor<T>
) -> Result<Tensor<T>> {
    // TODO: Implement efficient parallel transpose + matmul
    // For now, use sequential implementation
    crate::ops::linalg::matmul_transpose_rhs(a, b)
}

/// Helper: parallel matrix multiplication with transposed LHS
fn par_matmul_transpose_lhs<T: TensorElement + Send + Sync>(
    a: &Tensor<T>, 
    b: &Tensor<T>
) -> Result<Tensor<T>> {
    // TODO: Implement efficient parallel transpose + matmul
    // For now, use sequential implementation
    crate::ops::linalg::matmul_transpose_lhs(a, b)
}

/// Parallel tensor statistics
pub fn par_statistics<T: TensorElement + Send + Sync + FromPrimitive>(input: &Tensor<T>) -> Result<TensorStats<T>> {
    let coordinator = get_global_coordinator();
    let config = coordinator.get_config();
    let size = input.numel();

    if config.auto_parallel && size >= config.parallel_threshold {
        // Parallel statistics computation
        let data = input.data();
        
        let (min, max, sum) = data.par_iter().cloned().fold(
            || (T::zero(), T::zero(), T::zero()),
            |(min, max, sum), x| {
                let new_min = if x < min || min == T::zero() { x } else { min };
                let new_max = if x > max { x } else { max };
                (new_min, new_max, sum + x)
            }
        ).reduce(
            || (T::zero(), T::zero(), T::zero()),
            |(min1, max1, sum1), (min2, max2, sum2)| {
                let final_min = if min1 < min2 || min1 == T::zero() { min1 } else { min2 };
                let final_max = if max1 > max2 { max1 } else { max2 };
                (final_min, final_max, sum1 + sum2)
            }
        );

        Ok(TensorStats {
            min,
            max,
            sum,
            mean: sum / T::from_usize(size).unwrap(),
            count: size,
        })
    } else {
        // Sequential statistics
        let data = input.data();
        let min = data.iter().cloned().fold(T::zero(), |a, b| if b < a || a == T::zero() { b } else { a });
        let max = data.iter().cloned().fold(T::zero(), |a, b| if b > a { b } else { a });
        let sum = data.iter().cloned().fold(T::zero(), |a, b| a + b);
        
        Ok(TensorStats {
            min,
            max,
            sum,
            mean: sum / T::from_usize(size).unwrap(),
            count: size,
        })
    }
}

/// Statistics about a tensor
#[derive(Debug, Clone)]
pub struct TensorStats<T: TensorElement> {
    pub min: T,
    pub max: T,
    pub sum: T,
    pub mean: T,
    pub count: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::Shape;
    use crate::autograd::parallel::init_parallel_autograd;

    #[test]
    fn test_parallel_add() -> Result<()> {
        // Initialize parallel autograd
        let config = crate::autograd::parallel::ParallelConfig {
            parallel_threshold: 2, // Low threshold for testing
            ..Default::default()
        };
        let _ = init_parallel_autograd(config);

        let a = Tensor::new(vec![1.0f32, 2.0, 3.0], Shape::from(&[3]))?;
        let b = Tensor::new(vec![4.0f32, 5.0, 6.0], Shape::from(&[3]))?;
        
        let result = par_add(&a, &b)?;
        assert_eq!(result.data(), &[5.0, 7.0, 9.0]);
        
        Ok(())
    }

    #[test]
    fn test_parallel_matmul() -> Result<()> {
        let config = crate::autograd::parallel::ParallelConfig {
            parallel_threshold: 4, // Low threshold for testing
            ..Default::default()
        };
        let _ = init_parallel_autograd(config);

        let a = Tensor::new(vec![1.0f32, 2.0, 3.0, 4.0], Shape::from(&[2, 2]))?;
        let b = Tensor::new(vec![5.0f32, 6.0, 7.0, 8.0], Shape::from(&[2, 2]))?;
        
        let result = par_matmul(&a, &b)?;
        // Expected: [1*5+2*7, 1*6+2*8] = [19, 22]
        //           [3*5+4*7, 3*6+4*8] = [43, 50]
        assert_eq!(result.data(), &[19.0, 22.0, 43.0, 50.0]);
        
        Ok(())
    }

    #[test]
    fn test_parallel_relu() -> Result<()> {
        let config = crate::autograd::parallel::ParallelConfig {
            parallel_threshold: 3, // Low threshold for testing
            ..Default::default()
        };
        let _ = init_parallel_autograd(config);

        let input = Tensor::new(vec![-2.0f32, -1.0, 0.0, 1.0, 2.0], Shape::from(&[5]))?;
        let result = par_relu(&input)?;
        assert_eq!(result.data(), &[0.0, 0.0, 0.0, 1.0, 2.0]);
        
        Ok(())
    }

    #[test]
    fn test_parallel_statistics() -> Result<()> {
        let input = Tensor::new(vec![1.0f32, 2.0, 3.0, 4.0, 5.0], Shape::from(&[5]))?;
        let stats = par_statistics(&input)?;
        
        assert_eq!(stats.min, 1.0);
        assert_eq!(stats.max, 5.0);
        assert_eq!(stats.sum, 15.0);
        assert_eq!(stats.mean, 3.0);
        assert_eq!(stats.count, 5);
        
        Ok(())
    }
} 
