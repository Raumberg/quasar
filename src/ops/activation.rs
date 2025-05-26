//! Activation functions with automatic parallelization

use crate::core::{Tensor, TensorElement};
use crate::error::{QuasarError, Result};
use crate::autograd::engine::{AutogradOp, with_global_engine, is_in_backward_pass};
use crate::autograd::graph::Operation;
use crate::autograd::parallel::{get_global_coordinator, parallel_elementwise};

/// ReLU activation function with automatic parallelization
pub fn relu<T: TensorElement>(input: &Tensor<T>) -> Result<Tensor<T>> {
    let coordinator = get_global_coordinator();
    let tensor_size = input.data().len();
    
    // Automatically choose parallel or sequential execution
    let result = if coordinator.should_parallelize(tensor_size) {
        // Parallel execution
        parallel_elementwise(&[input], |values| {
            if values[0] > T::zero() { values[0] } else { T::zero() }
        })?
    } else {
        // Sequential execution
        let result_data: Vec<T> = input.data().iter()
            .map(|&x| if x > T::zero() { x } else { T::zero() })
            .collect();
        Tensor::new(result_data, input.shape().clone())?
    };

    // Handle autograd registration (skip if in backward pass)
    if input.requires_grad_flag() && !is_in_backward_pass() {
        let mut result = result.requires_grad(true);
        register_relu_operation(input, &mut result)?;
        Ok(result)
    } else {
        Ok(result)
    }
}

/// Sigmoid activation function
pub fn sigmoid<T: TensorElement>(_input: &Tensor<T>) -> Result<Tensor<T>> {
    // TODO: Implement sigmoid with automatic parallelization
    todo!("Sigmoid activation")
}

/// Tanh activation function
pub fn tanh<T: TensorElement>(_input: &Tensor<T>) -> Result<Tensor<T>> {
    // TODO: Implement tanh with automatic parallelization
    todo!("Tanh activation")
}

/// ReLU operation struct
pub struct ReLUOp;

impl<T: TensorElement> AutogradOp<T> for ReLUOp {
    fn forward(&self, inputs: &[&Tensor<T>]) -> Result<Tensor<T>> {
        if inputs.len() != 1 {
            return Err(QuasarError::invalid_operation("ReLU requires exactly 1 input"));
        }
        relu(inputs[0])
    }

    fn backward(&self, grad_output: &Tensor<T>, inputs: &[&Tensor<T>]) -> Result<Vec<Tensor<T>>> {
        if inputs.len() != 1 {
            return Err(QuasarError::invalid_operation("ReLU requires exactly 1 input"));
        }
        
        let input = inputs[0];
        
        // Gradient of ReLU: 1 if x > 0, 0 otherwise
        let coordinator = get_global_coordinator();
        let tensor_size = input.data().len();
        
        let grad_input = if coordinator.should_parallelize(tensor_size) {
            // Parallel gradient computation
            parallel_elementwise(&[input, grad_output], |values| {
                if values[0] > T::zero() { values[1] } else { T::zero() }
            })?
        } else {
            // Sequential gradient computation
            let grad_data: Vec<T> = input.data().iter()
                .zip(grad_output.data().iter())
                .map(|(&x, &grad)| if x > T::zero() { grad } else { T::zero() })
                .collect();
            Tensor::new(grad_data, input.shape().clone())?
        };
        
        Ok(vec![grad_input])
    }
}

/// Helper function to register ReLU operations in global computational graph
fn register_relu_operation<T: TensorElement>(
    input: &Tensor<T>,
    result: &mut Tensor<T>,
) -> Result<()> {
    with_global_engine::<T, _, _>(|engine| {
        // Get or register input node
        let input_node = if let Some(node_id) = input.node_id() {
            node_id
        } else if input.requires_grad_flag() {
            engine.get_or_register_leaf(input)
        } else {
            engine.register_leaf(input.clone())
        };
        
        let input_nodes = vec![input_node];
        
        // Create gradient function with saved input
        let grad_fn = create_relu_gradient_function(input.clone());
        
        // Register operation
        let output_node = engine.register_operation(
            Operation::ReLU,
            input_nodes,
            result.clone(),
            grad_fn,
        )?;
        
        // Set node ID for result tensor
        result.set_node_id(output_node);
        
        Ok(())
    })
}

/// Create gradient function for ReLU with saved input
fn create_relu_gradient_function<T: TensorElement>(
    input: Tensor<T>,
) -> Box<dyn Fn(&Tensor<T>, &[&Tensor<T>]) -> Result<Vec<Tensor<T>>> + Send + Sync> {
    Box::new(move |grad_output: &Tensor<T>, _inputs: &[&Tensor<T>]| -> Result<Vec<Tensor<T>>> {
        // Use saved input for gradient computation
        let coordinator = get_global_coordinator();
        let tensor_size = input.data().len();
        
        let grad_input = if coordinator.should_parallelize(tensor_size) {
            // Parallel gradient computation
            parallel_elementwise(&[&input, grad_output], |values| {
                if values[0] > T::zero() { values[1] } else { T::zero() }
            })?
        } else {
            // Sequential gradient computation
            let grad_data: Vec<T> = input.data().iter()
                .zip(grad_output.data().iter())
                .map(|(&x, &grad)| if x > T::zero() { grad } else { T::zero() })
                .collect();
            Tensor::new(grad_data, input.shape().clone())?
        };
        
        Ok(vec![grad_input])
    })
} 