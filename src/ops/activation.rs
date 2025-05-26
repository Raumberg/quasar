//! Activation functions

use crate::core::{Tensor, TensorElement};
use crate::error::{QuasarError, Result};
use crate::autograd::engine::{AutogradOp, with_global_engine};
use crate::autograd::graph::Operation;

/// ReLU activation function
pub fn relu<T: TensorElement>(input: &Tensor<T>) -> Result<Tensor<T>> {
    let relu_op = ReLUOp;
    let result = relu_op.forward(&[input])?;

    // Set requires_grad for result if input requires grad
    if input.requires_grad_flag() {
        let mut result = result.requires_grad(true);
        
        // Register operation in global computational graph
        register_unary_operation(Operation::ReLU, input, &mut result, Box::new(relu_op))?;
        Ok(result)
    } else {
        Ok(result)
    }
}

/// Sigmoid activation
pub fn sigmoid<T: TensorElement>(_input: &Tensor<T>) -> Result<Tensor<T>> {
    // TODO: Implement with SIMD optimizations
    todo!("Sigmoid activation")
}

/// Tanh activation
pub fn tanh<T: TensorElement>(_input: &Tensor<T>) -> Result<Tensor<T>> {
    // TODO: Implement with SIMD optimizations
    todo!("Tanh activation")
}

/// ReLU operation struct
pub struct ReLUOp;

impl<T: TensorElement> AutogradOp<T> for ReLUOp {
    fn forward(&self, inputs: &[&Tensor<T>]) -> Result<Tensor<T>> {
        if inputs.len() != 1 {
            return Err(QuasarError::invalid_operation("ReLU requires exactly 1 input"));
        }
        
        let input = inputs[0];
        let result_data: Vec<T> = input.data().iter()
            .map(|&x| if x > T::zero() { x } else { T::zero() })
            .collect();

        Tensor::new(result_data, input.shape().clone())
    }

    fn backward(&self, grad_output: &Tensor<T>, inputs: &[&Tensor<T>]) -> Result<Vec<Tensor<T>>> {
        if inputs.len() != 1 {
            return Err(QuasarError::invalid_operation("ReLU requires exactly 1 input"));
        }
        
        let input = inputs[0];
        
        // Gradient of ReLU: 1 if x > 0, 0 if x <= 0
        let grad_data: Vec<T> = input.data().iter()
            .zip(grad_output.data().iter())
            .map(|(&x, &grad)| if x > T::zero() { grad } else { T::zero() })
            .collect();

        let grad_input = Tensor::new(grad_data, input.shape().clone())?;
        Ok(vec![grad_input])
    }
}

/// Helper function to register unary operations in global computational graph
fn register_unary_operation<T: TensorElement>(
    operation: Operation<T>,
    input: &Tensor<T>,
    result: &mut Tensor<T>,
    autograd_op: Box<dyn AutogradOp<T>>,
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
        let grad_fn = create_unary_gradient_function(autograd_op, input.clone());
        
        // Register operation
        let output_node = engine.register_operation(
            operation,
            input_nodes,
            result.clone(),
            grad_fn,
        )?;
        
        // Set node ID for result tensor
        result.set_node_id(output_node);
        
        Ok(())
    })
}

/// Create gradient function from autograd operation with saved input
fn create_unary_gradient_function<T: TensorElement>(
    autograd_op: Box<dyn AutogradOp<T>>,
    input: Tensor<T>,
) -> Box<dyn Fn(&Tensor<T>, &[&Tensor<T>]) -> Result<Vec<Tensor<T>>> + Send + Sync> {
    Box::new(move |grad_output: &Tensor<T>, _inputs: &[&Tensor<T>]| -> Result<Vec<Tensor<T>>> {
        // Use saved input instead of the ones passed from graph
        let saved_inputs = vec![&input];
        autograd_op.backward(grad_output, &saved_inputs)
    })
} 