//! Arithmetic operations for tensors with automatic parallelization

use crate::core::{Tensor, TensorElement};
use crate::error::{QuasarError, Result};
use crate::autograd::engine::{AddOp, SubOp, MulOp, DivOp, AutogradOp, with_global_engine, is_in_backward_pass};
use crate::autograd::graph::Operation;
use crate::autograd::parallel::{get_global_coordinator, parallel_elementwise};

/// Element-wise addition of two tensors with automatic parallelization
pub fn add<T: TensorElement>(a: &Tensor<T>, b: &Tensor<T>) -> Result<Tensor<T>> {
    // Check shape compatibility
    if a.shape() != b.shape() {
        return Err(QuasarError::shape_mismatch(
            a.shape().dims(), b.shape().dims()
        ));
    }

    // Automatically choose parallel or sequential execution
    let coordinator = get_global_coordinator();
    let tensor_size = a.data().len();
    
    let result = if coordinator.should_parallelize(tensor_size) {
        // Parallel execution
        parallel_elementwise(&[a, b], |values| values[0] + values[1])?
    } else {
        // Sequential execution
        let result_data: Vec<T> = a.data().iter()
            .zip(b.data().iter())
            .map(|(&x, &y)| x + y)
            .collect();
        Tensor::new(result_data, a.shape().clone())?
    };

    // Handle autograd registration (skip if in backward pass)
    let requires_grad = (a.requires_grad_flag() || b.requires_grad_flag()) && !is_in_backward_pass();
    if requires_grad {
        let mut result = result.requires_grad(true);
        register_binary_operation(Operation::Add, a, b, &mut result, Box::new(AddOp))?;
        Ok(result)
    } else {
        Ok(result)
    }
}

/// Element-wise subtraction of two tensors
pub fn sub<T: TensorElement>(a: &Tensor<T>, b: &Tensor<T>) -> Result<Tensor<T>> {
    // Check shape compatibility
    if a.shape() != b.shape() {
        return Err(QuasarError::shape_mismatch(
            a.shape().dims(), b.shape().dims()
        ));
    }

    // Automatically choose parallel or sequential execution
    let coordinator = get_global_coordinator();
    let tensor_size = a.data().len();
    
    let result = if coordinator.should_parallelize(tensor_size) {
        // Parallel execution
        parallel_elementwise(&[a, b], |values| values[0] - values[1])?
    } else {
        // Sequential execution
        let result_data: Vec<T> = a.data().iter()
            .zip(b.data().iter())
            .map(|(&x, &y)| x - y)
            .collect();
        Tensor::new(result_data, a.shape().clone())?
    };

    // Handle autograd registration (skip if in backward pass)
    let requires_grad = (a.requires_grad_flag() || b.requires_grad_flag()) && !is_in_backward_pass();
    if requires_grad {
        let mut result = result.requires_grad(true);
        register_binary_operation(Operation::Sub, a, b, &mut result, Box::new(SubOp))?;
        Ok(result)
    } else {
        Ok(result)
    }
}

/// Element-wise multiplication of two tensors with automatic parallelization
pub fn mul<T: TensorElement>(a: &Tensor<T>, b: &Tensor<T>) -> Result<Tensor<T>> {
    // Check shape compatibility
    if a.shape() != b.shape() {
        return Err(QuasarError::shape_mismatch(
            a.shape().dims(), b.shape().dims()
        ));
    }

    // Automatically choose parallel or sequential execution
    let coordinator = get_global_coordinator();
    let tensor_size = a.data().len();
    
    let result = if coordinator.should_parallelize(tensor_size) {
        // Parallel execution
        parallel_elementwise(&[a, b], |values| values[0] * values[1])?
    } else {
        // Sequential execution
        let result_data: Vec<T> = a.data().iter()
            .zip(b.data().iter())
            .map(|(&x, &y)| x * y)
            .collect();
        Tensor::new(result_data, a.shape().clone())?
    };

    // Handle autograd registration (skip if in backward pass)
    let requires_grad = (a.requires_grad_flag() || b.requires_grad_flag()) && !is_in_backward_pass();
    if requires_grad {
        let mut result = result.requires_grad(true);
        register_binary_operation(Operation::Mul, a, b, &mut result, Box::new(MulOp))?;
        Ok(result)
    } else {
        Ok(result)
    }
}

/// Element-wise division of two tensors
pub fn div<T: TensorElement>(a: &Tensor<T>, b: &Tensor<T>) -> Result<Tensor<T>> {
    // Check shape compatibility
    if a.shape() != b.shape() {
        return Err(QuasarError::shape_mismatch(
            a.shape().dims(), b.shape().dims()
        ));
    }

    // Check for division by zero
    for &val in b.data() {
        if val == T::zero() {
            return Err(QuasarError::division_by_zero());
        }
    }

    // Automatically choose parallel or sequential execution
    let coordinator = get_global_coordinator();
    let tensor_size = a.data().len();
    
    let result = if coordinator.should_parallelize(tensor_size) {
        // Parallel execution
        parallel_elementwise(&[a, b], |values| values[0] / values[1])?
    } else {
        // Sequential execution
        let result_data: Vec<T> = a.data().iter()
            .zip(b.data().iter())
            .map(|(&x, &y)| x / y)
            .collect();
        Tensor::new(result_data, a.shape().clone())?
    };

    // Handle autograd registration (skip if in backward pass)
    let requires_grad = (a.requires_grad_flag() || b.requires_grad_flag()) && !is_in_backward_pass();
    if requires_grad {
        let mut result = result.requires_grad(true);
        register_binary_operation(Operation::Div, a, b, &mut result, Box::new(DivOp))?;
        Ok(result)
    } else {
        Ok(result)
    }
}

/// Helper function to register binary operations in global computational graph
fn register_binary_operation<T: TensorElement>(
    operation: Operation<T>,
    a: &Tensor<T>,
    b: &Tensor<T>,
    result: &mut Tensor<T>,
    autograd_op: Box<dyn AutogradOp<T>>,
) -> Result<()> {
    with_global_engine::<T, _, _>(|engine| {
        // Get or register input nodes
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
        
        // Create gradient function with saved inputs
        let grad_fn = create_gradient_function(autograd_op, a.clone(), b.clone());
        
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

/// Create gradient function from autograd operation with saved inputs
fn create_gradient_function<T: TensorElement>(
    autograd_op: Box<dyn AutogradOp<T>>,
    input_a: Tensor<T>,
    input_b: Tensor<T>,
) -> Box<dyn Fn(&Tensor<T>, &[&Tensor<T>]) -> Result<Vec<Tensor<T>>> + Send + Sync> {
    Box::new(move |grad_output: &Tensor<T>, _inputs: &[&Tensor<T>]| -> Result<Vec<Tensor<T>>> {
        // Use saved inputs instead of the ones passed from graph
        let saved_inputs = vec![&input_a, &input_b];
        autograd_op.backward(grad_output, &saved_inputs)
    })
}

// Implement AutogradOp for operations
impl<T: TensorElement> AutogradOp<T> for AddOp {
    fn forward(&self, inputs: &[&Tensor<T>]) -> Result<Tensor<T>> {
        if inputs.len() != 2 {
            return Err(QuasarError::invalid_operation("Add requires exactly 2 inputs"));
        }
        add(inputs[0], inputs[1])
    }

    fn backward(&self, grad_output: &Tensor<T>, _inputs: &[&Tensor<T>]) -> Result<Vec<Tensor<T>>> {
        // Gradient of addition: d/dx (x + y) = 1, d/dy (x + y) = 1
        Ok(vec![grad_output.clone(), grad_output.clone()])
    }
}

impl<T: TensorElement> AutogradOp<T> for SubOp {
    fn forward(&self, inputs: &[&Tensor<T>]) -> Result<Tensor<T>> {
        if inputs.len() != 2 {
            return Err(QuasarError::invalid_operation("Sub requires exactly 2 inputs"));
        }
        sub(inputs[0], inputs[1])
    }

    fn backward(&self, grad_output: &Tensor<T>, _inputs: &[&Tensor<T>]) -> Result<Vec<Tensor<T>>> {
        // Gradient of subtraction: d/dx (x - y) = 1, d/dy (x - y) = -1
        let neg_ones = Tensor::new(vec![-T::one(); grad_output.numel()], grad_output.shape().clone())?;
        let neg_grad = mul(grad_output, &neg_ones)?;
        Ok(vec![grad_output.clone(), neg_grad])
    }
}

impl<T: TensorElement> AutogradOp<T> for MulOp {
    fn forward(&self, inputs: &[&Tensor<T>]) -> Result<Tensor<T>> {
        if inputs.len() != 2 {
            return Err(QuasarError::invalid_operation("Mul requires exactly 2 inputs"));
        }
        mul(inputs[0], inputs[1])
    }

    fn backward(&self, grad_output: &Tensor<T>, inputs: &[&Tensor<T>]) -> Result<Vec<Tensor<T>>> {
        if inputs.len() != 2 {
            return Err(QuasarError::invalid_operation("Mul requires exactly 2 inputs"));
        }
        
        // Gradient of multiplication: d/dx (x * y) = y, d/dy (x * y) = x
        let grad_x = mul(grad_output, inputs[1])?;
        let grad_y = mul(grad_output, inputs[0])?;
        
        Ok(vec![grad_x, grad_y])
    }
}

impl<T: TensorElement> AutogradOp<T> for DivOp {
    fn forward(&self, inputs: &[&Tensor<T>]) -> Result<Tensor<T>> {
        if inputs.len() != 2 {
            return Err(QuasarError::invalid_operation("Div requires exactly 2 inputs"));
        }
        div(inputs[0], inputs[1])
    }

    fn backward(&self, grad_output: &Tensor<T>, inputs: &[&Tensor<T>]) -> Result<Vec<Tensor<T>>> {
        if inputs.len() != 2 {
            return Err(QuasarError::invalid_operation("Div requires exactly 2 inputs"));
        }
        
        let x = inputs[0];
        let y = inputs[1];
        
        // Gradient of division: d/dx (x / y) = 1/y, d/dy (x / y) = -x/y²
        let grad_x = div(grad_output, y)?;
        
        let y_squared = mul(y, y)?;
        let neg_ones = Tensor::new(vec![-T::one(); x.numel()], x.shape().clone())?;
        let neg_x = mul(x, &neg_ones)?;
        let grad_y_temp = div(&neg_x, &y_squared)?;
        let grad_y = mul(grad_output, &grad_y_temp)?;
        
        Ok(vec![grad_x, grad_y])
    }
} 