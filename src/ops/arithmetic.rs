//! Arithmetic operations for tensors

use crate::core::{Tensor, TensorElement};
use crate::error::{QuasarError, Result};
use crate::autograd::engine::{AddOp, SubOp, MulOp, DivOp, AutogradOp, with_global_engine};
use crate::autograd::graph::Operation;

/// Element-wise addition of two tensors
pub fn add<T: TensorElement>(a: &Tensor<T>, b: &Tensor<T>) -> Result<Tensor<T>> {
    // Check shape compatibility
    if a.shape() != b.shape() {
        return Err(QuasarError::shape_mismatch(
            a.shape().dims(), b.shape().dims()
        ));
    }

    // Perform forward pass
    let add_op = AddOp;
    let result = add_op.forward(&[a, b])?;

    // Set requires_grad for result if any input requires grad
    let requires_grad = a.requires_grad_flag() || b.requires_grad_flag();
    if requires_grad {
        let mut result = result.requires_grad(true);
        
        // Register operation in global computational graph
        register_binary_operation(Operation::Add, a, b, &mut result, Box::new(add_op))?;
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

    // Perform forward pass
    let sub_op = SubOp;
    let result = sub_op.forward(&[a, b])?;

    // Set requires_grad for result if any input requires grad
    let requires_grad = a.requires_grad_flag() || b.requires_grad_flag();
    if requires_grad {
        let mut result = result.requires_grad(true);
        
        // Register operation in global computational graph
        register_binary_operation(Operation::Sub, a, b, &mut result, Box::new(sub_op))?;
        Ok(result)
    } else {
        Ok(result)
    }
}

/// Element-wise multiplication of two tensors
pub fn mul<T: TensorElement>(a: &Tensor<T>, b: &Tensor<T>) -> Result<Tensor<T>> {
    // Check shape compatibility
    if a.shape() != b.shape() {
        return Err(QuasarError::shape_mismatch(
            a.shape().dims(), b.shape().dims()
        ));
    }

    // Perform forward pass
    let mul_op = MulOp;
    let result = mul_op.forward(&[a, b])?;

    // Set requires_grad for result if any input requires grad
    let requires_grad = a.requires_grad_flag() || b.requires_grad_flag();
    if requires_grad {
        let mut result = result.requires_grad(true);
        
        // Register operation in global computational graph
        register_binary_operation(Operation::Mul, a, b, &mut result, Box::new(mul_op))?;
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

    // Perform forward pass
    let div_op = DivOp;
    let result = div_op.forward(&[a, b])?;

    // Set requires_grad for result if any input requires grad
    let requires_grad = a.requires_grad_flag() || b.requires_grad_flag();
    if requires_grad {
        let mut result = result.requires_grad(true);
        
        // Register operation in global computational graph
        register_binary_operation(Operation::Div, a, b, &mut result, Box::new(div_op))?;
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
    // Use global engine for all operations
    with_global_engine::<T, _, _>(|engine| {
        // Get or register input nodes (handle tensors that don't require gradients)
        let input_node_a = if let Some(node_id) = a.node_id() {
            node_id
        } else if a.requires_grad_flag() {
            // Get or register as leaf node (avoids duplicates)
            engine.get_or_register_leaf(a)
        } else {
            // For tensors that don't require gradients, register them as constants
            engine.register_leaf(a.clone())
        };
        
        let input_node_b = if let Some(node_id) = b.node_id() {
            node_id
        } else if b.requires_grad_flag() {
            // Get or register as leaf node (avoids duplicates)
            engine.get_or_register_leaf(b)
        } else {
            // For tensors that don't require gradients, register them as constants
            engine.register_leaf(b.clone())
        };
        
        let input_nodes = vec![input_node_a, input_node_b];
        
        // Create gradient function with saved inputs and requires_grad flags
        let grad_fn = create_gradient_function_with_flags(
            autograd_op, 
            a.clone(), 
            b.clone(),
            a.requires_grad_flag(),
            b.requires_grad_flag()
        );
        
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

/// Create gradient function from autograd operation with saved inputs and requires_grad flags
fn create_gradient_function_with_flags<T: TensorElement>(
    autograd_op: Box<dyn AutogradOp<T>>,
    input_a: Tensor<T>,
    input_b: Tensor<T>,
    a_requires_grad: bool,
    b_requires_grad: bool,
) -> Box<dyn Fn(&Tensor<T>, &[&Tensor<T>]) -> Result<Vec<Tensor<T>>> + Send + Sync> {
    Box::new(move |grad_output: &Tensor<T>, _inputs: &[&Tensor<T>]| -> Result<Vec<Tensor<T>>> {
        // Use saved inputs instead of the ones passed from graph
        let saved_inputs = vec![&input_a, &input_b];
        let mut gradients = autograd_op.backward(grad_output, &saved_inputs)?;
        
        // Zero out gradients for tensors that don't require gradients
        if !a_requires_grad && gradients.len() > 0 {
            gradients[0] = Tensor::zeros(gradients[0].shape().clone())?;
        }
        if !b_requires_grad && gradients.len() > 1 {
            gradients[1] = Tensor::zeros(gradients[1].shape().clone())?;
        }
        
        Ok(gradients)
    })
}

/// Create gradient function from autograd operation with saved inputs (legacy)
fn create_gradient_function<T: TensorElement>(
    autograd_op: Box<dyn AutogradOp<T>>,
    input_a: Tensor<T>,
    input_b: Tensor<T>,
) -> Box<dyn Fn(&Tensor<T>, &[&Tensor<T>]) -> Result<Vec<Tensor<T>>>> {
    Box::new(move |grad_output: &Tensor<T>, _inputs: &[&Tensor<T>]| -> Result<Vec<Tensor<T>>> {
        // Use saved inputs instead of the ones passed from graph
        let saved_inputs = vec![&input_a, &input_b];
        autograd_op.backward(grad_output, &saved_inputs)
    })
}

// Raw arithmetic implementations (without autograd)
impl AddOp {
    /// Raw addition implementation
    pub fn add_raw<T: TensorElement>(a: &Tensor<T>, b: &Tensor<T>) -> Result<Tensor<T>> {
        let result_data: Vec<T> = a.data().iter()
            .zip(b.data().iter())
            .map(|(&x, &y)| x + y)
            .collect();

        Tensor::new(result_data, a.shape().clone())
    }
}

impl SubOp {
    /// Raw subtraction implementation
    pub fn sub_raw<T: TensorElement>(a: &Tensor<T>, b: &Tensor<T>) -> Result<Tensor<T>> {
        let result_data: Vec<T> = a.data().iter()
            .zip(b.data().iter())
            .map(|(&x, &y)| x - y)
            .collect();

        Tensor::new(result_data, a.shape().clone())
    }
}

impl MulOp {
    /// Raw multiplication implementation
    pub fn mul_raw<T: TensorElement>(a: &Tensor<T>, b: &Tensor<T>) -> Result<Tensor<T>> {
        let result_data: Vec<T> = a.data().iter()
            .zip(b.data().iter())
            .map(|(&x, &y)| x * y)
            .collect();

        Tensor::new(result_data, a.shape().clone())
    }
}

impl DivOp {
    /// Raw division implementation
    pub fn div_raw<T: TensorElement>(a: &Tensor<T>, b: &Tensor<T>) -> Result<Tensor<T>> {
        let result_data: Vec<T> = a.data().iter()
            .zip(b.data().iter())
            .map(|(&x, &y)| x / y)
            .collect();

        Tensor::new(result_data, a.shape().clone())
    }
}

// Implement AutogradOp for operations using raw implementations
impl<T: TensorElement> AutogradOp<T> for AddOp {
    fn forward(&self, inputs: &[&Tensor<T>]) -> Result<Tensor<T>> {
        if inputs.len() != 2 {
            return Err(QuasarError::invalid_operation("Add requires exactly 2 inputs"));
        }
        Self::add_raw(inputs[0], inputs[1])
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
        Self::sub_raw(inputs[0], inputs[1])
    }

    fn backward(&self, grad_output: &Tensor<T>, _inputs: &[&Tensor<T>]) -> Result<Vec<Tensor<T>>> {
        // Gradient of subtraction: d/dx (x - y) = 1, d/dy (x - y) = -1
        let neg_ones = Tensor::new(vec![-T::one(); grad_output.numel()], grad_output.shape().clone())?;
        let neg_grad = MulOp::mul_raw(grad_output, &neg_ones)?;
        Ok(vec![grad_output.clone(), neg_grad])
    }
}

impl<T: TensorElement> AutogradOp<T> for MulOp {
    fn forward(&self, inputs: &[&Tensor<T>]) -> Result<Tensor<T>> {
        if inputs.len() != 2 {
            return Err(QuasarError::invalid_operation("Mul requires exactly 2 inputs"));
        }
        Self::mul_raw(inputs[0], inputs[1])
    }

    fn backward(&self, grad_output: &Tensor<T>, inputs: &[&Tensor<T>]) -> Result<Vec<Tensor<T>>> {
        if inputs.len() != 2 {
            return Err(QuasarError::invalid_operation("Mul requires exactly 2 inputs"));
        }
        
        // Gradient of multiplication: d/dx (x * y) = y, d/dy (x * y) = x
        let grad_x = Self::mul_raw(grad_output, inputs[1])?;
        let grad_y = Self::mul_raw(grad_output, inputs[0])?;
        
        Ok(vec![grad_x, grad_y])
    }
}

impl<T: TensorElement> AutogradOp<T> for DivOp {
    fn forward(&self, inputs: &[&Tensor<T>]) -> Result<Tensor<T>> {
        if inputs.len() != 2 {
            return Err(QuasarError::invalid_operation("Div requires exactly 2 inputs"));
        }
        Self::div_raw(inputs[0], inputs[1])
    }

    fn backward(&self, grad_output: &Tensor<T>, inputs: &[&Tensor<T>]) -> Result<Vec<Tensor<T>>> {
        if inputs.len() != 2 {
            return Err(QuasarError::invalid_operation("Div requires exactly 2 inputs"));
        }
        
        let x = inputs[0];
        let y = inputs[1];
        
        // Gradient of division: d/dx (x / y) = 1/y, d/dy (x / y) = -x/y²
        let grad_x = Self::div_raw(grad_output, y)?;
        
        let y_squared = MulOp::mul_raw(y, y)?;
        let neg_ones = Tensor::new(vec![-T::one(); x.numel()], x.shape().clone())?;
        let neg_x = MulOp::mul_raw(x, &neg_ones)?;
        let grad_y_temp = Self::div_raw(&neg_x, &y_squared)?;
        let grad_y = MulOp::mul_raw(grad_output, &grad_y_temp)?;
        
        Ok(vec![grad_x, grad_y])
    }
} 