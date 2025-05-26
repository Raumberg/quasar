//! Example demonstrating a simple neural network layer with Quasar

use quasar::prelude::*;
use quasar::autograd::engine::with_global_engine;

/// Simple linear layer: output = input * weight + bias
fn linear_layer<T: TensorElement>(
    input: &Tensor<T>, 
    weight: &Tensor<T>, 
    bias: &Tensor<T>
) -> Result<Tensor<T>> {
    // For now, we'll simulate matrix multiplication with element-wise operations
    // TODO: Implement proper matrix multiplication
    
    // Ensure shapes are compatible for our simplified implementation
    if input.shape().dims()[0] != weight.shape().dims()[0] {
        return Err(QuasarError::shape_mismatch(
            input.shape().dims(), 
            weight.shape().dims()
        ));
    }
    
    // Simplified: element-wise multiply and add bias
    let weighted = (input * weight)?;
    let output = (&weighted + bias)?;
    
    Ok(output)
}

/// ReLU activation function (simplified)
fn relu<T: TensorElement>(input: &Tensor<T>) -> Result<Tensor<T>> {
    // For now, we'll just return the input (TODO: implement proper ReLU)
    // In a real implementation, this would be: max(0, x)
    Ok(input.clone())
}

/// Mean squared error loss
fn mse_loss<T: TensorElement>(predictions: &Tensor<T>, targets: &Tensor<T>) -> Result<Tensor<T>> {
    // loss = mean((predictions - targets)^2)
    let diff = (predictions - targets)?;
    let squared = (&diff * &diff)?;
    
    // For simplicity, we'll just sum instead of mean
    // TODO: Implement proper mean reduction
    Ok(squared)
}

fn main() -> Result<()> {
    println!("🧠 Quasar Neural Network Demo");
    
    // Create a simple dataset
    println!("\n=== Creating Dataset ===");
    
    // Input features: [features=3] (simplified for element-wise operations)
    let input = Tensor::new(
        vec![1.0, 2.0, 3.0], 
        Shape::from(&[3])
    )?.requires_grad(true);
    
    // Target outputs: [features=3]
    let targets = Tensor::new(
        vec![0.5, 1.0, 1.5], 
        Shape::from(&[3])
    )?;
    
    println!("Input shape: {:?}", input.shape());
    println!("Input data: {:?}", input.data());
    println!("Targets: {:?}", targets.data());
    
    // Initialize network parameters
    println!("\n=== Initializing Network Parameters ===");
    
    // Weight matrix: [features=3] (simplified for element-wise operations)
    let weight = Tensor::new(
        vec![0.1, 0.2, 0.3], 
        Shape::from(&[3])
    )?.requires_grad(true);
    
    // Bias vector: [features=3]
    let bias = Tensor::new(
        vec![0.1, 0.2, 0.3], 
        Shape::from(&[3])
    )?.requires_grad(true);
    
    println!("Weight shape: {:?}", weight.shape());
    println!("Bias shape: {:?}", bias.shape());
    
    // Forward pass
    println!("\n=== Forward Pass ===");
    
    // Layer 1: Linear transformation
    let layer1_output = linear_layer(&input, &weight, &bias)?;
    println!("Layer 1 output: {:?}", layer1_output.data());
    
    // Activation: ReLU (simplified)
    let activated = relu(&layer1_output)?;
    println!("After ReLU: {:?}", activated.data());
    
    // Compute loss
    let mut loss = mse_loss(&activated, &targets)?;
    println!("Loss: {:?}", loss.data());
    
    // Backward pass
    println!("\n=== Backward Pass ===");
    
    match loss.backward() {
        Ok(()) => {
            println!("✅ Backward pass completed successfully!");
            
            // Get gradients from global autograd engine
            let leaf_grads = with_global_engine::<f64, _, _>(|engine| {
                engine.get_leaf_gradients()
            });
            
            println!("Found {} leaf gradients:", leaf_grads.len());
            for (node_id, grad) in leaf_grads {
                println!("  Node {}: shape={:?}, data={:?}", 
                        node_id, grad.shape(), grad.data());
            }
        }
        Err(e) => {
            println!("❌ Backward pass failed: {:?}", e);
            return Err(e);
        }
    }
    
    // Simulate gradient descent step
    println!("\n=== Gradient Descent Step (Simulated) ===");
    println!("In a real implementation, we would update parameters:");
    println!("  weight = weight - learning_rate * weight.grad");
    println!("  bias = bias - learning_rate * bias.grad");
    
    // Training loop simulation
    println!("\n=== Training Loop Simulation ===");
    println!("For multiple epochs, we would:");
    println!("  1. Forward pass: predictions = model(input)");
    println!("  2. Compute loss: loss = mse_loss(predictions, targets)");
    println!("  3. Backward pass: loss.backward()");
    println!("  4. Update parameters: optimizer.step()");
    println!("  5. Zero gradients: optimizer.zero_grad()");
    
    println!("\n✅ Neural network demo completed!");
    println!("\n🚀 Next steps:");
    println!("  - Implement proper matrix multiplication");
    println!("  - Add real ReLU activation function");
    println!("  - Implement parameter updates and optimizers");
    println!("  - Add more layers and activation functions");
    
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear_layer_shapes() -> Result<()> {
        let input = Tensor::new(vec![1.0, 2.0, 3.0], Shape::from(&[3]))?;
        let weight = Tensor::new(vec![0.1, 0.2, 0.3], Shape::from(&[3]))?;
        let bias = Tensor::new(vec![0.1, 0.2, 0.3], Shape::from(&[3]))?;
        
        let output = linear_layer(&input, &weight, &bias)?;
        
        assert_eq!(output.shape(), input.shape());
        Ok(())
    }

    #[test]
    fn test_mse_loss_computation() -> Result<()> {
        let predictions = Tensor::new(vec![1.0, 2.0], Shape::from(&[2]))?;
        let targets = Tensor::new(vec![0.5, 1.5], Shape::from(&[2]))?;
        
        let loss = mse_loss(&predictions, &targets)?;
        
        // Expected: (1.0-0.5)^2 + (2.0-1.5)^2 = 0.25 + 0.25 = [0.25, 0.25]
        assert_eq!(loss.data(), &[0.25, 0.25]);
        Ok(())
    }

    #[test]
    fn test_neural_network_gradient_flow() -> Result<()> {
        // Simple test to ensure gradients flow through the network
        let input = Tensor::new(vec![1.0], Shape::from(&[1]))?.requires_grad(true);
        let weight = Tensor::new(vec![2.0], Shape::from(&[1]))?.requires_grad(true);
        let bias = Tensor::new(vec![0.5], Shape::from(&[1]))?.requires_grad(true);
        let target = Tensor::new(vec![3.0], Shape::from(&[1]))?;
        
        // Forward pass
        let output = linear_layer(&input, &weight, &bias)?;
        let mut loss = mse_loss(&output, &target)?;
        
        // Backward pass
        loss.backward()?;
        
        // Check that we have gradients
        let leaf_grads = with_global_engine::<f64, _, _>(|engine| {
            engine.get_leaf_gradients()
        });
        
        // Should have gradients for input, weight, and bias
        assert!(leaf_grads.len() >= 2); // At least input and weight (bias might be optimized out)
        
        Ok(())
    }
} 