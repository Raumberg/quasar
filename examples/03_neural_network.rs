//! Simple neural network example
//! 
//! This example demonstrates:
//! - Building a simple neural network layer
//! - Forward pass through the network
//! - Backward pass and gradient computation
//! - Basic training loop concept

use quasar::prelude::*;
use std::time::Instant;

fn main() -> Result<()> {
    println!("🚀 Quasar Neural Network Example\n");

    // 1. Create a simple linear layer
    println!("1. Creating Neural Network Components:");
    
    // Input: 3 features
    // Hidden layer: 4 neurons  
    // Output: 2 classes
    
    // Initialize weights and biases
    let weights1 = Tensor::new(
        vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2],
        Shape::from(&[3, 4])
    )?.requires_grad(true);
    
    let bias1 = Tensor::new(
        vec![0.1, 0.2, 0.3, 0.4],
        Shape::from(&[1, 4])
    )?.requires_grad(true);
    
    let weights2 = Tensor::new(
        vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
        Shape::from(&[4, 2])
    )?.requires_grad(true);
    
    let bias2 = Tensor::new(
        vec![0.1, 0.2],
        Shape::from(&[1, 2])
    )?.requires_grad(true);
    
    println!("   Weights1 shape: {:?}", weights1.shape());
    println!("   Bias1 shape: {:?}", bias1.shape());
    println!("   Weights2 shape: {:?}", weights2.shape());
    println!("   Bias2 shape: {:?}", bias2.shape());

    // 2. Forward pass
    println!("\n2. Forward Pass:");
    
    // Input data (reshape to 1x3 for matrix multiplication)
    let input = Tensor::new(vec![1.0, 2.0, 3.0], Shape::from(&[1, 3]))?;
    println!("   Input: {:?}", input.data());
    
    // First layer: linear transformation
    let hidden = matmul(&input, &weights1)?;
    let hidden = add(&hidden, &bias1)?;
    println!("   Hidden (before activation): {:?}", hidden.data());
    
    // ReLU activation
    let hidden_activated = relu(&hidden)?;
    println!("   Hidden (after ReLU): {:?}", hidden_activated.data());
    
    // Second layer
    let output = matmul(&hidden_activated, &weights2)?;
    let mut output = add(&output, &bias2)?;
    println!("   Output: {:?}", output.data());

    // 3. Backward pass
    println!("\n3. Backward Pass:");
    
    let start_time = Instant::now();
    output.backward()?;
    let backward_time = start_time.elapsed();
    
    println!("   Backward pass completed in {:?}", backward_time);
    
    // Get gradients for weights and biases
    let mut gradients = Vec::new();
    with_global_engine::<f32, _, _>(|engine| {
        let leaf_grads = engine.get_leaf_gradients();
        println!("   Found {} leaf gradients", leaf_grads.len());
        
        for (node_id, grad) in leaf_grads {
            gradients.push((node_id, grad));
        }
    });
    
    // Display some gradients
    for (i, (node_id, grad)) in gradients.iter().enumerate().take(2) {
        println!("   Gradient for node {}: shape {:?}, first few values: {:?}", 
                node_id, grad.shape(), &grad.data()[..grad.data().len().min(4)]);
    }

    // 4. Different input example
    println!("\n4. Different Input Example:");
    
    // Create different input
    let input2 = Tensor::new(vec![0.5, 1.5, 2.5], Shape::from(&[1, 3]))?;
    println!("   Input2: {:?}", input2.data());
    
    // Forward pass for different input
    let hidden2 = matmul(&input2, &weights1)?;
    let hidden2 = add(&hidden2, &bias1)?;
    let hidden2_activated = relu(&hidden2)?;
    let output2 = matmul(&hidden2_activated, &weights2)?;
    let output2 = add(&output2, &bias2)?;
    
    println!("   Output2: {:?}", output2.data());

    // 5. Loss computation (simplified)
    println!("\n5. Loss Computation:");
    
    // Simple mean squared error loss
    let target = Tensor::new(vec![1.0, 0.0], Shape::from(&[1, 2]))?;
    println!("   Target: {:?}", target.data());
    
    let diff = sub(&output, &target)?;
    let squared_diff = mul(&diff, &diff)?;
    
    println!("   Squared difference: {:?}", squared_diff.data());
    println!("   Loss computation completed (no backward pass for simplicity)");

    // 6. Performance demonstration
    println!("\n6. Performance Test:");
    
    // Larger network for performance testing
    let large_input = Tensor::new(vec![1.0; 100], Shape::from(&[1, 100]))?;
    let large_weights = Tensor::new(vec![0.01; 10000], Shape::from(&[100, 100]))?;
    
    let start_time = Instant::now();
    let large_output = matmul(&large_input, &large_weights)?;
    let forward_time = start_time.elapsed();
    
    println!("   Large matrix multiplication (100x100): {:?}", forward_time);
    println!("   Output shape: {:?}", large_output.shape());
    println!("   First few output values: {:?}", &large_output.data()[..5]);

    println!("\n✅ Neural network example completed successfully!");
    println!("💡 Note: This demonstrates the building blocks for deep learning!");
    
    Ok(())
} 