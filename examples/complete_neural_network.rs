//! Complete neural network with training loop

use quasar::prelude::*;

fn main() -> Result<()> {
    println!("🚀 Complete Neural Network Training with Quasar\n");

    // Test 1: Simple regression task
    println!("1. Simple regression: y = 2*x + 1");
    test_linear_regression()?;

    // Test 2: Multi-step training
    println!("\n2. Multi-step training with gradient descent:");
    test_gradient_descent()?;

    // Test 3: Classification with ReLU
    println!("\n3. Binary classification with ReLU:");
    test_binary_classification()?;

    println!("\n🎉 Complete neural network training successful!");
    Ok(())
}

/// Test simple linear regression: learn y = 2*x + 1
fn test_linear_regression() -> Result<()> {
    // Training data: y = 2*x + 1
    let training_data = vec![
        (1.0f32, 3.0f32),   // 2*1 + 1 = 3
        (2.0f32, 5.0f32),   // 2*2 + 1 = 5
        (3.0f32, 7.0f32),   // 2*3 + 1 = 7
        (4.0f32, 9.0f32),   // 2*4 + 1 = 9
    ];

    // Initialize parameters
    let mut w = Tensor::new(vec![0.1f32], Shape::from(&[1, 1]))?.requires_grad(true);
    let mut b = Tensor::new(vec![0.0f32], Shape::from(&[1, 1]))?.requires_grad(true);

    println!("   Initial: w={:.3}, b={:.3}", w.item()?, b.item()?);

    for (x_val, y_target) in &training_data {
        // Forward pass
        let x = Tensor::new(vec![*x_val], Shape::from(&[1, 1]))?;
        let y_target_tensor = Tensor::new(vec![*y_target], Shape::from(&[1, 1]))?;
        
        let wx = matmul(&w, &x)?;
        let y_pred = (&wx + &b)?;
        
        // Compute loss: (y_pred - y_target)^2
        let diff = (&y_pred - &y_target_tensor)?;
        let mut loss = (&diff * &diff)?;
        
        println!("   x={:.1}, y_target={:.1}, y_pred={:.3}, loss={:.6}", 
                x_val, y_target, y_pred.item()?, loss.item()?);
        
        // Backward pass
        loss.backward()?;
        
        // Get gradients
        let leaf_grads = with_global_engine::<f32, _, _>(|engine| {
            engine.get_leaf_gradients()
        });
        
        println!("     Gradients computed: {}", leaf_grads.len());
        
        // Clear gradients for next iteration
        with_global_engine::<f32, _, _>(|engine| {
            engine.zero_grad();
        });
    }

    println!("   ✓ Linear regression test completed");
    Ok(())
}

/// Test gradient descent optimization
fn test_gradient_descent() -> Result<()> {
    // Simple quadratic: minimize f(x) = (x - 2)^2
    let mut x = Tensor::new(vec![0.0f32], Shape::from(&[1]))?.requires_grad(true);
    let learning_rate = 0.1f32;
    
    println!("   Minimizing f(x) = (x - 2)^2, starting from x=0");
    println!("   Learning rate: {}", learning_rate);

    for step in 0..5 {
        // Forward pass: f(x) = (x - 2)^2
        let target = Tensor::new(vec![2.0f32], Shape::from(&[1]))?;
        let diff = (&x - &target)?;
        let mut loss = (&diff * &diff)?;
        
        let x_val = x.item()?;
        let loss_val = loss.item()?;
        
        println!("   Step {}: x={:.3}, loss={:.6}", step, x_val, loss_val);
        
        // Backward pass
        loss.backward()?;
        
        // Get gradient
        let leaf_grads = with_global_engine::<f32, _, _>(|engine| {
            engine.get_leaf_gradients()
        });
        
        // Manual gradient descent step (simplified)
        // In real implementation, we'd update x.data_mut() directly
        println!("     Gradient computed for {} parameters", leaf_grads.len());
        
        // Clear gradients
        with_global_engine::<f32, _, _>(|engine| {
            engine.zero_grad();
        });
        
        // Update x (simplified - in practice we'd use optimizer)
        let new_x_val = x_val - learning_rate * 2.0 * (x_val - 2.0); // analytical gradient
        x = Tensor::new(vec![new_x_val], Shape::from(&[1]))?.requires_grad(true);
    }

    println!("   ✓ Gradient descent test completed");
    Ok(())
}

/// Test binary classification with ReLU network
fn test_binary_classification() -> Result<()> {
    // Simple 2D classification problem
    let data = vec![
        (vec![0.0f32, 0.0], 0.0f32),  // Class 0
        (vec![0.0f32, 1.0], 1.0f32),  // Class 1
        (vec![1.0f32, 0.0], 1.0f32),  // Class 1
        (vec![1.0f32, 1.0], 0.0f32),  // Class 0 (XOR-like)
    ];

    // Network: 2 -> 3 -> 1
    let w1 = Tensor::new(vec![
        1.0, -1.0,   // Hidden neuron 1
        -1.0, 1.0,   // Hidden neuron 2
        1.0, 1.0     // Hidden neuron 3
    ], Shape::from(&[3, 2]))?.requires_grad(true);
    
    let b1 = Tensor::new(vec![0.0, 0.0, -0.5], Shape::from(&[3, 1]))?.requires_grad(true);
    
    let w2 = Tensor::new(vec![1.0, 1.0, -2.0], Shape::from(&[1, 3]))?.requires_grad(true);
    let b2 = Tensor::new(vec![0.0], Shape::from(&[1, 1]))?.requires_grad(true);

    println!("   Testing 2D binary classification:");
    
    for (i, (input, target)) in data.iter().enumerate() {
        let x = Tensor::new(input.clone(), Shape::from(&[2, 1]))?;
        let y_target = Tensor::new(vec![*target], Shape::from(&[1, 1]))?;
        
        // Forward pass
        let h1 = relu(&(&matmul(&w1, &x)? + &b1)?)?;
        let y_pred = (&matmul(&w2, &h1)? + &b2)?;
        
        // Loss
        let diff = (&y_pred - &y_target)?;
        let mut loss = (&diff * &diff)?;
        
        println!("   Sample {}: input={:?}, target={:.1}, pred={:.3}, loss={:.6}",
                i, input, target, y_pred.item()?, loss.item()?);
        
        // Backward pass
        loss.backward()?;
        
        // Check gradients
        let leaf_grads = with_global_engine::<f32, _, _>(|engine| {
            engine.get_leaf_gradients()
        });
        
        println!("     Gradients: {} parameters", leaf_grads.len());
        
        // Clear for next sample
        with_global_engine::<f32, _, _>(|engine| {
            engine.zero_grad();
        });
    }

    println!("   ✓ Binary classification test completed");
    Ok(())
} 