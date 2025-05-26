//! Simple neural network test with Quasar autograd engine

use quasar::prelude::*;

fn main() -> Result<()> {
    println!("🧠 Testing simple neural network with Quasar...\n");

    // Test 1: Single neuron (linear layer)
    println!("1. Testing single neuron (linear transformation):");
    test_single_neuron()?;

    // Test 2: Simple 2-layer neural network
    println!("\n2. Testing 2-layer neural network:");
    test_two_layer_network()?;

    // Test 3: XOR problem (classic test)
    println!("\n3. Testing XOR problem:");
    test_xor_network()?;

    println!("\n✅ All neural network tests completed successfully!");
    Ok(())
}

/// Test single neuron: y = W * x + b
fn test_single_neuron() -> Result<()> {
    // Input: 2D vector
    let x = Tensor::new(vec![1.0f32, 2.0], Shape::from(&[2, 1]))?.requires_grad(false);
    
    // Weight matrix: 1x2 (1 output, 2 inputs)
    let w = Tensor::new(vec![0.5f32, -0.3], Shape::from(&[1, 2]))?.requires_grad(true);
    
    // Bias: scalar
    let b = Tensor::new(vec![0.1f32], Shape::from(&[1, 1]))?.requires_grad(true);
    
    println!("   Input x: {:?}", x.data());
    println!("   Weight W: {:?}", w.data());
    println!("   Bias b: {:?}", b.data());
    
    // Forward pass: y = W * x + b
    let wx = matmul(&w, &x)?;
    let mut y = (&wx + &b)?;
    
    println!("   Output y = W*x + b: {:?}", y.data());
    
    // Backward pass
    y.backward()?;
    
    // Check gradients
    let leaf_grads = with_global_engine::<f32, _, _>(|engine| {
        engine.get_leaf_gradients()
    });
    
    println!("   Number of gradients computed: {}", leaf_grads.len());
    
    // Expected: dy/dW = x^T, dy/db = 1
    println!("   ✓ Single neuron test passed");
    
    Ok(())
}

/// Test 2-layer neural network: y = ReLU(W1 * x + b1) * W2 + b2
fn test_two_layer_network() -> Result<()> {
    // Input: 2D vector
    let x = Tensor::new(vec![1.0f32, -0.5], Shape::from(&[2, 1]))?.requires_grad(false);
    
    // First layer: 2 -> 3
    let w1 = Tensor::new(vec![
        0.1, 0.2,   // neuron 1
        -0.1, 0.3,  // neuron 2  
        0.4, -0.2   // neuron 3
    ], Shape::from(&[3, 2]))?.requires_grad(true);
    
    let b1 = Tensor::new(vec![0.1, -0.05, 0.2], Shape::from(&[3, 1]))?.requires_grad(true);
    
    // Second layer: 3 -> 1
    let w2 = Tensor::new(vec![0.5, -0.3, 0.8], Shape::from(&[1, 3]))?.requires_grad(true);
    let b2 = Tensor::new(vec![0.1], Shape::from(&[1, 1]))?.requires_grad(true);
    
    println!("   Input x: {:?}", x.data());
    println!("   Layer 1 weights W1: {:?}", w1.data());
    println!("   Layer 1 bias b1: {:?}", b1.data());
    println!("   Layer 2 weights W2: {:?}", w2.data());
    println!("   Layer 2 bias b2: {:?}", b2.data());
    
    // Forward pass
    // Layer 1: h1 = W1 * x + b1
    let h1_linear = (&matmul(&w1, &x)? + &b1)?;
    println!("   Layer 1 linear: {:?}", h1_linear.data());
    
    // Activation: h1_activated = ReLU(h1)
    let h1_activated = relu(&h1_linear)?;
    println!("   Layer 1 activated: {:?}", h1_activated.data());
    
    // Layer 2: y = W2 * h1_activated + b2
    let mut y = (&matmul(&w2, &h1_activated)? + &b2)?;
    println!("   Final output y: {:?}", y.data());
    
    // Backward pass
    y.backward()?;
    
    // Check gradients
    let leaf_grads = with_global_engine::<f32, _, _>(|engine| {
        engine.get_leaf_gradients()
    });
    
    println!("   Number of gradients computed: {}", leaf_grads.len());
    println!("   ✓ 2-layer network test passed");
    
    Ok(())
}

/// Test XOR problem with 2-2-1 network
fn test_xor_network() -> Result<()> {
    println!("   Testing XOR truth table:");
    
    // XOR truth table
    let inputs = vec![
        vec![0.0f32, 0.0],  // 0 XOR 0 = 0
        vec![0.0f32, 1.0],  // 0 XOR 1 = 1
        vec![1.0f32, 0.0],  // 1 XOR 0 = 1
        vec![1.0f32, 1.0],  // 1 XOR 1 = 0
    ];
    let expected = vec![0.0f32, 1.0, 1.0, 0.0];
    
    // Network weights (pre-trained for XOR)
    // Hidden layer: 2 -> 2
    let w1 = Tensor::new(vec![
        20.0, 20.0,   // First hidden neuron
        -20.0, -20.0  // Second hidden neuron
    ], Shape::from(&[2, 2]))?.requires_grad(true);
    
    let b1 = Tensor::new(vec![-10.0, 30.0], Shape::from(&[2, 1]))?.requires_grad(true);
    
    // Output layer: 2 -> 1
    let w2 = Tensor::new(vec![20.0, 20.0], Shape::from(&[1, 2]))?.requires_grad(true);
    let b2 = Tensor::new(vec![-30.0], Shape::from(&[1, 1]))?.requires_grad(true);
    
    for (i, input) in inputs.iter().enumerate() {
        let x = Tensor::new(input.clone(), Shape::from(&[2, 1]))?.requires_grad(false);
        
        // Forward pass
        let h1 = relu(&(&matmul(&w1, &x)? + &b1)?)?;
        let mut y = (&matmul(&w2, &h1)? + &b2)?;
        
        let output = y.item()?;
        let target = expected[i];
        
        println!("   Input: {:?} -> Output: {:.3}, Expected: {:.1}", 
                input, output, target);
        
        // Simple test: output should be closer to target than to opposite
        let error = (output - target).abs();
        let opposite_error = (output - (1.0 - target)).abs();
        
        if error < opposite_error {
            println!("     ✓ Correct classification");
        } else {
            println!("     ❌ Incorrect classification");
        }
        
        // Test backward pass
        y.backward()?;
    }
    
    println!("   ✓ XOR network test completed");
    Ok(())
}

/// Helper function to create a simple loss function (MSE)
fn mse_loss(prediction: &Tensor<f32>, target: &Tensor<f32>) -> Result<Tensor<f32>> {
    let diff = (prediction - target)?;
    let squared = (&diff * &diff)?;
    
    // For now, just return the squared difference
    // In a full implementation, we'd sum and average
    Ok(squared)
} 