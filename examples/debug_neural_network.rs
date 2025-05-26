//! Debug neural network computations

use quasar::prelude::*;

fn main() -> Result<()> {
    println!("🔍 Debugging neural network computations...\n");

    // Test 1: Manual matrix multiplication verification
    println!("1. Testing matrix multiplication manually:");
    test_matmul_manual()?;

    // Test 2: Step-by-step single neuron
    println!("\n2. Step-by-step single neuron:");
    test_single_neuron_debug()?;

    // Test 3: ReLU function verification
    println!("\n3. Testing ReLU function:");
    test_relu_debug()?;

    Ok(())
}

fn test_matmul_manual() -> Result<()> {
    // Simple 2x2 * 2x1 multiplication
    let a = Tensor::new(vec![1.0f32, 2.0, 3.0, 4.0], Shape::from(&[2, 2]))?;
    let b = Tensor::new(vec![5.0f32, 6.0], Shape::from(&[2, 1]))?;
    
    println!("   Matrix A (2x2): {:?}", a.data());
    println!("   Matrix B (2x1): {:?}", b.data());
    
    let result = matmul(&a, &b)?;
    println!("   Result A*B: {:?}", result.data());
    
    // Manual calculation:
    // [1 2] * [5] = [1*5 + 2*6] = [17]
    // [3 4]   [6]   [3*5 + 4*6]   [39]
    
    println!("   Expected: [17.0, 39.0]");
    
    let expected = vec![17.0f32, 39.0];
    let actual = result.data();
    
    for (i, (&actual_val, &expected_val)) in actual.iter().zip(expected.iter()).enumerate() {
        let diff = (actual_val - expected_val).abs();
        println!("   Element {}: actual={:.3}, expected={:.3}, diff={:.6}", 
                i, actual_val, expected_val, diff);
        
        if diff < 1e-5 {
            println!("     ✓ Match");
        } else {
            println!("     ❌ Mismatch");
        }
    }
    
    Ok(())
}

fn test_single_neuron_debug() -> Result<()> {
    // Very simple case: 1x1 matrices
    println!("   Testing 1x1 case:");
    let w = Tensor::new(vec![2.0f32], Shape::from(&[1, 1]))?;
    let x = Tensor::new(vec![3.0f32], Shape::from(&[1, 1]))?;
    let b = Tensor::new(vec![1.0f32], Shape::from(&[1, 1]))?;
    
    println!("   W: {:?}", w.data());
    println!("   x: {:?}", x.data());
    println!("   b: {:?}", b.data());
    
    let wx = matmul(&w, &x)?;
    println!("   W*x: {:?} (expected: [6.0])", wx.data());
    
    let result = (&wx + &b)?;
    println!("   W*x + b: {:?} (expected: [7.0])", result.data());
    
    // Test 2x1 case
    println!("\n   Testing 2x1 case:");
    let w2 = Tensor::new(vec![0.5f32, -0.3], Shape::from(&[1, 2]))?;
    let x2 = Tensor::new(vec![1.0f32, 2.0], Shape::from(&[2, 1]))?;
    let b2 = Tensor::new(vec![0.1f32], Shape::from(&[1, 1]))?;
    
    println!("   W: {:?}", w2.data());
    println!("   x: {:?}", x2.data());
    println!("   b: {:?}", b2.data());
    
    let wx2 = matmul(&w2, &x2)?;
    println!("   W*x: {:?}", wx2.data());
    
    // Manual calculation: [0.5, -0.3] * [1.0, 2.0]^T = 0.5*1 + (-0.3)*2 = 0.5 - 0.6 = -0.1
    println!("   Expected W*x: [-0.1]");
    
    let result2 = (&wx2 + &b2)?;
    println!("   W*x + b: {:?}", result2.data());
    println!("   Expected W*x + b: [0.0]");
    
    Ok(())
}

fn test_relu_debug() -> Result<()> {
    let input = Tensor::new(vec![-2.0f32, -1.0, 0.0, 1.0, 2.0], Shape::from(&[5]))?;
    println!("   Input: {:?}", input.data());
    
    let output = relu(&input)?;
    println!("   ReLU output: {:?}", output.data());
    println!("   Expected: [0.0, 0.0, 0.0, 1.0, 2.0]");
    
    // Test with requires_grad
    let input_grad = input.requires_grad(true);
    let mut output_grad = relu(&input_grad)?;
    
    println!("   Testing backward pass...");
    output_grad.backward()?;
    
    let leaf_grads = with_global_engine::<f32, _, _>(|engine| {
        engine.get_leaf_gradients()
    });
    
    println!("   Number of gradients: {}", leaf_grads.len());
    
    Ok(())
} 