//! Test example for architectural improvements

use quasar::prelude::*;

fn main() -> Result<()> {
    println!("🌟 Testing Quasar architectural improvements...\n");

    // Test 1: Copy-on-write optimization
    println!("1. Testing copy-on-write optimization:");
    let x = Tensor::new(vec![1.0f32, 2.0, 3.0], Shape::from(&[3]))?;
    println!("   Original tensor ref count: {}", x.ref_count());
    
    let y = x.clone(); // Should share data
    println!("   After clone ref count: {}", x.ref_count());
    println!("   Clone ref count: {}", y.ref_count());
    println!("   Data is shared: {}", x.is_shared());
    
    // Test 2: Basic autograd with new architecture
    println!("\n2. Testing autograd with new architecture:");
    let a = Tensor::new(vec![2.0f32, 3.0], Shape::from(&[2]))?.requires_grad(true);
    let b = Tensor::new(vec![4.0f32, 5.0], Shape::from(&[2]))?.requires_grad(true);
    
    // Perform operations: result = (a + b) * (a - b) = a² - b²
    let sum = (&a + &b)?;
    let diff = (&a - &b)?;
    let mut result = (&sum * &diff)?;
    
    println!("   Input a: {:?}", a.data());
    println!("   Input b: {:?}", b.data());
    println!("   Result: {:?}", result.data());
    
    // Compute gradients
    result.backward()?;
    
    // Access gradients through global autograd engine
    let leaf_grads = with_global_engine::<f32, _, _>(|engine| {
        engine.get_leaf_gradients()
    });
    
    println!("   Number of leaf gradients: {}", leaf_grads.len());
    
    // Test 3: Graph statistics
    println!("\n3. Testing graph statistics:");
    let stats = with_global_engine::<f32, _, _>(|engine| {
        engine.stats()
    });
    
    println!("   Graph node count: {}", stats.node_count);
    println!("   Autograd enabled: {}", stats.enabled);
    
    // Test 4: Memory efficiency
    println!("\n4. Testing memory efficiency:");
    let large_tensor: Tensor<f32> = Tensor::zeros(Shape::from(&[1000, 1000]))?;
    println!("   Created large tensor: {}x{}", 1000, 1000);
    println!("   Ref count: {}", large_tensor.ref_count());
    
    let shared_tensor = large_tensor.clone();
    println!("   After sharing: ref count = {}", large_tensor.ref_count());
    println!("   Memory is shared: {}", large_tensor.is_shared());
    
    println!("\n✅ All tests completed successfully!");
    
    Ok(())
} 