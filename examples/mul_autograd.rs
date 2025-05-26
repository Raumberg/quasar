//! Multiplication autograd example

use quasar::prelude::*;

fn main() -> Result<()> {
    println!("🔢 Quasar Multiplication Autograd Example");
    
    // Create tensors that require gradients
    let x = Tensor::new(vec![2.0f32], Shape::from(&[1]))?.requires_grad(true);
    let y = Tensor::new(vec![3.0f32], Shape::from(&[1]))?.requires_grad(true);
    
    println!("\nInput tensors:");
    println!("x = {} (requires_grad={})", x, x.requires_grad_flag());
    println!("y = {} (requires_grad={})", y, y.requires_grad_flag());
    
    // Test forward pass with MulOp
    println!("\n🔄 Forward pass:");
    let mul_op = MulOp;
    let z = mul_op.forward(&[&x, &y])?;
    println!("z = x * y = {}", z);
    
    // Test backward pass
    println!("\n⬅️ Backward pass:");
    let grad_output = Tensor::ones(Shape::from(&[1]))?; // dL/dz = 1
    let gradients = mul_op.backward(&grad_output, &[&x, &y])?;
    
    println!("dL/dx = {}", gradients[0]);
    println!("dL/dy = {}", gradients[1]);
    
    // Expected: dL/dx = y = 3.0, dL/dy = x = 2.0
    assert_eq!(gradients[0].item()?, 3.0); // dL/dx = y
    assert_eq!(gradients[1].item()?, 2.0); // dL/dy = x
    
    println!("\n✅ Multiplication autograd test passed!");
    
    // Test more complex example: z = x * y + x
    println!("\n🧮 Complex example: z = x * y + x");
    let xy = mul_op.forward(&[&x, &y])?;
    let add_op = AddOp;
    let z_complex = add_op.forward(&[&xy, &x])?;
    println!("z = x * y + x = {} * {} + {} = {}", 
             x.item()?, y.item()?, x.item()?, z_complex.item()?);
    
    // Manual gradient computation for z = x * y + x
    // dz/dx = y + 1, dz/dy = x
    let expected_dx = y.item()? + 1.0; // 3 + 1 = 4
    let expected_dy = x.item()?;       // 2
    
    println!("Expected gradients: dz/dx = {}, dz/dy = {}", expected_dx, expected_dy);
    
    Ok(())
} 