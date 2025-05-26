//! Simple autograd example

use quasar::prelude::*;
use quasar::autograd::{AddOp, AutogradOp};

fn main() -> Result<()> {
    println!("🧠 Quasar Simple Autograd Example");
    
    // Create tensors that require gradients
    let x = Tensor::new(vec![2.0f32], Shape::from(&[1]))?.requires_grad(true);
    let y = Tensor::new(vec![3.0f32], Shape::from(&[1]))?.requires_grad(true);
    
    println!("\nInput tensors:");
    println!("x = {} (requires_grad={})", x, x.requires_grad_flag());
    println!("y = {} (requires_grad={})", y, y.requires_grad_flag());
    
    // Test forward pass with AddOp
    println!("\n🔄 Forward pass:");
    let add_op = AddOp;
    let z = add_op.forward(&[&x, &y])?;
    println!("z = x + y = {}", z);
    
    // Test backward pass
    println!("\n⬅️ Backward pass:");
    let grad_output = Tensor::ones(Shape::from(&[1]))?; // dL/dz = 1
    let gradients = add_op.backward(&grad_output, &[&x, &y])?;
    
    println!("dL/dx = {}", gradients[0]);
    println!("dL/dy = {}", gradients[1]);
    
    // Expected: both gradients should be [1.0] for addition
    assert_eq!(gradients[0].item()?, 1.0);
    assert_eq!(gradients[1].item()?, 1.0);
    
    println!("\n✅ Simple autograd test passed!");
    
    Ok(())
} 