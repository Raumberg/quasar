//! Basic usage example for Quasar autograd engine

use quasar::prelude::*;

fn main() -> Result<()> {
    println!("🌟 Quasar Autograd Engine - Basic Usage Example");
    
    // Create some tensors
    let x = Tensor::new(vec![2.0f32, 3.0, 4.0], Shape::from(&[3]))?;
    let y = Tensor::<f32>::ones(Shape::from(&[3]))?;
    
    println!("x = {}", x);
    println!("y = {}", y);
    
    // Test tensor creation and basic properties
    println!("\nTensor properties:");
    println!("x.shape() = {:?}", x.shape().dims());
    println!("x.ndim() = {}", x.ndim());
    println!("x.numel() = {}", x.numel());
    
    // Test gradient computation setup
    let x_grad = x.requires_grad(true);
    println!("\nx with gradients enabled:");
    println!("x_grad.requires_grad = {}", x_grad.requires_grad_flag());
    
    // Test reshape
    let x_reshaped = x_grad.reshape(Shape::from(&[1, 3]))?;
    println!("\nReshaped tensor:");
    println!("x_reshaped.shape() = {:?}", x_reshaped.shape().dims());
    
    // Test scalar extraction
    let scalar = Tensor::new(vec![42.0f32], Shape::from(&[1]))?;
    println!("\nScalar tensor:");
    println!("scalar.item() = {}", scalar.item()?);
    
    println!("\n✅ Basic operations completed successfully!");
    
    Ok(())
} 