//! Example demonstrating basic arithmetic operations

use quasar::prelude::*;

fn main() -> Result<()> {
    println!("🧮 Quasar Arithmetic Operations Example");
    
    // Create test tensors
    let a = Tensor::new(vec![1.0f32, 2.0, 3.0], Shape::from(&[3]))?;
    let b = Tensor::new(vec![4.0f32, 5.0, 6.0], Shape::from(&[3]))?;
    
    println!("\nInput tensors:");
    println!("a = {}", a);
    println!("b = {}", b);
    
    // Test addition
    println!("\n➕ Addition:");
    let c = (&a + &b)?;
    println!("a + b = {}", c);
    
    // Test subtraction
    println!("\n➖ Subtraction:");
    let d = (&b - &a)?;
    println!("b - a = {}", d);
    
    // Test multiplication
    println!("\n✖️ Multiplication:");
    let e = (&a * &b)?;
    println!("a * b = {}", e);
    
    // Test division
    println!("\n➗ Division:");
    let f = (&b / &a)?;
    println!("b / a = {}", f);
    
    // Test chained operations
    println!("\n🔗 Chained operations:");
    let temp = (&a + &b)?;
    let g = (&temp * &a)?;
    println!("(a + b) * a = {}", g);
    
    // Test with different shapes (should fail)
    println!("\n❌ Testing shape mismatch:");
    let x = Tensor::new(vec![1.0f32, 2.0], Shape::from(&[2]))?;
    match &a + &x {
        Ok(_) => println!("Unexpected success!"),
        Err(e) => println!("Expected error: {}", e),
    }
    
    println!("\n✅ Arithmetic operations test completed!");
    
    Ok(())
} 