//! Basic tensor operations example
//! 
//! This example demonstrates:
//! - Creating tensors with different shapes
//! - Basic arithmetic operations (add, sub, mul, div)
//! - Tensor properties and methods
//! - Error handling

use quasar::prelude::*;

fn main() -> Result<()> {
    println!("🚀 Quasar Basic Operations Example\n");

    // 1. Creating tensors
    println!("1. Creating Tensors:");
    
    // Scalar tensor
    let scalar = Tensor::from_scalar(42.0f32);
    println!("   Scalar: {}", scalar);
    
    // Vector tensor
    let vector = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], Shape::from(&[4]))?;
    println!("   Vector: {}", vector);
    
    // Matrix tensor
    let matrix = Tensor::new(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 
        Shape::from(&[2, 3])
    )?;
    println!("   Matrix: {}", matrix);
    
    // Special tensors
    let zeros = Tensor::<f32>::zeros(Shape::from(&[2, 2]))?;
    let ones = Tensor::<f32>::ones(Shape::from(&[2, 2]))?;
    println!("   Zeros 2x2: {:?}", zeros.data());
    println!("   Ones 2x2: {:?}", ones.data());

    // 2. Basic arithmetic operations
    println!("\n2. Arithmetic Operations:");
    
    let a = Tensor::new(vec![1.0, 2.0, 3.0], Shape::from(&[3]))?;
    let b = Tensor::new(vec![4.0, 5.0, 6.0], Shape::from(&[3]))?;
    
    println!("   a = {:?}", a.data());
    println!("   b = {:?}", b.data());
    
    // Addition
    let sum = add(&a, &b)?;
    println!("   a + b = {:?}", sum.data());
    
    // Subtraction
    let diff = sub(&a, &b)?;
    println!("   a - b = {:?}", diff.data());
    
    // Multiplication
    let product = mul(&a, &b)?;
    println!("   a * b = {:?}", product.data());
    
    // Division
    let quotient = div(&a, &b)?;
    println!("   a / b = {:?}", quotient.data());

    // 3. Operator overloading
    println!("\n3. Operator Overloading:");
    
    let result1 = (&a + &b)?;
    let result2 = (&a * &b)?;
    println!("   Using + operator: {:?}", result1.data());
    println!("   Using * operator: {:?}", result2.data());

    // 4. Tensor properties
    println!("\n4. Tensor Properties:");
    
    let tensor_3d = Tensor::new(
        vec![1.0; 24], 
        Shape::from(&[2, 3, 4])
    )?;
    
    println!("   Shape: {:?}", tensor_3d.shape());
    println!("   Dimensions: {}", tensor_3d.ndim());
    println!("   Total elements: {}", tensor_3d.numel());
    println!("   Data type: {:?}", tensor_3d.dtype());
    println!("   Is scalar: {}", tensor_3d.is_scalar());

    // 5. Reshaping
    println!("\n5. Reshaping:");
    
    let original = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], Shape::from(&[6]))?;
    println!("   Original shape: {:?}", original.shape());
    
    let reshaped = original.reshape(Shape::from(&[2, 3]))?;
    println!("   Reshaped to 2x3: {:?}", reshaped.shape());
    println!("   Data: {:?}", reshaped.data());

    // 6. Error handling
    println!("\n6. Error Handling:");
    
    let small = Tensor::new(vec![1.0, 2.0], Shape::from(&[2]))?;
    let large = Tensor::new(vec![1.0, 2.0, 3.0], Shape::from(&[3]))?;
    
    match add(&small, &large) {
        Ok(_) => println!("   Unexpected success!"),
        Err(e) => println!("   Expected error: {}", e),
    }

    println!("\n✅ Basic operations completed successfully!");
    Ok(())
} 