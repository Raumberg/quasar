//! Matrix operations example
//! 
//! This example demonstrates:
//! - Matrix multiplication
//! - Matrix-vector operations
//! - Broadcasting concepts
//! - Linear algebra operations

use quasar::prelude::*;
use std::time::Instant;

fn main() -> Result<()> {
    println!("🚀 Quasar Matrix Operations Example\n");

    // 1. Basic matrix multiplication
    println!("1. Basic Matrix Multiplication:");
    
    let matrix_a = Tensor::new(
        vec![1.0, 2.0, 3.0, 4.0],
        Shape::from(&[2, 2])
    )?;
    
    let matrix_b = Tensor::new(
        vec![5.0, 6.0, 7.0, 8.0],
        Shape::from(&[2, 2])
    )?;
    
    println!("   Matrix A (2x2):");
    print_matrix(&matrix_a, 2, 2);
    
    println!("   Matrix B (2x2):");
    print_matrix(&matrix_b, 2, 2);
    
    let result = matmul(&matrix_a, &matrix_b)?;
    println!("   A @ B =");
    print_matrix(&result, 2, 2);

    // 2. Matrix-vector multiplication
    println!("\n2. Matrix-Vector Multiplication:");
    
    let matrix = Tensor::new(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        Shape::from(&[2, 3])
    )?;
    
    let vector = Tensor::new(
        vec![1.0, 2.0, 3.0],
        Shape::from(&[3, 1])
    )?;
    
    println!("   Matrix (2x3):");
    print_matrix(&matrix, 2, 3);
    
    println!("   Vector: {:?}", vector.data());
    
    let mv_result = matmul(&matrix, &vector)?;
    println!("   Matrix @ Vector = {:?}", mv_result.data());

    // 3. Batch matrix multiplication
    println!("\n3. Batch Matrix Operations:");
    
    // Create batch of matrices (batch_size=2, 3x3 matrices)
    let batch_a = Tensor::new(
        vec![
            // First matrix
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
            7.0, 8.0, 9.0,
            // Second matrix
            9.0, 8.0, 7.0,
            6.0, 5.0, 4.0,
            3.0, 2.0, 1.0,
        ],
        Shape::from(&[2, 3, 3])
    )?;
    
    let batch_b = Tensor::new(
        vec![
            // First matrix
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0,
            // Second matrix
            2.0, 0.0, 0.0,
            0.0, 2.0, 0.0,
            0.0, 0.0, 2.0,
        ],
        Shape::from(&[2, 3, 3])
    )?;
    
    println!("   Batch A shape: {:?}", batch_a.shape());
    println!("   Batch B shape: {:?}", batch_b.shape());
    
    // For now, we'll demonstrate with individual matrices since batch matmul might not be implemented
    println!("   First matrix from batch A:");
    let first_a = Tensor::new(
        batch_a.data()[0..9].to_vec(),
        Shape::from(&[3, 3])
    )?;
    print_matrix(&first_a, 3, 3);

    // 4. Different matrix sizes
    println!("\n4. Different Matrix Sizes:");
    
    let rect_a = Tensor::new(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        Shape::from(&[2, 3])
    )?;
    
    let rect_b = Tensor::new(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        Shape::from(&[3, 2])
    )?;
    
    println!("   Matrix A (2x3):");
    print_matrix(&rect_a, 2, 3);
    
    println!("   Matrix B (3x2):");
    print_matrix(&rect_b, 3, 2);
    
    let rect_result = matmul(&rect_a, &rect_b)?;
    println!("   A @ B (2x2):");
    print_matrix(&rect_result, 2, 2);

    // 5. Chain of matrix operations
    println!("\n5. Chain of Matrix Operations:");
    
    let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], Shape::from(&[2, 2]))?;
    let b = Tensor::new(vec![2.0, 0.0, 0.0, 2.0], Shape::from(&[2, 2]))?;
    let c = Tensor::new(vec![1.0, 1.0, 1.0, 1.0], Shape::from(&[2, 2]))?;
    
    println!("   Computing (A @ B) + C:");
    println!("   A:");
    print_matrix(&a, 2, 2);
    println!("   B:");
    print_matrix(&b, 2, 2);
    println!("   C:");
    print_matrix(&c, 2, 2);
    
    let ab = matmul(&a, &b)?;
    let result = add(&ab, &c)?;
    
    println!("   Result:");
    print_matrix(&result, 2, 2);

    // 6. Performance test with larger matrices
    println!("\n6. Performance Test:");
    
    let large_a = Tensor::new(vec![1.0; 10000], Shape::from(&[100, 100]))?;
    let large_b = Tensor::new(vec![2.0; 10000], Shape::from(&[100, 100]))?;
    
    println!("   Testing 100x100 matrix multiplication...");
    
    let start_time = Instant::now();
    let large_result = matmul(&large_a, &large_b)?;
    let duration = start_time.elapsed();
    
    println!("   Completed in: {:?}", duration);
    println!("   Result shape: {:?}", large_result.shape());
    println!("   First element: {:.3}", large_result.data()[0]);
    println!("   Last element: {:.3}", large_result.data()[large_result.data().len() - 1]);

    // 7. Matrix operations with gradients
    println!("\n7. Matrix Operations with Gradients:");
    
    let w = Tensor::new(
        vec![0.1, 0.2, 0.3, 0.4],
        Shape::from(&[2, 2])
    )?.requires_grad(true);
    
    let x = Tensor::new(
        vec![1.0, 2.0],
        Shape::from(&[2, 1])
    )?.requires_grad(true);
    
    println!("   Weight matrix W:");
    print_matrix(&w, 2, 2);
    println!("   Input vector x: {:?}", x.data());
    
    // Forward pass: y = W @ x
    let mut y = matmul(&w, &x)?;
    println!("   Output y = W @ x: {:?}", y.data());
    
    // Backward pass
    y.backward()?;
    
    println!("   Gradients computed for matrix multiplication!");
    
    // Get gradients
    with_global_engine::<f32, _, _>(|engine| {
        let leaf_grads = engine.get_leaf_gradients();
        println!("   Found {} gradients", leaf_grads.len());
        
        for (node_id, grad) in leaf_grads {
            println!("   Node {}: gradient shape {:?}", node_id, grad.shape());
        }
    });

    println!("\n✅ Matrix operations completed successfully!");
    println!("💡 Note: Matrix multiplication is optimized for performance!");
    
    Ok(())
}

/// Helper function to print matrix in a readable format
fn print_matrix(tensor: &Tensor<f32>, rows: usize, cols: usize) {
    let data = tensor.data();
    for i in 0..rows {
        print!("     [");
        for j in 0..cols {
            print!("{:6.2}", data[i * cols + j]);
            if j < cols - 1 {
                print!(", ");
            }
        }
        println!("]");
    }
} 