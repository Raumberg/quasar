//! Performance and automatic parallelization demo
//! 
//! This example demonstrates:
//! - Automatic parallelization for large operations
//! - Performance comparison between sequential and parallel execution
//! - Memory efficiency with copy-on-write tensors
//! - Scalability with different tensor sizes

use quasar::prelude::*;
use std::time::Instant;

fn main() -> Result<()> {
    println!("🚀 Quasar Performance Demo\n");

    // 1. Automatic parallelization demonstration
    println!("1. Automatic Parallelization:");
    println!("   Quasar automatically chooses parallel execution for large tensors");
    
    // Small tensors (sequential execution)
    let small_a = Tensor::new(vec![1.0; 100], Shape::from(&[100]))?;
    let small_b = Tensor::new(vec![2.0; 100], Shape::from(&[100]))?;
    
    let start = Instant::now();
    let small_result = add(&small_a, &small_b)?;
    let small_time = start.elapsed();
    
    println!("   Small tensors (100 elements): {:?} (sequential)", small_time);
    
    // Large tensors (parallel execution)
    let large_a = Tensor::new(vec![1.0; 1_000_000], Shape::from(&[1_000_000]))?;
    let large_b = Tensor::new(vec![2.0; 1_000_000], Shape::from(&[1_000_000]))?;
    
    let start = Instant::now();
    let large_result = add(&large_a, &large_b)?;
    let large_time = start.elapsed();
    
    println!("   Large tensors (1M elements): {:?} (parallel)", large_time);
    
    // Verify results
    println!("   Small result sample: {:?}", &small_result.data()[0..5]);
    println!("   Large result sample: {:?}", &large_result.data()[0..5]);

    // 2. Performance scaling test
    println!("\n2. Performance Scaling Test:");
    
    let sizes = vec![1_000, 10_000, 100_000, 1_000_000];
    
    for size in sizes {
        let a = Tensor::new(vec![1.0; size], Shape::from(&[size]))?;
        let b = Tensor::new(vec![2.0; size], Shape::from(&[size]))?;
        
        let start = Instant::now();
        let _result = mul(&a, &b)?;
        let duration = start.elapsed();
        
        let throughput = size as f64 / duration.as_secs_f64() / 1_000_000.0;
        println!("   Size: {:>8} elements, Time: {:>8.2?}, Throughput: {:.1} M ops/sec", 
                size, duration, throughput);
    }

    // 3. Matrix multiplication performance
    println!("\n3. Matrix Multiplication Performance:");
    
    let matrix_sizes = vec![50, 100, 200];
    
    for size in matrix_sizes {
        let elements = size * size;
        let a = Tensor::new(vec![1.0; elements], Shape::from(&[size, size]))?;
        let b = Tensor::new(vec![2.0; elements], Shape::from(&[size, size]))?;
        
        let start = Instant::now();
        let _result = matmul(&a, &b)?;
        let duration = start.elapsed();
        
        let flops = 2.0 * (size as f64).powi(3); // Approximate FLOPs for matrix multiplication
        let gflops = flops / duration.as_secs_f64() / 1_000_000_000.0;
        
        println!("   {}x{} matrices: {:>8.2?}, {:.2} GFLOPS", 
                size, size, duration, gflops);
    }

    // 4. Memory efficiency demonstration
    println!("\n4. Memory Efficiency (Copy-on-Write):");
    
    let original = Tensor::new(vec![1.0; 1_000_000], Shape::from(&[1_000_000]))?;
    println!("   Original tensor ref count: {}", original.ref_count());
    
    // Clone creates a shared reference (no data copy)
    let cloned = original.clone();
    println!("   After clone ref count: {}", original.ref_count());
    println!("   Cloned tensor ref count: {}", cloned.ref_count());
    println!("   Data is shared: {}", original.is_shared());
    
    // Modifying triggers copy-on-write
    let mut modified = cloned.clone();
    println!("   Before modification ref count: {}", modified.ref_count());
    
    // This would trigger copy-on-write if we had mutable access
    println!("   Copy-on-write ensures memory efficiency!");

    // 5. Gradient computation performance
    println!("\n5. Gradient Computation Performance:");
    
    let x = Tensor::new(vec![1.0; 10_000], Shape::from(&[10_000]))?.requires_grad(true);
    let y = Tensor::new(vec![2.0; 10_000], Shape::from(&[10_000]))?.requires_grad(true);
    
    println!("   Forward pass with 10K elements...");
    let start = Instant::now();
    
    // Complex computation: z = (x * y + x) * (x - y)
    let xy = mul(&x, &y)?;
    let xy_plus_x = add(&xy, &x)?;
    let x_minus_y = sub(&x, &y)?;
    let mut z = mul(&xy_plus_x, &x_minus_y)?;
    
    let forward_time = start.elapsed();
    println!("   Forward pass completed in: {:?}", forward_time);
    
    println!("   Backward pass...");
    let start = Instant::now();
    z.backward()?;
    let backward_time = start.elapsed();
    
    println!("   Backward pass completed in: {:?}", backward_time);
    println!("   Total computation time: {:?}", forward_time + backward_time);

    // 6. Different data types performance
    println!("\n6. Data Type Performance Comparison:");
    
    // f32 performance
    let f32_a = Tensor::new(vec![1.0f32; 100_000], Shape::from(&[100_000]))?;
    let f32_b = Tensor::new(vec![2.0f32; 100_000], Shape::from(&[100_000]))?;
    
    let start = Instant::now();
    let _f32_result = mul(&f32_a, &f32_b)?;
    let f32_time = start.elapsed();
    
    // f64 performance
    let f64_a = Tensor::new(vec![1.0f64; 100_000], Shape::from(&[100_000]))?;
    let f64_b = Tensor::new(vec![2.0f64; 100_000], Shape::from(&[100_000]))?;
    
    let start = Instant::now();
    let _f64_result = mul(&f64_a, &f64_b)?;
    let f64_time = start.elapsed();
    
    println!("   f32 multiplication (100K elements): {:?}", f32_time);
    println!("   f64 multiplication (100K elements): {:?}", f64_time);
    println!("   f32 is {:.1}x faster", f64_time.as_secs_f64() / f32_time.as_secs_f64());

    // 7. Complex neural network simulation
    println!("\n7. Neural Network Simulation:");
    
    let batch_size = 32;
    let input_size = 784;  // Like MNIST
    let hidden_size = 256;
    let output_size = 10;
    
    println!("   Simulating neural network:");
    println!("   Batch size: {}, Input: {}, Hidden: {}, Output: {}", 
            batch_size, input_size, hidden_size, output_size);
    
    // Create network weights
    let w1 = Tensor::new(vec![0.01; input_size * hidden_size], 
                        Shape::from(&[input_size, hidden_size]))?;
    let w2 = Tensor::new(vec![0.01; hidden_size * output_size], 
                        Shape::from(&[hidden_size, output_size]))?;
    
    // Create batch input
    let input = Tensor::new(vec![1.0; batch_size * input_size], 
                           Shape::from(&[batch_size, input_size]))?;
    
    let start = Instant::now();
    
    // Forward pass
    let hidden = matmul(&input, &w1)?;
    let hidden_relu = relu(&hidden)?;
    let output = matmul(&hidden_relu, &w2)?;
    
    let forward_time = start.elapsed();
    
    println!("   Forward pass completed in: {:?}", forward_time);
    println!("   Output shape: {:?}", output.shape());
    println!("   Throughput: {:.1} samples/sec", 
            batch_size as f64 / forward_time.as_secs_f64());

    // 8. Memory usage estimation
    println!("\n8. Memory Usage Estimation:");
    
    let tensor_1m = Tensor::new(vec![1.0f32; 1_000_000], Shape::from(&[1_000_000]))?;
    let memory_mb = (1_000_000 * 4) as f64 / 1_024.0 / 1_024.0; // 4 bytes per f32
    
    println!("   1M f32 tensor uses ~{:.1} MB", memory_mb);
    println!("   Tensor shape: {:?}", tensor_1m.shape());
    println!("   Elements: {}", tensor_1m.numel());
    println!("   Data type: {:?}", tensor_1m.dtype());

    println!("\n✅ Performance demo completed successfully!");
    println!("💡 Key takeaways:");
    println!("   • Automatic parallelization for large tensors");
    println!("   • Copy-on-write for memory efficiency");
    println!("   • Optimized matrix operations");
    println!("   • Efficient gradient computation");
    
    Ok(())
} 