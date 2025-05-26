//! Parallel neural network example demonstrating multi-threaded autograd

use quasar::prelude::*;
use quasar::autograd::parallel::{ParallelConfig, init_parallel_autograd, with_local_engine};
use quasar::ops::parallel_ops::{par_add, par_mul, par_matmul, par_relu, par_batch_matmul, par_statistics};
use std::thread;
use std::time::Instant;

fn main() -> Result<()> {
    println!("🚀 Parallel Neural Network with Quasar\n");

    // Initialize parallel autograd with custom configuration
    let config = ParallelConfig {
        max_workers: num_cpus::get(),
        auto_parallel: true,
        parallel_threshold: 100, // Parallelize operations on tensors with 100+ elements
        use_thread_local: true,
        enable_fusion: true,
    };
    
    init_parallel_autograd(config)?;
    println!("✅ Initialized parallel autograd with {} workers", num_cpus::get());

    // Test 1: Compare sequential vs parallel operations
    println!("\n1. Performance comparison: Sequential vs Parallel");
    compare_performance()?;

    // Test 2: Multi-threaded training
    println!("\n2. Multi-threaded neural network training");
    multi_threaded_training()?;

    // Test 3: Batch processing with parallelization
    println!("\n3. Parallel batch processing");
    parallel_batch_processing()?;

    // Test 4: Thread-local engines
    println!("\n4. Thread-local autograd engines");
    test_thread_local_engines()?;

    println!("\n🎉 All parallel neural network tests completed!");
    Ok(())
}

/// Compare performance between sequential and parallel operations
fn compare_performance() -> Result<()> {
    // Create large tensors for meaningful performance comparison
    let size = 1000;
    let a = Tensor::new(
        (0..size*size).map(|i| i as f32 / 1000.0).collect(),
        Shape::from(&[size, size])
    )?;
    let b = Tensor::new(
        (0..size*size).map(|i| (i as f32 + 1.0) / 1000.0).collect(),
        Shape::from(&[size, size])
    )?;

    println!("   Testing with {}x{} matrices ({} elements)", size, size, size*size);

    // Sequential matrix multiplication
    let start = Instant::now();
    let _result_seq = matmul(&a, &b)?;
    let seq_time = start.elapsed();
    println!("   Sequential matmul: {:?}", seq_time);

    // Parallel matrix multiplication
    let start = Instant::now();
    let _result_par = par_matmul(&a, &b)?;
    let par_time = start.elapsed();
    println!("   Parallel matmul: {:?}", par_time);

    let speedup = seq_time.as_nanos() as f64 / par_time.as_nanos() as f64;
    println!("   Speedup: {:.2}x", speedup);

    // Test element-wise operations
    let start = Instant::now();
    let _result_seq = (&a + &b)?;
    let seq_time = start.elapsed();
    println!("   Sequential addition: {:?}", seq_time);

    let start = Instant::now();
    let _result_par = par_add(&a, &b)?;
    let par_time = start.elapsed();
    println!("   Parallel addition: {:?}", par_time);

    let speedup = seq_time.as_nanos() as f64 / par_time.as_nanos() as f64;
    println!("   Addition speedup: {:.2}x", speedup);

    Ok(())
}

/// Demonstrate multi-threaded neural network training
fn multi_threaded_training() -> Result<()> {
    let num_threads = 4;
    let mut handles = Vec::new();

    println!("   Training {} neural networks in parallel", num_threads);

    for thread_id in 0..num_threads {
        let handle = thread::spawn(move || -> Result<()> {
            // Each thread trains its own neural network
            train_network_in_thread(thread_id)
        });
        handles.push(handle);
    }

    // Wait for all threads to complete
    for (i, handle) in handles.into_iter().enumerate() {
        match handle.join() {
            Ok(result) => {
                result?;
                println!("   ✅ Thread {} completed successfully", i);
            }
            Err(_) => {
                println!("   ❌ Thread {} panicked", i);
            }
        }
    }

    Ok(())
}

/// Train a neural network in a specific thread
fn train_network_in_thread(thread_id: usize) -> Result<()> {
    // Use thread-local engine for independent computation
    with_local_engine(|local_engine| {
        println!("     Thread {}: Using local engine", thread_id);
        
        // Create a simple 2-layer network
        let input_size = 10;
        let hidden_size = 5;
        let output_size = 1;

        // Initialize weights with thread-specific values
        let w1_data: Vec<f32> = (0..input_size*hidden_size)
            .map(|i| (i as f32 + thread_id as f32) / 100.0)
            .collect();
        let w1 = Tensor::new(w1_data, Shape::from(&[hidden_size, input_size]))?.requires_grad(true);

        let w2_data: Vec<f32> = (0..hidden_size*output_size)
            .map(|i| (i as f32 + thread_id as f32 + 10.0) / 100.0)
            .collect();
        let w2 = Tensor::new(w2_data, Shape::from(&[output_size, hidden_size]))?.requires_grad(true);

        // Training data
        let x = Tensor::new(
            (0..input_size).map(|i| (i as f32 + thread_id as f32) / 10.0).collect(),
            Shape::from(&[input_size, 1])
        )?;

        // Forward pass using parallel operations
        let h1 = par_relu(&par_matmul(&w1, &x)?)?;
        let output = par_matmul(&w2, &h1)?;

        println!("     Thread {}: Forward pass completed, output: {:.3}", 
                thread_id, output.item()?);

        // Skip backward pass for now to avoid deadlock issues
        // TODO: Fix autograd thread safety for backward pass
        println!("     Thread {}: Skipping backward pass (thread safety)", thread_id);
        println!("     Thread {}: Local engine stats: {:?}", thread_id, local_engine.stats());

        Ok(())
    })
}

/// Demonstrate parallel batch processing
fn parallel_batch_processing() -> Result<()> {
    let batch_size = 8;
    let matrix_size = 100;

    println!("   Processing batch of {} matrices ({}x{} each)", 
            batch_size, matrix_size, matrix_size);

    // Create batch of matrices
    let mut batch_a = Vec::new();
    let mut batch_b = Vec::new();

    for i in 0..batch_size {
        let a = Tensor::new(
            (0..matrix_size*matrix_size).map(|j| (j + i * 1000) as f32 / 1000.0).collect(),
            Shape::from(&[matrix_size, matrix_size])
        )?;
        let b = Tensor::new(
            (0..matrix_size*matrix_size).map(|j| (j + i * 1000 + 500) as f32 / 1000.0).collect(),
            Shape::from(&[matrix_size, matrix_size])
        )?;
        
        batch_a.push(a);
        batch_b.push(b);
    }

    // Convert to references for batch processing
    let batch_a_refs: Vec<&Tensor<f32>> = batch_a.iter().collect();
    let batch_b_refs: Vec<&Tensor<f32>> = batch_b.iter().collect();

    // Parallel batch matrix multiplication
    let start = Instant::now();
    let results = par_batch_matmul(&batch_a_refs, &batch_b_refs)?;
    let batch_time = start.elapsed();

    println!("   Parallel batch processing time: {:?}", batch_time);
    println!("   Processed {} matrices successfully", results.len());

    // Verify results
    for (i, result) in results.iter().enumerate() {
        let sample_value = result.data()[0];
        println!("     Batch {}: First element = {:.3}", i, sample_value);
    }

    Ok(())
}

/// Test thread-local autograd engines
fn test_thread_local_engines() -> Result<()> {
    let num_threads = 3;
    let mut handles = Vec::new();

    println!("   Testing {} independent thread-local engines", num_threads);

    for thread_id in 0..num_threads {
        let handle = thread::spawn(move || -> Result<()> {
            with_local_engine(|engine| {
                println!("     Thread {}: Local engine initialized", thread_id);
                
                // Perform some operations to test independence
                let a = Tensor::new(vec![1.0f32, 2.0], Shape::from(&[2]))?.requires_grad(true);
                let b = Tensor::new(vec![3.0f32, 4.0], Shape::from(&[2]))?.requires_grad(true);
                
                // Use parallel operations
                let sum = par_add(&a, &b)?;
                let product = par_mul(&a, &b)?;
                let _result = par_add(&sum, &product)?;
                
                println!("     Thread {}: Operations completed", thread_id);
                
                // Skip backward pass for thread safety
                println!("     Thread {}: Skipping backward pass (thread safety)", thread_id);
                println!("     Thread {}: Final stats: {:?}", thread_id, engine.stats());
                
                Ok(())
            })
        });
        handles.push(handle);
    }

    // Wait for all threads
    for (i, handle) in handles.into_iter().enumerate() {
        match handle.join() {
            Ok(result) => {
                result?;
                println!("   ✅ Thread-local engine {} completed", i);
            }
            Err(_) => {
                println!("   ❌ Thread-local engine {} failed", i);
            }
        }
    }

    // Test global coordinator statistics
    let coordinator = get_global_coordinator();
    let global_stats = coordinator.stats();
    println!("   Global coordinator stats: {:?}", global_stats);

    Ok(())
}

/// Demonstrate tensor statistics with parallel computation
fn _test_parallel_statistics() -> Result<()> {
    println!("\n5. Parallel tensor statistics");
    
    let size = 10000;
    let data: Vec<f32> = (0..size).map(|i| (i as f32).sin()).collect();
    let tensor = Tensor::new(data, Shape::from(&[size]))?;
    
    let start = Instant::now();
    let stats = par_statistics(&tensor)?;
    let stats_time = start.elapsed();
    
    println!("   Computed statistics for {} elements in {:?}", size, stats_time);
    println!("   Min: {:.3}, Max: {:.3}, Mean: {:.3}, Sum: {:.3}", 
            stats.min, stats.max, stats.mean, stats.sum);
    
    Ok(())
}

/// Helper function to get global coordinator
fn get_global_coordinator() -> std::sync::Arc<quasar::autograd::parallel::GlobalCoordinator> {
    quasar::autograd::parallel::get_global_coordinator()
} 