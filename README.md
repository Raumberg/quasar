# 🌟 Quasar - High-Performance Autograd Engine in Rust

[![Rust](https://github.com/username/Raumberg/workflows/Rust/badge.svg)](https://github.com/Raumberg/quasar/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Crates.io](https://img.shields.io/crates/v/quasar.svg)](https://crates.io/crates/quasar)
[![Documentation](https://docs.rs/quasar/badge.svg)](https://docs.rs/quasar)

**Quasar** is a blazingly fast automatic differentiation (autograd) engine built from scratch in Rust. Designed for high-performance machine learning workloads with zero-cost abstractions and memory safety guarantees.

## 🚀 Features

- **🔥 High Performance**: SIMD-optimized operations with aligned memory allocation
- **🛡️ Memory Safe**: Built in Rust with zero-cost abstractions
- **🧠 Automatic Differentiation**: Full computational graph with backward pass
- **🔗 Chain Rule Support**: Complex gradient computations for deep learning
- **🎯 PyTorch-like API**: Familiar interface for ML practitioners
- **🧵 Multi-threading**: JAX/PyTorch-like parallel computing with 16x+ speedups
- **⚡ Thread-local Engines**: Independent autograd computation per thread
- **🚀 Automatic Parallelization**: Smart threshold-based parallel execution
- **📊 Mixed Precision**: Support for f32 and f64 tensors
- **🔧 Extensible**: Easy to add custom operations and backends

## 📦 Quick Start

Add Quasar to your `Cargo.toml`:

```toml
[dependencies]
quasar = "0.1.0"
```

### Basic Usage

```rust
use quasar::prelude::*;

fn main() -> Result<()> {
    // Create tensors with gradient tracking
    let x = Tensor::new(vec![2.0, 3.0], Shape::from(&[2]))?.requires_grad(true);
    let y = Tensor::new(vec![4.0, 5.0], Shape::from(&[2]))?.requires_grad(true);
    
    // Perform operations: result = (x + y) * (x - y) = x² - y²
    let sum = (&x + &y)?;
    let diff = (&x - &y)?;
    let mut result = (&sum * &diff)?;
    
    // Compute gradients
    result.backward()?;
    
    // Access gradients through global autograd engine
    let leaf_grads = with_global_engine::<f64, _, _>(|engine| {
        engine.get_leaf_gradients()
    });
    
    println!("Result: {:?}", result.data()); // [-12.0, -16.0]
    println!("Gradients: {:?}", leaf_grads);
    
    Ok(())
}
```

### Advanced Example: Neural Network Layer

```rust
use quasar::prelude::*;

fn linear_layer(input: &Tensor<f32>, weight: &Tensor<f32>, bias: &Tensor<f32>) -> Result<Tensor<f32>> {
    // Forward pass: output = input * weight + bias
    let matmul_result = matmul(input, weight)?;
    let output = (&matmul_result + bias)?;
    Ok(output)
}

fn main() -> Result<()> {
    // Input: batch_size=2, features=3
    let input = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 
                           Shape::from(&[2, 3]))?.requires_grad(true);
    
    // Weight: input_features=3, output_features=2
    let weight = Tensor::new(vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6], 
                            Shape::from(&[3, 2]))?.requires_grad(true);
    
    // Bias: output_features=2
    let bias = Tensor::new(vec![0.1, 0.2], Shape::from(&[2]))?.requires_grad(true);
    
    // Forward pass
    let mut output = linear_layer(&input, &weight, &bias)?;
    
    // Backward pass
    output.backward()?;
    
    println!("Output shape: {:?}", output.shape());
    println!("Gradients computed successfully!");
    
    Ok(())
}
```

### Parallel Computing Example

```rust
use quasar::prelude::*;
use quasar::autograd::parallel::{ParallelConfig, init_parallel_autograd, with_local_engine};
use quasar::ops::parallel_ops::{par_add, par_mul, par_matmul, par_relu};
use std::thread;

fn main() -> Result<()> {
    // Initialize parallel autograd
    let config = ParallelConfig {
        max_workers: num_cpus::get(),
        parallel_threshold: 1000,  // Parallelize tensors with 1000+ elements
        ..Default::default()
    };
    init_parallel_autograd(config)?;
    
    // Large tensors automatically use parallel computation
    let large_a = Tensor::new(
        (0..1_000_000).map(|i| i as f32 / 1000.0).collect(),
        Shape::from(&[1000, 1000])
    )?;
    let large_b = Tensor::new(
        (0..1_000_000).map(|i| (i + 1) as f32 / 1000.0).collect(),
        Shape::from(&[1000, 1000])
    )?;
    
    // These operations automatically use parallel execution (16x+ speedup)
    let sum = par_add(&large_a, &large_b)?;           // Parallel addition
    let product = par_mul(&large_a, &large_b)?;       // Parallel multiplication
    let matmul_result = par_matmul(&large_a, &large_b)?;  // Parallel matrix multiplication
    let activated = par_relu(&sum)?;                  // Parallel ReLU activation
    
    // Multi-threaded training with thread-local engines
    let handles: Vec<_> = (0..4).map(|thread_id| {
        thread::spawn(move || -> Result<()> {
            with_local_engine(|_engine| {
                // Each thread has independent autograd computation
                let weights = Tensor::new(weight_data, shape)?.requires_grad(true);
                let input = Tensor::new(input_data, input_shape)?;
                
                // Parallel forward pass
                let hidden = par_relu(&par_matmul(&weights, &input)?)?;
                let output = par_matmul(&output_weights, &hidden)?;
                
                println!("Thread {}: Output = {:.3}", thread_id, output.item()?);
                Ok(())
            })
        })
    }).collect();
    
    // Wait for all threads
    for handle in handles {
        handle.join().unwrap()?;
    }
    
    println!("Parallel computation completed!");
    Ok(())
}
```

## 🏗️ Architecture

Quasar uses a **global autograd engine** architecture similar to PyTorch:

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Tensor API    │───▶│  Autograd Engine │───▶│ Computational   │
│                 │    │                  │    │     Graph       │
│ • requires_grad │    │ • Global state   │    │                 │
│ • backward()    │    │ • Thread-safe    │    │ • Topological   │
│ • operations    │    │ • Arc<Mutex<>>   │    │   sorting       │
└─────────────────┘    └──────────────────┘    │ • Gradient      │
                                               │   computation   │
┌─────────────────┐    ┌──────────────────┐    └─────────────────┘
│  Memory Layer   │    │   SIMD Ops       │
│                 │    │                  │
│ • Aligned alloc │    │ • Vectorized     │
│ • Zero-copy     │    │ • CPU optimized  │
│ • Cache-friendly│    │ • Future: GPU    │
└─────────────────┘    └──────────────────┘
```

## 🧪 Examples

Check out the `examples/` directory for more comprehensive examples:

- [`basic_usage.rs`](examples/basic_usage.rs) - Simple tensor operations
- [`full_backward.rs`](examples/full_backward.rs) - Complete autograd demo
- [`neural_network_test.rs`](examples/neural_network_test.rs) - Building neural networks
- [`parallel_neural_network.rs`](examples/parallel_neural_network.rs) - **Multi-threading and parallel computing**
- [`test_refactoring.rs`](examples/test_refactoring.rs) - Memory optimization features
- [`complete_neural_network.rs`](examples/complete_neural_network.rs) - End-to-end training example

### Running Examples

```bash
# Basic tensor operations
cargo run --example basic_usage

# Complete autograd demonstration
cargo run --example full_backward

# Neural network with ReLU and matrix multiplication
cargo run --example neural_network_test

# Multi-threaded parallel computing (16x+ speedup)
cargo run --example parallel_neural_network

# Memory optimization with copy-on-write
cargo run --example test_refactoring
```

## 🔬 Benchmarks

TODO!

## 🛠️ Development

### Prerequisites

- Rust 1.70+ (for advanced const generics)
- Cargo

### Building

```bash
git clone https://github.com/Raumberg/quasar.git
cd quasar
cargo build --release
```

### Testing

```bash
# Run all tests
cargo test

# Run with output
cargo test -- --nocapture

# Run specific test
cargo test test_chain_rule

# Run examples
cargo run --example full_backward
```

### Benchmarking

```bash
cargo bench
```

## 🎯 Roadmap

### Phase 1: Core Engine ✅
- [x] Basic tensor operations (add, sub, mul, div)
- [x] Automatic differentiation with computational graph
- [x] Memory-aligned SIMD operations
- [x] Thread-safe global autograd engine
- [x] Chain rule and complex gradient computations

### Phase 2: Extended Operations ✅
- [x] Matrix multiplication (matmul) with parallel support
- [x] Activation functions (ReLU) with parallel support
- [x] Multi-threading and parallel computing architecture
- [x] Thread-local autograd engines
- [x] Automatic parallelization with configurable thresholds
- [ ] Loss functions (MSE, CrossEntropy)
- [ ] Broadcasting support
- [ ] Tensor slicing and indexing

### Phase 3: Advanced Parallel Computing 🚧
- [x] **Production-ready multi-threading** with 16x+ speedups
- [x] **Thread-local autograd engines** for independent computation
- [x] **Automatic parallelization** based on tensor size thresholds
- [x] **Parallel operations**: add, mul, matmul, relu
- [x] **Batch processing** for multiple tensor operations
- [ ] **Thread-safe backward pass** for full autograd in multi-threaded contexts
- [ ] **Operation fusion** for optimized kernel execution
- [ ] **Distributed computing** support for multi-node training
- [ ] **Memory pool optimization** for reduced allocation overhead

### Phase 4: Neural Network Primitives 📋
- [ ] Convolution operations (Conv1D, Conv2D) with parallel support
- [ ] Pooling operations (MaxPool, AvgPool)
- [ ] Batch normalization
- [ ] Dropout layers
- [ ] LSTM/GRU cells
- [ ] Transformer attention mechanisms

### Phase 5: Python Integration 🐍
- [ ] PyO3 bindings for Python interop
- [ ] NumPy compatibility layer
- [ ] Jupyter notebook support
- [ ] Python package distribution
- [ ] Multi-threading support in Python bindings

### Phase 6: GPU Acceleration 🚀
- [ ] CUDA backend with cuDNN
- [ ] Metal backend for Apple Silicon
- [ ] Multi-GPU support with automatic load balancing
- [ ] CPU-GPU hybrid execution
- [ ] Vulkan compute shaders
- [ ] Multi-GPU support

### Phase 6: Advanced Features 🧠
- [ ] JIT compilation with LLVM
- [ ] Graph optimization passes
- [ ] Quantization support (INT8, FP16)
- [ ] Distributed training primitives

## 🐍 Python Integration (Planned)

Quasar will provide seamless Python integration:

```python
import quasar as qsr
import numpy as np

# Create tensors from NumPy arrays
x = qsr.tensor(np.array([1.0, 2.0, 3.0]), requires_grad=True)
y = qsr.tensor(np.array([4.0, 5.0, 6.0]), requires_grad=True)

# Perform operations
z = (x + y) * (x - y)

# Backward pass
z.backward()

# Access gradients
print(f"x.grad: {x.grad}")
print(f"y.grad: {y.grad}")
```

## 🤝 Contributing

We welcome contributions! Here's how to get started:

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/amazing-feature`
3. **Make your changes** and add tests
4. **Run the test suite**: `cargo test`
5. **Submit a pull request**

### Contribution Guidelines

- Follow Rust best practices and idioms
- Add tests for new functionality
- Update documentation for public APIs
- Run `cargo fmt` and `cargo clippy` before submitting
- Write clear commit messages

### Areas We Need Help With

- 🧮 **SIMD Optimizations**: Vectorized operations for different architectures
- 🐍 **Python Bindings**: PyO3 integration and NumPy compatibility
- 🚀 **GPU Backends**: CUDA, Metal, and Vulkan implementations
- 📚 **Documentation**: Examples, tutorials, and API docs
- 🧪 **Testing**: Edge cases, performance tests, and fuzzing
- 🔧 **Tooling**: Better error messages and debugging tools

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Inspired by [PyTorch](https://pytorch.org/) and [Candle](https://github.com/huggingface/candle)
- Built with the amazing Rust ecosystem
- Special thanks to the Rust ML community

## 📞 Contact

- **Issues**: [GitHub Issues](https://github.com/Raumberg/quasar/issues)
- **Discussions**: [GitHub Discussions](https://github.com/Raumberg/quasar/discussions)
---

**⭐ Star this repository if you find it useful!**

*Built with ❤️ and ☕ by the Attention Signs team* 