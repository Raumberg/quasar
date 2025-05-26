# Contributing to Quasar 🌟

Thank you for your interest in contributing to Quasar! We welcome contributions from everyone, whether you're fixing a bug, adding a feature, improving documentation, or just asking questions.

## 🚀 Quick Start

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/Raumberg/quasar.git
   cd quasar
   ```
3. **Create a branch** for your changes:
   ```bash
   git checkout -b feature/your-feature-name
   ```
4. **Make your changes** and test them
5. **Submit a pull request**

## 🛠️ Development Setup

### Prerequisites

- **Rust 1.70+** (for advanced const generics)
- **Git**
- **A good text editor** (VS Code with rust-analyzer recommended)

### Building

```bash
# Clone the repository
git clone https://github.com/Raumberg/quasar.git
cd quasar

# Build in debug mode
cargo build

# Build in release mode (optimized)
cargo build --release

# Run tests
cargo test

# Run with output
cargo test -- --nocapture

# Run specific test
cargo test test_chain_rule

# Run examples
cargo run --example full_backward
```

### Code Quality Tools

We use several tools to maintain code quality:

```bash
# Format code
cargo fmt

# Check for common mistakes
cargo clippy

# Check for security vulnerabilities
cargo audit

# Generate documentation
cargo doc --open
```

## 📋 What Can You Contribute?

### 🐛 Bug Reports

Found a bug? Please create an issue with:

- **Clear description** of the problem
- **Steps to reproduce** the issue
- **Expected vs actual behavior**
- **Environment details** (OS, Rust version, etc.)
- **Minimal code example** if possible

### ✨ Feature Requests

Have an idea for a new feature? Create an issue with:

- **Clear description** of the feature
- **Use case** - why is this needed?
- **Proposed API** if you have ideas
- **Implementation notes** if you know how it could work

### 🔧 Code Contributions

We especially welcome contributions in these areas:

#### 🧮 SIMD Optimizations
- Vectorized operations for different CPU architectures
- AVX2/AVX512 implementations
- ARM NEON support
- Performance benchmarking

#### 🐍 Python Bindings
- PyO3 integration
- NumPy compatibility layer
- Python packaging and distribution
- Jupyter notebook support

#### 🚀 GPU Backends
- CUDA implementation with cuDNN
- Metal backend for Apple Silicon
- Vulkan compute shaders
- OpenCL support

#### 📚 Documentation
- API documentation improvements
- Tutorial writing
- Example creation
- Blog posts about internals

#### 🧪 Testing
- Edge case testing
- Performance regression tests
- Fuzzing for robustness
- Integration tests

## 📝 Coding Guidelines

### Rust Style

We follow standard Rust conventions:

- **Use `cargo fmt`** for formatting
- **Follow Rust naming conventions** (snake_case for functions, PascalCase for types)
- **Write documentation** for public APIs
- **Add tests** for new functionality
- **Use meaningful variable names**
- **Prefer explicit over implicit**

### Code Organization

```rust
// Good: Clear, documented function
/// Performs element-wise addition of two tensors
/// 
/// # Arguments
/// * `a` - First tensor
/// * `b` - Second tensor
/// 
/// # Returns
/// Result containing the sum tensor or an error
/// 
/// # Examples
/// ```
/// let a = Tensor::new(vec![1.0, 2.0], Shape::from(&[2]))?;
/// let b = Tensor::new(vec![3.0, 4.0], Shape::from(&[2]))?;
/// let result = add(&a, &b)?;
/// ```
pub fn add<T: TensorElement>(a: &Tensor<T>, b: &Tensor<T>) -> Result<Tensor<T>> {
    // Implementation...
}
```

### Error Handling

- **Use `Result<T>` for fallible operations**
- **Provide meaningful error messages**
- **Use custom error types** from `crate::error`
- **Don't panic in library code** (except for truly unrecoverable errors)

```rust
// Good: Descriptive error
if a.shape() != b.shape() {
    return Err(QuasarError::shape_mismatch(
        a.shape().dims(), 
        b.shape().dims()
    ));
}

// Bad: Generic error
if a.shape() != b.shape() {
    return Err(QuasarError::invalid_operation("shapes don't match"));
}
```

### Performance Considerations

- **Profile before optimizing**
- **Use SIMD when beneficial**
- **Minimize allocations in hot paths**
- **Consider cache locality**
- **Benchmark performance-critical changes**

## 🧪 Testing

### Writing Tests

- **Unit tests** for individual functions
- **Integration tests** for complex workflows
- **Property-based tests** for mathematical operations
- **Benchmark tests** for performance-critical code

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_addition_basic() -> Result<()> {
        let a = Tensor::new(vec![1.0, 2.0], Shape::from(&[2]))?;
        let b = Tensor::new(vec![3.0, 4.0], Shape::from(&[2]))?;
        let result = add(&a, &b)?;
        
        assert_eq!(result.data(), &[4.0, 6.0]);
        Ok(())
    }

    #[test]
    fn test_addition_shape_mismatch() {
        let a = Tensor::new(vec![1.0, 2.0], Shape::from(&[2])).unwrap();
        let b = Tensor::new(vec![1.0], Shape::from(&[1])).unwrap();
        
        assert!(add(&a, &b).is_err());
    }
}
```

### Running Tests

```bash
# Run all tests
cargo test

# Run tests with output
cargo test -- --nocapture

# Run specific test
cargo test test_addition

# Run tests in specific module
cargo test ops::arithmetic

# Run ignored tests
cargo test -- --ignored

# Run benchmarks
cargo bench
```

## 📖 Documentation

### API Documentation

- **Document all public APIs** with `///` comments
- **Include examples** in documentation
- **Explain parameters and return values**
- **Note any panics or special behavior**

### Examples

When adding examples:

- **Create self-contained examples** in `examples/`
- **Add comments explaining the code**
- **Show both basic and advanced usage**
- **Include error handling**

## 🔄 Pull Request Process

### Before Submitting

1. **Run the full test suite**: `cargo test`
2. **Check formatting**: `cargo fmt --check`
3. **Run clippy**: `cargo clippy -- -D warnings`
4. **Update documentation** if needed
5. **Add tests** for new functionality

### PR Description

Include in your PR description:

- **What** does this PR do?
- **Why** is this change needed?
- **How** does it work?
- **Testing** - what tests did you add/run?
- **Breaking changes** - if any

### Review Process

1. **Automated checks** must pass (CI/CD)
2. **Code review** by maintainers
3. **Discussion** and iteration if needed
4. **Merge** once approved

## 🏷️ Commit Messages

Use clear, descriptive commit messages:

```bash
# Good
git commit -m "feat: add SIMD-optimized matrix multiplication"
git commit -m "fix: handle edge case in gradient computation"
git commit -m "docs: add examples for custom operations"

# Bad
git commit -m "fix stuff"
git commit -m "update"
git commit -m "wip"
```

### Commit Types

- `feat:` - New feature
- `fix:` - Bug fix
- `docs:` - Documentation changes
- `style:` - Code style changes (formatting, etc.)
- `refactor:` - Code refactoring
- `test:` - Adding or updating tests
- `perf:` - Performance improvements
- `chore:` - Maintenance tasks

## 🎯 Areas We Need Help

### High Priority

- **🧮 SIMD Optimizations**: Vectorized operations for different architectures
- **🐍 Python Bindings**: PyO3 integration and NumPy compatibility
- **📚 Documentation**: More examples and tutorials

### Medium Priority

- **🚀 GPU Backends**: CUDA, Metal, and Vulkan implementations
- **🧪 Testing**: Edge cases and performance tests
- **🔧 Tooling**: Better error messages and debugging tools

### Future Work

- **🧠 Advanced Features**: JIT compilation, graph optimization
- **📊 Quantization**: INT8, FP16 support
- **🌐 Distributed**: Multi-GPU and distributed training

## 💬 Communication

- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: General questions and ideas
- **Discord**: Real-time chat (link in README)

## 📄 License

By contributing to Quasar, you agree that your contributions will be licensed under the MIT License.

## 🙏 Recognition

All contributors will be recognized in our README and release notes. We appreciate every contribution, no matter how small!

---

**Happy coding! 🚀** 