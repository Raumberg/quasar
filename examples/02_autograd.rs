//! Automatic differentiation example
//! 
//! This example demonstrates:
//! - Creating tensors that require gradients
//! - Forward pass computation
//! - Backward pass and gradient computation
//! - Chain rule in action

use quasar::prelude::*;

fn main() -> Result<()> {
    println!("🚀 Quasar Automatic Differentiation Example\n");

    // 1. Simple gradient computation
    println!("1. Simple Gradient Computation:");
    println!("   Computing gradients for f(x, y) = x * y + x");
    
    let x = Tensor::new(vec![2.0, 3.0], Shape::from(&[2]))?.requires_grad(true);
    let y = Tensor::new(vec![4.0, 5.0], Shape::from(&[2]))?.requires_grad(true);
    
    println!("   x = {:?}", x.data());
    println!("   y = {:?}", y.data());
    
    // Forward pass: z = x * y + x
    let xy = mul(&x, &y)?;
    let mut z = add(&xy, &x)?;
    
    println!("   z = x * y + x = {:?}", z.data());
    
    // Backward pass
    z.backward()?;
    
    // Get gradients manually from engine (temporary solution)
    let (mut x_grad, mut y_grad) = (None, None);
    with_global_engine::<f32, _, _>(|engine| {
        let leaf_grads = engine.get_leaf_gradients();
        for (node_id, grad) in leaf_grads {
            if node_id == 0 {
                x_grad = Some(grad);
            } else if node_id == 1 {
                y_grad = Some(grad);
            }
        }
    });
    
    if let Some(grad) = x_grad {
        println!("   dz/dx = {:?} (expected: [5.0, 6.0])", grad.data());
    }
    if let Some(grad) = y_grad {
        println!("   dz/dy = {:?} (expected: [2.0, 3.0])", grad.data());
    }

    // 2. More complex computation
    println!("\n2. Complex Gradient Computation:");
    println!("   Computing gradients for f(a, b) = (a + b) * (a - b)");
    
    let a = Tensor::new(vec![3.0, 4.0], Shape::from(&[2]))?.requires_grad(true);
    let b = Tensor::new(vec![1.0, 2.0], Shape::from(&[2]))?.requires_grad(true);
    
    println!("   a = {:?}", a.data());
    println!("   b = {:?}", b.data());
    
    // Forward pass: f = (a + b) * (a - b) = a² - b²
    let sum_ab = add(&a, &b)?;
    let diff_ab = sub(&a, &b)?;
    let mut f = mul(&sum_ab, &diff_ab)?;
    
    println!("   f = (a + b) * (a - b) = {:?}", f.data());
    
    // Backward pass
    f.backward()?;
    
    // Get gradients
    let (mut a_grad, mut b_grad) = (None, None);
    with_global_engine::<f32, _, _>(|engine| {
        let leaf_grads = engine.get_leaf_gradients();
        for (node_id, grad) in leaf_grads {
            if node_id == 0 {
                a_grad = Some(grad);
            } else if node_id == 1 {
                b_grad = Some(grad);
            }
        }
    });
    
    if let Some(grad) = a_grad {
        println!("   df/da = {:?} (expected: [6.0, 8.0] = 2*a)", grad.data());
    }
    if let Some(grad) = b_grad {
        println!("   df/db = {:?} (expected: [-2.0, -4.0] = -2*b)", grad.data());
    }

    // 3. Scalar function
    println!("\n3. Scalar Function:");
    println!("   Computing gradient for f(x) = x²");
    
    let x_scalar = Tensor::new(vec![5.0], Shape::from(&[1]))?.requires_grad(true);
    println!("   x = {:?}", x_scalar.data());
    
    // Forward pass: f = x²
    let mut f_scalar = mul(&x_scalar, &x_scalar)?;
    println!("   f = x² = {:?}", f_scalar.data());
    
    // Backward pass
    f_scalar.backward()?;
    
    // Get gradient
    with_global_engine::<f32, _, _>(|engine| {
        let leaf_grads = engine.get_leaf_gradients();
        for (node_id, grad) in leaf_grads {
            if node_id == 0 {
                println!("   df/dx = {:?} (expected: [10.0] = 2*x)", grad.data());
                break;
            }
        }
    });

    // 4. Chain rule demonstration
    println!("\n4. Chain Rule Demonstration:");
    println!("   Computing gradients for f(x) = (x + 1)² where x = [1, 2]");
    
    let x_chain = Tensor::new(vec![1.0, 2.0], Shape::from(&[2]))?.requires_grad(true);
    let ones = Tensor::new(vec![1.0, 1.0], Shape::from(&[2]))?;
    
    println!("   x = {:?}", x_chain.data());
    
    // Forward pass: f = (x + 1)²
    let x_plus_1 = add(&x_chain, &ones)?;
    let mut f_chain = mul(&x_plus_1, &x_plus_1)?;
    
    println!("   f = (x + 1)² = {:?}", f_chain.data());
    
    // Backward pass
    f_chain.backward()?;
    
    // Get gradient
    with_global_engine::<f32, _, _>(|engine| {
        let leaf_grads = engine.get_leaf_gradients();
        for (node_id, grad) in leaf_grads {
            if node_id == 0 {
                println!("   df/dx = {:?} (expected: [4.0, 6.0] = 2*(x+1))", grad.data());
                break;
            }
        }
    });

    println!("\n✅ Automatic differentiation completed successfully!");
    println!("💡 Note: Gradients are computed using the chain rule automatically!");
    
    Ok(())
} 