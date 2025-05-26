//! Example demonstrating full backward pass with computational graph

use quasar::prelude::*;
use quasar::autograd::engine::with_global_engine;

fn main() -> Result<()> {
    println!("🌟 Quasar Full Backward Pass Demo");
    
    // Create input tensors with requires_grad
    let x = Tensor::new(vec![2.0, 3.0], Shape::from(&[2]))?.requires_grad(true);
    let y = Tensor::new(vec![4.0, 5.0], Shape::from(&[2]))?.requires_grad(true);
    
    println!("\nInput tensors:");
    println!("x = {:?}", x);
    println!("y = {:?}", y);
    
    // Test chained operations: result = (x + y) * (x - y)
    println!("\n=== Computing result = (x + y) * (x - y) ===");
    let sum = (&x + &y)?;
    let diff = (&x - &y)?;
    let result = (&sum * &diff)?;
    
    println!("sum = x + y = {:?}", sum.data());
    println!("diff = x - y = {:?}", diff.data());
    println!("result = sum * diff = {:?}", result.data());
    
    // Manually compute expected result: (x + y) * (x - y) = x² - y²
    // x = [2, 3], y = [4, 5]
    // x² = [4, 9], y² = [16, 25]
    // x² - y² = [4-16, 9-25] = [-12, -16] ✓
    
    // Test backward pass
    println!("\n=== Backward Pass ===");
    let mut result_mut = result.clone();
    
    match result_mut.backward() {
        Ok(()) => {
            println!("✅ Backward pass completed successfully!");
            
            // Check result gradient (should be ones)
            if let Some(grad) = result_mut.grad() {
                println!("Result gradient: {:?}", grad.data());
                assert_eq!(grad.data(), &[1.0, 1.0]);
            }
        }
        Err(e) => {
            println!("❌ Backward pass failed: {:?}", e);
            return Err(e);
        }
    }
    
    // Test gradients for leaf nodes
    println!("\n=== Checking Leaf Gradients ===");
    
    // Get gradients from global autograd engine
    let leaf_grads = with_global_engine::<f64, _, _>(|engine| {
        engine.get_leaf_gradients()
    });
    
    println!("Found {} leaf gradients:", leaf_grads.len());
    for (node_id, grad) in leaf_grads {
        println!("  Node {}: {:?}", node_id, grad.data());
    }
    
    // Expected gradients for result = (x + y) * (x - y) = x² - y²:
    // ∂result/∂x = ∂(x² - y²)/∂x = 2x = [4, 6]
    // ∂result/∂y = ∂(x² - y²)/∂y = -2y = [-8, -10]
    
    println!("\n✅ Full backward pass demo completed!");
    
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_backward() -> Result<()> {
        let x = Tensor::new(vec![2.0], Shape::from(&[1]))?
            .requires_grad(true);
        let y = Tensor::new(vec![3.0], Shape::from(&[1]))?
            .requires_grad(true);
        
        // z = x * y = 6.0
        let mut z = (&x * &y)?;
        z.backward()?;
        
        // dz/dx = y = 3.0
        // dz/dy = x = 2.0
        
        // Get gradients from global autograd engine
        let leaf_grads = with_global_engine::<f64, _, _>(|engine| {
            engine.get_leaf_gradients()
        });
        
        // We should have 2 leaf gradients
        assert_eq!(leaf_grads.len(), 2);
        
        // Check that gradients are correct (order might vary)
        let mut grad_values: Vec<f64> = leaf_grads.values()
            .map(|grad| grad.data()[0])
            .collect();
        grad_values.sort_by(|a, b| a.partial_cmp(b).unwrap());
        
        // Expected: [2.0, 3.0] (sorted)
        assert_eq!(grad_values, vec![2.0, 3.0]);
        
        Ok(())
    }

    #[test]
    fn test_chain_rule() -> Result<()> {
        let x = Tensor::new(vec![2.0], Shape::from(&[1]))?
            .requires_grad(true);
        
        // z = x * x + x = x² + x
        let x_squared = (&x * &x)?;
        let mut z = (&x_squared + &x)?;
        z.backward()?;
        
        // dz/dx = 2x + 1 = 2*2 + 1 = 5
        
        // Get gradients from global autograd engine
        let leaf_grads = with_global_engine::<f64, _, _>(|engine| {
            engine.get_leaf_gradients()
        });
        
        // We should have 1 leaf gradient (x appears twice but it's the same tensor)
        assert_eq!(leaf_grads.len(), 1);
        
        let grad_value = leaf_grads.values().next().unwrap().data()[0];
        assert_eq!(grad_value, 5.0);
        
        Ok(())
    }

    #[test]
    fn test_division_gradients() -> Result<()> {
        let x = Tensor::new(vec![4.0], Shape::from(&[1]))?
            .requires_grad(true);
        let y = Tensor::new(vec![2.0], Shape::from(&[1]))?
            .requires_grad(true);
        
        // z = x / y = 2.0
        let mut z = (&x / &y)?;
        z.backward()?;
        
        // dz/dx = 1/y = 0.5
        // dz/dy = -x/y² = -4/4 = -1.0
        
        // Get gradients from global autograd engine
        let leaf_grads = with_global_engine::<f64, _, _>(|engine| {
            engine.get_leaf_gradients()
        });
        
        // We should have 2 leaf gradients
        assert_eq!(leaf_grads.len(), 2);
        
        // Check that gradients are correct (order might vary)
        let mut grad_values: Vec<f64> = leaf_grads.values()
            .map(|grad| grad.data()[0])
            .collect();
        grad_values.sort_by(|a, b| a.partial_cmp(b).unwrap());
        
        // Expected: [-1.0, 0.5] (sorted)
        assert_eq!(grad_values, vec![-1.0, 0.5]);
        
        Ok(())
    }
} 