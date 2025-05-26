//! Tests for automatic differentiation

use quasar::prelude::*;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_forward() -> Result<()> {
        let x = Tensor::new(vec![2.0], Shape::from(&[1]))?;
        let y = Tensor::new(vec![3.0], Shape::from(&[1]))?;
        
        let add_op = AddOp;
        let result = add_op.forward(&[&x, &y])?;
        
        assert_eq!(result.item()?, 5.0);
        Ok(())
    }

    #[test]
    fn test_add_backward() -> Result<()> {
        let x = Tensor::new(vec![2.0], Shape::from(&[1]))?;
        let y = Tensor::new(vec![3.0], Shape::from(&[1]))?;
        let grad_output = Tensor::ones(Shape::from(&[1]))?;
        
        let add_op = AddOp;
        let gradients = add_op.backward(&grad_output, &[&x, &y])?;
        
        assert_eq!(gradients.len(), 2);
        assert_eq!(gradients[0].item()?, 1.0); // dL/dx = 1
        assert_eq!(gradients[1].item()?, 1.0); // dL/dy = 1
        Ok(())
    }

    #[test]
    fn test_mul_forward() -> Result<()> {
        let x = Tensor::new(vec![2.0], Shape::from(&[1]))?;
        let y = Tensor::new(vec![3.0], Shape::from(&[1]))?;
        
        let mul_op = MulOp;
        let result = mul_op.forward(&[&x, &y])?;
        
        assert_eq!(result.item()?, 6.0);
        Ok(())
    }

    #[test]
    fn test_mul_backward() -> Result<()> {
        let x = Tensor::new(vec![2.0], Shape::from(&[1]))?;
        let y = Tensor::new(vec![3.0], Shape::from(&[1]))?;
        let grad_output = Tensor::ones(Shape::from(&[1]))?;
        
        let mul_op = MulOp;
        let gradients = mul_op.backward(&grad_output, &[&x, &y])?;
        
        assert_eq!(gradients.len(), 2);
        assert_eq!(gradients[0].item()?, 3.0); // dL/dx = y = 3
        assert_eq!(gradients[1].item()?, 2.0); // dL/dy = x = 2
        Ok(())
    }

    #[test]
    fn test_vector_add_backward() -> Result<()> {
        let x = Tensor::new(vec![1.0, 2.0, 3.0], Shape::from(&[3]))?;
        let y = Tensor::new(vec![4.0, 5.0, 6.0], Shape::from(&[3]))?;
        let grad_output = Tensor::ones(Shape::from(&[3]))?;
        
        let add_op = AddOp;
        let gradients = add_op.backward(&grad_output, &[&x, &y])?;
        
        // All gradients should be 1.0 for addition
        for i in 0..3 {
            assert_eq!(gradients[0].item_at(&[i])?, 1.0);
            assert_eq!(gradients[1].item_at(&[i])?, 1.0);
        }
        Ok(())
    }

    #[test]
    fn test_vector_mul_backward() -> Result<()> {
        let x = Tensor::new(vec![2.0, 3.0, 4.0], Shape::from(&[3]))?;
        let y = Tensor::new(vec![5.0, 6.0, 7.0], Shape::from(&[3]))?;
        let grad_output = Tensor::ones(Shape::from(&[3]))?;
        
        let mul_op = MulOp;
        let gradients = mul_op.backward(&grad_output, &[&x, &y])?;
        
        // dL/dx = y, dL/dy = x
        assert_eq!(gradients[0].item_at(&[0])?, 5.0); // y[0]
        assert_eq!(gradients[0].item_at(&[1])?, 6.0); // y[1]
        assert_eq!(gradients[0].item_at(&[2])?, 7.0); // y[2]
        
        assert_eq!(gradients[1].item_at(&[0])?, 2.0); // x[0]
        assert_eq!(gradients[1].item_at(&[1])?, 3.0); // x[1]
        assert_eq!(gradients[1].item_at(&[2])?, 4.0); // x[2]
        Ok(())
    }

    #[test]
    fn test_matrix_add_backward() -> Result<()> {
        let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], Shape::from(&[2, 2]))?;
        let y = Tensor::new(vec![5.0, 6.0, 7.0, 8.0], Shape::from(&[2, 2]))?;
        let grad_output = Tensor::ones(Shape::from(&[2, 2]))?;
        
        let add_op = AddOp;
        let gradients = add_op.backward(&grad_output, &[&x, &y])?;
        
        // All gradients should be 1.0 for addition
        for i in 0..2 {
            for j in 0..2 {
                assert_eq!(gradients[0].item_at(&[i, j])?, 1.0);
                assert_eq!(gradients[1].item_at(&[i, j])?, 1.0);
            }
        }
        Ok(())
    }

    #[test]
    fn test_chain_rule_manual() -> Result<()> {
        // Test z = x * y + x manually
        let x = Tensor::new(vec![2.0], Shape::from(&[1]))?;
        let y = Tensor::new(vec![3.0], Shape::from(&[1]))?;
        
        // Forward: xy = x * y
        let mul_op = MulOp;
        let xy = mul_op.forward(&[&x, &y])?;
        assert_eq!(xy.item()?, 6.0);
        
        // Forward: z = xy + x
        let add_op = AddOp;
        let z = add_op.forward(&[&xy, &x])?;
        assert_eq!(z.item()?, 8.0); // 6 + 2 = 8
        
        // Backward: dz/d(xy) = 1, dz/dx_direct = 1
        let grad_output = Tensor::ones(Shape::from(&[1]))?;
        let grad_add = add_op.backward(&grad_output, &[&xy, &x])?;
        assert_eq!(grad_add[0].item()?, 1.0); // dz/d(xy)
        assert_eq!(grad_add[1].item()?, 1.0); // dz/dx_direct
        
        // Backward: d(xy)/dx = y, d(xy)/dy = x
        let grad_mul = mul_op.backward(&grad_add[0], &[&x, &y])?;
        assert_eq!(grad_mul[0].item()?, 3.0); // d(xy)/dx = y = 3
        assert_eq!(grad_mul[1].item()?, 2.0); // d(xy)/dy = x = 2
        
        // Total gradients: dz/dx = d(xy)/dx + dz/dx_direct = 3 + 1 = 4
        let total_grad_x = grad_mul[0].item()? + grad_add[1].item()?;
        let total_grad_y = grad_mul[1].item()?;
        
        assert_eq!(total_grad_x, 4.0); // dz/dx = y + 1 = 4
        assert_eq!(total_grad_y, 2.0); // dz/dy = x = 2
        
        Ok(())
    }

    #[test]
    fn test_zero_gradients() -> Result<()> {
        let x = Tensor::new(vec![5.0], Shape::from(&[1]))?;
        let y = Tensor::zeros(Shape::from(&[1]))?;
        let grad_output = Tensor::ones(Shape::from(&[1]))?;
        
        let mul_op = MulOp;
        let gradients = mul_op.backward(&grad_output, &[&x, &y])?;
        
        assert_eq!(gradients[0].item()?, 0.0); // dL/dx = y = 0
        assert_eq!(gradients[1].item()?, 5.0); // dL/dy = x = 5
        Ok(())
    }

    #[test]
    fn test_large_tensor_gradients() -> Result<()> {
        // Test with 100-element tensors
        let data_x: Vec<f32> = (1..=100).map(|i| i as f32).collect();
        let data_y: Vec<f32> = (101..=200).map(|i| i as f32).collect();
        
        let x = Tensor::new(data_x, Shape::from(&[100]))?;
        let y = Tensor::new(data_y, Shape::from(&[100]))?;
        let grad_output = Tensor::ones(Shape::from(&[100]))?;
        
        let mul_op = MulOp;
        let gradients = mul_op.backward(&grad_output, &[&x, &y])?;
        
        // Check a few elements
        assert_eq!(gradients[0].item_at(&[0])?, 101.0); // y[0] = 101
        assert_eq!(gradients[0].item_at(&[99])?, 200.0); // y[99] = 200
        assert_eq!(gradients[1].item_at(&[0])?, 1.0);   // x[0] = 1
        assert_eq!(gradients[1].item_at(&[99])?, 100.0); // x[99] = 100
        
        Ok(())
    }

    #[test]
    fn test_autograd_op_wrong_inputs() {
        let x = Tensor::new(vec![1.0], Shape::from(&[1])).unwrap();
        
        let add_op = AddOp;
        
        // Should fail with wrong number of inputs
        assert!(add_op.forward(&[&x]).is_err());
        assert!(add_op.forward(&[&x, &x, &x]).is_err());
    }
} 