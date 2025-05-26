//! Tests for arithmetic operations

use quasar::prelude::*;

#[cfg(test)]
mod tests {
    use super::*;

    // Helper function to create test tensors
    fn create_test_tensors() -> Result<(Tensor<f32>, Tensor<f32>)> {
        let a = Tensor::new(vec![1.0, 2.0, 3.0], Shape::from(&[3]))?;
        let b = Tensor::new(vec![4.0, 5.0, 6.0], Shape::from(&[3]))?;
        Ok((a, b))
    }

    #[test]
    fn test_addition_basic() -> Result<()> {
        let (a, b) = create_test_tensors()?;
        let result = (&a + &b)?;
        
        assert_eq!(result.item_at(&[0])?, 5.0);
        assert_eq!(result.item_at(&[1])?, 7.0);
        assert_eq!(result.item_at(&[2])?, 9.0);
        Ok(())
    }

    #[test]
    fn test_subtraction_basic() -> Result<()> {
        let (a, b) = create_test_tensors()?;
        let result = (&b - &a)?;
        
        assert_eq!(result.item_at(&[0])?, 3.0);
        assert_eq!(result.item_at(&[1])?, 3.0);
        assert_eq!(result.item_at(&[2])?, 3.0);
        Ok(())
    }

    #[test]
    fn test_multiplication_basic() -> Result<()> {
        let (a, b) = create_test_tensors()?;
        let result = (&a * &b)?;
        
        assert_eq!(result.item_at(&[0])?, 4.0);
        assert_eq!(result.item_at(&[1])?, 10.0);
        assert_eq!(result.item_at(&[2])?, 18.0);
        Ok(())
    }

    #[test]
    fn test_division_basic() -> Result<()> {
        let (a, b) = create_test_tensors()?;
        let result = (&b / &a)?;
        
        assert_eq!(result.item_at(&[0])?, 4.0);
        assert_eq!(result.item_at(&[1])?, 2.5);
        assert_eq!(result.item_at(&[2])?, 2.0);
        Ok(())
    }

    #[test]
    fn test_scalar_operations() -> Result<()> {
        let a = Tensor::new(vec![5.0], Shape::from(&[1]))?;
        let b = Tensor::new(vec![3.0], Shape::from(&[1]))?;
        
        let add_result = (&a + &b)?;
        assert_eq!(add_result.item()?, 8.0);
        
        let mul_result = (&a * &b)?;
        assert_eq!(mul_result.item()?, 15.0);
        
        Ok(())
    }

    #[test]
    fn test_large_tensors() -> Result<()> {
        // Create 1000-element tensors
        let data_a: Vec<f32> = (0..1000).map(|i| i as f32).collect();
        let data_b: Vec<f32> = (0..1000).map(|i| (i + 1) as f32).collect();
        
        let a = Tensor::new(data_a, Shape::from(&[1000]))?;
        let b = Tensor::new(data_b, Shape::from(&[1000]))?;
        
        let result = (&a + &b)?;
        
        // Check first few elements
        assert_eq!(result.item_at(&[0])?, 1.0);   // 0 + 1
        assert_eq!(result.item_at(&[1])?, 3.0);   // 1 + 2
        assert_eq!(result.item_at(&[999])?, 1999.0); // 999 + 1000
        
        Ok(())
    }

    #[test]
    fn test_matrix_operations() -> Result<()> {
        // 2x3 matrices
        let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], Shape::from(&[2, 3]))?;
        let b = Tensor::new(vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0], Shape::from(&[2, 3]))?;
        
        let result = (&a + &b)?;
        
        assert_eq!(result.item_at(&[0, 0])?, 8.0);   // 1 + 7
        assert_eq!(result.item_at(&[0, 1])?, 10.0);  // 2 + 8
        assert_eq!(result.item_at(&[1, 2])?, 18.0);  // 6 + 12
        
        Ok(())
    }

    #[test]
    fn test_3d_tensors() -> Result<()> {
        // 2x2x2 tensor
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let a = Tensor::new(data.clone(), Shape::from(&[2, 2, 2]))?;
        let b = Tensor::new(data, Shape::from(&[2, 2, 2]))?;
        
        let result = (&a + &b)?;
        
        assert_eq!(result.item_at(&[0, 0, 0])?, 2.0);  // 1 + 1
        assert_eq!(result.item_at(&[1, 1, 1])?, 16.0); // 8 + 8
        
        Ok(())
    }

    #[test]
    fn test_zero_operations() -> Result<()> {
        let a = Tensor::<f32>::zeros(Shape::from(&[3]))?;
        let b = Tensor::<f32>::ones(Shape::from(&[3]))?;
        
        let add_result = (&a + &b)?;
        assert_eq!(add_result.item_at(&[0])?, 1.0);
        
        let mul_result = (&a * &b)?;
        assert_eq!(mul_result.item_at(&[0])?, 0.0);
        
        Ok(())
    }

    #[test]
    fn test_negative_numbers() -> Result<()> {
        let a = Tensor::new(vec![-1.0, -2.0, -3.0], Shape::from(&[3]))?;
        let b = Tensor::new(vec![1.0, 2.0, 3.0], Shape::from(&[3]))?;
        
        let add_result = (&a + &b)?;
        assert_eq!(add_result.item_at(&[0])?, 0.0);
        assert_eq!(add_result.item_at(&[1])?, 0.0);
        assert_eq!(add_result.item_at(&[2])?, 0.0);
        
        let mul_result = (&a * &b)?;
        assert_eq!(mul_result.item_at(&[0])?, -1.0);
        assert_eq!(mul_result.item_at(&[1])?, -4.0);
        assert_eq!(mul_result.item_at(&[2])?, -9.0);
        
        Ok(())
    }

    #[test]
    fn test_shape_mismatch() {
        let a = Tensor::new(vec![1.0, 2.0], Shape::from(&[2])).unwrap();
        let b = Tensor::new(vec![1.0, 2.0, 3.0], Shape::from(&[3])).unwrap();
        
        assert!((&a + &b).is_err());
        assert!((&a * &b).is_err());
        assert!((&a - &b).is_err());
        assert!((&a / &b).is_err());
    }

    #[test]
    fn test_division_by_zero() {
        let a = Tensor::new(vec![1.0, 2.0], Shape::from(&[2])).unwrap();
        let b = Tensor::new(vec![0.0, 1.0], Shape::from(&[2])).unwrap();
        
        assert!((&a / &b).is_err());
    }

    #[test]
    fn test_chained_operations() -> Result<()> {
        let a = Tensor::new(vec![2.0], Shape::from(&[1]))?;
        let b = Tensor::new(vec![3.0], Shape::from(&[1]))?;
        let c = Tensor::new(vec![4.0], Shape::from(&[1]))?;
        
        // (a + b) * c
        let temp = (&a + &b)?;
        let result = (&temp * &c)?;
        
        assert_eq!(result.item()?, 20.0); // (2 + 3) * 4 = 20
        
        Ok(())
    }

    #[test]
    fn test_requires_grad_propagation() -> Result<()> {
        let a = Tensor::new(vec![1.0], Shape::from(&[1]))?.requires_grad(true);
        let b = Tensor::new(vec![2.0], Shape::from(&[1]))?; // no grad
        
        let result = (&a + &b)?;
        assert!(result.requires_grad_flag());
        
        let result2 = (&b + &a)?;
        assert!(result2.requires_grad_flag());
        
        Ok(())
    }
} 