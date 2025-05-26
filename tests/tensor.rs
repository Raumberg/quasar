//! Tests for tensor functionality

use quasar::prelude::*;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_creation() -> Result<()> {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let tensor = Tensor::new(data, Shape::from(&[2, 2]))?;
        
        assert_eq!(tensor.shape().dims(), &[2, 2]);
        assert_eq!(tensor.numel(), 4);
        assert_eq!(tensor.ndim(), 2);
        assert!(!tensor.is_scalar());
        
        Ok(())
    }

    #[test]
    fn test_scalar_tensor() -> Result<()> {
        let tensor = Tensor::new(vec![42.0], Shape::from(&[1]))?;
        
        assert!(tensor.is_scalar());
        assert_eq!(tensor.item()?, 42.0);
        
        Ok(())
    }

    #[test]
    fn test_zeros_and_ones() -> Result<()> {
        let zeros = Tensor::<f32>::zeros(Shape::from(&[3, 2]))?;
        let ones = Tensor::<f32>::ones(Shape::from(&[3, 2]))?;
        
        assert_eq!(zeros.shape().dims(), &[3, 2]);
        assert_eq!(ones.shape().dims(), &[3, 2]);
        
        for i in 0..3 {
            for j in 0..2 {
                assert_eq!(zeros.item_at(&[i, j])?, 0.0);
                assert_eq!(ones.item_at(&[i, j])?, 1.0);
            }
        }
        
        Ok(())
    }

    #[test]
    fn test_requires_grad() -> Result<()> {
        let tensor = Tensor::new(vec![1.0, 2.0], Shape::from(&[2]))?;
        assert!(!tensor.requires_grad_flag());
        assert!(tensor.grad().is_none());
        
        let grad_tensor = tensor.requires_grad(true);
        assert!(grad_tensor.requires_grad_flag());
        // Gradient is None until backward() is called
        assert!(grad_tensor.grad().is_none());
        
        Ok(())
    }

    #[test]
    fn test_reshape() -> Result<()> {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], Shape::from(&[2, 3]))?;
        
        let reshaped = tensor.reshape(Shape::from(&[3, 2]))?;
        assert_eq!(reshaped.shape().dims(), &[3, 2]);
        assert_eq!(reshaped.numel(), 6);
        
        // Data should be the same
        assert_eq!(reshaped.item_at(&[0, 0])?, 1.0);
        assert_eq!(reshaped.item_at(&[2, 1])?, 6.0);
        
        Ok(())
    }

    #[test]
    fn test_reshape_invalid_size() {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], Shape::from(&[2, 2])).unwrap();
        
        // Should fail - different total size
        assert!(tensor.reshape(Shape::from(&[3, 2])).is_err());
        assert!(tensor.reshape(Shape::from(&[5])).is_err());
    }

    #[test]
    fn test_item_at_access() -> Result<()> {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let tensor = Tensor::new(data, Shape::from(&[2, 3]))?;
        
        // Row-major order: [1, 2, 3]
        //                  [4, 5, 6]
        assert_eq!(tensor.item_at(&[0, 0])?, 1.0);
        assert_eq!(tensor.item_at(&[0, 1])?, 2.0);
        assert_eq!(tensor.item_at(&[0, 2])?, 3.0);
        assert_eq!(tensor.item_at(&[1, 0])?, 4.0);
        assert_eq!(tensor.item_at(&[1, 1])?, 5.0);
        assert_eq!(tensor.item_at(&[1, 2])?, 6.0);
        
        Ok(())
    }

    #[test]
    fn test_3d_tensor_access() -> Result<()> {
        let data: Vec<f32> = (1..=24).map(|i| i as f32).collect();
        let tensor = Tensor::new(data, Shape::from(&[2, 3, 4]))?;
        
        assert_eq!(tensor.ndim(), 3);
        assert_eq!(tensor.numel(), 24);
        
        // Check a few elements
        assert_eq!(tensor.item_at(&[0, 0, 0])?, 1.0);
        assert_eq!(tensor.item_at(&[0, 0, 3])?, 4.0);
        assert_eq!(tensor.item_at(&[1, 2, 3])?, 24.0);
        
        Ok(())
    }

    #[test]
    fn test_index_out_of_bounds() {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], Shape::from(&[2, 2])).unwrap();
        
        assert!(tensor.item_at(&[2, 0]).is_err()); // row out of bounds
        assert!(tensor.item_at(&[0, 2]).is_err()); // col out of bounds
        assert!(tensor.item_at(&[0]).is_err());    // wrong dimensions
        assert!(tensor.item_at(&[0, 0, 0]).is_err()); // too many dimensions
    }

    #[test]
    fn test_large_tensor() -> Result<()> {
        // Create 10000-element tensor
        let data: Vec<f32> = (0..10000).map(|i| i as f32).collect();
        let tensor = Tensor::new(data, Shape::from(&[100, 100]))?;
        
        assert_eq!(tensor.shape().dims(), &[100, 100]);
        assert_eq!(tensor.numel(), 10000);
        
        // Check a few elements
        assert_eq!(tensor.item_at(&[0, 0])?, 0.0);
        assert_eq!(tensor.item_at(&[0, 99])?, 99.0);
        assert_eq!(tensor.item_at(&[99, 99])?, 9999.0);
        
        Ok(())
    }

    #[test]
    fn test_tensor_display() -> Result<()> {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0], Shape::from(&[3]))?;
        let display_str = format!("{}", tensor);
        
        assert!(display_str.contains("1.0"));
        assert!(display_str.contains("2.0"));
        assert!(display_str.contains("3.0"));
        
        Ok(())
    }

    #[test]
    fn test_tensor_debug() -> Result<()> {
        let tensor = Tensor::new(vec![1.0, 2.0], Shape::from(&[2]))?.requires_grad(true);
        let debug_str = format!("{:?}", tensor);
        
        assert!(debug_str.contains("shape"));
        assert!(debug_str.contains("requires_grad"));
        assert!(debug_str.contains("true"));
        
        Ok(())
    }

    #[test]
    fn test_empty_tensor_creation() {
        // Should fail to create tensor with zero size
        // Empty shape creation should panic (as expected)
        let result = std::panic::catch_unwind(|| {
            Shape::from(&[] as &[usize])
        });
        assert!(result.is_err());
        
        // Zero-sized dimensions should fail
        let result = std::panic::catch_unwind(|| {
            Shape::from(&[0])
        });
        assert!(result.is_err());
        
        let result = std::panic::catch_unwind(|| {
            Shape::from(&[3, 0])
        });
        assert!(result.is_err());
    }

    #[test]
    fn test_f64_tensors() -> Result<()> {
        let tensor = Tensor::new(vec![1.0f64, 2.0, 3.0], Shape::from(&[3]))?;
        
        assert_eq!(tensor.item_at(&[0])?, 1.0f64);
        assert_eq!(tensor.item_at(&[1])?, 2.0f64);
        assert_eq!(tensor.item_at(&[2])?, 3.0f64);
        
        Ok(())
    }

    #[test]
    fn test_tensor_clone() -> Result<()> {
        let original = Tensor::new(vec![1.0, 2.0, 3.0], Shape::from(&[3]))?.requires_grad(true);
        let cloned = original.clone();
        
        assert_eq!(original.shape().dims(), cloned.shape().dims());
        assert_eq!(original.requires_grad_flag(), cloned.requires_grad_flag());
        
        // Data should be the same
        for i in 0..3 {
            assert_eq!(original.item_at(&[i])?, cloned.item_at(&[i])?);
        }
        
        Ok(())
    }

    #[test]
    fn test_item_single_element() -> Result<()> {
        let tensor = Tensor::new(vec![42.0], Shape::from(&[1]))?;
        assert_eq!(tensor.item()?, 42.0);
        
        Ok(())
    }
} 