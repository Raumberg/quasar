//! Tests for memory management

use quasar::memory::AlignedVec;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_aligned_vec_creation() -> Result<(), Box<dyn std::error::Error>> {
        let vec = AlignedVec::<f32>::new(100)?;
        
        assert_eq!(vec.len(), 100);
        assert_eq!(vec.capacity(), 100);
        
        // Check alignment
        let ptr = vec.as_ptr() as usize;
        assert_eq!(ptr % 64, 0); // Should be 64-byte aligned
        
        Ok(())
    }

    #[test]
    fn test_aligned_vec_from_vec() -> Result<(), Box<dyn std::error::Error>> {
        let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
        let aligned = AlignedVec::from_vec(data)?;
        
        assert_eq!(aligned.len(), 5);
        assert_eq!(aligned[0], 1.0);
        assert_eq!(aligned[4], 5.0);
        
        // Check alignment
        let ptr = aligned.as_ptr() as usize;
        assert_eq!(ptr % 64, 0);
        
        Ok(())
    }

    #[test]
    fn test_aligned_vec_indexing() -> Result<(), Box<dyn std::error::Error>> {
        let mut vec = AlignedVec::<f32>::new(5)?;
        
        // Test mutable indexing
        vec[0] = 10.0;
        vec[1] = 20.0;
        vec[4] = 50.0;
        
        // Test immutable indexing
        assert_eq!(vec[0], 10.0);
        assert_eq!(vec[1], 20.0);
        assert_eq!(vec[4], 50.0);
        
        Ok(())
    }

    #[test]
    fn test_aligned_vec_fill() -> Result<(), Box<dyn std::error::Error>> {
        let mut vec = AlignedVec::<f32>::new(100)?;
        vec.fill(42.0);
        
        for i in 0..100 {
            assert_eq!(vec[i], 42.0);
        }
        
        Ok(())
    }

    #[test]
    fn test_aligned_vec_clone() -> Result<(), Box<dyn std::error::Error>> {
        let mut original = AlignedVec::<f32>::new(5)?;
        original[0] = 1.0;
        original[1] = 2.0;
        original[2] = 3.0;
        
        let cloned = original.clone();
        
        assert_eq!(original.len(), cloned.len());
        assert_eq!(original[0], cloned[0]);
        assert_eq!(original[1], cloned[1]);
        assert_eq!(original[2], cloned[2]);
        
        // Check that both are properly aligned
        let ptr1 = original.as_ptr() as usize;
        let ptr2 = cloned.as_ptr() as usize;
        assert_eq!(ptr1 % 64, 0);
        assert_eq!(ptr2 % 64, 0);
        
        Ok(())
    }

    #[test]
    fn test_aligned_vec_large() -> Result<(), Box<dyn std::error::Error>> {
        // Test with large allocation
        let vec = AlignedVec::<f64>::new(10000)?;
        
        assert_eq!(vec.len(), 10000);
        
        // Check alignment for f64
        let ptr = vec.as_ptr() as usize;
        assert_eq!(ptr % 64, 0);
        
        Ok(())
    }

    #[test]
    fn test_aligned_vec_zero_size() {
        // Should handle zero-size allocation gracefully
        let result = AlignedVec::<f32>::new(0);
        assert!(result.is_ok());
        
        let vec = result.unwrap();
        assert_eq!(vec.len(), 0);
    }

    #[test]
    fn test_aligned_vec_as_slice() -> Result<(), Box<dyn std::error::Error>> {
        let data = vec![1.0f32, 2.0, 3.0, 4.0];
        let aligned = AlignedVec::from_vec(data)?;
        
        let slice = aligned.as_slice();
        assert_eq!(slice.len(), 4);
        assert_eq!(slice[0], 1.0);
        assert_eq!(slice[3], 4.0);
        
        Ok(())
    }

    #[test]
    fn test_aligned_vec_as_mut_slice() -> Result<(), Box<dyn std::error::Error>> {
        let mut aligned = AlignedVec::<f32>::new(4)?;
        
        {
            let slice = aligned.as_mut_slice();
            slice[0] = 10.0;
            slice[1] = 20.0;
            slice[2] = 30.0;
            slice[3] = 40.0;
        }
        
        assert_eq!(aligned[0], 10.0);
        assert_eq!(aligned[1], 20.0);
        assert_eq!(aligned[2], 30.0);
        assert_eq!(aligned[3], 40.0);
        
        Ok(())
    }

    #[test]
    fn test_aligned_vec_different_types() -> Result<(), Box<dyn std::error::Error>> {
        // Test with different numeric types
        let vec_f32 = AlignedVec::<f32>::new(10)?;
        let vec_f64 = AlignedVec::<f64>::new(10)?;
        let vec_i32 = AlignedVec::<i32>::new(10)?;
        
        // All should be properly aligned
        assert_eq!(vec_f32.as_ptr() as usize % 64, 0);
        assert_eq!(vec_f64.as_ptr() as usize % 64, 0);
        assert_eq!(vec_i32.as_ptr() as usize % 64, 0);
        
        Ok(())
    }

    #[test]
    fn test_memory_layout_consistency() -> Result<(), Box<dyn std::error::Error>> {
        // Test that memory layout is consistent across operations
        let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let aligned = AlignedVec::from_vec(data)?;
        
        // Clone and verify data integrity
        let cloned = aligned.clone();
        
        for i in 0..8 {
            assert_eq!(aligned[i], cloned[i]);
            assert_eq!(aligned[i], (i + 1) as f32);
        }
        
        Ok(())
    }
} 