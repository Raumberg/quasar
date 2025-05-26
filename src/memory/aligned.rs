//! Memory management for SIMD-aligned data

use crate::error::{QuasarError, Result};
use std::alloc::{alloc, dealloc, Layout};
use std::ops::{Deref, DerefMut, Index, IndexMut};
use std::ptr::NonNull;
use std::slice;

/// SIMD alignment (32 bytes for AVX2, 64 bytes for AVX512)
const SIMD_ALIGNMENT: usize = 64;

/// Vector with SIMD-aligned memory
#[derive(Debug)]
pub struct AlignedVec<T> {
    ptr: NonNull<T>,
    len: usize,
    capacity: usize,
}

impl<T> AlignedVec<T> {
    /// Create new aligned vector with given capacity
    pub fn new(capacity: usize) -> Result<Self> {
        if capacity == 0 {
            return Ok(Self {
                ptr: NonNull::dangling(),
                len: 0,
                capacity: 0,
            });
        }

        let layout = Self::layout_for_capacity(capacity)?;
        let ptr = unsafe { alloc(layout) as *mut T };
        
        if ptr.is_null() {
            return Err(QuasarError::memory_error("Failed to allocate aligned memory"));
        }

        Ok(Self {
            ptr: unsafe { NonNull::new_unchecked(ptr) },
            len: capacity,
            capacity,
        })
    }

    /// Create aligned vector from existing Vec
    pub fn from_vec(mut vec: Vec<T>) -> Result<Self> {
        let len = vec.len();
        let capacity = vec.capacity();
        
        if len == 0 {
            return Ok(Self {
                ptr: NonNull::dangling(),
                len: 0,
                capacity: 0,
            });
        }

        // Allocate aligned memory
        let mut aligned = Self::new(len)?;
        
        // Copy data
        unsafe {
            std::ptr::copy_nonoverlapping(
                vec.as_ptr(),
                aligned.ptr.as_ptr(),
                len
            );
        }
        
        // Prevent vec from dropping its data
        unsafe { vec.set_len(0); }
        
        Ok(aligned)
    }

    /// Get layout for given capacity
    fn layout_for_capacity(capacity: usize) -> Result<Layout> {
        Layout::from_size_align(
            capacity * std::mem::size_of::<T>(),
            SIMD_ALIGNMENT
        ).map_err(|_| QuasarError::memory_error("Invalid memory layout"))
    }

    /// Get length
    pub fn len(&self) -> usize {
        self.len
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Get capacity
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Get raw pointer
    pub fn as_ptr(&self) -> *const T {
        if self.capacity == 0 {
            std::ptr::null()
        } else {
            self.ptr.as_ptr()
        }
    }

    /// Get mutable raw pointer
    pub fn as_mut_ptr(&mut self) -> *mut T {
        if self.capacity == 0 {
            std::ptr::null_mut()
        } else {
            self.ptr.as_ptr()
        }
    }

    /// Get as slice
    pub fn as_slice(&self) -> &[T] {
        if self.len == 0 {
            &[]
        } else {
            unsafe { slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
        }
    }

    /// Get as mutable slice
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        if self.len == 0 {
            &mut []
        } else {
            unsafe { slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
        }
    }

    /// Fill with value
    pub fn fill(&mut self, value: T) 
    where 
        T: Copy 
    {
        for i in 0..self.len {
            unsafe {
                *self.ptr.as_ptr().add(i) = value;
            }
        }
    }

    /// Check if memory is properly aligned
    pub fn is_aligned(&self) -> bool {
        (self.ptr.as_ptr() as usize) % SIMD_ALIGNMENT == 0
    }
}

impl<T> Drop for AlignedVec<T> {
    fn drop(&mut self) {
        if self.capacity > 0 {
            unsafe {
                // Drop all elements
                for i in 0..self.len {
                    std::ptr::drop_in_place(self.ptr.as_ptr().add(i));
                }
                
                // Deallocate memory
                let layout = Self::layout_for_capacity(self.capacity).unwrap();
                dealloc(self.ptr.as_ptr() as *mut u8, layout);
            }
        }
    }
}

impl<T> Deref for AlignedVec<T> {
    type Target = [T];

    fn deref(&self) -> &Self::Target {
        self.as_slice()
    }
}

impl<T> DerefMut for AlignedVec<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.as_mut_slice()
    }
}

impl<T> Index<usize> for AlignedVec<T> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        &self.as_slice()[index]
    }
}

impl<T> IndexMut<usize> for AlignedVec<T> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.as_mut_slice()[index]
    }
}

impl<T: Clone> Clone for AlignedVec<T> {
    fn clone(&self) -> Self {
        let mut new_vec = Self::new(self.len).unwrap();
        for i in 0..self.len {
            unsafe {
                *new_vec.ptr.as_ptr().add(i) = self[i].clone();
            }
        }
        new_vec
    }
}

unsafe impl<T: Send> Send for AlignedVec<T> {}
unsafe impl<T: Sync> Sync for AlignedVec<T> {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_aligned_vec_creation() {
        let vec = AlignedVec::<f32>::new(100).unwrap();
        assert_eq!(vec.len(), 100);
        assert_eq!(vec.capacity(), 100);
        assert!(vec.is_aligned());
    }

    #[test]
    fn test_from_vec() {
        let data = vec![1.0f32, 2.0, 3.0, 4.0];
        let aligned = AlignedVec::from_vec(data).unwrap();
        assert_eq!(aligned.len(), 4);
        assert_eq!(aligned[0], 1.0);
        assert_eq!(aligned[3], 4.0);
        assert!(aligned.is_aligned());
    }

    #[test]
    fn test_fill() {
        let mut vec = AlignedVec::<f32>::new(10).unwrap();
        vec.fill(42.0);
        for i in 0..10 {
            assert_eq!(vec[i], 42.0);
        }
    }
} 