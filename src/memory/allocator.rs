//! Custom allocators for tensor memory management

use std::alloc::{Layout, GlobalAlloc, System};

/// Trait for tensor-specific allocators
pub trait TensorAllocator {
    /// Allocate aligned memory
    unsafe fn alloc(&self, layout: Layout) -> *mut u8;
    
    /// Deallocate memory
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout);
}

/// SIMD-aligned allocator
pub struct SimdAllocator {
    alignment: usize,
}

impl SimdAllocator {
    /// Create new SIMD allocator with specified alignment
    pub fn new(alignment: usize) -> Self {
        Self { alignment }
    }
    
    /// Create allocator with default SIMD alignment (64 bytes)
    pub fn default_simd() -> Self {
        Self::new(64)
    }
}

impl TensorAllocator for SimdAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let aligned_layout = Layout::from_size_align(
            layout.size(),
            self.alignment.max(layout.align())
        ).unwrap();
        
        unsafe { System.alloc(aligned_layout) }
    }
    
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        let aligned_layout = Layout::from_size_align(
            layout.size(),
            self.alignment.max(layout.align())
        ).unwrap();
        
        unsafe { System.dealloc(ptr, aligned_layout); }
    }
} 