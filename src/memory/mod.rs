//! Memory management for high-performance tensor operations

pub mod aligned;
pub mod allocator;
pub mod pool;

// Re-exports
pub use aligned::AlignedVec;
pub use allocator::{TensorAllocator, SimdAllocator};
pub use pool::MemoryPool; 