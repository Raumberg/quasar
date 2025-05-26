//! Computational backends

pub mod cpu;

#[cfg(feature = "gpu")]
pub mod gpu;

// Re-exports
pub use cpu::CpuBackend; 