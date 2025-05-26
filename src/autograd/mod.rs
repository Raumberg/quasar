//! Automatic differentiation engine

pub mod engine;
pub mod graph;
pub mod function;
pub mod variable;
pub mod parallel;

// Re-exports
pub use engine::{AutogradEngine, AutogradOp, AddOp, MulOp, SubOp, DivOp};
pub use graph::{ComputationGraph, ComputationNode};
pub use function::{AutogradFunction, Function};
pub use variable::Variable;
pub use parallel::{ParallelConfig, ParallelTensorOps, init_parallel_autograd, with_local_engine}; 