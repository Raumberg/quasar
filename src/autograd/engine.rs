//! Global autograd engine for automatic differentiation

use crate::core::{Tensor, TensorElement};
use crate::error::{QuasarError, Result};
use crate::autograd::graph::{ComputationGraph, NodeId, Operation};
use std::collections::HashMap;
use std::sync::{Mutex, Arc, OnceLock};
use std::any::TypeId;
use std::cell::RefCell;

/// Thread-local flag to track if we're in backward pass
thread_local! {
    static IN_BACKWARD_PASS: RefCell<bool> = RefCell::new(false);
}

/// Check if we're currently in a backward pass
pub fn is_in_backward_pass() -> bool {
    IN_BACKWARD_PASS.with(|flag| *flag.borrow())
}

/// Execute function with backward pass context
pub fn with_backward_context<F, R>(f: F) -> R
where
    F: FnOnce() -> R,
{
    IN_BACKWARD_PASS.with(|flag| {
        let old_value = *flag.borrow();
        *flag.borrow_mut() = true;
        let result = f();
        *flag.borrow_mut() = old_value;
        result
    })
}

/// Global autograd engines storage - thread-safe initialization
static GLOBAL_ENGINES: OnceLock<GlobalEngines> = OnceLock::new();

/// Container for all global engines
struct GlobalEngines {
    f32_engine: Arc<Mutex<AutogradEngine<f32>>>,
    f64_engine: Arc<Mutex<AutogradEngine<f64>>>,
}

impl GlobalEngines {
    fn new() -> Self {
        Self {
            f32_engine: Arc::new(Mutex::new(AutogradEngine::new())),
            f64_engine: Arc::new(Mutex::new(AutogradEngine::new())),
        }
    }
}

/// Main autograd engine
pub struct AutogradEngine<T: TensorElement> {
    /// Computational graph
    graph: ComputationGraph<T>,
    /// Whether gradient computation is enabled
    enabled: bool,
}

impl<T: TensorElement> AutogradEngine<T> {
    /// Create new autograd engine
    pub fn new() -> Self {
        Self {
            graph: ComputationGraph::new(),
            enabled: true,
        }
    }

    /// Enable gradient computation
    pub fn enable(&mut self) {
        self.enabled = true;
    }

    /// Disable gradient computation
    pub fn disable(&mut self) {
        self.enabled = false;
    }

    /// Check if gradients are enabled
    pub fn is_enabled(&self) -> bool {
        self.enabled && !is_in_backward_pass()
    }

    /// Register a leaf tensor (requires_grad=true)
    pub fn register_leaf(&mut self, tensor: Tensor<T>) -> NodeId {
        self.graph.add_leaf_simple(tensor)
    }

    /// Find existing leaf node by tensor data (to avoid duplicates)
    pub fn find_leaf_by_data(&self, tensor: &Tensor<T>) -> Option<NodeId> {
        self.graph.find_leaf_by_data(tensor)
    }

    /// Get or register a leaf tensor (avoids duplicates)
    pub fn get_or_register_leaf(&mut self, tensor: &Tensor<T>) -> NodeId {
        if let Some(node_id) = self.find_leaf_by_data(tensor) {
            node_id
        } else {
            self.register_leaf(tensor.clone())
        }
    }

    /// Register an operation
    pub fn register_operation(
        &mut self,
        operation: Operation<T>,
        input_nodes: Vec<NodeId>,
        output: Tensor<T>,
        grad_fn: Box<dyn Fn(&Tensor<T>, &[&Tensor<T>]) -> Result<Vec<Tensor<T>>> + Send + Sync>,
    ) -> Result<NodeId> {
        if !self.is_enabled() {
            return Err(QuasarError::invalid_operation("Autograd is disabled or in backward pass"));
        }

        // Add operation node
        let node_id = self.graph.add_operation_simple(operation, input_nodes, output, grad_fn);
        Ok(node_id)
    }

    /// Execute backward pass from given node
    pub fn backward(&mut self, node_id: NodeId, grad_output: Tensor<T>) -> Result<()> {
        with_backward_context(|| {
            self.graph.backward(node_id, grad_output)
        })
    }

    /// Get gradient for a node
    pub fn get_gradient(&self, node_id: NodeId) -> Option<&Tensor<T>> {
        self.graph.get_gradient(node_id)
    }

    /// Get all leaf node gradients
    pub fn get_leaf_gradients(&self) -> HashMap<NodeId, Tensor<T>> {
        self.graph.get_leaf_gradients()
    }

    /// Clear all gradients
    pub fn zero_grad(&mut self) {
        // Only clear gradients, not the entire graph
        self.graph.clear_gradients();
    }

    /// Get graph statistics
    pub fn stats(&self) -> AutogradStats {
        AutogradStats {
            node_count: self.graph.node_count(),
            enabled: self.enabled,
        }
    }
}

/// Statistics about the autograd engine
#[derive(Debug, Clone)]
pub struct AutogradStats {
    pub node_count: usize,
    pub enabled: bool,
}

/// Get global engine for type T
pub fn get_global_engine<T: TensorElement>() -> Arc<Mutex<AutogradEngine<T>>> {
    let engines = GLOBAL_ENGINES.get_or_init(GlobalEngines::new);
    
    if TypeId::of::<T>() == TypeId::of::<f32>() {
        // SAFETY: We've checked the type ID matches f32
        unsafe { std::mem::transmute(engines.f32_engine.clone()) }
    } else if TypeId::of::<T>() == TypeId::of::<f64>() {
        // SAFETY: We've checked the type ID matches f64
        unsafe { std::mem::transmute(engines.f64_engine.clone()) }
    } else {
        // For other types, create a new engine (shouldn't happen in practice)
        Arc::new(Mutex::new(AutogradEngine::new()))
    }
}

/// Execute operation with global engine
pub fn with_global_engine<T: TensorElement, F, R>(f: F) -> R
where
    F: FnOnce(&mut AutogradEngine<T>) -> R,
{
    let engine = get_global_engine::<T>();
    let mut engine_guard = engine.lock().unwrap();
    f(&mut *engine_guard)
}

/// Autograd operation trait
pub trait AutogradOp<T: TensorElement>: Send + Sync {
    /// Forward pass
    fn forward(&self, inputs: &[&Tensor<T>]) -> Result<Tensor<T>>;
    
    /// Backward pass - compute gradients for inputs
    fn backward(&self, grad_output: &Tensor<T>, inputs: &[&Tensor<T>]) -> Result<Vec<Tensor<T>>>;
}

/// Addition operation
pub struct AddOp;

/// Subtraction operation
pub struct SubOp;

/// Multiplication operation
pub struct MulOp;

/// Division operation
pub struct DivOp;

impl<T: TensorElement> Default for AutogradEngine<T> {
    fn default() -> Self {
        Self::new()
    }
} 