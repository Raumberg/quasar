//! Computational graph for automatic differentiation

use crate::core::{Tensor, TensorElement};
use crate::error::{QuasarError, Result};
use std::collections::{HashMap, HashSet};
use std::time::{Duration, Instant};

/// Unique identifier for nodes in the computational graph
pub type NodeId = usize;

/// Configuration for graph lifecycle management
#[derive(Debug, Clone)]
pub struct GraphConfig {
    /// Maximum number of nodes before triggering cleanup
    pub max_nodes: usize,
    /// Maximum age of unused nodes before cleanup (in seconds)
    pub max_node_age_secs: u64,
    /// Enable automatic garbage collection
    pub auto_gc: bool,
}

impl Default for GraphConfig {
    fn default() -> Self {
        Self {
            max_nodes: 10000,
            max_node_age_secs: 300, // 5 minutes
            auto_gc: true,
        }
    }
}

/// Metadata for graph nodes
#[derive(Debug)]
struct NodeMetadata {
    /// When this node was created
    created_at: Instant,
    /// When this node was last accessed
    last_accessed: Instant,
    /// Reference count (how many other nodes depend on this)
    ref_count: usize,
}

impl NodeMetadata {
    fn new() -> Self {
        let now = Instant::now();
        Self {
            created_at: now,
            last_accessed: now,
            ref_count: 0,
        }
    }

    fn touch(&mut self) {
        self.last_accessed = Instant::now();
    }

    fn age(&self) -> Duration {
        self.created_at.elapsed()
    }

    fn idle_time(&self) -> Duration {
        self.last_accessed.elapsed()
    }
}

/// Computational graph node
pub struct ComputationNode<T: TensorElement> {
    /// Unique identifier
    pub id: NodeId,
    /// Operation that created this node
    pub operation: Operation<T>,
    /// Input node IDs
    pub inputs: Vec<NodeId>,
    /// Output tensor stored directly
    pub output: Tensor<T>,
    /// Gradient function for backward pass
    pub grad_fn: Option<Box<dyn Fn(&Tensor<T>, &[&Tensor<T>]) -> Result<Vec<Tensor<T>>> + Send + Sync>>,
    /// Node metadata for lifecycle management
    metadata: NodeMetadata,
}

/// Types of operations in the computational graph
#[derive(Debug, Clone)]
pub enum Operation<T: TensorElement> {
    /// Leaf node (input tensor)
    Leaf,
    /// Addition operation
    Add,
    /// Subtraction operation
    Sub,
    /// Multiplication operation
    Mul,
    /// Division operation
    Div,
    /// Matrix multiplication
    MatMul,
    /// ReLU activation
    ReLU,
    /// Custom operation
    Custom(String),
    /// Placeholder for other operations
    _Phantom(std::marker::PhantomData<T>),
}

/// Main computational graph
pub struct ComputationGraph<T: TensorElement> {
    /// All nodes in the graph
    nodes: HashMap<NodeId, ComputationNode<T>>,
    /// Next available node ID
    next_id: NodeId,
    /// Tensors that require gradients (leaf nodes)
    leaf_nodes: HashSet<NodeId>,
    /// Gradient storage for each node
    gradients: HashMap<NodeId, Tensor<T>>,
    /// Configuration for graph lifecycle management
    config: GraphConfig,
}

impl<T: TensorElement> ComputationGraph<T> {
    /// Create new computational graph
    pub fn new() -> Self {
        Self {
            nodes: HashMap::new(),
            next_id: 0,
            leaf_nodes: HashSet::new(),
            gradients: HashMap::new(),
            config: GraphConfig::default(),
        }
    }

    /// Create new computational graph with custom config
    pub fn with_config(config: GraphConfig) -> Self {
        Self {
            nodes: HashMap::new(),
            next_id: 0,
            leaf_nodes: HashSet::new(),
            gradients: HashMap::new(),
            config,
        }
    }

    /// Add a leaf node (input tensor) - simplified version
    pub fn add_leaf_simple(&mut self, tensor: Tensor<T>) -> NodeId {
        let id = self.next_id;
        self.next_id += 1;

        let node = ComputationNode {
            id,
            operation: Operation::Leaf,
            inputs: Vec::new(),
            output: tensor,
            grad_fn: None,
            metadata: NodeMetadata::new(),
        };

        self.nodes.insert(id, node);
        self.leaf_nodes.insert(id);
        id
    }

    /// Find existing leaf node by tensor data (to avoid duplicates)
    pub fn find_leaf_by_data(&self, tensor: &Tensor<T>) -> Option<NodeId> {
        for &leaf_id in &self.leaf_nodes {
            if let Some(node) = self.nodes.get(&leaf_id) {
                // Compare tensor data and shape
                if node.output.data() == tensor.data() && node.output.shape() == tensor.shape() {
                    return Some(leaf_id);
                }
            }
        }
        None
    }

    /// Add an operation node - simplified version
    pub fn add_operation_simple(
        &mut self,
        operation: Operation<T>,
        inputs: Vec<NodeId>,
        output: Tensor<T>,
        grad_fn: Box<dyn Fn(&Tensor<T>, &[&Tensor<T>]) -> Result<Vec<Tensor<T>>> + Send + Sync>,
    ) -> NodeId {
        let id = self.next_id;
        self.next_id += 1;

        let node = ComputationNode {
            id,
            operation,
            inputs,
            output,
            grad_fn: Some(grad_fn),
            metadata: NodeMetadata::new(),
        };

        self.nodes.insert(id, node);
        id
    }

    /// Perform topological sort for backward pass
    pub fn topological_sort(&self, start_node: NodeId) -> Result<Vec<NodeId>> {
        let mut visited = HashSet::new();
        let mut temp_visited = HashSet::new();
        let mut result = Vec::new();

        self.dfs_topological(start_node, &mut visited, &mut temp_visited, &mut result)?;
        
        result.reverse(); // Reverse to get correct order for backward pass
        Ok(result)
    }

    /// Depth-first search for topological sorting
    fn dfs_topological(
        &self,
        node_id: NodeId,
        visited: &mut HashSet<NodeId>,
        temp_visited: &mut HashSet<NodeId>,
        result: &mut Vec<NodeId>,
    ) -> Result<()> {
        if temp_visited.contains(&node_id) {
            return Err(QuasarError::invalid_operation("Cycle detected in computational graph"));
        }
        
        if visited.contains(&node_id) {
            return Ok(());
        }

        temp_visited.insert(node_id);

        if let Some(node) = self.nodes.get(&node_id) {
            for &input_id in &node.inputs {
                self.dfs_topological(input_id, visited, temp_visited, result)?;
            }
        }

        temp_visited.remove(&node_id);
        visited.insert(node_id);
        result.push(node_id);

        Ok(())
    }

    /// Execute backward pass from given node
    pub fn backward(&mut self, start_node: NodeId, grad_output: Tensor<T>) -> Result<()> {
        // Clear previous gradients
        self.gradients.clear();

        // Initialize gradient for output node
        self.gradients.insert(start_node, grad_output);

        // Get topological order
        let topo_order = self.topological_sort(start_node)?;

        // Process nodes in reverse topological order
        for &node_id in &topo_order {
            // Get node information first to avoid borrowing conflicts
            let (has_grad_fn, input_ids, input_tensors) = {
                let node = match self.nodes.get(&node_id) {
                    Some(node) if node.grad_fn.is_some() => node,
                    _ => continue,
                };

                // Get input tensors
                let mut input_tensors = Vec::new();
                for &input_id in &node.inputs {
                    if let Some(input_node) = self.nodes.get(&input_id) {
                        input_tensors.push(input_node.output.clone());
                    }
                }

                (true, node.inputs.clone(), input_tensors)
            };

            if !has_grad_fn {
                continue;
            }

            // Get gradient for this node
            let node_grad = match self.gradients.get(&node_id) {
                Some(grad) => grad.clone(),
                None => return Err(QuasarError::invalid_operation("Missing gradient for node")),
            };

            // Get the gradient function and compute gradients
            let input_grads = {
                let node = self.nodes.get(&node_id).unwrap();
                if let Some(ref grad_fn) = node.grad_fn {
                    let input_tensor_refs: Vec<&Tensor<T>> = input_tensors.iter().collect();
                    grad_fn(&node_grad, &input_tensor_refs)?
                } else {
                    continue;
                }
            };

            // Accumulate gradients for input nodes
            for (i, &input_id) in input_ids.iter().enumerate() {
                if i < input_grads.len() {
                    self.accumulate_gradient(input_id, input_grads[i].clone())?;
                }
            }
        }

        Ok(())
    }

    /// Accumulate gradient for a node (sum if gradient already exists)
    fn accumulate_gradient(&mut self, node_id: NodeId, grad: Tensor<T>) -> Result<()> {
        if let Some(existing_grad) = self.gradients.get(&node_id) {
            // Sum gradients
            let summed_grad = crate::ops::arithmetic::add(existing_grad, &grad)?;
            self.gradients.insert(node_id, summed_grad);
        } else {
            self.gradients.insert(node_id, grad);
        }
        Ok(())
    }

    /// Get gradient for a node
    pub fn get_gradient(&self, node_id: NodeId) -> Option<&Tensor<T>> {
        self.gradients.get(&node_id)
    }

    /// Get all leaf node gradients
    pub fn get_leaf_gradients(&self) -> HashMap<NodeId, Tensor<T>> {
        let mut result = HashMap::new();
        for &leaf_id in &self.leaf_nodes {
            if let Some(grad) = self.gradients.get(&leaf_id) {
                result.insert(leaf_id, grad.clone());
            }
        }
        result
    }

    /// Check if node exists
    pub fn has_node(&self, node_id: NodeId) -> bool {
        self.nodes.contains_key(&node_id)
    }

    /// Get node count
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Clear the graph
    pub fn clear(&mut self) {
        self.nodes.clear();
        self.leaf_nodes.clear();
        self.gradients.clear();
        self.next_id = 0;
    }

    /// Perform automatic garbage collection if needed
    pub fn maybe_gc(&mut self) {
        if !self.config.auto_gc {
            return;
        }

        // Check if we need to trigger GC
        let should_gc = self.nodes.len() > self.config.max_nodes ||
                       self.has_old_nodes();

        if should_gc {
            self.garbage_collect();
        }
    }

    /// Check if there are old unused nodes
    fn has_old_nodes(&self) -> bool {
        let max_age = Duration::from_secs(self.config.max_node_age_secs);
        self.nodes.values().any(|node| {
            node.metadata.ref_count == 0 && node.metadata.idle_time() > max_age
        })
    }

    /// Perform garbage collection - remove unused old nodes
    pub fn garbage_collect(&mut self) {
        let max_age = Duration::from_secs(self.config.max_node_age_secs);
        let mut to_remove = Vec::new();

        // Update reference counts first
        self.update_ref_counts();

        // Find nodes to remove
        for (id, node) in &self.nodes {
            // Don't remove leaf nodes or recently used nodes
            if self.leaf_nodes.contains(id) {
                continue;
            }

            // Remove if old and unused
            if node.metadata.ref_count == 0 && node.metadata.idle_time() > max_age {
                to_remove.push(*id);
            }
        }

        // Remove nodes
        for id in to_remove {
            self.nodes.remove(&id);
            self.gradients.remove(&id);
        }
    }

    /// Update reference counts for all nodes
    fn update_ref_counts(&mut self) {
        // Reset all ref counts
        for node in self.nodes.values_mut() {
            node.metadata.ref_count = 0;
        }

        // Collect all input dependencies first
        let mut ref_counts: HashMap<NodeId, usize> = HashMap::new();
        for node in self.nodes.values() {
            for &input_id in &node.inputs {
                *ref_counts.entry(input_id).or_insert(0) += 1;
            }
        }

        // Update ref counts
        for (node_id, count) in ref_counts {
            if let Some(node) = self.nodes.get_mut(&node_id) {
                node.metadata.ref_count = count;
            }
        }
    }

    /// Get graph statistics including memory usage
    pub fn graph_stats(&self) -> GraphStats {
        let total_nodes = self.nodes.len();
        let leaf_nodes = self.leaf_nodes.len();
        let gradient_nodes = self.gradients.len();
        
        let old_nodes = self.nodes.values()
            .filter(|node| {
                let max_age = Duration::from_secs(self.config.max_node_age_secs);
                node.metadata.idle_time() > max_age
            })
            .count();

        GraphStats {
            total_nodes,
            leaf_nodes,
            gradient_nodes,
            old_nodes,
            next_id: self.next_id,
            config: self.config.clone(),
        }
    }
}

impl<T: TensorElement> Default for ComputationGraph<T> {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics about the computational graph
#[derive(Debug, Clone)]
pub struct GraphStats {
    pub total_nodes: usize,
    pub leaf_nodes: usize,
    pub gradient_nodes: usize,
    pub old_nodes: usize,
    pub next_id: NodeId,
    pub config: GraphConfig,
} 