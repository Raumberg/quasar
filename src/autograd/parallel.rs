//! Parallel autograd engine for multi-threaded computation

use crate::core::{Tensor, TensorElement};
use crate::error::{QuasarError, Result};
use crate::autograd::graph::{ComputationGraph, NodeId, Operation};
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex, RwLock, Condvar};
use std::thread::{self, ThreadId};
use std::cell::RefCell;
use rayon::prelude::*;
use crossbeam::channel::{self, Receiver, Sender};
use std::time::Instant;
use std::sync::OnceLock;

/// Thread-local autograd engine for independent computation
thread_local! {
    static LOCAL_ENGINE: RefCell<Option<LocalAutogradEngine>> = RefCell::new(None);
}

/// Global coordinator instance with automatic initialization
static GLOBAL_COORDINATOR: OnceLock<Arc<GlobalCoordinator>> = OnceLock::new();

/// Configuration for parallel execution
#[derive(Debug, Clone)]
pub struct ParallelConfig {
    /// Maximum number of worker threads
    pub max_workers: usize,
    /// Enable automatic parallelization of operations
    pub auto_parallel: bool,
    /// Minimum tensor size for parallel operations
    pub parallel_threshold: usize,
    /// Enable thread-local engines
    pub use_thread_local: bool,
    /// Enable operation fusion
    pub enable_fusion: bool,
}

impl Default for ParallelConfig {
    fn default() -> Self {
        Self {
            max_workers: num_cpus::get(),
            auto_parallel: true,
            parallel_threshold: 1000,
            use_thread_local: true,
            enable_fusion: true,
        }
    }
}

/// Thread-local autograd engine for independent computations
pub struct LocalAutogradEngine {
    /// Local computational graph
    graph: ComputationGraph<f32>, // TODO: Make generic
    /// Thread ID for debugging
    thread_id: ThreadId,
    /// Whether this engine is enabled
    enabled: bool,
    /// Statistics
    stats: LocalEngineStats,
}

/// Statistics for local engine
#[derive(Debug, Default)]
pub struct LocalEngineStats {
    pub operations_count: usize,
    pub parallel_operations: usize,
    pub fused_operations: usize,
    pub cache_hits: usize,
}

impl LocalAutogradEngine {
    fn new() -> Self {
        Self {
            graph: ComputationGraph::new(),
            thread_id: thread::current().id(),
            enabled: true,
            stats: LocalEngineStats::default(),
        }
    }

    /// Register operation in local graph
    pub fn register_operation<T: TensorElement>(
        &mut self,
        operation: Operation<T>,
        input_nodes: Vec<NodeId>,
        output: Tensor<T>,
        grad_fn: Box<dyn Fn(&Tensor<T>, &[&Tensor<T>]) -> Result<Vec<Tensor<T>>> + Send + Sync>,
    ) -> Result<NodeId> {
        if !self.enabled {
            return Err(QuasarError::invalid_operation("Local autograd is disabled"));
        }

        self.stats.operations_count += 1;
        
        // Check if operation can be parallelized
        if should_parallelize(&output) {
            self.stats.parallel_operations += 1;
        }

        // TODO: Implement generic version
        // For now, return dummy node ID
        Ok(0)
    }

    /// Get statistics
    pub fn stats(&self) -> &LocalEngineStats {
        &self.stats
    }
}

/// Global coordinator for managing parallel autograd operations
pub struct GlobalCoordinator {
    /// Configuration
    config: RwLock<ParallelConfig>,
    /// Worker thread pool
    worker_pool: Arc<rayon::ThreadPool>,
    /// Task queue for complex operations
    task_queue: Mutex<VecDeque<ParallelTask>>,
    /// Condition variable for task availability
    task_available: Condvar,
    /// Operation cache for fusion
    operation_cache: RwLock<HashMap<String, CachedOperation>>,
    /// Global statistics
    global_stats: RwLock<GlobalStats>,
}

/// Task for parallel execution
#[derive(Debug)]
pub struct ParallelTask {
    pub id: usize,
    pub operation_type: String,
    pub priority: TaskPriority,
    pub estimated_cost: f64,
    pub dependencies: Vec<usize>,
}

/// Task priority levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum TaskPriority {
    Low = 0,
    Normal = 1,
    High = 2,
    Critical = 3,
}

/// Cached operation for fusion optimization
#[derive(Debug, Clone)]
pub struct CachedOperation {
    pub pattern: String,
    pub fused_impl: String, // Serialized fused operation
    pub hit_count: usize,
    pub last_used: Instant,
}

/// Global statistics across all threads
#[derive(Debug, Default)]
pub struct GlobalStats {
    pub total_operations: usize,
    pub parallel_operations: usize,
    pub active_threads: usize,
    pub cache_hit_rate: f64,
    pub average_parallelization: f64,
}

impl GlobalCoordinator {
    fn new(config: ParallelConfig) -> Self {
        let worker_pool = rayon::ThreadPoolBuilder::new()
            .num_threads(config.max_workers)
            .build()
            .expect("Failed to create thread pool");

        Self {
            config: RwLock::new(config),
            worker_pool: Arc::new(worker_pool),
            task_queue: Mutex::new(VecDeque::new()),
            task_available: Condvar::new(),
            operation_cache: RwLock::new(HashMap::new()),
            global_stats: RwLock::new(GlobalStats::default()),
        }
    }

    /// Submit task for parallel execution
    pub fn submit_task(&self, task: ParallelTask) -> Result<()> {
        let mut queue = self.task_queue.lock().unwrap();
        queue.push_back(task);
        self.task_available.notify_one();
        Ok(())
    }

    /// Execute operation in parallel if beneficial
    pub fn maybe_parallelize<T: TensorElement, F>(
        &self,
        operation: F,
        tensor_size: usize,
    ) -> Result<Tensor<T>>
    where
        F: FnOnce() -> Result<Tensor<T>> + Send,
    {
        let config = self.config.read().unwrap();
        
        if config.auto_parallel && tensor_size >= config.parallel_threshold {
            // Execute in parallel
            self.execute_parallel(operation)
        } else {
            // Execute sequentially
            operation()
        }
    }

    /// Execute operation in parallel using worker pool
    fn execute_parallel<T: TensorElement, F>(
        &self,
        operation: F,
    ) -> Result<Tensor<T>>
    where
        F: FnOnce() -> Result<Tensor<T>> + Send,
    {
        // Update statistics
        {
            let mut stats = self.global_stats.write().unwrap();
            stats.parallel_operations += 1;
        }

        // Execute using rayon thread pool
        let result = self.worker_pool.install(|| operation());
        result
    }

    /// Get global statistics
    pub fn stats(&self) -> GlobalStats {
        let stats = self.global_stats.read().unwrap();
        GlobalStats {
            total_operations: stats.total_operations,
            parallel_operations: stats.parallel_operations,
            active_threads: stats.active_threads,
            cache_hit_rate: stats.cache_hit_rate,
            average_parallelization: stats.average_parallelization,
        }
    }

    /// Update configuration
    pub fn update_config(&self, new_config: ParallelConfig) {
        let mut config = self.config.write().unwrap();
        *config = new_config;
    }

    /// Get current configuration
    pub fn get_config(&self) -> ParallelConfig {
        self.config.read().unwrap().clone()
    }

    /// Check if operation should be parallelized based on tensor size
    pub fn should_parallelize(&self, tensor_size: usize) -> bool {
        let config = self.config.read().unwrap();
        config.auto_parallel && tensor_size >= config.parallel_threshold
    }
}

/// Automatically initialize parallel autograd with optimal defaults
fn auto_init_parallel() -> Arc<GlobalCoordinator> {
    let config = ParallelConfig {
        max_workers: num_cpus::get(),
        auto_parallel: true,
        parallel_threshold: 1000, // Optimal threshold for most workloads
        use_thread_local: true,
        enable_fusion: true,
    };
    
    Arc::new(GlobalCoordinator::new(config))
}

/// Get global coordinator with automatic initialization
pub fn get_global_coordinator() -> Arc<GlobalCoordinator> {
    GLOBAL_COORDINATOR.get_or_init(auto_init_parallel).clone()
}

/// Initialize parallel autograd (optional - will auto-initialize if not called)
pub fn init_parallel_autograd(config: ParallelConfig) -> Result<()> {
    let coordinator = Arc::new(GlobalCoordinator::new(config));
    GLOBAL_COORDINATOR.set(coordinator)
        .map_err(|_| QuasarError::invalid_operation("Parallel autograd already initialized"))?;
    Ok(())
}

/// Execute with thread-local engine
pub fn with_local_engine<F, R>(f: F) -> R
where
    F: FnOnce(&mut LocalAutogradEngine) -> R,
{
    LOCAL_ENGINE.with(|engine_cell| {
        let mut engine_opt = engine_cell.borrow_mut();
        if engine_opt.is_none() {
            *engine_opt = Some(LocalAutogradEngine::new());
        }
        
        let engine = engine_opt.as_mut().unwrap();
        f(engine)
    })
}

/// Parallel matrix multiplication using rayon
pub fn parallel_matmul<T: TensorElement + Send + Sync>(
    a: &Tensor<T>, 
    b: &Tensor<T>
) -> Result<Tensor<T>> {
    let a_shape = a.shape();
    let b_shape = b.shape();
    
    if a_shape.ndim() != 2 || b_shape.ndim() != 2 {
        return Err(QuasarError::invalid_operation("Parallel matmul requires 2D tensors"));
    }
    
    let a_dims = a_shape.dims();
    let b_dims = b_shape.dims();
    
    if a_dims[1] != b_dims[0] {
        return Err(QuasarError::shape_mismatch(a_dims, b_dims));
    }
    
    let m = a_dims[0];
    let k = a_dims[1];
    let n = b_dims[1];
    
    // Use parallel computation for large matrices
    let coordinator = get_global_coordinator();
    let config = coordinator.config.read().unwrap();
    
    if config.auto_parallel && m * n >= config.parallel_threshold {
        // Parallel implementation using rayon
        let result_data: Vec<T> = (0..m).into_par_iter().flat_map(|i| {
            (0..n).into_par_iter().map(move |j| {
                let mut sum = T::zero();
                for l in 0..k {
                    let a_val = a.data()[i * k + l];
                    let b_val = b.data()[l * n + j];
                    sum = sum + a_val * b_val;
                }
                sum
            }).collect::<Vec<_>>()
        }).collect();
        
        let result_shape = crate::core::Shape::from(&[m, n]);
        Tensor::new(result_data, result_shape)
    } else {
        // Fall back to sequential implementation - use raw implementation to avoid recursion
        let mut result_data = vec![T::zero(); m * n];
        
        for i in 0..m {
            for j in 0..n {
                let mut sum = T::zero();
                for l in 0..k {
                    let a_val = a.data()[i * k + l];
                    let b_val = b.data()[l * n + j];
                    sum = sum + a_val * b_val;
                }
                result_data[i * n + j] = sum;
            }
        }
        
        let result_shape = crate::core::Shape::from(&[m, n]);
        Tensor::new(result_data, result_shape)
    }
}

/// Parallel element-wise operations
pub fn parallel_elementwise<T, F>(
    tensors: &[&Tensor<T>], 
    operation: F
) -> Result<Tensor<T>>
where
    T: TensorElement + Send + Sync,
    F: Fn(&[T]) -> T + Send + Sync,
{
    if tensors.is_empty() {
        return Err(QuasarError::invalid_operation("No tensors provided"));
    }
    
    let shape = tensors[0].shape().clone();
    let size = shape.total_elements();
    
    // Check all tensors have same shape
    for tensor in tensors.iter().skip(1) {
        if tensor.shape() != &shape {
            return Err(QuasarError::shape_mismatch(
                shape.dims(), tensor.shape().dims()
            ));
        }
    }
    
    let coordinator = get_global_coordinator();
    let config = coordinator.config.read().unwrap();
    
    if config.auto_parallel && size >= config.parallel_threshold {
        // Parallel element-wise operation
        let result_data: Vec<T> = (0..size).into_par_iter().map(|i| {
            let values: Vec<T> = tensors.iter().map(|t| t.data()[i]).collect();
            operation(&values)
        }).collect();
        
        Tensor::new(result_data, shape)
    } else {
        // Sequential operation
        let result_data: Vec<T> = (0..size).map(|i| {
            let values: Vec<T> = tensors.iter().map(|t| t.data()[i]).collect();
            operation(&values)
        }).collect();
        
        Tensor::new(result_data, shape)
    }
}

/// Check if operation should be parallelized
fn should_parallelize<T: TensorElement>(tensor: &Tensor<T>) -> bool {
    let coordinator = get_global_coordinator();
    let config = coordinator.config.read().unwrap();
    
    config.auto_parallel && tensor.numel() >= config.parallel_threshold
}

/// Parallel backward pass
pub fn parallel_backward<T: TensorElement + Send + Sync>(
    node_id: NodeId,
    grad_output: Tensor<T>,
) -> Result<()> {
    // TODO: Implement parallel backward pass
    // This would involve:
    // 1. Topological sorting of the graph
    // 2. Identifying independent branches that can be computed in parallel
    // 3. Using thread pool to compute gradients for independent nodes
    // 4. Synchronizing results
    
    with_local_engine(|engine| {
        // For now, delegate to local engine
        // In full implementation, this would coordinate parallel execution
        Ok(())
    })
}

/// Operation fusion for optimization
pub struct OperationFusion {
    /// Patterns to detect fusable operations
    patterns: HashMap<String, FusionRule>,
}

/// Rule for fusing operations
#[derive(Debug, Clone)]
pub struct FusionRule {
    pub pattern: String,
    pub replacement: String,
    pub benefit_score: f64,
}

impl OperationFusion {
    pub fn new() -> Self {
        let mut patterns = HashMap::new();
        
        // Add common fusion patterns
        patterns.insert(
            "add_mul".to_string(),
            FusionRule {
                pattern: "add(x, y) -> mul(result, z)".to_string(),
                replacement: "fused_add_mul(x, y, z)".to_string(),
                benefit_score: 1.5,
            }
        );
        
        patterns.insert(
            "relu_matmul".to_string(),
            FusionRule {
                pattern: "matmul(x, y) -> relu(result)".to_string(),
                replacement: "fused_matmul_relu(x, y)".to_string(),
                benefit_score: 2.0,
            }
        );
        
        Self { patterns }
    }
    
    /// Attempt to fuse operations
    pub fn try_fuse(&self, operations: &[String]) -> Option<String> {
        // Simple pattern matching - in real implementation would be more sophisticated
        if operations.len() >= 2 {
            let pattern = format!("{} -> {}", operations[0], operations[1]);
            for (name, rule) in &self.patterns {
                if pattern.contains(&rule.pattern) {
                    return Some(rule.replacement.clone());
                }
            }
        }
        None
    }
}

/// Parallel tensor operations trait
pub trait ParallelTensorOps<T: TensorElement> {
    /// Parallel addition
    fn par_add(&self, other: &Tensor<T>) -> Result<Tensor<T>>;
    
    /// Parallel multiplication
    fn par_mul(&self, other: &Tensor<T>) -> Result<Tensor<T>>;
    
    /// Parallel matrix multiplication
    fn par_matmul(&self, other: &Tensor<T>) -> Result<Tensor<T>>;
}

impl<T: TensorElement + Send + Sync> ParallelTensorOps<T> for Tensor<T> {
    fn par_add(&self, other: &Tensor<T>) -> Result<Tensor<T>> {
        parallel_elementwise(&[self, other], |values| values[0] + values[1])
    }
    
    fn par_mul(&self, other: &Tensor<T>) -> Result<Tensor<T>> {
        parallel_elementwise(&[self, other], |values| values[0] * values[1])
    }
    
    fn par_matmul(&self, other: &Tensor<T>) -> Result<Tensor<T>> {
        parallel_matmul(self, other)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::Shape;

    #[test]
    fn test_parallel_config() {
        let config = ParallelConfig::default();
        assert!(config.max_workers > 0);
        assert!(config.auto_parallel);
    }

    #[test]
    fn test_local_engine() {
        with_local_engine(|engine| {
            assert!(engine.enabled);
            assert_eq!(engine.stats.operations_count, 0);
        });
    }

    #[test]
    fn test_parallel_elementwise() -> Result<()> {
        let a = Tensor::new(vec![1.0f32, 2.0, 3.0], Shape::from(&[3]))?;
        let b = Tensor::new(vec![4.0f32, 5.0, 6.0], Shape::from(&[3]))?;
        
        let result = parallel_elementwise(&[&a, &b], |values| values[0] + values[1])?;
        assert_eq!(result.data(), &[5.0, 7.0, 9.0]);
        
        Ok(())
    }
} 