//! Profiling utilities for performance analysis

use std::time::{Duration, Instant};
use std::collections::HashMap;

/// Simple profiler for measuring operation performance
pub struct Profiler {
    timings: HashMap<String, Vec<Duration>>,
}

impl Profiler {
    /// Create new profiler
    pub fn new() -> Self {
        Self {
            timings: HashMap::new(),
        }
    }
    
    /// Start timing an operation
    pub fn start(&self, _name: &str) -> ProfilerGuard {
        ProfilerGuard {
            start_time: Instant::now(),
            name: _name.to_string(),
        }
    }
    
    /// Record timing
    pub fn record(&mut self, name: String, duration: Duration) {
        self.timings.entry(name).or_insert_with(Vec::new).push(duration);
    }
    
    /// Get statistics for an operation
    pub fn stats(&self, name: &str) -> Option<ProfileStats> {
        self.timings.get(name).map(|durations| {
            let total: Duration = durations.iter().sum();
            let count = durations.len();
            let avg = total / count as u32;
            let min = *durations.iter().min().unwrap();
            let max = *durations.iter().max().unwrap();
            
            ProfileStats {
                count,
                total,
                avg,
                min,
                max,
            }
        })
    }
    
    /// Print all statistics
    pub fn print_stats(&self) {
        for (name, _) in &self.timings {
            if let Some(stats) = self.stats(name) {
                println!("{}: {:?}", name, stats);
            }
        }
    }
}

impl Default for Profiler {
    fn default() -> Self {
        Self::new()
    }
}

/// RAII guard for automatic timing
pub struct ProfilerGuard {
    start_time: Instant,
    name: String,
}

impl Drop for ProfilerGuard {
    fn drop(&mut self) {
        let duration = self.start_time.elapsed();
        // TODO: Record to global profiler
        println!("{}: {:?}", self.name, duration);
    }
}

/// Statistics for profiled operations
#[derive(Debug, Clone)]
pub struct ProfileStats {
    pub count: usize,
    pub total: Duration,
    pub avg: Duration,
    pub min: Duration,
    pub max: Duration,
} 