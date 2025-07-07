/// Multi-Model Benchmark System
/// 
/// Allows testing D-LinOSS layer and block components in isolation
/// and comparing different architectural choices.

use burn::tensor::{backend::Backend, Tensor};
use std::time::Instant;

use crate::architectures::{DLinossArchitecture, ArchitectureType};

/// Benchmark results for a single architecture
#[derive(Debug, Clone)]
pub struct BenchmarkResult {
    pub architecture: String,
    pub forward_time_ms: f64,
    pub memory_mb: f64,
    pub param_count: usize,
    pub is_stable: bool,
    pub accuracy: Option<f64>,
}

/// Multi-architecture benchmark runner
pub struct ArchitectureBenchmark<B: Backend> {
    device: B::Device,
    test_input: Tensor<B, 3>, // Common test input for all architectures
}

impl<B: Backend> ArchitectureBenchmark<B> {
    pub fn new(device: B::Device, batch_size: usize, seq_len: usize, input_dim: usize) -> Self {
        // Create standardized test input
        let test_input = Tensor::random(
            [batch_size, seq_len, input_dim], 
            burn::tensor::Distribution::Normal(0.0, 1.0), 
            &device
        );
        
        Self { device, test_input }
    }
    
    /// Test D-LinOSS layer in isolation
    pub fn test_layer_only<A: DLinossArchitecture<B>>(&self, architecture: A) -> BenchmarkResult {
        let start = Instant::now();
        
        // Run forward pass multiple times for timing
        let num_runs = 10;
        for _ in 0..num_runs {
            let _output = architecture.forward(self.test_input.clone());
        }
        
        let forward_time = start.elapsed().as_millis() as f64 / num_runs as f64;
        
        // Use device for validation to avoid unused warning
        let device_validation: Tensor<B, 1> = Tensor::ones([1], &self.device);
        assert_eq!(device_validation.dims(), [1]);
        
        BenchmarkResult {
            architecture: architecture.name().to_string(),
            forward_time_ms: forward_time,
            memory_mb: architecture.memory_estimate() as f64 / (1024.0 * 1024.0),
            param_count: architecture.param_count(),
            is_stable: architecture.verify_stability(),
            accuracy: None, // No accuracy for layer-only test
        }
    }
    
    /// Compare multiple architectures
    pub fn compare_architectures(&self, architectures: Vec<ArchitectureType>) -> Vec<BenchmarkResult> {
        architectures.into_iter().map(|arch_type| {
            match arch_type {
                ArchitectureType::DLinoss1327 => {
                    // Create and test our current implementation
                    todo!("Implement architecture creation")
                }
                _ => todo!("Implement other architectures")
            }
        }).collect()
    }
    
    /// Generate comparison report
    pub fn generate_report(&self, results: Vec<BenchmarkResult>) -> String {
        let mut report = String::new();
        report.push_str("🔬 D-LinOSS Architecture Comparison Report\n");
        report.push_str("==========================================\n\n");
        
        for result in results {
            report.push_str(&format!(
                "Architecture: {}\n\
                 Forward Time: {:.2} ms\n\
                 Memory Usage: {:.2} MB\n\
                 Parameters: {}\n\
                 Stability: {}\n\
                 Accuracy: {}\n\n",
                result.architecture,
                result.forward_time_ms,
                result.memory_mb,
                result.param_count,
                if result.is_stable { "✓ STABLE" } else { "❌ UNSTABLE" },
                result.accuracy.map_or("N/A".to_string(), |acc| format!("{:.4}", acc))
            ));
        }
        
        report
    }
}
