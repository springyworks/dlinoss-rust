/// Multi-Architecture System for D-LinOSS Variants
/// 
/// This module provides a flexible trait-based system for testing different
/// D-LinOSS architectures and comparing their performance.

use burn::tensor::{backend::Backend, Tensor};

/// Core trait that all D-LinOSS architectures must implement
pub trait DLinossArchitecture<B: Backend> {
    type Config: Clone;
    
    /// Create new architecture instance
    fn new(config: &Self::Config, device: &B::Device) -> Self;
    
    /// Forward pass
    fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 3>;
    
    /// Architecture name for logging/comparison
    fn name(&self) -> &'static str;
    
    /// Number of parameters
    fn param_count(&self) -> usize;
    
    /// Verify mathematical stability
    fn verify_stability(&self) -> bool;
    
    /// Get memory usage estimate
    fn memory_estimate(&self) -> usize {
        self.param_count() * std::mem::size_of::<f32>()
    }
}

/// Available architecture variants
pub enum ArchitectureType {
    /// Original D-LinOSS from paper (our current 1327 implementation)
    DLinoss1327,
    /// Simplified D-LinOSS for faster training
    DLinossLite,
    /// Multi-scale D-LinOSS with different time constants
    DLinossMultiScale,
    /// Hybrid CNN + D-LinOSS
    DLinossCNN,
}

impl ArchitectureType {
    /// Get all available architectures
    pub fn all() -> Vec<Self> {
        vec![
            Self::DLinoss1327,
            Self::DLinossLite,
            Self::DLinossMultiScale,
            Self::DLinossCNN,
        ]
    }
    
    /// Get architecture name
    pub fn name(&self) -> &'static str {
        match self {
            Self::DLinoss1327 => "D-LinOSS-1327",
            Self::DLinossLite => "D-LinOSS-Lite", 
            Self::DLinossMultiScale => "D-LinOSS-MultiScale",
            Self::DLinossCNN => "D-LinOSS-CNN",
        }
    }
}

// TODO: Implement additional architecture variants
// pub mod dlinoss_1327;
// pub mod dlinoss_lite;
// pub mod dlinoss_multiscale;
// pub mod dlinoss_cnn;
