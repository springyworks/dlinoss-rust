//! D-LinOSS: Damped Linear Oscillatory State-Space Layers and Blocks

// Custom Burn extensions - functionality missing from Burn
pub mod burnadd;

// Core mathematical D-LinOSS implementation following arXiv:2505.12171
pub mod dlinoss_core;

#[cfg(test)]
pub mod test_dlinoss_core;

// Layer and block implementations using the corrected core
pub mod dlinoss_layer;
pub mod dlinoss_block;

// Core modules
pub mod data;
pub mod device;
pub mod inference;
pub mod model;
pub mod training;

// Multi-architecture system
pub mod architectures;
pub mod benchmark;

// Re-exports for convenience (only working implementations)
pub use dlinoss_core::*;
pub use dlinoss_layer::{DLinossLayer, DLinossLayerConfig};
pub use dlinoss_block::DLinossBlock;
