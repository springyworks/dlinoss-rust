//! D-LinOSS: Damped Linear Oscillatory State-Space Layers and Blocks

// Mathematical D-LinOSS implementation following arXiv:2505.12171
pub mod dlinoss_1327;
pub mod dlinoss_block_1327;
pub mod parallel_scan_1327;
pub mod model_1327;

// Legacy implementations (working baseline)
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

// Re-exports for convenience
pub use dlinoss_1327::{DLinoss1327, DLinoss1327Config};
pub use dlinoss_block_1327::{DLinossBlock1327, DLinossBlock1327Config};
pub use model_1327::{Model1327, ModelConfig};
pub use dlinoss_layer::{DLinossLayer, DLinossLayerConfig};
pub use dlinoss_block::DLinossBlock;
