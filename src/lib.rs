//! D-LinOSS: Damped Linear Oscillatory State-Space Layers and Blocks
pub use dlinoss_layer::{DLinossLayer, DLinossLayerConfig};
pub use dlinoss_block::DLinossBlock;
pub mod dlinoss_layer;
pub mod dlinoss_block;
// Note: If you are following the Burn Book guide this file can be ignored.
// This lib.rs file is added only for convenience so that the code in this
// guide can be reused.
pub mod data;
pub mod device;
pub mod inference;
pub mod model;
pub mod training;
