//! Centralized device initialization for D-LinOSS
//! 
//! This module provides a single place to configure GPU acceleration,
//! automatically setting up Vulkan backend for optimal performance.

use burn::backend::{wgpu::WgpuDevice, Wgpu, Autodiff};

/// Initialize WGPU device with Vulkan backend for optimal GPU performance
/// 
/// This function automatically configures Vulkan as the graphics API,
/// eliminating the need to set WGPU_BACKEND environment variables.
/// 
/// # Returns
/// - `WgpuDevice` configured with Vulkan backend
/// 
/// # Example
/// ```rust
/// use dlinoss_rust::device::init_device;
/// 
/// let device = init_device();
/// // Device is now ready for GPU acceleration
/// ```
pub fn init_device() -> WgpuDevice {
    println!("🚀 Initializing WGPU device with Vulkan backend...");
    
    let device = WgpuDevice::default();
    
    // Set Vulkan as the graphics API for optimal GPU performance
    burn::backend::wgpu::init_setup::<burn::backend::wgpu::graphics::Vulkan>(
        &device,
        Default::default(),
    );
    
    println!("✓ Vulkan backend initialized successfully");
    println!("  Device: {:?}", device);
    
    device
}

/// Type alias for the Wgpu backend used throughout the project
pub type Backend = Wgpu<f32, i32>;

/// Type alias for autodiff backend used in training
pub type AutodiffBackend = Autodiff<Backend>;

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::Tensor;
    
    #[test]
    fn test_device_initialization() {
        let device = init_device();
        
        // Test that we can create tensors on the device
        let test_tensor = Tensor::<Backend, 1>::from_data([1.0, 2.0, 3.0], &device);
        assert_eq!(test_tensor.dims(), [3]);
        
        // Test basic computation
        let result = test_tensor.clone() + test_tensor;
        assert_eq!(result.dims(), [3]);
    }
}
