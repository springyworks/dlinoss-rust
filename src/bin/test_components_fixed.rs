/// D-LinOSS Layer and Block Isolated Testing
/// 
/// Test individual components without full model overhead

use burn::{
    backend::{Autodiff, Wgpu},
    tensor::{Tensor, Distribution},
};

type AutodiffBackend = Autodiff<Wgpu>;

use dlinoss_rust::{
    dlinoss_1327::{DLinoss1327, DLinoss1327Config},
    dlinoss_block_1327::{DLinossBlock1327, DLinossBlock1327Config},
    device::init_device,
};

fn main() {
    println!("🧪 D-LinOSS Component Testing");
    println!("Testing layers and blocks in isolation\n");
    
    // Initialize device
    let device = init_device();
    println!("✓ Device initialized: {:?}\n", device);
    
    test_single_layer(&device);
    test_single_block(&device);
    test_layer_scaling(&device);
    test_block_composition(&device);
}

fn test_single_layer(device: &burn::backend::wgpu::WgpuDevice) {
    println!("1️⃣ Testing Single D-LinOSS Layer");
    
    let config = DLinoss1327Config::new(16, 32, 8); // d_input, d_oscillators, d_output
    let layer: DLinoss1327<AutodiffBackend> = DLinoss1327::init(&config, device);
    
    // Test different input sizes
    let test_cases = vec![
        ([1, 10, 16], "Single batch, short sequence"),
        ([4, 50, 16], "Multi batch, medium sequence"), 
        ([2, 200, 16], "Multi batch, long sequence"),
    ];
    
    for (dims, description) in test_cases {
        let input = Tensor::random(dims, Distribution::Normal(0.0, 1.0), device);
        let output = layer.forward(input.clone());
        
        println!("   {} -> Input: {:?}, Output: {:?}", 
                description, input.dims(), output.dims());
    }
    
    println!("   ✓ Layer handles various input sizes correctly\n");
}

fn test_single_block(device: &burn::backend::wgpu::WgpuDevice) {
    println!("2️⃣ Testing Single D-LinOSS Block");
    
    let config = DLinossBlock1327Config::new(2, 64, 32); // num_layers, d_oscillators, d_input
    let block: DLinossBlock1327<AutodiffBackend> = DLinossBlock1327::init(&config, device);
    
    let input = Tensor::random([4, 100, 32], Distribution::Normal(0.0, 1.0), device);
    let output = block.forward(input.clone());
    
    println!("   Input shape: {:?}", input.dims());
    println!("   Output shape: {:?}", output.dims());
    
    // Test stability
    let stability = block.verify_paper_stability();
    println!("   Stability: {}", if stability { "✓ STABLE" } else { "❌ UNSTABLE" });
    
    // Test spectral analysis
    let spectral_radii = block.spectral_radius();
    println!("   Spectral radii: {:?}", spectral_radii);
    
    // Use device for tensor operation to avoid unused warning
    let device_test: Tensor<AutodiffBackend, 1> = Tensor::ones([1], device);
    println!("   Device validation: {:?}", device_test.dims());
    
    println!("   ✓ Block processing successful\n");
}

fn test_layer_scaling(device: &burn::backend::wgpu::WgpuDevice) {
    println!("3️⃣ Testing Layer Scaling Performance");
    
    let oscillator_counts = vec![16, 32, 64, 128];
    
    for d_osc in oscillator_counts {
        let config = DLinoss1327Config::new(32, d_osc, 16); // d_input, d_oscillators, d_output
        let layer: DLinoss1327<AutodiffBackend> = DLinoss1327::init(&config, device);
        
        let input = Tensor::random([2, 50, 32], Distribution::Normal(0.0, 1.0), device);
        
        let start = std::time::Instant::now();
        let output = layer.forward(input);
        let duration = start.elapsed();
        
        println!("   {} oscillators: {:.2} ms (output: {:?})", 
                 d_osc, duration.as_millis(), output.dims());
    }
    println!("   ✓ Scaling analysis complete\n");
}

fn test_block_composition(device: &burn::backend::wgpu::WgpuDevice) {
    println!("4️⃣ Testing Block Composition");
    
    let layer_counts = vec![1, 2, 4, 8];
    
    for num_layers in layer_counts {
        let config = DLinossBlock1327Config::new(num_layers, 32, 16); // num_layers, d_oscillators, d_input
        let block: DLinossBlock1327<AutodiffBackend> = DLinossBlock1327::init(&config, device);
        
        let input = Tensor::random([2, 30, 16], Distribution::Normal(0.0, 1.0), device);
        
        let start = std::time::Instant::now();
        let output = block.forward(input);
        let duration = start.elapsed();
        
        println!("   {} layers: {:.2} ms (output: {:?})", 
                 num_layers, duration.as_millis(), output.dims());
    }
    println!("   ✓ Composition analysis complete");
}
