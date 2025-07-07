use burn::tensor::Tensor;
use dlinoss_rust::{
    DLinossLayer, DLinossLayerConfig, 
    model::ModelConfig,
    device::{init_device, Backend}
};

fn main() {
    // Initialize device with centralized Vulkan setup
    let device = init_device();
    println!("WGPU device selected: {:?}", device);
    
    // Check if we can create tensors on GPU
    let test_tensor = Tensor::<Backend, 2>::random([100, 100], burn::tensor::Distribution::Normal(0.0, 1.0), &device);
    println!("Test tensor created on device: {:?}", test_tensor.device());
    println!("Test tensor shape: {:?}", test_tensor.dims());
    
    // Perform GPU computation to verify it's working
    let gpu_result = test_tensor.clone().matmul(test_tensor.transpose());
    println!("GPU matrix multiplication completed. Result shape: {:?}", gpu_result.dims());
    
    // Test D-LinOSS layer creation with proper config
    let dlinoss_config = DLinossLayerConfig::dlinoss_config(32, 32, 32);
    
    let dlinoss_layer: DLinossLayer<Backend> = DLinossLayer::new(&dlinoss_config, &device);
    
    // Test forward pass with random input (batch_size, sequence_len, d_input)
    let input = Tensor::random([2, 64, 32], burn::tensor::Distribution::Uniform(-1.0, 1.0), &device);
    println!("D-LinOSS input tensor device: {:?}", input.device());
    let output = dlinoss_layer.forward(input);
    
    println!("D-LinOSS layer forward pass successful!");
    println!("Output shape: {:?}", output.dims());
    println!("D-LinOSS output tensor device: {:?}", output.device());
    
    // Test full model with MLP head 
    let model_config = ModelConfig {
        num_classes: 10,
        hidden_size: 512,
        dropout: 0.5,
        use_dlinoss: false,
    };
    
    let model = model_config.init::<Backend>(&device);
    
    // Test model forward pass
    let model_input = Tensor::random([2, 28, 28], burn::tensor::Distribution::Uniform(0.0, 1.0), &device);
    let model_output = model.forward(model_input);
    
    println!("Full model with MLP head successful!");
    println!("Model output shape: {:?}", model_output.dims());
    
    // Now test D-LinOSS model 
    let dlinoss_model_config = ModelConfig {
        num_classes: 10,
        hidden_size: 64,
        dropout: 0.5,
        use_dlinoss: true,
    };
    
    let dlinoss_model = dlinoss_model_config.init::<Backend>(&device);
    let dlinoss_input = Tensor::random([2, 28, 28], burn::tensor::Distribution::Uniform(0.0, 1.0), &device);
    let dlinoss_output = dlinoss_model.forward(dlinoss_input);
    
    println!("Full model with D-LinOSS head successful!");
    println!("D-LinOSS model output shape: {:?}", dlinoss_output.dims());
    println!("D-LinOSS output device: {:?}", dlinoss_output.device());
    
    // Additional GPU verification
    println!("\n=== GPU Verification ===");
    println!("Backend: Wgpu with Vulkan");
    println!("Device type used: {:?}", device);
    
    // Try to verify WGPU adapter info if possible
    println!("Large tensor operations test...");
    let large_a = Tensor::<Backend, 2>::random([512, 512], burn::tensor::Distribution::Normal(0.0, 1.0), &device);
    let large_b = Tensor::<Backend, 2>::random([512, 512], burn::tensor::Distribution::Normal(0.0, 1.0), &device);
    let large_result = large_a.matmul(large_b);
    println!("Large matrix multiplication (512x512) completed on device: {:?}", large_result.device());
}
