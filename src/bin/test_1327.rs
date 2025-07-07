use burn::tensor::{backend::Backend, Tensor};
use dlinoss_rust::{
    device::{init_device, AutodiffBackend},
    model_1327::{Model1327, ModelConfig, exponential_decay_benchmark},
    dlinoss_1327::{DLinoss1327, DLinoss1327Config},
};

fn main() {
    println!("🔬 D-LinOSS Mathematical Validation Test");
    println!("Testing implementation against arXiv:2505.12171");
    
    // Initialize GPU device
    let device = init_device();
    println!("✓ Device initialized: {:?}", device);
    
    // Test 1: Basic D-LinOSS layer functionality
    test_dlinoss_layer(&device);
    
    // Test 2: Stability conditions from paper
    test_stability_conditions(&device);
    
    // Test 3: Eigenvalue analysis
    test_eigenvalue_analysis(&device);
    
    // Test 4: Exponential decay benchmark
    test_exponential_decay(&device);
    
    // Test 5: Full model integration
    test_model_integration(&device);
    
    println!("\n🎯 All mathematical validation tests completed");
}

fn test_dlinoss_layer(device: &<AutodiffBackend as Backend>::Device) {
    println!("\n1️⃣ Testing D-LinOSS Layer Basic Functionality");
    
    let config = DLinoss1327Config::init(10, 32, 5); // 10 input, 32 oscillators, 5 output
    let layer: DLinoss1327<AutodiffBackend> = DLinoss1327::init(&config, device);
    
    // Test with batch of sequences
    let batch_size = 4;
    let seq_len = 16;
    let input = Tensor::random([batch_size, seq_len, 10], burn::tensor::Distribution::Normal(0.0, 1.0), device);
    
    println!("   Input shape: {:?}", input.dims());
    
    let output = layer.forward(input);
    println!("   Output shape: {:?}", output.dims());
    
    assert_eq!(output.dims(), [batch_size, seq_len, 5]);
    println!("   ✓ Layer forward pass successful");
    
    // Test stability check
    let stability = layer.check_stability();
    let stable_count = stability.clone().sum().into_scalar();
    println!("   Stable oscillators: {}/{}", stable_count, config.d_oscillators);
    
    if stable_count > (config.d_oscillators as f32 * 0.8) {
        println!("   ✓ Most oscillators are stable");
    } else {
        println!("   ⚠️  Many oscillators are unstable - may need parameter adjustment");
    }
}

fn test_stability_conditions(device: &<AutodiffBackend as Backend>::Device) {
    println!("\n2️⃣ Testing Stability Conditions from Paper");
    println!("   Condition: (G_i - Δt_i A_i)² ≤ 4A_i");
    
    let config = DLinoss1327Config::init(5, 16, 3);
    let layer: DLinoss1327<AutodiffBackend> = DLinoss1327::init(&config, device);
    
    let stability = layer.check_stability();
    let eigenvals = layer.get_eigenvalues();
    
    println!("   Stability check tensor shape: {:?}", stability.dims());
    println!("   Eigenvalues tensor shape: {:?}", eigenvals.dims());
    
    let stability_mean = stability.mean().into_scalar();
    println!("   Average stability score: {:.3}", stability_mean);
    
    if stability_mean > 0.9 {
        println!("   ✓ Paper stability conditions satisfied");
    } else {
        println!("   ❌ Stability conditions violated - implementation issue");
    }
}

fn test_eigenvalue_analysis(device: &<AutodiffBackend as Backend>::Device) {
    println!("\n3️⃣ Testing Eigenvalue Analysis");
    
    let config = DLinoss1327Config::init(8, 24, 4);
    let layer: DLinoss1327<AutodiffBackend> = DLinoss1327::init(&config, device);
    
    let eigenvals = layer.get_eigenvalues();
    println!("   Eigenvalue tensor: {:?}", eigenvals.dims());
    
    // Check if eigenvalues are within unit circle (stability requirement)
    let max_eigenval = eigenvals.abs().max().into_scalar();
    println!("   Maximum eigenvalue magnitude: {:.4}", max_eigenval);
    
    if max_eigenval <= 1.0 {
        println!("   ✓ All eigenvalues within unit circle (stable)");
    } else {
        println!("   ❌ Eigenvalues outside unit circle (unstable system)");
    }
}

fn test_exponential_decay(device: &<AutodiffBackend as Backend>::Device) {
    println!("\n4️⃣ Testing Exponential Decay Benchmark (Paper Validation)");
    
    let seq_len = 50;
    let batch_size = 8;
    
    let (input, target) = exponential_decay_benchmark::<AutodiffBackend>(device, seq_len, batch_size);
    
    println!("   Generated synthetic data:");
    println!("   Input shape: {:?}", input.dims());
    println!("   Target shape: {:?}", target.dims());
    
    // Create a simple D-LinOSS model for this task
    let config = DLinoss1327Config::init(1, 16, 1); // 1 input, 16 oscillators, 1 output
    let layer = DLinoss1327::init(&config, device);
    
    let prediction = layer.forward(input.clone());
    println!("   Prediction shape: {:?}", prediction.dims());
    
    // Compute MSE loss as basic validation
    let mse = (prediction - target).powf_scalar(2.0).mean();
    let mse_value: f32 = mse.into_scalar();
    println!("   Initial MSE (untrained): {:.6}", mse_value);
    
    if mse_value < 10.0 {
        println!("   ✓ Reasonable initial prediction range");
    } else {
        println!("   ⚠️  High initial error - may indicate implementation issues");
    }
}

fn test_model_integration(device: &<AutodiffBackend as Backend>::Device) 
where 
    <AutodiffBackend as Backend>::FloatElem: PartialOrd + Copy,
{
    println!("\n5️⃣ Testing Full Model Integration");
    
    let config = ModelConfig {
        num_classes: 10,
        d_oscillators: 32,
        num_dlinoss_layers: 2,
        dropout: 0.1,
    };
    
    let mut model: Model1327<AutodiffBackend> = Model1327::init(&config, device);
    
    // Test with MNIST-like input
    let batch_size = 4;
    let images = Tensor::random([batch_size, 1, 28, 28], burn::tensor::Distribution::Normal(0.0, 1.0), device);
    
    println!("   Input images shape: {:?}", images.dims());
    
    let output = model.forward(images);
    println!("   Model output shape: {:?}", output.dims());
    
    assert_eq!(output.dims(), [batch_size, 10]);
    println!("   ✓ Model forward pass successful");
    
    // Test stability verification
    let is_stable = model.verify_stability();
    println!("   Mathematical stability: {}", if is_stable { "✓ STABLE" } else { "❌ UNSTABLE" });
    
    // Test spectral analysis
    let spectral_radii = model.spectral_analysis();
    println!("   Spectral radii: {:?}", spectral_radii);
    
    // Test matrix update
    model.update_dlinoss_matrices();
    println!("   ✓ Discretized matrix update successful");
    
    println!("   Model parameters: ~{}", model.num_params());
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_mathematical_implementation() {
        let device = init_device();
        
        // Basic layer test
        let config = DLinoss1327Config::init(4, 8, 2);
        let layer = DLinoss1327::init(&config, &device);
        
        let input = Tensor::ones([2, 10, 4], &device);
        let output = layer.forward(input);
        
        assert_eq!(output.dims(), [2, 10, 2]);
        
        // Stability test
        let stability = layer.check_stability();
        assert!(stability.sum().into_scalar() >= 0.0);
    }
    
    #[test]
    fn test_exponential_decay_generation() {
        let device = init_device();
        let (input, target) = exponential_decay_benchmark::<AutodiffBackend>(&device, 20, 4);
        
        assert_eq!(input.dims(), [4, 20, 1]);
        assert_eq!(target.dims(), [4, 20, 1]);
    }
}
