use burn::prelude::*;
use dlinoss_rust::*;

fn main() {
    println!("🎯 D-LinOSS Numerical Stability Demo");
    println!("=====================================");
    
    let device = dlinoss_rust::device::init_device();
    
    // Test different SSM sizes
    let test_sizes = vec![32, 64, 128, 256];
    
    for ssm_size in test_sizes {
        println!("\n--- Testing SSM Size: {} ---", ssm_size);
        
        // Initialize stable parameters
        let step = 0.001;
        let (a_diag, g_diag) = dlinoss_rust::dlinoss_core::init_stable_parameters::<dlinoss_rust::device::Backend>(
            ssm_size, step, &device
        );
        
        // Verify stability
        let is_stable = dlinoss_rust::dlinoss_core::check_stability_condition(&a_diag, &g_diag, step);
        println!("Stability check: {}", if is_stable { "✅ STABLE" } else { "❌ UNSTABLE" });
        
        // Create test input
        let batch_size = 4;
        let seq_len = 32;
        let input_dim = 16;
        
        let b_matrix: Tensor<dlinoss_rust::device::Backend, 2> = Tensor::random([input_dim, ssm_size], 
            burn::tensor::Distribution::Normal(0.0, 0.1), &device);
        
        let input_sequence: Tensor<dlinoss_rust::device::Backend, 3> = Tensor::random([batch_size, seq_len, input_dim], 
            burn::tensor::Distribution::Normal(0.0, 1.0), &device);
        
        // Run D-LinOSS
        let start = std::time::Instant::now();
        let output = dlinoss_rust::dlinoss_core::apply_damped_linoss_imex(
            a_diag, g_diag, b_matrix, input_sequence, step, &device
        );
        let duration = start.elapsed();
        
        // Check output stability
        let max_value = output.abs().max().into_scalar();
        let is_finite = max_value.is_finite();
        
        println!("Computation time: {:?}", duration);
        println!("Max output value: {:.2e}", max_value);
        println!("Is finite: {}", is_finite);
        println!("Status: {}", if is_finite && max_value < 1e6 { "✅ STABLE" } else { "❌ UNSTABLE" });
        
        assert!(is_finite, "Output must be finite");
        assert!(max_value < 1e6, "Output must be bounded (got {:.2e})", max_value);
    }
    
    println!("\n🎉 All tests passed! D-LinOSS numerical stability fixed!");
}
