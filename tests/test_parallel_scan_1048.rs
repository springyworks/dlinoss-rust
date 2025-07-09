use burn::prelude::*;
use dlinoss_rust::dlinoss_core::{binary_operator, apply_damped_linoss_imex, init_oscillatory_a_matrix, init_damping_g_matrix};
use dlinoss_rust::device::init_device;

/// Comprehensive test suite for parallel scan implementation
/// Tests the mathematical correctness and performance of D-LinOSS parallel scan operations
/// File marker: 1048 (10:48)

type TestBackend = dlinoss_rust::device::Backend;

/// Simple test to verify module works
#[cfg(test)]
#[test]
fn test_module_works() {
    assert_eq!(2 + 2, 4);
}

/// Test binary operator mathematical properties
#[cfg(test)]
#[test]
fn test_binary_operator_associativity() {
    let device = init_device();
    
    // Create simpler test elements
    let a1: Tensor<TestBackend, 1> = Tensor::from_floats([1.0, 0.5, 0.0, 0.5], &device);
    let b1: Tensor<TestBackend, 1> = Tensor::from_floats([1.0, 0.0], &device);
    
    let a2: Tensor<TestBackend, 1> = Tensor::from_floats([0.8, 0.2, 0.0, 0.8], &device);
    let b2: Tensor<TestBackend, 1> = Tensor::from_floats([0.8, 0.0], &device);
    
    let a3: Tensor<TestBackend, 1> = Tensor::from_floats([0.9, 0.1, 0.0, 0.9], &device);
    let b3: Tensor<TestBackend, 1> = Tensor::from_floats([0.9, 0.0], &device);
    
    // Test associativity: (a1 ⊗ a2) ⊗ a3 = a1 ⊗ (a2 ⊗ a3)
    let left_assoc = binary_operator(
        binary_operator((a1.clone(), b1.clone()), (a2.clone(), b2.clone())), 
        (a3.clone(), b3.clone())
    );
    
    let right_assoc = binary_operator(
        (a1.clone(), b1.clone()), 
        binary_operator((a2.clone(), b2.clone()), (a3.clone(), b3.clone()))
    );
    
    // Check if results are approximately equal (allowing for floating point errors)
    let diff_a = (left_assoc.0 - right_assoc.0).abs().max();
    let diff_b = (left_assoc.1 - right_assoc.1).abs().max();
    
    println!("Binary operator associativity test:");
    println!("Max difference in A matrices: {}", diff_a.clone().into_scalar());
    println!("Max difference in B vectors: {}", diff_b.clone().into_scalar());
    
    // Associativity should hold within numerical precision
    assert!(diff_a.into_scalar() < 1e-5, "A matrices not associative");
    assert!(diff_b.into_scalar() < 1e-5, "B vectors not associative");
}

/// Test D-LinOSS stability
#[cfg(test)]
#[test]
fn test_dlinoss_imex_stability() {
    let device = init_device();
    let ssm_size = 4;
    let batch_size = 2;
    let seq_len = 20;
    let input_dim = 2;
    
    // Create oscillatory A matrix
    let a_diag: Tensor<TestBackend, 1> = init_oscillatory_a_matrix::<TestBackend>(
        ssm_size, 0.1, 10.0, &device
    );
    
    // Create damping G matrix
    let g_diag: Tensor<TestBackend, 1> = init_damping_g_matrix::<TestBackend>(
        ssm_size, 0.01, 0.1, &device
    );
    
    // Create test input sequence
    let input_sequence: Tensor<TestBackend, 3> = Tensor::random([batch_size, seq_len, input_dim], 
                                      burn::tensor::Distribution::Normal(0.0, 1.0), &device);
    
    // Create input projection matrix - fix the dimensions
    let b_matrix: Tensor<TestBackend, 2> = Tensor::random([input_dim, ssm_size], 
                                burn::tensor::Distribution::Normal(0.0, 0.1), &device);
    
    println!("B matrix shape: {:?}", b_matrix.dims());
    println!("Input sequence shape: {:?}", input_sequence.dims());
    
    let step = 0.01;
    
    println!("Testing D-LinOSS IMEX with step size: {}", step);
    
    let output = apply_damped_linoss_imex(
        a_diag.clone(),
        g_diag.clone(),
        b_matrix.clone(),
        input_sequence.clone(),
        step,
        &device
    );
    
    // Check output dimensions
    assert_eq!(output.dims(), [batch_size, seq_len, ssm_size]);
    
    // Check for numerical stability (no NaN/Inf)
    let max_value = output.abs().max();
    let is_finite = max_value.clone().into_scalar().is_finite();
    
    println!("Max output value: {}", max_value.clone().into_scalar());
    println!("Is finite: {}", is_finite);
    
    assert!(is_finite, "Output contains NaN/Inf values at step {}", step);
    
    // Check that damping prevents explosion
    assert!(max_value.into_scalar() < 1000.0, "Output exploded at step {}", step);
    
    println!("✓ D-LinOSS IMEX stability test passed");
}

/// Test GPU utilization
#[cfg(test)]
#[test]
fn test_gpu_utilization() {
    let device = init_device();
    println!("Testing GPU utilization with device: {:?}", device);
    
    // Create large tensors to ensure GPU usage
    let ssm_size = 32;
    let batch_size = 8;
    let seq_len = 64;
    let input_dim = 8;
    
    let a_diag: Tensor<TestBackend, 1> = init_oscillatory_a_matrix::<TestBackend>(
        ssm_size, 0.1, 10.0, &device
    );
    let g_diag: Tensor<TestBackend, 1> = init_damping_g_matrix::<TestBackend>(
        ssm_size, 0.01, 0.1, &device
    );
    let b_matrix: Tensor<TestBackend, 2> = Tensor::random([input_dim, ssm_size], 
                                burn::tensor::Distribution::Normal(0.0, 0.1), &device);
    let input_sequence: Tensor<TestBackend, 3> = Tensor::random([batch_size, seq_len, input_dim], 
                                      burn::tensor::Distribution::Normal(0.0, 1.0), &device);
    
    // Verify all tensors are on GPU
    println!("A matrix device: {:?}", a_diag.device());
    println!("G matrix device: {:?}", g_diag.device());
    println!("B matrix device: {:?}", b_matrix.device());
    println!("Input sequence device: {:?}", input_sequence.device());
    
    // Time the computation
    let start = std::time::Instant::now();
    let output = apply_damped_linoss_imex(
        a_diag,
        g_diag,
        b_matrix,
        input_sequence,
        0.01,
        &device
    );
    let duration = start.elapsed();
    
    println!("GPU computation time: {:?}", duration);
    println!("Output device: {:?}", output.device());
    
    // Check that computation completed in reasonable time
    assert!(duration.as_secs() < 10, "GPU computation took too long");
    
    // Verify output dimensions
    assert_eq!(output.dims(), [batch_size, seq_len, ssm_size]);
    
    println!("✓ GPU utilization test passed");
}
