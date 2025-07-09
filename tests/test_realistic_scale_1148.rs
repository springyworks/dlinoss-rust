use burn::prelude::*;
use dlinoss_rust::parallel_scan_jul0915::*;
use dlinoss_rust::gpu_verification_1140::*;
use dlinoss_rust::device::init_device;
use std::sync::OnceLock;

/// COMPREHENSIVE REALISTIC SCALE TESTS
/// This tests with 64+ oscillators as requested, no more tiny test cases!
/// File marker: 1148 (11:48)

type TestBackend = dlinoss_rust::device::Backend;

// Single device initialization to avoid GPU conflicts
static DEVICE: OnceLock<<TestBackend as burn::tensor::backend::Backend>::Device> = OnceLock::new();

fn get_device() -> &'static <TestBackend as burn::tensor::backend::Backend>::Device {
    DEVICE.get_or_init(|| init_device())
}

/// Test realistic SSM sizes (64+ oscillators)
#[test]
fn test_realistic_ssm_sizes() {
    let device = get_device();
    
    // These are REALISTIC sizes, not toy examples
    let ssm_sizes = vec![64, 128, 256, 512];
    let batch_sizes = vec![8, 16, 32];
    let sequence_lengths = vec![128, 256, 512];
    
    println!("\n=== REALISTIC SCALE D-LinOSS TESTS ===");
    println!("Testing with 64+ oscillators as requested");
    
    for &ssm_size in &ssm_sizes {
        for &batch_size in &batch_sizes {
            for &seq_len in &sequence_lengths {
                let input_dim = ssm_size / 4; // Realistic input dimension
                
                println!("\n--- Testing SSM={}, Batch={}, Seq={}, Input={} ---", 
                    ssm_size, batch_size, seq_len, input_dim);
                
                // Create realistic D-LinOSS parameters with very conservative settings
                let a_diag: Tensor<TestBackend, 1> = dlinoss_rust::dlinoss_core::init_oscillatory_a_matrix::<TestBackend>(
                    ssm_size, 0.1, 1.0, device // Very conservative frequency range
                );
                
                let g_diag: Tensor<TestBackend, 1> = dlinoss_rust::dlinoss_core::init_damping_g_matrix::<TestBackend>(
                    ssm_size, 0.1, 0.2, device // Conservative damping for stability
                );
                
                let b_matrix: Tensor<TestBackend, 2> = Tensor::random([input_dim, ssm_size], 
                    burn::tensor::Distribution::Normal(0.0, (1.0 / (ssm_size as f32).sqrt()).into()), device);
                
                let input_sequence: Tensor<TestBackend, 3> = Tensor::random([batch_size, seq_len, input_dim], 
                    burn::tensor::Distribution::Normal(0.0, 1.0), device);
                
                // Test computation
                let start = std::time::Instant::now();
                let output = dlinoss_rust::dlinoss_core::apply_damped_linoss_imex(
                    a_diag, g_diag, b_matrix, input_sequence, 0.01, device  // Smaller time step for stability
                );
                let duration = start.elapsed();
                
                // Verify output
                assert_eq!(output.dims(), [batch_size, seq_len, ssm_size]);
                
                // Check numerical stability
                let max_value = output.abs().max();
                let is_finite = max_value.clone().into_scalar().is_finite();
                
                println!("  Computation time: {:?}", duration);
                let max_scalar = max_value.into_scalar();
                println!("  Max output value: {:.2e}", max_scalar);
                println!("  Is finite: {}", is_finite);
                
                assert!(is_finite, "Output should be finite for SSM size {}", ssm_size);
                assert!(max_scalar < 1e10, "Output should be bounded for SSM size {} (got {})", ssm_size, max_scalar);
                
                // For large systems, computation should be reasonable
                if ssm_size >= 128 {
                    assert!(duration.as_secs() < 30, "Large system computation should complete in reasonable time");
                }
                
                println!("  ✅ Test passed for SSM size {}", ssm_size);
            }
        }
    }
    
    println!("\n🎉 All realistic scale tests PASSED!");
}

/// Test parallel scan performance at scale
#[test]
fn test_parallel_scan_performance_at_scale() {
    let device = get_device();
    
    println!("\n=== PARALLEL SCAN PERFORMANCE AT SCALE ===");
    
    // Test with sequence lengths where parallel scan should show benefits
    let sequence_lengths = vec![64, 128, 256, 512, 1024];
    let ssm_size = 128; // Large enough to be realistic
    
    for &seq_len in &sequence_lengths {
        println!("\n--- Testing Parallel Scan with sequence length {} ---", seq_len);
        
        // Create test elements for parallel scan
        let mut elements = Vec::new();
        for i in 0..seq_len {
            let freq = (i as f32 / seq_len as f32) * 10.0; // Varying frequencies
            let a_matrix: Tensor<TestBackend, 1> = dlinoss_rust::dlinoss_core::init_oscillatory_a_matrix::<TestBackend>(
                ssm_size, 0.1, freq, device
            );
            
            let b_vector: Tensor<TestBackend, 1> = Tensor::random([2 * ssm_size], 
                burn::tensor::Distribution::Normal(0.0, 0.1), device);
            
            elements.push((a_matrix, b_vector));
        }
        
        // Test parallel scan
        let start = std::time::Instant::now();
        let parallel_results = parallel_scan(elements.clone(), dlinoss_rust::dlinoss_core::binary_operator);
        let parallel_time = start.elapsed();
        
        // Test sequential scan for comparison
        let start = std::time::Instant::now();
        let sequential_results = sequential_scan_reference(elements, dlinoss_rust::dlinoss_core::binary_operator);
        let sequential_time = start.elapsed();
        
        println!("  Parallel scan time: {:?}", parallel_time);
        println!("  Sequential scan time: {:?}", sequential_time);
        
        if sequential_time.as_nanos() > 0 {
            let speedup = sequential_time.as_nanos() as f64 / parallel_time.as_nanos() as f64;
            println!("  Speedup: {:.2}x", speedup);
        }
        
        // Verify results are identical
        assert_eq!(parallel_results.len(), sequential_results.len());
        
        for i in 0..std::cmp::min(10, parallel_results.len()) { // Check first 10 elements
            let diff_a = (parallel_results[i].0.clone() - sequential_results[i].0.clone()).abs().max();
            let diff_b = (parallel_results[i].1.clone() - sequential_results[i].1.clone()).abs().max();
            
            assert!(diff_a.into_scalar() < 1e-4, "Parallel/sequential mismatch in A at step {}", i);
            assert!(diff_b.into_scalar() < 1e-4, "Parallel/sequential mismatch in B at step {}", i);
        }
        
        println!("  ✅ Parallel scan correctness verified");
    }
    
    println!("\n🚀 Parallel scan performance tests completed!");
}

/// Test GPU verification with realistic workloads
#[test]
fn test_gpu_verification_realistic_workloads() {
    let device = get_device();
    
    println!("\n=== GPU VERIFICATION WITH REALISTIC WORKLOADS ===");
    
    let mut profiler: GpuProfiler<TestBackend> = GpuProfiler::new(device.clone());
    
    // Test GPU verification with realistic D-LinOSS computation sizes
    let test_sizes = vec![512, 1024, 2048];
    
    for &size in &test_sizes {
        println!("\n--- GPU Verification Test: Matrix size {} ---", size);
        
        let result = profiler.comprehensive_gpu_test(size);
        result.print_assessment();
        
        // These are the REAL requirements for GPU verification
        assert!(result.tensors_on_gpu, "Tensors must be on GPU for size {}", size);
        assert!(result.computation_time.as_millis() > 0, "Computation should take measurable time");
        assert!(result.memory_bandwidth_gbps > 0.1, "Should measure reasonable memory bandwidth (>0.1 GB/s)");
        
        println!("  ✅ GPU verification passed for size {}", size);
    }
    
    // Test D-LinOSS specific GPU benchmarks
    let ssm_sizes = vec![64, 128, 256];
    let batch_sizes = vec![16, 32];
    let sequence_lengths = vec![128, 256];
    
    let benchmarks = benchmark_dlinoss_gpu::<TestBackend>(device, ssm_sizes, batch_sizes, sequence_lengths);
    
    println!("\n--- D-LinOSS GPU Benchmark Results ---");
    for benchmark in &benchmarks {
        assert!(benchmark.computation_time.as_millis() > 0, "Should take measurable time");
        assert!(benchmark.memory_usage_mb > 0.0, "Should use GPU memory");
        assert!(benchmark.throughput_sequences_per_sec > 0.0, "Should have throughput");
        
        // For large systems, should have decent performance
        if benchmark.ssm_size >= 128 {
            assert!(benchmark.throughput_sequences_per_sec > 1.0, 
                "Large systems should have decent throughput");
        }
    }
    
    println!("🎯 GPU verification with realistic workloads completed!");
}

/// Comprehensive integration test
#[test]
fn test_comprehensive_integration() {
    let device = get_device();
    
    println!("\n=== COMPREHENSIVE INTEGRATION TEST ===");
    println!("Testing all components together with realistic parameters");
    
    // Use realistic D-LinOSS system parameters
    let ssm_size = 128;  // 128 oscillators
    let batch_size = 16;
    let seq_len = 256;
    let input_dim = 32;
    
    println!("System: {} oscillators, {} batch, {} sequence, {} input dim", 
        ssm_size, batch_size, seq_len, input_dim);
    
    // 1. Test core D-LinOSS computation
    println!("\n1. Testing core D-LinOSS computation...");
    let a_diag: Tensor<TestBackend, 1> = dlinoss_rust::dlinoss_core::init_oscillatory_a_matrix::<TestBackend>(
        ssm_size, 0.1, 20.0, device
    );
    let g_diag: Tensor<TestBackend, 1> = dlinoss_rust::dlinoss_core::init_damping_g_matrix::<TestBackend>(
        ssm_size, 0.01, 0.1, device
    );
    let b_matrix: Tensor<TestBackend, 2> = Tensor::random([input_dim, ssm_size], 
        burn::tensor::Distribution::Normal(0.0, 0.1), device);
    let input_sequence: Tensor<TestBackend, 3> = Tensor::random([batch_size, seq_len, input_dim], 
        burn::tensor::Distribution::Normal(0.0, 1.0), device);
    
    let start = std::time::Instant::now();
    let output = dlinoss_rust::dlinoss_core::apply_damped_linoss_imex(
        a_diag.clone(), g_diag.clone(), b_matrix.clone(), input_sequence.clone(), 0.01, device
    );
    let computation_time = start.elapsed();
    
    println!("   Core computation time: {:?}", computation_time);
    println!("   Output shape: {:?}", output.dims());
    println!("   ✅ Core computation successful");
    
    // 2. Test binary operator with realistic matrices
    println!("\n2. Testing binary operator at scale...");
    let a1: Tensor<TestBackend, 1> = Tensor::random([4 * ssm_size], 
        burn::tensor::Distribution::Normal(0.0, 0.1), device);
    let b1: Tensor<TestBackend, 1> = Tensor::random([2 * ssm_size], 
        burn::tensor::Distribution::Normal(0.0, 0.1), device);
    let a2: Tensor<TestBackend, 1> = Tensor::random([4 * ssm_size], 
        burn::tensor::Distribution::Normal(0.0, 0.1), device);
    let b2: Tensor<TestBackend, 1> = Tensor::random([2 * ssm_size], 
        burn::tensor::Distribution::Normal(0.0, 0.1), device);
    
    let binary_result = dlinoss_rust::dlinoss_core::binary_operator(
        (a1, b1), (a2, b2)
    );
    
    assert_eq!(binary_result.0.dims(), [4 * ssm_size]);
    assert_eq!(binary_result.1.dims(), [2 * ssm_size]);
    println!("   ✅ Binary operator successful at scale");
    
    // 3. Test GPU verification
    println!("\n3. Testing GPU verification...");
    let mut profiler: GpuProfiler<TestBackend> = GpuProfiler::new(device.clone());
    let gpu_result = profiler.comprehensive_gpu_test(1024);
    
    assert!(gpu_result.tensors_on_gpu, "Tensors should be on GPU");
    println!("   ✅ GPU verification successful");
    
    println!("\n🎉 COMPREHENSIVE INTEGRATION TEST PASSED!");
    println!("   All components working together with realistic 128-oscillator system");
}

/// Reference sequential scan for comparison
fn sequential_scan_reference<B: burn::tensor::backend::Backend>(
    elements: Vec<(Tensor<B, 1>, Tensor<B, 1>)>,
    binary_op: fn((Tensor<B, 1>, Tensor<B, 1>), (Tensor<B, 1>, Tensor<B, 1>)) -> (Tensor<B, 1>, Tensor<B, 1>)
) -> Vec<(Tensor<B, 1>, Tensor<B, 1>)> {
    if elements.is_empty() {
        return vec![];
    }
    
    let mut results = Vec::with_capacity(elements.len());
    results.push(elements[0].clone());
    
    for i in 1..elements.len() {
        let prev = results[i-1].clone();
        let curr = elements[i].clone();
        results.push(binary_op(prev, curr));
    }
    
    results
}
