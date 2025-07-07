use burn::tensor::Tensor;
use dlinoss_rust::device::{init_device, Backend};

fn main() {
    println!("=== WGPU Device Information Test with Vulkan ===");
    
    // Use centralized device initialization
    let device = init_device();
    
    // Try creating a tensor to force device initialization
    let test_tensor = Tensor::<Backend, 1>::from_data([1.0, 2.0, 3.0], &device);
    println!("   Tensor created successfully on: {:?}", test_tensor.device());
    
    // Test with actual computation to verify GPU usage
    println!("\n2. Testing computation performance:");
    
    // Small computation (should be fast on any device)
    let start = std::time::Instant::now();
    let small_a = Tensor::<Backend, 2>::random([100, 100], burn::tensor::Distribution::Normal(0.0, 1.0), &device);
    let small_b = Tensor::<Backend, 2>::random([100, 100], burn::tensor::Distribution::Normal(0.0, 1.0), &device);
    let _small_result = small_a.matmul(small_b);
    let small_time = start.elapsed();
    println!("   Small matrix mult (100x100): {:?}", small_time);
    
    // Large computation (should show difference between CPU/GPU)
    let start = std::time::Instant::now();
    let large_a = Tensor::<Backend, 2>::random([1000, 1000], burn::tensor::Distribution::Normal(0.0, 1.0), &device);
    let large_b = Tensor::<Backend, 2>::random([1000, 1000], burn::tensor::Distribution::Normal(0.0, 1.0), &device);
    let _large_result = large_a.matmul(large_b);
    let large_time = start.elapsed();
    println!("   Large matrix mult (1000x1000): {:?}", large_time);
    
    // Very large computation to stress test
    let start = std::time::Instant::now();
    let huge_a = Tensor::<Backend, 2>::random([2000, 2000], burn::tensor::Distribution::Normal(0.0, 1.0), &device);
    let huge_b = Tensor::<Backend, 2>::random([2000, 2000], burn::tensor::Distribution::Normal(0.0, 1.0), &device);
    let _huge_result = huge_a.matmul(huge_b);
    let huge_time = start.elapsed();
    println!("   Huge matrix mult (2000x2000): {:?}", huge_time);
    
    println!("\n3. Summary:");
    println!("   - If times are very fast (microseconds), likely using GPU");
    println!("   - If times scale roughly linearly with size, likely using CPU");
    println!("   - WGPU backend: Enabled");
    println!("   - Tensors successfully created and computed on: {:?}", device);
    
    // Additional test: memory-intensive operations
    println!("\n4. Memory test (GPU should handle this well):");
    let start = std::time::Instant::now();
    let mem_test = Tensor::<Backend, 2>::random([5000, 1000], burn::tensor::Distribution::Normal(0.0, 1.0), &device);
    let _mem_result = mem_test.sum();
    let mem_time = start.elapsed();
    println!("   Large tensor sum (5M elements): {:?}", mem_time);
}
