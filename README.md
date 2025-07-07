# D-LinOSS Rust Implementation

**Damped Linear Oscillatory State-Space Models** - Mathematical implementation following arXiv:2505.12171

## 🚀 Available Cargo Commands

### **Understanding `cargo run`**

`cargo run` is Rust's command to **compile and execute** your project in one step:

1. **Compilation**: Checks dependencies, compiles source code
2. **Execution**: Runs the resulting binary immediately
3. **Caching**: Reuses compiled artifacts when source unchanged

**Basic Syntax**: `cargo run [OPTIONS] [--] [ARGS]`

### **⚠️ Important: Multiple Binaries**

**ERROR**: `cargo run` could not determine which binary to run.

This project has **7 available binaries**:

- `dlinoss-rust` (main binary)
- `gpu_test`, `infer`, `print`, `test_1327`, `test_components_fixed`, `train`

**SOLUTION**: Always use `--bin <name>` to specify which binary to run:

```bash
# ❌ This will fail:
cargo run

# ✅ This works:
cargo run --bin test_1327
cargo run --bin gpu_test
cargo run --bin train --release
```

### **Cargo Run Variants**

```bash
# Development build (fast compile, slower runtime)
cargo run --bin <binary_name>

# Release build (slow compile, faster runtime)  
cargo run --bin <binary_name> --release

# With arguments passed to the binary
cargo run --bin <binary_name> -- arg1 arg2

# Show what cargo would do (dry run)
cargo run --bin <binary_name> --verbose
```

### **Core Testing & Validation**

```bash
# Complete mathematical validation of D-LinOSS implementation
cargo run --bin test_1327

# Test individual D-LinOSS components (layers & blocks) in isolation
cargo run --bin test_components_fixed

# GPU/WGPU device initialization and capability test
cargo run --bin gpu_test
```

### **Training & Inference**

```bash
# Train D-LinOSS model on dataset
cargo run --bin train

# Run inference with trained model
cargo run --bin infer

# Display model architecture and parameters
cargo run --bin print
```

### **Development & Release Builds**

```bash
# DEVELOPMENT BUILD (default)
# - Fast compilation, slower runtime
# - Debug symbols included, optimizations disabled
# - Best for development and testing
cargo run --bin <binary_name>

# RELEASE BUILD (optimized)  
# - Slower compilation, faster runtime
# - Full optimizations enabled, no debug symbols
# - Best for production and benchmarking
cargo run --bin <binary_name> --release

# CHECK COMPILATION (no execution)
# - Only compiles, doesn't run
# - Fast way to check for errors
cargo check

# BUILD WITHOUT RUNNING
# - Compiles and creates binary, doesn't execute
# - Useful for CI/CD pipelines
cargo build --release
```

### **Cargo Run Variants & Options**

```bash
# BINARY SELECTION
cargo run --bin test_1327              # Run specific binary
cargo run                              # Run default binary (src/main.rs)

# BUILD PROFILES  
cargo run --release                    # Optimized build
cargo run --debug                      # Debug build (default)

# WORKSPACE MANAGEMENT
cargo run -p dlinoss-rust --bin train  # Run binary from specific package

# FEATURE FLAGS
cargo run --features "gpu,parallel"    # Enable specific features
cargo run --no-default-features        # Disable default features

# ENVIRONMENT CONTROL
RUST_LOG=debug cargo run --bin test_1327     # Set logging level
RUST_BACKTRACE=1 cargo run --bin test_1327   # Enable stack traces

# OUTPUT CONTROL
cargo run --bin test_1327 --quiet      # Suppress cargo output
cargo run --bin test_1327 --verbose    # Verbose compilation info

# CROSS-COMPILATION
cargo run --target x86_64-pc-windows-gnu  # Cross-compile target

# PASSING ARGUMENTS TO BINARY
cargo run --bin train -- --epochs 100 --lr 0.001  # Args after '--'
```

### **Build vs Run Comparison**

| Command | Action | Use Case |
|---------|--------|----------|
| `cargo check` | Compile only, no binary | Fast error checking |
| `cargo build` | Compile + create binary | CI/CD, manual execution |
| `cargo run` | Compile + execute | Development, testing |
| `cargo test` | Compile + run tests | Unit/integration testing |

## 📊 Test Results Summary

- **✅ Mathematical Validation**: All paper stability conditions verified
- **✅ GPU Acceleration**: WGPU Vulkan backend working
- **✅ Parallel Scan**: O(log N) associative scan implementation
- **✅ Component Testing**: Layer and block isolation testing
- **✅ Performance**: Scaling analysis for different architectures

## 🧮 D-LinOSS Mathematical Implementation

This implementation provides:

1. **Mathematically Correct D-LinOSS**: Following arXiv:2505.12171 exactly
2. **Second-order ODE System**: `x''(t) = -A x(t) - G x'(t) + B u(t)`
3. **Trainable Damping Matrix**: Learnable `G` for energy dissipation
4. **Parallel Scan Algorithms**: O(log N) sequence processing
5. **Stability Verification**: Eigenvalue analysis and paper conditions
6. **Multi-Architecture Framework**: Ready for variant implementations

📖 **[Complete Mathematical Documentation](MATHEMATICAL_FOUNDATION.md)** - Comprehensive mathematical foundation with full derivations, proofs, and physical interpretations.

## 🔬 Testing Framework

### **Full Mathematical Validation** (`test_1327`)

- D-LinOSS layer functionality testing
- Paper stability condition verification  
- Eigenvalue analysis and spectral radius
- Exponential decay benchmark validation
- Full model integration testing

### **Component Isolation Testing** (`test_components_fixed`)

- Single layer testing with various input sizes
- Block composition testing (1-8 layers)
- Performance scaling analysis (16-128 oscillators)
- Mathematical stability verification
- Device validation and GPU usage confirmation

### **GPU Testing** (`gpu_test`)

- WGPU Vulkan backend initialization
- Device capability detection
- Tensor operation verification
- Memory and performance profiling

## 🏗️ Architecture Overview

```text
src/
├── dlinoss_1327.rs          # Core D-LinOSS layer (mathematically correct)
├── dlinoss_block_1327.rs    # Multi-layer D-LinOSS blocks
├── parallel_scan_1327.rs    # O(log N) parallel scan algorithms
├── model_1327.rs            # CNN + D-LinOSS integration
├── architectures/           # Multi-architecture trait system
├── benchmark.rs             # Performance comparison framework
└── bin/
    ├── test_1327.rs         # Mathematical validation suite
    ├── test_components_fixed.rs # Component isolation testing
    ├── gpu_test.rs          # GPU capability testing
    ├── train.rs             # Model training
    ├── infer.rs             # Model inference
    └── print.rs             # Model inspection
```

## ⚡ Quick Start

1. **Validate Implementation**:

   ```bash
   cargo run --bin test_1327
   ```

2. **Test Components**:

   ```bash
   cargo run --bin test_components_fixed
   ```

3. **Check GPU Setup**:

   ```bash
   cargo run --bin gpu_test
   ```

4. **Train Model**:

   ```bash
   cargo run --bin train --release
   ```

## 🔧 Development Notes

- **Zero Warnings**: All unused variables and imports eliminated
- **Real Implementations**: No fake/placeholder functions
- **Proper Tensor Operations**: Using Burn's TensorData API correctly
- **GPU Utilization**: WGPU Vulkan backend verified working
- **Mathematical Accuracy**: All formulations match arXiv:2505.12171

---

## Legacy Usage (Original)

- Train: `cargo run --bin train --release`
- Inference: `cargo run --bin infer --release`
- Print model: `cargo run --bin print --release`
