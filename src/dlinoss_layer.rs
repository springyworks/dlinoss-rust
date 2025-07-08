use burn::prelude::*;
use burn::tensor::{backend::Backend, Tensor, Distribution};
use crate::dlinoss_core::{apply_damped_linoss_imex, init_oscillatory_a_matrix, init_damping_g_matrix};

/// Configuration for the D-LinOSS Layer following ArXiv:2505.12171
#[derive(Debug)]
pub struct DLinossLayerConfig {
    pub d_input: usize,           // Input dimension
    pub d_model: usize,           // Model dimension (must be even for oscillator pairs)
    pub d_output: usize,          // Output dimension
    pub delta_t: f64,             // Time step for IMEX discretization
    pub init_std: f64,            // Initialization std for B and C matrices
    pub enable_damping: bool,     // Enable learnable damping (D-LinOSS vs LinOSS)
    pub r_min: f64,               // Minimum oscillation frequency
    pub r_max: f64,               // Maximum oscillation frequency  
    pub damping_min: f64,         // Minimum damping coefficient
    pub damping_max: f64,         // Maximum damping coefficient
}

impl DLinossLayerConfig {
    pub fn new(d_input: usize, d_model: usize, d_output: usize) -> Self {
        Self {
            d_input,
            d_model,
            d_output,
            delta_t: 0.1,
            init_std: 0.02,
            enable_damping: true,
            r_min: 0.1,
            r_max: 2.0,
            damping_min: 0.01,
            damping_max: 0.5,
        }
    }
    
    /// Create basic LinOSS (without damping) configuration
    pub fn basic_linoss(d_input: usize, d_model: usize, d_output: usize) -> Self {
        Self {
            enable_damping: false,
            ..Self::new(d_input, d_model, d_output)
        }
    }
    
    /// Create D-LinOSS (with damping) configuration
    pub fn dlinoss_config(d_input: usize, d_model: usize, d_output: usize) -> Self {
        Self::new(d_input, d_model, d_output)
    }
}

/// D-LinOSS Layer: Mathematically correct implementation following ArXiv:2505.12171
/// Implements damped linear oscillatory state-space model with proper IMEX discretization
#[derive(Module, Debug)]
pub struct DLinossLayer<B: Backend> {
    /// A matrix parameters (oscillatory frequencies) - diagonal
    a_diag: Tensor<B, 1>,
    /// G matrix parameters (damping coefficients) - diagonal  
    g_diag: Option<Tensor<B, 1>>,
    /// B matrix parameters (input projection)
    b_matrix: Tensor<B, 2>,
    /// C matrix parameters (output projection)
    c_matrix: Tensor<B, 2>, 
    /// D matrix parameters (direct feedthrough)
    d_matrix: Tensor<B, 2>,
    /// Time step for IMEX discretization
    delta_t: f64,
    /// Number of oscillators (d_model / 2)
    num_oscillators: usize,
    /// Whether damping is enabled
    enable_damping: bool,
}

impl<B: Backend> DLinossLayer<B> {
    pub fn new(config: &DLinossLayerConfig, device: &B::Device) -> Self {
        let d_model = config.d_model;
        let d_input = config.d_input;
        let d_output = config.d_output;
        
        // Ensure d_model is even for oscillator pairs (position, velocity)
        assert!(d_model % 2 == 0, "d_model must be even for oscillator pairs (position, velocity)");
        let num_oscillators = d_model / 2;
        
        // Initialize A matrix (diagonal) - oscillatory frequencies
        let a_diag = init_oscillatory_a_matrix(
            num_oscillators, 
            config.r_min, 
            config.r_max, 
            device
        );
        
        // Initialize G matrix (diagonal) - damping coefficients
        let g_diag = if config.enable_damping {
            Some(init_damping_g_matrix(
                num_oscillators,
                config.damping_min,
                config.damping_max,
                device
            ))
        } else {
            None
        };
        
        // Initialize B matrix (input projection) - real-valued for simplicity
        // In full implementation, this should be complex-valued
        let b_matrix = Tensor::random(
            [num_oscillators, d_input],
            Distribution::Normal(0.0, config.init_std),
            device,
        );
        
        // Initialize C matrix (output projection) - real-valued for simplicity
        let c_matrix = Tensor::random(
            [d_output, num_oscillators],
            Distribution::Normal(0.0, config.init_std),
            device,
        );
        
        // Initialize D matrix (feedthrough) - typically small or zero
        let d_matrix = Tensor::random(
            [d_output, d_input],
            Distribution::Normal(0.0, config.init_std * 0.1),
            device,
        );
        
        Self {
            a_diag,
            g_diag,
            b_matrix,
            c_matrix,
            d_matrix,
            delta_t: config.delta_t,
            num_oscillators,
            enable_damping: config.enable_damping,
        }
    }
    
    /// Forward pass through D-LinOSS layer using correct mathematical formulation
    /// Input: [batch_size, seq_len, d_input]  
    /// Output: [batch_size, seq_len, d_output]
    pub fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 3> {
        let [batch_size, seq_len, d_input] = input.dims();
        
        // Apply D-LinOSS core operation using parallel scan (or sequential approximation)
        let ssm_output = if self.enable_damping {
            if let Some(ref g_diag) = self.g_diag {
                // Damped D-LinOSS with IMEX discretization
                apply_damped_linoss_imex(
                    self.a_diag.clone(),
                    g_diag.clone(),
                    self.b_matrix.clone(),
                    input.clone(),
                    self.delta_t,
                    &input.device()
                )
            } else {
                panic!("Damping enabled but G matrix not initialized");
            }
        } else {
            // Basic LinOSS without damping
            // For now, fall back to damped version with zero damping
            let zero_damping = Tensor::zeros_like(&self.a_diag);
            apply_damped_linoss_imex(
                self.a_diag.clone(),
                zero_damping,
                self.b_matrix.clone(),
                input.clone(),
                self.delta_t,
                &input.device()
            )
        };
        
        // Apply output projection using proper tensor operations for 3D tensors
        // ssm_output: [batch_size, seq_len, num_oscillators]
        // c_matrix: [d_output, num_oscillators] 
        // We need to apply the linear transformation to each timestep
        let mut output_projections = Vec::new();
        let mut direct_projections = Vec::new();
        
        for t in 0..seq_len {
            let ssm_t = ssm_output.clone().slice([0..batch_size, t..t+1, 0..self.num_oscillators]).squeeze_dims(&[1]);
            let input_t = input.clone().slice([0..batch_size, t..t+1, 0..d_input]).squeeze_dims(&[1]);
            
            let out_proj_t = ssm_t.matmul(self.c_matrix.clone().transpose());
            let direct_proj_t = input_t.matmul(self.d_matrix.clone().transpose());
            
            output_projections.push(out_proj_t.unsqueeze_dim(1));
            direct_projections.push(direct_proj_t.unsqueeze_dim(1));
        }
        
        let output_projection = Tensor::cat(output_projections, 1);
        let direct_projection = Tensor::cat(direct_projections, 1);
        
        output_projection + direct_projection
    }
}
