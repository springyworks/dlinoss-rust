use burn::prelude::*;
use burn::nn::{Gelu, Linear, LinearConfig, Dropout, DropoutConfig};
use crate::dlinoss_core::{apply_damped_linoss_imex, init_oscillatory_a_matrix, init_damping_g_matrix};

/// Configuration for DLinossLayer
#[derive(Config, Debug)]
pub struct DLinossLayerConfig {
    /// Number of oscillators (state space size)
    pub num_oscillators: usize,
    /// Input dimension
    pub input_dim: usize,
    /// Output dimension
    pub output_dim: usize,
    /// Use damping
    #[config(default = true)]
    pub use_damping: bool,
    /// Minimum damping coefficient
    #[config(default = 0.001)]
    pub damping_min: f32,  // Changed from f64 to f32
    /// Maximum damping coefficient
    #[config(default = 0.1)]
    pub damping_max: f32,  // Changed from f64 to f32
    /// Minimum oscillation period (for frequency initialization)
    #[config(default = 0.01)]
    pub r_min: f32,  // Changed from f64 to f32
    /// Maximum oscillation period
    #[config(default = 10.0)]
    pub r_max: f32,  // Changed from f64 to f32
    /// Time step
    #[config(default = 0.01)]
    pub delta_t: f32,  // Changed from f64 to f32
    /// Dropout rate
    #[config(default = 0.0)]
    pub dropout: f64,
}

impl DLinossLayerConfig {
    /// Create config for D-LinOSS layer (backwards compatibility)
    pub fn dlinoss_config(input_dim: usize, output_dim: usize, num_oscillators: usize) -> Self {
        Self::new(num_oscillators, input_dim, output_dim)
    }
}

/// DLinossLayer implementation
#[derive(Module, Debug)]
pub struct DLinossLayer<B: Backend> {
    /// Input projection (B matrix in state space formulation)
    pub input_proj: Linear<B>,
    /// Output projection (C matrix in state space formulation)
    pub output_proj: Linear<B>,
    /// Oscillatory matrix A (diagonal, complex eigenvalues)
    pub a_diag: Tensor<B, 1>,
    /// Damping matrix G (diagonal, negative real values) 
    pub g_diag: Option<Tensor<B, 1>>,
    /// Time step
    pub delta_t: f32,
    /// Activation function
    pub activation: Gelu,
    /// Dropout layer
    pub dropout: Dropout,
}

impl<B: Backend> DLinossLayer<B> {
    pub fn new(config: &DLinossLayerConfig, device: &B::Device) -> Self {
        let num_oscillators = config.num_oscillators;
        let input_dim = config.input_dim;
        let output_dim = config.output_dim;
        
        // Initialize input projection (B matrix)
        let input_proj = LinearConfig::new(input_dim, num_oscillators)
            .with_bias(false)
            .init(device);
        
        // Initialize output projection (C matrix)
        let output_proj = LinearConfig::new(num_oscillators, output_dim)
            .with_bias(true)
            .init(device);
        
        // Initialize oscillatory matrix (real part from discretization)
        let a_diag = init_oscillatory_a_matrix::<B>(
            num_oscillators, 
            config.r_min, 
            config.r_max, 
            device
        );
        
        // Initialize damping matrix if enabled
        let g_diag = if config.use_damping {
            Some(init_damping_g_matrix::<B>(
                num_oscillators,
                config.damping_min,
                config.damping_max,
                device
            ))
        } else {
            None
        };
        
        let activation = Gelu::new();
        let dropout = DropoutConfig::new(config.dropout).init();
        
        Self {
            input_proj,
            output_proj,
            a_diag,
            g_diag,
            delta_t: config.delta_t,
            activation,
            dropout,
        }
    }
    
    /// Forward pass for 2D input
    pub fn forward_2d(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        let device = input.device();
        let [_batch_size, _input_dim] = input.dims();
        
        // Expand to 3D for processing
        let input_3d = input.unsqueeze::<3>().swap_dims(1, 2); // [batch, 1, input_dim]
        
        // Get B matrix from input projection weights
        let b_matrix = self.input_proj.weight.val().transpose();
        
        // Apply damped LinOSS
        let state_sequence = if let Some(g_diag) = &self.g_diag {
            apply_damped_linoss_imex(
                self.a_diag.clone(),
                g_diag.clone(),
                b_matrix,
                input_3d,
                self.delta_t,
                &device
            )
        } else {
            // Without damping, use zeros for g_diag
            let g_diag = Tensor::zeros_like(&self.a_diag);
            apply_damped_linoss_imex(
                self.a_diag.clone(),
                g_diag,
                b_matrix,
                input_3d,
                self.delta_t,
                &device
            )
        };
        
        // Squeeze back to 2D
        let state_flat = state_sequence.squeeze(1); // [batch_size, ssm_size]
        
        // Apply output projection
        let output = self.output_proj.forward(state_flat);
        let output = self.activation.forward(output);
        self.dropout.forward(output)
    }
    
    /// Forward pass for 3D input
    pub fn forward_3d(&self, input: Tensor<B, 3>) -> Tensor<B, 3> {
        let device = input.device();
        let [batch_size, seq_len, _] = input.dims();
        
        // Get B matrix from input projection weights
        let b_matrix = self.input_proj.weight.val().transpose();
        
        // Apply damped LinOSS
        let state_sequence = if let Some(g_diag) = &self.g_diag {
            apply_damped_linoss_imex(
                self.a_diag.clone(),
                g_diag.clone(),
                b_matrix,
                input,
                self.delta_t,
                &device
            )
        } else {
            // Without damping, use zeros for g_diag
            let g_diag = Tensor::zeros_like(&self.a_diag);
            apply_damped_linoss_imex(
                self.a_diag.clone(),
                g_diag,
                b_matrix,
                input,
                self.delta_t,
                &device
            )
        };
        
        // Reshape for output projection
        let state_flat = state_sequence.reshape([batch_size * seq_len, self.a_diag.dims()[0]]);
        
        // Apply output projection
        let output_flat = self.output_proj.forward(state_flat);
        let output_flat = self.activation.forward(output_flat);
        let output_flat = self.dropout.forward(output_flat);
        
        // Reshape back to 3D - use the correct output dimension
        let output_dim = self.output_proj.weight.dims()[1];
        output_flat.reshape([batch_size, seq_len, output_dim])
    }
    
    /// Generic forward pass that handles both 2D and 3D inputs
    pub fn forward<const D: usize>(&self, _input: Tensor<B, D>) -> Tensor<B, D> {
        // We can't dynamically convert between tensor dimensions at runtime
        // This method needs to be called with specific dimensions
        panic!("Use forward_2d() or forward_3d() directly instead of generic forward()")
    }
}
