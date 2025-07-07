use burn::prelude::*;
use burn::nn::{LayerNorm, LayerNormConfig};
use burn::tensor::{backend::Backend, Tensor, Distribution};

/// Configuration for the D-LinOSS Layer
#[derive(Config, Debug)]
pub struct DLinossLayerConfig {
    pub d_input: usize,           // Input dimension
    pub d_model: usize,           // Model dimension (must be even)
    pub d_output: usize,          // Output dimension
    pub delta_t: f64,             // Time step for discretization
    #[config(default = "0.02")]
    pub init_std: f64,            // Initialization std
    #[config(default = "true")]
    pub enable_layer_norm: bool,  // Layer normalization
    #[config(default = "true")]
    pub enable_damping: bool,     // Enable learnable damping
    #[config(default = "0.1")]
    pub init_damping: f64,        // Initial damping coefficient
    #[config(default = "4")]
    pub num_damping_scales: usize,// Number of damping timescales
}

impl DLinossLayerConfig {
    pub fn dlinoss_config(d_input: usize, d_model: usize, d_output: usize) -> Self {
        Self {
            d_input,
            d_model,
            d_output,
            delta_t: 0.1,
            init_std: 0.02,
            enable_layer_norm: true,
            enable_damping: true,
            init_damping: 0.1,
            num_damping_scales: 4,
        }
    }
    
    pub fn basic_linoss(d_input: usize, d_model: usize, d_output: usize) -> Self {
        Self {
            enable_damping: false,
            ..Self::dlinoss_config(d_input, d_model, d_output)
        }
    }
}

/// D-LinOSS Layer: Damped Linear Oscillatory State-Space Layer  
#[derive(Module, Debug)]
pub struct DLinossLayer<B: Backend> {
    /// A matrix parameters (oscillatory dynamics)
    a_matrix: Tensor<B, 2>,
    /// B matrix parameters (input projection)
    b_matrix: Tensor<B, 2>,
    /// C matrix parameters (output projection) 
    c_matrix: Tensor<B, 2>,
    /// D matrix parameters (direct feedthrough)
    d_matrix: Tensor<B, 2>,
    /// Optional layer normalization
    layer_norm: Option<LayerNorm<B>>,
    /// Configuration parameters
    enable_damping: bool,
    delta_t: f64,
    d_input: usize,
    d_model: usize,
    d_output: usize,
}

impl<B: Backend> DLinossLayer<B> {
    pub fn new(config: &DLinossLayerConfig, device: &B::Device) -> Self {
        let d_model = config.d_model;
        let d_input = config.d_input;
        let d_output = config.d_output;
        
        // Ensure d_model is even for oscillatory pairs
        assert!(d_model % 2 == 0, "d_model must be even for oscillatory pairs");
        
        // Initialize A matrix with oscillatory structure
        let a_matrix = Self::init_oscillatory_a_matrix(config, device);
        
        // Initialize B matrix (input projection)
        let b_matrix = Tensor::random(
            [d_model, d_input],
            Distribution::Normal(0.0, config.init_std),
            device,
        );
        
        // Initialize C matrix (output projection)
        let c_matrix = Tensor::random(
            [d_output, d_model],
            Distribution::Normal(0.0, config.init_std),
            device,
        );
        
        // Initialize D matrix (feedthrough)
        let d_matrix = Tensor::random(
            [d_output, d_input],
            Distribution::Normal(0.0, config.init_std * 0.1),
            device,
        );
        
        // Optional layer normalization
        let layer_norm = if config.enable_layer_norm {
            Some(LayerNormConfig::new(d_model).init(device))
        } else {
            None
        };
        
        Self {
            a_matrix,
            b_matrix,
            c_matrix,
            d_matrix,
            layer_norm,
            enable_damping: config.enable_damping,
            delta_t: config.delta_t,
            d_input,
            d_model,
            d_output,
        }
    }
    
    /// Initialize oscillatory A matrix with analytical damped harmonic oscillator discretization
    fn init_oscillatory_a_matrix(config: &DLinossLayerConfig, device: &B::Device) -> Tensor<B, 2> {
        let d_model = config.d_model;
        let num_oscillators = d_model / 2;
        
        // Create block diagonal matrix for oscillatory dynamics
        let mut a_data = vec![0.0; d_model * d_model];
        
        for i in 0..num_oscillators {
            let freq = 0.1 + (i as f64 / num_oscillators as f64) * 2.0;
            let dt = config.delta_t;
            let base_damping = if config.enable_damping { config.init_damping } else { 0.0 };
            
            // Damped harmonic oscillator discretization
            let omega = freq;
            let gamma = base_damping;
            
            // Analytical solution for damped harmonic oscillator
            let exp_gamma_dt = (-gamma * dt).exp();
            let omega_d = (omega * omega - gamma * gamma).sqrt().max(0.01);
            
            let cos_term = (omega_d * dt).cos();
            let sin_term = (omega_d * dt).sin();
            
            // State transition matrix for damped oscillator [x, ẋ]
            let a11 = exp_gamma_dt * (cos_term + gamma * sin_term / omega_d);
            let a12 = exp_gamma_dt * sin_term / omega_d;
            let a21 = -exp_gamma_dt * omega * omega * sin_term / omega_d;
            let a22 = exp_gamma_dt * (cos_term - gamma * sin_term / omega_d);
            
            // Fill the 2x2 block for this oscillator
            let row_offset = i * 2;
            let col_offset = i * 2;
            
            a_data[row_offset * d_model + col_offset] = a11;
            a_data[row_offset * d_model + col_offset + 1] = a12;
            a_data[(row_offset + 1) * d_model + col_offset] = a21;
            a_data[(row_offset + 1) * d_model + col_offset + 1] = a22;
        }
        
        Tensor::<B, 1>::from_floats(a_data.as_slice(), device).reshape([d_model, d_model])
    }
    
    /// Forward pass through D-LinOSS layer
    pub fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 3> {
        let [batch_size, seq_len, _] = input.dims();
        let d_model = self.d_model;
        
        // Sequential processing with proper matrix operations
        let mut hidden_state: Tensor<B, 2> = Tensor::zeros([batch_size, d_model], &input.device());
        let mut all_outputs = Vec::new();
        
        for t in 0..seq_len {
            // Get input at time t
            let input_t = input.clone().slice([0..batch_size, t..t+1, 0..self.d_input]).squeeze(1);
            
            // Project input: B^T * u_t
            let projected_input = input_t.clone().matmul(self.b_matrix.clone().transpose());
            
            // State transition: h_{t+1} = A^T * h_t + B^T * u_t
            hidden_state = hidden_state.matmul(self.a_matrix.clone().transpose()) + projected_input;
            
            // Apply layer normalization if enabled
            if let Some(ref ln) = self.layer_norm {
                hidden_state = ln.forward(hidden_state.clone());
            }
            
            // Output projection: y_t = C * h_t + D * u_t
            let output_projection = hidden_state.clone().matmul(self.c_matrix.clone().transpose());
            let direct_projection = input_t.clone().matmul(self.d_matrix.clone().transpose());
            let output_t = output_projection + direct_projection;
            
            all_outputs.push(output_t.unsqueeze_dim(1));
        }
        
        // Stack all outputs: Vec<Tensor<B, 2>> -> Tensor<B, 3>
        Tensor::cat(all_outputs, 1)
    }
}
