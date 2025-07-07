use burn::prelude::*;
use burn::nn::{Linear, LinearConfig, LayerNorm, LayerNormConfig};
use burn::tensor::{backend::Backend, Tensor, Distribution};

/// Configuration for the D-LinOSS Layer
#[derive(Config, Debug)]
pub struct DLinossLayerConfig {
    pub d_input: usize,           // Input dimension
    pub d_model: usize,           // Model dimension (must be even)
    pub d_output: usize,          // Output dimension
    pub delta_t: f64,             // Time step for discretization
    #[config(default = "0.05")]
    pub init_std: f64,            // Initialization std
    #[config(default = "true")]
    pub enable_layer_norm: bool,  // Layer normalization
    #[config(default = "true")]
    pub enable_damping: bool,     // Enable learnable damping
    #[config(default = "0.1")]
    pub init_damping: f64,        // Initial damping coefficient
    #[config(default = "1")]
    pub num_damping_scales: usize,// Number of damping timescales
}

impl DLinossLayerConfig {
    pub fn new_dlinoss(d_input: usize, d_model: usize, d_output: usize) -> Self {
        Self {
            d_input,
            d_model,
            d_output,
            delta_t: 1.0,
            init_std: 0.05,
            enable_layer_norm: true,
            enable_damping: true,
            init_damping: 0.1,
            num_damping_scales: 1,
        }
    }
    pub fn vanilla_linoss(d_input: usize, d_model: usize, d_output: usize) -> Self {
        Self {
            enable_damping: false,
            ..Self::new_dlinoss(d_input, d_model, d_output)
        }
    }
}

/// D-LinOSS Layer: Damped Linear Oscillatory State-Space Layer
#[derive(Module, Debug)]
pub struct DLinossLayer<B: Backend> {
    #[module]
    pub input_proj: Linear<B>,
    #[module]
    pub output_proj: Linear<B>,
    #[module]
    pub layer_norm: Option<LayerNorm<B>>,
    pub freq: Tensor<B, 1>,
    pub damp: Tensor<B, 1>,
    pub d_input: usize,
    pub d_model: usize,
    pub d_output: usize,
    pub delta_t: f64,
}

impl<B: Backend> DLinossLayer<B> {
    pub fn new(config: &DLinossLayerConfig, device: &B::Device) -> Self {
        let freq = Tensor::random([config.d_model], Distribution::Normal(0.0, config.init_std), device);
        let damp = Tensor::full([config.d_model], config.init_damping, device);
        let input_proj = LinearConfig::new(config.d_input, config.d_model).init(device);
        let output_proj = LinearConfig::new(config.d_model, config.d_output).init(device);
        let layer_norm = if config.enable_layer_norm {
            Some(LayerNormConfig::new(config.d_model).init(device))
        } else {
            None
        };
        Self {
            freq,
            damp,
            input_proj,
            output_proj,
            layer_norm,
            d_input: config.d_input,
            d_model: config.d_model,
            d_output: config.d_output,
            delta_t: config.delta_t,
        }
    }

    /// Forward pass for a sequence [batch, seq_len, d_input]
    pub fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 3> {
        let [batch, seq_len, _] = input.dims();
        let d_model = self.d_model;
        let x_proj = self.input_proj.forward(input); // [batch, seq_len, d_model]

        // D-LinOSS recurrence: analytical update for each time step
        // State: [batch, d_model*2] (position, velocity)
        let mut state: Tensor<B, 2> = Tensor::zeros([batch, 2 * d_model], &x_proj.device());
        let freq = self.freq.clone(); // [d_model]
        let damp = self.damp.clone(); // [d_model]
        let dt = self.delta_t;

        for t in 0..seq_len {
            let input_t = x_proj.clone().slice([0..batch, t..t + 1, 0..d_model]).squeeze(1); // [batch, d_model]
            let x = state.clone().slice([0..batch, 0..d_model]);
            let v = state.clone().slice([0..batch, d_model..2 * d_model]);

            let exp_gamma = (damp.clone() * (-dt)).exp();
            let omega_d = (freq.clone() * freq.clone() - damp.clone() * damp.clone()).abs().sqrt();
            let cos_term = (omega_d.clone() * dt).cos();
            let sin_term = (omega_d.clone() * dt).sin();

            let a11 = exp_gamma.clone() * (cos_term.clone() + damp.clone() * sin_term.clone() / omega_d.clone());
            let a12 = exp_gamma.clone() * sin_term.clone() / omega_d.clone();
            let a21 = exp_gamma.clone() * freq.clone() * freq.clone() * sin_term.clone() / omega_d.clone() * (-1.0);
            let a22 = exp_gamma.clone() * (cos_term.clone() - damp.clone() * sin_term.clone() / omega_d.clone());

            let x_new = a11 * x.clone() + a12 * v.clone() + input_t;
            let v_new = a21 * x + a22 * v;
            state = Tensor::cat(vec![x_new, v_new], 1);
        }
        let out = self.output_proj.forward(state.slice([0..batch, 0..d_model]));
        let out = out.reshape([batch, 1, self.d_output]);
        if let Some(norm) = &self.layer_norm {
            norm.forward(out)
        } else {
            out
        }
    }
}
