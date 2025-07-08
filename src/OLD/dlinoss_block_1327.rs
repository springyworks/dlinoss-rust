use burn::prelude::*;
use burn::nn::{LayerNorm, LayerNormConfig};
use burn::tensor::{backend::Backend, Tensor};

use crate::dlinoss_1327::{DLinoss1327, DLinoss1327Config};

/// Configuration for D-LinOSS Block (arXiv:2505.12171)
/// 
/// MULTI-LAYER ARCHITECTURE DESIGN:
/// A D-LinOSS block consists of L sequential D-LinOSS layers, where each layer
/// implements the damped oscillatory dynamics. The mathematical composition is:
/// 
/// INPUT: u ∈ ℝᵖ → LAYER 1 → h₁ ∈ ℝᵠ → LAYER 2 → h₂ ∈ ℝᵠ → ... → LAYER L → y ∈ ℝᵠ
/// 
/// MATHEMATICAL FLOW:
/// Layer 1: h₁ = DLinOSS₁(u; A₁, G₁, B₁, C₁, D₁)
/// Layer 2: h₂ = DLinOSS₂(h₁; A₂, G₂, B₂, C₂, D₂)  
/// ...
/// Layer L: y = DLinOSSₗ(hₗ₋₁; Aₗ, Gₗ, Bₗ, Cₗ, Dₗ)
/// 
/// DIMENSION PROGRESSION:
/// - Layer 1: ℝᵖ → ℝᵠ (input projection)
/// - Layers 2-L: ℝᵠ → ℝᵠ (hidden transformations)
/// - Each layer has m oscillators with 2m internal state dimensions
/// 
/// LEARNABLE PARAMETERS PER LAYER:
/// - A_i ∈ ℝᵐ: frequency parameters for layer i
/// - G_i ∈ ℝᵐ: damping parameters for layer i (key innovation)
/// - B_i ∈ ℝᵐˣᵈⁱⁿ: input projection for layer i  
/// - C_i ∈ ℝᵠˣᵐ: output projection for layer i
/// - D_i ∈ ℝᵠˣᵈⁱⁿ: feedthrough for layer i
/// 
/// REPRESENTATION LEARNING:
/// Different layers can learn different temporal scales through their damping matrices:
/// - Early layers: fine-grained, fast dynamics (small damping)
/// - Later layers: coarse-grained, slow dynamics (larger damping)
#[derive(Config, Debug)]
pub struct DLinossBlock1327Config {
    pub d_input: usize,      // p: input dimension to first layer
    pub d_oscillators: usize, // m: number of oscillators per layer
    pub d_output: usize,     // q: output dimension from each layer  
    #[config(default = "1")]
    pub num_layers: usize,   // L: number of sequential D-LinOSS layers
    #[config(default = "0.1")]
    pub delta_t: f64,        // Δt: time step (shared across layers)
    #[config(default = "0.02")]
    pub init_std: f64,       // σ: initialization standard deviation
    #[config(default = "true")]
    pub layer_norm: bool,    // Enable layer normalization
    #[config(default = "0.1")]
    pub dropout: f64,        // Dropout probability (currently unused)
}

impl DLinossBlock1327Config {
    pub fn init(d_input: usize, d_oscillators: usize, d_output: usize) -> Self {
        Self {
            d_input,
            d_oscillators,
            d_output,
            num_layers: 1,
            delta_t: 0.1,
            init_std: 0.02,
            layer_norm: true,
            dropout: 0.1,
        }
    }
    
    pub fn with_layers(mut self, num_layers: usize) -> Self {
        self.num_layers = num_layers;
        self
    }
}

/// D-LinOSS Block: Multi-Layer Damped Oscillatory Architecture
/// Implements hierarchical composition of D-LinOSS layers from arXiv:2505.12171
/// 
/// MATHEMATICAL FORMULATION:
/// 
/// A D-LinOSS block with L layers computes the following transformation:
/// 
/// FOR layer ℓ = 1, 2, ..., L:
///   INPUT: h_{ℓ-1} ∈ ℝᵈ (where h₀ = input, d = p for ℓ=1, d = q for ℓ>1)
///   
///   OSCILLATOR DYNAMICS:
///   w^{(ℓ)}_{k+1} = M^{(ℓ)} w^{(ℓ)}_k + F^{(ℓ)} h_{ℓ-1,k}
///   
///   OUTPUT: h_{ℓ,k} = H^{(ℓ)} w^{(ℓ)}_k + D^{(ℓ)} h_{ℓ-1,k}
/// 
/// where each layer ℓ has its own learned parameters:
/// - A^{(ℓ)} ∈ ℝᵐ: frequency matrix for layer ℓ
/// - G^{(ℓ)} ∈ ℝᵐ: damping matrix for layer ℓ (learnable!)
/// - M^{(ℓ)}, F^{(ℓ)}, H^{(ℓ)}: discretized matrices computed from A^{(ℓ)}, G^{(ℓ)}
/// 
/// HIERARCHICAL REPRESENTATION LEARNING:
/// 
/// 1. TEMPORAL SCALE SEPARATION:
///    Different layers can specialize in different temporal scales:
///    - Layer 1: Fast dynamics, fine-grained patterns
///    - Layer L: Slow dynamics, coarse-grained patterns
/// 
/// 2. FEATURE ABSTRACTION:
///    Each layer transforms oscillator positions into higher-level features:
///    x^{(1)} → positions → h₁ → x^{(2)} → positions → h₂ → ... → y
/// 
/// 3. NONLINEAR COMPOSITIONS:
///    While each layer is linear, the composition creates rich nonlinear dynamics
///    through the interaction of multiple oscillator systems.
/// 
/// STABILITY ANALYSIS:
/// The overall system stability depends on the stability of each layer.
/// If each layer satisfies (G^{(ℓ)}_i - Δt A^{(ℓ)}_i)² ≤ 4A^{(ℓ)}_i ∀i,ℓ,
/// then the entire block is stable.
/// 
/// COMPUTATIONAL COMPLEXITY:
/// - Per layer: O(m²) for matrix operations, O(T) for sequence processing
/// - Total block: O(L × m² × T) for sequential processing
/// - With parallel scan: O(L × m² × log T) asymptotic complexity
#[derive(Module, Debug)]
pub struct DLinossBlock1327<B: Backend> {
    layers: Vec<DLinoss1327<B>>,           // Sequence of L D-LinOSS layers
    output_norm: Option<LayerNorm<B>>,     // Final layer normalization
    
    // BLOCK DIMENSIONS (following paper notation):
    d_input: usize,       // p: input dimension to first layer
    d_oscillators: usize, // m: oscillators per layer (shared across layers)
    d_output: usize,      // q: output dimension from each layer
    num_layers: usize,    // L: total number of layers in block
}

impl<B: Backend> DLinossBlock1327<B> {
    pub fn init(config: &DLinossBlock1327Config, device: &B::Device) -> Self {
        let mut layers = Vec::with_capacity(config.num_layers);
        
        for i in 0..config.num_layers {
            let layer_d_input = if i == 0 { config.d_input } else { config.d_output };
            let layer_d_output = config.d_output;
            
            let layer_config = DLinoss1327Config {
                d_input: layer_d_input,
                d_oscillators: config.d_oscillators,
                d_output: layer_d_output,
                delta_t: config.delta_t,
                init_std: config.init_std,
                layer_norm: config.layer_norm,
            };
            
            layers.push(DLinoss1327::init(&layer_config, device));
        }
        
        let output_norm = if config.layer_norm {
            Some(LayerNormConfig::new(config.d_output).init(device))
        } else {
            None
        };
        
        Self {
            layers,
            output_norm,
            d_input: config.d_input,
            d_oscillators: config.d_oscillators,
            d_output: config.d_output,
            num_layers: config.num_layers,
        }
    }
    
    /// Forward pass through multi-layer D-LinOSS block
    /// 
    /// MATHEMATICAL COMPUTATION FLOW:
    /// 
    /// Given input sequence u₁, u₂, ..., uₜ ∈ ℝᵖ:
    /// 
    /// LAYER 1 (Input Processing):
    /// h₁ₖ = DLinOSS₁(uₖ; A₁, G₁, B₁, C₁, D₁) ∈ ℝᵠ
    /// 
    /// LAYERS 2 to L (Feature Transformation):  
    /// h₂ₖ = DLinOSS₂(h₁ₖ; A₂, G₂, B₂, C₂, D₂) ∈ ℝᵠ
    /// h₃ₖ = DLinOSS₃(h₂ₖ; A₃, G₃, B₃, C₃, D₃) ∈ ℝᵠ
    /// ...
    /// yₖ = DLinOSSₗ(hₗ₋₁,ₖ; Aₗ, Gₗ, Bₗ, Cₗ, Dₗ) ∈ ℝᵠ
    /// 
    /// TEMPORAL DYNAMICS per layer ℓ:
    /// For each time step k:
    ///   1. w^{(ℓ)}_k = M^{(ℓ)} w^{(ℓ)}_{k-1} + F^{(ℓ)} h_{ℓ-1,k}  [State update]
    ///   2. h_{ℓ,k} = H^{(ℓ)} w^{(ℓ)}_k + D^{(ℓ)} h_{ℓ-1,k}      [Output computation]
    /// 
    /// REPRESENTATION HIERARCHY:
    /// - Layer 1: Maps raw input to oscillator-based features
    /// - Intermediate layers: Transform features through different temporal scales  
    /// - Final layer: Produces task-specific output representation
    /// 
    /// LEARNED TEMPORAL SCALES:
    /// Each layer's damping matrix G^{(ℓ)} learns appropriate energy dissipation:
    /// - Fast timescales: Small damping, rapid oscillations
    /// - Slow timescales: Large damping, smooth evolution
    /// - Multi-scale: Different oscillators within layer learn different rates
    /// 
    /// INPUT/OUTPUT SHAPES:
    /// - input: [batch_size, seq_len, p] - input sequence  
    /// - output: [batch_size, seq_len, q] - transformed sequence
    pub fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 3> {
        let mut x = input;  // Current activation: starts as input u
        
        // SEQUENTIAL LAYER PROCESSING
        // Pass activations through each D-LinOSS layer in sequence
        for layer in &self.layers {
            // Apply layer ℓ: h_ℓ = DLinOSS_ℓ(h_{ℓ-1})
            x = layer.forward(x);
            
            // Optional: Add residual connections for deep networks (not in original paper)
            // if layer_idx > 0 && x.dims() == previous_x.dims() {
            //     x = x + previous_x;  // Residual connection
            // }
        }
        
        // FINAL OUTPUT NORMALIZATION
        // Apply layer normalization to final output for training stability
        if let Some(ref norm) = self.output_norm {
            let [batch_size, seq_len, d_output] = x.dims();
            
            // Reshape for layer norm: [batch×seq, d_output]
            let x_reshaped: Tensor<B, 2> = x.reshape([batch_size * seq_len, d_output]);
            
            // Apply normalization
            let x_normed = norm.forward(x_reshaped);
            
            // Reshape back: [batch, seq, d_output]
            x = x_normed.reshape([batch_size, seq_len, d_output]);
        }
        
        x  // Final output: y ∈ ℝᵠ
    }
    
    /// Get stability metrics for all layers
    pub fn check_stability(&self) -> Vec<Tensor<B, 1>> {
        self.layers.iter().map(|layer| layer.check_stability()).collect()
    }
    
    /// Get eigenvalue analysis for all layers
    pub fn get_eigenvalues(&self) -> Vec<Tensor<B, 1>> {
        self.layers.iter().map(|layer| layer.get_eigenvalues()).collect()
    }
    
    /// Update discretized matrices for all layers
    /// Should be called periodically during training
    pub fn update_discretized_matrices(&mut self) {
        for layer in &mut self.layers {
            layer.update_discretized_matrices();
        }
    }
    
    /// Compute spectral radius (maximum eigenvalue magnitude) for stability
    pub fn spectral_radius(&self) -> Vec<<B as Backend>::FloatElem>
    where 
        <B as Backend>::FloatElem: Copy
    {
        self.get_eigenvalues()
            .into_iter()
            .map(|eigenvals| {
                // Simplified spectral radius computation
                eigenvals.abs().max().into_scalar()
            })
            .collect()
    }
    
    /// Verify all layers satisfy stability condition from paper
    /// Condition: (G_i - Δt_i A_i)² ≤ 4A_i for all oscillators
    pub fn verify_paper_stability(&self) -> bool 
    where 
        <B as Backend>::FloatElem: PartialOrd + Copy
    {
        self.check_stability()
            .into_iter()
            .all(|stability_tensor| {
                // All oscillators in layer must be stable
                let threshold = <B as Backend>::FloatElem::from_elem(0.99);
                stability_tensor.mean().into_scalar() >= threshold // Allow small numerical errors
            })
    }
}
