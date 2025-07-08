use burn::prelude::*;
use burn::nn::{
    conv::{Conv2d, Conv2dConfig},
    pool::{AdaptiveAvgPool2d, AdaptiveAvgPool2dConfig},
    Dropout, DropoutConfig, Relu,
};
use burn::tensor::{backend::Backend, Tensor, TensorData};

use crate::dlinoss_block_1327::{DLinossBlock1327, DLinossBlock1327Config};

/// Configuration for hybrid CNN + D-LinOSS architecture (arXiv:2505.12171)
/// 
/// ARCHITECTURAL DESIGN PHILOSOPHY:
/// This model combines the spatial feature extraction capabilities of CNNs
/// with the temporal dynamics modeling of D-LinOSS layers.
/// 
/// MATHEMATICAL PIPELINE:
/// Input Image → CNN Feature Extraction → Sequence Processing → D-LinOSS → Classification
/// 
/// CNN COMPONENT:
/// - Extracts spatial features from input images
/// - Reduces spatial dimensions while increasing feature depth
/// - Output: feature vector ∈ ℝᶠ where F = CNN feature dimension
/// 
/// D-LinOSS COMPONENT:
/// - Treats CNN features as single time step in sequence (for image classification)
/// - Can be extended to video sequences for temporal modeling
/// - Learns oscillatory dynamics for feature transformation
/// - Each oscillator processes CNN features through damped harmonic motion
/// 
/// HYPERPARAMETER SELECTION:
/// - num_classes: q (output dimension) - target classification categories
/// - d_oscillators: m (number of oscillators) - controls model capacity
/// - num_dlinoss_layers: L (layer depth) - enables hierarchical feature learning
/// - dropout: regularization to prevent overfitting
#[derive(Config, Debug)]
pub struct ModelConfig {
    pub num_classes: usize,        // q: Number of output classes
    #[config(default = "64")]
    pub d_oscillators: usize,      // m: Number of oscillators per D-LinOSS layer
    #[config(default = "2")]
    pub num_dlinoss_layers: usize, // L: Number of sequential D-LinOSS layers
    #[config(default = "0.5")]
    pub dropout: f64,              // Dropout probability for regularization
}

/// Hybrid CNN + D-LinOSS Model for Image Classification (arXiv:2505.12171)
/// 
/// FULL MATHEMATICAL ARCHITECTURE:
/// 
/// 1. CNN FEATURE EXTRACTION STAGE:
///    Input: I ∈ ℝᴴˣᵂˣᴄ (height × width × channels)
///    
///    Conv2D Layer 1: I → F₁ ∈ ℝᴴ¹ˣᵂ¹ˣᶜ¹ 
///    Activation: F₁ → ReLU(F₁)
///    Conv2D Layer 2: F₁ → F₂ ∈ ℝᴴ²ˣᵂ²ˣᶜ²
///    Pooling: F₂ → P ∈ ℝ⁸ˣ⁸ˣᶜ² (adaptive average pooling)
///    Flatten: P → f ∈ ℝᶠ where F = 8 × 8 × C₂
/// 
/// 2. SEQUENCE PREPARATION:
///    For image classification: treat as single time step
///    f → u ∈ ℝ¹ˣᶠ (sequence length = 1)
///    
///    For video/temporal data: f_t → u ∈ ℝᵀˣᶠ (sequence length = T)
/// 
/// 3. D-LinOSS DYNAMICS STAGE:
///    Input: u ∈ ℝᵀˣᶠ (CNN features as driving input)
///    
///    LAYER 1: u → h₁ via oscillator dynamics
///    - A₁ ∈ ℝᵐ: frequency parameters
///    - G₁ ∈ ℝᵐ: learnable damping parameters  
///    - State evolution: w₁ₖ₊₁ = M₁ w₁ₖ + F₁ uₖ
///    - Output: h₁ₖ = H₁ w₁ₖ + D₁ uₖ ∈ ℝᵠ
///    
///    LAYER L: hₗ₋₁ → hₗ via oscillator dynamics
///    - Each layer learns different temporal scales via Gₗ
///    - Final output: y ∈ ℝᵠ (classification logits)
/// 
/// 4. CLASSIFICATION:
///    y → softmax(y) → probability distribution over classes
/// 
/// KEY MATHEMATICAL PROPERTIES:
/// - Each D-LinOSS layer satisfies stability condition: (G_i - Δt A_i)² ≤ 4A_i
/// - CNN provides spatial inductive bias, D-LinOSS provides temporal dynamics
/// - Learned damping allows adaptive timescale modeling
/// - Composition enables hierarchical feature learning
/// 
/// PHYSICAL INTERPRETATION:
/// CNN features drive a system of coupled damped harmonic oscillators.
/// Each oscillator learns appropriate resonance frequency (A_i) and 
/// damping coefficient (G_i) for the classification task.
#[derive(Module, Debug)]
pub struct Model1327<B: Backend> {
    // CNN COMPONENTS for spatial feature extraction:
    conv1: Conv2d<B>,              // First convolutional layer: 1→8 channels
    conv2: Conv2d<B>,              // Second convolutional layer: 8→16 channels
    pool: AdaptiveAvgPool2d,       // Adaptive pooling to fixed 8×8 spatial size
    dropout: Dropout,              // Regularization layer
    activation: Relu,              // ReLU activation function
    
    // D-LinOSS COMPONENTS for temporal dynamics:
    dlinoss_block: DLinossBlock1327<B>, // Multi-layer oscillatory system
    
    // MODEL METADATA:
    params: usize,                 // Total parameter count for analysis
}

impl<B: Backend> Model1327<B> {
    pub fn init(config: &ModelConfig, device: &B::Device) -> Self {
        let [out1, out2] = [8, 16];
        
        let conv1 = Conv2dConfig::new([1, out1], [3, 3]).init(device);
        let conv2 = Conv2dConfig::new([out1, out2], [3, 3]).init(device);
        let pool = AdaptiveAvgPool2dConfig::new([8, 8]).init();
        let dropout = DropoutConfig::new(config.dropout).init();
        let activation = Relu::new();
        
        // Calculate flattened CNN feature size: 16 channels * 8 * 8 = 1024
        let cnn_features = out2 * 8 * 8;
        
        // D-LinOSS block configuration
        let dlinoss_config = DLinossBlock1327Config::init(
            cnn_features,              // Input from CNN features
            config.d_oscillators,      // Number of oscillators (m)
            config.num_classes,        // Output classes
        ).with_layers(config.num_dlinoss_layers);
        
        let dlinoss_block = DLinossBlock1327::init(&dlinoss_config, device);
        
        // Calculate total parameters
        let params = conv1.num_params() + conv2.num_params() + 
                    (config.d_oscillators * 2 * 5) + // A, G, B, C, D matrices approximation
                    (config.num_classes * config.d_oscillators) + 
                    (config.num_classes * cnn_features); // Rough parameter count
        
        Self {
            conv1,
            conv2,
            pool,
            dropout,
            activation,
            dlinoss_block,
            params,
        }
    }

    pub fn forward(&self, images: Tensor<B, 4>) -> Tensor<B, 2> {
        let [batch_size, channels, height, width] = images.dims();
        
        // CNN feature extraction
        let x = images.detach(); // Remove from computation graph for efficiency
        let x = self.conv1.forward(x);
        let x = self.activation.forward(x);
        let x = self.conv2.forward(x);
        let x = self.activation.forward(x);
        let x = self.dropout.forward(x);
        let x = self.pool.forward(x);
        
        // Flatten CNN features using original dimensions for validation
        let [_, c_out, h_out, w_out] = x.dims();
        
        // Validate dimensions are consistent with input processing
        assert!(c_out > 0 && h_out > 0 && w_out > 0, 
                "CNN output dimensions invalid: {}x{}x{} from input {}x{}x{}", 
                c_out, h_out, w_out, channels, height, width);
        
        let x = x.reshape([batch_size, c_out * h_out * w_out]);
        
        // Create sequence for D-LinOSS: [batch, seq_len=1, features]
        // For image classification, we treat it as a single time step
        let x = x.unsqueeze_dim(1);
        
        // D-LinOSS processing
        let x = self.dlinoss_block.forward(x);
        
        // Extract final classification: [batch, 1, num_classes] -> [batch, num_classes]
        let x = x.squeeze(1);
        
        x
    }
    
    pub fn num_params(&self) -> usize {
        self.params
    }
    
    /// Verify mathematical stability of D-LinOSS layers
    pub fn verify_stability(&self) -> bool 
    where 
        <B as Backend>::FloatElem: PartialOrd + Copy
    {
        self.dlinoss_block.verify_paper_stability()
    }
    
    /// Get spectral analysis of D-LinOSS layers
    pub fn spectral_analysis(&self) -> Vec<<B as Backend>::FloatElem>
    where 
        <B as Backend>::FloatElem: Copy
    {
        self.dlinoss_block.spectral_radius()
    }
    
    /// Update discretized matrices (should be called periodically during training)
    pub fn update_dlinoss_matrices(&mut self) {
        self.dlinoss_block.update_discretized_matrices();
    }
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self::new(10) // Default to 10 classes
    }
}

impl ModelConfig {
    pub fn init() -> Self {
        Self {
            num_classes: 10,
            d_oscillators: 64,
            num_dlinoss_layers: 2,
            dropout: 0.5,
        }
    }
}

/// Create model with proper device backend
pub fn create_model_1327<B: Backend>(device: &B::Device) -> Model1327<B> {
    let config = ModelConfig::init();
    Model1327::init(&config, device)
}

/// Synthetic Exponential Decay Benchmark (inspired by arXiv:2505.12171)
/// 
/// MATHEMATICAL TEST SETUP:
/// This benchmark validates D-LinOSS's ability to learn exponential decay dynamics,
/// which is a fundamental test for oscillatory systems with damping.
/// 
/// TEST SIGNAL GENERATION:
/// For each sequence, generate:
/// 1. Input: u(t) = δ(t) * A  (impulse at t=0 with amplitude A)
/// 2. Target: y(t) = A * exp(-γt)  (exponential decay with rate γ)
/// 
/// where:
/// - A ~ U(-1, 1): random initial amplitude
/// - γ ~ U(0.1, 0.9): random decay rate
/// - δ(t): Dirac delta function (impulse)
/// 
/// PHYSICAL INTERPRETATION:
/// This simulates a damped oscillator with:
/// - Initial displacement: A
/// - Critical damping: γ
/// - No external forcing after t=0
/// 
/// The D-LinOSS system should learn to:
/// 1. Map impulse input to appropriate oscillator state
/// 2. Configure damping matrix G to match target decay rate γ
/// 3. Output exponentially decaying signal matching target
/// 
/// SUCCESS CRITERIA:
/// - Low MSE between predicted and target exponential curves
/// - Learned damping parameters should correlate with target decay rates
/// - System should generalize to unseen decay rates
/// 
/// MATHEMATICAL VALIDATION:
/// For a second-order oscillator with critical damping:
/// x''(t) + 2γx'(t) + γ²x(t) = 0
/// Solution: x(t) = A(1 + γt)exp(-γt) ≈ A exp(-γt) for small γt
/// 
/// INPUT/OUTPUT SHAPES:
/// - input: [batch_size, seq_len, 1] - impulse sequences
/// - target: [batch_size, seq_len, 1] - exponential decay sequences
pub fn exponential_decay_benchmark<B: Backend>(
    device: &B::Device,
    seq_len: usize,     // T: sequence length (time steps)
    batch_size: usize,  // B: number of sequences in batch
) -> (Tensor<B, 3>, Tensor<B, 3>) {
    // Generate synthetic exponential decay sequences
    let mut input_data = Vec::new();   // Impulse inputs: δ(t) * A
    let mut target_data = Vec::new();  // Exponential targets: A * exp(-γt)
    
    for _ in 0..batch_size {
        // RANDOM PARAMETER SAMPLING:
        let decay_rate = 0.1 + (rand::random::<f64>() * 0.8); // γ ∈ [0.1, 0.9]
        let initial_value = rand::random::<f64>() * 2.0 - 1.0; // A ∈ [-1, 1]
        
        let mut sequence_input = Vec::new();
        let mut sequence_target = Vec::new();
        
        // SEQUENCE GENERATION:
        for t in 0..seq_len {
            let time = t as f64 * 0.1;  // Δt = 0.1 time units
            
            // Input: impulse at t=0, zero elsewhere
            let input_val = if t == 0 { initial_value } else { 0.0 };
            
            // Target: exponential decay y(t) = A * exp(-γt)
            let target_val = initial_value * (-decay_rate * time).exp();
            
            sequence_input.push(input_val as f32);
            sequence_target.push(target_val as f32);
        }
        
        input_data.push(sequence_input);
        target_data.push(sequence_target);
    }
    
    // TENSOR CONSTRUCTION using Burn TensorData API:
    let input_flat: Vec<f32> = input_data.into_iter().flatten().collect();
    let target_flat: Vec<f32> = target_data.into_iter().flatten().collect();
    
    // Create TensorData with proper dimensions [batch_size, seq_len, 1]
    let input_data = TensorData::new(input_flat, [batch_size, seq_len, 1]);
    let target_data = TensorData::new(target_flat, [batch_size, seq_len, 1]);
    
    // Convert to device tensors
    let input_tensor = Tensor::<B, 3>::from_data(input_data, device);
    let target_tensor = Tensor::<B, 3>::from_data(target_data, device);
    
    (input_tensor, target_tensor)
}

// TRAINING SUPPORT for Model1327
use burn::{
    train::{ClassificationOutput, TrainOutput, TrainStep, ValidStep},
    nn::loss::CrossEntropyLossConfig,
    tensor::backend::AutodiffBackend,
};
use crate::data::MnistBatch;

impl<B: Backend> Model1327<B> {
    /// Forward pass for classification training
    pub fn forward_classification(
        &self,
        images: Tensor<B, 3>,
        targets: Tensor<B, 1, Int>,
    ) -> ClassificationOutput<B> {
        // Add batch dimension for the forward pass
        let [batch_size, height, width] = images.dims();
        let images_4d = images.reshape([batch_size, 1, height, width]);
        let output = self.forward(images_4d);
        let loss = CrossEntropyLossConfig::new()
            .init(&output.device())
            .forward(output.clone(), targets.clone());

        ClassificationOutput::new(loss, output, targets)
    }
}

impl<B: AutodiffBackend> TrainStep<MnistBatch<B>, ClassificationOutput<B>> for Model1327<B> {
    fn step(&self, batch: MnistBatch<B>) -> TrainOutput<ClassificationOutput<B>> {
        let item = self.forward_classification(batch.images, batch.targets);
        TrainOutput::new(self, item.loss.backward(), item)
    }
}

impl<B: Backend> ValidStep<MnistBatch<B>, ClassificationOutput<B>> for Model1327<B> {
    fn step(&self, batch: MnistBatch<B>) -> ClassificationOutput<B> {
        self.forward_classification(batch.images, batch.targets)
    }
}
