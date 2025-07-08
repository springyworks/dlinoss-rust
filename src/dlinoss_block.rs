use burn::prelude::*;
use burn::nn::LayerNorm;
use crate::{DLinossLayer, DLinossLayerConfig};

/// DLinossBlock: A stack of DLinossLayer(s) with optional skip connections and normalization.
#[derive(Module, Debug)]
pub struct DLinossBlock<B: Backend> {
    pub layers: Vec<DLinossLayer<B>>,
    pub norm: Option<LayerNorm<B>>,
}

impl<B: Backend> DLinossBlock<B> {
    pub fn new(config: &DLinossLayerConfig, num_layers: usize, device: &B::Device) -> Self {
        let mut layers = Vec::with_capacity(num_layers);
        for _ in 0..num_layers {
            layers.push(DLinossLayer::new(config, device));
        }
        let norm = None; // Layer normalization removed as per paper analysis
        Self { layers, norm }
    }

    /// Forward pass through all layers in the block
    pub fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 3> {
        let mut x = input;
        for layer in &self.layers {
            x = layer.forward_3d(x);
        }
        if let Some(norm) = &self.norm {
            norm.forward(x)
        } else {
            x
        }
    }
}
