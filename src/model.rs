use burn::{
	nn::{
		Dropout, DropoutConfig, Linear, LinearConfig, Relu,
		conv::{Conv2d, Conv2dConfig},
		pool::{AdaptiveAvgPool2d, AdaptiveAvgPool2dConfig},
	},
	prelude::*,
};
use crate::{DLinossBlock, DLinossLayerConfig};

#[derive(Module, Debug)]
pub struct Model<B: Backend> {
	conv1: Conv2d<B>,
	conv2: Conv2d<B>,
	pool: AdaptiveAvgPool2d,
	dropout: Dropout,
	activation: Relu,
	// Option 1: Use MLP head
	linear1: Option<Linear<B>>,
	linear2: Option<Linear<B>>,
	// Option 2: Use D-LinOSS block head
	dlinoss_block: Option<DLinossBlock<B>>,
}

#[derive(Config, Debug)]
pub struct ModelConfig {
	pub num_classes: usize,
	pub hidden_size: usize,
	#[config(default = "0.5")]
	pub dropout: f64,
	#[config(default = "false")]
	pub use_dlinoss: bool,
}

impl ModelConfig {
	/// Returns the initialized model.
	pub fn init<B: Backend>(&self, device: &B::Device) -> Model<B> {
		if self.use_dlinoss {
			// Use D-LinOSS block as head
			let dlinoss_config = DLinossLayerConfig::dlinoss_config(16 * 8 * 8, self.hidden_size, self.num_classes);
			Model {
				conv1: Conv2dConfig::new([1, 8], [3, 3]).init(device),
				conv2: Conv2dConfig::new([8, 16], [3, 3]).init(device),
				pool: AdaptiveAvgPool2dConfig::new([8, 8]).init(),
				activation: Relu::new(),
				linear1: None,
				linear2: None,
				dlinoss_block: Some(DLinossBlock::new(&dlinoss_config, 1, device)),
				dropout: DropoutConfig::new(self.dropout).init(),
			}
		} else {
			// Use MLP head
			Model {
				conv1: Conv2dConfig::new([1, 8], [3, 3]).init(device),
				conv2: Conv2dConfig::new([8, 16], [3, 3]).init(device),
				pool: AdaptiveAvgPool2dConfig::new([8, 8]).init(),
				activation: Relu::new(),
				linear1: Some(LinearConfig::new(16 * 8 * 8, self.hidden_size).init(device)),
				linear2: Some(LinearConfig::new(self.hidden_size, self.num_classes).init(device)),
				dlinoss_block: None,
				dropout: DropoutConfig::new(self.dropout).init(),
			}
		}
	}
}

impl<B: Backend> Model<B> {
	/// # Shapes
	///   - Images [batch_size, height, width]
	///   - Output [batch_size, class_prob]
	pub fn forward(&self, images: Tensor<B, 3>) -> Tensor<B, 2> {
		let [batch_size, height, width] = images.dims();
		let x = images.reshape([batch_size, 1, height, width]);
		let x = self.conv1.forward(x);
		let x = self.dropout.forward(x);
		let x = self.conv2.forward(x);
		let x = self.dropout.forward(x);
		let x = self.activation.forward(x);
		let x = self.pool.forward(x);
		let x = x.reshape([batch_size, 16 * 8 * 8]);
		if let Some(dlinoss_block) = &self.dlinoss_block {
			// D-LinOSS expects [batch, seq_len, d_input], so treat each sample as a sequence of length 1
			let x = x.reshape([batch_size, 1, 16 * 8 * 8]);
			let out = dlinoss_block.forward(x); // [batch, 1, num_classes]
			out.squeeze(1) // [batch, num_classes]
		} else {
			let mut x = x;
			if let Some(linear1) = &self.linear1 {
				x = linear1.forward(x);
				x = self.dropout.forward(x);
				x = self.activation.forward(x);
			}
			if let Some(linear2) = &self.linear2 {
				x = linear2.forward(x);
			}
			x
		}
	}
}
