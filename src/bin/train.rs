#![recursion_limit = "256"]

use burn::{
    backend::{Autodiff, Wgpu},
    data::{dataloader::DataLoaderBuilder, dataset::vision::MnistDataset},
    optim::AdamConfig,
    prelude::*,
    record::CompactRecorder,
    train::{
        metric::{AccuracyMetric, LossMetric},
        LearnerBuilder,
    },
};
use dlinoss_rust::{
    data::MnistBatcher,
    model::ModelConfig,
};

#[derive(Config)]
pub struct TrainingConfig {
    pub model: ModelConfig,
    pub optimizer: AdamConfig,
    #[config(default = 10)]
    pub num_epochs: usize,
    #[config(default = 64)]
    pub batch_size: usize,
    #[config(default = 4)]
    pub num_workers: usize,
    #[config(default = 42)]
    pub seed: u64,
    #[config(default = 1.0e-4)]
    pub learning_rate: f64,
}

fn create_artifact_dir(artifact_dir: &str) {
    // Remove existing artifacts before to get an accurate learner summary
    std::fs::remove_dir_all(artifact_dir).ok();
    std::fs::create_dir_all(artifact_dir).ok();
}

fn main() {
    type MyBackend = Wgpu<f32, i32>;
    type MyAutodiffBackend = Autodiff<MyBackend>;

    println!("🧠 D-LinOSS Training - Using Burn's Built-in Progress Display");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    // Create a default Wgpu device
    let device = burn::backend::wgpu::WgpuDevice::default();
    println!("🚀 Initializing WGPU device: {:?}", device);

    // All the training artifacts will be saved in this directory
    let artifact_dir = "/tmp/dlinoss-training";
    create_artifact_dir(artifact_dir);

    // Create training configuration
    let config = TrainingConfig {
        model: ModelConfig::new(10, 512).with_use_dlinoss(true), // Enable D-LinOSS
        optimizer: AdamConfig::new(),
        num_epochs: 5,
        batch_size: 32,
        num_workers: 4,
        seed: 42,
        learning_rate: 1.0e-3,
    };

    println!("📊 Model Configuration:");
    println!("  - Classes: {}", config.model.num_classes);
    println!("  - Hidden Size: {}", config.model.hidden_size);
    println!("  - Dropout: {:.2}", config.model.dropout);
    println!("  - D-LinOSS: {}", if config.model.use_dlinoss { "✓ ENABLED" } else { "✗ DISABLED" });
    println!();

    config
        .save(format!("{artifact_dir}/config.json"))
        .expect("Config should be saved successfully");

    MyBackend::seed(config.seed);

    let batcher = MnistBatcher::default();

    println!("📦 Loading MNIST dataset...");
    let dataloader_train = DataLoaderBuilder::new(batcher.clone())
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(MnistDataset::train());

    let dataloader_test = DataLoaderBuilder::new(batcher)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(MnistDataset::test());

    println!("🎯 Starting D-LinOSS training with Burn's built-in progress display...");
    println!();

    let learner = LearnerBuilder::new(artifact_dir)
        .metric_train_numeric(AccuracyMetric::new())
        .metric_valid_numeric(AccuracyMetric::new())
        .metric_train_numeric(LossMetric::new())
        .metric_valid_numeric(LossMetric::new())
        .with_file_checkpointer(CompactRecorder::new())
        .devices(vec![device.clone()])
        .num_epochs(config.num_epochs)
        .summary()
        .build(
            config.model.init::<MyAutodiffBackend>(&device),
            config.optimizer.init(),
            config.learning_rate,
        );

    let model_trained = learner.fit(dataloader_train, dataloader_test);

    println!();
    println!("💾 Saving trained model...");
    model_trained
        .save_file(format!("{artifact_dir}/model"), &CompactRecorder::new())
        .expect("Trained model should be saved successfully");

    println!("✅ Training completed successfully!");
    println!("📁 Artifacts saved in: {}", artifact_dir);
}
