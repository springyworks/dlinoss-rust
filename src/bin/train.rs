#![recursion_limit = "131"]
use burn::{
    data::dataset::Dataset,
    optim::AdamConfig,
};
use dlinoss_rust::{
    inference,
    model::ModelConfig,
    training::{self, TrainingConfig},
    device::{init_device, Backend, AutodiffBackend},
};

fn main() {
    // Use centralized device initialization
    let device = init_device();

    // All the training artifacts will be saved in this directory
    let artifact_dir = "/tmp/dlinoss";

    // Train the model
    training::train::<AutodiffBackend>(
        artifact_dir,
        TrainingConfig::new(
            ModelConfig {
                num_classes: 10,
                hidden_size: 64, // Smaller hidden size for D-LinOSS
                dropout: 0.5,
                use_dlinoss: true, // Test with D-LinOSS
            }, 
            AdamConfig::new()
        ),
        device.clone(),
    );

    // Infer the model
    inference::infer::<Backend>(
        artifact_dir,
        device,
        burn::data::dataset::vision::MnistDataset::test()
            .get(42)
            .unwrap(),
    );
}
