#![recursion_limit = "131"]
use burn::data::dataset::Dataset;
use dlinoss_rust::{inference, device::{init_device, Backend}};

fn main() {
    // Use centralized device initialization
    let device = init_device();

    // All the training artifacts are saved in this directory
    let artifact_dir = "/tmp/dlinoss";

    // Infer the model
    inference::infer::<Backend>(
        artifact_dir,
        device,
        burn::data::dataset::vision::MnistDataset::test()
            .get(42)
            .unwrap(),
    );
}
