#![recursion_limit = "131"]
use burn::{backend::WebGpu, data::dataset::Dataset};
use dlinoss_rust::inference;

fn main() {
    type MyBackend = WebGpu<f32, i32>;

    let device = burn::backend::wgpu::WgpuDevice::default();

    // All the training artifacts are saved in this directory
    let artifact_dir = "/tmp/dlinoss";

    // Infer the model
    inference::infer::<MyBackend>(
        artifact_dir,
        device,
        burn::data::dataset::vision::MnistDataset::test()
            .get(42)
            .unwrap(),
    );
}
