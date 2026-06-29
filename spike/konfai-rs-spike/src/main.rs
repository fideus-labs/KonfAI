use burn::backend::NdArray;
use burn::tensor::{Tensor, TensorData};
use std::io::Read;

mod model {
    include!(concat!(env!("OUT_DIR"), "/model/model.rs"));
}
use model::Model;

type B = NdArray<f32>;

fn read_f32(path: &str) -> Vec<f32> {
    let mut buf = Vec::new();
    std::fs::File::open(path).unwrap().read_to_end(&mut buf).unwrap();
    buf.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

fn main() {
    let device = Default::default();
    let model: Model<B> = Model::default();

    let input_v = read_f32("assets/input_f32.bin");
    let reference = read_f32("assets/ref_f32.bin");

    let input = Tensor::<B, 4>::from_data(TensorData::new(input_v, [1, 1, 256, 256]), &device);
    let output = model.forward(input);
    let out_v: Vec<f32> = output.into_data().to_vec().unwrap();

    let n = out_v.len().min(reference.len());
    let mae: f64 = (0..n).map(|i| (out_v[i] as f64 - reference[i] as f64).abs()).sum::<f64>() / n as f64;
    let mx: f64 = (0..n).map(|i| (out_v[i] as f64 - reference[i] as f64).abs()).fold(0.0, f64::max);
    println!("burn out len={} ref len={} | MAE={:.3e} max={:.3e}", out_v.len(), reference.len(), mae, mx);
    println!("{}", if mae < 1e-3 { "SPIKE_BURN_OK" } else { "SPIKE_BURN_DIVERGE" });
}
