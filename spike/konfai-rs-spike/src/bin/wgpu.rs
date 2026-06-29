use burn::backend::wgpu::{Wgpu, WgpuDevice};
use burn::tensor::{Tensor, TensorData};
use std::io::Read;
mod model { include!(concat!(env!("OUT_DIR"), "/model/model.rs")); }
use model::Model;
type B = Wgpu<f32, i32>;
fn read_f32(p: &str) -> Vec<f32> {
    let mut b = Vec::new();
    std::fs::File::open(p).unwrap().read_to_end(&mut b).unwrap();
    b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}
fn main() {
    let device = WgpuDevice::default();
    let model: Model<B> = Model::default();
    let inp = read_f32("assets-impact/input_f32.bin");
    let rf = read_f32("assets-impact/ref_f32.bin");
    let input = Tensor::<B, 4>::from_data(TensorData::new(inp, [1, 5, 256, 256]), &device);
    let out: Vec<f32> = model.forward(input).into_data().to_vec().unwrap();
    let n = out.len().min(rf.len());
    let mae: f64 = (0..n).map(|i| (out[i] as f64 - rf[i] as f64).abs()).sum::<f64>() / n as f64;
    let mx: f64 = (0..n).map(|i| (out[i] as f64 - rf[i] as f64).abs()).fold(0.0, f64::max);
    println!("[WGPU] impact_synth burn out len={} | MAE={:.3e} max={:.3e}", out.len(), mae, mx);
    println!("{}", if mae < 1e-3 { "IMPACT_WGPU_OK" } else { "IMPACT_WGPU_DIVERGE" });
}
