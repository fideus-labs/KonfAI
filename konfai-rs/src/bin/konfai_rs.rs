// SPDX-License-Identifier: Apache-2.0
//! konfai-rs CLI: run portable inference of the baked KonfAI model.

use clap::{Parser, Subcommand, ValueEnum};
use konfai_rs::{infer_2d, Manifest};
use std::fs;
use std::io::{Read, Write};

#[derive(Parser)]
#[command(name = "konfai-rs", about = "Portable KonfAI inference (Burn + ONNX)")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Run inference on a single fixed-size patch (raw little-endian f32 in/out).
    Infer {
        #[arg(long)]
        manifest: String,
        #[arg(long)]
        input: String,
        #[arg(long)]
        output: String,
        #[arg(long, value_enum, default_value_t = BackendKind::Cpu)]
        backend: BackendKind,
    },
}

#[derive(Clone, ValueEnum)]
enum BackendKind {
    Cpu,
    Wgpu,
}

fn read_f32(path: &str) -> Vec<f32> {
    let mut buf = Vec::new();
    fs::File::open(path)
        .unwrap_or_else(|e| panic!("konfai-rs: cannot open {path}: {e}"))
        .read_to_end(&mut buf)
        .unwrap();
    buf.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

fn main() {
    match Cli::parse().cmd {
        Cmd::Infer { manifest, input, output, backend } => {
            let m = Manifest::from_file(&manifest);
            assert_eq!(m.patch.dim, 2, "konfai-rs: only 2D/2.5D patches supported for now");
            let (h, w) = (m.patch.size[0], m.patch.size[1]);
            let data = read_f32(&input);

            let out = match backend {
                BackendKind::Cpu => {
                    use burn::backend::NdArray;
                    infer_2d::<NdArray<f32>>(&Default::default(), data, m.input.channels, h, w)
                }
                BackendKind::Wgpu => {
                    use burn::backend::wgpu::{Wgpu, WgpuDevice};
                    infer_2d::<Wgpu<f32, i32>>(&WgpuDevice::default(), data, m.input.channels, h, w)
                }
            };

            let mut f = fs::File::create(&output).unwrap_or_else(|e| panic!("konfai-rs: cannot write {output}: {e}"));
            for v in &out {
                f.write_all(&v.to_le_bytes()).unwrap();
            }
            eprintln!("konfai-rs: {} values ({}x{}, {} out-ch) -> {}", out.len(), h, w, m.output.channels, output);
        }
    }
}
