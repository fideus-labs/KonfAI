// SPDX-License-Identifier: Apache-2.0
//! konfai-rs — portable, Python-free inference for KonfAI models via Burn + ONNX.
//!
//! The model architecture is generated at build time from `models/model.onnx`
//! (see build.rs). This crate provides the runtime: manifest parsing and a
//! backend-generic forward over a single patch. Volume tiling + NIfTI I/O land
//! in a later phase; today the unit of work is one fixed-size patch.

use burn::prelude::Backend;
use burn::tensor::{Tensor, TensorData};
use serde::Deserialize;

pub mod generated {
    include!(concat!(env!("OUT_DIR"), "/model/model.rs"));
}
use generated::Model;

/// The konfai-rs contract emitted by `konfai export` alongside `model.onnx`.
#[derive(Debug, Deserialize)]
pub struct Manifest {
    pub input: ChannelSpec,
    pub output: ChannelSpec,
    pub patch: PatchSpec,
}

#[derive(Debug, Deserialize)]
pub struct ChannelSpec {
    pub channels: usize,
}

#[derive(Debug, Deserialize)]
pub struct PatchSpec {
    pub size: Vec<usize>,
    pub dim: usize,
}

impl Manifest {
    pub fn from_file(path: &str) -> Self {
        let text = std::fs::read_to_string(path)
            .unwrap_or_else(|e| panic!("konfai-rs: cannot read manifest {path}: {e}"));
        serde_json::from_str(&text).unwrap_or_else(|e| panic!("konfai-rs: invalid manifest {path}: {e}"))
    }
}

/// Run the baked model on one 2D patch `[1, C, H, W]`; returns the flat output.
pub fn infer_2d<B: Backend>(device: &B::Device, input: Vec<f32>, channels: usize, height: usize, width: usize) -> Vec<f32> {
    let model = Model::<B>::default();
    let tensor = Tensor::<B, 4>::from_data(TensorData::new(input, [1, channels, height, width]), device);
    model.forward(tensor).into_data().to_vec().expect("f32 output")
}
