use burn_onnx::ModelGen;
fn main() {
    ModelGen::new()
        .input("assets-impact/model.onnx")
        .out_dir("model/")
        .run_from_script();
}
