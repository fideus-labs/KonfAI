"""Phase-0 spike: export the REAL impact_synth (sCT) generator to a self-contained ONNX.

impact_synth MR = a KonfAI Network wrapping `segmentation_models_pytorch.UnetPlusPlus`
(resnet34 encoder, in_channels=5 [2.5D], classes=1) + a Tanh head. 2D / 2.5D, no 3D conv.

Download the variant's Model.py + config from HF, instantiate the generator (random weights
are fine to prove the import/op-coverage + chain parity), export via the dynamo exporter,
inline weights, and check onnxruntime parity. Writes assets for the Rust/burn-onnx half.

Env: HF_VARIANT (default MR), SPIKE_ASSETS (output dir). Needs `segmentation-models-pytorch`,
`onnx`, `onnxruntime`, `onnxscript` in the env.
"""

import os
import sys
import tempfile

import numpy as np
import torch
import torch.nn as nn

os.environ.setdefault("KONFAI_CONFIG_MODE", "Done")
os.environ.setdefault("KONFAI_config_file", os.path.join(tempfile.gettempdir(), "d.yml"))

VARIANT = os.environ.get("HF_VARIANT", "MR")
ASSETS = os.environ["SPIKE_ASSETS"]
SHAPE = (1, 5, 256, 256)  # 2.5D: 5 channels, 256x256


def main() -> None:
    from huggingface_hub import hf_hub_download

    work = tempfile.mkdtemp()
    for f in (f"{VARIANT}/Model.py", f"{VARIANT}/Prediction.yml"):
        hf_hub_download("VBoussot/ImpactSynth", f, local_dir=work)
    sys.path.insert(0, os.path.join(work, VARIANT))
    from Model import UNetpp  # noqa

    try:
        model = UNetpp(nb_channel=SHAPE[1], outputs_criterions=None)
    except Exception:
        model = UNetpp(nb_channel=SHAPE[1])
    model.eval()
    print(f">>> params: {sum(p.numel() for p in model.parameters()):,}")

    x = torch.randn(*SHAPE)
    names = []
    with torch.no_grad():
        for n, t in model.named_forward(x):
            names.append(n)
    out_name = names[-1]  # Head.Tanh for impact_synth
    print(f">>> export head: {out_name}")

    class Wrap(nn.Module):
        def __init__(self, net, nm):
            super().__init__()
            self.net, self.nm = net, nm

        def forward(self, x):
            o = x
            for n, t in self.net.named_forward(x):
                if n == self.nm:
                    o = t
            return o

    wrap = Wrap(model, out_name).eval()
    with torch.no_grad():
        ref = wrap(x)

    tmp = os.path.join(tempfile.gettempdir(), "impact.onnx")
    with torch.no_grad():
        torch.onnx.export(wrap, x, tmp, opset_version=18,
                          input_names=["input"], output_names=["output"], dynamo=True)

    import onnx
    os.makedirs(ASSETS, exist_ok=True)
    onnx.save(onnx.load(tmp), os.path.join(ASSETS, "model.onnx"), save_as_external_data=False)

    import onnxruntime as ort
    sess = ort.InferenceSession(os.path.join(ASSETS, "model.onnx"), providers=["CPUExecutionProvider"])
    o = sess.run(None, {"input": x.numpy().astype(np.float32)})[0]
    x.numpy().astype(np.float32).tofile(os.path.join(ASSETS, "input_f32.bin"))
    o.astype(np.float32).tofile(os.path.join(ASSETS, "ref_f32.bin"))
    mae = float(np.mean(np.abs(o.astype(np.float64) - ref.numpy().astype(np.float64))))
    print(f">>> onnxruntime parity: out {o.shape} MAE={mae:.3e}")
    print("IMPACT_EXPORT_OK" if mae < 1e-4 else "IMPACT_EXPORT_DIVERGE")


if __name__ == "__main__":
    main()
