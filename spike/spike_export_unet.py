"""Phase-0 spike (v2): export a KonfAI UNet via named_forward + a thin wrapper.

The Network.forward returns per-output-group results (empty without .init()). The raw
graph is reachable via ModuleArgsDict.named_forward, which yields (dotted_name, tensor)
for every module. We pick a named head (Softmax = float logits, good for parity) and
export THAT through a small wrapper -> the "what to export" decision in the plan, made concrete.
"""

import os
import tempfile

import numpy as np
import torch
import torch.nn as nn

os.environ.setdefault("KONFAI_CONFIG_MODE", "Done")
os.environ.setdefault("KONFAI_config_file", os.path.join(tempfile.gettempdir(), "spike_dummy.yml"))

from konfai.models.segmentation.UNet import UNet  # noqa: E402

PATCH = (1, 1, 256, 256)


class HeadExport(nn.Module):
    """Run the routed graph, return one named output (by exact dotted name)."""

    def __init__(self, net: nn.Module, out_name: str) -> None:
        super().__init__()
        self.net = net
        self.out_name = out_name

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x
        for name, t in self.net.named_forward(x):
            if name == self.out_name:
                out = t
        return out


def main() -> None:
    model = UNet(dim=2, channels=[1, 32, 64, 128, 256], nb_class=2)
    model.eval()
    print(f">>> params: {sum(p.numel() for p in model.parameters()):,}")

    x = torch.randn(*PATCH)

    print(">>> named_forward output names:")
    names = []
    with torch.no_grad():
        for name, t in model.named_forward(x):
            names.append((name, tuple(t.shape), str(t.dtype)))
    for n, s, d in names:
        print(f"    {n:55s} {s} {d}")

    # top-level full-res inference head (matches the default outputs_criterions key
    # "UNetBlock_0:Head:Argmax"); fall back to the last Softmax yielded.
    soft = [n for n, _, d in names if n.endswith("Softmax")]
    preferred = "UNetBlock_0.Head.Softmax"
    out_name = preferred if preferred in [n for n, _, _ in names] else (soft[-1] if soft else names[-1][0])
    print(f">>> exporting head: {out_name}")

    wrapper = HeadExport(model, out_name).eval()
    with torch.no_grad():
        ref = wrapper(x)
    print(f"    ref output shape {tuple(ref.shape)} dtype {ref.dtype}")

    onnx_path = os.path.join(tempfile.gettempdir(), "spike_unet.onnx")

    def _do_export(dynamo: bool) -> None:
        with torch.no_grad():
            torch.onnx.export(
                wrapper, x, onnx_path, opset_version=17,
                input_names=["input"], output_names=["output"], dynamic_axes=None,
                dynamo=dynamo,
            )

    exported_via = None
    try:
        print(">>> attempt A: dynamo=True exporter")
        _do_export(True)
        exported_via = "dynamo"
    except Exception as e:  # noqa: BLE001
        print(f"    dynamo export failed: {type(e).__name__}: {str(e)[:140]}")
        print(">>> attempt B: restore std nn.Module.state_dict on the Network + legacy exporter")
        import types
        # KonfAI Network overrides state_dict() with a custom signature that the jit tracer
        # cannot call. Temporarily rebind the standard one for the duration of export.
        for m in model.modules():
            m.state_dict = types.MethodType(nn.Module.state_dict, m)
        try:
            _do_export(False)
            exported_via = "legacy+state_dict-restore"
        except Exception as e2:  # noqa: BLE001
            print(f"!!! legacy export also failed: {type(e2).__name__}: {str(e2)[:200]}")
            raise
    print(f">>> exported via: {exported_via} -> {onnx_path}")

    import onnxruntime as ort

    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    ort_out = sess.run(None, {"input": x.numpy().astype(np.float32)})[0]
    ref_np = ref.numpy()
    mae = float(np.mean(np.abs(ort_out.astype(np.float64) - ref_np.astype(np.float64))))
    mx = float(np.max(np.abs(ort_out.astype(np.float64) - ref_np.astype(np.float64))))
    print(f">>> onnxruntime parity: shape {ort_out.shape} | MAE={mae:.3e} max={mx:.3e}")

    # persist artifacts for the Rust/burn-onnx half of the spike
    assets = os.environ.get("SPIKE_ASSETS")
    if assets:
        os.makedirs(assets, exist_ok=True)
        # dynamo export writes weights as EXTERNAL data (.onnx.data sidecar). burn-onnx's
        # ModelGen needs a self-contained file -> reload (pulls in the sidecar) and re-save
        # with weights embedded in a single .onnx.
        import onnx
        m = onnx.load(onnx_path)  # loads external data from the sidecar in the same dir
        onnx.save(m, os.path.join(assets, "model.onnx"), save_as_external_data=False)
        x.numpy().astype(np.float32).tofile(os.path.join(assets, "input_f32.bin"))
        ort_out.astype(np.float32).tofile(os.path.join(assets, "ref_f32.bin"))
        with open(os.path.join(assets, "shapes.txt"), "w") as f:
            f.write(f"input={tuple(x.shape)}\noutput={tuple(ort_out.shape)}\n")
        print(f">>> artifacts written to {assets}")

    print("SPIKE_EXPORT_OK" if mae < 1e-4 else "SPIKE_EXPORT_DIVERGE")


if __name__ == "__main__":
    main()
