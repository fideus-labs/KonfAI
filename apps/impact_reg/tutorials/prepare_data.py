"""The tutorial pair: a head and neck CT (fixed) and the MRI of the same patient (moving), from the public
SynthRAD2025 cases of the KonfAI demo dataset, at 2 mm. The MRI is moved by a known transform, a rigid one and a smooth
deformation of up to about 8 mm, so every tutorial can check its result: label maps and landmarks are written for both
images, the moving ones through that transform. A rigid registration recovers the rigid part only; the deformation is
what a deformable preset is there for.

    python prepare_data.py [output_dir]      # default: ./data
"""

import json
import sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from huggingface_hub import hf_hub_download

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "data")
OUT.mkdir(parents=True, exist_ok=True)


def fetch(name: str, pixel: int) -> sitk.Image:
    path = hf_hub_download("VBoussot/konfai-demo", f"Synthesis/1HNA001/{name}.mha", repo_type="dataset")
    return sitk.ReadImage(path, pixel)


ct, mr, body = fetch("CT", sitk.sitkFloat32), fetch("MR", sitk.sitkFloat32), fetch("MASK", sitk.sitkUInt8)

# A 2 mm grid over the same extent: a few million voxels, registered in minutes.
size = [round(n * s / 2.0) for n, s in zip(ct.GetSize(), ct.GetSpacing(), strict=True)]
grid = sitk.Image(size, sitk.sitkFloat32)
grid.SetSpacing([2.0, 2.0, 2.0])
grid.SetOrigin(ct.GetOrigin())


def on(image: sitk.Image, transform: sitk.Transform, fill: float, nearest: bool = False) -> sitk.Image:
    interpolator = sitk.sitkNearestNeighbor if nearest else sitk.sitkLinear
    return sitk.Resample(image, grid, transform, interpolator, fill, image.GetPixelID())


identity = sitk.Transform(3, sitk.sitkIdentity)
fixed, fixed_body = on(ct, identity, -1000.0), on(body, identity, 0, nearest=True)

# Labels on the fixed CT: 1 soft tissue, 2 bone, 3 air inside the body.
hu, inside = sitk.GetArrayFromImage(fixed), sitk.GetArrayFromImage(fixed_body) > 0
labels = np.zeros(hu.shape, np.uint8)
labels[inside & (hu > -200)] = 1
labels[inside & (hu > 250)] = 2
labels[inside & (hu < -400)] = 3
fixed_labels = sitk.GetImageFromArray(labels)
fixed_labels.CopyInformation(fixed)

# The known misalignment psi: moving(y) = the anatomy at psi(y) = T(y + B(y)), T rigid and B three smooth Gaussian
# bumps (about 12 mm each, sigma 45 mm: gradients under 0.2, so psi is invertible). A registration recovers psi's inverse,
# which maps each fixed point to its moving partner.
center = fixed.TransformContinuousIndexToPhysicalPoint([(n - 1) / 2 for n in size])
T = sitk.Euler3DTransform()
T.SetCenter(center)
T.SetRotation(*np.deg2rad([4.0, -3.0, 6.0]))
T.SetTranslation([7.0, -5.0, 9.0])
# Points well inside the body: the bumps' centres and the landmarks are drawn from them.
candidates = np.argwhere(sitk.GetArrayFromImage(sitk.BinaryErode(fixed_body, [8, 8, 8])) > 0)
SIGMA = 45.0
BUMPS = [  # (centre, displacement), mm: three places of the anatomy, each pushed its own way
    (fixed.TransformIndexToPhysicalPoint([int(v) for v in candidates[i][::-1]]), amplitude)
    for i, amplitude in zip(
        np.random.default_rng(1).choice(len(candidates), 3, replace=False),
        [[9.0, -6.0, 4.5], [-4.5, 7.5, -7.5], [6.0, 6.0, 9.0]],
        strict=True,
    )
]


def bumps(points: np.ndarray) -> np.ndarray:
    """B at physical points [N, 3] (x, y, z), mm."""
    out = np.zeros_like(points)
    for centre, amplitude in BUMPS:
        weight = np.exp(-np.sum((points - np.array(centre)) ** 2, axis=1) / (2 * SIGMA**2))
        out += weight[:, None] * np.array(amplitude)
    return out


# psi as a displacement field on the moving grid: psi(y) - y at every voxel (numpy order z, y, x; vectors x, y, z).
index = np.stack(np.meshgrid(*[np.arange(n) for n in size], indexing="ij"), -1).reshape(-1, 3)  # x, y, z
points = np.array(grid.GetOrigin()) + index * 2.0
matrix, offset = np.array(T.GetMatrix()).reshape(3, 3), np.array(T.GetTranslation()) + np.array(center)
displaced = points + bumps(points)
field = (displaced - np.array(center)) @ matrix.T + offset - points  # T(y + B(y)) - y
field_image = sitk.GetImageFromArray(field.reshape(*size, 3).transpose(2, 1, 0, 3), isVector=True)
field_image.CopyInformation(grid)
psi = sitk.DisplacementFieldTransform(sitk.Cast(field_image, sitk.sitkVectorFloat64))
moving = on(mr, psi, 0.0)
moving_labels = sitk.Resample(fixed_labels, grid, psi, sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)

# Landmarks: points inside the body on the fixed image, and where they went on the moving one.
rng = np.random.default_rng(0)
fixed_points = [
    fixed.TransformIndexToPhysicalPoint([int(v) for v in p[::-1]])
    for p in candidates[rng.choice(len(candidates), 16, replace=False)]
]
# q = psi^-1(p): T(q + B(q)) = p, so q = T^-1(p) - B(q), a fixed point reached in a few steps since B's gradient is small.
inverse = T.GetInverse()
targets = np.array([inverse.TransformPoint(p) for p in fixed_points])
moving_points = targets.copy()
for _ in range(50):
    moving_points = targets - bumps(moving_points)
moving_points = [tuple(float(v) for v in point) for point in moving_points]


def write_fcsv(points: list, path: Path) -> None:
    rows = [
        "# Markups fiducial file version = 4.11",
        "# CoordinateSystem = LPS",
        "# columns = id,x,y,z,ow,ox,oy,oz,vis,sel,lock,label,desc,associatedNodeID",
    ]
    rows += [f"{i},{x:.3f},{y:.3f},{z:.3f},0,0,0,1,1,1,0,L{i},," for i, (x, y, z) in enumerate(points)]
    path.write_text("\n".join(rows) + "\n")


sitk.WriteImage(fixed, str(OUT / "fixed_ct.mha"), True)
sitk.WriteImage(moving, str(OUT / "moving_mr.mha"), True)
sitk.WriteImage(on(mr, identity, 0.0), str(OUT / "aligned_mr.mha"), True)  # the MRI as the dataset aligned it
sitk.WriteImage(fixed_body, str(OUT / "fixed_mask.mha"), True)
sitk.WriteImage(fixed_labels, str(OUT / "fixed_labels.mha"), True)
sitk.WriteImage(moving_labels, str(OUT / "moving_labels.mha"), True)
write_fcsv(fixed_points, OUT / "fixed_landmarks.fcsv")
write_fcsv(moving_points, OUT / "moving_landmarks.fcsv")
(OUT / "truth.json").write_text(
    json.dumps(
        {
            "rotation_deg": [4, -3, 6],
            "translation_mm": [7, -5, 9],
            "center": center,
            "bumps_mm": BUMPS,
            "sigma_mm": SIGMA,
        },
        indent=1,
    )
)
print(f"{OUT}/: fixed_ct.mha and moving_mr.mha ({' x '.join(map(str, size))} voxels at 2 mm), labels, landmarks, mask")
