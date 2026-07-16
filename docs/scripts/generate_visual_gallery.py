# Copyright (c) 2025 Valentin Boussot
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

"""Generate the deterministic transform and augmentation gallery used by the docs."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from konfai.data.augmentation import Brightness, Contrast, CutOUT, Flip, Noise, Rotate
from konfai.data.transform import (
    Clip,
    Crop,
    Gradient,
    Normalize,
    Padding,
    Permute,
    ResampleToResolution,
    ResampleToShape,
    Standardize,
)
from konfai.utils.dataset import Attribute
from PIL import Image, ImageOps

OUTPUT = Path(__file__).parents[1] / "source" / "_static" / "gallery"
TRANSFORMS_DIR = OUTPUT / "transforms"
AUGMENTATIONS_DIR = OUTPUT / "augmentations"
IMAGE_SIZE = 512
VIEWPORT_BACKGROUND = "#121819"


def make_phantom(size: int = IMAGE_SIZE) -> torch.Tensor:
    """Return a CT-like 2-D phantom with anatomy-shaped regions and acquisition artifacts."""
    y, x = np.mgrid[-1 : 1 : complex(size), -1 : 1 : complex(size)]
    body = ((x / 0.80) ** 2 + (y / 0.94) ** 2) <= 1
    left_lung = (((x + 0.29) / 0.24) ** 2 + ((y + 0.10) / 0.40) ** 2) <= 1
    right_lung = (((x - 0.29) / 0.24) ** 2 + ((y + 0.10) / 0.40) ** 2) <= 1
    heart = (((x + 0.04) / 0.24) ** 2 + ((y - 0.12) / 0.31) ** 2) <= 1
    vertebra = ((x / 0.10) ** 2 + ((y - 0.42) / 0.12) ** 2) <= 1
    lesion = (((x - 0.27) / 0.07) ** 2 + ((y + 0.03) / 0.09) ** 2) <= 1

    image = np.full((size, size), -1.15, dtype=np.float32)
    image[body] = 0.05
    image[left_lung | right_lung] = -0.72
    image[heart] = 0.38
    image[vertebra] = 1.20
    image[lesion] = 0.22
    image += 0.035 * np.sin(28 * x) * body
    image += 0.018 * np.cos(34 * y) * body
    return torch.from_numpy(image).unsqueeze(0)


def load_medical_slice(path: Path, slice_index: int | None) -> torch.Tensor:
    """Load one anonymous 2-D slice from a public documentation volume."""
    import SimpleITK as sitk

    reader = sitk.ImageFileReader()
    reader.SetFileName(str(path))
    reader.ReadImageInformation()
    size_x, size_y, size_z = reader.GetSize()
    index = size_z // 2 if slice_index is None else slice_index
    if not 0 <= index < size_z:
        raise ValueError(f"slice_index must be in [0, {size_z - 1}], got {index}")
    reader.SetExtractIndex([0, 0, index])
    reader.SetExtractSize([size_x, size_y, 1])
    plane = sitk.GetArrayFromImage(reader.Execute()).squeeze().astype(np.float32)
    return torch.from_numpy(plane).unsqueeze(0)


def apply_augmentation(augmentation, tensor: torch.Tensor, seed: int) -> torch.Tensor:
    """Run an augmentation through its public per-case lifecycle."""
    torch.manual_seed(seed)
    augmentation.load(1.0)
    augmentation.state_init(0, [list(tensor.shape[1:])], [Attribute()])
    return augmentation("PHANTOM", 0, [tensor.clone()])[0]


def display_array(tensor: torch.Tensor, low: float = -1.2, high: float = 1.25) -> np.ndarray:
    array = tensor.detach().cpu().squeeze().float().numpy()
    array = np.clip((array - low) / (high - low), 0, 1)
    return np.asarray(array * 255, dtype=np.uint8)


def grayscale_medical(array: np.ndarray) -> np.ndarray:
    """Render a classic medical grayscale without pure black or clipped white."""
    softened = 18 + np.asarray(array, dtype=np.float32) / 255 * 219
    return np.asarray(np.clip(softened, 0, 255), dtype=np.uint8)


def medical_image(tensor: torch.Tensor, window: tuple[float, float] = (-1.2, 1.25)) -> Image.Image:
    """Render medical pixels only; the documentation owns labels and statistics."""
    image = Image.fromarray(grayscale_medical(display_array(tensor, *window)), mode="L").convert("RGB")
    return ImageOps.pad(
        image,
        (IMAGE_SIZE, IMAGE_SIZE),
        method=Image.Resampling.BILINEAR,
        color=VIEWPORT_BACKGROUND,
    )


def save_images(images: list[tuple[str, Image.Image]], output_dir: Path) -> None:
    """Write image-only proof assets for responsive HTML composition."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for slug, image in images:
        image.save(output_dir / f"{slug}.png", optimize=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, help="Optional MHA/NIfTI volume used for the image galleries.")
    parser.add_argument("--slice-index", type=int, help="Axial slice index; defaults to the middle slice.")
    parser.add_argument("--transforms-dir", type=Path, default=TRANSFORMS_DIR)
    parser.add_argument("--augmentations-dir", type=Path, default=AUGMENTATIONS_DIR)
    args = parser.parse_args()

    source = load_medical_slice(args.input, args.slice_index) if args.input else make_phantom()
    is_medical_input = args.input is not None

    clip_min, clip_max = (-1000.0, 100.0) if is_medical_input else (-0.75, 0.15)
    source_window = (-1000.0, 1000.0) if is_medical_input else (-1.2, 1.25)
    clipped = Clip(min_value=clip_min, max_value=clip_max)("IMAGE", source.clone(), Attribute())
    normalized = Normalize(min_value=-1, max_value=1, inverse=True)("IMAGE", clipped.clone(), Attribute())
    standardized = Standardize(inverse=True)("PHANTOM", source.clone(), Attribute())

    source_shape = list(normalized.shape[1:])
    target_shape = [220, 220]
    shape_attribute = Attribute()
    shape_attribute["Spacing"] = np.asarray([0.8, 0.8])
    resampled_shape = ResampleToShape(shape=target_shape)("IMAGE", normalized.clone(), shape_attribute)

    resolution_attribute = Attribute()
    resolution_attribute["Spacing"] = np.asarray([0.8, 0.8])
    target_spacing = [1.25, 0.55]
    resampled_resolution = ResampleToResolution(spacing=target_spacing, inverse=True)(
        "IMAGE", normalized.clone(), resolution_attribute
    )

    padded = Padding(padding=[28, 28, 18, 18], mode="constant:-1")("IMAGE", normalized.clone(), Attribute())
    crop_margin_y = max(12, source_shape[0] // 10)
    crop_margin_x = max(12, source_shape[1] // 10)
    crop_attribute = Attribute()
    crop_attribute["Origin"] = np.asarray([0.0, 0.0])
    crop_attribute["Spacing"] = np.asarray([1.0, 1.0])
    crop_attribute["Direction"] = np.eye(2, dtype=np.float64).flatten()
    crop_attribute["box"] = np.asarray([[crop_margin_y, crop_margin_y], [crop_margin_x, crop_margin_x]], dtype=np.int64)
    cropped = Crop()("IMAGE", normalized.clone(), crop_attribute)
    permuted = Permute(dims="1|0")("IMAGE", normalized.clone(), Attribute())
    gradient = Gradient()("IMAGE", normalized.clone(), Attribute())

    transform_images = [
        ("source", medical_image(source, source_window)),
        ("clip", medical_image(clipped, source_window)),
        ("normalize", medical_image(normalized, (-1, 1))),
        ("standardize", medical_image(standardized, (-2.5, 2.5))),
        ("resample-shape", medical_image(resampled_shape, (-1, 1))),
        ("resample-spacing", medical_image(resampled_resolution, (-1, 1))),
        ("padding", medical_image(padded, (-1, 1))),
        ("crop", medical_image(cropped, (-1, 1))),
        ("permute", medical_image(permuted, (-1, 1))),
        ("gradient", medical_image(gradient, (0, 1))),
    ]
    save_images(transform_images, args.transforms_dir)

    augmentation_source = normalized
    flipped = apply_augmentation(Flip(f_prob=[0.0, 1.0]), augmentation_source, seed=7)
    rotated = apply_augmentation(Rotate(a_min=18, a_max=18), augmentation_source, seed=7)
    bright = apply_augmentation(Brightness(b_std=0.35), augmentation_source, seed=5)
    contrast = apply_augmentation(Contrast(c_std=0.75), augmentation_source, seed=3)
    noise = Noise(n_std=0.65)
    torch.manual_seed(11)
    noise.load(0.55)
    noise.state_init(0, [list(augmentation_source.shape[1:])], [Attribute()])
    noisy = noise("IMAGE", 0, [augmentation_source.clone()])[0]
    cutout = apply_augmentation(CutOUT(c_prob=1, cutout_size=0.34, value=-1), augmentation_source, seed=13)
    save_images(
        [
            ("source", medical_image(augmentation_source, (-1, 1))),
            ("flip", medical_image(flipped, (-1, 1))),
            ("rotate", medical_image(rotated, (-1, 1))),
            ("brightness", medical_image(bright, (-1, 1))),
            ("contrast", medical_image(contrast, (-1, 1))),
            ("noise", medical_image(noisy, (-1, 1))),
            ("cutout", medical_image(cutout, (-1, 1))),
        ],
        args.augmentations_dir,
    )


if __name__ == "__main__":
    main()
