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

"""Generate the real OME-Zarr regional-read figure used by the documentation."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
from konfai.utils.dataset import Dataset
from konfai.utils.ome_zarr import get_ome_zarr_info
from PIL import Image, ImageDraw, ImageEnhance, ImageFont, ImageOps

OUTPUT = Path(__file__).parents[1] / "source" / "_static" / "gallery" / "scale-omezarr.webp"
MOBILE_OUTPUT = Path(__file__).parents[1] / "source" / "_static" / "gallery" / "scale-omezarr-mobile.webp"
PAGE_WIDTH = 1530
PAGE_HEIGHT = 900
MOBILE_PAGE_WIDTH = 500
MOBILE_PAGE_HEIGHT = 2147
CARD_WIDTH = 470
CARD_HEIGHT = 535
INK = "#123f3e"
MUTED = "#607775"
ACCENT = "#1a9a91"
BACKGROUND = "#e8f0ef"
CARD = "#f8fbfa"


def fonts() -> tuple[ImageFont.FreeTypeFont | ImageFont.ImageFont, ...]:
    """Load the documentation font family with a portable fallback."""
    try:
        return (
            ImageFont.truetype("DejaVuSans-Bold.ttf", 36),
            ImageFont.truetype("DejaVuSans-Bold.ttf", 22),
            ImageFont.truetype("DejaVuSans.ttf", 16),
            ImageFont.truetype("DejaVuSansMono.ttf", 14),
        )
    except OSError:
        fallback = ImageFont.load_default()
        return fallback, fallback, fallback, fallback


def robust_gray(array: np.ndarray) -> Image.Image:
    """Render sparse microscopy intensities with a robust logarithmic window."""
    values = np.asarray(array, dtype=np.float32)
    positive = values[values > 0]
    if not positive.size:
        return Image.new("L", (values.shape[1], values.shape[0]))
    low, high = np.percentile(positive, (2.0, 99.7))
    scaled = np.clip((values - low) / max(float(high - low), 1.0), 0, 1)
    scaled = np.log1p(12 * scaled) / math.log(13)
    return Image.fromarray(np.asarray(scaled * 255, dtype=np.uint8), mode="L")


def read_plane(
    dataset: Dataset,
    group: str,
    case: str,
    z: int,
    y: slice = slice(None),
    x: slice = slice(None),
) -> np.ndarray:
    """Read one bounded CZYX window through KonfAI's public dataset API."""
    data, _ = dataset.read_data_slice(
        group,
        case,
        (slice(None), slice(z, z + 1), y, x),
    )
    return np.asarray(data[0, 0])


def fit(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    """Fit a high-contrast microscopy panel without changing its aspect ratio."""
    fitted = ImageOps.contain(image.convert("RGB"), size, method=Image.Resampling.LANCZOS)
    fitted = ImageEnhance.Sharpness(fitted).enhance(1.2)
    result = Image.new("RGB", size, "#061110")
    result.paste(fitted, ((size[0] - fitted.width) // 2, (size[1] - fitted.height) // 2))
    return result


def card(
    canvas: Image.Image,
    x: int,
    y: int,
    title: str,
    subtitle: str,
    image: Image.Image,
    footer: str,
) -> None:
    """Render one gallery card."""
    _, heading_font, body_font, mono_font = fonts()
    draw = ImageDraw.Draw(canvas)
    draw.rounded_rectangle((x, y, x + CARD_WIDTH, y + CARD_HEIGHT), radius=22, fill=CARD)
    draw.rounded_rectangle((x + 24, y + 20, x + 84, y + 26), radius=3, fill=ACCENT)
    draw.text((x + 24, y + 38), title, font=heading_font, fill=INK)
    draw.text((x + 24, y + 72), subtitle, font=body_font, fill=MUTED)
    canvas.paste(fit(image, (422, 388)), (x + 24, y + 105))
    draw.rounded_rectangle((x + 24, y + 502, x + 446, y + 528), radius=10, fill="#deebe9")
    draw.text((x + 35, y + 506), footer.upper(), font=mono_font, fill=INK)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="Dataset root containing case directories.")
    parser.add_argument("--case", default="822175")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--mobile-output", type=Path, default=MOBILE_OUTPUT)
    args = parser.parse_args()

    fine = Dataset(args.root, "omezarr")
    coarse = Dataset(args.root, "omezarr@1")
    shape0, _ = fine.get_infos("Volume", args.case)
    shape1, _ = coarse.get_infos("Volume", args.case)
    store = args.root / args.case / "Volume.ome.zarr"
    info0 = get_ome_zarr_info(store, level=0)

    z0 = shape0[1] // 2
    z1 = min(shape1[1] - 1, round((z0 + 0.5) * shape1[1] / shape0[1] - 0.5))
    patch_size = 512
    y0 = min(max(0, shape0[2] // 2 - patch_size // 2), shape0[2] - patch_size)
    x0 = min(max(0, shape0[3] // 2 - patch_size // 2), shape0[3] - patch_size)

    overview = read_plane(coarse, "Volume", args.case, z1)
    patch = read_plane(
        fine,
        "Volume",
        args.case,
        z0,
        slice(y0, y0 + patch_size),
        slice(x0, x0 + patch_size),
    )
    mask = read_plane(
        fine,
        "Mask",
        args.case,
        z0,
        slice(y0, y0 + patch_size),
        slice(x0, x0 + patch_size),
    )

    overview_image = robust_gray(overview).convert("RGB")
    overview_draw = ImageDraw.Draw(overview_image)
    scale_y = shape0[2] / shape1[2]
    scale_x = shape0[3] / shape1[3]
    overview_box = (
        round(x0 / scale_x),
        round(y0 / scale_y),
        round((x0 + patch_size) / scale_x),
        round((y0 + patch_size) / scale_y),
    )
    overview_draw.rounded_rectangle(overview_box, radius=4, outline="#2ce0d1", width=4)

    patch_image = robust_gray(patch).convert("RGB")
    patch_draw = ImageDraw.Draw(patch_image)
    chunk_z, chunk_y, chunk_x = (int(axis_chunks[0]) for axis_chunks in info0["chunks"])
    for boundary in range(((x0 // chunk_x) + 1) * chunk_x, x0 + patch_size, chunk_x):
        patch_draw.line((boundary - x0, 0, boundary - x0, patch_size), fill="#2ce0d1", width=2)
    for boundary in range(((y0 // chunk_y) + 1) * chunk_y, y0 + patch_size, chunk_y):
        patch_draw.line((0, boundary - y0, patch_size, boundary - y0), fill="#2ce0d1", width=2)

    overlay = np.asarray(robust_gray(patch).convert("RGB"), dtype=np.float32)
    selected = mask > 0
    tint = np.zeros_like(overlay)
    tint[..., 0], tint[..., 1], tint[..., 2] = 24, 210, 192
    overlay[selected] = 0.6 * overlay[selected] + 0.4 * tint[selected]
    overlay_image = Image.fromarray(np.asarray(np.clip(overlay, 0, 255), dtype=np.uint8), mode="RGB")

    canvas = Image.new("RGB", (PAGE_WIDTH, PAGE_HEIGHT), BACKGROUND)
    draw = ImageDraw.Draw(canvas)
    title_font, heading_font, body_font, mono_font = fonts()
    draw.text((42, 30), "Scale from storage, not from RAM", font=title_font, fill=INK)
    draw.text(
        (44, 80),
        "A real ExaSPIM pyramid: metadata first, then only the spatial region required by the patch.",
        font=body_font,
        fill=MUTED,
    )
    badges = [
        f"{shape0[1]} x {shape0[2]} x {shape0[3]}",
        f"{np.prod(shape0[1:]) / 1e9:.2f} billion voxels",
        f"chunks {chunk_z} x {chunk_y} x {chunk_x}",
        f"{info0['n_levels']} pyramid levels",
    ]
    badge_x = 44
    for label in badges:
        bounds = draw.textbbox((0, 0), label, font=mono_font)
        width = bounds[2] - bounds[0] + 28
        draw.rounded_rectangle((badge_x, 119, badge_x + width, 153), radius=14, fill="#d6e8e5")
        draw.text((badge_x + 14, 126), label, font=mono_font, fill=INK)
        badge_x += width + 12

    card(
        canvas,
        30,
        176,
        "1. Multiscale overview",
        f"Level 1 · one slice z={z1}/{shape1[1] - 1}",
        overview_image,
        f"read 1 x {shape1[2]} x {shape1[3]} voxels",
    )
    card(
        canvas,
        530,
        176,
        "2. Native source region",
        f"Level 0 · z={z0}, 512² requested",
        patch_image,
        "actual chunk boundaries in cyan",
    )
    card(
        canvas,
        1030,
        176,
        "3. Matching mask",
        "Same geometry · same source window",
        overlay_image,
        "volume + mask ready for preprocessing",
    )

    raw_volume_gib = int(np.prod(shape0[1:])) * np.dtype(info0["dtype"]).itemsize / 1024**3
    draw.rounded_rectangle((30, 738, PAGE_WIDTH - 30, 868), radius=22, fill=INK)
    columns = [
        (58, "FULL SOURCE", f"{raw_volume_gib:.2f} GiB raw"),
        (510, "REQUESTED WINDOW", f"{patch.nbytes / 1024**2:.2f} MiB raw"),
        (970, "EXECUTION", "transform → batch → GPU"),
    ]
    for x, label, value in columns:
        draw.text((x, 760), label, font=mono_font, fill="#9ec5c0")
        draw.text((x, 790), value, font=title_font, fill="white")
    draw.text((442, 790), "→", font=title_font, fill="#43d4c7")
    draw.text((900, 790), "→", font=title_font, fill="#43d4c7")

    mobile_canvas = Image.new("RGB", (MOBILE_PAGE_WIDTH, MOBILE_PAGE_HEIGHT), BACKGROUND)
    mobile_draw = ImageDraw.Draw(mobile_canvas)
    mobile_draw.multiline_text(
        (26, 22),
        "Scale from storage,\nnot from RAM",
        font=title_font,
        fill=INK,
        spacing=0,
    )
    mobile_draw.multiline_text(
        (27, 111),
        "A real ExaSPIM pyramid: metadata first.\nThen only the region required by the patch.",
        font=body_font,
        fill=MUTED,
        spacing=3,
    )
    badge_x = 27
    badge_y = 165
    for label in badges:
        bounds = mobile_draw.textbbox((0, 0), label, font=mono_font)
        width = bounds[2] - bounds[0] + 28
        if badge_x + width > MOBILE_PAGE_WIDTH - 27:
            badge_x = 27
            badge_y += 44
        mobile_draw.rounded_rectangle((badge_x, badge_y, badge_x + width, badge_y + 34), radius=14, fill="#d6e8e5")
        mobile_draw.text((badge_x + 14, badge_y + 7), label, font=mono_font, fill=INK)
        badge_x += width + 12

    card(
        mobile_canvas,
        15,
        264,
        "1. Multiscale overview",
        f"Level 1 · one slice z={z1}/{shape1[1] - 1}",
        overview_image,
        f"read 1 x {shape1[2]} x {shape1[3]} voxels",
    )
    card(
        mobile_canvas,
        15,
        817,
        "2. Native source region",
        f"Level 0 · z={z0}, 512² requested",
        patch_image,
        "actual chunk boundaries in cyan",
    )
    card(
        mobile_canvas,
        15,
        1370,
        "3. Matching mask",
        "Same geometry · same source window",
        overlay_image,
        "volume + mask ready for preprocessing",
    )

    mobile_draw.rounded_rectangle((15, 1923, MOBILE_PAGE_WIDTH - 15, 2124), radius=22, fill=INK)
    mobile_rows = [
        (1940, "FULL SOURCE", f"{raw_volume_gib:.2f} GiB raw"),
        (1999, "REQUESTED WINDOW", f"{patch.nbytes / 1024**2:.2f} MiB raw"),
        (2058, "EXECUTION", "transform → batch → GPU"),
    ]
    for y, label, value in mobile_rows:
        mobile_draw.text((38, y), label, font=mono_font, fill="#9ec5c0")
        mobile_draw.text((38, y + 17), value, font=heading_font, fill="white")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.output, quality=92, method=6)
    print(f"Wrote {args.output}")
    args.mobile_output.parent.mkdir(parents=True, exist_ok=True)
    mobile_canvas.save(args.mobile_output, quality=92, method=6)
    print(f"Wrote {args.mobile_output}")


if __name__ == "__main__":
    main()
