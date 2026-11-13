# SPDX-License-Identifier: Apache-2.0
"""Rendering: what the model is shown, and the record of where each pixel sits in the world.

A view is one plane of the case (or a montage of planes), composed from the case's volumes in a mode the task
defines, displayed in radiological convention (the patient's left on the right for axial and coronal planes, the
anterior side on the left of a sagittal plane, superior at the top), at isotropic pixels, with a header, direction
letters and a scale bar. The :class:`~konfai_scout.report.View` beside the PNG carries the affine that takes a pixel
back to millimetres, so anything the model points at is a physical location.
"""

from __future__ import annotations

import hashlib
import io
from pathlib import Path

import numpy as np
from konfai.data.geometry import Grid, WorldBox
from konfai.utils.ITK import field_jacobian_determinant
from PIL import Image, ImageDraw, ImageFont

from .case import InspectionCase, Volume
from .config import ScoutConfig
from .derived import Derived
from .planes import (
    PlaneFrame,
    axis_world_vector,
    box_center,
    box_to_dict,
    dominant_world_axis,
    index_along,
    plane_frame,
    same_grid,
    window_of,
)
from .report import View

PALETTE = [
    (230, 25, 75),
    (60, 180, 75),
    (255, 225, 25),
    (0, 130, 200),
    (245, 130, 48),
    (145, 30, 180),
    (70, 240, 240),
    (240, 50, 230),
    (210, 245, 60),
    (250, 190, 212),
    (0, 128, 128),
    (220, 190, 255),
    (170, 110, 40),
    (255, 250, 200),
    (128, 0, 0),
    (170, 255, 195),
    (128, 128, 0),
    (255, 215, 180),
    (0, 0, 128),
    (128, 128, 128),
]
WINDOW_PRESETS: dict[str, tuple[float, float]] = {
    "ct_soft": (-160.0, 240.0),
    "ct_lung": (-1200.0, 200.0),
    "ct_bone": (-500.0, 1300.0),
    "ct_brain": (0.0, 80.0),
    "ct_full": (-1000.0, 1000.0),
}
MODES: dict[str, tuple[str, ...]] = {
    "segmentation": ("image", "prediction", "truth", "both", "disagreement"),
    "registration": (
        "fixed",
        "moving",
        "warped",
        "before",
        "checkerboard",
        "overlay",
        "difference",
        "field_magnitude",
        "field_arrows",
        "grid",
        "jacobian",
        "labels",
        "landmarks",
    ),
    "synthesis": ("synthetic", "target", "source", "difference", "side_by_side", "checkerboard"),
}
DEFAULT_MODE = {"segmentation": "prediction", "registration": "checkerboard", "synthesis": "synthetic"}
_VIRIDIS = np.array(
    [
        (0.267, 0.005, 0.329),
        (0.283, 0.141, 0.458),
        (0.254, 0.265, 0.530),
        (0.207, 0.372, 0.553),
        (0.164, 0.471, 0.558),
        (0.128, 0.567, 0.551),
        (0.135, 0.659, 0.518),
        (0.267, 0.749, 0.441),
        (0.478, 0.821, 0.318),
        (0.741, 0.873, 0.150),
        (0.993, 0.906, 0.144),
    ]
)
_HOT = np.array([(0.0, 0.0, 0.0), (0.6, 0.0, 0.0), (1.0, 0.3, 0.0), (1.0, 0.8, 0.0), (1.0, 1.0, 1.0)])
_COOLWARM = np.array(
    [(0.23, 0.30, 0.75), (0.60, 0.72, 0.95), (0.86, 0.86, 0.86), (0.95, 0.65, 0.55), (0.71, 0.016, 0.15)]
)
_COLORMAPS = {"viridis": _VIRIDIS, "hot": _HOT, "coolwarm": _COOLWARM}


class ViewError(ValueError):
    """A view that cannot be rendered as asked: the message is for the model."""


def label_color(label: int) -> tuple[float, float, float]:
    r, g, b = PALETTE[(int(label) - 1) % len(PALETTE)]
    return (r / 255, g / 255, b / 255)


def to_unit(plane: np.ndarray, window: tuple[float, float]) -> np.ndarray:
    low, high = window
    if high <= low:
        high = low + 1.0
    return np.clip((np.asarray(plane, dtype=np.float32) - low) / (high - low), 0.0, 1.0)


#: The roles a registration's warped labels come from: given warped, or moving ones the field carries.
WARPED_LABEL_ROLES = {"warped_labels", "moving_labels"}
#: The band above every view that holds its header.
HEADER_PX = 18
#: The registration modes drawn over one image, and which.
GRAY_REGISTRATION = {
    **dict.fromkeys(("fixed", "field_arrows", "grid", "labels", "landmarks", "field_magnitude", "jacobian"), "fixed"),
    "moving": "moving_on_fixed",
    "warped": "warped",
}


def heat(values: np.ndarray, name: str) -> tuple[np.ndarray, float]:
    """Values coloured from 0 to their 99th percentile, and that percentile."""
    scale = float(max(np.percentile(values, 99), 1e-6))
    return colormap(values / scale, name), round(scale, 3)


def colormap(unit: np.ndarray, name: str) -> np.ndarray:
    anchors = _COLORMAPS[name]
    positions = np.linspace(0.0, 1.0, len(anchors))
    unit = np.clip(np.asarray(unit, dtype=np.float32), 0.0, 1.0)
    return np.stack([np.interp(unit, positions, anchors[:, channel]) for channel in range(3)], axis=-1)


def contours(mask: np.ndarray) -> np.ndarray:
    """The pixels of a mask that touch the outside of it, 4-connected; the image's edge is not an outside."""
    from scipy import ndimage

    mask = np.asarray(mask, dtype=bool)
    return mask & ~ndimage.binary_erosion(mask, border_value=1)


def fill_labels(rgb: np.ndarray, labels: np.ndarray, alpha: float, only: int | None = None) -> np.ndarray:
    out = rgb.copy()
    for label in (int(v) for v in np.unique(labels) if v != 0):
        if only is not None and label != only:
            continue
        mask = labels == label
        out[mask] = (1 - alpha) * out[mask] + alpha * np.asarray(label_color(label))
    return out


def draw_contours(
    rgb: np.ndarray, labels: np.ndarray, only: int | None = None, color: tuple[float, float, float] | None = None
) -> np.ndarray:
    out = rgb.copy()
    for label in (int(v) for v in np.unique(labels) if v != 0):
        if only is not None and label != only:
            continue
        out[contours(labels == label)] = color if color is not None else label_color(label)
    return out


def checkerboard(a: np.ndarray, b: np.ndarray, tiles: int = 8) -> np.ndarray:
    rows, cols = np.indices(a.shape)
    size = max(1, max(a.shape) // tiles)
    board = ((rows // size) + (cols // size)) % 2 == 0
    return np.where(board, a, b)


def _grid_text(grid: Grid) -> str:
    return (
        f"{list(grid.size_zyx)} voxels in array order, spacing {[round(float(s), 4) for s in grid.spacing_xyz]} mm, "
        f"origin {[round(float(o), 3) for o in grid.origin_xyz]} mm"
    )


class Renderer:
    """Renders the case's views and keeps the record of each one."""

    def __init__(self, case: InspectionCase, work: Path, config: ScoutConfig, evidence: Path) -> None:
        self.case = case
        self.config = config
        self.evidence = Path(evidence)
        self.evidence.mkdir(parents=True, exist_ok=True)
        self.derived = Derived(case, Path(work)) if case.task == "registration" else None
        self._windows: dict[str, tuple[float, float]] = {}
        self._count = 0
        self.views: list[View] = []
        self.font = ImageFont.load_default(size=11)

    # ------------------------------------------------------------------ volumes by role

    @property
    def grid(self) -> Grid:
        return self.case.grid

    def modes(self) -> list[str]:
        available = list(MODES[self.case.task])
        if self.case.task == "segmentation" and "truth" not in self.case.inputs:
            available = [m for m in available if m not in ("truth", "both", "disagreement")]
        if self.case.task == "registration":
            if not ({"fixed_labels"} | WARPED_LABEL_ROLES) & self.case.inputs.keys():
                available.remove("labels")
            if "fixed_landmarks" not in self.case.inputs:
                available.remove("landmarks")
        if self.case.task == "synthesis" and "target" not in self.case.inputs:
            available = [m for m in available if m not in ("target", "difference", "side_by_side", "checkerboard")]
        return available

    def volume(self, role: str) -> Volume:
        """A role's volume, refused unless it samples the reference grid: Scout views (and measures) the roles of
        one case on one grid, a given field or warped labels included."""
        volume = self._volume(role)
        if not same_grid(volume.grid, self.grid):
            raise ViewError(
                f"{role} is on another grid than {self.case.image_role} ({_grid_text(volume.grid)} vs "
                f"{_grid_text(self.grid)}); Scout views the roles of one case on one grid"
            )
        return volume

    def _volume(self, role: str) -> Volume:
        if self.case.task == "registration":
            assert self.derived is not None
            if role == "warped":
                return self.derived.warped()
            if role == "moving_on_fixed":
                return self.derived.moving_on_fixed()
            if role == "field":
                return self.derived.field()
            if role == "warped_labels":
                warped = self.derived.warped_labels()
                if warped is None:
                    raise ViewError("this case has no moving labels")
                return warped
        return self.case.volume(role)

    def window(self, role: str) -> tuple[float, float]:
        if role not in self._windows:  # KonfAI's bounded passes: the volume is never held for its window
            volume = self.volume(role)
            # the 99.9th: a hyperintense lesion (DWI, contrast) fills well under 1 % of a volume and must not saturate
            low, high = volume.quantile(0.01), volume.quantile(0.999)
            self._windows[role] = (low, high if high > low else low + 1.0)
        return self._windows[role]

    def resolve_window(self, window: object, role: str) -> tuple[float, float]:
        if window is None or window == "auto":
            return self.window(role)
        if isinstance(window, str):
            if window not in WINDOW_PRESETS:
                raise ViewError(
                    f"unknown window preset {window!r}; presets: {', '.join(WINDOW_PRESETS)}, or [low, high]"
                )
            # ponytail: a CT stored as HU + 1024 is told by its 1st percentile (no air below 0); a scan of HU with
            # no value below 0 in 99% of it would be shifted too. Read the stored offset from the header if it matters.
            offset = 1024.0 if self.window(role)[0] >= 0 else 0.0
            return WINDOW_PRESETS[window][0] + offset, WINDOW_PRESETS[window][1] + offset
        values = [float(v) for v in np.asarray(window, dtype=float).ravel()]
        if len(values) != 2:
            raise ViewError("a window is a preset name or [low, high]")
        return values[0], values[1]

    # ------------------------------------------------------------------ public rendering

    def render(
        self,
        plane: str,
        mode: str | None = None,
        index: int | None = None,
        position_mm: object = None,
        window: object = None,
        zoom: WorldBox | None = None,
        label: int | None = None,
        sent: bool = True,
    ) -> tuple[bytes, View]:
        """One slice: by index, by the world position it passes through, or the centre of the zoom box."""
        mode = mode or DEFAULT_MODE[self.case.task]
        if mode not in self.modes():
            raise ViewError(f"mode {mode!r} is not available for this case; modes: {', '.join(self.modes())}")
        frame = plane_frame(self.grid, plane)
        axis = frame.array_axis
        if index is None:
            if position_mm is not None:
                index = index_along(self.grid, axis, position_mm)
            elif zoom is not None:
                index = index_along(self.grid, axis, box_center(zoom))
            else:
                index = self.grid.size_zyx[axis] // 2
        index = int(index)
        if not 0 <= index < self.grid.size_zyx[axis]:
            raise ViewError(f"slice {index} is outside the {plane} range 0..{self.grid.size_zyx[axis] - 1}")
        crop = self._crop(zoom, frame) if zoom is not None else None
        rgb, extra, resolved_window = self._compose(frame, index, mode, window, label, crop)
        self._count += 1
        view_id = f"v{self._count:03d}"  # on the image before anything else: a model cites what it reads there
        header = self._header_text(frame, index, mode, resolved_window, extra, crop).format(id=view_id)
        image, mapping = self._display(rgb, frame, index, crop, header)
        view = self._record(view_id, frame, index, mode, resolved_window, image, mapping, extra, crop, sent)
        return self._png(image, view), view

    def montage(
        self,
        plane: str,
        mode: str | None = None,
        count: int | None = None,
        window: object = None,
        label: int | None = None,
        sent: bool = True,
    ) -> tuple[bytes, View]:
        """``count`` equally spaced slices of a plane, tiled: the coverage of a volume at a glance."""
        mode = mode or DEFAULT_MODE[self.case.task]
        if mode not in self.modes():
            raise ViewError(f"mode {mode!r} is not available for this case; modes: {', '.join(self.modes())}")
        frame = plane_frame(self.grid, plane)
        depth = self.grid.size_zyx[frame.array_axis]
        count = max(1, min(int(count or self.config.coverage_slices), depth, 16))
        indices = (
            [int(i) for i in np.linspace(0, depth - 1, count + 2)[1:-1]]
            if depth > count + 1
            else list(range(depth))[:count]
        )
        indices = sorted(set(indices))
        tiles = []
        resolved_window: list[float] | None = None
        per_tile = max(160, self.config.max_px // max(1, int(np.ceil(np.sqrt(len(indices))))))
        for index in indices:
            rgb, extra, resolved = self._compose(frame, index, mode, window, label, None)
            resolved_window = resolved
            image, mapping = self._display(rgb, frame, index, None, None, max_px=per_tile)
            position = self._position(frame, index)
            caption = f"{self._axis_letter(frame)}={position[self._world_axis(frame)]:.1f}mm #{index}"
            self._caption(image, caption)
            tiles.append((image, mapping, index, position))
        columns = int(np.ceil(np.sqrt(len(tiles))))
        rows = int(np.ceil(len(tiles) / columns))
        width = max(t[0].width for t in tiles)
        height = max(t[0].height for t in tiles)
        gap = 3
        header_h = HEADER_PX
        sheet = Image.new(
            "RGB", (columns * width + (columns - 1) * gap, header_h + rows * height + (rows - 1) * gap), (20, 20, 20)
        )
        placed = []
        for n, (image, mapping, index, position) in enumerate(tiles):
            x = (n % columns) * (width + gap)
            y = header_h + (n // columns) * (height + gap)
            sheet.paste(image, (x, y))
            placed.append(
                {
                    "index": index,
                    "position_mm": [round(float(v), 3) for v in position],
                    "offset_px": [x, y],
                    "px_to_mm": mapping,
                }
            )
        draw = ImageDraw.Draw(sheet)
        extra = {"tiles": placed, "indices": indices}
        title = (
            f"{{id}} {self._plane_name(frame)} montage of {len(indices)} slices | {mode} | "
            f"window {self._window_text(resolved_window)}"
        )
        self._count += 1
        view_id = f"v{self._count:03d}"
        draw.text((4, 3), title.format(id=view_id), fill=(255, 255, 255), font=self.font)
        box = self.grid.world_box()
        view = View(
            id=view_id,
            kind="montage",
            plane=plane,
            array_axis=frame.array_axis,
            index=None,
            position_mm=None,
            box_mm=box_to_dict(box),
            mode=mode,
            window=resolved_window,
            size_px=[sheet.width, sheet.height],
            px_to_mm=None,
            path="",
            sha256="",
            sent=sent,
            extra=extra,
        )
        self.views.append(view)
        return self._png(sheet, view), view

    # ------------------------------------------------------------------ composition

    def _crop(self, zoom: WorldBox, frame: PlaneFrame) -> tuple[slice, slice]:
        window = window_of(self.grid, zoom)
        if window is None:
            raise ViewError("the zoom box lies outside the volume")
        rows, cols = window[frame.rows_axis], window[frame.cols_axis]
        return self._widen(rows, self.grid.size_zyx[frame.rows_axis]), self._widen(
            cols, self.grid.size_zyx[frame.cols_axis]
        )

    @staticmethod
    def _widen(part: slice, extent: int, minimum: int = 8) -> slice:
        start, stop = part.start, part.stop
        while stop - start < minimum and (start > 0 or stop < extent):
            start, stop = max(0, start - 1), min(extent, stop + 1)
        return slice(start, stop)

    def _plane(self, role: str, frame: PlaneFrame, index: int, crop: tuple[slice, slice] | None) -> np.ndarray:
        plane = self.volume(role).plane(frame.array_axis, index)
        if crop is not None:
            plane = plane[..., crop[0], crop[1]]
        return plane

    def _compose(
        self,
        frame: PlaneFrame,
        index: int,
        mode: str,
        window: object,
        label: int | None,
        crop: tuple[slice, slice] | None,
    ) -> tuple[np.ndarray, dict[str, object], list[float] | None]:
        task = self.case.task
        extra: dict[str, object] = {}
        resolved: tuple[float, float] | None = None

        def unit(role: str, window_role: str | None = None) -> np.ndarray:
            nonlocal resolved
            resolved = self.resolve_window(window, window_role or role)
            return to_unit(self._plane(role, frame, index, crop), resolved)

        def gray(role: str, window_role: str | None = None) -> np.ndarray:
            return np.repeat(unit(role, window_role)[..., None], 3, axis=-1)

        if task == "segmentation":
            rgb = gray("image")
            if mode == "prediction":
                pred = self._plane("prediction", frame, index, crop)
                rgb = draw_contours(fill_labels(rgb, pred, 0.35, label), pred, label)
                extra["labels"] = [int(v) for v in np.unique(pred) if v != 0]
            elif mode == "truth":
                truth = self._plane("truth", frame, index, crop)
                rgb = draw_contours(rgb, truth, label)
                extra["labels"] = [int(v) for v in np.unique(truth) if v != 0]
            elif mode == "both":
                pred = self._plane("prediction", frame, index, crop)
                truth = self._plane("truth", frame, index, crop)
                rgb = draw_contours(fill_labels(rgb, pred, 0.35, label), truth, label, color=(1.0, 1.0, 1.0))
                extra["legend"] = "prediction filled in the label's colour, truth as white contours"
            elif mode == "disagreement":
                pred = self._plane("prediction", frame, index, crop)
                truth = self._plane("truth", frame, index, crop)
                if label is not None:
                    pred, truth = pred == label, truth == label
                else:
                    pred, truth = pred != 0, truth != 0
                rgb = rgb.copy()
                for mask, color in (
                    (pred & ~truth, (1.0, 0.1, 0.1)),
                    (~pred & truth, (0.1, 0.3, 1.0)),
                    (pred & truth, (0.1, 0.9, 0.1)),
                ):
                    rgb[mask] = 0.5 * rgb[mask] + 0.5 * np.asarray(color)
                extra["legend"] = (
                    "red: prediction only (false positive), blue: truth only (false negative), green: both"
                )
        elif task == "registration":
            if mode in GRAY_REGISTRATION:
                rgb = gray(GRAY_REGISTRATION[mode])
            elif mode == "before":
                fixed, moving = unit("fixed"), unit("moving_on_fixed")
                rgb = np.stack([fixed, moving, fixed], axis=-1)
                extra["legend"] = "fixed in magenta, moving (not registered) in green: grey where they agree"
            elif mode == "overlay":
                fixed, warped = unit("fixed"), unit("warped")
                rgb = np.stack([fixed, warped, fixed], axis=-1)
                extra["legend"] = "fixed in magenta, warped moving in green: grey where they agree"
            elif mode == "checkerboard":
                rgb = np.repeat(checkerboard(unit("fixed"), unit("warped"))[..., None], 3, axis=-1)
                extra["legend"] = (
                    "alternating tiles of the fixed and the warped moving image: edges continue when aligned"
                )
            elif mode == "difference":
                resolved = self.resolve_window(window, "fixed")
                rgb, scale = heat(
                    np.abs(
                        self._plane("fixed", frame, index, crop).astype(np.float32)
                        - self._plane("warped", frame, index, crop)
                    ),
                    "hot",
                )
                extra["scale"] = {"difference_max": scale, "legend": "black 0 to white at the 99th percentile"}
            else:
                raise ViewError(f"unknown mode {mode}")
            if mode == "field_magnitude":
                field = self._plane("field", frame, index, crop).astype(np.float32)
                colours, scale = heat(np.linalg.norm(field, axis=0), "viridis")
                rgb = 0.4 * rgb + 0.6 * colours
                extra["scale"] = {"magnitude_max_mm": scale, "legend": "dark 0 mm to yellow at the 99th percentile"}
            elif mode == "jacobian":
                rgb = self._jacobian_overlay(rgb, frame, index, crop, extra)
            elif mode == "labels":
                if WARPED_LABEL_ROLES & self.case.inputs.keys():
                    rgb = fill_labels(rgb, self._plane("warped_labels", frame, index, crop), 0.35, label)
                if "fixed_labels" in self.case.inputs:
                    fixed_labels = self._plane("fixed_labels", frame, index, crop)
                    rgb = draw_contours(rgb, fixed_labels, label, color=(1.0, 1.0, 1.0))
                extra["legend"] = "warped moving labels filled in colour, fixed labels as white contours"
        else:  # synthesis
            if mode in ("synthetic", "target", "source"):
                rgb = gray(mode)
            elif mode == "difference":
                resolved = self.resolve_window(window, "synthetic")
                rgb, scale = heat(
                    np.abs(
                        self._plane("synthetic", frame, index, crop).astype(np.float32)
                        - self._plane("target", frame, index, crop)
                    ),
                    "hot",
                )
                extra["scale"] = {"difference_max": scale, "legend": "black 0 to white at the 99th percentile"}
            elif mode == "checkerboard":  # one window, the target's: the comparison is of intensities
                rgb = np.repeat(checkerboard(unit("synthetic", "target"), unit("target"))[..., None], 3, axis=-1)
                extra["legend"] = "alternating tiles of the synthetic and the real target image"
            elif mode == "side_by_side":
                left, right = gray("synthetic", "target"), gray("target")
                rgb = np.concatenate([left, np.ones((left.shape[0], 2, 3)), right], axis=1)
                extra["legend"] = "synthetic on the left, real target on the right"
            else:
                raise ViewError(f"unknown mode {mode}")
        return rgb, extra, (list(resolved) if resolved is not None else None)

    def _jacobian_overlay(
        self, rgb: np.ndarray, frame: PlaneFrame, index: int, crop: tuple[slice, slice] | None, extra: dict[str, object]
    ) -> np.ndarray:
        field = self.volume("field")
        lo = [0, 0, 0]
        hi = list(self.grid.size_zyx)
        lo[frame.array_axis] = max(0, index - 1)
        hi[frame.array_axis] = min(self.grid.size_zyx[frame.array_axis], index + 2)
        if crop is not None:
            lo[frame.rows_axis], hi[frame.rows_axis] = crop[0].start, crop[0].stop
            lo[frame.cols_axis], hi[frame.cols_axis] = crop[1].start, crop[1].stop
        block = field.region(tuple(lo), tuple(hi)).astype(np.float64)  # type: ignore[arg-type]
        determinant = field_jacobian_determinant(np.asarray(block, np.float64), self.grid.world_to_index.matrix)
        plane = np.take(determinant, index - lo[frame.array_axis], axis=frame.array_axis)
        folded = plane <= 0
        unit = np.clip(np.log2(np.clip(plane, 1e-9, None)) / 2 + 0.5, 0.0, 1.0)  # det 0.25..4 over the map
        overlay = colormap(unit, "coolwarm")
        out = 0.4 * rgb + 0.6 * overlay
        out[folded] = (1.0, 0.0, 1.0)
        extra["scale"] = {
            "legend": "blue: det J 0.25 (shrinking), grey 1, red 4 (expanding); magenta: det J <= 0 (fold)",
            "folded_pixels": int(folded.sum()),
            "det_min": round(float(plane.min()), 4),
            "det_max": round(float(plane.max()), 4),
        }
        return out

    # ------------------------------------------------------------------ display

    def _world_axis(self, frame: PlaneFrame) -> int:
        return dominant_world_axis(self.grid, frame.array_axis)[0]

    def _axis_letter(self, frame: PlaneFrame) -> str:
        return "xyz"[self._world_axis(frame)]

    def _position(self, frame: PlaneFrame, index: int, crop: tuple[slice, slice] | None = None) -> np.ndarray:
        """The world point at the centre of what is shown: the slice's, or the zoom's."""
        center = [(n - 1) / 2 for n in reversed(self.grid.size_zyx)]
        center[2 - frame.array_axis] = float(index)
        if crop is not None:
            center[2 - frame.rows_axis] = (crop[0].start + crop[0].stop - 1) / 2
            center[2 - frame.cols_axis] = (crop[1].start + crop[1].stop - 1) / 2
        return self.grid.index_to_world.apply(np.array(center))

    def _plane_name(self, frame: PlaneFrame) -> str:
        """The plane, and its tilt when the grid is oblique: the world coordinate then changes across the view."""
        normal = axis_world_vector(self.grid, frame.array_axis)
        cosine = abs(normal[self._world_axis(frame)]) / max(float(np.linalg.norm(normal)), 1e-12)
        tilt = float(np.degrees(np.arccos(min(cosine, 1.0))))
        return frame.plane + (f" tilted {tilt:.0f}°" if tilt >= 1 else "")

    def _window_text(self, window: list[float] | None) -> str:
        return "n/a" if window is None else f"[{window[0]:.4g}, {window[1]:.4g}]"

    def _header_text(
        self,
        frame: PlaneFrame,
        index: int,
        mode: str,
        window: list[float] | None,
        extra: dict[str, object],
        crop: tuple[slice, slice] | None,
    ) -> str:
        position = self._position(frame, index, crop)
        depth = self.grid.size_zyx[frame.array_axis]
        text = (
            f"{{id}} {self._plane_name(frame)} {self._axis_letter(frame)}={position[self._world_axis(frame)]:.1f}mm "
            f"(slice {index}/{depth - 1}) | {mode} | window {self._window_text(window)}"
        )
        scale = extra.get("scale")
        if isinstance(scale, dict):
            numbers = {k: v for k, v in scale.items() if k != "legend"}
            text += " | " + ", ".join(f"{k}={v}" for k, v in numbers.items())
        return text

    def _display(
        self,
        rgb: np.ndarray,
        frame: PlaneFrame,
        index: int,
        crop: tuple[slice, slice] | None,
        header: str | None,
        max_px: int | None = None,
    ) -> tuple[Image.Image, dict[str, list[float]]]:
        """Flips, isotropic pixels, size limits, header, direction letters, scale bar; and the pixel-to-world affine."""
        array = np.asarray(rgb, dtype=np.float32)
        rows0 = crop[0].start if crop is not None else 0
        cols0 = crop[1].start if crop is not None else 0
        height0, width0 = array.shape[:2]
        u_step = axis_world_vector(self.grid, frame.cols_axis)
        v_step = axis_world_vector(self.grid, frame.rows_axis)
        first = np.zeros(3)
        first[frame.array_axis] = index
        first[frame.rows_axis] = rows0
        first[frame.cols_axis] = cols0
        origin = self.grid.index_to_world.apply(first[::-1])  # array order -> xyz
        if frame.flip_cols:
            array = array[:, ::-1]
            origin = origin + (width0 - 1) * u_step
            u_step = -u_step
        if frame.flip_rows:
            array = array[::-1, :]
            origin = origin + (height0 - 1) * v_step
            v_step = -v_step
        drawn_width = array.shape[1]  # side_by_side widens the array beyond the plane
        mm_u, mm_v = float(np.linalg.norm(u_step)), float(np.linalg.norm(v_step))
        finest = min(mm_u, mm_v)
        width1 = max(1, round(drawn_width * mm_u / finest))
        height1 = max(1, round(height0 * mm_v / finest))
        # the longest edge takes the target size: a small volume or a zoom is enlarged (nearest: every voxel a
        # crisp block), a large plane reduced
        target = max_px or self.config.max_px
        longest = max(width1, height1)
        if longest != target:
            factor = target / longest
            width1, height1 = max(1, round(width1 * factor)), max(1, round(height1 * factor))
        image = Image.fromarray((np.clip(array, 0, 1) * 255).astype(np.uint8))
        upscaling = width1 >= drawn_width and height1 >= height0
        image = image.resize((width1, height1), Image.Resampling.NEAREST if upscaling else Image.Resampling.BILINEAR)
        u_px = u_step * (drawn_width / width1)
        v_px = v_step * (height0 / height1)
        origin_px = origin - 0.5 * u_step - 0.5 * v_step + 0.5 * u_px + 0.5 * v_px
        header_h = HEADER_PX if header else 0
        canvas = Image.new("RGB", (image.width, image.height + header_h), (20, 20, 20))
        canvas.paste(image, (0, header_h))
        draw = ImageDraw.Draw(canvas)
        if header:
            draw.text((4, 3), header, fill=(255, 255, 255), font=self.font)
        self._edges(draw, canvas, frame, header_h)
        self._scale_bar(draw, canvas, float(np.linalg.norm(u_px)))
        mapping = {
            "origin": [round(float(v), 4) for v in origin_px - header_h * v_px],
            "u": [round(float(v), 6) for v in u_px],
            "v": [round(float(v), 6) for v in v_px],
        }
        return canvas, mapping

    def _edges(self, draw: ImageDraw.ImageDraw, canvas: Image.Image, frame: PlaneFrame, header_h: int) -> None:
        left, right, top, bottom = frame.edges
        width, height = canvas.width, canvas.height
        mid_y = header_h + (height - header_h) // 2
        for text, xy in (
            (left, (3, mid_y - 6)),
            (right, (width - 11, mid_y - 6)),
            (top, (width // 2 - 3, header_h + 2)),
            (bottom, (width // 2 - 3, height - 14)),
        ):
            x, y = xy
            draw.rectangle((x - 2, y - 1, x + 9, y + 12), fill=(0, 0, 0))
            draw.text((x, y), text, fill=(255, 255, 0), font=self.font)

    def _scale_bar(self, draw: ImageDraw.ImageDraw, canvas: Image.Image, mm_per_px: float) -> None:
        for length in (50.0, 20.0, 10.0, 5.0, 2.0, 1.0, 0.5, 0.1):
            pixels = length / mm_per_px if mm_per_px > 0 else 0
            if 20 <= pixels <= canvas.width * 0.4:
                x0, y0 = 14, canvas.height - 8
                draw.line((x0, y0, x0 + int(pixels), y0), fill=(255, 255, 255), width=2)
                draw.text((x0, y0 - 14), f"{length:g} mm", fill=(255, 255, 255), font=self.font)
                return

    def _caption(self, image: Image.Image, text: str) -> None:
        draw = ImageDraw.Draw(image)
        draw.rectangle((0, 0, min(image.width, 8 + 7 * len(text)), 14), fill=(0, 0, 0))
        draw.text((3, 1), text, fill=(255, 255, 255), font=self.font)

    def _record(
        self,
        view_id: str,
        frame: PlaneFrame,
        index: int,
        mode: str,
        window: list[float] | None,
        image: Image.Image,
        mapping: dict[str, list[float]],
        extra: dict[str, object],
        crop: tuple[slice, slice] | None,
        sent: bool,
    ) -> View:
        lo = [0, 0, 0]
        hi = list(self.grid.size_zyx)
        lo[frame.array_axis], hi[frame.array_axis] = index, index + 1
        if crop is not None:
            lo[frame.rows_axis], hi[frame.rows_axis] = crop[0].start, crop[0].stop
            lo[frame.cols_axis], hi[frame.cols_axis] = crop[1].start, crop[1].stop
        box = self.grid.world_box(tuple(slice(a, b) for a, b in zip(lo, hi, strict=True)))
        position = self._position(frame, index, crop)
        view = View(
            id=view_id,
            kind="slice",
            plane=frame.plane,
            array_axis=frame.array_axis,
            index=index,
            position_mm=[round(float(v), 3) for v in position],
            box_mm=box_to_dict(box),
            mode=mode,
            window=window,
            size_px=[image.width, image.height],
            px_to_mm=mapping,
            path="",
            sha256="",
            sent=sent,
            extra=extra,
        )
        if mode in ("field_arrows", "grid", "landmarks"):
            self._draw_vectors(image, view, frame, index, crop, mode)
        self.views.append(view)
        return view

    def _png(self, image: Image.Image, view: View) -> bytes:
        buffer = io.BytesIO()
        image.save(buffer, format="PNG", optimize=True)
        data = buffer.getvalue()
        view.sha256 = hashlib.sha256(data).hexdigest()
        path = self.evidence / f"{view.id}.png"
        path.write_bytes(data)
        view.path = f"evidence/{view.id}.png"
        return data

    # ------------------------------------------------------------------ vector drawings (registration)

    def _to_pixel(self, view: View, points: np.ndarray) -> np.ndarray:
        """World points to (col, row) pixels of a slice view, by the view's own affine."""
        assert view.px_to_mm is not None
        origin = np.asarray(view.px_to_mm["origin"])
        basis = np.stack([view.px_to_mm["u"], view.px_to_mm["v"]], axis=1)  # [3, 2]
        solution, *_ = np.linalg.lstsq(basis, (np.asarray(points, dtype=float) - origin).T, rcond=None)
        return solution.T

    def _draw_vectors(
        self, image: Image.Image, view: View, frame: PlaneFrame, index: int, crop: tuple[slice, slice] | None, mode: str
    ) -> None:
        draw = ImageDraw.Draw(image)
        if mode == "landmarks":
            self._draw_landmarks(draw, view)
            return
        field = self._plane("field", frame, index, crop).astype(np.float64)  # [3, H, W] in array orientation
        if frame.flip_cols:
            field = field[:, :, ::-1]
        if frame.flip_rows:
            field = field[:, ::-1, :]
        height0, width0 = field.shape[1:]
        assert view.px_to_mm is not None
        u, v = np.asarray(view.px_to_mm["u"]), np.asarray(view.px_to_mm["v"])
        scale_x, scale_y = view.size_px[0] / width0, (view.size_px[1] - HEADER_PX) / height0
        step = max(12, int(min(view.size_px) / 18))
        rows, columns = range(0, view.size_px[1] - HEADER_PX), range(0, view.size_px[0])

        def moved(row: int, col: int) -> tuple[float, float]:
            """Where the field takes the fixed point drawn at (col, row) under the header."""
            vector = field[:, min(height0 - 1, int(row / scale_y)), min(width0 - 1, int(col / scale_x))]
            return col + np.dot(vector, u) / np.dot(u, u), row + HEADER_PX + np.dot(vector, v) / np.dot(v, v)

        if mode == "field_arrows":
            for row in rows[step // 2 :: step]:
                for col in columns[step // 2 :: step]:
                    x1, y1 = moved(row, col)
                    draw.line((col, row + HEADER_PX, x1, y1), fill=(255, 255, 0), width=1)
                    draw.ellipse((x1 - 1.5, y1 - 1.5, x1 + 1.5, y1 + 1.5), fill=(255, 80, 0))
            view.extra["legend"] = "arrows: the displacement of the fixed point at their tail, in the plane, to scale"
        elif mode == "grid":
            fine = max(1, step // 4)
            for row in rows[::step]:
                draw.line([moved(row, col) for col in columns[::fine]], fill=(0, 255, 255), width=1)
            for col in columns[::step]:
                draw.line([moved(row, col) for row in rows[::fine]], fill=(0, 255, 255), width=1)
            view.extra["legend"] = (
                "a regular grid of the fixed image pushed by the field: crossings and folds show as crushed cells"
            )

    def _draw_landmarks(self, draw: ImageDraw.ImageDraw, view: View) -> None:
        assert self.derived is not None
        fixed = self.case.landmarks("fixed_landmarks")
        moving = self.case.landmarks("moving_landmarks")
        moved = self.derived.moved_landmarks()
        if fixed is None or view.position_mm is None:
            return
        normal = axis_world_vector(self.grid, view.array_axis)
        thickness = float(np.linalg.norm(normal))
        unit_normal = normal / max(thickness, 1e-9)
        position = np.asarray(view.position_mm)
        drawn = 0
        for n, point in enumerate(fixed):
            if abs(np.dot(point - position, unit_normal)) > thickness / 2:
                continue
            col, row = self._to_pixel(view, point[None])[0]
            draw.ellipse((col - 4, row - 4, col + 4, row + 4), outline=(255, 255, 0), width=2)
            if moved is not None:
                mc, mr = self._to_pixel(view, moved[n][None])[0]
                draw.line((col - 4, row, col + 4, row), fill=(255, 0, 255), width=1)
                draw.line((col, row - 4, col, row + 4), fill=(255, 0, 255), width=1)
                if moving is not None and n < len(moving):
                    tc, tr = self._to_pixel(view, moving[n][None])[0]
                    draw.line((mc, mr, tc, tr), fill=(0, 255, 255), width=2)
                    draw.rectangle((tc - 3, tr - 3, tc + 3, tr + 3), outline=(0, 255, 255), width=2)
            drawn += 1
        view.extra["legend"] = (
            "yellow circle: fixed landmark; cyan square: the corresponding moving landmark, joined "
            "by the residual (cyan) from where the field carries the fixed point"
        )
        view.extra["landmarks_in_plane"] = drawn
