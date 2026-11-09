# SPDX-License-Identifier: Apache-2.0
"""Where to look: a deterministic survey that hands the model leads, each a region and the reason it stands out.
A lead is not a finding; the model looks at it and judges. Segmentation: label boundaries that run where the image
has no edge, components unlike their label's main one, and with a truth the clusters of disagreement. Registration:
folded voxels, and blocks where the edges of the warped and fixed images disagree. Synthesis: clusters of error
against the target that are more than a seam along an edge. Volumes are read by blocks; a label map whole only when
it is small, as for the label summary. The figures in a reason rank the lead: they are not measurements."""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import asdict, dataclass

import numpy as np
from scipy import ndimage

from .measure import jacobian_determinant
from .planes import box_to_dict
from .views import Renderer

MAX_LEADS = 8
BLOCK = 64
CELL_MM = 16.0
WEAK = 0.55  # a boundary voxel is edgeless below this fraction of the label's median boundary edge
_FULL = np.ones((3, 3, 3), dtype=bool)


@dataclass
class Lead:
    id: str
    kind: str
    box_mm: dict[str, list[float]]
    reason: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


Found = tuple[str, tuple[slice, ...], str]  # kind, region (array order), reason


def survey(renderer: Renderer) -> list[Lead]:
    find = {"segmentation": _segmentation, "registration": _registration, "synthesis": _synthesis}
    found = find[renderer.case.task](renderer)[:MAX_LEADS]
    return [
        Lead(f"l{n:02d}", kind, box_to_dict(renderer.grid.world_box(region)), reason)
        for n, (kind, region, reason) in enumerate(found, start=1)
    ]


def leads_text(leads: list[Lead], active: bool) -> str:
    if not leads:
        return "Leads: the survey found nothing that stands out."
    lines = [
        "Leads, from a deterministic survey of the case: where the result is most likely wrong, not findings."
        + (" Look at each through a slice of its box (a view zoomed on it) before concluding." if active else "")
    ]
    lines += [f"- {lead.id} {lead.kind}, box {json.dumps(lead.box_mm)}: {lead.reason}" for lead in leads]
    return "\n".join(lines)


# ---------------------------------------------------------------------------------------------- helpers


def _blocks(size_zyx: tuple[int, ...], block: tuple[int, int, int], margin: int) -> Iterator[tuple[tuple, tuple]]:
    """Blocks covering the grid: the inner window, and the window read with ``margin`` voxels around it."""
    for z in range(0, size_zyx[0], block[0]):
        for y in range(0, size_zyx[1], block[1]):
            for x in range(0, size_zyx[2], block[2]):
                lo = (z, y, x)
                hi = tuple(min(a + b, n) for a, b, n in zip(lo, block, size_zyx, strict=True))
                olo = tuple(max(a - margin, 0) for a in lo)
                ohi = tuple(min(b + margin, n) for b, n in zip(hi, size_zyx, strict=True))
                yield (lo, hi), (olo, ohi)


def _inner(lo: tuple, hi: tuple, olo: tuple) -> tuple[slice, ...]:
    return tuple(slice(a - o, b - o) for a, b, o in zip(lo, hi, olo, strict=True))


def _edges(image: np.ndarray, spacing_zyx: np.ndarray) -> np.ndarray:
    """Edge strength: the gradient magnitude, per millimetre, of the image smoothed over one voxel."""
    smooth = ndimage.gaussian_filter(image.astype(np.float32), 1.0)
    return np.sqrt(sum(g**2 for g in np.gradient(smooth, *spacing_zyx)))


def _clusters(mask: np.ndarray, minimum: int) -> list[tuple[int, tuple[slice, ...]]]:
    """Connected clusters of at least ``minimum`` voxels, largest first, with their array windows."""
    labelled, count = ndimage.label(mask, structure=_FULL)
    if count == 0:
        return []
    sizes = np.bincount(labelled.ravel())[1:]
    windows = ndimage.find_objects(labelled)
    return sorted(((int(sizes[i]), windows[i]) for i in range(count) if sizes[i] >= minimum), key=lambda c: -c[0])


def _zyx(values: object) -> tuple[int, int, int]:
    z, y, x = (int(v) for v in values)  # type: ignore[attr-defined]
    return z, y, x


def _offset(window: tuple[slice, ...], lo: tuple[int, ...]) -> tuple[slice, ...]:
    return tuple(slice(s.start + o, s.stop + o) for s, o in zip(window, lo, strict=True))


# ---------------------------------------------------------------------------------------------- segmentation


def _segmentation(r: Renderer) -> list[Found]:
    case, grid = r.case, r.grid
    if max(grid.size_zyx) > 512:  # the same limit as the label summary: a big map is not read whole
        return []
    size = _zyx(grid.size_zyx)
    prediction = r.volume("prediction").region((0, 0, 0), size)[0]
    truth = r.volume("truth").region((0, 0, 0), size)[0] if "truth" in case.inputs else None
    spacing = np.asarray(grid.spacing_xyz, dtype=np.float64)[::-1]
    margin = np.ceil(10.0 / spacing).astype(int)
    components: list[tuple[float, Found]] = []
    edgeless: list[tuple[int, Found]] = []
    for label in (int(v) for v in np.unique(prediction) if v != 0):
        name = case.labels.get(label, f"label {label}") if case.labels else f"label {label}"
        where = np.argwhere(prediction == label)
        lo = _zyx(np.maximum(where.min(axis=0) - margin, 0))
        hi = _zyx(np.minimum(where.max(axis=0) + margin + 1, grid.size_zyx))
        window = tuple(slice(a, b) for a, b in zip(lo, hi, strict=True))
        image = r.volume(case.image_role).region(lo, hi)[0].astype(np.float32)
        mask, other = prediction[window] == label, (prediction[window] != 0) & (prediction[window] != label)
        components += [
            (score, (k, _offset(w, lo), why)) for score, (k, w, why) in _components(image, mask, other, name)
        ]
        edgeless += [(size, (k, _offset(w, lo), why)) for size, (k, w, why) in _edgeless(image, mask, spacing, name)]
    found = [f for _, f in sorted(components, key=lambda c: -c[0])[:3]]
    found += [f for _, f in sorted(edgeless, key=lambda c: -c[0])[:3]]
    if truth is not None:
        found += _disagreement(prediction, truth, case.labels or {})[: MAX_LEADS - len(found)]
    return found


def _components(image: np.ndarray, mask: np.ndarray, other: np.ndarray, name: str) -> list[tuple[float, Found]]:
    """Components whose contrast against their own surroundings is unlike the main component's: a lesion label's
    satellite sitting on tissue that looks like nothing the main lesion sits on."""
    labelled, count = ndimage.label(mask, structure=_FULL)
    if count < 2:
        return []
    sizes = np.bincount(labelled.ravel())[1:]
    windows = ndimage.find_objects(labelled)
    spread = float(np.subtract(*np.percentile(image, [90, 10]))) or 1.0

    def contrast(i: int) -> float:
        pad = tuple(slice(max(s.start - 3, 0), s.stop + 3) for s in windows[i])
        piece = labelled[pad] == i + 1
        shell = ndimage.binary_dilation(piece, iterations=2) & ~mask[pad] & ~other[pad]
        if not shell.any():
            return 0.0
        return float(np.median(image[pad][piece]) - np.median(image[pad][shell])) / spread

    order = list(np.argsort(sizes)[::-1][:20])
    main = contrast(int(order[0]))
    out = []
    for i in order[1:]:
        own = contrast(int(i))
        departure = abs(own - main) / max(abs(main), 0.05)
        if departure > 0.5:
            out.append(
                (
                    departure,
                    (
                        "component",
                        windows[i],
                        f"a separate component of {name} ({int(sizes[i])} voxels) whose contrast against its own "
                        f"surroundings is {own:+.2f}, against {main:+.2f} for the main component (in units of the "
                        "image's 10-90 % spread)",
                    ),
                )
            )
    return out


def _edgeless(image: np.ndarray, mask: np.ndarray, spacing_zyx: np.ndarray, name: str) -> list[tuple[int, Found]]:
    """Stretches of the label's boundary where the image has almost no edge: a cut through a homogeneous lesion, a
    hole, a leak into tissue that does not differ, a component on nothing."""
    surface = mask & ~ndimage.binary_erosion(mask)
    if surface.sum() < 30:
        return []
    edges = _edges(image, spacing_zyx)
    reference = float(np.median(edges[surface]))
    if reference <= 0:
        return []
    weak = surface & (edges < WEAK * reference)
    out = []
    for size, window in _clusters(weak, max(15, int(0.03 * surface.sum())))[:3]:
        strength = float(edges[window][weak[window]].mean())
        out.append(
            (
                size,
                (
                    "edgeless_boundary",
                    window,
                    f"{size} voxels of the {name} boundary where the image has almost no edge (strength "
                    f"{strength:.3g} per mm, median {reference:.3g} along the whole boundary)",
                ),
            )
        )
    return out


def _disagreement(prediction: np.ndarray, truth: np.ndarray, names: dict[int, str]) -> list[Found]:
    out: list[tuple[int, Found]] = []
    for label in (int(v) for v in np.union1d(np.unique(prediction), np.unique(truth)) if v != 0):
        name = names.get(label, f"label {label}")
        reference = int((truth == label).sum())
        for kind, mask, what in (
            ("false_positive", (prediction == label) & (truth != label), "the prediction adds"),
            ("false_negative", (truth == label) & (prediction != label), "the prediction misses"),
        ):
            for size, window in _clusters(mask, max(10, int(0.01 * reference)))[:3]:
                out.append((size, (kind, window, f"{size} voxels of {name} {what} against the truth")))
    return [f for _, f in sorted(out, key=lambda c: -c[0])]


# ---------------------------------------------------------------------------------------------- registration


def _registration(r: Renderer) -> list[Found]:
    grid = r.grid
    spacing = np.asarray(grid.spacing_xyz, dtype=np.float64)[::-1]
    size = tuple(grid.size_zyx)
    found: list[Found] = []
    field = r.volume("field")
    folded = np.zeros(size, dtype=bool)
    for (lo, hi), (olo, ohi) in _blocks(size, (BLOCK,) * 3, 1):
        det = jacobian_determinant(field.region(olo, ohi), grid)
        folded[tuple(slice(a, b) for a, b in zip(lo, hi, strict=True))] = det[_inner(lo, hi, olo)] <= 0
    for count, window in _clusters(folded, 1)[:3]:
        found.append(("folding", window, f"{count} voxels where the field folds (the map's determinant is <= 0)"))
    found += _edge_disagreement(r, spacing, size)[: MAX_LEADS - len(found)]
    return found


def _edge_disagreement(r: Renderer, spacing: np.ndarray, size: tuple[int, ...]) -> list[Found]:
    """Cells of about 16 mm where the edges of the warped image correlate with the fixed image's much less than
    elsewhere in the case: the edges, not the intensities, so that two modalities compare."""
    fixed, warped = r.volume(r.case.image_role), r.volume("warped")
    cell = tuple(int(max(4, round(CELL_MM / s))) for s in spacing)
    cells: list[tuple[float, float, tuple[int, int, int]]] = []
    for (lo, hi), (olo, ohi) in _blocks(size, tuple(4 * c for c in cell), 3):  # type: ignore[arg-type]
        inner = _inner(lo, hi, olo)
        f, w = fixed.region(olo, ohi)[0], warped.region(olo, ohi)[0]
        ef, ew = _edges(f, spacing)[inner], _edges(w, spacing)[inner]
        # where the warp read outside the moving image it wrote its fill, 0: an edge of coverage, not of alignment
        unread = ((w == 0) & (f != 0))[inner]
        for z in range(0, ef.shape[0], cell[0]):
            for y in range(0, ef.shape[1], cell[1]):
                for x in range(0, ef.shape[2], cell[2]):
                    part = (slice(z, z + cell[0]), slice(y, y + cell[1]), slice(x, x + cell[2]))
                    a, b = ef[part].ravel(), ew[part].ravel()
                    if a.size < 8 or a.std() == 0 or b.std() == 0 or unread[part].mean() > 0.05:
                        continue
                    index = ((lo[0] + z) // cell[0], (lo[1] + y) // cell[1], (lo[2] + x) // cell[2])
                    cells.append((float(a.mean()), float(np.corrcoef(a, b)[0, 1]), index))
    if len(cells) < 8:
        return []
    structure = np.median([c[0] for c in cells])
    kept = [c for c in cells if c[0] >= structure]  # cells with edges to compare
    scores = np.array([c[1] for c in kept])
    median = float(np.median(scores))
    spread = float(np.median(np.abs(scores - median))) or 0.01
    shape = tuple(-(-n // c) for n, c in zip(size, cell, strict=True))
    low = np.zeros(shape, dtype=bool)
    value = np.full(shape, np.nan)
    for _, ncc, index in kept:
        value[index] = ncc
        low[index] = ncc < median - 3 * spread and ncc < 0.9 * median
    out = []
    for count, window in _clusters(low, 1)[:3]:
        worst = float(np.nanmin(value[window][low[window]]))
        cells_window = tuple(slice(w.start * c, min(w.stop * c, n)) for w, c, n in zip(window, cell, size, strict=True))
        out.append(
            (
                "edge_disagreement",
                cells_window,
                f"{count} cell(s) of about {CELL_MM:.0f} mm where the edges of the warped and fixed images agree "
                f"poorly (edge correlation down to {worst:.2f}, median {median:.2f} over the case)",
            )
        )
    return out


# ---------------------------------------------------------------------------------------------- synthesis


def _synthesis(r: Renderer) -> list[Found]:
    """Clusters where the synthetic departs from the target by several times the case's typical error, smoothed so
    that noise does not count, and off the target's strongest edges where a slight shift or blur is expected."""
    case, grid = r.case, r.grid
    if "target" not in case.inputs:
        return []
    synthetic, target = r.volume("synthetic"), r.volume("target")
    mask = r.volume("mask") if "mask" in case.inputs else None
    spacing = np.asarray(grid.spacing_xyz, dtype=np.float64)[::-1]
    size = tuple(grid.size_zyx)

    def error(olo: tuple, ohi: tuple) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """The error smoothed inside the mask only (a normalised convolution: what lies outside the mask, where the
        synthetic need not be anything, does not bleed in), the target, and the mask."""
        t = target.region(olo, ohi)[0].astype(np.float32)
        inside = mask.region(olo, ohi)[0] > 0 if mask is not None else np.ones(t.shape, bool)
        raw = (synthetic.region(olo, ohi)[0].astype(np.float32) - t) * inside
        weight = ndimage.gaussian_filter(inside.astype(np.float32), 1.5)
        return np.where(inside, ndimage.gaussian_filter(raw, 1.5) / np.maximum(weight, 1e-3), 0.0), t, inside

    errors, edges, values = [], [], []
    blocks = list(_blocks(size, (BLOCK,) * 3, 3))
    for (lo, hi), (olo, ohi) in blocks:  # pass 1: the case's typical error, its intensity range, its strong edges
        inner = _inner(lo, hi, olo)
        e, t, inside = error(olo, ohi)
        e, inside = e[inner], inside[inner]
        errors.append(np.abs(e[inside])[::7])
        edges.append(_edges(t, spacing)[inner][inside][::7])
        values.append(t[inner][inside][::7])
    if not errors or not np.concatenate(errors).size:
        return []
    low, high = np.percentile(np.concatenate(values), [1, 99])
    # several times the typical error, and at least 3 % of the target's range: a quiet case's noise is no lead
    threshold = max(4 * float(np.median(np.concatenate(errors))), 0.03 * float(high - low))
    strong = float(np.percentile(np.concatenate(edges), 90))
    if threshold <= 0:
        return []
    hot = np.zeros(size, dtype=bool)
    signed = np.zeros(size, dtype=np.float16)
    for (lo, hi), (olo, ohi) in blocks:  # pass 2: where the error is several times that, off the strong edges
        inner = _inner(lo, hi, olo)
        window = tuple(slice(a, b) for a, b in zip(lo, hi, strict=True))
        e, t, inside = error(olo, ohi)
        hot[window] = ((np.abs(e) > threshold) & (_edges(t, spacing) < strong) & inside)[inner]
        signed[window] = e[inner]
    hot = ndimage.binary_opening(hot, structure=_FULL)  # a seam one voxel thick is not a cluster
    out = []
    for count, window in _clusters(hot, 30)[:5]:
        mean = float(signed[window][hot[window]].astype(np.float32).mean())
        out.append(
            (
                "error_cluster",
                window,
                f"{count} voxels where the synthetic is {'brighter' if mean > 0 else 'darker'} than the target by "
                f"{abs(mean):.3g} on average (smoothed inside the mask), away from the target's strongest edges",
            )
        )
    return out
