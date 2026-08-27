from pathlib import Path

# ---------------------------------------------------------------- what a whole-volume scan decodes


def test_a_scan_raises_its_grain_onto_the_store_block_it_would_decode_anyway() -> None:
    """A chunked store decodes whole blocks, so a scan stepping finer decodes the same block again
    at every step inside it -- measured at 85x, and 170x where the step straddled two. Where the
    budget holds a whole block the grain is raised to it, which reads each block once."""
    from konfai.utils.dataset import _scan_block_on_the_store_grid

    rows, held = _scan_block_on_the_store_grid(rows=3, extent=512, plane=1000, granularity=[64], budget=1 << 30)
    assert rows == 64, "raised onto the grid"
    assert held == 64 * 3 * 1000 * 4, "an aligned block decodes itself and nothing more"


def test_a_scan_that_cannot_afford_a_whole_block_is_charged_for_the_one_it_decodes() -> None:
    """Where the budget cannot hold a stored block the grain stays fine -- and what the store
    decodes is charged, so the plan refuses instead of the kernel."""
    from konfai.utils.dataset import _scan_block_on_the_store_grid

    tight = 4 * 1000 * 4 * 8  # far under one 64-row block
    rows, held = _scan_block_on_the_store_grid(rows=3, extent=512, plane=1000, granularity=[64], budget=tight)
    assert rows == 3, "the grain the budget bought"
    assert held > 3 * 3 * 1000 * 4, "and the block it really decodes, charged"


def test_an_unchunked_store_is_charged_for_what_it_is_asked_for() -> None:
    from konfai.utils.dataset import _scan_block_on_the_store_grid

    rows, held = _scan_block_on_the_store_grid(rows=7, extent=512, plane=1000, granularity=None, budget=1 << 30)
    assert (rows, held) == (7, 7 * 3 * 1000 * 4)


# ---------------------------------------------------------------- what a memmapped store serves


def test_a_memmapped_store_declares_the_band_a_region_read_touches(tmp_path: "Path") -> None:
    """A memmap is served band by band: the read maps the outermost axis the window spans and every
    axis below it whole, so a region narrower than a plane touches a plane's pages and the kernel
    counts them. Priced on the voxels it asked for instead, the shape search answered a cube under
    96 MiB and the run held MORE at a smaller budget (77 MiB at 64 against 58 at 96).

    Declared the way a chunked store declares its chunk, so one mechanism prices both.
    """
    import numpy as np
    import SimpleITK as sitk
    from konfai.utils.dataset import Dataset, chunk_hull_voxels

    case = tmp_path / "Dataset" / "CASE_000"
    case.mkdir(parents=True)
    image = sitk.GetImageFromArray(np.zeros((40, 24, 16), dtype=np.int16))
    image.SetSpacing((1.0, 1.0, 1.0))
    sitk.WriteImage(image, str(case / "CT.mha"), useCompression=False)

    granularity = Dataset(str(tmp_path / "Dataset"), "mha").read_granularity("CT", "CASE_000")
    assert granularity == (1, 1, 24, 16), "one step along the banded axis, everything below it whole"

    # And the hull that grain implies IS the band: a cube costs its rows times the whole plane.
    spatial, cube = [40, 24, 16], (slice(8, 16), slice(4, 12), slice(2, 10))
    assert chunk_hull_voxels(cube, granularity[1:], spatial) == 8 * 24 * 16
    slab = (slice(8, 16), slice(0, 24), slice(0, 16))
    assert chunk_hull_voxels(slab, granularity[1:], spatial) == 8 * 24 * 16, "a full-plane slab costs itself"


def test_a_compressed_store_declares_no_grain_because_it_serves_no_bounded_region(tmp_path: "Path") -> None:
    """Compressed, ITK decodes forward from the start and stops on the region, so a read costs its
    END offset: 14.6 / 41.8 / 70.5 / 112.6 ms for one 32^3 region at z = 0 / 64 / 128 / 224 of a
    256^3 volume whose whole decode is 102 ms. There is no block to align to, only a prefix, and
    ``bounded_region_reads`` is what says so -- the grain stays unstated."""
    import numpy as np
    import SimpleITK as sitk
    from konfai.utils.dataset import Dataset

    case = tmp_path / "Dataset" / "CASE_000"
    case.mkdir(parents=True)
    image = sitk.GetImageFromArray(np.zeros((40, 24, 16), dtype=np.int16))
    image.SetSpacing((1.0, 1.0, 1.0))
    sitk.WriteImage(image, str(case / "CT.mha"), useCompression=True)

    dataset = Dataset(str(tmp_path / "Dataset"), "mha")
    assert dataset.read_granularity("CT", "CASE_000") is None
    assert not dataset.bounded_region_reads("CT", "CASE_000")
