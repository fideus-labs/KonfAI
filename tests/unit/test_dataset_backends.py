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
