import os
import types
from contextlib import nullcontext
from pathlib import Path

import konfai_apps.app as app_module
import pytest


def test_run_distributed_app_uses_requested_workspace_and_restores_cwd(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    user_dir = tmp_path / "user"
    run_dir = tmp_path / "workspace"
    user_dir.mkdir()
    run_dir.mkdir()
    monkeypatch.chdir(user_dir)
    monkeypatch.setattr(app_module, "MinimalLog", nullcontext)

    visited: dict[str, Path] = {}

    @app_module.run_distributed_app
    def wrapped(tmp_dir: Path) -> None:
        visited["cwd"] = Path.cwd()
        (Path.cwd() / "result.txt").write_text("ok", encoding="utf-8")

    wrapped(tmp_dir=run_dir)

    assert visited["cwd"] == run_dir
    assert Path.cwd() == user_dir
    assert (run_dir / "result.txt").read_text(encoding="utf-8") == "ok"


def test_run_distributed_app_cleans_auto_created_temporary_workspace(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    user_dir = tmp_path / "user"
    auto_root = tmp_path / "temp_root"
    auto_dir = auto_root / "konfai_test_auto_workspace"
    auto_root.mkdir()
    auto_dir.mkdir(exist_ok=True)
    user_dir.mkdir()
    monkeypatch.chdir(user_dir)
    monkeypatch.setattr(app_module, "MinimalLog", nullcontext)
    monkeypatch.setattr(app_module.tempfile, "gettempdir", lambda: str(auto_root))
    monkeypatch.setattr(app_module.tempfile, "mkdtemp", lambda prefix: str(auto_dir))

    visited: dict[str, Path] = {}

    @app_module.run_distributed_app
    def wrapped() -> None:
        visited["cwd"] = Path.cwd()
        (Path.cwd() / "result.txt").write_text("ok", encoding="utf-8")

    wrapped()

    assert os.path.normcase(os.path.realpath(visited["cwd"])) == os.path.normcase(os.path.realpath(auto_dir))
    assert Path.cwd() == user_dir
    assert auto_dir.exists() is False


def test_run_distributed_app_restores_cwd_after_keyboard_interrupt(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    user_dir = tmp_path / "user"
    run_dir = tmp_path / "workspace"
    user_dir.mkdir()
    run_dir.mkdir()
    monkeypatch.chdir(user_dir)
    monkeypatch.setattr(app_module, "MinimalLog", nullcontext)

    @app_module.run_distributed_app
    def wrapped(tmp_dir: Path) -> None:
        raise KeyboardInterrupt

    with pytest.raises(SystemExit) as excinfo:
        wrapped(tmp_dir=run_dir)

    assert excinfo.value.code == 130
    assert Path.cwd() == user_dir
    assert "Manual interruption" in capsys.readouterr().out


def test_run_distributed_app_resolves_relative_output_before_chdir(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    user_dir = tmp_path / "user"
    run_dir = tmp_path / "workspace"
    user_dir.mkdir()
    run_dir.mkdir()
    monkeypatch.chdir(user_dir)
    monkeypatch.setattr(app_module, "MinimalLog", nullcontext)

    seen: dict[str, Path] = {}

    @app_module.run_distributed_app
    def wrapped(output: Path, tmp_dir: Path) -> None:
        seen["output"] = output
        seen["cwd"] = Path.cwd()
        output.mkdir(parents=True, exist_ok=True)
        (output / "artifact.txt").write_text("ok", encoding="utf-8")

    wrapped(output=Path("Results"), tmp_dir=run_dir)

    assert seen["cwd"] == run_dir
    assert seen["output"] == (user_dir / "Results")
    assert (user_dir / "Results" / "artifact.txt").read_text(encoding="utf-8") == "ok"


def test_run_distributed_app_resolves_relative_inputs_before_chdir(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    user_dir = tmp_path / "user"
    run_dir = tmp_path / "workspace"
    user_dir.mkdir()
    run_dir.mkdir()
    monkeypatch.chdir(user_dir)
    monkeypatch.setattr(app_module, "MinimalLog", nullcontext)

    seen: dict[str, object] = {}

    @app_module.run_distributed_app
    def wrapped(inputs: list[list[Path]], mask: list[list[Path]] | None, dataset: Path, tmp_dir: Path) -> None:
        seen["inputs"] = inputs
        seen["mask"] = mask
        seen["dataset"] = dataset

    wrapped(
        inputs=[[Path("a.mha")], [Path("b.mha")]],
        mask=None,
        dataset=Path("Dataset"),
        tmp_dir=run_dir,
    )

    assert seen["inputs"] == [[user_dir / "a.mha"], [user_dir / "b.mha"]]
    assert seen["mask"] is None
    assert seen["dataset"] == user_dir / "Dataset"


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("patient.1.nii.gz", ".nii.gz"),
        ("scan.mha", ".mha"),
        ("a.b.c.nrrd", ".nrrd"),
        ("volume.nii", ".nii"),
    ],
)
def test_supported_suffix_handles_multi_dot_names(name: str, expected: str) -> None:
    assert app_module.KonfAIApp._supported_suffix(Path(name)) == expected


def test_dataset_writer_preserves_registered_extension_for_multidot_names(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "patient.1.nii.gz").write_bytes(b"volume")
    monkeypatch.chdir(tmp_path)

    app = app_module.KonfAIApp.__new__(app_module.KonfAIApp)
    app._write_inputs_to_dataset([[data_dir]])

    volume = tmp_path / "Dataset" / "P000" / "Volume_0.nii.gz"
    assert volume.is_symlink() or volume.exists()
    assert Path(os.readlink(volume)).name == "patient.1.nii.gz"


class _FakeSseResponse:
    def __init__(self, lines: list[str], status_code: int = 200) -> None:
        self._lines = lines
        self.status_code = status_code

    def raise_for_status(self) -> None:
        return None

    def iter_lines(self, decode_unicode: bool = False):  # type: ignore[no-untyped-def]
        yield from self._lines

    def __enter__(self) -> "_FakeSseResponse":
        return self

    def __exit__(self, *exc: object) -> bool:
        return False


def _stub_client(monkeypatch: pytest.MonkeyPatch, lines: list[str]) -> "app_module.KonfAIAppClient":
    client = app_module.KonfAIAppClient.__new__(app_module.KonfAIAppClient)
    client.remote_server = types.SimpleNamespace(  # type: ignore[attr-defined]
        get_url=lambda: "http://server",
        get_headers=lambda: {},
    )
    monkeypatch.setattr(app_module.requests, "get", lambda *a, **k: _FakeSseResponse(lines))
    return client


def test_stream_logs_raises_on_error_marker(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _stub_client(monkeypatch, ["data: hello", "data: __ERROR__ boom happened", "data: __DONE__"])

    with pytest.raises(RuntimeError, match="boom happened"):
        client.stream_logs("job123")


def test_stream_logs_returns_on_done_marker(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    client = _stub_client(monkeypatch, ["data: hello", "data: __DONE__"])

    client.stream_logs("job123")

    assert "hello" in capsys.readouterr().out


def _scramble_directory_scans(monkeypatch: pytest.MonkeyPatch, reversed_dir_names: set[str]) -> None:
    """Force `Path.rglob` to enumerate the given directories in reverse name order."""
    original_rglob = Path.rglob

    def scrambled_rglob(self: Path, pattern: str):  # type: ignore[no-untyped-def]
        entries = list(original_rglob(self, pattern))
        return iter(sorted(entries, reverse=self.name in reversed_dir_names))

    monkeypatch.setattr(Path, "rglob", scrambled_rglob)


def _case_name(link: Path) -> str:
    return Path(os.readlink(link)).name.split(".")[0]


def test_list_supported_files_returns_directory_scan_in_sorted_order(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    for name in ("case_b", "case_c", "case_a"):
        (data_dir / f"{name}.nii.gz").write_bytes(b"volume")
    _scramble_directory_scans(monkeypatch, {"data"})

    files = app_module.KonfAIApp._list_supported_files([data_dir])

    assert [f.name for f in files] == ["case_a.nii.gz", "case_b.nii.gz", "case_c.nii.gz"]


def test_list_supported_files_preserves_explicit_file_order(tmp_path: Path) -> None:
    first = tmp_path / "case_b.nii.gz"
    second = tmp_path / "case_a.nii.gz"
    first.write_bytes(b"volume")
    second.write_bytes(b"volume")

    files = app_module.KonfAIApp._list_supported_files([first, second])

    assert files == [first, second]


def test_dataset_writers_pair_cases_by_sorted_name_across_directories(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    preds = tmp_path / "preds"
    refs = tmp_path / "refs"
    masks = tmp_path / "masks"
    for directory in (preds, refs, masks):
        directory.mkdir()
        for name in ("case_b", "case_c", "case_a"):
            (directory / f"{name}.nii.gz").write_bytes(b"volume")
    _scramble_directory_scans(monkeypatch, {"refs", "masks"})
    monkeypatch.chdir(tmp_path)

    app = app_module.KonfAIApp.__new__(app_module.KonfAIApp)
    app._write_inputs_to_dataset([[preds]])
    app._write_gt_to_dataset([[refs]])
    app._write_mask_or_default([[masks]])

    cases = sorted((tmp_path / "Dataset").iterdir())
    assert [c.name for c in cases] == ["P000", "P001", "P002"]
    assert [_case_name(c / "Volume_0.nii.gz") for c in cases] == ["case_a", "case_b", "case_c"]
    for case in cases:
        assert _case_name(case / "Volume_0.nii.gz") == _case_name(case / "Reference_0.nii.gz")
        assert _case_name(case / "Volume_0.nii.gz") == _case_name(case / "Mask_0.nii.gz")


class _FakePostResponse:
    def __init__(self, captured: dict, files: list, data: dict) -> None:
        self.status_code = 200
        captured["files"] = files
        captured["data"] = data

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return {"job_id": "job-1"}

    def __enter__(self) -> "_FakePostResponse":
        return self

    def __exit__(self, *exc: object) -> bool:
        return False


def _stub_remote_submission(monkeypatch: pytest.MonkeyPatch) -> dict:
    captured: dict = {}

    def fake_post(url, files, data, headers, timeout):  # type: ignore[no-untyped-def]
        return _FakePostResponse(captured, list(files), dict(data))

    monkeypatch.setattr(app_module.requests, "post", fake_post)
    monkeypatch.setattr(app_module.KonfAIAppClient, "stream_logs", lambda self, job_id: None)
    monkeypatch.setattr(app_module.KonfAIAppClient, "download_result", lambda self, job_id, output: None)

    client = app_module.KonfAIAppClient.__new__(app_module.KonfAIAppClient)
    client.app = "demo"  # type: ignore[attr-defined]
    client.remote_server = types.SimpleNamespace(  # type: ignore[attr-defined]
        get_url=lambda: "http://server",
        get_headers=lambda: {},
    )
    captured["client"] = client
    return captured


def test_run_remote_job_encodes_group_boundaries_and_preserves_order(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    first = tmp_path / "a.nii.gz"
    second = tmp_path / "b.nii.gz"
    first.write_bytes(b"a")
    second.write_bytes(b"b")

    captured = _stub_remote_submission(monkeypatch)
    client = captured["client"]

    # Two multi-channel groups of one file each -> "1,1", files sent in order.
    client.infer(inputs=[[first], [second]], output=tmp_path / "out")

    assert captured["data"]["inputs_groups"] == "1,1"
    field_names = [key for key, _ in captured["files"]]
    assert field_names == ["inputs", "inputs"]
    sent_names = [Path(handle.name).name for _, handle in captured["files"]]
    assert sent_names == ["a.nii.gz", "b.nii.gz"]


def test_run_remote_job_counts_directory_expansion_in_group_sizes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    data_dir = tmp_path / "cohort"
    data_dir.mkdir()
    (data_dir / "case_a.nii.gz").write_bytes(b"a")
    (data_dir / "case_b.nii.gz").write_bytes(b"b")

    captured = _stub_remote_submission(monkeypatch)
    client = captured["client"]

    # A single group holding a directory expands to two files -> "2".
    client.infer(inputs=[[data_dir]], output=tmp_path / "out")

    assert captured["data"]["inputs_groups"] == "2"
    field_names = [key for key, _ in captured["files"]]
    assert field_names == ["inputs", "inputs"]
    sent_names = sorted(Path(handle.name).name for _, handle in captured["files"])
    assert sent_names == ["case_a.nii.gz", "case_b.nii.gz"]


def test_run_remote_job_encodes_gpu_ids_as_csv(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    volume = tmp_path / "a.nii.gz"
    volume.write_bytes(b"a")

    captured = _stub_remote_submission(monkeypatch)
    client = captured["client"]

    client.infer(inputs=[[volume]], output=tmp_path / "out", gpu=[0, 1])

    # A multi-GPU selection travels as a single CSV field so the server keeps every id.
    assert captured["data"]["gpu"] == "0,1"


def test_run_remote_job_encodes_empty_gpu_selection_as_empty_string(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    volume = tmp_path / "a.nii.gz"
    volume.write_bytes(b"a")

    captured = _stub_remote_submission(monkeypatch)
    client = captured["client"]

    client.infer(inputs=[[volume]], output=tmp_path / "out")


def test_run_remote_job_packs_dataset_directory_as_single_zip(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import io
    import zipfile

    dataset = tmp_path / "cohort"
    (dataset / "P000").mkdir(parents=True)
    (dataset / "P000" / "Volume_0.mha").write_bytes(b"v0")
    (dataset / "P000" / "Volume_1.mha").write_bytes(b"v1")

    captured: dict = {}

    class _Resp:
        status_code = 200

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {"job_id": "job-1"}

        def __enter__(self) -> "_Resp":
            return self

        def __exit__(self, *exc: object) -> bool:
            return False

    def fake_post(url, files, data, headers, timeout):  # type: ignore[no-untyped-def]
        captured["files"] = [(field, Path(handle.name).name, handle.read()) for field, handle in files]
        captured["data"] = dict(data)
        return _Resp()

    monkeypatch.setattr(app_module.requests, "post", fake_post)
    monkeypatch.setattr(app_module.KonfAIAppClient, "stream_logs", lambda self, job_id: None)
    monkeypatch.setattr(app_module.KonfAIAppClient, "download_result", lambda self, job_id, output: None)

    client = app_module.KonfAIAppClient.__new__(app_module.KonfAIAppClient)
    client.app = "demo"  # type: ignore[attr-defined]
    client.remote_server = types.SimpleNamespace(  # type: ignore[attr-defined]
        get_url=lambda: "http://server",
        get_headers=lambda: {},
    )

    client.fine_tune(dataset=dataset, output=tmp_path / "out")

    # The dataset directory travels as exactly one zip file field, never as text.
    dataset_fields = [entry for entry in captured["files"] if entry[0] == "dataset"]
    assert len(dataset_fields) == 1
    _, filename, payload = dataset_fields[0]
    assert filename == "dataset.zip"
    with zipfile.ZipFile(io.BytesIO(payload)) as zf:
        names = set(zf.namelist())
    assert {"P000/Volume_0.mha", "P000/Volume_1.mha"} <= names
    assert "dataset" not in captured["data"]
    assert "inputs_groups" not in captured["data"]

    # No explicit GPU selection encodes as an empty string (auto mode on the server).
    assert captured["data"]["gpu"] == ""
