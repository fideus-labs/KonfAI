# SPDX-License-Identifier: Apache-2.0
import json

from konfai_scout.cli import main


def test_inspect_render_export(fixtures, tmp_path, capsys, monkeypatch):
    case, _ = fixtures["segmentation"]
    out = tmp_path / "active"
    code = main(
        [
            "inspect",
            "--task",
            "segmentation",
            "--name",
            "c1",
            "--input",
            f"image={case.inputs['image']}",
            "--input",
            f"prediction={case.inputs['prediction']}",
            "--input",
            f"truth={case.inputs['truth']}",
            "--labels",
            "1=sphere,2=cube",
            "--out",
            str(out),
            "--provider",
            "mock",
            "--max-tool-calls",
            "12",
            "--second-reading",
            "--json",
        ]
    )
    assert code == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["case"] == "c1" and summary["verdict"] and (out / "report.md").exists()
    assert summary["second_reading"]["observations"] == 1 and summary["observations"][0]["second_reading"] == "unsure"
    case.save(tmp_path / "case.json")
    code = main(
        [
            "inspect",
            "--case",
            str(tmp_path / "case.json"),
            "--out",
            str(tmp_path / "fixed"),
            "--provider",
            "mock",
            "--mode",
            "fixed",
            "--no-blind",
        ]
    )
    assert code == 0 and "->" in capsys.readouterr().out
    assert json.loads((tmp_path / "fixed" / "report.json").read_text())["config"]["blind"] is False
    assert main(["render", str(out)]) == 0
    assert main(["export", str(out), "--level", "counts", "--out", str(tmp_path / "counts.json")]) == 0
    counts = json.loads((tmp_path / "counts.json").read_text())
    assert set(counts) == {"level", "verdict", "counts"} and counts["verdict"] == summary["verdict"]


def test_refusals_before_any_request(fixtures, tmp_path, capsys, monkeypatch):
    case, _ = fixtures["segmentation"]
    assert (
        main(["inspect", "--task", "segmentation", "--input", f"image={case.inputs['image']}", "--out", str(tmp_path)])
        == 2
    )
    assert "needs" in capsys.readouterr().err
    monkeypatch.setenv("ANTHROPIC_API_KEY", "not-a-key")
    code = main(["inspect", "--case", str(tmp_path / "nope.json"), "--out", str(tmp_path)])
    assert code == 2
    case.save(tmp_path / "case.json")
    code = main(
        ["inspect", "--case", str(tmp_path / "case.json"), "--out", str(tmp_path / "r"), "--provider", "anthropic"]
    )
    assert code == 2 and "allow-cloud-images" in capsys.readouterr().err
    assert not (tmp_path / "r" / "report.json").exists()


def test_a_depth_sets_the_defaults_and_an_explicit_flag_wins():
    from konfai_scout.cli import _config_from_args, _parser
    from konfai_scout.config import ScoutConfig

    def config(*flags):
        return _config_from_args(_parser().parse_args(["inspect", "--out", "o", *flags]))

    quick = config("--depth", "quick", "--max-tool-calls", "20")
    assert (quick.effort, quick.budget.max_tool_calls, quick.second_reading) == ("low", 20, False)
    assert config("--depth", "quick").budget.max_tool_calls == 10
    deep = ScoutConfig(depth="deep", effort="medium")
    assert (deep.effort, deep.budget.max_tool_calls, deep.second_reading) == ("medium", 40, True)
    assert config().budget.max_tool_calls == 25 and config().effort is None
    assert ScoutConfig.from_dict(quick.to_dict()).budget.max_tool_calls == 20  # a saved config reads back as it was


def test_the_survey_can_be_turned_off(fixtures, tmp_path):
    import json

    case, _ = fixtures["segmentation"]
    case.save(tmp_path / "case.json")
    main(["inspect", "--case", str(tmp_path / "case.json"), "--out", str(tmp_path / "out"), "--no-survey"])
    report = json.loads((tmp_path / "out" / "report.json").read_text())
    assert report["config"]["survey"] is False and report["leads"] == []
