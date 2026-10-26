"""What the tools' engines share."""

from impact_reg_konfai.models.tools import gpu_environment


def test_a_tool_s_subprocess_runs_on_the_physical_gpu_behind_this_process_index(monkeypatch) -> None:
    # The subprocess reads CUDA_VISIBLE_DEVICES afresh: the local index 1 of a process that sees GPUs 3 and 5 is 5.
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3,5")
    assert gpu_environment(1)["CUDA_VISIBLE_DEVICES"] == "5"
    on_cpu = gpu_environment(-1, PYTHONPATH="x")
    assert on_cpu["CUDA_VISIBLE_DEVICES"] == "" and on_cpu["PYTHONPATH"] == "x"
