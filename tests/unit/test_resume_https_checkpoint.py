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

"""Regression test: RESUME must keep an https:// checkpoint URL intact through build_train."""

from konfai.utils.runtime import State


def test_build_train_keeps_https_checkpoint_url(monkeypatch) -> None:
    # build_train must not wrap an https:// URL in Path(): that collapses '//' into 'https:/…',
    # which then fails both the startswith('https://') check and Path.exists() at load time.
    import konfai.trainer as trainer_module

    recorded: dict[str, object] = {}

    class _DummyTrainer:
        def set_model(self, path_to_model) -> None:
            recorded["model"] = path_to_model

        def set_lr(self, lr) -> None:
            recorded["lr"] = lr

    monkeypatch.setattr(trainer_module, "configure_workflow_environment", lambda **kwargs: None)
    monkeypatch.setattr(
        trainer_module,
        "apply_config",
        lambda *args, **kwargs: lambda cls: lambda: _DummyTrainer(),
    )

    url = "https://example.com/weights/ckpt.pt"
    trainer_module.build_train(command=State.RESUME, model=url)

    assert recorded["model"] == url
