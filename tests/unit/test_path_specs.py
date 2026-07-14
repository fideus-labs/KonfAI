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

from konfai.utils.utils import split_path_spec


def test_split_path_spec_supports_unix_style_dataset_specs() -> None:
    assert split_path_spec("./Dataset") == ("./Dataset", None, "mha")
    assert split_path_spec("./Dataset:mha") == ("./Dataset", None, "mha")
    assert split_path_spec("./Dataset:a:mha", allowed_flags={"a", "i"}) == ("./Dataset", "a", "mha")
    assert split_path_spec("./Predictions/TRAIN_01/Dataset:i:mha", allowed_flags={"a", "i"}) == (
        "./Predictions/TRAIN_01/Dataset",
        "i",
        "mha",
    )


def test_split_path_spec_supports_windows_paths_without_breaking_drive_letters() -> None:
    assert split_path_spec(r"C:\Dataset") == (r"C:\Dataset", None, "mha")
    assert split_path_spec(r"C:\Dataset:mha") == (r"C:\Dataset", None, "mha")
    assert split_path_spec(r"C:\Dataset:a:mha", allowed_flags={"a", "i"}) == (r"C:\Dataset", "a", "mha")
