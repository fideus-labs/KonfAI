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


"""Which cases a run reads."""

import os
from collections.abc import Sequence

import numpy as np

from konfai.utils.dataset import Attribute


class Subset:
    def __init__(
        self,
        subset: str | list[int] | list[str] | None = None,
        shuffle: bool = True,
        shuffle_window: int | None = None,
    ) -> None:
        self.subset = subset
        self.shuffle = shuffle
        self.shuffle_window = shuffle_window

    @staticmethod
    def _read_names_from_file(filename: str) -> list[str]:
        with open(filename) as f:
            return [name.strip() for name in f if name.strip()]

    def requires_infos(self) -> bool:
        """Return whether this subset implementation needs per-sample metadata."""
        return self.__class__.__call__ is not Subset.__call__

    @staticmethod
    def _is_slice_selector(subset: str) -> bool:
        start, sep, end = subset.partition(":")
        if sep == "":
            return False
        return start.lstrip("-").isdigit() and end.lstrip("-").isdigit()

    def _resolve_selector(self, subset: str | int, names: list[str]) -> tuple[set[int], bool]:
        size = len(names)
        name_to_index = {name: i for i, name in enumerate(names)}

        if isinstance(subset, int):
            return {subset}, False
        if subset.startswith("~"):
            excluded = subset[1:]
            if os.path.exists(excluded):
                exclude_names = set(self._read_names_from_file(excluded))
                return {i for i, name in enumerate(names) if name in exclude_names}, True
            if excluded in name_to_index:
                return {name_to_index[excluded]}, True
            return set(), True
        if os.path.exists(subset):
            selected_names = set(self._read_names_from_file(subset))
            return {i for i, name in enumerate(names) if name in selected_names}, False
        if self._is_slice_selector(subset):
            start, _, end = subset.partition(":")
            # Negative bounds count from the end, Python-slice style: '0:-2' keeps all but the last two.
            bounds = [int(bound) + size if int(bound) < 0 else int(bound) for bound in (start, end)]
            r = np.clip(np.asarray(bounds), 0, size)
            return set(range(int(r[0]), int(r[1]))), False
        if subset in name_to_index:
            return {name_to_index[subset]}, False
        return set(), False

    def _resolve_selectors(self, selectors: Sequence[str | int], names: list[str]) -> list[int]:
        """The positions ``selectors`` keep in ``names``: inclusions united, exclusions subtracted,
        and a list of only exclusions defined against the full list."""
        include_index: set[int] = set()
        exclude_index: set[int] = set()
        has_include = False
        for selector in selectors:
            resolved_index, is_exclusion = self._resolve_selector(selector, names)
            if is_exclusion:
                exclude_index.update(resolved_index)
            else:
                include_index.update(resolved_index)
                has_include = True
        index_set = include_index if has_include else set(range(len(names)))
        return sorted(index_set.difference(exclude_index))

    @staticmethod
    def _excludes(selector: str | int) -> bool:
        return isinstance(selector, str) and selector.startswith("~")

    def _included_names(self, selector: str | int) -> set[str] | None:
        """The cases ``selector`` keeps, or ``None`` when only the cohort can say which they are.
        Read in ``_resolve_selector``'s order, a file before a slice, so the walk asks for what the
        selection then keeps."""
        if isinstance(selector, int):
            return None  # a position, and positions are defined against the full sorted list
        if os.path.exists(selector):
            return set(self._read_names_from_file(selector))
        if self._is_slice_selector(selector):
            return None
        return {selector}

    def required_names(self) -> set[str] | None:
        """The cases this subset keeps, when it can name them, or ``None`` when it needs the cohort.

        A root is asked for these instead of for everything it holds. Exclusions do not appear:
        they only remove from what the inclusions brought, and a subset that ONLY excludes is
        defined against the full list, so it asks for it.
        """
        if self.requires_infos():
            return None  # a subclass selecting on geometry: it is handed every case, and picks
        selectors = self.subset if isinstance(self.subset, list) else [self.subset]
        included = [self._included_names(s) for s in selectors if s is not None and not self._excludes(s)]
        return None if not included or None in included else set().union(*included)  # type: ignore[arg-type]

    def __call__(self, names: list[str], infos: dict[str, tuple[list[int], Attribute]]) -> set[str]:
        names = sorted(names)

        if self.subset is None:
            index = list(range(0, len(names)))
        elif isinstance(self.subset, list):
            # An empty list selects nothing: only a list of ONLY exclusions reads as "everything but".
            index = self._resolve_selectors(self.subset, names) if self.subset else []
        else:
            index = self._resolve_selectors([self.subset], names)
        return {names[i] for i in index}

    def __str__(self):
        return f"Subset : {self.subset} shuffle : {self.shuffle} shuffle_window : {self.shuffle_window}"


class PredictionSubset(Subset):
    def __init__(self, subset: str | list[int] | list[str] | None = None) -> None:
        super().__init__(subset, False, None)
