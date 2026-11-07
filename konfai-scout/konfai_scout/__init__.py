# SPDX-License-Identifier: Apache-2.0
"""KonfAI Scout: an agent that inspects segmentation, registration and synthesis results visually and reports
localised defects with evidence and KonfAI Eval measurements."""

from __future__ import annotations

from .agent import Inspector, inspect, scout_version
from .case import InspectionCase, Volume
from .config import Budget, ScoutConfig
from .geometry import Grid, WorldBox
from .report import Report, compare, render_markdown

__version__ = scout_version()
__all__ = [
    "Budget",
    "Grid",
    "InspectionCase",
    "Inspector",
    "Report",
    "ScoutConfig",
    "Volume",
    "WorldBox",
    "compare",
    "inspect",
    "render_markdown",
    "__version__",
]
