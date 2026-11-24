# SPDX-License-Identifier: Apache-2.0
"""KonfAI Scout: an agent that inspects segmentation, registration and synthesis results visually and reports
localised defects with evidence and KonfAI Eval measurements."""

from __future__ import annotations

from .agent import Inspector, inspect, scout_version
from .case import InspectionCase, Volume
from .config import Budget, ScoutConfig
from .report import Report, export, render_markdown

__version__ = scout_version()
__all__ = [
    "Budget",
    "InspectionCase",
    "Inspector",
    "Report",
    "ScoutConfig",
    "Volume",
    "__version__",
    "export",
    "inspect",
    "render_markdown",
]
