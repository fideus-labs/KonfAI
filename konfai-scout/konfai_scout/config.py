# SPDX-License-Identifier: Apache-2.0
"""What an inspection is configured with: the provider, the mode, the budget, the privacy gate, the criteria."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

MODES = ("active", "fixed")
PROVIDERS = ("anthropic", "openai", "claude-code", "codex", "mock")
DEFAULT_MODELS = {
    "anthropic": "claude-opus-5-5",
    "openai": "gpt-5",
    "claude-code": "claude-opus-5-5",  # Claude under the user's Claude Code login, through the Claude Agent SDK
    "codex": "default",  # the Codex CLI's own default model under the user's ChatGPT login; the report names it
    "mock": "heuristic",
}

#: How far an inspection looks and what it may spend: ``quick`` for a cheap first pass, ``deep`` for a case that
#: matters. A depth changes the defaults only: a field set to anything but its standard value keeps that value.
DEPTHS: dict[str, dict[str, object]] = {
    "quick": {"effort": "low", "budget": {"max_tool_calls": 10}},
    "standard": {},
    "deep": {"effort": "high", "second_reading": True, "budget": {"max_tool_calls": 40}},
}


@dataclass
class Budget:
    """What one inspection may spend. ``None`` lifts a limit. ``max_tool_calls`` counts the calls that explore
    (views, montages, measurements, statistics): a ``note`` or the ``conclude`` that records them is never refused."""

    max_tool_calls: int = 25
    max_images: int = 60
    max_seconds: float | None = 900.0
    max_usd: float | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass
class ScoutConfig:
    provider: str = "mock"
    model: str | None = None
    #: ``quick``, ``standard`` or ``deep`` (``DEPTHS``): the budget, the effort and the second reading in one word.
    depth: str = "standard"
    #: An OpenAI-compatible server (vLLM, Ollama, LM Studio) for ``provider: openai``; the official API when unset.
    base_url: str | None = None
    #: The environment variable holding the key; the SDK's own default when unset.
    api_key_env: str | None = None
    #: Claude's effort (low, medium, high, xhigh, max); the model's default when unset.
    effort: str | None = None
    max_output_tokens: int = 16000
    mode: str = "active"
    budget: Budget = field(default_factory=Budget)
    #: Images leave this machine only when this is True. A provider on a loopback address is local; set
    #: ``local_provider`` for another address you know to be local.
    allow_cloud_images: bool = False
    local_provider: bool = False
    #: Longest edge of a rendered view, in pixels; what a model is handed.
    max_px: int = 768
    #: Slices per plane in the coverage montage both modes start from.
    coverage_slices: int = 9
    #: Criteria per measurement kind, KonfAI's names and arguments; ``"labels": "auto"`` becomes the labels present.
    criteria: dict[str, dict[str, dict[str, object]]] = field(default_factory=dict)
    #: When a measured value is an error, by KonfAI's result name: ``{"Dice": {"below": 0.8}, "MAE": {"above": 100}}``,
    #: added to ``DEFAULT_THRESHOLDS``. A value confirms an error only past its threshold; a name without one, never.
    thresholds: dict[str, dict[str, float]] = field(default_factory=dict)
    #: Prices per million tokens, ``{model: {"input": x, "output": y, "cache_read": z}}``, added to the defaults.
    prices: dict[str, dict[str, float]] = field(default_factory=dict)
    provider_retries: int = 3
    retry_wait_seconds: float = 5.0
    #: Ask the Claude API to continue on a fallback model after a safety refusal. Off: a report then names the model
    #: that produced every word of it.
    anthropic_fallbacks: bool = False
    #: Keep KonfAI's evaluation workspaces of every region measurement beside the report.
    keep_evaluations: bool = True
    #: The model is not told the case's name, its notes (the method, the dataset) or its prior evaluation: it judges
    #: the images and its own measurements. Off (``--no-blind``) for a study of what that knowledge changes.
    blind: bool = True
    #: Free text appended to the system prompt (a site's own rules, a study's instructions).
    extra_instructions: str = ""
    #: After the inspection, a fresh session of the same model re-reads every error observation blind: crops of its
    #: box in a few modes and the claim, without the method, the history or the measurements. Its answer (visible,
    #: not visible, unsure) is recorded on the observation; nothing is removed.
    second_reading: bool = False
    #: The deterministic survey of where to look; off only to measure what it brings.
    survey: bool = True
    #: The tools that measure a region (`measure`, `region_stats`); off only to measure what they bring.
    measure: bool = True

    def __post_init__(self) -> None:
        if self.mode not in MODES:
            raise ValueError(f"mode is one of {MODES}, not {self.mode!r}")
        if self.provider not in PROVIDERS:
            raise ValueError(f"provider is one of {PROVIDERS}, not {self.provider!r}")
        if self.depth not in DEPTHS:
            raise ValueError(f"depth is one of {tuple(DEPTHS)}, not {self.depth!r}")
        if isinstance(self.budget, dict):
            self.budget = Budget(**self.budget)
        if self.model is None:
            self.model = DEFAULT_MODELS[self.provider]
        standard = ScoutConfig.__dataclass_fields__
        for key, value in DEPTHS[self.depth].items():
            if key == "budget":
                for limit, amount in value.items():  # type: ignore[attr-defined]
                    if getattr(self.budget, limit) == getattr(Budget(), limit):
                        setattr(self.budget, limit, amount)
            elif getattr(self, key) == standard[key].default:
                setattr(self, key, value)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @staticmethod
    def from_dict(data: dict[str, object]) -> ScoutConfig:
        known = {k: v for k, v in data.items() if k in ScoutConfig.__dataclass_fields__}
        return ScoutConfig(**known)  # type: ignore[arg-type]

    @staticmethod
    def load(path: Path | str) -> ScoutConfig:
        return ScoutConfig.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
