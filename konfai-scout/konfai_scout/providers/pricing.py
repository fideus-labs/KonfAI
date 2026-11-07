# SPDX-License-Identifier: Apache-2.0
"""What a request costs, in US dollars per million tokens, for the models whose public prices are known. A model
not listed costs ``None``: the report then says the cost is unknown rather than guessing."""

from __future__ import annotations

from .base import Usage

#: input, output, cache read (per million tokens); cache writes are billed at 1.25 x input.
PRICES: dict[str, dict[str, float]] = {
    "claude-fable-5-1": {"input": 10.0, "output": 50.0, "cache_read": 0.25},
    "claude-fable-5": {"input": 10.0, "output": 50.0, "cache_read": 0.25},
    "claude-opus-5-5": {"input": 4.0, "output": 20.0, "cache_read": 0.20},
    "claude-opus-5": {"input": 5.0, "output": 25.0, "cache_read": 0.50},
    "claude-opus-4-8": {"input": 5.0, "output": 25.0, "cache_read": 0.50},
    "claude-opus-4-7": {"input": 5.0, "output": 25.0, "cache_read": 0.50},
    "claude-opus-4-6": {"input": 5.0, "output": 25.0, "cache_read": 0.50},
    "claude-sonnet-5-5": {"input": 2.0, "output": 10.0, "cache_read": 0.20},
    "claude-sonnet-5": {"input": 2.0, "output": 10.0, "cache_read": 0.20},
    "claude-sonnet-4-6": {"input": 3.0, "output": 15.0, "cache_read": 0.30},
    "claude-haiku-5-5": {"input": 0.10, "output": 0.50, "cache_read": 0.01},
    "claude-haiku-4-5": {"input": 1.0, "output": 5.0, "cache_read": 0.10},
}


def price_of(model: str, extra: dict[str, dict[str, float]] | None = None) -> dict[str, float] | None:
    table = {**PRICES, **(extra or {})}
    if model in table:
        return table[model]
    for name, price in table.items():  # a dated or suffixed id of a listed model
        if model.startswith(name):
            return price
    return None


def cost_usd(usage: Usage, price: dict[str, float] | None) -> float | None:
    if price is None:
        return None
    million = 1_000_000
    cache_write = price.get("cache_write", price["input"] * 1.25)
    return (
        usage.input_tokens * price["input"]
        + usage.output_tokens * price["output"]
        + usage.cache_read_tokens * price.get("cache_read", price["input"])
        + usage.cache_write_tokens * cache_write
    ) / million
