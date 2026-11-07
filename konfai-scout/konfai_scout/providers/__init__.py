# SPDX-License-Identifier: Apache-2.0
"""The vision-language models Scout can drive, and the gate every image passes before leaving this machine."""

from __future__ import annotations

from ..config import ScoutConfig
from .base import ImageContent, Provider, ProviderError, ToolCall, ToolResult, ToolSpec, Turn, Usage, is_loopback

__all__ = [
    "ImageContent",
    "Provider",
    "ProviderError",
    "ToolCall",
    "ToolResult",
    "ToolSpec",
    "Turn",
    "Usage",
    "make_provider",
    "check_privacy",
    "is_remote",
]


def make_provider(config: ScoutConfig, client: object = None) -> Provider:
    model = config.model or ""
    if config.provider == "anthropic":
        from .anthropic import AnthropicProvider

        return AnthropicProvider(
            model,
            effort=config.effort,
            max_tokens=config.max_output_tokens,
            api_key_env=config.api_key_env,
            base_url=config.base_url,
            fallbacks=config.anthropic_fallbacks,
            client=client,
        )
    if config.provider == "openai":
        from .openai import OpenAIProvider

        return OpenAIProvider(
            model,
            base_url=config.base_url,
            api_key_env=config.api_key_env,
            local=config.local_provider,
            max_tokens=config.max_output_tokens,
            client=client,
        )
    from .mock import HeuristicProvider

    return HeuristicProvider(model or "heuristic")


def is_remote(config: ScoutConfig) -> bool:
    """Whether the configured provider is a service off this machine, decided from the configuration alone."""
    if config.provider == "mock" or config.local_provider:
        return False
    if config.provider == "openai" and is_loopback(config.base_url):
        return False
    return True


def check_privacy(config: ScoutConfig) -> None:
    """Images leave this machine only when the caller said so: checked before any client or request exists."""
    if is_remote(config) and not config.allow_cloud_images:
        raise ProviderError(
            f"the {config.provider} provider ({config.model}) is a cloud service and this inspection would send "
            "images of the case to it. Set allow_cloud_images (CLI: --allow-cloud-images) to authorise the transfer, "
            "or use a local "
            "server (openai provider with a loopback base_url, or local_provider: true)."
        )
