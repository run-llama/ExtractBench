"""Shared Gemini API pricing helpers."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, date, datetime
from typing import Any

# Gemini pricing: USD per million tokens (input, output).
# Thinking tokens are billed at the output token rate.
# Source: https://ai.google.dev/gemini-api/docs/pricing (2026-03-25)
GEMINI_PRICING_PER_MILLION: dict[str, tuple[float, float]] = {
    # model-prefix: (input_per_M, output_per_M)
    "gemini-3.7-flash": (0.75, 3.75),
    "gemini-3.6-flash": (1.50, 7.50),
    "gemini-3.5-flash": (1.50, 9.00),
    "gemini-3-flash": (0.50, 3.00),
    "gemini-3.1-flash-lite": (0.25, 1.50),
    "gemini-3.5-flash-lite": (0.30, 2.50),
    "gemini-2.5-flash": (0.30, 2.50),
    "gemini-2.5-flash-lite": (0.10, 0.40),
    "gemini-2.0-flash": (0.10, 0.40),
    "gemini-2.0-flash-lite": (0.075, 0.30),
    "gemini-2.5-pro": (1.25, 10.00),
    "gemini-3.1-pro": (2.00, 12.00),
    # Legacy 1.5 series (<=128k prompt tier; Gemini 1.5 prices were tiered above 128k)
    "gemini-1.5-pro": (1.25, 5.00),
    "gemini-1.5-flash": (0.075, 0.30),
    "gemini-1.5-flash-8b": (0.0375, 0.15),
}

# Gemini context caching pricing: USD per million tokens / per million token-hours.
# Source: https://ai.google.dev/gemini-api/docs/pricing (2026-04-05)
# NOTE: gemini-3.5-flash cache-hit (0.15) follows the table's consistent 0.1x-input
# convention (every flash entry below: cache_hit == 0.1 x input rate); confirm against
# the published cache pricing if billing exactness on cached tokens matters.
GEMINI_CONTEXT_CACHE_PRICING_PER_MILLION: dict[str, tuple[float, float]] = {
    # model-prefix: (cache_hit_per_M, storage_per_M_token_hour)
    "gemini-3.7-flash": (0.075, 0.50),
    "gemini-3.6-flash": (0.15, 1.00),
    "gemini-3.5-flash": (0.15, 1.00),
    "gemini-3-flash": (0.05, 1.00),
    "gemini-3.1-flash-lite": (0.025, 1.00),
    "gemini-3.5-flash-lite": (0.03, 1.00),
    "gemini-2.5-flash": (0.03, 1.00),
    "gemini-2.5-flash-lite": (0.01, 1.00),
    "gemini-2.5-pro": (0.125, 4.50),
    "gemini-3.1-pro": (0.20, 4.50),
}


def gemini_pricing_per_million(model: str) -> tuple[float, float]:
    """Return (input_rate, output_rate) in USD per million tokens."""

    return _longest_prefix_match(model, GEMINI_PRICING_PER_MILLION, default=(0.0, 0.0))


def gemini_context_cache_pricing_per_million(model: str) -> tuple[float, float]:
    """Return (cache_hit_rate, storage_rate) in USD per million tokens."""

    return _longest_prefix_match(model, GEMINI_CONTEXT_CACHE_PRICING_PER_MILLION, default=(0.0, 0.0))


def _longest_prefix_match(
    model: str,
    pricing: dict[str, tuple[float, float]],
    *,
    default: tuple[float, float],
) -> tuple[float, float]:
    matches = [(prefix, rates) for prefix, rates in pricing.items() if model.startswith(prefix)]
    return max(matches, key=lambda item: len(item[0]))[1] if matches else default


def cost_from_request_usage(model: str, usage: Mapping[str, Any], *, as_of: date | None = None) -> float | None:
    """Price native CLI requests, including hidden thinking; unsupported accounting stays unknown.

    Argon introductory rates end 2026-11-30:
    https://ai.google.dev/gemini-api/docs/latest-model
    The CLI's Flash and Pro helpers use the shared standard Gemini pricing maps.
    """
    if any(type(usage.get(key)) is not int or usage[key] < 0 for key in ("promptTokenCount", "totalTokenCount")):
        return None
    prompt, total = usage["promptTokenCount"], usage["totalTokenCount"]
    cached = usage.get("cachedContentTokenCount", 0)
    if type(cached) is not int or not 0 <= cached <= prompt or total < prompt:
        return None
    if usage.get("toolUsePromptTokenCount", 0) != 0:
        return None
    if model.startswith("gemini-4-argon"):
        day = datetime.now(UTC).date() if as_of is None else as_of
        input_rate, output_rate, cache_rate = (2.0, 10.0, 0.1) if day < date(2026, 12, 1) else (4.0, 20.0, 0.2)
    elif model.startswith(("gemini-3-flash-preview", "gemini-3.1-pro-preview")):
        input_rate, output_rate = gemini_pricing_per_million(model)
        cache_rate, _ = gemini_context_cache_pricing_per_million(model)
        if model.startswith("gemini-3.1-pro-preview") and prompt > 200_000:
            input_rate, output_rate, cache_rate = (4.0, 18.0, 0.4)
    else:
        return None
    return float(((prompt - cached) * input_rate + cached * cache_rate + (total - prompt) * output_rate) / 1e6)
