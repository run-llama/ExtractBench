"""Shared helper to build operational RunStat entries from an InferenceResult."""

import json
import math
import warnings
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from extract_bench.inference.cost_contract import (
    ATTEMPT_ACCOUNTING,
    COST_COMPLETE,
    COST_PER_PAGE_USD,
    COST_USD,
    KNOWN_COST_USD,
    RAW_COST_COMPLETE,
)
from extract_bench.schemas.evaluation import RunStat
from extract_bench.schemas.pipeline_io import InferenceResult
from extract_bench.schemas.product import ProductType

COST_STAT_NAMES = {COST_USD, KNOWN_COST_USD, COST_PER_PAGE_USD, COST_COMPLETE}

# Stat keys to look for in raw_output, with their units
_RAW_OUTPUT_STATS: list[tuple[str, str]] = [
    ("credits_used", "credits"),
    # Credit-billed pipelines (LlamaExtract) report their charge in credits and
    # bill the extract leg against `num_pages_billed` — the *effective* billed
    # pages, which a large schema pushes above the document's page count — while
    # the parse leg bills against pages parsed. Carrying both legs separately is
    # what makes a per-page cost reconstructable from the report.
    ("credits_per_page", "credits/page"),
    ("parse_credits", "credits"),
    ("extract_credits", "credits"),
    ("agent_usage_credits", "credits"),
    ("agent_usage_usd", "$"),
    ("total_credits", "credits"),
    ("num_pages_billed", "pages"),
    (COST_USD, "$"),
    (COST_PER_PAGE_USD, "$/page"),
    ("input_cost_usd", "$"),
    ("tool_use_prompt_cost_usd", "$"),
    ("cached_input_cost_usd", "$"),
    ("output_and_thinking_cost_usd", "$"),
    ("cache_storage_cost_usd", "$"),
    # Non-token charges from server-side tool use, and the cache-write term some
    # model families bill separately. Without these the per-term cost breakdown
    # under-sums against cost_usd — by roughly a third on a GPT-5.6 tool run,
    # where the container fee plus cache writes carry that much of the bill.
    ("cache_write_cost_usd", "$"),
    ("container_cost_usd", "$"),
    ("container_time_cost_usd", "$"),
    ("tool_surcharge_usd", "$"),
    ("token_cost_usd", "$"),
    ("acu_content_extraction_cost_usd", "$"),
    ("acu_contextualization_cost_usd", "$"),
    # Token metrics (when available from provider)
    ("input_tokens", "tokens"),
    ("tool_use_prompt_tokens", "tokens"),
    ("cached_content_tokens", "tokens"),
    ("cache_read_tokens", "tokens"),
    ("cache_write_tokens", "tokens"),
    ("output_tokens", "tokens"),
    ("total_tokens", "tokens"),
    ("thinking_tokens", "tokens"),
    ("num_api_calls", "calls"),
    # A tool pipeline where the model never called the tool is otherwise
    # indistinguishable in a report from one where it did.
    ("num_tool_calls", "calls"),
    ("num_containers", "containers"),
    ("num_regions", "regions"),
    ("input_tokens_per_page", "tokens/page"),
    ("tool_use_prompt_tokens_per_page", "tokens/page"),
    ("cached_content_tokens_per_page", "tokens/page"),
    ("output_tokens_per_page", "tokens/page"),
]


def apply_cost_sidecar(raw: dict[str, Any], path: Path) -> None:
    """A damaged optional bill cannot discard a valid prediction or imply full coverage."""
    if not path.exists():
        return
    try:
        accounting = json.loads(path.read_text(encoding="utf-8"))
        known = accounting[KNOWN_COST_USD]
        complete = accounting[COST_COMPLETE]
        if not isinstance(complete, bool):
            raise ValueError("cost_complete must be boolean")
        if known is not None and (
            isinstance(known, bool) or not isinstance(known, (int, float)) or not math.isfinite(known) or known < 0
        ):
            raise ValueError("known_cost_usd must be nonnegative and finite")
    except (OSError, ValueError, KeyError, TypeError) as error:
        warnings.warn(f"Unable to read cost accounting {path}: {error}", stacklevel=2)
        existing = raw.get(ATTEMPT_ACCOUNTING)
        known = existing.get(KNOWN_COST_USD) if isinstance(existing, dict) else raw.get(COST_USD)
        accounting = {KNOWN_COST_USD: known, COST_COMPLETE: False}
    raw[ATTEMPT_ACCOUNTING] = accounting


def build_cost_stats(raw: Mapping[str, Any]) -> list[RunStat]:
    """Publish known charges separately when any attempt has incomplete billing."""
    accounting = raw.get(ATTEMPT_ACCOUNTING)
    if isinstance(accounting, dict):
        known = accounting[KNOWN_COST_USD]
        complete = accounting[COST_COMPLETE]
    else:
        known = raw.get(COST_USD)
        complete = raw.get(RAW_COST_COMPLETE, known is not None)
    complete = known is not None and complete is True
    stats = [RunStat(name=COST_COMPLETE, value=float(complete), unit="ratio")]
    if known is not None:
        stats.append(RunStat(name=KNOWN_COST_USD, value=known, unit="$"))
        if complete:
            stats.append(RunStat(name=COST_USD, value=known, unit="$"))
    if complete:
        pages = raw.get("num_pages")
        per_page = raw.get(COST_PER_PAGE_USD)
        if isinstance(pages, (int, float)) and pages > 0 and (isinstance(accounting, dict) or per_page is None):
            per_page = known / pages
        if per_page is not None:
            stats.append(RunStat(name=COST_PER_PAGE_USD, value=per_page, unit="$/page"))
    return stats


def build_operational_stats(inference_result: InferenceResult) -> list[RunStat]:
    """Build operational stats from an inference result.

    Extracts latency from the dedicated field and cost-related stats
    from raw_output (pre-computed by the provider).
    """
    raw = inference_result.raw_output
    stats = build_cost_stats(raw) if inference_result.product_type == ProductType.EXTRACT else []

    # Latency (from dedicated field)
    if inference_result.latency_in_ms is not None:
        stats.append(RunStat(name="latency_ms", value=float(inference_result.latency_in_ms), unit="ms"))

    # Per-page latency (computed from latency and num_pages)
    num_pages = raw.get("num_pages")
    if inference_result.latency_in_ms is not None and isinstance(num_pages, (int, float)) and num_pages > 0:
        stats.append(
            RunStat(
                name="latency_ms_per_page",
                value=float(inference_result.latency_in_ms) / float(num_pages),
                unit="ms/page",
            )
        )

    # Cost and token stats (from raw_output, pre-computed by provider)
    for key, unit in _RAW_OUTPUT_STATS:
        if inference_result.product_type == ProductType.EXTRACT and key in COST_STAT_NAMES:
            continue
        value = raw.get(key)
        if value is not None:
            stats.append(RunStat(name=key, value=float(value), unit=unit))

    return stats
