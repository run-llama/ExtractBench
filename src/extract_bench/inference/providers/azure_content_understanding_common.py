"""Shared helpers for the Azure Content Understanding providers.

Content Understanding exposes one analyzer API that both EXTRACT
(``providers/extract/azure_content_understanding.py``) and PARSE
(``providers/parse/azure_content_understanding.py``) sit on top of, with a
single billing model. Everything that must not drift between the two —
endpoint normalization, the analyze/poll HTTP contract and the usage → cost
translation — lives here.

Pricing defaults are the published Azure retail rates for the
``Azure Content Understanding`` meters (uniform across regions as of
2026-07-30, verified via https://prices.azure.com/api/retail/prices):

============================================  ===================
Meter                                          Rate
============================================  ===================
``Doc Content Extraction Min Pages``           $0.01 / 1K pages
``Doc Content Extraction Basic Pages``         $1.00 / 1K pages
``Doc Content Extraction Standard Pages``      $5.00 / 1K pages
``Std Contextualization Tokens``               $0.001 / 1K tokens
============================================  ===================

LLM/embedding tokens are billed on the caller's Microsoft Foundry deployment
rather than by Content Understanding, so completion pricing is a separate
table keyed by model name. Those defaults are the *global* deployment rates
(gpt-4.1 at $2.00/$8.00 per 1M in/out); data-zone and regional deployments run
10-37% higher, so a pipeline on one should override
``completion_model_pricing_per_1m``.
"""

import time
from collections.abc import Callable, Collection, Mapping
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from typing import Any
from urllib.parse import urlparse, urlunparse

import requests

from extract_bench.inference.providers.base import (
    ProviderPermanentError,
    ProviderRateLimitError,
    ProviderTransientError,
)

DEFAULT_API_VERSION = "2025-11-01"
TERMINAL_STATUSES = {"succeeded", "failed", "canceled"}
TRANSIENT_STATUS_CODES = {408, 425, 429, 500, 502, 503, 504}
DEFAULT_POLL_INTERVAL_S = 10.0
# Wall-clock wait for one analyze/create operation. Retry-After sleeps count
# as elapsed waiting rather than consuming a fixed number of polls.
DEFAULT_POLL_TIMEOUT_S = 6600.0
MAX_RETRY_AFTER_S = 60.0
DEFAULT_RATE_LIMIT_RETRIES = 5

DEFAULT_DOCUMENT_PAGE_MINIMAL_PRICE_PER_1000 = 0.01
DEFAULT_DOCUMENT_PAGE_BASIC_PRICE_PER_1000 = 1.0
DEFAULT_DOCUMENT_PAGE_STANDARD_PRICE_PER_1000 = 5.0
DEFAULT_CONTEXTUALIZATION_PRICE_PER_1M = 1.0
DEFAULT_COMPLETION_MODEL_PRICING_PER_1M: dict[str, tuple[float, float]] = {
    "gpt-4.1": (2.0, 8.0),
    "gpt-4.1-mini": (0.4, 1.6),
}


def public_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Pipeline config minus credentials and transport knobs, safe to persist."""
    hidden = {"api_key", "endpoint", "timeout", "poll_interval", "poll_timeout"}
    return {key: value for key, value in config.items() if key not in hidden}


def content_understanding_endpoint(endpoint: str) -> str:
    """Strip any path/query from a resource endpoint, keeping scheme + host."""
    parsed = urlparse(endpoint)
    if parsed.hostname is None:
        return endpoint
    return urlunparse(parsed._replace(path="", params="", query="", fragment=""))


def apply_usage_cost_fields(raw_output: dict[str, Any], config: Mapping[str, Any]) -> None:
    """Attach dashboard cost/token fields derived from the analyze ``usage`` block."""
    usage = raw_output.get("usage")
    if not isinstance(usage, Mapping):
        return

    breakdown = usage_cost_breakdown(usage, config)
    if len(breakdown) == 0:
        return

    total_cost = float(breakdown["total_cost_usd"])
    raw_output["acu_cost_breakdown"] = breakdown
    raw_output["cost_usd"] = total_cost
    raw_output["input_cost_usd"] = breakdown["llm_input_cost_usd"]
    raw_output["output_and_thinking_cost_usd"] = breakdown["llm_output_cost_usd"]
    raw_output["acu_content_extraction_cost_usd"] = breakdown["content_extraction_cost_usd"]
    raw_output["acu_contextualization_cost_usd"] = breakdown["contextualization_cost_usd"]
    page_count = usage_document_pages(usage)
    if page_count > 0:
        raw_output["num_pages"] = page_count
        # The page is the billing unit for content extraction, so report it
        # alongside cost the way the metered pipelines do.
        raw_output["num_pages_billed"] = page_count
        raw_output["cost_per_page_usd"] = total_cost / page_count

    tokens = usage.get("tokens")
    if isinstance(tokens, Mapping):
        raw_output["input_tokens"] = usage_token_count(tokens, "-input")
        raw_output["output_tokens"] = usage_token_count(tokens, "-output")
        raw_output["total_tokens"] = raw_output["input_tokens"] + raw_output["output_tokens"]


def usage_cost_breakdown(usage: Mapping[str, Any], config: Mapping[str, Any]) -> dict[str, float]:
    """Split the analyze ``usage`` block into per-meter costs.

    Content extraction is metered per page at one of three levels (the service
    charges for the work it actually performed, not the analyzer requested),
    contextualization is a fixed per-page token charge on generative features,
    and LLM tokens are billed against the caller's Foundry deployment.
    """
    page_standard = number(usage.get("documentPagesStandard"))
    page_basic = number(usage.get("documentPagesBasic"))
    page_minimal = number(usage.get("documentPagesMinimal"))
    page_standard_rate = float_config(
        config,
        "document_page_standard_price_per_1000",
        DEFAULT_DOCUMENT_PAGE_STANDARD_PRICE_PER_1000,
    )
    page_basic_rate = float_config(
        config,
        "document_page_basic_price_per_1000",
        DEFAULT_DOCUMENT_PAGE_BASIC_PRICE_PER_1000,
    )
    page_minimal_rate = float_config(
        config,
        "document_page_minimal_price_per_1000",
        DEFAULT_DOCUMENT_PAGE_MINIMAL_PRICE_PER_1000,
    )
    content_extraction_cost = (
        page_standard / 1000.0 * page_standard_rate
        + page_basic / 1000.0 * page_basic_rate
        + page_minimal / 1000.0 * page_minimal_rate
    )

    contextualization_tokens = number(usage.get("contextualizationTokens", usage.get("contextualizationToken")))
    contextualization_rate = float_config(
        config,
        "contextualization_price_per_1m",
        DEFAULT_CONTEXTUALIZATION_PRICE_PER_1M,
    )
    contextualization_cost = contextualization_tokens / 1_000_000.0 * contextualization_rate

    input_tokens = 0.0
    output_tokens = 0.0
    input_cost = 0.0
    output_cost = 0.0
    tokens = usage.get("tokens")
    if isinstance(tokens, Mapping):
        for key, value in tokens.items():
            token_count = number(value)
            if token_count <= 0:
                continue
            token_key = str(key)
            if token_key.endswith("-input"):
                model = token_key[: -len("-input")]
                input_rate, _ = completion_model_rates(model, config)
                input_tokens += token_count
                input_cost += token_count / 1_000_000.0 * input_rate
            elif token_key.endswith("-output"):
                model = token_key[: -len("-output")]
                _, output_rate = completion_model_rates(model, config)
                output_tokens += token_count
                output_cost += token_count / 1_000_000.0 * output_rate

    total_cost = content_extraction_cost + contextualization_cost + input_cost + output_cost
    return {
        "total_cost_usd": total_cost,
        "content_extraction_cost_usd": content_extraction_cost,
        "contextualization_cost_usd": contextualization_cost,
        "llm_input_cost_usd": input_cost,
        "llm_output_cost_usd": output_cost,
        "document_pages_standard": page_standard,
        "document_pages_basic": page_basic,
        "document_pages_minimal": page_minimal,
        "contextualization_tokens": contextualization_tokens,
        "llm_input_tokens": input_tokens,
        "llm_output_tokens": output_tokens,
    }


def usage_document_pages(usage: Mapping[str, Any]) -> float:
    return (
        number(usage.get("documentPagesStandard"))
        + number(usage.get("documentPagesBasic"))
        + number(usage.get("documentPagesMinimal"))
    )


def usage_token_count(tokens: Mapping[str, Any], suffix: str) -> int:
    return int(sum(number(value) for key, value in tokens.items() if str(key).endswith(suffix)))


def completion_model_rates(model: str, config: Mapping[str, Any]) -> tuple[float, float]:
    pricing = config.get("completion_model_pricing_per_1m")
    if isinstance(pricing, Mapping):
        value = pricing.get(model)
        if isinstance(value, Mapping):
            return number(value.get("input")), number(value.get("output"))
        if isinstance(value, (list, tuple)) and len(value) >= 2:
            return number(value[0]), number(value[1])

    for prefix, rates in sorted(DEFAULT_COMPLETION_MODEL_PRICING_PER_1M.items(), key=lambda item: -len(item[0])):
        if model.startswith(prefix):
            return rates
    return (0.0, 0.0)


def float_config(config: Mapping[str, Any], key: str, default: float) -> float:
    value = config.get(key)
    return number(value) if value is not None else default


def number(value: Any) -> float:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return 0.0
    return 0.0


def retry_after_seconds(response: requests.Response, default: float) -> float:
    """Seconds to wait from a 429 ``Retry-After`` header, capped at ``MAX_RETRY_AFTER_S``."""
    raw = response.headers.get("Retry-After")
    wait = default
    if raw is not None and str(raw).strip() != "":
        try:
            wait = float(raw)
        except ValueError:
            try:
                when = parsedate_to_datetime(str(raw))
                if when.tzinfo is None:
                    when = when.replace(tzinfo=UTC)
                wait = (when - datetime.now(UTC)).total_seconds()
            except (TypeError, ValueError, OverflowError, IndexError):
                wait = default
    return max(0.0, min(wait, MAX_RETRY_AFTER_S))


def poll_deadline(timeout_s: float) -> float:
    return time.monotonic() + timeout_s


def sleep_for_poll_interval(deadline: float, interval: float) -> bool:
    """Sleep until the next poll. False means the poll timeout has elapsed."""
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        return False
    time.sleep(min(interval, remaining))
    return deadline - time.monotonic() > 0


def content_understanding_request(
    session: requests.Session,
    method: str,
    url: str,
    *,
    context: str,
    timeout: float,
    default_retry_after_s: float = DEFAULT_POLL_INTERVAL_S,
    max_rate_limit_retries: int = DEFAULT_RATE_LIMIT_RETRIES,
    sleep: Callable[[float], None] = time.sleep,
    accepted_status_codes: Collection[int] = (),
    **kwargs: Any,
) -> requests.Response:
    """Send one ACU HTTP call, sleeping on HTTP 429 + Retry-After instead of abandoning the job."""
    last_rate_limited: requests.Response | None = None
    for attempt in range(max_rate_limit_retries + 1):
        try:
            response = session.request(method, url, timeout=timeout, **kwargs)
        except requests.RequestException as exc:
            raise ProviderTransientError(f"Azure Content Understanding {context} request failed: {exc}") from exc
        if response.status_code in accepted_status_codes:
            return response
        if response.status_code == 429:
            last_rate_limited = response
            if attempt >= max_rate_limit_retries:
                break
            sleep(retry_after_seconds(response, default_retry_after_s))
            continue
        if response.status_code in TRANSIENT_STATUS_CODES:
            raise ProviderTransientError(
                f"Azure Content Understanding transient error during {context}: {response.status_code} {response.text}"
            )
        if response.status_code >= 400:
            raise ProviderPermanentError(
                f"Azure Content Understanding error during {context}: {response.status_code} {response.text}"
            )
        return response
    assert last_rate_limited is not None
    raise ProviderRateLimitError(f"Azure Content Understanding rate limited during {context}: {last_rate_limited.text}")


_RATE_LIMIT_CODES = frozenset({"429", "ratelimit", "ratelimited", "toomanyrequests"})


def raise_operation_failure(body: Mapping[str, Any], *, context: str) -> None:
    """Classify a terminal failed ACU operation.

    HTTP 429s are already ``ProviderRateLimitError``. Azure often instead finishes
    the operation as ``status: Failed`` and nests a RateLimit / 429 under
    ``error.innererror``. Classify these consistently with HTTP rate limits.
    """
    if _is_rate_limit_error(body):
        raise ProviderRateLimitError(f"Azure Content Understanding rate limited during {context}: {body}")
    raise ProviderPermanentError(f"Azure Content Understanding {context} failed: {body}")


def _is_rate_limit_error(value: Any) -> bool:
    if isinstance(value, Mapping):
        if _is_rate_limit_code(value.get("code")):
            return True
        message = value.get("message")
        if isinstance(message, str) and "ratelimit" in _alnum_lower(message):
            return True
        return any(_is_rate_limit_error(value.get(key)) for key in ("error", "innererror", "innerError", "details"))
    if isinstance(value, list):
        return any(_is_rate_limit_error(item) for item in value)
    return False


def _is_rate_limit_code(code: Any) -> bool:
    if isinstance(code, bool):
        return False
    if isinstance(code, int):
        return code == 429
    if isinstance(code, str):
        return _alnum_lower(code) in _RATE_LIMIT_CODES
    return False


def _alnum_lower(text: str) -> str:
    return "".join(character for character in text.lower() if character.isalnum())


def response_json(response: requests.Response) -> dict[str, Any]:
    try:
        payload = response.json()
    except ValueError:
        return {}
    return payload if isinstance(payload, dict) else {"value": payload}


def analysis_result(raw_output: Mapping[str, Any]) -> dict[str, Any]:
    """The ``result`` payload of a completed analyze operation."""
    result = raw_output.get("result", raw_output)
    if isinstance(result, Mapping):
        return dict(result)
    return {}


def first_document_content(result: Mapping[str, Any]) -> Mapping[str, Any]:
    contents = result.get("contents")
    if isinstance(contents, list):
        for content in contents:
            if isinstance(content, Mapping) and content.get("kind") == "document":
                return content
        for content in contents:
            if isinstance(content, Mapping):
                return content
    return {}


def document_contents(result: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    """Every document content block, in service order.

    A single-input analyze call returns one block; segmented analyzers and
    multi-input (pro mode) calls return several.
    """
    contents = result.get("contents")
    if not isinstance(contents, list):
        return []
    documents = [c for c in contents if isinstance(c, Mapping) and c.get("kind") == "document"]
    if len(documents) > 0:
        return documents
    return [c for c in contents if isinstance(c, Mapping)]
