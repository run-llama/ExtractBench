"""Provider for Datalab EXTRACT.

Uses the Datalab /api/v1/extract endpoint via the official datalab-python-sdk.
Citations are returned per field as block IDs (e.g. ``"{field}_citations": ["...id..."]``)
inside ``extraction_schema_json``; the corresponding bounding boxes live in the
``json`` document tree, where every block carries an ``id`` and a pixel
``bbox = [x1, y1, x2, y2]``. This provider walks both structures to produce
normalized ``FieldCitation`` entries with COCO ``[x, y, w, h]`` bboxes in
``[0, 1]`` page-relative coordinates.
"""

import asyncio
import dataclasses
import json
import math
import os
import time
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any, Literal
from urllib.parse import quote
from uuid import uuid4

import aiohttp
from datalab_sdk import AsyncDatalabClient
from datalab_sdk.models import ExtractOptions
from pypdf import PdfReader

from extract_bench.inference.events import emit_provider_event
from extract_bench.inference.limits import execution_budget
from extract_bench.inference.providers.base import (
    Provider,
    ProviderConfigError,
    ProviderPermanentError,
    ProviderTransientError,
)
from extract_bench.inference.providers.registry import register_provider
from extract_bench.schemas.extract_output import ExtractOutput, FieldCitation
from extract_bench.schemas.pipeline import PipelineSpec
from extract_bench.schemas.pipeline_io import (
    InferenceRequest,
    InferenceResult,
    RawInferenceResult,
)
from extract_bench.schemas.product import ProductType

DATALAB_PARSE_COST_PER_PAGE_USD = {
    "fast": 0.004,
    "balanced": 0.004,
    "accurate": 0.01,
}
DATALAB_EXTRACT_COST_PER_PAGE_USD = {
    "turbo": 0.006,
    "fast": 0.006,
    "balanced": 0.015,
    "accurate": 0.02,
}


ConfidencePollingStatus = Literal["scores_ready", "scoring_failed", "deadline_exceeded", "poll_error"]


class ConfidencePollingStatusValues:
    SCORES_READY: ConfidencePollingStatus = "scores_ready"
    SCORING_FAILED: ConfidencePollingStatus = "scoring_failed"
    DEADLINE_EXCEEDED: ConfidencePollingStatus = "deadline_exceeded"
    POLL_ERROR: ConfidencePollingStatus = "poll_error"


@register_provider("datalab_extract")
class DatalabExtractProvider(Provider):
    """
    Provider for Datalab EXTRACT.

    Submits a document plus JSON schema to ``POST /api/v1/extract`` and polls
    until completion. The completed response contains ``extraction_schema_json``
    (a JSON string of the filled schema, augmented with ``{field}_citations``
    block-id references and optional ``{field}_score`` confidence objects) and
    a ``json`` document tree whose blocks carry pixel ``bbox`` coordinates.
    See https://documentation.datalab.to/docs/recipes/structured-extraction/api-overview .
    """

    def __init__(self, provider_name: str, base_config: dict[str, Any] | None = None):
        """
        Initialize the provider.

        :param provider_name: Name of the provider
        :param base_config: Optional configuration with:
            - ``api_key``: Datalab API key (defaults to DATALAB_API_KEY env var)
            - ``parse_mode``: Document parse mode — ``fast`` / ``balanced`` /
              ``accurate``. Omit to use Datalab’s default; ``mode`` is a legacy alias.
            - ``extraction_mode``: Extraction mode — ``turbo`` / ``fast`` / ``balanced`` /
              ``accurate``. Turbo skips parsing and returns no citations or scores.
              Fast mode also waits for optional confidence scores.
            - ``max_pages``: Maximum number of pages to process (optional).
            - ``page_range``: Comma-separated page ranges (e.g. ``"0-2,4"``).
            - ``output_format``: Conversion output format the extract endpoint
              should also emit — one of ``markdown``, ``json``, ``html``,
              ``chunks``. Defaults to ``json`` so the parse tree (with per-
              block ``id`` + ``bbox``) is available for citation resolution.
            - ``save_checkpoint``: Save a checkpoint after processing.
            - ``skip_cache``: Bypass server-side cache (default: False).
            - ``poll_interval``: Seconds between polls (default: 3).
        """
        super().__init__(provider_name, base_config)

        self._api_key = self.base_config.get("api_key") or os.getenv("DATALAB_API_KEY")
        if not self._api_key:
            raise ProviderConfigError(
                "Datalab API key is required. Set DATALAB_API_KEY environment variable or pass api_key in base_config."
            )

        self._extraction_mode = self.base_config.get("extraction_mode")
        self._parse_mode = (
            None
            if self._extraction_mode == "turbo"
            else self.base_config.get("parse_mode", self.base_config.get("mode"))
        )
        if self._parse_mode is not None and self._parse_mode not in DATALAB_PARSE_COST_PER_PAGE_USD:
            raise ProviderConfigError(
                f"Unsupported Datalab parse mode: {self._parse_mode!r}. "
                f"Expected one of {sorted(DATALAB_PARSE_COST_PER_PAGE_USD)}."
            )
        if self._extraction_mode is not None and self._extraction_mode not in DATALAB_EXTRACT_COST_PER_PAGE_USD:
            raise ProviderConfigError(
                f"Unsupported Datalab extraction mode: {self._extraction_mode!r}. "
                f"Expected one of {sorted(DATALAB_EXTRACT_COST_PER_PAGE_USD)}."
            )
        self._max_pages = self.base_config.get("max_pages")
        self._page_range = self.base_config.get("page_range")
        # Default to "json" so we always have the parse tree available for
        # resolving block-id citations into bounding boxes.
        self._output_format = self.base_config.get("output_format", "json")
        self._save_checkpoint = self.base_config.get("save_checkpoint", False)
        self._skip_cache = self.base_config.get("skip_cache", False)
        if not math.isfinite(self._poll_interval) or self._poll_interval <= 0:
            raise ProviderConfigError("poll_interval must be positive and finite")

    async def _extract_pdf_async(self, pdf_path: str, schema: dict[str, Any]) -> dict[str, Any]:
        """Call Datalab Extract and return the full response as a plain dict."""
        raw_response: dict[str, Any] = {}
        try:
            extraction_started = time.perf_counter()
            pdf_page_count = _read_pdf_page_count(pdf_path)
            options = ExtractOptions(
                page_schema=json.dumps(_adapt_schema_for_datalab(schema)),
                mode=self._parse_mode,
                output_format=self._output_format,
                save_checkpoint=self._save_checkpoint,
                skip_cache=self._skip_cache,
            )
            if self._extraction_mode is not None:
                options.extraction_mode = self._extraction_mode
            if self._max_pages is not None:
                options.max_pages = self._max_pages
            if self._page_range is not None:
                options.page_range = self._page_range

            async with AsyncDatalabClient(api_key=self._api_key) as client:
                # The SDK accepts a poll count, not a deadline. Size its sleep
                # allowance from the remaining budget; run_async owns the cutoff.
                result = await client.extract(
                    pdf_path,
                    options=options,
                    max_polls=math.ceil(execution_budget().remaining() / self._poll_interval),
                    poll_interval=self._poll_interval,  # type: ignore[arg-type]
                )

            extraction_latency_ms = (time.perf_counter() - extraction_started) * 1000
            raw_response = dataclasses.asdict(result)
            # Repeated component IDs replace cumulative charges within an attempt.
            # The SDK's request_id is optional; a local ID keeps separate calls
            # distinct when absent. Choose once and reuse through confidence polls.
            request_id = raw_response.get("request_id")
            component_id = f"datalab/{request_id if request_id is not None else uuid4().hex}"
            page_count = _coerce_positive_page_count(raw_response.get("page_count")) or pdf_page_count
            if page_count is not None:
                raw_response["page_count"] = page_count
            if self._extraction_mode is not None:
                _attach_datalab_extract_pricing(
                    raw_response,
                    page_count=page_count,
                    parse_mode=self._parse_mode,
                    extraction_mode=self._extraction_mode,
                )
            self._record_cost(raw_response, component_id)
            emit_provider_event(
                "stage.completed",
                provider="datalab",
                stage="extract",
                raw_output=raw_response,
            )
            if self._extraction_mode == "fast":
                raw_response = await self._wait_for_confidence(
                    raw_response,
                    base_url=client.base_url,
                    component_id=component_id,
                )

            # Client-observed time includes upload, parsing, and extraction.
            # Confidence waiting is measured separately; neither is server compute time.
            raw_response["extraction_latency_ms"] = extraction_latency_ms
            raw_response["_config"] = {
                "mode": self._parse_mode,
                "parse_mode": self._parse_mode,
                "extraction_mode": self._extraction_mode,
                "max_pages": self._max_pages,
                "page_range": self._page_range,
                "output_format": self._output_format,
                "save_checkpoint": self._save_checkpoint,
                "skip_cache": self._skip_cache,
            }

            return raw_response

        except (ProviderTransientError, ProviderPermanentError):
            raise
        except Exception as e:
            debug_payload = {
                "raw_response": raw_response,
                "cost_accounting": {
                    "known_cost_usd": raw_response.get("cost_usd"),
                    "complete": raw_response.get("_cost_complete", False),
                },
            }
            error_str = str(e).lower()
            transient_keywords = ["timeout", "network", "connection", "503", "502", "504", "429", "rate limit"]
            if any(keyword in error_str for keyword in transient_keywords):
                raise ProviderTransientError(
                    f"Transient error during extraction: {e}",
                    debug_payload=debug_payload,
                ) from e
            raise ProviderPermanentError(
                f"Error during extraction: {e}",
                debug_payload=debug_payload,
            ) from e

    def _record_cost(self, raw_response: dict[str, Any], component_id: str) -> None:
        _promote_api_cost_breakdown(raw_response)
        cost = raw_response.get("cost_usd")
        raw_response["_cost_complete"] = cost is not None
        if cost is not None:
            emit_provider_event(
                "cost.component",
                component_id=component_id,
                cost_usd=cost,
                usage=raw_response.get("cost_breakdown"),
            )

    async def _wait_for_confidence(
        self,
        completed: dict[str, Any],
        *,
        base_url: str,
        component_id: str,
    ) -> dict[str, Any]:
        """Refresh the completed job through the documented REST result endpoint.

        Fast scores arrive after extraction. A failed or interrupted score wait
        leaves the completed extraction usable; it never submits another job.
        """
        confidence_started = time.perf_counter()
        budget = execution_budget()
        status: ConfidencePollingStatus
        polls = 0
        request_id = completed.get("request_id")
        if completed.get("extraction_score_average") is not None:
            status = ConfidencePollingStatusValues.SCORES_READY
        elif completed.get("extraction_scoring_status") == "failed":
            status = ConfidencePollingStatusValues.SCORING_FAILED
        elif not isinstance(request_id, str) or request_id == "":
            raise ValueError("Completed Fast extraction is missing its request_id")
        else:
            url = f"{base_url.rstrip('/')}/api/v1/extract/{quote(request_id, safe='')}"
            async with aiohttp.ClientSession() as session:
                while True:
                    try:
                        await budget.run_async(asyncio.sleep(self._poll_interval))
                        async with session.get(
                            url,
                            headers={"X-API-Key": self._api_key},
                            timeout=aiohttp.ClientTimeout(total=budget.remaining()),
                            allow_redirects=False,
                        ) as response:
                            response.raise_for_status()
                            refreshed = await response.json()
                        budget.remaining()
                        if not isinstance(refreshed, dict):
                            raise ValueError("Confidence response must be an object")
                        polls += 1
                        if refreshed.get("status") != "complete" or refreshed.get("success") is False:
                            raise ValueError(f"Expected completed extraction, got {refreshed.get('status')!r}")
                        # Regional results can be signed downloads. Credentials
                        # belong only on the API request, not on the storage URL.
                        result_url = refreshed.get("result_url")
                        if result_url is not None:
                            if not isinstance(result_url, str) or result_url == "":
                                raise ValueError("Result URL must be a nonempty string")
                            async with session.get(
                                result_url,
                                timeout=aiohttp.ClientTimeout(total=budget.remaining()),
                            ) as response:
                                response.raise_for_status()
                                downloaded = await response.json()
                            budget.remaining()
                            if not isinstance(downloaded, dict):
                                raise ValueError("Downloaded result must be an object")
                            refreshed = {
                                **downloaded,
                                **{key: value for key, value in refreshed.items() if value is not None},
                            }
                        if refreshed.get("success") is False:
                            raise ValueError("Completed extraction result reports success=false")
                        if refreshed.get("extraction_scoring_status") == "failed":
                            status = ConfidencePollingStatusValues.SCORING_FAILED
                            break
                        completed = {
                            **completed,
                            **{key: value for key, value in refreshed.items() if value is not None},
                        }
                        self._record_cost(completed, component_id)
                        if completed.get("extraction_score_average") is not None:
                            status = ConfidencePollingStatusValues.SCORES_READY
                            break
                    except (TimeoutError, asyncio.CancelledError):
                        status = ConfidencePollingStatusValues.DEADLINE_EXCEEDED
                        break
                    except aiohttp.ClientError as error:
                        completed["_confidence_poll_error"] = type(error).__name__
                        status = ConfidencePollingStatusValues.POLL_ERROR
                        break
        completed["confidence_wait_ms"] = (time.perf_counter() - confidence_started) * 1000
        completed["_confidence_polling"] = {"status": status, "polls": polls}
        return completed

    def run_inference(self, pipeline: PipelineSpec, request: InferenceRequest) -> RawInferenceResult:
        if request.product_type != ProductType.EXTRACT:
            raise ProviderPermanentError(
                f"DatalabExtractProvider only supports EXTRACT product type, got {request.product_type}"
            )
        if not request.schema_override:
            raise ProviderPermanentError(
                "schema_override is required for EXTRACT product type. "
                "Provide a JSON schema in InferenceRequest.schema_override"
            )

        started_at = datetime.now()

        file_path = Path(request.source_file_path)
        if not file_path.exists():
            raise ProviderPermanentError(f"File not found: {file_path}")

        try:
            raw_output = asyncio.run(
                execution_budget().run_async(
                    self._extract_pdf_async(pdf_path=str(file_path), schema=request.schema_override)
                )
            )

            completed_at = datetime.now()
            latency_ms = int((completed_at - started_at).total_seconds() * 1000)

            return RawInferenceResult(
                request=request,
                pipeline=pipeline,
                pipeline_name=pipeline.pipeline_name,
                product_type=request.product_type,
                raw_output=raw_output,
                started_at=started_at,
                completed_at=completed_at,
                latency_in_ms=latency_ms,
            )

        except (ProviderPermanentError, ProviderTransientError):
            raise
        except Exception as e:
            raise ProviderPermanentError(f"Unexpected error during inference: {e}") from e

    def normalize(self, raw_result: RawInferenceResult) -> InferenceResult:
        if raw_result.product_type != ProductType.EXTRACT:
            raise ProviderPermanentError(
                f"DatalabExtractProvider only supports EXTRACT product type, got {raw_result.product_type}"
            )

        extraction_json_str = raw_result.raw_output.get("extraction_schema_json")
        extraction_payload: Any = {}
        if isinstance(extraction_json_str, str) and extraction_json_str:
            try:
                extraction_payload = json.loads(extraction_json_str)
            except json.JSONDecodeError:
                extraction_payload = {}

        extracted_data = _strip_citation_and_score_keys(extraction_payload)

        block_lookup = _build_block_lookup(raw_result.raw_output.get("json"))
        citations = _build_field_citations(extraction_payload, block_lookup)

        output = ExtractOutput(
            task_type="extract",
            example_id=raw_result.request.example_id,
            pipeline_name=raw_result.pipeline_name,
            extracted_data=extracted_data if isinstance(extracted_data, (dict, list)) else {},
            field_citations=citations,
        )

        return InferenceResult(
            request=raw_result.request,
            pipeline_name=raw_result.pipeline_name,
            product_type=raw_result.product_type,
            raw_output=raw_result.raw_output,
            output=output,
            started_at=raw_result.started_at,
            completed_at=raw_result.completed_at,
            latency_in_ms=raw_result.latency_in_ms,
        )


# ---------------------------------------------------------------------------
# Pricing helpers
# ---------------------------------------------------------------------------


def _read_pdf_page_count(pdf_path: str) -> int | None:
    try:
        return len(PdfReader(pdf_path).pages)
    except Exception:
        return None


def _coerce_positive_page_count(value: Any) -> int | None:
    if not isinstance(value, (int, float)) or value <= 0:
        return None
    return int(value)


def _attach_datalab_extract_pricing(
    raw_output: dict[str, Any],
    *,
    page_count: int | None,
    parse_mode: str | None,
    extraction_mode: str,
) -> None:
    if page_count is None or page_count <= 0:
        return

    if extraction_mode == "turbo":
        parse_cost_per_page = 0.0
    else:
        if parse_mode is None:
            # Server-selected parsing has no local rate assumption. Use the
            # reported cost breakdown; missing billing data stays unknown.
            return
        parse_cost_per_page = DATALAB_PARSE_COST_PER_PAGE_USD[parse_mode]
    extract_cost_per_page = DATALAB_EXTRACT_COST_PER_PAGE_USD[extraction_mode]
    parse_cost = parse_cost_per_page * page_count
    extract_cost = extract_cost_per_page * page_count
    total_cost = parse_cost + extract_cost

    raw_output.setdefault("page_count", page_count)
    raw_output["parse_cost_usd"] = parse_cost
    raw_output["extract_cost_usd"] = extract_cost
    raw_output["cost_usd"] = total_cost
    raw_output["cost_per_page_usd"] = total_cost / page_count
    raw_output["datalab_pricing"] = {
        "pricing_basis": "datalab_public_pricing_per_1000_pages",
        "parse_mode": parse_mode,
        "parse_cost_per_page_usd": parse_cost_per_page,
        "extraction_mode": extraction_mode,
        "extract_cost_per_page_usd": extract_cost_per_page,
    }


def _promote_api_cost_breakdown(raw_output: dict[str, Any]) -> None:
    # Datalab returns ``cost_breakdown`` as a dict of cent-denominated keys
    # (``final_cost_cents`` post-discount, ``list_cost_cents`` pre-discount).
    cost_breakdown = raw_output.get("cost_breakdown")
    if not isinstance(cost_breakdown, Mapping):
        return
    final_cents = cost_breakdown.get("final_cost_cents")
    if not isinstance(final_cents, (int, float)):
        final_cents = cost_breakdown.get("list_cost_cents")
    if not isinstance(final_cents, (int, float)):
        return

    cost_usd = float(final_cents) / 100.0
    raw_output["cost_usd"] = cost_usd
    page_count = _coerce_positive_page_count(raw_output.get("page_count"))
    if page_count is not None:
        raw_output["cost_per_page_usd"] = cost_usd / page_count


# ---------------------------------------------------------------------------
# Schema adapter
# ---------------------------------------------------------------------------


def _adapt_schema_for_datalab(schema: Any) -> Any:
    """Normalize a JSON Schema so Datalab's /api/v1/extract accepts it.

    Datalab returns HTTP 500 on schemas that use the JSON Schema union form
    ``"type": ["string", "null"]`` (a common way to mark a nullable scalar).
    Empirically a payload identical to one that 500s succeeds once those
    unions are collapsed to a single non-null type. This adapter is a pure
    shape transform: it recursively walks ``properties``, ``items``, and
    object dicts, and replaces any ``type`` list with its first non-null
    member (falling back to ``"string"``). Nothing else is touched, so the
    semantics of the user's schema are preserved.
    """
    if isinstance(schema, Mapping):
        out: dict[str, Any] = {}
        for key, value in schema.items():
            if key == "type" and isinstance(value, list):
                non_null = [t for t in value if t != "null"]
                out[key] = non_null[0] if non_null else "string"
            else:
                out[key] = _adapt_schema_for_datalab(value)
        return out
    if isinstance(schema, list):
        return [_adapt_schema_for_datalab(item) for item in schema]
    return schema


# ---------------------------------------------------------------------------
# Citation resolution helpers
# ---------------------------------------------------------------------------


_CITATIONS_SUFFIX = "_citations"
_SCORE_SUFFIX = "_score"
_META_SUFFIX = "_meta"
_SIDECAR_SUFFIXES = (_CITATIONS_SUFFIX, _SCORE_SUFFIX, _META_SUFFIX)


def _strip_citation_and_score_keys(node: Any) -> Any:
    """Return a copy of ``node`` with ``*_citations``, ``*_score`` and ``*_meta`` keys removed.

    Datalab interleaves the citation/score metadata with the actual field
    values inside ``extraction_schema_json``. The bench schema stores those as
    structured ``FieldCitation`` objects separately, so the extracted_data
    payload should be the clean schema-shaped dict.
    """
    if isinstance(node, Mapping):
        cleaned: dict[str, Any] = {}
        for key, value in node.items():
            if isinstance(key, str) and key.endswith(_SIDECAR_SUFFIXES):
                continue
            cleaned[key] = _strip_citation_and_score_keys(value)
        return cleaned
    if isinstance(node, list):
        return [_strip_citation_and_score_keys(item) for item in node]
    return node


def _build_block_lookup(json_tree: Any) -> dict[str, dict[str, Any]]:
    """Flatten Datalab's JSON document tree into ``block_id -> info`` lookup.

    The tree shape is::

        {"children": [<page>, ...], "metadata": {...}}

    Each page has ``block_type == "Page"``, a pixel ``bbox = [x1, y1, x2, y2]``,
    a 1-indexed or 0-indexed ``page_number`` (we fall back to enumeration
    order), and ``children`` containing nested blocks. Blocks carry ``id``,
    ``bbox``, ``block_type``, and optionally ``html``.
    """
    lookup: dict[str, dict[str, Any]] = {}
    if not isinstance(json_tree, Mapping):
        return lookup

    pages = json_tree.get("children")
    if not isinstance(pages, Sequence):
        return lookup

    page_idx_counter = 0
    for page in pages:
        if not isinstance(page, Mapping):
            continue
        if page.get("block_type") != "Page":
            continue

        page_bbox = page.get("bbox")
        page_w, page_h = _page_dims(page_bbox)
        page_number_raw = page.get("page_number")
        if isinstance(page_number_raw, int) and page_number_raw > 0:
            page_number = page_number_raw
        else:
            page_number = page_idx_counter + 1
        page_idx_counter += 1

        _walk_blocks(
            page,
            lookup=lookup,
            page_number=page_number,
            page_w=page_w,
            page_h=page_h,
        )

    return lookup


def _walk_blocks(
    node: Mapping[str, Any],
    *,
    lookup: dict[str, dict[str, Any]],
    page_number: int,
    page_w: float,
    page_h: float,
) -> None:
    block_id = node.get("id")
    if isinstance(block_id, str) and block_id and block_id not in lookup:
        lookup[block_id] = {
            "page_number": page_number,
            "page_w": page_w,
            "page_h": page_h,
            "bbox": node.get("bbox"),
            "html": node.get("html"),
            "block_type": node.get("block_type"),
        }

    children = node.get("children")
    if isinstance(children, Sequence):
        for child in children:
            if isinstance(child, Mapping):
                _walk_blocks(
                    child,
                    lookup=lookup,
                    page_number=page_number,
                    page_w=page_w,
                    page_h=page_h,
                )


def _page_dims(page_bbox: Any) -> tuple[float, float]:
    if isinstance(page_bbox, Sequence) and len(page_bbox) >= 4:
        try:
            w = float(page_bbox[2])
            h = float(page_bbox[3])
            if w > 0 and h > 0:
                return w, h
        except (TypeError, ValueError):
            pass
    return 1.0, 1.0


def _build_field_citations(
    payload: Any,
    block_lookup: dict[str, dict[str, Any]],
) -> list[FieldCitation]:
    citations: list[FieldCitation] = []
    _collect_citations(payload, path=[], block_lookup=block_lookup, citations=citations)
    return _dedupe_citations(citations)


def _collect_citations(
    node: Any,
    *,
    path: list[str],
    block_lookup: dict[str, dict[str, Any]],
    citations: list[FieldCitation],
) -> None:
    if isinstance(node, Mapping):
        for key, value in node.items():
            if not isinstance(key, str):
                continue
            if key.endswith(_CITATIONS_SUFFIX):
                field_name = key[: -len(_CITATIONS_SUFFIX)]
                field_path = ".".join([*path, field_name]) if path or field_name else field_name
                score_obj = node.get(field_name + _SCORE_SUFFIX)
                if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
                    for block_id in value:
                        citation = _citation_from_block_id(
                            block_id=block_id,
                            field_path=field_path,
                            block_lookup=block_lookup,
                            score_obj=score_obj,
                        )
                        if citation is not None:
                            citations.append(citation)
                continue
            if key.endswith(_SCORE_SUFFIX):
                field_name = key[: -len(_SCORE_SUFFIX)]
                field_value = node.get(field_name)
                if field_name in node and not isinstance(field_value, (Mapping, list)):
                    confidence = _score_confidence(value)
                    block_ids = node.get(field_name + _CITATIONS_SUFFIX)
                    if confidence is not None and (block_ids is None or block_ids == []):
                        citations.append(
                            FieldCitation(
                                field_path=".".join([*path, field_name]),
                                page=None,
                                bbox=None,
                                source="datalab",
                                confidence=confidence,
                                metadata=_score_metadata(value),
                            ),
                        )
                continue
            if key.endswith(_META_SUFFIX):
                continue
            _collect_citations(
                value,
                path=[*path, key],
                block_lookup=block_lookup,
                citations=citations,
            )
        return
    if isinstance(node, list):
        for idx, item in enumerate(node):
            _collect_citations(
                item,
                path=[*path, str(idx)],
                block_lookup=block_lookup,
                citations=citations,
            )


def _citation_from_block_id(
    *,
    block_id: Any,
    field_path: str,
    block_lookup: dict[str, dict[str, Any]],
    score_obj: Any,
) -> FieldCitation | None:
    if not isinstance(block_id, str) or not block_id:
        return None
    info = block_lookup.get(block_id)
    if info is None:
        # The id is still useful for downstream debugging even if we can't
        # resolve a bbox (e.g. the user asked for output_format != "json").
        return FieldCitation(
            field_path=field_path,
            page=None,
            bbox=None,
            source="datalab",
            confidence=_score_confidence(score_obj),
            metadata={"block_id": block_id, **_score_metadata(score_obj)},
        )

    bbox_pixels = info.get("bbox")
    page_w = float(info.get("page_w") or 1.0)
    page_h = float(info.get("page_h") or 1.0)
    bbox_norm = _normalize_bbox(bbox_pixels, page_w, page_h)

    reference_text = info.get("html") if isinstance(info.get("html"), str) else None

    confidence = _score_confidence(score_obj)

    metadata: dict[str, Any] = {
        "block_id": block_id,
        "block_type": info.get("block_type"),
        **_score_metadata(score_obj),
    }

    return FieldCitation(
        field_path=field_path,
        page=int(info["page_number"]),
        bbox=bbox_norm,
        polygon=None,
        reference_text=reference_text,
        confidence=confidence,
        source="datalab",
        metadata=metadata,
    )


def _normalize_bbox(bbox_pixels: Any, page_w: float, page_h: float) -> list[float] | None:
    if not isinstance(bbox_pixels, Sequence) or len(bbox_pixels) < 4:
        return None
    try:
        x1 = float(bbox_pixels[0])
        y1 = float(bbox_pixels[1])
        x2 = float(bbox_pixels[2])
        y2 = float(bbox_pixels[3])
    except (TypeError, ValueError):
        return None
    if page_w <= 0 or page_h <= 0:
        return None
    return [x1 / page_w, y1 / page_h, (x2 - x1) / page_w, (y2 - y1) / page_h]


def _score_confidence(score_obj: Any) -> float | None:
    if score_obj is None:
        return None
    score = score_obj["score"]
    if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score):
        return None
    return max(0.0, min(1.0, (float(score) - 1.0) / 4.0))


def _score_metadata(score_obj: Any) -> dict[str, Any]:
    if not isinstance(score_obj, Mapping):
        return {}
    out: dict[str, Any] = {}
    if "score" in score_obj:
        out["score"] = score_obj.get("score")
    if "reasoning" in score_obj:
        out["reasoning"] = score_obj.get("reasoning")
    return out


def _dedupe_citations(citations: list[FieldCitation]) -> list[FieldCitation]:
    seen: set[tuple[Any, ...]] = set()
    deduped: list[FieldCitation] = []
    for c in citations:
        key = (
            c.field_path,
            c.page,
            tuple(c.bbox) if c.bbox is not None else None,
            (c.metadata or {}).get("block_id"),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(c)
    return deduped
