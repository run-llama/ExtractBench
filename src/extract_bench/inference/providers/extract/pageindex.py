"""Provider for PageIndex Page Extract (https://pageindex.ai/extract).

Page Extract takes a JSON Schema and returns JSON that fits it. The provider sends
the PDF with the schema, unmodified, in one request that starts an extraction job
(``POST /extract`` with ``file``), and polls the job.
``cite_sources`` asks the extraction to record the page each value was read from;
the job is billed by the page either way.
API reference: https://pageindex.ai/docs/api-reference/extract

Environment: ``PAGEINDEX_API_KEY``.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import requests

from extract_bench.inference.cost_accounting import record_response_cost
from extract_bench.inference.cost_contract import (
    COMPLETE,
    COST_ACCOUNTING,
    COST_PER_PAGE_USD,
    COST_USD,
    KNOWN_COST_USD,
    RAW_COST_COMPLETE,
)
from extract_bench.inference.events import emit_provider_event
from extract_bench.inference.limits import execution_budget
from extract_bench.inference.providers.base import (
    Provider,
    ProviderConfigError,
    ProviderPermanentError,
    ProviderRateLimitError,
    ProviderTransientError,
)
from extract_bench.inference.providers.extract.citations import format_field_path
from extract_bench.inference.providers.registry import register_provider
from extract_bench.schemas.extract_output import ExtractOutput, FieldCitation
from extract_bench.schemas.pipeline import PipelineSpec
from extract_bench.schemas.pipeline_io import (
    InferenceRequest,
    InferenceResult,
    RawInferenceResult,
)
from extract_bench.schemas.product import ProductType

API_URL = "https://api.pageindex.ai"
API_KEY_ENV = "PAGEINDEX_API_KEY"
TIERS = ("standard", "pro")
USD_PER_CREDIT = 0.01  # usage.credits: one credit is one US cent
HTTP_TIMEOUT_S = 120.0
UPLOAD_TIMEOUT_S = 600.0
# A job exists once it is created: a runner retry would start and bill a second one,
# so errors while polling it are ridden out here, up to this many in a row.
MAX_POLL_ERRORS = 10
PENDING_JOB_STATUSES = frozenset({"pending", "running"})


class _ApiError(Exception):
    def __init__(self, status: int | None, message: str):
        super().__init__(message)
        self.status = status


def _citations(extracted: Any, field_metadata: Any) -> list[FieldCitation]:
    """Page citations from ``{"/rows/0": {"page": 3}, "/rows/0/name": {"page": 4}}``.

    The scorer reads pages on scalar fields, so a list item's page is given to each
    scalar in that item. Page Extract also cites values one by one; a pointer to the
    value itself is more specific than its item's and wins, so each field has one page.
    """
    if not isinstance(field_metadata, dict):
        return []
    pages: dict[str, tuple[int, int]] = {}  # field path -> (depth of the pointer, page)
    for pointer, meta in field_metadata.items():
        page = meta.get("page") if isinstance(meta, dict) else None
        if not isinstance(pointer, str) or not isinstance(page, int) or page < 1:
            continue
        node, tokens = extracted, []
        for raw in pointer.split("/")[1:]:
            key = raw.replace("~1", "/").replace("~0", "~")
            if isinstance(node, list) and key.isdigit() and int(key) < len(node):
                node, token = node[int(key)], f"[{key}]"
            elif isinstance(node, dict) and key in node:
                node, token = node[key], key
            else:
                break
            tokens.append(token)
        else:
            for path in _scalar_paths(node, tokens):
                field_path = format_field_path(path)
                if len(tokens) >= pages.get(field_path, (-1, 0))[0]:
                    pages[field_path] = (len(tokens), page)
    return [FieldCitation(field_path=path, page=page, source="pageindex") for path, (_, page) in pages.items()]


def _scalar_paths(node: Any, path: list[str]) -> list[list[str]]:
    if isinstance(node, dict):
        return [leaf for key, child in node.items() for leaf in _scalar_paths(child, [*path, key])]
    if isinstance(node, list):
        return [leaf for i, child in enumerate(node) for leaf in _scalar_paths(child, [*path, f"[{i}]"])]
    return [path]


@register_provider("pageindex")
class PageIndexExtractProvider(Provider):
    """Page Extract over its public REST API.

    Config keys (``base_config``): ``tier`` (``standard`` or ``pro``) and ``poll_interval``.
    """

    def __init__(self, provider_name: str, base_config: dict[str, Any] | None = None):
        super().__init__(provider_name, base_config)
        api_key = os.getenv(API_KEY_ENV)
        if not api_key:
            raise ProviderConfigError(f"{API_KEY_ENV} environment variable is required for the pageindex provider.")
        self._tier = self.base_config.get("tier", "standard")
        if self._tier not in TIERS:
            raise ProviderConfigError(f"tier must be one of {TIERS}, got {self._tier!r}")
        self._session = requests.Session()
        self._session.headers.update({"Authorization": f"Bearer {api_key}"})
        self._started: dict[str, datetime] = {}  # example_id -> first attempt's start

    def _request(self, method: str, url: str, **kwargs: Any) -> dict[str, Any]:
        kwargs.setdefault("timeout", HTTP_TIMEOUT_S)
        try:
            response = self._session.request(method, url, **kwargs)
        except requests.RequestException as exc:
            raise _ApiError(None, f"{method} {url}: {exc}") from exc
        try:
            data = response.json()
        except ValueError:
            data = None
        if response.ok and isinstance(data, dict):
            return data
        detail = data.get("detail") if isinstance(data, dict) else None
        raise _ApiError(
            response.status_code, f"{method} {url}: HTTP {response.status_code}: {detail or response.text[:500]}"
        )

    def _start_job(self, file_path: Path, schema: dict[str, Any]) -> dict[str, Any]:
        configuration = {"data_schema": schema, "tier": self._tier, "cite_sources": True}
        try:
            return self._request(
                "POST",
                f"{API_URL}/extract",
                data={"configuration": json.dumps(configuration)},
                files={"file": (file_path.name, file_path.read_bytes(), "application/pdf")},
                timeout=UPLOAD_TIMEOUT_S,
            )
        except _ApiError as exc:
            # Turned away before a job was started: safe for the runner to retry.
            if exc.status == 429:
                raise ProviderRateLimitError(f"Rate limited starting the job: {exc}") from exc
            if exc.status in (502, 503):
                raise ProviderTransientError(f"Transient error starting the job: {exc}") from exc
            # A job may have started (and would be billed twice by a retry).
            raise ProviderPermanentError(f"Starting the job failed, not retried: {exc}") from exc

    def _cancel_job(self, job_id: str) -> None:
        """A job cancelled before it completes is not charged."""
        try:
            self._request("POST", f"{API_URL}/extract/{job_id}/cancel")
        except _ApiError:
            pass  # best effort: the deadline has already failed the document

    def _wait_for_job(self, job: dict[str, Any]) -> dict[str, Any]:
        budget = execution_budget()
        errors = 0
        try:
            while job.get("status") in PENDING_JOB_STATUSES:
                budget.wait(self._poll_interval)
                try:
                    job = self._request("GET", f"{API_URL}/extract/{job['id']}")
                    errors = 0
                except _ApiError as exc:
                    transient = exc.status is None or exc.status == 429 or exc.status >= 500
                    errors += 1
                    if not transient or errors >= MAX_POLL_ERRORS:
                        self._cancel_job(job["id"])
                        raise ProviderPermanentError(
                            f"Polling job {job['id']} failed: {exc}", job_id=job["id"]
                        ) from exc
        except TimeoutError:
            self._cancel_job(job["id"])
            raise
        if job.get("status") == "failed":
            # A failed job is not charged and a new one may finish: the runner's retry starts one.
            failure = {"status": "failed", "error_code": job.get("error_code")}
            record_response_cost(0.0, failure, component_id=f"pageindex/extract/{job['id']}")
            raise ProviderTransientError(
                f"Job {job['id']} failed ({job.get('error_code')}): {job.get('error_message')}",
                job_id=job["id"],
                debug_payload=job,
            )
        if job.get("status") != "completed":
            raise ProviderPermanentError(
                f"Job {job['id']} {job.get('status')}: {job.get('error_message')}", job_id=job["id"], debug_payload=job
            )
        return job

    def run_inference(self, pipeline: PipelineSpec, request: InferenceRequest) -> RawInferenceResult:
        if request.product_type != ProductType.EXTRACT:
            raise ProviderPermanentError(f"pageindex provider only supports EXTRACT, got {request.product_type.value}")
        if not request.schema_override:
            raise ProviderPermanentError("pageindex provider requires schema_override (the document's JSON Schema)")
        file_path = Path(request.source_file_path)
        if not file_path.is_file():
            raise ProviderPermanentError(f"Source file not found: {file_path}")

        # A retry keeps the first attempt's start, so latency covers every attempt.
        started_at = self._started.setdefault(request.example_id, datetime.now())
        try:
            job = self._start_job(file_path, request.schema_override)
            emit_provider_event("vendor.job_created", provider="pageindex", stage="extract", job_id=job["id"])
            # Its bill is unknown until it ends, but its ID is on disk from now on: a job
            # that runs on after an interrupted run can still be found and reconciled.
            record_response_cost(None, {}, component_id=f"pageindex/extract/{job['id']}")
            job = self._wait_for_job(job)
        except (ProviderTransientError, ProviderRateLimitError) as exc:
            # No job was charged: none started, or it failed.
            exc.debug_payload = {**(exc.debug_payload or {}), COST_ACCOUNTING: {KNOWN_COST_USD: 0.0, COMPLETE: True}}
            raise
        self._started.pop(request.example_id, None)
        completed_at = datetime.now()
        emit_provider_event("stage.completed", provider="pageindex", stage="extract", job_id=job["id"], raw_output=job)

        pages = int(job["usage"]["pages"])
        cost = float(job["usage"]["credits"]) * USD_PER_CREDIT
        record_response_cost(cost, job["usage"], component_id=f"pageindex/extract/{job['id']}")
        raw_output = {
            "job": job,
            "tier": self._tier,
            "num_pages": pages,
            COST_USD: cost,
            RAW_COST_COMPLETE: True,
            COST_PER_PAGE_USD: cost / pages,
        }
        return RawInferenceResult(
            request=request,
            pipeline=pipeline,
            pipeline_name=pipeline.pipeline_name,
            product_type=request.product_type,
            raw_output=raw_output,
            started_at=started_at,
            completed_at=completed_at,
            latency_in_ms=int((completed_at - started_at).total_seconds() * 1000),
        )

    def normalize(self, raw_result: RawInferenceResult) -> InferenceResult:
        job = raw_result.raw_output.get("job") or {}
        extracted = job.get("extract_result")
        if not isinstance(extracted, dict):
            raise ProviderPermanentError(f"Job {job.get('id')} returned no JSON object: {extracted!r:.200}")
        output = ExtractOutput(
            example_id=raw_result.request.example_id,
            pipeline_name=raw_result.pipeline_name,
            extracted_data=extracted,
            field_citations=_citations(extracted, (job.get("extract_metadata") or {}).get("field_metadata")),
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
