"""Provider for Sarvam Extract (https://docs.sarvam.ai/doc-ai/getting-started/overview).

Sarvam Extract takes a JSON Schema and returns JSON that fits it, with a page, a
normalized box and a confidence score for each value. The provider sends the file and
the schema, unmodified, in one request that creates and starts an extraction job
(``POST /doc-ai/v1/job/extract``), then polls the job and reads ``/results``.
``mode`` selects ``basic`` (page-by-page) or ``advanced`` (adds a whole-document pass).
Pricing is flat per processed page, and only successfully processed pages are billed.

Environment: ``SARVAM_API_KEY``.
"""

from __future__ import annotations

import json
import os
import uuid
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

API_URL = "https://api.sarvam.ai/doc-ai/v1"
API_KEY_ENV = "SARVAM_API_KEY"
MODES = ("basic", "advanced")
HTTP_TIMEOUT_S = 120.0
UPLOAD_TIMEOUT_S = 600.0
# A job exists once it is created: errors while polling it are ridden out here, up to
# this many in a row, instead of starting (and billing) a second job.
MAX_POLL_ERRORS = 10
MAX_SUBMIT_ATTEMPTS = 3
PENDING_JOB_STATUSES = frozenset({"pending", "running"})


class _ApiError(Exception):
    def __init__(self, status: int | None, message: str):
        super().__init__(message)
        self.status = status


def _is_scalar(value: Any) -> bool:
    return not isinstance(value, (dict, list))


def _citations(extracted: Any, annotations: Any) -> list[FieldCitation]:
    """Citations from ``annotations``, which mirrors ``result`` key for key and index for index.

    A leaf annotation is ``{"confidence": 0-1, "sources": [{"page_num", "bbox_norm": [x1, y1, x2, y2]}]}``.
    The first source is used. ``region_norm`` (the wider context region) is not used as a box.
    """
    citations: list[FieldCitation] = []

    def walk(value: Any, annotation: Any, path: list[str]) -> None:
        if isinstance(value, dict) and isinstance(annotation, dict) and "sources" not in annotation:
            for key, child in value.items():
                walk(child, annotation.get(key), [*path, key])
        elif isinstance(value, list) and isinstance(annotation, list):
            for index, child in enumerate(value):
                walk(child, annotation[index] if index < len(annotation) else None, [*path, f"[{index}]"])
        elif _is_scalar(value) and isinstance(annotation, dict):
            sources = annotation.get("sources") or []
            source = sources[0] if sources and isinstance(sources[0], dict) else None
            page = source.get("page_num") if source else None
            if not isinstance(page, int) or page < 1:
                return
            bbox = None
            box = source.get("bbox_norm")
            if isinstance(box, list) and len(box) == 4 and all(isinstance(v, (int, float)) for v in box):
                x1, y1, x2, y2 = (float(v) for v in box)
                bbox = [x1, y1, max(x2 - x1, 0.0), max(y2 - y1, 0.0)]
            confidence = annotation.get("confidence")
            citations.append(
                FieldCitation(
                    field_path=format_field_path(path),
                    page=page,
                    bbox=bbox,
                    confidence=float(confidence) if isinstance(confidence, (int, float)) else None,
                    source="sarvam",
                )
            )

    walk(extracted, annotations, [])
    return citations


@register_provider("sarvam_extract")
class SarvamExtractProvider(Provider):
    """Sarvam Extract over its public REST API.

    Config keys (``base_config``): ``mode`` (``basic`` or ``advanced``), ``language``
    (BCP-47 code, default ``en-IN``), ``price_per_page_usd`` (required, the list price
    for the mode) and ``poll_interval``.
    """

    def __init__(self, provider_name: str, base_config: dict[str, Any] | None = None):
        super().__init__(provider_name, base_config)
        api_key = os.getenv(API_KEY_ENV)
        if not api_key:
            raise ProviderConfigError(
                f"{API_KEY_ENV} environment variable is required for the sarvam_extract provider."
            )
        self._mode = self.base_config.get("mode", "basic")
        if self._mode not in MODES:
            raise ProviderConfigError(f"mode must be one of {MODES}, got {self._mode!r}")
        self._language = self.base_config.get("language", "en-IN")
        price = self.base_config.get("price_per_page_usd")
        if not isinstance(price, (int, float)) or price <= 0:
            raise ProviderConfigError("price_per_page_usd (the mode's list price in USD) is required")
        self._price_per_page_usd = float(price)
        self._session = requests.Session()
        self._session.headers.update({"api-subscription-key": api_key})
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
        detail = (data.get("detail") or data.get("title")) if isinstance(data, dict) else None
        raise _ApiError(
            response.status_code, f"{method} {url}: HTTP {response.status_code}: {detail or response.text[:500]}"
        )

    def _start_job(self, file_path: Path, schema: dict[str, Any]) -> dict[str, Any]:
        # The same Idempotency-Key on every attempt: a retry after a lost response returns
        # the original job instead of starting and billing a second one.
        headers = {"Idempotency-Key": uuid.uuid4().hex}
        content = file_path.read_bytes()
        form = {
            "schema": json.dumps(schema),
            "language": self._language,
            "output_format": "json",
            "mode": self._mode,
        }
        last: _ApiError | None = None
        for _ in range(MAX_SUBMIT_ATTEMPTS):
            try:
                return self._request(
                    "POST",
                    f"{API_URL}/job/extract",
                    headers=headers,
                    data=form,
                    files={"file": (file_path.name, content, "application/pdf")},
                    timeout=UPLOAD_TIMEOUT_S,
                )
            except _ApiError as exc:
                last = exc
                if exc.status == 429:
                    raise ProviderRateLimitError(f"Rate limited starting the job: {exc}") from exc
                if exc.status is not None and exc.status < 500:
                    raise ProviderPermanentError(f"Starting the job failed: {exc}") from exc
        raise ProviderTransientError(f"Starting the job failed after {MAX_SUBMIT_ATTEMPTS} attempts: {last}")

    def _wait_for_job(self, job: dict[str, Any]) -> dict[str, Any]:
        job_id = job["job_id"]
        budget = execution_budget()
        errors = 0
        status = job
        while status.get("status") in PENDING_JOB_STATUSES:
            budget.wait(self._poll_interval)
            try:
                status = self._request("GET", f"{API_URL}/job/{job_id}/status")
                errors = 0
            except _ApiError as exc:
                transient = exc.status is None or exc.status == 429 or exc.status >= 500
                errors += 1
                if not transient or errors >= MAX_POLL_ERRORS:
                    raise ProviderPermanentError(f"Polling job {job_id} failed: {exc}", job_id=job_id) from exc
        state = status.get("status")
        if state in ("failed", "rejected") or state == "partially_completed":
            # A failed job bills no pages and a new one may finish: the runner's retry starts one.
            # A partial result is not scored as if it were whole: its failed pages would read as blanks.
            record_response_cost(0.0, status.get("usage") or {}, component_id=f"sarvam/extract/{job_id}")
            error = ProviderTransientError if state != "rejected" else ProviderPermanentError
            raise error(f"Job {job_id} {state}: usage={status.get('usage')}", job_id=job_id, debug_payload=status)
        if state != "completed":
            raise ProviderPermanentError(f"Job {job_id} ended in {state!r}", job_id=job_id, debug_payload=status)
        return status

    def run_inference(self, pipeline: PipelineSpec, request: InferenceRequest) -> RawInferenceResult:
        if request.product_type != ProductType.EXTRACT:
            raise ProviderPermanentError(
                f"sarvam_extract provider only supports EXTRACT, got {request.product_type.value}"
            )
        if not request.schema_override:
            raise ProviderPermanentError(
                "sarvam_extract provider requires schema_override (the document's JSON Schema)"
            )
        file_path = Path(request.source_file_path)
        if not file_path.is_file():
            raise ProviderPermanentError(f"Source file not found: {file_path}")

        # A retry keeps the first attempt's start, so latency covers every attempt.
        started_at = self._started.setdefault(request.example_id, datetime.now())
        try:
            job = self._start_job(file_path, request.schema_override)
            emit_provider_event("vendor.job_created", provider="sarvam", stage="extract", job_id=job["job_id"])
            record_response_cost(None, {}, component_id=f"sarvam/extract/{job['job_id']}")
            self._wait_for_job(job)
            results = self._request("GET", f"{API_URL}/job/{job['job_id']}/results")
        except (ProviderTransientError, ProviderRateLimitError) as exc:
            exc.debug_payload = {**(exc.debug_payload or {}), COST_ACCOUNTING: {KNOWN_COST_USD: 0.0, COMPLETE: True}}
            raise
        except _ApiError as exc:
            raise ProviderPermanentError(
                f"Reading job {job['job_id']} results failed: {exc}", job_id=job["job_id"]
            ) from exc
        self._started.pop(request.example_id, None)
        completed_at = datetime.now()
        emit_provider_event("stage.completed", provider="sarvam", stage="extract", job_id=job["job_id"])

        usage = results.get("usage") or {}
        pages = int(usage.get("pages_succeeded") or 0)
        if pages < 1:
            raise ProviderPermanentError(f"Job {job['job_id']} reported no processed pages: {usage}")
        cost = pages * self._price_per_page_usd
        record_response_cost(cost, usage, component_id=f"sarvam/extract/{job['job_id']}")
        raw_output = {
            "job": results,
            "mode": self._mode,
            "num_pages": pages,
            COST_USD: cost,
            RAW_COST_COMPLETE: True,
            COST_PER_PAGE_USD: self._price_per_page_usd,
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
        extracted = job.get("result")
        if not isinstance(extracted, dict) or not extracted:
            raise ProviderPermanentError(f"Job {job.get('job_id')} returned no JSON object: {extracted!r:.200}")
        output = ExtractOutput(
            example_id=raw_result.request.example_id,
            pipeline_name=raw_result.pipeline_name,
            extracted_data=extracted,
            field_citations=_citations(extracted, job.get("annotations")),
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
