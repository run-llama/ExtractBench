"""Provider for LandingAI ADE v2 (DPT-3 Parse Jobs + v2 Extract Jobs).

The v1 provider (``landingai.py``) calls the Gen1 ``client.parse`` / ``client.extract``
endpoints. DPT-3 parse models and the v2 extract models (for example
``extract-20261111``) are only served by the v2 APIs, which this provider uses:

1. ``client.v2.parse_jobs`` parses the PDF (no 100-page sync limit).
2. ``client.v2.extract_jobs`` extracts with the benchmark schema, passed unchanged,
   from the v2 Parse Markdown.
3. Field citations map each extracted value's Markdown ranges to the parse
   structure's atomic boxes, word-level on ``dpt-3-pro-20261111`` (see
   ``extract_landingai_v2_field_citations``).

Both stages run as async jobs and are polled against the document's shared
execution budget, so the harness deadline and cancellation are respected.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

from landingai_ade import (
    APIConnectionError,
    APIStatusError,
    InternalServerError,
    LandingAIADE,
    RateLimitError,
)

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
from extract_bench.inference.limits import DEFAULT_POLL_INTERVAL_S, execution_budget
from extract_bench.inference.providers.base import (
    Provider,
    ProviderConfigError,
    ProviderPermanentError,
    ProviderRateLimitError,
    ProviderTransientError,
)
from extract_bench.inference.providers.extract.citations import extract_landingai_v2_field_citations
from extract_bench.inference.providers.registry import register_provider
from extract_bench.schemas.extract_output import ExtractOutput
from extract_bench.schemas.pipeline import PipelineSpec
from extract_bench.schemas.pipeline_io import (
    InferenceRequest,
    InferenceResult,
    RawInferenceResult,
)
from extract_bench.schemas.product import ProductType

# Keys this provider reads from the pipeline config; nothing else is forwarded.
_CONFIG_KEYS = frozenset(
    {"api_key", "model", "parse_model", "grounding", "service_tier", "parse_options", "poll_interval_s"}
)
_SERVICE_TIERS = frozenset({"standard", "priority"})
# A single status request should never hold the poll loop for long.
_STATUS_REQUEST_TIMEOUT_S = 60.0


def _coerce_nonnegative(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if number >= 0 else None


def _billing_credits(metadata: Mapping[str, Any] | None) -> float | None:
    """Credits charged for one v2 stage: ``metadata.billing.total_credits``."""
    if not isinstance(metadata, Mapping):
        return None
    billing = metadata.get("billing")
    if isinstance(billing, Mapping):
        credits = _coerce_nonnegative(billing.get("total_credits"))
        if credits is not None:
            return credits
    # Extract results also carry a flat credit_usage field.
    return _coerce_nonnegative(metadata.get("credit_usage"))


class _JobFailed(Exception):
    """A v2 job reached a terminal non-success state."""

    def __init__(self, stage: str, job_id: str, status: str, code: str | None, message: str | None):
        self.stage = stage
        self.job_id = job_id
        self.status = status
        self.code = code
        super().__init__(f"{stage} job {job_id} {status}: {code or ''} {message or ''}".strip())


@register_provider("landingai_ade_v2_extract")
class LandingAIADEV2ExtractProvider(Provider):
    """LandingAI ADE v2: DPT-3 Parse Jobs followed by v2 Extract Jobs."""

    # LandingAI list price: $0.01 per credit (same conversion as the v1 provider).
    CREDIT_RATE_USD = 0.01

    def __init__(
        self,
        provider_name: str,
        base_config: dict[str, Any] | None = None,
    ):
        """
        Initialize the provider.

        :param provider_name: Name of the provider
        :param base_config: Optional configuration with:
            - `api_key`: LandingAI API key (defaults to LANDING_AI_API_KEY, then VISION_AGENT_API_KEY)
            - `model`: v2 extract model, e.g. "extract-20261111" (default: server default)
            - `parse_model`: v2 parse model, e.g. "dpt-3-pro-20261111" (default: server default)
            - `grounding`: request extraction ranges (default True, the API default)
            - `service_tier`: "standard" or "priority" for both jobs (default: API default)
            - `parse_options`: optional v2 parse `options` mapping
            - `poll_interval_s`: seconds between job status checks
        """
        super().__init__(provider_name, base_config)

        unknown = set(self.base_config) - _CONFIG_KEYS
        if unknown:
            raise ProviderConfigError(f"Unknown landingai_ade_v2_extract config keys: {sorted(unknown)}")

        self._api_key = (
            self.base_config.get("api_key") or os.getenv("LANDING_AI_API_KEY") or os.getenv("VISION_AGENT_API_KEY")
        )
        if not self._api_key:
            raise ProviderConfigError(
                "LandingAI API key is required. "
                "Set LANDING_AI_API_KEY (or VISION_AGENT_API_KEY) or pass api_key in base_config."
            )

        tier = self.base_config.get("service_tier")
        if tier is not None and tier not in _SERVICE_TIERS:
            raise ProviderConfigError(f"service_tier must be one of {sorted(_SERVICE_TIERS)}, got {tier!r}")

        self._model: str | None = self.base_config.get("model")
        self._parse_model: str | None = self.base_config.get("parse_model")
        self._grounding: bool = bool(self.base_config.get("grounding", True))
        self._service_tier: str | None = tier
        self._parse_options: Mapping[str, Any] | None = self.base_config.get("parse_options")
        self._poll_interval_s = float(self.base_config.get("poll_interval_s", DEFAULT_POLL_INTERVAL_S))

        self._client = LandingAIADE(apikey=self._api_key)

    # ------------------------------------------------------------------ jobs

    def _wait_for_job(
        self,
        *,
        stage: str,
        job_id: str,
        get_job: Callable[..., Any],
    ) -> Mapping[str, Any]:
        """Poll a v2 job until it is terminal and return its result payload."""
        budget = execution_budget()
        while True:
            job = get_job(job_id, timeout=min(_STATUS_REQUEST_TIMEOUT_S, budget.remaining()))
            status = getattr(getattr(job, "status", None), "value", getattr(job, "status", None))
            if status == "completed":
                raw = getattr(job, "raw", None) or {}
                payload = raw.get("result")
                if payload is None:
                    payload = raw.get("data")
                if not isinstance(payload, Mapping):
                    raise ProviderPermanentError(
                        f"LandingAI {stage} job {job_id} completed without an inline result (was output_save_url set?)"
                    )
                return payload
            if status in ("failed", "cancelled"):
                error = getattr(job, "error", None)
                raise _JobFailed(
                    stage,
                    job_id,
                    str(status),
                    getattr(error, "code", None),
                    getattr(error, "message", None),
                )
            budget.wait(self._poll_interval_s)

    def _parse(self, document_path: Path) -> Mapping[str, Any]:
        kwargs: dict[str, Any] = {}
        if self._parse_model:
            kwargs["model"] = self._parse_model
        if self._service_tier:
            kwargs["service_tier"] = self._service_tier
        if self._parse_options:
            kwargs["options"] = dict(self._parse_options)
        budget = execution_budget()
        job = self._client.v2.parse_jobs.create(
            document=document_path,
            timeout=min(_STATUS_REQUEST_TIMEOUT_S * 5, budget.remaining()),
            **kwargs,
        )
        return self._wait_for_job(stage="parse", job_id=job.job_id, get_job=self._client.v2.parse_jobs.get)

    def _extract(self, markdown: str, schema: dict[str, Any] | str) -> Mapping[str, Any]:
        kwargs: dict[str, Any] = {"grounding": self._grounding}
        if self._model:
            kwargs["model"] = self._model
        if self._service_tier:
            kwargs["service_tier"] = self._service_tier
        budget = execution_budget()
        job = self._client.v2.extract_jobs.create(
            schema=schema,
            markdown=markdown,
            timeout=min(_STATUS_REQUEST_TIMEOUT_S * 5, budget.remaining()),
            **kwargs,
        )
        return self._wait_for_job(stage="extract", job_id=job.job_id, get_job=self._client.v2.extract_jobs.get)

    # ------------------------------------------------------------------ cost

    def _record_stage_cost(self, accounting: dict[str, Any], stage: str, metadata: Mapping[str, Any] | None) -> None:
        credits = _billing_credits(metadata)
        if credits is None:
            return
        accounting[f"{stage}_credits"] = credits
        job_id = (metadata or {}).get("job_id") or "unknown"
        record_response_cost(
            credits * self.CREDIT_RATE_USD,
            dict(metadata or {}),
            component_id=f"landingai_v2/{stage}/{job_id}",
        )

    def _cost_accounting(self, accounting: Mapping[str, Any]) -> dict[str, Any]:
        credits = [accounting[key] for key in ("parse_credits", "extract_credits") if key in accounting]
        return {
            KNOWN_COST_USD: sum(credits) * self.CREDIT_RATE_USD if credits else None,
            COMPLETE: len(credits) == 2,
        }

    def _apply_cost_fields(self, result: dict[str, Any], accounting: Mapping[str, Any], pages: float | None) -> None:
        snapshot = self._cost_accounting(accounting)
        result.update(accounting)
        result[COST_USD] = snapshot[KNOWN_COST_USD]
        result[RAW_COST_COMPLETE] = snapshot[COMPLETE]
        if snapshot[KNOWN_COST_USD] is not None:
            result["credits_used"] = sum(accounting.values())
        if pages is not None and pages > 0:
            result["num_pages"] = pages
            if snapshot[KNOWN_COST_USD] is not None:
                result[COST_PER_PAGE_USD] = snapshot[KNOWN_COST_USD] / pages

    # ------------------------------------------------------------- pipeline

    def _extract_from_document(self, document_path: Path, schema: dict[str, Any] | str) -> dict[str, Any]:
        """Parse then extract one document; return the raw output dictionary."""
        accounting: dict[str, Any] = {}
        try:
            parse_payload = self._parse(document_path)
            parse_metadata = parse_payload.get("metadata")
            self._record_stage_cost(accounting, "parse", parse_metadata)
            emit_provider_event("stage.completed", provider="landingai", stage="parse", raw_output=parse_payload)

            markdown = parse_payload.get("markdown") or ""
            extract_payload = self._extract(markdown, schema)
            extract_metadata = extract_payload.get("metadata")
            self._record_stage_cost(accounting, "extract", extract_metadata)
            emit_provider_event("stage.completed", provider="landingai", stage="extract", raw_output=extract_payload)

            extraction = extract_payload.get("extraction")
            result: dict[str, Any] = {
                "data": extraction if isinstance(extraction, Mapping) else {},
                "extraction_metadata": extract_payload.get("extraction_metadata"),
                "parse_structure": parse_payload.get("structure"),
                "parse_metadata": parse_metadata,
                "extract_metadata": extract_metadata,
                "warnings": extract_payload.get("warnings"),
                "schema_violation_error": extract_payload.get("schema_violation_error"),
            }
            pages = _coerce_nonnegative((parse_metadata or {}).get("page_count"))
            self._apply_cost_fields(result, accounting, pages)
            return result

        except TimeoutError:
            raise
        except Exception as error:
            raise self._classify(error, accounting) from error

    def _classify(self, error: Exception, accounting: Mapping[str, Any]) -> Exception:
        """Map an SDK or job error to the harness's retry categories."""
        debug_payload = {COST_ACCOUNTING: self._cost_accounting(accounting)}
        if isinstance(error, (ProviderPermanentError, ProviderTransientError, ProviderRateLimitError)):
            return error
        if isinstance(error, RateLimitError):
            return ProviderRateLimitError(f"Rate limited: {error}", debug_payload=debug_payload)
        if isinstance(error, (APIConnectionError, InternalServerError)):
            return ProviderTransientError(f"Transient LandingAI error: {error}", debug_payload=debug_payload)
        if isinstance(error, APIStatusError):
            if error.status_code in (408, 502, 503, 504):
                return ProviderTransientError(f"Transient LandingAI error: {error}", debug_payload=debug_payload)
            return ProviderPermanentError(f"LandingAI request rejected: {error}", debug_payload=debug_payload)
        if isinstance(error, _JobFailed):
            text = str(error).lower()
            if any(word in text for word in ("timeout", "timed out", "unavailable", "internal")):
                return ProviderTransientError(f"LandingAI job failed: {error}", debug_payload=debug_payload)
            return ProviderPermanentError(f"LandingAI job failed: {error}", debug_payload=debug_payload)
        text = str(error).lower()
        if any(word in text for word in ("429", "rate limit", "too many requests")):
            return ProviderRateLimitError(f"Rate limited: {error}", debug_payload=debug_payload)
        if any(word in text for word in ("timeout", "network", "connection", "502", "503", "504")):
            return ProviderTransientError(f"Transient LandingAI error: {error}", debug_payload=debug_payload)
        return ProviderPermanentError(f"Error during LandingAI extraction: {error}", debug_payload=debug_payload)

    def run_inference(self, pipeline: PipelineSpec, request: InferenceRequest) -> RawInferenceResult:
        """Run inference and return raw results."""
        if request.product_type != ProductType.EXTRACT:
            raise ProviderPermanentError(
                f"LandingAIADEV2ExtractProvider only supports EXTRACT product type, got {request.product_type}"
            )
        if not request.schema_override:
            raise ProviderPermanentError(
                "schema_override is required for EXTRACT product type. "
                "Provide a JSON schema in InferenceRequest.schema_override"
            )

        file_path = Path(request.source_file_path)
        if not file_path.exists():
            raise ProviderPermanentError(f"File not found: {file_path}")

        started_at = datetime.now()
        raw_output = self._extract_from_document(document_path=file_path, schema=request.schema_override)
        completed_at = datetime.now()
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
        """Normalize raw inference result to produce ExtractOutput."""
        if raw_result.product_type != ProductType.EXTRACT:
            raise ProviderPermanentError(
                f"LandingAIADEV2ExtractProvider only supports EXTRACT product type, got {raw_result.product_type}"
            )
        raw_output = raw_result.raw_output
        extracted_data = raw_output.get("data", {})
        output = ExtractOutput(
            task_type="extract",
            example_id=raw_result.request.example_id,
            pipeline_name=raw_result.pipeline_name,
            extracted_data=extracted_data,
            field_citations=extract_landingai_v2_field_citations(
                raw_output.get("extraction_metadata"),
                raw_output.get("parse_structure"),
            ),
        )
        return InferenceResult(
            request=raw_result.request,
            pipeline_name=raw_result.pipeline_name,
            product_type=raw_result.product_type,
            raw_output=raw_output,
            output=output,
            started_at=raw_result.started_at,
            completed_at=raw_result.completed_at,
            latency_in_ms=raw_result.latency_in_ms,
        )
