"""Provider for Reducto EXTRACT."""

import asyncio
import logging
import math
import os
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, cast
from uuid import uuid4

from reducto import Reducto

from extract_bench.inference.cost_contract import (
    COMPLETE,
    COMPONENT_ID,
    COST_ACCOUNTING,
    COST_COMPONENT_EVENT,
    COST_PER_PAGE_USD,
    COST_USD,
    KNOWN_COST_USD,
    RAW_COST_COMPLETE,
    USAGE,
)
from extract_bench.inference.events import emit_provider_event
from extract_bench.inference.limits import execution_budget
from extract_bench.inference.providers.base import (
    Provider,
    ProviderConfigError,
    ProviderPermanentError,
    ProviderTransientError,
)
from extract_bench.inference.providers.extract.citations import extract_reducto_field_citations
from extract_bench.inference.providers.extract.direct_model_utils import page_count
from extract_bench.inference.providers.registry import register_provider
from extract_bench.schemas.extract_output import ExtractOutput
from extract_bench.schemas.pipeline import PipelineSpec
from extract_bench.schemas.pipeline_io import (
    InferenceRequest,
    InferenceResult,
    RawInferenceResult,
)
from extract_bench.schemas.product import ProductType

REDUCTO_USD_PER_CREDIT = 0.015
REDUCTO_DEEP_CREDITS_PER_PAGE = 4.0
REDUCTO_DEEP_CREDITS_PER_FIELD = 0.1
REDUCTO_DEEP_MIN_CREDITS_PER_DOCUMENT = 30.0
REDUCTO_PARSE_CREDITS_PER_PAGE = 1.0


def _adapt_schema_for_reducto(schema: dict[str, Any]) -> dict[str, Any]:
    """Promote `repeated_structure[name]` entries into `properties[name]`.

    Some benchmark schemas declare repeated array nodes (e.g. claims,
    payment_details) under a top-level `repeated_structure` annotation that
    LlamaExtract reads as authoritative. Reducto follows standard JSON
    Schema and ignores unknown top-level keys, so when the
    schema's `properties` dict is missing those arrays Reducto silently
    skips them and returns an empty extraction for those fields.

    This adapter is a pure shape transform: copy each `repeated_structure`
    entry into `properties` if not already there, and drop the
    `repeated_structure` key so the wire payload is plain JSON Schema.
    """
    if not isinstance(schema, dict):
        return schema
    rs = schema.get("repeated_structure")
    if not isinstance(rs, dict) or not rs:
        return schema
    out = dict(schema)
    props = dict(out.get("properties") or {})
    for name, defn in rs.items():
        if isinstance(defn, dict) and name not in props:
            props[name] = defn
    out["properties"] = props
    out.pop("repeated_structure", None)
    return out


def _unwrap_citation_values(value: Any) -> Any:
    if isinstance(value, dict):
        if "value" in value and "citations" in value:
            return _unwrap_citation_values(value.get("value"))
        return {key: _unwrap_citation_values(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_unwrap_citation_values(item) for item in value]
    return value


def _coerce_positive_float(value: Any) -> float | None:
    if isinstance(value, (int, float)) and value > 0:
        return float(value)
    return None


def _coerce_nonnegative_float(value: Any) -> float | None:
    if isinstance(value, (int, float)) and math.isfinite(value) and value >= 0:
        return float(value)
    return None


def _attach_cost_stats(
    raw_output: dict[str, Any],
    pipeline_config: dict[str, Any],
    source_file_path: str | None = None,
) -> None:
    """Surface Reducto usage credits as top-level report stats.

    Standard Extract uses returned ``usage.credits``. Deep Extract uses one
    deterministic total: the greater of 30 credits or 4 credits per page plus
    0.1 credits per returned field, followed by 1 parse credit per page.
    """
    config = pipeline_config if isinstance(pipeline_config, dict) else {}
    response_config = raw_output.get("_config")
    response_config = response_config if isinstance(response_config, dict) else {}
    deep_extract = bool(config.get("deep_extract") or response_config.get("deep_extract"))

    usage = raw_output.get("usage")
    usage_dict = usage if isinstance(usage, dict) else {}
    num_pages = _coerce_positive_float(usage_dict.get("num_pages"))
    if num_pages is None and source_file_path:
        counted_pages = page_count(Path(source_file_path))
        if counted_pages > 0:
            num_pages = float(counted_pages)
    if num_pages is not None:
        raw_output["num_pages"] = num_pages

    reported_credits = _coerce_nonnegative_float(usage_dict.get("credits"))
    if reported_credits is not None:
        raw_output["reported_credits"] = reported_credits
    credits_used = reported_credits

    if deep_extract and num_pages is not None:
        num_fields_value = usage_dict.get("num_fields")
        if isinstance(num_fields_value, (int, float)) and num_fields_value >= 0:
            billable_fields = float(num_fields_value)
            deep_credits = max(
                REDUCTO_DEEP_MIN_CREDITS_PER_DOCUMENT,
                REDUCTO_DEEP_CREDITS_PER_PAGE * num_pages + REDUCTO_DEEP_CREDITS_PER_FIELD * billable_fields,
            )
            parse_credits = REDUCTO_PARSE_CREDITS_PER_PAGE * num_pages

            raw_output["deep_extract_billable_fields"] = billable_fields
            raw_output["deep_extract_credits"] = deep_credits
            raw_output["parse_credits"] = parse_credits
            raw_output["deep_extract_cost_calculated"] = True
            credits_used = deep_credits + parse_credits

    if credits_used is None:
        return

    raw_output["credits_used"] = credits_used
    raw_output["total_credits"] = credits_used

    cost_usd = credits_used * REDUCTO_USD_PER_CREDIT
    raw_output[COST_USD] = cost_usd
    if num_pages is not None:
        raw_output["credits_per_page"] = credits_used / num_pages
        raw_output[COST_PER_PAGE_USD] = cost_usd / num_pages


@register_provider("reducto_extract")
class ReductoExtractProvider(Provider):
    """
    Provider for Reducto EXTRACT.

    This provider uses the Reducto API for extraction tasks.
    According to https://docs.reducto.ai/extraction/extract-overview
    """

    _attach_cost_stats = staticmethod(_attach_cost_stats)

    def _record_cost(
        self,
        raw: dict[str, Any],
        config: dict[str, Any],
        component_id: str,
        source_file_path: str | None = None,
    ) -> dict[str, Any]:
        self._attach_cost_stats(raw, config, source_file_path)
        cost = raw.get(COST_USD)
        raw[RAW_COST_COMPLETE] = cost is not None
        snapshot = {KNOWN_COST_USD: cost, COMPLETE: cost is not None}
        if cost is not None:
            emit_provider_event(
                COST_COMPONENT_EVENT,
                **{
                    COMPONENT_ID: component_id,
                    COST_USD: cost,
                    USAGE: raw.get("usage"),
                },
            )
        return snapshot

    def __init__(self, provider_name: str, base_config: dict[str, Any] | None = None):
        """
        Initialize the provider.

        :param provider_name: Name of the provider
        :param base_config: Optional configuration with:
            - `api_key`: Reducto API key (defaults to REDUCTO_API_KEY env var)
            - `system_prompt`: System prompt for extraction (optional)
            - `citations`: Whether to enable citations (default: False)
            - `array_extract`: Whether to use array extraction for long documents (default: False)
        """
        super().__init__(provider_name, base_config)

        # Get API key
        self._api_key = self.base_config.get("api_key") or os.getenv("REDUCTO_API_KEY")
        if not self._api_key:
            raise ProviderConfigError(
                "Reducto API key is required. Set REDUCTO_API_KEY environment variable or pass api_key in base_config."
            )

        # Get configuration with defaults
        self._default_system_prompt = self.base_config.get("system_prompt")
        self._default_citations = self.base_config.get("citations", False)
        self._default_array_extract = self.base_config.get("array_extract", False)

        # Cancellation flags for in-flight deep-extract poll loops, keyed by
        # example_id. cancel() sets the flag from the runner's timeout thread;
        # the poll loop observes it on its next iteration and stops instead of
        # polling forever (and racing the retry with a duplicate billed job).
        self._cancel_events: dict[str, threading.Event] = {}
        self._cancel_lock = threading.Lock()

    def _build_extract_config(self, pipeline: PipelineSpec, request: InferenceRequest) -> dict[str, Any]:
        """
        Build extract configuration by merging pipeline config with request overrides.

        :param pipeline: Pipeline specification
        :param request: Inference request with optional config_override
        :return: Configuration dictionary for Reducto extract
        """
        # Start with pipeline config
        config_dict = dict(pipeline.config)

        # Merge with request config_override if provided
        if request.config_override:
            config_dict.update(request.config_override)

        return config_dict

    async def _extract_pdf_async(
        self,
        pdf_path: str,
        schema: dict[str, Any],
        system_prompt: str | None = None,
        citations: bool = False,
        array_extract: bool = False,
        deep_extract: bool = False,
        cancel_event: threading.Event | None = None,
        extract_model: str | None = None,
    ) -> dict[str, Any]:
        """
        Extract data from a PDF using Reducto API (async).

        :param pdf_path: Path to the PDF file
        :param schema: JSON schema for extraction
        :param system_prompt: Optional system prompt
        :param citations: Whether to enable citations
        :param array_extract: Whether to use array extraction
        :return: Raw API response as dictionary
        :raises ProviderError: For any API errors
        """
        budget = execution_budget()
        accounting: dict[str, Any] = {KNOWN_COST_USD: None, COMPLETE: False}
        last_response: dict[str, Any] = {}
        client = Reducto(api_key=self._api_key, timeout=budget.remaining(), max_retries=0)
        try:
            if cancel_event is not None and cancel_event.is_set():
                budget.cancel()
            upload = await budget.run_async(
                asyncio.to_thread(client.upload, file=Path(pdf_path), timeout=budget.remaining()),
            )
            instructions: dict[str, Any] = {"schema": _adapt_schema_for_reducto(schema)}
            if system_prompt:
                instructions["system_prompt"] = system_prompt
            settings: dict[str, Any] = {
                "citations": {"enabled": citations, "numerical_confidence": citations},
                "array_extract": array_extract,
                "deep_extract": deep_extract,
                "include_images": False,
                "optimize_for_latency": False,
            }
            if extract_model:
                settings["alpha"] = {"extract_model": extract_model}
            if cancel_event is not None and cancel_event.is_set():
                budget.cancel()
            if deep_extract:
                # Async jobs avoid expiring presigned tokens on long extracts.
                # Each stage shares the document deadline, including upload time.
                def submit_job() -> Any:
                    submitted = client.extract.run_job(
                        input=cast(Any, upload),
                        instructions=cast(Any, instructions),
                        settings=cast(Any, settings),
                        timeout=budget.remaining(),
                    )
                    emit_provider_event(
                        "vendor.job_created",
                        provider="reducto",
                        stage="extract",
                        job_id=submitted.job_id,
                    )
                    return submitted

                submitted = await budget.run_async(asyncio.to_thread(submit_job))
                job_id = submitted.job_id
                try:
                    while True:
                        if cancel_event is not None and cancel_event.is_set():
                            budget.cancel()

                        def poll_job() -> Any:
                            nonlocal accounting, last_response
                            job = client.job.get(job_id, timeout=budget.remaining())
                            last_response = job.model_dump()
                            if job.status in ("Completed", "Failed", "Cancelled"):
                                result_payload = last_response.get("result")
                                cost_payload = result_payload if isinstance(result_payload, dict) else last_response
                                accounting = self._record_cost(
                                    cost_payload,
                                    {"deep_extract": deep_extract and job.status == "Completed"},
                                    f"reducto/{job_id}",
                                    pdf_path if job.status == "Completed" else None,
                                )
                            if job.status == "Completed":
                                emit_provider_event(
                                    "stage.completed",
                                    provider="reducto",
                                    stage="extract",
                                    job_id=job_id,
                                    raw_output=last_response,
                                )
                            return job

                        job = await budget.run_async(asyncio.to_thread(poll_job))
                        status = getattr(job, "status", None)
                        if status == "Completed":
                            result = job.result
                            break
                        if status in ("Failed", "Cancelled"):
                            reason = getattr(job, "reason", None) or "unknown"
                            raise ProviderTransientError(
                                f"Reducto deep-extract job {job_id} {status}: {reason}",
                                job_id=job_id,
                                debug_payload={"response": last_response, COST_ACCOUNTING: accounting},
                            )
                        await budget.run_async(asyncio.sleep(self._poll_interval))
                except (TimeoutError, asyncio.CancelledError):
                    # Cancellation is cleanup: give it a bounded request after
                    # extraction time is exhausted, without submitting new work.
                    emit_provider_event("vendor.cancellation_requested", provider="reducto", job_id=job_id)
                    try:
                        cancellation = await asyncio.to_thread(client.job.cancel, job_id, timeout=10)
                        emit_provider_event(
                            "vendor.cancellation_response",
                            provider="reducto",
                            job_id=job_id,
                            response=cancellation,
                        )
                        if isinstance(cancellation, dict):
                            result_payload = cancellation.get("result")
                            cost_payload = result_payload if isinstance(result_payload, dict) else cancellation
                            if isinstance(cost_payload.get("usage"), dict):
                                accounting = self._record_cost(cost_payload, {}, f"reducto/{job_id}")
                    except Exception as cancel_error:
                        logging.getLogger(__name__).warning("Reducto job cancellation failed", exc_info=True)
                        emit_provider_event(
                            "vendor.cancellation_failed",
                            provider="reducto",
                            job_id=job_id,
                            error=str(cancel_error),
                        )
                    else:
                        emit_provider_event("vendor.cancellation_completed", provider="reducto", job_id=job_id)
                    raise
            else:

                def extract_sync() -> Any:
                    nonlocal accounting, last_response
                    result = cast(Any, client.extract.run)(
                        input=upload,
                        instructions=instructions,
                        settings=settings,
                        timeout=budget.remaining(),
                    )
                    last_response = result.model_dump()
                    # Repeated component IDs replace cumulative charges within an attempt.
                    # The synchronous SDK result has an optional job_id. Use it when
                    # present; a local ID keeps separate calls distinct when absent.
                    response_id = last_response.get("job_id")
                    component_id = f"reducto/{response_id if response_id is not None else uuid4().hex}"
                    accounting = self._record_cost(last_response, {}, component_id, pdf_path)
                    emit_provider_event(
                        "stage.completed",
                        provider="reducto",
                        stage="extract",
                        raw_output=last_response,
                    )
                    return result

                result = await budget.run_async(asyncio.to_thread(extract_sync))

            # Capture the original Reducto API response as-is. The result is a
            # union of response models (or None for a job that completed with
            # no payload); the hasattr duck-typing below handles every shape,
            # so widen to Any rather than narrowing each union member.
            result_data: Any = result
            raw_response: dict[str, Any]
            try:
                # Try Pydantic v2 first
                if hasattr(result_data, "model_dump"):
                    raw_response = result_data.model_dump()
                # Try Pydantic v1
                elif hasattr(result_data, "dict"):
                    raw_response = result_data.dict()
                else:
                    # Fallback: manually extract if not a Pydantic model
                    raw_response = {}
                    for attr in ["job_id", "duration", "pdf_url", "studio_link", "usage", "data"]:
                        if hasattr(result, attr):
                            value = getattr(result, attr)
                            if not callable(value):
                                raw_response[attr] = value
            except Exception:
                # If model_dump fails, fall back to manual extraction
                raw_response = {}
                for attr in ["job_id", "duration", "pdf_url", "studio_link", "usage", "data"]:
                    if hasattr(result, attr):
                        value = getattr(result, attr)
                        if not callable(value):
                            raw_response[attr] = value

            # Also store the configuration used for reference
            raw_response[RAW_COST_COMPLETE] = accounting[COMPLETE]
            raw_response[COST_USD] = accounting[KNOWN_COST_USD]
            raw_response["_config"] = {
                "system_prompt": system_prompt,
                "citations": citations,
                "array_extract": array_extract,
                "deep_extract": deep_extract,
                "extract_model": extract_model,
            }

            return raw_response

        except (TimeoutError, ProviderPermanentError):
            raise
        except Exception as e:
            # Check if it's a transient error (network, timeout, etc.)
            if isinstance(e, ProviderTransientError):
                raise
            debug_payload = {"response": last_response, COST_ACCOUNTING: accounting}
            error_str = str(e).lower()
            # 401 "invalid access token" is a presigned-token expiry on slow
            # (deep) jobs, not a real auth failure — retrying re-uploads with
            # a fresh token. 429s are vendor rate limits.
            transient_keywords = [
                "timeout",
                "network",
                "connection",
                "503",
                "502",
                "504",
                "429",
                "rate limit",
                "invalid access token",
            ]
            if any(keyword in error_str for keyword in transient_keywords):
                raise ProviderTransientError(
                    f"Transient error during extraction: {e}",
                    debug_payload=debug_payload,
                ) from e
            else:
                raise ProviderPermanentError(
                    f"Error during extraction: {e}",
                    debug_payload=debug_payload,
                ) from e
        finally:
            client.close()

    def run_inference(self, pipeline: PipelineSpec, request: InferenceRequest) -> RawInferenceResult:
        """
        Run inference and return raw results.

        :param pipeline: Pipeline specification
        :param request: Inference request (must include schema_override for EXTRACT)
        :return: Raw inference result
        :raises ProviderError: For any provider-related failures
        """
        if request.product_type != ProductType.EXTRACT:
            raise ProviderPermanentError(
                f"ReductoExtractProvider only supports EXTRACT product type, got {request.product_type}"
            )

        # Schema is required for extraction
        if not request.schema_override:
            raise ProviderPermanentError(
                "schema_override is required for EXTRACT product type. "
                "Provide a JSON schema in InferenceRequest.schema_override"
            )

        started_at = datetime.now()

        # Check if file exists
        file_path = Path(request.source_file_path)
        if not file_path.exists():
            raise ProviderPermanentError(f"File not found: {file_path}")

        try:
            # Build extract config
            extract_config = self._build_extract_config(pipeline, request)

            # Get system_prompt from config or default
            system_prompt = extract_config.get("system_prompt") or self._default_system_prompt

            # Get citations setting from config or default
            citations = extract_config.get("citations", self._default_citations)

            # Get array_extract setting from config or default
            array_extract = extract_config.get("array_extract", self._default_array_extract)

            # Get deep_extract setting from config (default off; deep_extract is
            # Reducto's agentic high-quality mode and bills at a separate rate).
            deep_extract = extract_config.get("deep_extract", False)

            # Alpha extract model version (e.g. "v4"); omitted -> Reducto default.
            extract_model = extract_config.get("extract_model")

            # Run extraction (async), registering a cancel flag so a runner
            # timeout can stop the deep-extract poll loop.
            cancel_event = threading.Event()
            with self._cancel_lock:
                self._cancel_events[request.example_id] = cancel_event
            try:
                raw_output = asyncio.run(
                    self._extract_pdf_async(
                        pdf_path=str(file_path),
                        schema=request.schema_override,
                        system_prompt=system_prompt,
                        citations=citations,
                        array_extract=array_extract,
                        deep_extract=deep_extract,
                        cancel_event=cancel_event,
                        extract_model=extract_model,
                    )
                )
            finally:
                with self._cancel_lock:
                    self._cancel_events.pop(request.example_id, None)

            completed_at = datetime.now()
            latency_ms = int((completed_at - started_at).total_seconds() * 1000)

            self._attach_cost_stats(
                raw_output,
                extract_config,
                str(file_path),
            )

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

        except ProviderPermanentError:
            # Re-raise provider errors as-is
            raise
        except ProviderTransientError:
            # Re-raise provider errors as-is
            raise
        except Exception as e:
            # Wrap unexpected errors
            raise ProviderPermanentError(f"Unexpected error during inference: {e}") from e

    def cancel(self, example_id: str) -> bool:
        """Signal cancellation before another upload, extraction, or job poll."""
        with self._cancel_lock:
            event = self._cancel_events.get(example_id)
        if event is None:
            return False
        event.set()
        return True

    def normalize(self, raw_result: RawInferenceResult) -> InferenceResult:
        """
        Normalize raw inference result to produce ExtractOutput.

        :param raw_result: Raw inference result from run_inference()
        :return: Inference result with both raw and normalized outputs
        :raises ProviderError: For any normalization failures
        """
        if raw_result.product_type != ProductType.EXTRACT:
            raise ProviderPermanentError(
                f"ReductoExtractProvider only supports EXTRACT product type, got {raw_result.product_type}"
            )

        extract_config = dict(raw_result.pipeline.config)
        if raw_result.request.config_override:
            extract_config.update(raw_result.request.config_override)

        self._attach_cost_stats(
            raw_result.raw_output,
            extract_config,
            raw_result.request.source_file_path,
        )

        # Extract the structured data from the response
        # Reducto returns data in the "result" field as a list
        result = raw_result.raw_output.get("result", [])
        # If result is a list, take the first element (should be a dict)
        if isinstance(result, list) and len(result) > 0:
            extracted_data = _unwrap_citation_values(result[0])
        elif isinstance(result, dict):
            # Fallback: if it's already a dict, use it directly
            extracted_data = _unwrap_citation_values(result)
        else:
            # Empty list or unexpected type
            extracted_data = {}

        output = ExtractOutput(
            task_type="extract",
            example_id=raw_result.request.example_id,
            pipeline_name=raw_result.pipeline_name,
            extracted_data=extracted_data,
            field_citations=list(extract_reducto_field_citations(raw_result.raw_output)),
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
