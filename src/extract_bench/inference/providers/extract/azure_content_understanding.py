"""Provider for Azure Content Understanding GA EXTRACT."""

import base64
import hashlib
import json
import math
import mimetypes
import os
import re
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from threading import Lock
from time import perf_counter
from typing import Any
from urllib.parse import urlencode

import requests

from extract_bench.inference.providers.azure_content_understanding_common import (
    DEFAULT_API_VERSION as _DEFAULT_API_VERSION,
)
from extract_bench.inference.providers.azure_content_understanding_common import (
    DEFAULT_POLL_INTERVAL_S,
    DEFAULT_POLL_TIMEOUT_S,
    content_understanding_request,
    poll_deadline,
    raise_operation_failure,
    sleep_for_poll_interval,
)
from extract_bench.inference.providers.azure_content_understanding_common import (
    TERMINAL_STATUSES as _TERMINAL_STATUSES,
)
from extract_bench.inference.providers.azure_content_understanding_common import (
    analysis_result as _analysis_result,
)
from extract_bench.inference.providers.azure_content_understanding_common import (
    apply_usage_cost_fields as _apply_usage_cost_fields,
)
from extract_bench.inference.providers.azure_content_understanding_common import (
    content_understanding_endpoint as _content_understanding_endpoint,
)
from extract_bench.inference.providers.azure_content_understanding_common import (
    first_document_content as _first_document_content,
)
from extract_bench.inference.providers.azure_content_understanding_common import (
    public_config as _public_config,
)
from extract_bench.inference.providers.azure_content_understanding_common import (
    response_json as _response_json,
)
from extract_bench.inference.providers.base import (
    Provider,
    ProviderConfigError,
    ProviderPermanentError,
    ProviderRateLimitError,
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

_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")
_SOURCE_REGION_RE = re.compile(r"D\(([^)]*)\)")


@register_provider("azure_content_understanding")
class AzureContentUnderstandingProvider(Provider):
    """Run Azure Content Understanding analyzer-backed extraction."""

    def __init__(self, provider_name: str, base_config: dict[str, Any] | None = None):
        super().__init__(provider_name, base_config)

        endpoint = self.base_config.get("endpoint") or os.getenv("AZURE_CONTENT_UNDERSTANDING_ENDPOINT")
        api_key = self.base_config.get("api_key") or os.getenv("AZURE_CONTENT_UNDERSTANDING_KEY")
        if endpoint is None or endpoint == "":
            raise ProviderConfigError(
                "Azure Content Understanding endpoint is required. "
                "Set AZURE_CONTENT_UNDERSTANDING_ENDPOINT or pass endpoint in config."
            )
        if api_key is None or api_key == "":
            raise ProviderConfigError(
                "Azure Content Understanding API key is required. "
                "Set AZURE_CONTENT_UNDERSTANDING_KEY or pass api_key in config."
            )

        self._endpoint = _content_understanding_endpoint(str(endpoint)).rstrip("/")
        self._api_key = str(api_key)
        self._api_version = str(self.base_config.get("api_version", _DEFAULT_API_VERSION))
        self._timeout = float(self.base_config.get("timeout", 300))
        self._poll_interval = float(self.base_config.get("poll_interval", DEFAULT_POLL_INTERVAL_S))
        self._poll_timeout = float(self.base_config.get("poll_timeout", DEFAULT_POLL_TIMEOUT_S))
        self._session = requests.Session()
        self._analyzer_locks_guard = Lock()
        self._analyzer_locks: dict[str, Lock] = {}

    def run_inference(self, pipeline: PipelineSpec, request: InferenceRequest) -> RawInferenceResult:
        if request.product_type != ProductType.EXTRACT:
            raise ProviderPermanentError(
                f"AzureContentUnderstandingProvider only supports EXTRACT, got {request.product_type}"
            )
        if not request.schema_override:
            raise ProviderPermanentError("schema_override is required for EXTRACT. Provide a JSON schema.")

        file_path = Path(request.source_file_path)
        if not file_path.exists():
            raise ProviderPermanentError(f"File not found: {file_path}")

        started_at = datetime.now()
        total_started = perf_counter()
        try:
            analyzer_id, analyzer_payload, field_name_map, analyzer_outcome = self._ensure_analyzer(
                pipeline, request.schema_override
            )
            analyzer_ready = perf_counter()
            raw_output = self._analyze_binary(analyzer_id, file_path)
            document_completed = perf_counter()
            processing_duration_ms = round((document_completed - analyzer_ready) * 1000)
            raw_output["_analyzer_setup"] = {"outcome": analyzer_outcome}
            raw_output["_analyzer_id"] = analyzer_id
            raw_output["_analyzer_payload"] = analyzer_payload
            raw_output["_field_name_map"] = field_name_map
            raw_output["_config"] = _public_config(pipeline.config)
            _apply_usage_cost_fields(raw_output, pipeline.config)

            total_completed = perf_counter()
            timings = raw_output.setdefault("_client_timings", {})
            timings.update(
                analyzer_setup_and_readiness_ms=round((analyzer_ready - total_started) * 1000),
                document_processing_ms=processing_duration_ms,
                postprocessing_ms=round((total_completed - document_completed) * 1000),
                inference_total_ms=round((total_completed - total_started) * 1000),
            )
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
                processing_duration_ms=processing_duration_ms,
                source="acu_client_document_processing",
                availability={"processing": True},
            )
        except (ProviderPermanentError, ProviderRateLimitError, ProviderTransientError):
            raise
        except Exception as exc:
            raise ProviderPermanentError(
                f"Unexpected error during Azure Content Understanding extraction: {exc}"
            ) from exc

    def normalize(self, raw_result: RawInferenceResult) -> InferenceResult:
        if raw_result.product_type != ProductType.EXTRACT:
            raise ProviderPermanentError(
                f"AzureContentUnderstandingProvider only supports EXTRACT, got {raw_result.product_type}"
            )

        normalization_started = perf_counter()
        result = _analysis_result(raw_result.raw_output)
        content = _first_document_content(result)
        fields = content.get("fields") if isinstance(content, Mapping) else {}
        fields = _restore_field_names(fields, raw_result.raw_output.get("_field_name_map"))
        extracted_data = _field_value({"type": "object", "valueObject": fields})
        page_sizes = _page_sizes(content)
        document_text = _document_text(content)
        citations = _field_citations(fields, page_sizes, document_text=document_text)

        output = ExtractOutput(
            task_type="extract",
            example_id=raw_result.request.example_id,
            pipeline_name=raw_result.pipeline_name,
            extracted_data=extracted_data if isinstance(extracted_data, (dict, list)) else {},
            field_citations=citations,
        )
        if "_client_timings" in raw_result.raw_output:
            raw_result.raw_output["_client_timings"]["normalization_ms"] = round(
                (perf_counter() - normalization_started) * 1000
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
            processing_duration_ms=raw_result.processing_duration_ms,
            retrieval_duration_ms=raw_result.retrieval_duration_ms,
            source=raw_result.source,
            availability=raw_result.availability,
        )

    def _build_analyzer(self, pipeline: PipelineSpec, schema: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        return _build_analyzer_payload_and_field_map(schema, pipeline.config)

    def _ensure_analyzer(
        self, pipeline: PipelineSpec, schema: dict[str, Any]
    ) -> tuple[str, dict[str, Any], dict[str, Any], str]:
        analyzer_payload, field_name_map = self._build_analyzer(pipeline, schema)
        config_hash = hashlib.sha256(
            json.dumps(analyzer_payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()[:24]
        prefix = _analyzer_id_part(str(pipeline.config.get("analyzer_id_prefix", pipeline.pipeline_name)))[:32]
        analyzer_id = f"bench{prefix}{config_hash}"[:64]
        with self._analyzer_locks_guard:
            lock = self._analyzer_locks.setdefault(analyzer_id, Lock())
        # The runner shares a provider across its document threads. Serialize only
        # setup for the same analyzer; document analysis stays concurrent.
        with lock:
            url = self._url(f"/contentunderstanding/analyzers/{analyzer_id}")
            response = self._request("GET", url, context="get reusable analyzer", accepted_status_codes=(404,))
            outcome = "reused_ready"
            if response.status_code == 404:
                response = self._request(
                    "PUT",
                    self._url(f"/contentunderstanding/analyzers/{analyzer_id}", {"allowReplace": "false"}),
                    json=analyzer_payload,
                    headers={"Content-Type": "application/json"},
                    context="create analyzer",
                    accepted_status_codes=(409,),
                )
                # A different process can create the same content-addressed ID
                # between our GET and PUT. Verify it rather than replacing it.
                outcome = "reused_after_create_conflict" if response.status_code == 409 else "created"
                if response.status_code != 409:
                    operation_url = response.headers.get("Operation-Location")
                    if operation_url is not None and operation_url != "":
                        self._poll_operation(operation_url, context=f"create analyzer {analyzer_id}")
                response = self._request("GET", url, context="verify created analyzer")
            body = _response_json(response)
            if outcome == "reused_ready" and body.get("status") == "creating":
                outcome = "reused_after_readiness_wait"
            self._wait_for_matching_analyzer(analyzer_id, analyzer_payload, body)
        return analyzer_id, analyzer_payload, field_name_map, outcome

    def _wait_for_matching_analyzer(
        self, analyzer_id: str, expected: Mapping[str, Any], body: Mapping[str, Any]
    ) -> None:
        deadline = poll_deadline(self._poll_timeout)
        while True:
            if body.get("analyzerId") != analyzer_id or not _analyzer_definition_matches(body, expected):
                raise ProviderPermanentError(
                    f"Azure Content Understanding analyzer {analyzer_id} does not match the requested definition"
                )
            status = str(body.get("status", "")).lower()
            if status == "ready":
                return
            if status == "failed":
                raise_operation_failure(body, context=f"analyzer {analyzer_id}")
            if status != "creating":
                raise ProviderPermanentError(
                    f"Azure Content Understanding analyzer {analyzer_id} has status {status!r}"
                )
            if not sleep_for_poll_interval(deadline, self._poll_interval):
                raise ProviderTransientError(
                    f"Timed out waiting for Azure Content Understanding analyzer {analyzer_id}"
                )
            response = self._request(
                "GET", self._url(f"/contentunderstanding/analyzers/{analyzer_id}"), context="wait for reusable analyzer"
            )
            body = _response_json(response)

    def _analyze_binary(self, analyzer_id: str, file_path: Path) -> dict[str, Any]:
        preparation_started = perf_counter()
        mime_type = mimetypes.guess_type(file_path.name)[0] or "application/octet-stream"
        url = self._url(f"/contentunderstanding/analyzers/{analyzer_id}:analyze")
        payload = {
            "inputs": [
                {
                    "name": file_path.name,
                    "mimeType": mime_type,
                    "data": base64.b64encode(file_path.read_bytes()).decode("ascii"),
                }
            ]
        }
        submission_started = perf_counter()
        response = self._request(
            "POST",
            url,
            json=payload,
            headers={"Content-Type": "application/json"},
            context="analyze",
        )
        operation_url = response.headers.get("Operation-Location")
        body = _response_json(response)
        submission_completed = perf_counter()
        if operation_url is not None and operation_url != "":
            body = self._poll_operation(operation_url, context=f"analyze {file_path.name}")
        polling_completed = perf_counter()
        status = str(body.get("status", "")).lower()
        if status in _TERMINAL_STATUSES - {"succeeded"} or body.get("error"):
            raise_operation_failure(body, context=f"analyze {file_path.name}")
        content = _first_document_content(_analysis_result(body))
        if (
            status not in {"", "succeeded"}
            or content.get("kind", "document") != "document"
            or not isinstance(content.get("fields"), Mapping)
        ):
            raise ProviderTransientError(
                "Azure Content Understanding analyze returned an incomplete or malformed result"
            )
        body["_client_timings"] = {
            "document_preparation_ms": round((submission_started - preparation_started) * 1000),
            "document_submission_ms": round((submission_completed - submission_started) * 1000),
            "document_polling_ms": round((polling_completed - submission_completed) * 1000)
            if operation_url is not None and operation_url != ""
            else 0,
        }
        return body

    def _poll_analyzer_ready(self, analyzer_id: str) -> None:
        url = self._url(f"/contentunderstanding/analyzers/{analyzer_id}")
        deadline = poll_deadline(self._poll_timeout)
        while True:
            response = self._request("GET", url, context=f"get analyzer {analyzer_id}")
            body = _response_json(response)
            status = str(body.get("status", "")).lower()
            if status == "ready":
                return
            if status == "failed":
                raise_operation_failure(body, context=f"analyzer {analyzer_id}")
            if not sleep_for_poll_interval(deadline, self._poll_interval):
                break
        raise ProviderTransientError(f"Timed out waiting for Azure Content Understanding analyzer {analyzer_id}")

    def _poll_operation(self, operation_url: str, *, context: str) -> dict[str, Any]:
        deadline = poll_deadline(self._poll_timeout)
        while True:
            response = self._request(
                "GET",
                operation_url,
                context=context,
                headers={"x-ms-diagnostics": "true"},
            )
            body = _response_json(response)
            status = str(body.get("status", "")).lower()
            if status in _TERMINAL_STATUSES:
                if status == "succeeded":
                    return body
                raise_operation_failure(body, context=context)
            if not sleep_for_poll_interval(deadline, self._poll_interval):
                break
        raise ProviderTransientError(f"Timed out waiting for Azure Content Understanding {context}")

    def _url(self, path: str, extra_params: Mapping[str, str] | None = None) -> str:
        params = {"api-version": self._api_version}
        if extra_params is not None and len(extra_params) > 0:
            params.update(extra_params)
        return f"{self._endpoint}{path}?{urlencode(params)}"

    def _request(self, method: str, url: str, *, context: str, **kwargs: Any) -> requests.Response:
        headers = dict(kwargs.pop("headers", {}) or {})
        headers["Ocp-Apim-Subscription-Key"] = self._api_key
        return content_understanding_request(
            self._session,
            method,
            url,
            headers=headers,
            timeout=self._timeout,
            context=context,
            default_retry_after_s=self._poll_interval,
            **kwargs,
        )


def _analyzer_definition_matches(body: Mapping[str, Any], expected: Mapping[str, Any]) -> bool:
    """Match all writable properties, allowing only documented default additions.

    GET includes read-only properties such as status and supportedModels. Azure
    may also expand omitted defaults; unknown non-null settings fail closed.
    """
    read_only = {"analyzerId", "status", "createdAt", "lastModifiedAt", "supportedModels", "warnings"}
    defaults: dict[str, Any] = {"dynamicFieldSchema": False, "processingLocation": "global"}
    observed = {key: value for key, value in body.items() if key not in read_only and value is not None}
    requested = dict(expected)
    for key, value in defaults.items():
        if key not in requested and observed.get(key) == value:
            observed.pop(key)
    config = observed.get("config")
    if isinstance(config, Mapping):
        config = {key: value for key, value in config.items() if value is not None}
        if "chartFormat" not in expected.get("config", {}) and config.get("chartFormat") == "chartJs":
            config.pop("chartFormat")
        observed["config"] = config
    return observed == requested


def _build_analyzer_payload(schema: dict[str, Any], config: Mapping[str, Any]) -> dict[str, Any]:
    payload, _ = _build_analyzer_payload_and_field_map(schema, config)
    return payload


def _build_analyzer_payload_and_field_map(
    schema: dict[str, Any], config: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    from extract_bench.inference.providers.extract.azure_content_understanding_extractbench import (
        _schema_fields,
    )

    fields, field_name_map = _schema_fields(schema, str(config.get("field_method", "extract")), strict_unions=False)
    payload = {
        "baseAnalyzerId": config.get("base_analyzer_id", "prebuilt-document"),
        "config": _content_config(config),
        "fieldSchema": {
            "name": str(config.get("field_schema_name", "BenchExtractSchema")),
            "description": str(config.get("field_schema_description", "Benchmark extraction schema")),
            "fields": fields,
            "definitions": {},
        },
    }
    models = config.get("models")
    if isinstance(models, Mapping):
        payload["models"] = {str(key): str(value) for key, value in models.items()}
    return payload, field_name_map


def _analyzer_id_part(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "", value) or "acu"


def _content_config(config: Mapping[str, Any]) -> dict[str, Any]:
    annotation_format = "markdown" if bool(config.get("enable_annotations", True)) else "none"
    return {
        "enableOcr": bool(config.get("enable_ocr", True)),
        "enableLayout": bool(config.get("enable_layout", True)),
        "enableFormula": bool(config.get("enable_formula", True)),
        "enableBarcode": bool(config.get("enable_barcode", True)),
        "returnDetails": bool(config.get("return_details", True)),
        "omitContent": bool(config.get("omit_content", False)),
        "tableFormat": str(config.get("table_format", "html")),
        "enableFigureDescription": bool(config.get("enable_figure_description", False)),
        "enableFigureAnalysis": bool(config.get("enable_figure_analysis", False)),
        "annotationFormat": annotation_format,
        "enableSegment": bool(config.get("enable_segment", False)),
        "segmentPerPage": bool(config.get("segment_per_page", False)),
        "estimateFieldSourceAndConfidence": bool(config.get("estimate_field_source_and_confidence", True)),
    }


def _safe_field_name(value: str, used_names: set[str]) -> str:
    parts = re.findall(r"[A-Za-z0-9]+", value)
    candidate = "".join(part[:1].upper() + part[1:] for part in parts) or "Field"
    if not candidate[0].isalpha():
        candidate = f"Field{candidate}"
    if candidate not in used_names:
        used_names.add(candidate)
        return candidate
    suffix = hashlib.sha1(value.encode()).hexdigest()[:8]
    deduped = f"{candidate}{suffix}"
    counter = 2
    while deduped in used_names:
        deduped = f"{candidate}{suffix}{counter}"
        counter += 1
    used_names.add(deduped)
    return deduped


def _field_value(field: Any) -> Any:
    if not isinstance(field, Mapping):
        return field
    field_type = field.get("type")
    if field_type == "object":
        value = field.get("valueObject")
        if isinstance(value, Mapping):
            return {str(key): _field_value(child) for key, child in value.items()}
        return {}
    if field_type == "array":
        value = field.get("valueArray")
        if isinstance(value, list):
            return [_field_value(item) for item in value]
        return []
    for key in ("valueString", "valueNumber", "valueInteger", "valueBoolean", "valueDate", "valueTime", "valueJson"):
        if key in field:
            return field[key]
    return None


def _restore_field_names(fields: Any, field_name_map: Any) -> Any:
    if not isinstance(fields, Mapping) or not isinstance(field_name_map, Mapping):
        return fields
    restored: dict[str, Any] = {}
    for safe_name, field in fields.items():
        mapping = field_name_map.get(str(safe_name))
        original_name = mapping.get("name") if isinstance(mapping, Mapping) else safe_name
        children = mapping.get("children") if isinstance(mapping, Mapping) else None
        restored[str(original_name)] = _restore_child_field_names(field, children)
    return restored


def _restore_child_field_names(field: Any, field_name_map: Any) -> Any:
    if not isinstance(field, Mapping):
        return field
    restored = dict(field)
    if restored.get("type") == "object" and isinstance(restored.get("valueObject"), Mapping):
        restored["valueObject"] = _restore_field_names(restored["valueObject"], field_name_map)
    elif restored.get("type") == "array" and isinstance(restored.get("valueArray"), list):
        restored["valueArray"] = [_restore_child_field_names(item, field_name_map) for item in restored["valueArray"]]
    return restored


def _page_sizes(content: Mapping[str, Any]) -> dict[int, tuple[float, float]]:
    sizes: dict[int, tuple[float, float]] = {}
    pages = content.get("pages")
    if not isinstance(pages, list):
        return sizes
    for page in pages:
        if not isinstance(page, Mapping):
            continue
        page_number = page.get("pageNumber")
        width = page.get("width")
        height = page.get("height")
        if isinstance(page_number, int) and isinstance(width, (int, float)) and isinstance(height, (int, float)):
            sizes[page_number] = (float(width), float(height))
    return sizes


def _document_text(content: Mapping[str, Any]) -> str | None:
    for key in ("markdown", "text", "content"):
        value = content.get(key)
        if isinstance(value, str):
            return value
    return None


def _field_citations(
    fields: Any,
    page_sizes: Mapping[int, tuple[float, float]],
    path: str = "",
    *,
    document_text: str | None = None,
) -> list[FieldCitation]:
    citations: list[FieldCitation] = []
    if not isinstance(fields, Mapping):
        return citations
    for name, field in fields.items():
        field_path = f"{path}.{name}" if len(path) > 0 else str(name)
        citations.extend(_field_citations_for_field(field_path, field, page_sizes, document_text=document_text))
    return citations


def _field_citations_for_field(
    field_path: str,
    field: Any,
    page_sizes: Mapping[int, tuple[float, float]],
    *,
    document_text: str | None = None,
) -> list[FieldCitation]:
    if not isinstance(field, Mapping):
        return []
    field_type = field.get("type")
    if field_type == "object" and isinstance(field.get("valueObject"), Mapping):
        return _field_citations(field["valueObject"], page_sizes, field_path, document_text=document_text)
    if field_type == "array" and isinstance(field.get("valueArray"), list):
        citations: list[FieldCitation] = []
        for index, item in enumerate(field["valueArray"]):
            citations.extend(
                _field_citations_for_field(f"{field_path}[{index}]", item, page_sizes, document_text=document_text)
            )
        return citations
    return _citations_from_field(field_path, field, page_sizes, document_text=document_text)


def _citations_from_field(
    field_path: str,
    field: Mapping[str, Any],
    page_sizes: Mapping[int, tuple[float, float]],
    *,
    document_text: str | None = None,
) -> list[FieldCitation]:
    source = field.get("source")
    if not isinstance(source, str) or len(source) == 0:
        return []
    selected_region_indexes, reference_text = _selected_source_region_indexes(field, document_text)
    regions_by_page: dict[int, set[int]] = {}
    for index, match in enumerate(_SOURCE_REGION_RE.finditer(source)):
        if selected_region_indexes is not None and index not in selected_region_indexes:
            continue
        region = _source_region_bbox(match.group(1))
        if region is not None:
            regions_by_page.setdefault(region[0], set()).add(index)
    # Keep the existing bare-source and missing-geometry handling.
    span_texts = _field_span_texts(field, document_text) if selected_region_indexes is not None else []
    page_groups: list[set[int] | None] = list(regions_by_page.values()) or [selected_region_indexes]
    citations = []
    for indexes in page_groups:
        page, bbox = _parse_source_bbox(source, page_sizes, selected_region_indexes=indexes)
        page_reference_text = " ".join(
            text.strip() for index, text in span_texts if indexes is None or index in indexes
        )
        citations.append(
            FieldCitation(
                field_path=field_path,
                page=page or 1,
                bbox=bbox,
                reference_text=page_reference_text or reference_text or _reference_text(field),
                confidence=float(field["confidence"]) if isinstance(field.get("confidence"), (int, float)) else None,
                source="azure_content_understanding",
                metadata={"source": source, "spans": field.get("spans")},
            )
        )
    return citations


def _reference_text(field: Mapping[str, Any]) -> str | None:
    value = _field_value(field)
    if value is None:
        return None
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


def _parse_source_bbox(
    source: str,
    page_sizes: Mapping[int, tuple[float, float]],
    *,
    selected_region_indexes: set[int] | None = None,
) -> tuple[int | None, list[float] | None]:
    regions: list[tuple[int, float, float, float, float]] = []
    for index, match in enumerate(_SOURCE_REGION_RE.finditer(source)):
        if selected_region_indexes is not None and index not in selected_region_indexes:
            continue
        region = _source_region_bbox(match.group(1))
        if region is not None:
            regions.append(region)
    if len(regions) == 0:
        region = _source_region_bbox(source)
        if region is not None:
            regions.append(region)
    if len(regions) == 0:
        return None, None

    page = regions[0][0]
    same_page_regions = [region for region in regions if region[0] == page]
    x0 = min(region[1] for region in same_page_regions)
    y0 = min(region[2] for region in same_page_regions)
    x1 = max(region[3] for region in same_page_regions)
    y1 = max(region[4] for region in same_page_regions)
    width, height = page_sizes.get(page, (1.0, 1.0))
    if width > 0 and height > 0 and math.isfinite(width) and math.isfinite(height):
        x0, x1 = x0 / width, x1 / width
        y0, y1 = y0 / height, y1 / height
    x0 = min(1.0, max(0.0, x0))
    y0 = min(1.0, max(0.0, y0))
    x1 = min(1.0, max(0.0, x1))
    y1 = min(1.0, max(0.0, y1))
    return page, [x0, y0, max(0.0, x1 - x0), max(0.0, y1 - y0)]


def _selected_source_region_indexes(
    field: Mapping[str, Any], document_text: str | None
) -> tuple[set[int] | None, str | None]:
    span_texts = _field_span_texts(field, document_text)
    if len(span_texts) == 0:
        return None, None

    value = _field_value(field)
    value_text = _reference_text(field)
    selected: set[int] = set()
    selected_texts: list[str] = []
    for index, text in span_texts:
        if _span_matches_value(text, value, value_text):
            selected.add(index)
            selected_texts.append(text.strip())

    if len(selected) == 0:
        return None, None
    reference_text = " ".join(text for text in selected_texts if len(text) > 0)
    return selected, reference_text or None


def _field_span_texts(field: Mapping[str, Any], document_text: str | None) -> list[tuple[int, str]]:
    if document_text is None or len(document_text) == 0:
        return []
    spans = field.get("spans")
    if not isinstance(spans, list):
        return []
    out: list[tuple[int, str]] = []
    for index, span in enumerate(spans):
        if not isinstance(span, Mapping):
            continue
        offset = span.get("offset")
        length = span.get("length")
        if not isinstance(offset, int) or not isinstance(length, int) or offset < 0 or length <= 0:
            continue
        text = document_text[offset : offset + length]
        if len(text) > 0:
            out.append((index, text))
    return out


def _span_matches_value(span_text: str, value: Any, value_text: str | None) -> bool:
    span_norm = _match_text(span_text)
    if len(span_norm) == 0:
        return False
    if isinstance(value, bool):
        accepted = {"true", "yes", "y", "checked"} if value else {"false", "no", "n", "unchecked"}
        return span_norm in accepted
    if value_text is None:
        return False
    value_norm = _match_text(value_text)
    if len(value_norm) == 0:
        return False
    return span_norm == value_norm or span_norm in value_norm or value_norm in span_norm


def _match_text(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def _source_region_bbox(source_region: str) -> tuple[int, float, float, float, float] | None:
    numbers = [float(match.group(0)) for match in _NUMBER_RE.finditer(source_region)]
    if len(numbers) < 5:
        return None
    page = int(numbers[0])
    coords = numbers[1:]
    if len(coords) >= 8:
        xs = coords[0::2]
        ys = coords[1::2]
        return page, min(xs), min(ys), max(xs), max(ys)
    x0, y0, width, height = coords[:4]
    return page, x0, y0, x0 + width, y0 + height
