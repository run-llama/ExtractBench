"""ExtractBench-native Azure Content Understanding provider."""

import os
from collections.abc import Mapping
from typing import Any

from extract_bench.inference.providers.base import (
    ProviderConfigError,
    ProviderPermanentError,
)
from extract_bench.inference.providers.extract.azure_content_understanding import (
    AzureContentUnderstandingProvider,
    _content_config,
    _safe_field_name,
)
from extract_bench.inference.providers.registry import register_provider
from extract_bench.schemas.pipeline import PipelineSpec
from extract_bench.schemas.pipeline_io import InferenceRequest, RawInferenceResult

_COMPLETION_DEPLOYMENT_TYPE = "GlobalStandard"
_COMPLETION_DATA_ZONE = "Global"
_COMPLETION_MODEL = "gpt-5.5"
_COMPLETION_SHORT_CONTEXT_PRICES_PER_1M = (5.0, 30.0)
_COMPLETION_LONG_CONTEXT_PRICES_PER_1M = (10.0, 45.0)
_COMPLETION_LONG_CONTEXT_THRESHOLD_TOKENS = 272_000
_PRICING_EFFECTIVE_FROM = "2026-05-01"
_PRICING_AS_OF = "2026-08-24"


@register_provider("azure_content_understanding_extractbench")
class AzureContentUnderstandingExtractBenchProvider(AzureContentUnderstandingProvider):
    """ACU extraction with an ExtractBench-aware JSON Schema adapter."""

    def __init__(self, provider_name: str, base_config: dict[str, Any] | None = None):
        config = dict(base_config or {})
        endpoint = config.get("endpoint") or os.getenv("AZURE_CONTENT_UNDERSTANDING_EXTRACTBENCH_ENDPOINT")
        api_key = config.get("api_key") or os.getenv("AZURE_CONTENT_UNDERSTANDING_EXTRACTBENCH_KEY")
        if endpoint is None or endpoint == "":
            raise ProviderConfigError(
                "acu-extract requires AZURE_CONTENT_UNDERSTANDING_EXTRACTBENCH_ENDPOINT or an explicit endpoint"
            )
        if api_key is None or api_key == "":
            raise ProviderConfigError(
                "acu-extract requires AZURE_CONTENT_UNDERSTANDING_EXTRACTBENCH_KEY or an explicit api_key"
            )
        config["endpoint"] = endpoint
        config["api_key"] = api_key
        super().__init__(provider_name, config)

    def run_inference(self, pipeline: PipelineSpec, request: InferenceRequest) -> RawInferenceResult:
        result = super().run_inference(pipeline, request)
        _apply_extractbench_completion_cost(result.raw_output, pipeline.config)
        return result

    def _build_analyzer(self, pipeline: PipelineSpec, schema: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        return _build_analyzer_payload_and_field_map(schema, pipeline.config)


def _apply_extractbench_completion_cost(raw_output: dict[str, Any], config: Mapping[str, Any]) -> None:
    """Reprice GPT-5.5 using the Global Standard context tier.

    The shared ACU helper handles the page and contextualization meters. GPT-5.5
    additionally has short- and long-context token rates, so this provider owns
    the model-specific adjustment without changing any legacy ACU pipeline.
    """
    usage = raw_output.get("usage")
    breakdown = raw_output.get("acu_cost_breakdown")
    if not isinstance(usage, Mapping) or not isinstance(breakdown, dict):
        raise ProviderPermanentError("Azure Content Understanding succeeded without auditable usage/cost data")

    models = config.get("models")
    expected_model = str(models.get("completion", "")) if isinstance(models, Mapping) else ""
    if expected_model != _COMPLETION_MODEL:
        raise ProviderPermanentError(
            f"acu-extract requires completion model {_COMPLETION_MODEL}; got {expected_model or 'none'}"
        )

    tokens = usage.get("tokens")
    if not isinstance(tokens, Mapping):
        raise ProviderPermanentError("Azure Content Understanding usage is missing completion token counts")

    input_key = f"{expected_model}-input"
    output_key = f"{expected_model}-output"
    completion_keys = {str(key) for key in tokens if str(key).endswith(("-input", "-output"))}
    unexpected_keys = completion_keys - {input_key, output_key}
    if len(unexpected_keys) > 0:
        raise ProviderPermanentError(
            f"Azure Content Understanding used unexpected completion meter(s): {sorted(unexpected_keys)}; "
            f"expected {expected_model}"
        )
    if input_key not in tokens and output_key not in tokens:
        raise ProviderPermanentError(
            f"Azure Content Understanding returned no token usage for configured model {expected_model}"
        )

    input_tokens = _number(tokens.get(input_key))
    output_tokens = _number(tokens.get(output_key))
    use_long_context = input_tokens > _COMPLETION_LONG_CONTEXT_THRESHOLD_TOKENS
    input_rate, output_rate = (
        _COMPLETION_LONG_CONTEXT_PRICES_PER_1M if use_long_context else _COMPLETION_SHORT_CONTEXT_PRICES_PER_1M
    )
    meter_context = "LongCo" if use_long_context else "ShortCo"

    input_cost = input_tokens / 1_000_000.0 * input_rate
    output_cost = output_tokens / 1_000_000.0 * output_rate
    content_cost = _number(breakdown.get("content_extraction_cost_usd"))
    contextualization_cost = _number(breakdown.get("contextualization_cost_usd"))
    total_cost = content_cost + contextualization_cost + input_cost + output_cost

    breakdown.update(
        {
            "total_cost_usd": total_cost,
            "llm_input_cost_usd": input_cost,
            "llm_output_cost_usd": output_cost,
            "llm_input_tokens": input_tokens,
            "llm_output_tokens": output_tokens,
            "completion_model": expected_model,
            "completion_deployment_type": _COMPLETION_DEPLOYMENT_TYPE,
            "completion_data_zone": _COMPLETION_DATA_ZONE,
            "completion_context_pricing_tier": "long" if use_long_context else "short",
            "completion_long_context_threshold_tokens": _COMPLETION_LONG_CONTEXT_THRESHOLD_TOKENS,
            "completion_input_price_per_1m": input_rate,
            "completion_output_price_per_1m": output_rate,
            "completion_input_meter": f"5.5 {meter_context} inp Gl 1M Tokens",
            "completion_output_meter": f"5.5 {meter_context} opt Gl 1M Tokens",
            "pricing_effective_from": _PRICING_EFFECTIVE_FROM,
            "pricing_as_of": _PRICING_AS_OF,
        }
    )
    raw_output["cost_usd"] = total_cost
    raw_output["input_cost_usd"] = input_cost
    raw_output["output_and_thinking_cost_usd"] = output_cost
    page_count = sum(
        _number(usage.get(key)) for key in ("documentPagesStandard", "documentPagesBasic", "documentPagesMinimal")
    )
    if page_count > 0:
        raw_output["cost_per_page_usd"] = total_cost / page_count


def _number(value: Any) -> float:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            pass
    return 0.0


def _build_analyzer_payload(schema: dict[str, Any], config: Mapping[str, Any]) -> dict[str, Any]:
    payload, _ = _build_analyzer_payload_and_field_map(schema, config)
    return payload


def _build_analyzer_payload_and_field_map(
    schema: dict[str, Any], config: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    field_method = str(config.get("field_method", "auto")).lower()
    if field_method not in {"auto", "classify", "extract", "generate"}:
        raise ProviderPermanentError(f"Unsupported Azure Content Understanding field method: {field_method}")
    fields, field_name_map = _schema_fields(schema, field_method)
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


def _schema_fields(
    schema: Mapping[str, Any],
    field_method: str,
    *,
    root_schema: Mapping[str, Any] | None = None,
    seen_refs: frozenset[str] = frozenset(),
    strict_unions: bool = True,
) -> tuple[dict[str, Any], dict[str, Any]]:
    root_schema = root_schema if root_schema is not None else schema
    resolved, active_refs = _resolve_schema_node(schema, root_schema, seen_refs, strict_unions)
    raw_properties = resolved.get("properties")
    properties = raw_properties if isinstance(raw_properties, Mapping) else {}
    fields: dict[str, Any] = {}
    field_name_map: dict[str, Any] = {}
    used_names: set[str] = set()
    for name, prop in properties.items():
        original_name = str(name)
        safe_name = _safe_field_name(original_name, used_names)
        node, children = _schema_node(
            prop,
            field_method,
            root_schema=root_schema,
            seen_refs=active_refs,
            strict_unions=strict_unions,
        )
        fields[safe_name] = node
        field_name_map[safe_name] = {"name": original_name, "children": children}
    return fields, field_name_map


def _schema_node(
    node: Any,
    field_method: str,
    *,
    root_schema: Mapping[str, Any],
    seen_refs: frozenset[str],
    strict_unions: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not isinstance(node, Mapping):
        fallback = {"type": "string", "estimateSourceAndConfidence": True}
        if field_method != "auto":
            fallback["method"] = field_method
        return fallback, {}

    resolved, active_refs = _resolve_schema_node(node, root_schema, seen_refs, strict_unions)
    node_type = _normalize_schema_type(resolved.get("type"), resolved)
    out: dict[str, Any] = {"type": node_type}
    field_name_map: dict[str, Any] = {}
    if isinstance(resolved.get("description"), str):
        out["description"] = resolved["description"]
    if isinstance(resolved.get("enum"), list):
        out["enum"] = [str(value) for value in resolved["enum"] if value is not None]

    if node_type == "object":
        raw_properties = resolved.get("properties")
        properties = raw_properties if isinstance(raw_properties, Mapping) else {}
        out["properties"], field_name_map = _schema_fields(
            {"properties": properties},
            field_method,
            root_schema=root_schema,
            seen_refs=active_refs,
            strict_unions=strict_unions,
        )
    elif node_type == "array":
        out["items"], field_name_map = _schema_node(
            resolved.get("items", {"type": "string"}),
            field_method,
            root_schema=root_schema,
            seen_refs=active_refs,
            strict_unions=strict_unions,
        )
    else:
        if field_method != "auto":
            out["method"] = field_method
        out["estimateSourceAndConfidence"] = True
    return out, field_name_map


def _resolve_schema_node(
    node: Mapping[str, Any],
    root_schema: Mapping[str, Any],
    seen_refs: frozenset[str],
    strict_unions: bool,
) -> tuple[dict[str, Any], frozenset[str]]:
    resolved = dict(node)
    active_refs = seen_refs

    ref = resolved.get("$ref")
    if isinstance(ref, str):
        if ref in active_refs:
            raise ProviderPermanentError(f"Recursive JSON Schema reference is not supported by ACU: {ref}")
        target = _resolve_local_schema_ref(ref, root_schema)
        target_resolved, active_refs = _resolve_schema_node(target, root_schema, active_refs | {ref}, strict_unions)
        resolved = {**target_resolved, **{key: value for key, value in resolved.items() if key != "$ref"}}

    if "allOf" in resolved:
        variants = resolved.pop("allOf")
        # ACU has no intersection operator. Unwrap a single schema, but never
        # guess how to merge multiple constraints or override a conflicting one.
        if not isinstance(variants, list) or len(variants) != 1 or not isinstance(variants[0], Mapping):
            raise ProviderPermanentError("ACU supports exactly one schema branch in JSON Schema allOf")
        composed, active_refs = _resolve_schema_node(variants[0], root_schema, active_refs, strict_unions)
        for key, value in resolved.items():
            if key in composed and composed[key] != value and key not in {"description", "title", "$comment"}:
                raise ProviderPermanentError(f"Conflicting JSON Schema allOf sibling: {key}")
        resolved = {**composed, **resolved}
        if isinstance(resolved.get("properties"), Mapping) and resolved.get("type", "object") != "object":
            raise ProviderPermanentError("Unsupported JSON Schema allOf object/type intersection")

    for union_key in ("anyOf", "oneOf"):
        raw_variants = resolved.get(union_key)
        if not isinstance(raw_variants, list):
            continue
        if (
            not strict_unions
            and sum(
                isinstance(variant, Mapping) and not _is_null_union_variant(variant, root_schema)
                for variant in raw_variants
            )
            != 1
        ):
            continue
        variants = [
            _resolve_schema_node(variant, root_schema, active_refs, strict_unions)
            for variant in raw_variants
            if isinstance(variant, Mapping)
        ]
        non_null = [variant for variant in variants if not _is_null_schema(variant[0])]
        if len(non_null) != 1:
            if not strict_unions:
                continue
            raise ProviderPermanentError(
                f"ACU requires exactly one non-null branch in JSON Schema {union_key}; got {len(non_null)}"
            )
        selected, active_refs = non_null[0]
        resolved = {
            **selected,
            **{key: value for key, value in resolved.items() if key != union_key},
        }

    return resolved, active_refs


def _resolve_local_schema_ref(ref: str, root_schema: Mapping[str, Any]) -> Mapping[str, Any]:
    if ref == "#":
        return root_schema
    if not ref.startswith("#/"):
        raise ProviderPermanentError(f"Only local JSON Schema references are supported by ACU: {ref}")

    target: Any = root_schema
    for raw_part in ref[2:].split("/"):
        part = raw_part.replace("~1", "/").replace("~0", "~")
        if not isinstance(target, Mapping) or part not in target:
            raise ProviderPermanentError(f"JSON Schema reference does not exist: {ref}")
        target = target[part]
    if not isinstance(target, Mapping):
        raise ProviderPermanentError(f"JSON Schema reference must point to an object: {ref}")
    return target


def _is_null_schema(node: Mapping[str, Any]) -> bool:
    node_type = node.get("type")
    if node_type == "null":
        return True
    enum = node.get("enum")
    return isinstance(enum, list) and enum == [None]


def _is_null_union_variant(node: Mapping[str, Any], root_schema: Mapping[str, Any]) -> bool:
    seen_refs: set[str] = set()
    while True:
        if _is_null_schema(node):
            return True
        ref = node.get("$ref")
        if not isinstance(ref, str) or ref in seen_refs:
            return False
        seen_refs.add(ref)
        try:
            target = _resolve_local_schema_ref(ref, root_schema)
        except ProviderPermanentError:
            return False
        node = {**target, **{key: value for key, value in node.items() if key != "$ref"}}


def _normalize_schema_type(value: Any, node: Mapping[str, Any]) -> str:
    if isinstance(value, list):
        non_null = [item for item in value if item != "null"]
        value = non_null[0] if len(non_null) > 0 else "string"
    if value in {"string", "number", "integer", "boolean", "array", "object"}:
        return str(value)
    if isinstance(node.get("properties"), Mapping):
        return "object"
    if "items" in node:
        return "array"
    return "string"
