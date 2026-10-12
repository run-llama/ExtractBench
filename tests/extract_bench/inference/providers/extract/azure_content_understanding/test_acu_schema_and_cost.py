import json

import pytest
import requests

from extract_bench.inference.providers.base import ProviderPermanentError, ProviderRateLimitError
from extract_bench.inference.providers.extract.azure_content_understanding_extractbench import (
    AzureContentUnderstandingExtractBenchProvider,
    _apply_extractbench_completion_cost,
    _build_analyzer_payload,
)


def _cost_config() -> dict:
    return {"models": {"completion": "gpt-5.5"}}


def _raw_usage(input_tokens: int, output_tokens: int) -> dict:
    return {
        "usage": {
            "documentPagesStandard": 1,
            "contextualizationTokens": 1000,
            "tokens": {
                "gpt-5.5-input": input_tokens,
                "gpt-5.5-output": output_tokens,
            },
        },
        "acu_cost_breakdown": {
            "content_extraction_cost_usd": 0.005,
            "contextualization_cost_usd": 0.001,
        },
    }


def test_cost_uses_gpt_5_5_global_standard_short_context_rates() -> None:
    raw_output = _raw_usage(10_000, 1_000)

    _apply_extractbench_completion_cost(raw_output, _cost_config())

    assert raw_output["cost_usd"] == pytest.approx(0.086)
    assert raw_output["input_cost_usd"] == pytest.approx(0.05)
    assert raw_output["output_and_thinking_cost_usd"] == pytest.approx(0.03)
    assert raw_output["acu_cost_breakdown"]["completion_context_pricing_tier"] == "short"
    assert raw_output["acu_cost_breakdown"]["completion_deployment_type"] == "GlobalStandard"
    assert raw_output["acu_cost_breakdown"]["completion_data_zone"] == "Global"
    assert raw_output["acu_cost_breakdown"]["completion_input_meter"] == "5.5 ShortCo inp Gl 1M Tokens"
    assert raw_output["acu_cost_breakdown"]["completion_output_meter"] == "5.5 ShortCo opt Gl 1M Tokens"
    assert raw_output["acu_cost_breakdown"]["pricing_effective_from"] == "2026-05-01"


def test_cost_uses_gpt_5_5_global_standard_long_context_rates() -> None:
    raw_output = _raw_usage(300_000, 10_000)

    _apply_extractbench_completion_cost(raw_output, _cost_config())

    assert raw_output["cost_usd"] == pytest.approx(3.456)
    assert raw_output["input_cost_usd"] == pytest.approx(3.0)
    assert raw_output["output_and_thinking_cost_usd"] == pytest.approx(0.45)
    assert raw_output["acu_cost_breakdown"]["completion_context_pricing_tier"] == "long"


def test_provider_uses_extractbench_specific_resource(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AZURE_CONTENT_UNDERSTANDING_ENDPOINT", "https://legacy.services.ai.azure.com")
    monkeypatch.setenv("AZURE_CONTENT_UNDERSTANDING_KEY", "legacy-key")
    monkeypatch.setenv(
        "AZURE_CONTENT_UNDERSTANDING_EXTRACTBENCH_ENDPOINT",
        "https://extractbench.services.ai.azure.com/api/projects/ignored",
    )
    monkeypatch.setenv("AZURE_CONTENT_UNDERSTANDING_EXTRACTBENCH_KEY", "extractbench-key")

    provider = AzureContentUnderstandingExtractBenchProvider("azure_content_understanding_extractbench")

    assert provider._endpoint == "https://extractbench.services.ai.azure.com"
    assert provider._api_key == "extractbench-key"


def test_cost_fails_closed_on_unexpected_completion_meter() -> None:
    raw_output = _raw_usage(10_000, 1_000)
    raw_output["usage"]["tokens"] = {
        "gpt-4.1-input": 10_000,
        "gpt-4.1-output": 1_000,
    }

    with pytest.raises(ProviderPermanentError, match="unexpected completion meter"):
        _apply_extractbench_completion_cost(raw_output, _cost_config())


def test_analyzer_payload_resolves_refs_nullable_unions_and_auto_method() -> None:
    schema = {
        "type": "object",
        "$defs": {
            "address": {
                "type": "object",
                "properties": {
                    "city": {"type": "string"},
                    "postal_code": {"type": ["string", "null"]},
                },
            },
            "line_item": {
                "type": "object",
                "properties": {
                    "description": {"type": "string"},
                    "amount": {"anyOf": [{"type": "number"}, {"type": "null"}]},
                },
            },
        },
        "properties": {
            "billing_address": {
                "anyOf": [{"$ref": "#/$defs/address"}, {"type": "null"}],
            },
            "line_items": {
                "type": "array",
                "items": {"$ref": "#/$defs/line_item"},
            },
        },
    }

    fields = _build_analyzer_payload(schema, {"field_method": "auto"})["fieldSchema"]["fields"]

    assert fields["BillingAddress"]["type"] == "object"
    assert fields["BillingAddress"]["properties"]["PostalCode"]["type"] == "string"
    assert fields["LineItems"]["items"]["type"] == "object"
    assert fields["LineItems"]["items"]["properties"]["Amount"] == {
        "type": "number",
        "estimateSourceAndConfidence": True,
    }
    assert "method" not in fields["LineItems"]["items"]["properties"]["Description"]


def test_extractbench_analyzer_rejects_multi_branch_union() -> None:
    schema = {
        "type": "object",
        "properties": {"ambiguous": {"anyOf": [{"type": "number"}, {"type": "string"}]}},
    }

    with pytest.raises(ProviderPermanentError, match="exactly one non-null branch"):
        _build_analyzer_payload(schema, {"field_method": "auto"})


def test_poll_operation_classifies_nested_azure_429_as_rate_limit() -> None:
    response = requests.Response()
    response.status_code = 200
    response._content = json.dumps(
        {
            "status": "Failed",
            "error": {
                "code": "InternalServerError",
                "innererror": {
                    "code": "ResourceError",
                    "innererror": {
                        "code": "RateLimit",
                        "message": "(RateLimit) 429, Rate limit exceeded for gpt-4.1",
                    },
                },
            },
        }
    ).encode()

    provider = AzureContentUnderstandingExtractBenchProvider.__new__(AzureContentUnderstandingExtractBenchProvider)
    provider._poll_timeout = 0
    provider._poll_interval = 0
    provider._request = lambda *args, **kwargs: response  # type: ignore[method-assign]

    with pytest.raises(ProviderRateLimitError, match="rate limited"):
        provider._poll_operation("https://example.test/operation", context="analyze document.pdf")
