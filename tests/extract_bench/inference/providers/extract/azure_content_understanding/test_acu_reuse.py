"""Offline analyzer lifecycle coverage; never connects to Azure."""

import copy
import json
from concurrent.futures import ThreadPoolExecutor
from threading import Event
from typing import Any
from urllib.parse import urlparse

import pytest
import requests

from extract_bench.inference.providers.base import ProviderPermanentError, ProviderTransientError
from extract_bench.inference.providers.extract.azure_content_understanding import (
    AzureContentUnderstandingProvider,
    _analyzer_definition_matches,
)
from extract_bench.inference.providers.extract.azure_content_understanding_extractbench import (
    AzureContentUnderstandingExtractBenchProvider,
)
from extract_bench.schemas.pipeline import PipelineSpec
from extract_bench.schemas.product import ProductType


def get_pipeline(name: str) -> PipelineSpec:
    return PipelineSpec(
        pipeline_name=name,
        provider_name="azure_content_understanding_extractbench"
        if name == "acu-extract"
        else "azure_content_understanding",
        product_type=ProductType.EXTRACT,
        config={
            "field_method": "generate" if name.endswith("gen") else "extract",
            "models": {
                "completion": "gpt-5.5" if name == "acu-extract" else "gpt-4.1",
                "embedding": "text-embedding-3-large",
            },
            "poll_interval": 3.0,
        },
    )


SCHEMA = {"type": "object", "properties": {"amount": {"type": "number"}}}


def response(code: int, body: dict[str, Any]) -> requests.Response:
    result = requests.Response()
    result.status_code = code
    result._content = json.dumps(body).encode()
    return result


@pytest.fixture
def provider() -> AzureContentUnderstandingProvider:
    return AzureContentUnderstandingProvider(
        "azure_content_understanding", {"endpoint": "https://example.test", "api_key": "offline"}
    )


def install_store(provider, monkeypatch):
    store = {}
    calls = []

    def request(method, url, *, context, **kwargs):
        analyzer_id = urlparse(url).path.rsplit("/", 1)[1]
        calls.append((method, analyzer_id, kwargs))
        if method == "GET":
            return response(200, store[analyzer_id]) if analyzer_id in store else response(404, {})
        assert method == "PUT"
        assert "allowReplace=false" in url
        assert analyzer_id not in store
        store[analyzer_id] = {**copy.deepcopy(kwargs["json"]), "analyzerId": analyzer_id, "status": "ready"}
        return response(201, store[analyzer_id])

    monkeypatch.setattr(provider, "_request", request)
    return store, calls


@pytest.mark.parametrize("pipeline_name", ["acu-ga-ext", "acu-ga-gen", "acu-extract"])
def test_cold_then_reused_ready_analyzer(provider, monkeypatch, pipeline_name):
    pipeline = get_pipeline(pipeline_name)
    if pipeline_name == "acu-extract":
        provider = AzureContentUnderstandingExtractBenchProvider(
            "azure_content_understanding_extractbench", {"endpoint": "https://example.test", "api_key": "offline"}
        )
    _, calls = install_store(provider, monkeypatch)
    first = provider._ensure_analyzer(pipeline, SCHEMA)
    second = provider._ensure_analyzer(pipeline, SCHEMA)
    assert first[:3] == second[:3]
    assert first[3] == "created"
    assert second[3] == "reused_ready"
    assert [call[0] for call in calls] == ["GET", "PUT", "GET", "GET"]


@pytest.mark.parametrize("change", ["schema", "method", "model", "content"])
def test_analyzer_identity_includes_entire_effective_payload(provider, monkeypatch, change):
    _, calls = install_store(provider, monkeypatch)
    pipeline = get_pipeline("acu-ga-ext").model_copy(deep=True)
    schema = copy.deepcopy(SCHEMA)
    first = provider._ensure_analyzer(pipeline, schema)
    if change == "schema":
        schema["properties"]["amount"]["description"] = "Total tax"
    elif change == "method":
        pipeline.config["field_method"] = "generate"
    elif change == "model":
        pipeline.config["models"] = {"completion": "gpt-4.1-mini", "embedding": "text-embedding-3-large"}
    else:
        pipeline.config["enable_formula"] = False
    second = provider._ensure_analyzer(pipeline, schema)
    assert first[0] != second[0]
    assert len([call for call in calls if call[0] == "PUT"]) == 2


@pytest.mark.parametrize("mutation", ["model", "config", "schema", "extra_setting", "id"])
def test_existing_mismatched_analyzer_is_never_reused_or_replaced(provider, monkeypatch, mutation):
    store, calls = install_store(provider, monkeypatch)
    pipeline = get_pipeline("acu-ga-ext")
    analyzer_id, _, _, _ = provider._ensure_analyzer(pipeline, SCHEMA)
    analyzer = store[analyzer_id]
    if mutation == "model":
        analyzer["models"]["completion"] = "wrong-model"
    elif mutation == "config":
        analyzer["config"]["enableFormula"] = False
    elif mutation == "schema":
        analyzer["fieldSchema"]["fields"]["Amount"]["description"] = "Different description"
    elif mutation == "extra_setting":
        analyzer["config"]["disableFaceBlurring"] = True
    else:
        analyzer["analyzerId"] = "wrong-id"
    with pytest.raises(ProviderPermanentError, match="does not match"):
        provider._ensure_analyzer(pipeline, SCHEMA)
    assert len([call for call in calls if call[0] == "PUT"]) == 1


@pytest.mark.parametrize("status", ["failed", "deleting", "unknown"])
def test_existing_nonready_analyzer_is_not_replaced(provider, monkeypatch, status):
    store, calls = install_store(provider, monkeypatch)
    pipeline = get_pipeline("acu-ga-ext")
    analyzer_id, _, _, _ = provider._ensure_analyzer(pipeline, SCHEMA)
    store[analyzer_id]["status"] = status
    with pytest.raises(ProviderPermanentError):
        provider._ensure_analyzer(pipeline, SCHEMA)
    assert len([call for call in calls if call[0] == "PUT"]) == 1


def test_existing_creating_analyzer_waits_and_rechecks_definition(provider, monkeypatch):
    store, calls = install_store(provider, monkeypatch)
    pipeline = get_pipeline("acu-ga-ext")
    analyzer_id, _, _, _ = provider._ensure_analyzer(pipeline, SCHEMA)
    store[analyzer_id]["status"] = "creating"

    def finish(*args):
        store[analyzer_id]["status"] = "ready"
        return True

    monkeypatch.setattr(
        "extract_bench.inference.providers.extract.azure_content_understanding.sleep_for_poll_interval", finish
    )
    result = provider._ensure_analyzer(pipeline, SCHEMA)
    assert result[3] == "reused_after_readiness_wait"
    assert len([call for call in calls if call[0] == "PUT"]) == 1


def test_same_configuration_concurrent_documents_create_once(provider, monkeypatch):
    store, calls = install_store(provider, monkeypatch)
    original_request = provider._request
    creating = Event()
    release = Event()

    def request(method, *args, **kwargs):
        if method == "PUT":
            creating.set()
            assert release.wait(5)
        return original_request(method, *args, **kwargs)

    monkeypatch.setattr(provider, "_request", request)
    pipeline = get_pipeline("acu-ga-ext")
    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(provider._ensure_analyzer, pipeline, SCHEMA)
        assert creating.wait(5)
        second = executor.submit(provider._ensure_analyzer, pipeline, SCHEMA)
        release.set()
        results = [first.result(), second.result()]
    assert [result[3] for result in results] == ["created", "reused_ready"]
    assert len(store) == 1
    assert len([call for call in calls if call[0] == "PUT"]) == 1


def test_remote_definition_accepts_only_documented_default_additions():
    expected = {"baseAnalyzerId": "prebuilt-document", "config": {"enableOcr": True}}
    body = {**copy.deepcopy(expected), "dynamicFieldSchema": False, "processingLocation": "global"}
    body["config"].update({"chartFormat": "chartJs", "locales": None})
    assert _analyzer_definition_matches(body, expected)
    body["processingLocation"] = "dataZone"
    assert not _analyzer_definition_matches(body, expected)


def test_analyzer_creation_race_verifies_remote_winner(provider, monkeypatch):
    pipeline = get_pipeline("acu-ga-ext")
    calls = []
    expected = provider._build_analyzer(pipeline, SCHEMA)[0]

    def request(method, url, *, context, **kwargs):
        calls.append(method)
        if method == "PUT":
            return response(409, {"error": {"code": "Conflict"}})
        if len(calls) == 1:
            return response(404, {})
        analyzer_id = urlparse(url).path.rsplit("/", 1)[1]
        return response(200, {**expected, "analyzerId": analyzer_id, "status": "ready"})

    monkeypatch.setattr(provider, "_request", request)
    result = provider._ensure_analyzer(pipeline, SCHEMA)
    assert result[3] == "reused_after_create_conflict"
    assert calls == ["GET", "PUT", "GET"]


def test_analyzer_readiness_timeout_does_not_replace(provider, monkeypatch):
    store, calls = install_store(provider, monkeypatch)
    pipeline = get_pipeline("acu-ga-ext")
    analyzer_id, _, _, _ = provider._ensure_analyzer(pipeline, SCHEMA)
    store[analyzer_id]["status"] = "creating"
    monkeypatch.setattr(
        "extract_bench.inference.providers.extract.azure_content_understanding.sleep_for_poll_interval",
        lambda *args: False,
    )
    with pytest.raises(ProviderTransientError, match="Timed out"):
        provider._ensure_analyzer(pipeline, SCHEMA)
    assert len([call for call in calls if call[0] == "PUT"]) == 1


def test_unrecognized_remote_configuration_is_not_ignored():
    expected = {"baseAnalyzerId": "prebuilt-document", "config": {"enableOcr": True}}
    assert not _analyzer_definition_matches({**expected, "futureSetting": True}, expected)
