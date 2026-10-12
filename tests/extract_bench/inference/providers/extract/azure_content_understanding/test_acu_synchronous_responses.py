"""ACU failures and malformed results must not become successful empty predictions."""

import json
from datetime import datetime

import pytest
import requests

from extract_bench.inference.providers.extract import azure_content_understanding as shared


@pytest.fixture
def provider_module():
    return shared


@pytest.mark.parametrize(
    "body,error",
    [
        (b"not JSON", "ProviderTransientError"),
        (b"null", "ProviderTransientError"),
        (b"[]", "ProviderTransientError"),
        ({}, "ProviderTransientError"),
        ({"status": "running"}, "ProviderTransientError"),
        ({"status": "succeeded", "result": {}}, "ProviderTransientError"),
        ({"result": {"contents": []}}, "ProviderTransientError"),
        ({"status": "succeeded", "result": {"contents": [{"kind": "image", "fields": {}}]}}, "ProviderTransientError"),
        ({"status": "failed", "error": {"code": "InvalidDocument"}}, "ProviderPermanentError"),
        ({"status": "canceled"}, "ProviderPermanentError"),
        ({"status": "failed", "error": {"innererror": {"code": "429"}}}, "ProviderRateLimitError"),
    ],
)
def test_invalid_synchronous_response_fails(provider_module, tmp_path, monkeypatch, body, error):
    provider = provider_module.AzureContentUnderstandingProvider(
        "azure_content_understanding", {"endpoint": "https://example.invalid", "api_key": "fake"}
    )
    source = tmp_path / "sample.pdf"
    source.write_bytes(b"local fixture")
    response = requests.Response()
    response.status_code = 200
    response._content = body if isinstance(body, bytes) else json.dumps(body).encode()
    monkeypatch.setattr(provider, "_request", lambda *args, **kwargs: response)
    with pytest.raises(getattr(provider_module, error)):
        provider._analyze_binary("fixture", source)


@pytest.mark.parametrize("wrapped", [False, True])
@pytest.mark.parametrize("with_kind", [False, True])
def test_synchronous_document_normalizes(provider_module, tmp_path, monkeypatch, wrapped, with_kind):
    content = {"contents": [{"kind": "document", "fields": {"total": {"type": "number", "valueNumber": 12}}}]}
    if not with_kind:
        del content["contents"][0]["kind"]
    body = {"status": "succeeded", "result": content} if wrapped else content
    provider = provider_module.AzureContentUnderstandingProvider(
        "azure_content_understanding", {"endpoint": "https://example.invalid", "api_key": "fake"}
    )
    source = tmp_path / "sample.pdf"
    source.write_bytes(b"local fixture")
    response = requests.Response()
    response.status_code = 200
    response._content = json.dumps(body).encode()
    monkeypatch.setattr(provider, "_request", lambda *args, **kwargs: response)
    result = provider._analyze_binary("fixture", source)
    pipeline = provider_module.PipelineSpec(
        pipeline_name="probe",
        provider_name="azure_content_understanding",
        product_type=provider_module.ProductType.EXTRACT,
        config={},
    )
    request = provider_module.InferenceRequest(
        example_id="probe", source_file_path=str(source), product_type=provider_module.ProductType.EXTRACT
    )
    raw = provider_module.RawInferenceResult(
        request=request,
        pipeline=pipeline,
        pipeline_name="probe",
        product_type=provider_module.ProductType.EXTRACT,
        raw_output=result,
        started_at=datetime.now(),
        completed_at=datetime.now(),
        latency_in_ms=0,
    )
    assert provider.normalize(raw).output.extracted_data == {"total": 12}


@pytest.mark.parametrize(
    "result,valid",
    [
        ({}, False),
        ({"contents": [{"kind": "image", "fields": {}}]}, False),
        ({"contents": [{"kind": "document", "fields": None}]}, False),
        ({"contents": [{"kind": "document", "fields": {}}]}, True),
    ],
)
def test_polled_analysis_validates_document_result(provider_module, tmp_path, monkeypatch, result, valid):
    provider = provider_module.AzureContentUnderstandingProvider(
        "azure_content_understanding", {"endpoint": "https://example.invalid", "api_key": "fake"}
    )
    source = tmp_path / "sample.pdf"
    source.write_bytes(b"local fixture")
    submission = requests.Response()
    submission.status_code = 202
    submission.headers["Operation-Location"] = "https://example.invalid/operations/1"
    submission._content = b"{}"
    completed = requests.Response()
    completed.status_code = 200
    body = {"status": "succeeded", "result": result}
    completed._content = json.dumps(body).encode()
    responses = iter([submission, completed])
    methods = []

    def request(method, *args, **kwargs):
        methods.append(method)
        return next(responses)

    monkeypatch.setattr(provider, "_request", request)
    if valid:
        assert provider._analyze_binary("fixture", source)["result"] == body["result"]
    else:
        with pytest.raises(provider_module.ProviderTransientError):
            provider._analyze_binary("fixture", source)
    assert methods == ["POST", "GET"]
